import torch
import torch.nn as nn
from .drcfnet import MSTCNProjection, GatedSplit, LiteMRU, MultiHeadGraphFusion, BimodalGCFModule
from .governance import StreamingGovernanceLayer
from .shadow_trainer import ImageBindProxyGenerator, ShadowTrainer

class StreamingDRCFNet(nn.Module):
    """
    Streaming-Native DRCFNet integrated with Governance Layer and Dual-Thread Shadow Training.
    Designed to process frames in real-time buffers (e.g. B=1, T=1).
    """
    def __init__(self, dim_v=35, dim_a=74, dim_t=300, dim_kg=300, d=128, n_heads=4, dropout=0.2, proxy_dim=1024):
        super(StreamingDRCFNet, self).__init__()
        self.d = d
        
        # 1. Base MS-TCN Extractors
        self.proj_v = MSTCNProjection(dim_v, d, dropout)
        self.proj_a = MSTCNProjection(dim_a, d, dropout)
        self.proj_t = MSTCNProjection(dim_t, d, dropout)
        
        # Note: Temporal Transformer omitted for pure streaming ultra-low latency,
        # or replaced by causal convolutions. For now, assuming MS-TCN provides local context.
        
        # 2. Governance Layer
        self.governance = StreamingGovernanceLayer(d_model=d)
        
        # 3. Thread B: Shadow Trainer components
        # Note: In production ONNX deployment, the ShadowTrainer runs in Go/Python backend.
        # Here we provide the PyTorch linkage to Meta's ImageBind.
        self.proxy_generator = ImageBindProxyGenerator()
        self.shadow_trainer = ShadowTrainer(proxy_dim=proxy_dim, target_dim=d)
        
        # Unlike the dummy extractors, ImageBind takes raw Modality inputs (audio wav, images, text)
        # So we no longer need dummy linear projections to proxy space here.
        
        # 4. Gated Split
        self.split_v = GatedSplit(d)
        self.split_a = GatedSplit(d)
        self.split_t = GatedSplit(d)
        
        # 5. Lite-MRU
        self.mru_t = LiteMRU(d_model=d//2, n_heads=n_heads, dropout=dropout)
        self.mru_a = LiteMRU(d_model=d//2, n_heads=n_heads, dropout=dropout)
        self.mru_v = LiteMRU(d_model=d//2, n_heads=n_heads, dropout=dropout)
        
        # 6. Graph Fusions
        self.het_gnn = MultiHeadGraphFusion(d_model=d//2, n_heads=n_heads)
        self.hom_gnn = MultiHeadGraphFusion(d_model=d//2, n_heads=n_heads)
        self.kg_placeholder = nn.Parameter(torch.randn(1, 1, d//2))
        self.kg_proj = nn.Linear(dim_kg, d//2)
        
        # 7. Bimodal GCF
        self.gcf_ta = BimodalGCFModule(d_model=d//2, n_heads=n_heads, dropout=dropout)
        self.gcf_av = BimodalGCFModule(d_model=d//2, n_heads=n_heads, dropout=dropout)
        self.gcf_vt = BimodalGCFModule(d_model=d//2, n_heads=n_heads, dropout=dropout)
        
        self.pair_attn = nn.Linear(d//2, 1)
        self.fc = nn.Sequential(
            nn.Linear(d//2, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1)
        )
        
        # Buffer for previous predictions (required for Entropy check in Governance)
        self.register_buffer('prev_preds', torch.zeros(1, 1))

    def forward(self, vision, audio, text, raw_inputs=None, kg_features=None, enable_shadow_training=True):
        """
        vision, audio, text: Live buffer chunks (e.g. 1 frame, shape B=1, T=1, D) for MS-TCN
        raw_inputs: Dict of raw ModalityType inputs for ImageBind proxy generation.
        """
        # Step 1: Base Projections (Produces the Real vectors)
        v_real = self.proj_v(vision.permute(0, 2, 1))
        a_real = self.proj_a(audio.permute(0, 2, 1))
        t_real = self.proj_t(text.permute(0, 2, 1))
        
        # Step 2: Generate Proxies using ImageBind
        with torch.no_grad():
            B = v_real.shape[0]
            if raw_inputs is not None:
                # Use ImageBind to extract 1024-D proxy embeddings directly from raw data
                proxy_embeddings = self.proxy_generator(raw_inputs)
                raw_proxy_v = proxy_embeddings.get(self.proxy_generator.ModalityType.VISION, torch.zeros(B, 1024).to(v_real.device))
                raw_proxy_a = proxy_embeddings.get(self.proxy_generator.ModalityType.AUDIO, torch.zeros(B, 1024).to(a_real.device))
                raw_proxy_t = proxy_embeddings.get(self.proxy_generator.ModalityType.TEXT, torch.zeros(B, 1024).to(t_real.device))
            else:
                # Fallback zero-tensors if no raw inputs are provided
                raw_proxy_v = torch.zeros(B, 1024).to(v_real.device)
                raw_proxy_a = torch.zeros(B, 1024).to(a_real.device)
                raw_proxy_t = torch.zeros(B, 1024).to(t_real.device)
                
            # ImageBind returns (B, 1024). We need (B, 1, 1024) for MS-TCN space compatibility.
            if raw_proxy_v.dim() == 2: raw_proxy_v = raw_proxy_v.unsqueeze(1)
            if raw_proxy_a.dim() == 2: raw_proxy_a = raw_proxy_a.unsqueeze(1)
            if raw_proxy_t.dim() == 2: raw_proxy_t = raw_proxy_t.unsqueeze(1)
            
        # Modality Adapters map proxy space -> MS-TCN space
        # These adapters are constantly updated by Thread B
        v_proxy_adapted = self.shadow_trainer.adapter_v(raw_proxy_v)
        a_proxy_adapted = self.shadow_trainer.adapter_a(raw_proxy_a)
        t_proxy_adapted = self.shadow_trainer.adapter_t(raw_proxy_t)
        
        # Step 3: Governance Layer routing and fallback
        # prev_preds converted to pseudo-probabilities for entropy calculation
        pseudo_probs = torch.sigmoid(self.prev_preds) 
        pseudo_probs = torch.cat([pseudo_probs, 1 - pseudo_probs], dim=-1)
        
        v_final, a_final, t_final, states = self.governance(
            v_real, a_real, t_real, 
            v_proxy_adapted, a_proxy_adapted, t_proxy_adapted, 
            pseudo_probs
        )
        
        # Step 4: Asynchronous update dispatch to Thread B
        if enable_shadow_training and self.training:
            self.shadow_trainer.push_data(
                v_real, a_real, t_real,
                raw_proxy_v, raw_proxy_a, raw_proxy_t,
                states
            )
            
        # Step 5: Downstream Model (Graph Fusion)
        msr_v, ssr_v = self.split_v(v_final)
        msr_a, ssr_a = self.split_a(a_final)
        msr_t, ssr_t = self.split_t(t_final)
        
        # Lite-MRU
        z_prime_t = self.mru_t(ssr_t, torch.cat([ssr_a, ssr_v], dim=1))
        z_prime_a = self.mru_a(ssr_a, torch.cat([ssr_t, ssr_v], dim=1))
        z_prime_v = self.mru_v(ssr_v, torch.cat([ssr_t, ssr_a], dim=1))
        
        # Graph Fusions
        B, T, _ = msr_t.shape
        kg_node = self.kg_placeholder.expand(B, T, -1) if kg_features is None else self.kg_proj(kg_features)
            
        h_m_t, h_m_a, h_m_v, h_kg = self.het_gnn([msr_t, msr_a, msr_v, kg_node])
        h_z_t, h_z_a, h_z_v = self.hom_gnn([z_prime_t, z_prime_a, z_prime_v])
        
        # Bimodal GCF
        o_ta = self.gcf_ta(h_m_t + h_z_t, h_m_a + h_z_a)
        o_av = self.gcf_av(h_m_a + h_z_a, h_m_v + h_z_v)
        o_vt = self.gcf_vt(h_m_v + h_z_v, h_m_t + h_z_t)
        
        # Final Prediction
        pairs = torch.stack([o_ta, o_av, o_vt], dim=1)
        pair_weights = F.softmax(self.pair_attn(pairs.mean(dim=2)), dim=1)
        combined = (pairs * pair_weights.unsqueeze(-1)).sum(dim=1)
        
        output = self.fc(combined.mean(dim=1))
        
        # Update buffer for next frame's entropy check
        self.prev_preds = output.detach()
        
        return output
