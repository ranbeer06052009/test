import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MSTCNProjection(nn.Module):
    """Multi-Scale Temporal Convolutional Projection"""
    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels // 4, 1)
        self.conv3 = nn.Conv1d(in_channels, out_channels // 4, 3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels // 4, 5, padding=2)
        self.skip = nn.Conv1d(in_channels, out_channels // 4, 1)
        
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (B, C, T)
        c1 = self.conv1(x)
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        sk = self.skip(x)
        
        out = torch.cat([c1, c3, c5, sk], dim=1) # (B, out, T)
        out = out.permute(0, 2, 1) # (B, T, out)
        out = self.norm(out)
        return self.dropout(F.relu(out))

class MultiHeadGraphFusion(nn.Module):
    """Multi-Head Graph Attention Fusion"""
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        self.w_o = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, nodes):
        # nodes is a list of 5 tensors, each (B, T, d)
        x = torch.stack(nodes, dim=2) # (B, T, N, d)
        B, T, N, d = x.size()
        
        # Project and split into heads
        q = self.w_q(x).view(B, T, N, self.n_heads, self.d_k).transpose(2, 3) # (B, T, h, N, dk)
        k = self.w_k(x).view(B, T, N, self.n_heads, self.d_k).transpose(2, 3)
        v = self.w_v(x).view(B, T, N, self.n_heads, self.d_k).transpose(2, 3)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) # (B, T, h, N, N)
        attn = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn, v) # (B, T, h, N, dk)
        out = out.transpose(2, 3).contiguous().view(B, T, N, d)
        out = self.w_o(out)
        
        # Residual and Norm
        out = self.layer_norm(x + out)
        return [out[:, :, i, :] for i in range(N)]

class GatedSplit(nn.Module):
    """Gated MSR/SSR Disentanglement Split"""
    def __init__(self, d_model):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_model // 2)
        self.w_msr = nn.Linear(d_model, d_model // 2)
        self.w_ssr = nn.Linear(d_model, d_model // 2)
        
    def forward(self, h):
        gate = torch.sigmoid(self.w_gate(h))
        msr = self.w_msr(h) * gate
        ssr = self.w_ssr(h) * (1 - gate)
        return msr, ssr

class LiteMRU(nn.Module):
    """Lite Modality Routing Unit for Cross-modal Attention"""
    def __init__(self, d_model, n_heads=4, dropout=0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key_value):
        attn_out, _ = self.mha(query=query, key=key_value, value=key_value)
        out = self.layer_norm(query + attn_out)
        return out

class BimodalGCFModule(nn.Module):
    """Bimodal Gated Controlled Fusion (from Image 1)"""
    def __init__(self, d_model, n_heads=4, dropout=0.2):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        
        self.w_r = nn.Linear(2 * d_model, d_model)
        self.w_c = nn.Linear(2 * d_model, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, z_r, z_c):
        m, _ = self.mha(query=z_r, key=z_c, value=z_c)
        
        h_r = z_r.mean(dim=1)
        h_c = z_c.mean(dim=1)
        h_concat = torch.cat([h_r, h_c], dim=-1)
        
        g_r = torch.sigmoid(self.w_r(h_concat)).unsqueeze(1)
        g_c = torch.sigmoid(self.w_c(h_concat)).unsqueeze(1)
        
        o_fusion = (g_r * z_r) + (g_c * m)
        
        o_norm = self.layer_norm(o_fusion)
        o_final = self.final_norm(o_norm + self.ffn(o_norm))
        return o_final

class DRCFNet(nn.Module):
    def __init__(self, dim_v=35, dim_a=74, dim_t=300, dim_kg=300, d=128, n_heads=4, dropout=0.2, num_layers=3):
        super(DRCFNet, self).__init__()
        self.d = d
        
        # Step 1: Multi-Scale Feature Projection
        self.proj_v = MSTCNProjection(dim_v, d, dropout)
        self.proj_a = MSTCNProjection(dim_a, d, dropout)
        self.proj_t = MSTCNProjection(dim_t, d, dropout)
        
        # Step 2: Temporal Transformer
        encoder_layer_v = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout, batch_first=True)
        self.transformer_v = nn.TransformerEncoder(encoder_layer_v, num_layers=num_layers)
        
        encoder_layer_a = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout, batch_first=True)
        self.transformer_a = nn.TransformerEncoder(encoder_layer_a, num_layers=num_layers)
        
        encoder_layer_t = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d*4, dropout=dropout, batch_first=True)
        self.transformer_t = nn.TransformerEncoder(encoder_layer_t, num_layers=num_layers)
        
        # Step 3: Gated MSR/SSR Split
        self.split_v = GatedSplit(d)
        self.split_a = GatedSplit(d)
        self.split_t = GatedSplit(d)
        
        # Step 4: Lite-MRU
        self.mru_t = LiteMRU(d_model=d//2, n_heads=n_heads, dropout=dropout)
        self.mru_a = LiteMRU(d_model=d//2, n_heads=n_heads, dropout=dropout)
        self.mru_v = LiteMRU(d_model=d//2, n_heads=n_heads, dropout=dropout)
        
        # Step 5: Dual-Graph Neural Fusion
        self.het_gnn = MultiHeadGraphFusion(d_model=d//2, n_heads=n_heads)
        self.hom_gnn = MultiHeadGraphFusion(d_model=d//2, n_heads=n_heads)
        
        # KG Node Integration
        self.kg_placeholder = nn.Parameter(torch.randn(1, 1, d//2))
        self.kg_proj = nn.Linear(dim_kg, d//2)
        
        # Step 6: Bimodal GCF in 3 Pairs
        self.gcf_ta = BimodalGCFModule(d_model=d//2, n_heads=n_heads, dropout=dropout)
        self.gcf_av = BimodalGCFModule(d_model=d//2, n_heads=n_heads, dropout=dropout)
        self.gcf_vt = BimodalGCFModule(d_model=d//2, n_heads=n_heads, dropout=dropout)
        
        # Positional Encoding
        self.pos_emb = nn.Parameter(torch.empty(1, 100, d))
        nn.init.uniform_(self.pos_emb, -0.01, 0.01)
        
        # Step 7: Final Prediction (Weighted Addition)
        self.pair_attn = nn.Linear(d//2, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(d//2, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, 1)
        )

    def forward(self, vision, audio, text, kg_features=None):
        # Step 1: MS-TCN Projection
        v = self.proj_v(vision.permute(0, 2, 1))
        a = self.proj_a(audio.permute(0, 2, 1))
        t = self.proj_t(text.permute(0, 2, 1))
        
        # Add Positional Encoding
        v = v + self.pos_emb[:, :v.size(1), :]
        a = a + self.pos_emb[:, :a.size(1), :]
        t = t + self.pos_emb[:, :t.size(1), :]
        
        # Step 2: Temporal Transformer
        h_v = self.transformer_v(v)
        h_a = self.transformer_a(a)
        h_t = self.transformer_t(t)
        
        # Step 3: Gated Split
        msr_v, ssr_v = self.split_v(h_v)
        msr_a, ssr_a = self.split_a(h_a)
        msr_t, ssr_t = self.split_t(h_t)
        
        # Step 4: Lite-MRU
        z_other_t = torch.cat([ssr_a, ssr_v], dim=1)
        z_other_a = torch.cat([ssr_t, ssr_v], dim=1)
        z_other_v = torch.cat([ssr_t, ssr_a], dim=1)
        
        z_prime_t = self.mru_t(ssr_t, z_other_t)
        z_prime_a = self.mru_a(ssr_a, z_other_a)
        z_prime_v = self.mru_v(ssr_v, z_other_v)
        
        # Step 5: Dual-Graph Fusion
        B, T, _ = msr_t.shape
        if kg_features is None:
            kg_node = self.kg_placeholder.expand(B, T, -1)
        else:
            kg_node = self.kg_proj(kg_features)
            
        het_nodes = [msr_t, msr_a, msr_v, kg_node]
        h_m_t, h_m_a, h_m_v, h_kg = self.het_gnn(het_nodes)
        
        hom_nodes = [z_prime_t, z_prime_a, z_prime_v]
        h_z_t, h_z_a, h_z_v = self.hom_gnn(hom_nodes)
        
        # Combine MSR and SSR nodes for Bimodal GCF
        H_T = h_m_t + h_z_t
        H_A = h_m_a + h_z_a
        H_V = h_m_v + h_z_v
        
        # Step 6: Bimodal GCF in 3 pairs
        o_ta = self.gcf_ta(H_T, H_A)
        o_av = self.gcf_av(H_A, H_V)
        o_vt = self.gcf_vt(H_V, H_T)
        
        # Step 7: Predict via Weighted Addition
        pairs = torch.stack([o_ta, o_av, o_vt], dim=1) # (B, 3, T, d//2)
        
        # Attention scores for pairs
        pair_pooled = pairs.mean(dim=2) # (B, 3, d//2)
        pair_scores = self.pair_attn(pair_pooled) # (B, 3, 1)
        pair_weights = F.softmax(pair_scores, dim=1) # (B, 3, 1)
        
        # Weighted addition of pairs
        combined = (pairs * pair_weights.unsqueeze(-1)).sum(dim=1) # (B, T, d//2)
        
        # Temporal pooling and classify
        final_rep = combined.mean(dim=1) # (B, d//2)
        output = self.fc(final_rep)
        
        return output, {
            'msr_v': msr_v, 'ssr_v': ssr_v,
            'msr_a': msr_a, 'ssr_a': ssr_a,
            'msr_t': msr_t, 'ssr_t': ssr_t,
            'kg_node': kg_node, 'h_kg': h_kg
        }
