import torch
import torch.nn as nn
import torch.nn.functional as F

class OnlineMMDDetector(nn.Module):
    """
    Online Maximum Mean Discrepancy (MMD) Detector.
    Maintains a rolling baseline window to detect covariate shift.
    """
    def __init__(self, feature_dim, window_size=60, threshold=0.05, kernel_sigma=1.0):
        super().__init__()
        self.window_size = window_size
        self.threshold = threshold
        self.kernel_sigma = kernel_sigma
        
        # Buffer to hold the baseline vectors (B, window_size, feature_dim)
        self.register_buffer('baseline_buffer', torch.zeros(1, window_size, feature_dim))
        self.register_buffer('buffer_idx', torch.zeros(1, dtype=torch.long))
        self.register_buffer('is_ready', torch.zeros(1, dtype=torch.bool))
        
    def compute_rbf_kernel(self, x, y):
        # x: (B, N, D), y: (B, M, D)
        x_norm = (x ** 2).sum(-1).unsqueeze(2) # (B, N, 1)
        y_norm = (y ** 2).sum(-1).unsqueeze(1) # (B, 1, M)
        dist = x_norm + y_norm - 2.0 * torch.bmm(x, y.transpose(1, 2))
        return torch.exp(-dist / (2.0 * self.kernel_sigma ** 2))

    def forward(self, x):
        """
        x: current feature vector (B, 1, D)
        Returns: drift_detected (bool tensor)
        """
        B = x.shape[0]
        if self.baseline_buffer.shape[0] != B:
            # Dynamically resize batch dimension if needed
            self.baseline_buffer = self.baseline_buffer.expand(B, -1, -1).clone()
            
        if not self.is_ready.item():
            # Fill the buffer
            idx = self.buffer_idx.item()
            self.baseline_buffer[:, idx:idx+1, :] = x.detach()
            self.buffer_idx += 1
            if self.buffer_idx.item() >= self.window_size:
                self.is_ready.fill_(True)
            return torch.zeros(B, dtype=torch.bool, device=x.device)
            
        # Compute MMD between x and baseline
        baseline = self.baseline_buffer
        
        # K(X, X) where X is just x
        k_xx = self.compute_rbf_kernel(x, x).mean(dim=(1,2))
        # K(Y, Y) where Y is baseline
        k_yy = self.compute_rbf_kernel(baseline, baseline).mean(dim=(1,2))
        # K(X, Y)
        k_xy = self.compute_rbf_kernel(x, baseline).mean(dim=(1,2))
        
        mmd_score = k_xx + k_yy - 2 * k_xy
        drift_detected = mmd_score > self.threshold
        
        # Update baseline using a rolling window approach
        # Move everything left by 1
        self.baseline_buffer = torch.roll(self.baseline_buffer, shifts=-1, dims=1)
        # Add new frame to the end
        self.baseline_buffer[:, -1:, :] = x.detach()
        
        return drift_detected

class DecisionMatrix(nn.Module):
    """
    Evaluates Prediction Confidence (Entropy) and Feature Quality (Variance).
    Outputs 3 States:
    0: NORMAL (No drift or acceptable variance)
    1: GOOD_DRIFT (Appearance changed but features sharp)
    2: BAD_DRIFT (Corruption/Camera blocked)
    """
    def __init__(self, entropy_threshold=1.5, variance_threshold=0.1):
        super().__init__()
        self.entropy_thresh = entropy_threshold
        self.var_thresh = variance_threshold
        
    def forward(self, x, softmax_preds, drift_detected):
        """
        x: latent feature vector (B, D) or (B, 1, D)
        softmax_preds: prediction probabilities (B, C)
        drift_detected: bool tensor (B)
        """
        # Calculate Variance (Feature Sharpness)
        var_x = torch.var(x, dim=-1) # (B,)
        high_variance = var_x > self.var_thresh
        
        # Calculate Entropy (Prediction Confidence)
        epsilon = 1e-8
        entropy = -torch.sum(softmax_preds * torch.log(softmax_preds + epsilon), dim=-1)
        low_entropy = entropy < self.entropy_thresh
        
        states = torch.zeros_like(drift_detected, dtype=torch.long)
        
        # Condition 2: Good Drift (Drift = True, Low Entropy, High Variance)
        good_drift = drift_detected & low_entropy & high_variance
        states[good_drift] = 1
        
        # Condition 3: Bad Drift (Drift = True, High Entropy, Low Variance)
        # We also trigger bad drift if variance completely collapses regardless of drift
        bad_drift = (drift_detected & (~low_entropy) & (~high_variance)) | (~high_variance)
        states[bad_drift] = 2
        
        return states

class SoftGateFallback(nn.Module):
    """
    Calculates weights to dynamically scale the imputed proxy vector 
    when the real vector is corrupted or missing.
    """
    def __init__(self, alpha=10.0, tau=0.5):
        super().__init__()
        # Learnable parameters for the sigmoid curve
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.tau = nn.Parameter(torch.tensor(tau))
        
    def forward(self, real_x, proxy_x):
        """
        Calculates cosine similarity and applies the sigmoid gate.
        If real_x is completely zeroed out (missing), sim is 0.
        """
        # (B, 1, D)
        epsilon = 1e-8
        sim = F.cosine_similarity(real_x + epsilon, proxy_x + epsilon, dim=-1) # (B, 1)
        
        # Soft gate weight
        w = 1.0 / (1.0 + torch.exp(-self.alpha * (sim - self.tau))) # (B, 1)
        
        # Scale proxy
        imputed_x = proxy_x * w.unsqueeze(-1)
        return imputed_x, w

class StreamingGovernanceLayer(nn.Module):
    """
    The orchestrator for the Drift Detection and Gating mechanism.
    Handles all three modalities (V, A, T).
    """
    def __init__(self, d_model=128):
        super().__init__()
        # Drift detectors per modality
        self.drift_v = OnlineMMDDetector(d_model)
        self.drift_a = OnlineMMDDetector(d_model)
        self.drift_t = OnlineMMDDetector(d_model)
        
        self.decision = DecisionMatrix()
        
        # Fallback gates per modality
        self.gate_v = SoftGateFallback()
        self.gate_a = SoftGateFallback()
        self.gate_t = SoftGateFallback()
        
    def forward(self, v_real, a_real, t_real, v_proxy, a_proxy, t_proxy, current_preds):
        """
        v_real, a_real, t_real: MS-TCN outputs (B, 1, d_model)
        v_proxy, a_proxy, t_proxy: Generated by Modality Adapters from Thread B
        current_preds: (B, C) classification probs from the *previous* frame for entropy check
        
        Returns: 
            v_final, a_final, t_final: The routed vectors to send to Graph Fusion
            states: Dict of states (0=Normal, 1=Good, 2=Bad) for Thread B to know when to halt backprop
        """
        # 1. Check Drift
        drift_v = self.drift_v(v_real)
        drift_a = self.drift_a(a_real)
        drift_t = self.drift_t(t_real)
        
        # 2. Evaluate States
        state_v = self.decision(v_real.squeeze(1), current_preds, drift_v)
        state_a = self.decision(a_real.squeeze(1), current_preds, drift_a)
        state_t = self.decision(t_real.squeeze(1), current_preds, drift_t)
        
        # 3. Soft Gate Fallback Activation
        # If Bad Drift (State == 2), use the soft gate imputed vector. Else, use real.
        
        v_imputed, w_v = self.gate_v(v_real, v_proxy)
        a_imputed, w_a = self.gate_a(a_real, a_proxy)
        t_imputed, w_t = self.gate_t(t_real, t_proxy)
        
        # Route depending on state
        B = v_real.shape[0]
        v_final = torch.where((state_v == 2).view(B, 1, 1), v_imputed, v_real)
        a_final = torch.where((state_a == 2).view(B, 1, 1), a_imputed, a_real)
        t_final = torch.where((state_t == 2).view(B, 1, 1), t_imputed, t_real)
        
        # If MULTIPLE modalities are missing (State == 2 for >1 modality),
        # we reduce the reliance on imputed vectors and rely more on the remaining healthy modality.
        # This is handled dynamically by the Graph Fusion downstream since the Graph nodes
        # for missing modalities will have lower magnitude due to w_v/w_a/w_t scaling.
        
        states = {'v': state_v, 'a': state_a, 't': state_t}
        return v_final, a_final, t_final, states
