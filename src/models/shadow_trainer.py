import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
import queue
import time

class ImageBindProxyGenerator(nn.Module):
    """
    Real Pretrained Model Wrapper: Meta's ImageBind.
    ImageBind natively projects Vision, Audio, and Text into a single joint 1024-D embedding space.
    This acts as the ultimate proxy generator. If Video is missing, we use the ImageBind Text/Audio 
    embeddings to generate the proxy Video embedding, which is then mapped by the Adapter to the MS-TCN space.
    """
    def __init__(self, device='cpu'):
        super().__init__()
        try:
            from imagebind.models import imagebind_model
            from imagebind.models.imagebind_model import ModalityType
            self.ModalityType = ModalityType
        except ImportError:
            raise ImportError("Please install ImageBind: pip install git+https://github.com/facebookresearch/ImageBind.git")
            
        # Load the frozen ImageBind Huge model
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(device)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, inputs):
        """
        inputs: dict mapping ModalityType to the raw tensor inputs required by ImageBind.
        Example: {ModalityType.VISION: video_tensor, ModalityType.AUDIO: audio_tensor, ModalityType.TEXT: text_tokens}
        Returns: Dict of 1024-D embeddings for each provided modality.
        """
        with torch.no_grad():
            embeddings = self.model(inputs)
        return embeddings

class ModalityAdapter(nn.Module):
    """
    A 2-Layer Neural Network that translates the generic proxy vector 
    into the specific MS-TCN latent space of the C2DTF model.
    These are the weights continuously updated by Thread B.
    """
    def __init__(self, proxy_dim=512, target_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proxy_dim, proxy_dim // 2),
            nn.ReLU(),
            nn.Linear(proxy_dim // 2, target_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class ShadowTrainer:
    """
    Thread B logic.
    Continuously receives valid (real) features and generated proxies from Thread A.
    Updates the Adapters via Cosine Similarity Loss unless BAD_DRIFT is triggered.
    """
    def __init__(self, proxy_dim=512, target_dim=128, lr=0.001):
        self.adapter_v = ModalityAdapter(proxy_dim, target_dim)
        self.adapter_a = ModalityAdapter(proxy_dim, target_dim)
        self.adapter_t = ModalityAdapter(proxy_dim, target_dim)
        
        self.optimizer = torch.optim.Adam([
            {'params': self.adapter_v.parameters()},
            {'params': self.adapter_a.parameters()},
            {'params': self.adapter_t.parameters()}
        ], lr=lr)
        
        # Loss function: Cosine Embedding Loss
        self.criterion = nn.CosineEmbeddingLoss()
        
        # Threading mechanisms
        self.training_queue = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._train_loop)
        
    def start(self):
        self.thread.start()
        
    def stop(self):
        self._stop_event.set()
        self.thread.join()
        
    def push_data(self, real_v, real_a, real_t, proxy_v_raw, proxy_a_raw, proxy_t_raw, states):
        """
        Called by Main Inference Thread A to pass data to Thread B.
        Data is detached to prevent graph retention issues across threads.
        """
        if not self.training_queue.full():
            self.training_queue.put({
                'real_v': real_v.detach(), 'real_a': real_a.detach(), 'real_t': real_t.detach(),
                'proxy_v': proxy_v_raw.detach(), 'proxy_a': proxy_a_raw.detach(), 'proxy_t': proxy_t_raw.detach(),
                'states': states
            })
            
    def _train_loop(self):
        """
        The background learning loop.
        """
        while not self._stop_event.is_set():
            try:
                data = self.training_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            states = data['states']
            
            # Forward pass through adapters
            adapted_v = self.adapter_v(data['proxy_v'])
            adapted_a = self.adapter_a(data['proxy_a'])
            adapted_t = self.adapter_t(data['proxy_t'])
            
            B = adapted_v.shape[0]
            target = torch.ones(B, device=adapted_v.device) # Target for cosine embedding loss (1 = similar)
            
            self.optimizer.zero_grad()
            total_loss = 0.0
            
            # Rule: HALT backpropagation instantly for a modality if BAD_DRIFT (State == 2)
            # This prevents corrupting the Adapter weights with bad real data.
            
            if (states['v'] != 2).all():
                loss_v = self.criterion(adapted_v.squeeze(1), data['real_v'].squeeze(1), target)
                total_loss += loss_v
                
            if (states['a'] != 2).all():
                loss_a = self.criterion(adapted_a.squeeze(1), data['real_a'].squeeze(1), target)
                total_loss += loss_a
                
            if (states['t'] != 2).all():
                loss_t = self.criterion(adapted_t.squeeze(1), data['real_t'].squeeze(1), target)
                total_loss += loss_t
                
            if isinstance(total_loss, torch.Tensor):
                total_loss.backward()
                self.optimizer.step()
