import torch
import sys
import os

from src.objectives import NeuroSymbolicLoss

def test_loss():
    print("Testing NeuroSymbolicLoss...")
    # Initialize loss with kg_dim=64 (from d=128 in drcfnet, so d//2=64)
    loss_fn = NeuroSymbolicLoss(kg_dim=64, lambda_logic=0.5)
    
    # Dummy data
    B = 2
    T = 50
    d = 64
    
    # Let's say model predicted 2.0 and -1.5
    pred = torch.tensor([[2.0], [-1.5]], requires_grad=True)
    truth = torch.tensor([[2.5], [-1.0]])
    
    # Dummy h_kg features
    h_kg = torch.randn(B, T, d, requires_grad=True)
    features = {'h_kg': h_kg}
    
    # Calculate loss
    loss, l_task, l_logic = loss_fn(pred, truth, features)
    
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Task Loss: {l_task.item():.4f}")
    print(f"Logic Penalty: {l_logic.item():.4f}")
    
    # Check backward pass
    loss.backward()
    
    print(f"Gradient on pred: {pred.grad.squeeze().tolist()}")
    print(f"Gradient on h_kg exists: {h_kg.grad is not None}")
    print("NeuroSymbolicLoss test passed successfully!")

if __name__ == '__main__':
    test_loss()
