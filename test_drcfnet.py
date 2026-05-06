import torch
import sys
import os

from src.models.drcfnet import DRCFNet

def test():
    print("Testing DRCFNet instantiation and forward pass...")
    # Initialize model
    model = DRCFNet(dim_v=35, dim_a=74, dim_t=300, dim_kg=300, d=128, n_heads=4, num_layers=2)
    
    # Create dummy inputs (Batch=2, Sequence Length=50, Feature Dims)
    B, T = 2, 50
    v = torch.randn(B, T, 35)
    a = torch.randn(B, T, 74)
    t = torch.randn(B, T, 300)
    
    # Test 1: Without external KG features (should use placeholder)
    print("Test 1: Forward without external KG...")
    try:
        out1, features1 = model(v, a, t)
        print(f"  Output shape: {out1.shape}")
        print(f"  h_kg shape: {features1['h_kg'].shape}")
    except Exception as e:
        print(f"  Error in Test 1: {e}")
        return
        
    # Test 2: With external KG features
    print("Test 2: Forward with external KG...")
    kg = torch.randn(B, T, 300)
    try:
        out2, features2 = model(v, a, t, kg_features=kg)
        print(f"  Output shape: {out2.shape}")
        print(f"  h_kg shape: {features2['h_kg'].shape}")
        print("All tests passed successfully!")
    except Exception as e:
        print(f"  Error in Test 2: {e}")

if __name__ == '__main__':
    test()
