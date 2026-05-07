import os
import torch
import numpy as np
from evaluation.performance import eval_affect

def train_epoch(model, dataloader, optimizer, criterion, device='cuda', clip_grad=1.0):
    model.train()
    total_loss = 0.0
    total_task = 0.0
    total_logic = 0.0
    
    for batch in dataloader:
        vision, audio, text, labels = batch
        vision = vision.to(device)
        audio = audio.to(device)
        text = text.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        logits, features = model(vision, audio, text, kg_features=None)
        
        loss, l_task, l_logic = criterion(logits, labels, features)
        loss.backward()
        
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        total_task += l_task.item()
        total_logic += l_logic.item()
        
    num_batches = len(dataloader)
    return total_loss / num_batches, total_task / num_batches, total_logic / num_batches


def test(model, dataloader, criterion, device='cuda', return_preds=False):
    model.eval()
    total_loss = 0.0
    total_task = 0.0
    total_logic = 0.0
    
    all_preds = []
    all_truths = []
    
    with torch.no_grad():
        for batch in dataloader:
            vision, audio, text, labels = batch
            vision = vision.to(device)
            audio = audio.to(device)
            text = text.to(device)
            labels = labels.to(device)
            
            logits, features = model(vision, audio, text, kg_features=None)
            
            loss, l_task, l_logic = criterion(logits, labels, features)
            
            total_loss += loss.item()
            total_task += l_task.item()
            total_logic += l_logic.item()
            
            all_preds.append(logits.cpu())
            all_truths.append(labels.cpu())
            
    num_batches = len(dataloader)
    
    preds = torch.cat(all_preds, dim=0)
    truths = torch.cat(all_truths, dim=0)
    
    acc = eval_affect(truths, preds)
    
    metrics = {
        'Loss': total_loss / num_batches,
        'Task': total_task / num_batches,
        'Logic': total_logic / num_batches,
        'Accuracy': acc
    }
    
    if return_preds:
        return metrics, preds, truths
    return metrics


def train(model, train_loader, valid_loader, criterion, optimizer, epochs=50, device='cuda', scheduler=None, clip_grad=1.0, save_path='best_model.pth'):
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'train_task': [], 'valid_task': [], 'train_logic': [], 'valid_logic': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        t_loss, t_task, t_logic = train_epoch(model, train_loader, optimizer, criterion, device, clip_grad)
        
        val_metrics = test(model, valid_loader, criterion, device)
        v_loss = val_metrics['Loss']
        v_acc = val_metrics['Accuracy']
        
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)
        history['train_task'].append(t_task)
        history['valid_task'].append(val_metrics['Task'])
        history['train_logic'].append(t_logic)
        history['valid_logic'].append(val_metrics['Logic'])
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc:.4f}")
        
        if scheduler is not None:
            scheduler.step(v_loss)
            
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), save_path)
            print(f"  >> Saved best model with Val Loss: {v_loss:.4f}")
            
    print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
    
    # Load best model weights before returning
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        
    return model, history
