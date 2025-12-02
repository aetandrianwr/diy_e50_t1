"""
Training script for optimized next location prediction model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import json
import random
from pathlib import Path
from tqdm import tqdm

from src.data import get_dataloaders
from src.models.optimized_model import create_model
from src.utils import get_performance_dict, calculate_correct_total_prediction


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss


class CosineAnnealingWarmup:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        locations = batch['locations'].to(device)
        users = batch['users'].to(device)
        weekdays = batch['weekdays'].to(device)
        start_mins = batch['start_mins'].to(device)
        durations = batch['durations'].to(device)
        time_diffs = batch['time_diffs'].to(device)
        lengths = batch['lengths'].to(device)
        targets = batch['targets'].to(device)
        
        with autocast():
            logits = model(locations, users, weekdays, start_mins, durations, time_diffs, lengths)
            loss = criterion(logits, targets)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def evaluate(model, data_loader, device, split_name='Val'):
    model.eval()
    
    all_results = {
        "correct@1": 0,
        "correct@3": 0,
        "correct@5": 0,
        "correct@10": 0,
        "rr": 0,
        "ndcg": 0,
        "f1": 0,
        "total": 0
    }
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc=f'{split_name}')
        for batch in pbar:
            locations = batch['locations'].to(device)
            users = batch['users'].to(device)
            weekdays = batch['weekdays'].to(device)
            start_mins = batch['start_mins'].to(device)
            durations = batch['durations'].to(device)
            time_diffs = batch['time_diffs'].to(device)
            lengths = batch['lengths'].to(device)
            targets = batch['targets'].to(device)
            
            logits = model(locations, users, weekdays, start_mins, durations, time_diffs, lengths)
            
            result_array, _, _ = calculate_correct_total_prediction(logits, targets)
            
            all_results["correct@1"] += result_array[0]
            all_results["correct@3"] += result_array[1]
            all_results["correct@5"] += result_array[2]
            all_results["correct@10"] += result_array[3]
            all_results["rr"] += result_array[4]
            all_results["ndcg"] += result_array[5]
            all_results["total"] += result_array[6]
    
    perf = get_performance_dict(all_results)
    return perf


def train(
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    num_epochs=100,
    learning_rate=0.002,
    warmup_epochs=8,
    patience=20,
    checkpoint_dir='checkpoints_v2',
    label_smoothing=0.15
):
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingWarmup(optimizer, warmup_epochs, num_epochs)
    scaler = GradScaler()
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    history = {
        'train_loss': [],
        'val_acc@1': [],
        'test_acc@1': [],
        'learning_rate': []
    }
    
    for epoch in range(num_epochs):
        lr = scheduler.step(epoch)
        history['learning_rate'].append(lr)
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        history['train_loss'].append(train_loss)
        
        val_perf = evaluate(model, val_loader, device, 'Val')
        val_acc = val_perf['acc@1']
        history['val_acc@1'].append(val_acc)
        
        test_perf = evaluate(model, test_loader, device, 'Test')
        test_acc = test_perf['acc@1']
        history['test_acc@1'].append(test_acc)
        
        print(f'\nEpoch {epoch}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Acc@1: {val_acc:.2f}% | Acc@5: {val_perf["acc@5"]:.2f}% | MRR: {val_perf["mrr"]:.2f}%')
        print(f'  Test Acc@1: {test_acc:.2f}% | Acc@5: {test_perf["acc@5"]:.2f}% | MRR: {test_perf["mrr"]:.2f}%')
        print(f'  LR: {lr:.6f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'test_acc': test_acc,
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
            
            print(f'  ✓ New best model! Val: {val_acc:.2f}%, Test: {test_acc:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch + 1}')
                break
    
    with open(os.path.join(checkpoint_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f'\n{"="*60}')
    print(f'Training completed!')
    print(f'Best Val Acc@1: {best_val_acc:.2f}%')
    print(f'Best Test Acc@1: {best_test_acc:.2f}%')
    print(f'{"="*60}')
    
    return history, best_val_acc, best_test_acc


def main():
    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    train_path = 'data/diy_skip_first_part_transformer_7_train.pk'
    val_path = 'data/diy_skip_first_part_transformer_7_validation.pk'
    test_path = 'data/diy_skip_first_part_transformer_7_test.pk'
    
    print('Loading data...')
    train_loader, val_loader, test_loader = get_dataloaders(
        train_path, val_path, test_path,
        batch_size=256,
        num_workers=4
    )
    
    num_locations = 7037
    num_users = 692
    
    print('\nCreating optimized model...')
    model = create_model(
        num_locations=num_locations,
        num_users=num_users,
        hidden_dim=92,  # Increased from 88
        num_heads=4,
        num_layers=3,   # Increased from 2
        dropout=0.15
    )
    
    model = model.to(device)
    num_params = model.count_parameters()
    print(f'Model parameters: {num_params:,}')
    assert num_params < 1_000_000, f'Model has {num_params:,} parameters (>= 1M)'
    
    print('\nStarting training...')
    history, best_val_acc, best_test_acc = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=100,
        learning_rate=0.002,
        warmup_epochs=8,
        patience=20,
        checkpoint_dir='checkpoints_v2',
        label_smoothing=0.15
    )
    
    print('\n\nFinal evaluation with best model...')
    checkpoint = torch.load('checkpoints_v2/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_test_perf = evaluate(model, test_loader, device, 'Final Test')
    
    print(f'\n{"="*60}')
    print(f'FINAL TEST RESULTS:')
    print(f'  Acc@1:  {final_test_perf["acc@1"]:.2f}%')
    print(f'  Acc@5:  {final_test_perf["acc@5"]:.2f}%')
    print(f'  Acc@10: {final_test_perf["acc@10"]:.2f}%')
    print(f'  MRR:    {final_test_perf["mrr"]:.2f}%')
    print(f'  NDCG:   {final_test_perf["ndcg"]:.2f}%')
    print(f'{"="*60}')
    
    with open('checkpoints_v2/final_results.json', 'w') as f:
        json.dump(final_test_perf, f, indent=2)
    
    return final_test_perf


if __name__ == '__main__':
    import torch.nn.functional as F
    main()
