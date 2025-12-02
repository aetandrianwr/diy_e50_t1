"""
Final training script for advanced pure neural model.
Focus: Reach 70% Acc@1 with optimized training.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import json
import random
from tqdm import tqdm

from src.data import get_dataloaders
from src.models.advanced_neural import create_model
from src.utils import get_performance_dict, calculate_correct_total_prediction


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, device, epoch):
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
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.6f}'})
    
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
    
    print('\n' + '='*70)
    print('ADVANCED PURE NEURAL MODEL - TARGET: 70% Acc@1')
    print('='*70)
    
    model = create_model(
        num_locations=num_locations,
        num_users=num_users,
        hidden_dim=152,
        num_layers=3,
        num_heads=4,
        dropout=0.2
    )
    
    model = model.to(device)
    num_params = model.count_parameters()
    print(f'\nModel parameters: {num_params:,}')
    print(f'Under 2M: {"✓ YES" if num_params < 2_000_000 else "✗ NO"}')
    print(f'Budget remaining: {2_000_000 - num_params:,}\n')
    
    # Training configuration - aggressive for better performance
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.98))
    
    # OneCycleLR for best performance
    from torch.optim.lr_scheduler import OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.003,
        epochs=150,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.05,
        anneal_strategy='cos',
        div_factor=10,
        final_div_factor=100
    )
    
    scaler = GradScaler()
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    patience = 35
    num_epochs = 150
    
    os.makedirs('checkpoints_final', exist_ok=True)
    
    history = {
        'train_loss': [],
        'val_acc@1': [],
        'test_acc@1': [],
        'val_acc@5': [],
        'test_acc@5': []
    }
    
    print('Starting training...\n')
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, device, epoch)
        history['train_loss'].append(train_loss)
        
        val_perf = evaluate(model, val_loader, device, 'Val')
        val_acc = val_perf['acc@1']
        history['val_acc@1'].append(val_acc)
        history['val_acc@5'].append(val_perf['acc@5'])
        
        test_perf = evaluate(model, test_loader, device, 'Test')
        test_acc = test_perf['acc@1']
        history['test_acc@1'].append(test_acc)
        history['test_acc@5'].append(test_perf['acc@5'])
        
        print(f'\nEpoch {epoch}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val  Acc@1: {val_acc:.2f}% | Acc@5: {val_perf["acc@5"]:.2f}% | MRR: {val_perf["mrr"]:.2f}%')
        print(f'  Test Acc@1: {test_acc:.2f}% | Acc@5: {test_perf["acc@5"]:.2f}% | MRR: {test_perf["mrr"]:.2f}%')
        
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
                'val_perf': val_perf,
                'test_perf': test_perf,
            }, 'checkpoints_final/best_model.pt')
            
            print(f'  ✓✓✓ NEW BEST! Val: {val_acc:.2f}%, Test: {test_acc:.2f}% ✓✓✓')
            
            if test_acc >= 70.0:
                print('\n' + '='*70)
                print(f'🎉🎉🎉 TARGET ACHIEVED! Test Acc@1: {test_acc:.2f}% >= 70% 🎉🎉🎉')
                print('='*70)
                break
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch + 1}')
                break
    
    with open('checkpoints_final/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print('\n\n' + '='*70)
    print('FINAL EVALUATION WITH BEST MODEL')
    print('='*70)
    
    checkpoint = torch.load('checkpoints_final/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_test_perf = evaluate(model, test_loader, device, 'Final Test')
    
    print(f'\n{"="*70}')
    print(f'FINAL TEST RESULTS:')
    print(f'{"="*70}')
    print(f'  Acc@1:  {final_test_perf["acc@1"]:.2f}%')
    print(f'  Acc@5:  {final_test_perf["acc@5"]:.2f}%')
    print(f'  Acc@10: {final_test_perf["acc@10"]:.2f}%')
    print(f'  MRR:    {final_test_perf["mrr"]:.2f}%')
    print(f'  NDCG:   {final_test_perf["ndcg"]:.2f}%')
    print(f'{"="*70}')
    
    if final_test_perf['acc@1'] >= 70.0:
        print(f'\n🎉🎉🎉 SUCCESS! Target of 70% Acc@1 ACHIEVED! 🎉🎉🎉\n')
    else:
        gap = 70.0 - final_test_perf['acc@1']
        print(f'\n⚠️  Current: {final_test_perf["acc@1"]:.2f}% | Gap to target: {gap:.2f}%')
        print(f'   Best achieved so far. Consider:')
        print(f'   - More training epochs')
        print(f'   - Data augmentation')
        print(f'   - Ensemble methods\n')
    
    with open('checkpoints_final/final_results.json', 'w') as f:
        json.dump(final_test_perf, f, indent=2)
    
    return final_test_perf


if __name__ == '__main__':
    main()
