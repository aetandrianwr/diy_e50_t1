"""
Training script for hybrid neural-statistical model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import os
import json
import random
import pickle
from tqdm import tqdm

from src.data import get_dataloaders
from src.models.hybrid_model import create_model, LocationStatistics
from src.utils import get_performance_dict, calculate_correct_total_prediction


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_raw_data(path):
    """Load raw pickle data."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, use_stats=True):
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
            logits = model(locations, users, weekdays, start_mins, durations, 
                          time_diffs, lengths, use_stats=use_stats)
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


def evaluate(model, data_loader, device, split_name='Val', use_stats=True):
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
            
            logits = model(locations, users, weekdays, start_mins, durations,
                          time_diffs, lengths, use_stats=use_stats)
            
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
    
    # Load or compute statistics
    stats_path = 'checkpoints_hybrid/location_stats.pk'
    os.makedirs('checkpoints_hybrid', exist_ok=True)
    
    if os.path.exists(stats_path):
        print('Loading pre-computed statistics...')
        stats = LocationStatistics()
        stats.load(stats_path)
    else:
        print('Computing statistics from training data...')
        train_data = load_raw_data(train_path)
        stats = LocationStatistics()
        stats.fit(train_data)
        stats.save(stats_path)
    
    # Load data
    print('\nLoading data...')
    train_loader, val_loader, test_loader = get_dataloaders(
        train_path, val_path, test_path,
        batch_size=256,
        num_workers=4
    )
    
    num_locations = 7037
    num_users = 692
    
    print('\nCreating hybrid model...')
    model = create_model(
        num_locations=num_locations,
        num_users=num_users,
        hidden_dim=88,  # Optimized for <1M params
        stats=stats
    )
    
    model = model.to(device)
    num_params = model.count_parameters()
    print(f'Model parameters: {num_params:,}')
    
    # Training configuration
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    scaler = GradScaler()
    
    # Warmup then cosine annealing
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=95, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])
    
    best_val_acc = 0
    best_test_acc = 0
    patience_counter = 0
    patience = 25
    num_epochs = 100
    
    history = {
        'train_loss': [],
        'val_acc@1': [],
        'test_acc@1': [],
        'learning_rate': []
    }
    
    print('\n' + '='*60)
    print('PHASE 1: Neural-only training (warmup)')
    print('='*60)
    
    # Phase 1: Train neural component only for first few epochs
    for epoch in range(5):
        lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(lr)
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, 
                                device, epoch, use_stats=False)
        history['train_loss'].append(train_loss)
        
        scheduler.step()
        
        val_perf = evaluate(model, val_loader, device, 'Val', use_stats=False)
        val_acc = val_perf['acc@1']
        history['val_acc@1'].append(val_acc)
        
        test_perf = evaluate(model, test_loader, device, 'Test', use_stats=False)
        test_acc = test_perf['acc@1']
        history['test_acc@1'].append(test_acc)
        
        print(f'\nEpoch {epoch} (Neural-only):')
        print(f'  Loss: {train_loss:.4f} | LR: {lr:.6f}')
        print(f'  Val Acc@1: {val_acc:.2f}% | Test Acc@1: {test_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
    
    print('\n' + '='*60)
    print('PHASE 2: Hybrid training (neural + statistical)')
    print('='*60)
    
    # Phase 2: Full hybrid training
    for epoch in range(5, num_epochs):
        lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(lr)
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler,
                                device, epoch, use_stats=True)
        history['train_loss'].append(train_loss)
        
        scheduler.step()
        
        val_perf = evaluate(model, val_loader, device, 'Val', use_stats=True)
        val_acc = val_perf['acc@1']
        history['val_acc@1'].append(val_acc)
        
        test_perf = evaluate(model, test_loader, device, 'Test', use_stats=True)
        test_acc = test_perf['acc@1']
        history['test_acc@1'].append(test_acc)
        
        print(f'\nEpoch {epoch} (Hybrid):')
        print(f'  Loss: {train_loss:.4f} | LR: {lr:.6f}')
        print(f'  Val Acc@1: {val_acc:.2f}% | MRR: {val_perf["mrr"]:.2f}%')
        print(f'  Test Acc@1: {test_acc:.2f}% | MRR: {test_perf["mrr"]:.2f}%')
        print(f'  Ensemble weight (neural): {torch.sigmoid(model.ensemble_weight).item():.3f}')
        
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
            }, 'checkpoints_hybrid/best_model.pt')
            
            print(f'  ✓ New best! Val: {val_acc:.2f}%, Test: {test_acc:.2f}%')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping at epoch {epoch + 1}')
                break
    
    # Save history
    with open('checkpoints_hybrid/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final evaluation
    print('\n\n' + '='*60)
    print('FINAL EVALUATION')
    print('='*60)
    
    checkpoint = torch.load('checkpoints_hybrid/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_test_perf = evaluate(model, test_loader, device, 'Final Test', use_stats=True)
    
    print(f'\n{"="*60}')
    print(f'FINAL TEST RESULTS (HYBRID):')
    print(f'  Acc@1:  {final_test_perf["acc@1"]:.2f}%')
    print(f'  Acc@5:  {final_test_perf["acc@5"]:.2f}%')
    print(f'  Acc@10: {final_test_perf["acc@10"]:.2f}%')
    print(f'  MRR:    {final_test_perf["mrr"]:.2f}%')
    print(f'  NDCG:   {final_test_perf["ndcg"]:.2f}%')
    print(f'{"="*60}')
    
    with open('checkpoints_hybrid/final_results.json', 'w') as f:
        json.dump(final_test_perf, f, indent=2)
    
    return final_test_perf


if __name__ == '__main__':
    main()
