# Next Location Prediction

Advanced next location prediction system using PyTorch with multi-feature fusion architecture.

## Project Structure

```
.
├── data/                           # Dataset files
├── src/
│   ├── models/                     # Model architectures
│   │   └── next_location_model.py
│   ├── data/                       # Dataset and data loading
│   │   └── dataset.py
│   └── utils/                      # Utilities and metrics
│       └── metrics.py
├── checkpoints/                    # Saved models
├── logs/                          # Training logs
├── train.py                       # Training script
└── README.md
```

## Model Architecture

The model uses a hierarchical architecture that leverages:

1. **Location Embeddings**: Learned representations for 7017+ locations
2. **User Embeddings**: Personalized context for 692+ users
3. **Multi-Scale Temporal Encoding**:
   - Weekday embeddings (7 categories)
   - Time-of-day (sin/cos encoding from minute-of-day)
   - Duration (log-normalized continuous)
   - Time gap between visits (0-7 categorical)
4. **Cross-Feature Attention**: Captures interactions between features
5. **Self-Attention with Relative Positional Encoding**: Models sequence dependencies
6. **Adaptive Gated Fusion**: Learns to combine features optimally

**Parameters**: ~900K (below 1M limit)

## Requirements

```bash
torch>=2.0.0
numpy
scikit-learn
tqdm
```

## Usage

### Training

```bash
python train.py
```

### Key Features

- **Mixed Precision Training**: Faster training with automatic mixed precision
- **Label Smoothing**: Reduces overfitting (smoothing=0.1)
- **Warmup + Cosine Annealing**: Learning rate schedule for stable convergence
- **Gradient Clipping**: Prevents exploding gradients
- **Early Stopping**: Stops when validation performance plateaus

## Dataset

The model expects pickle files with the following structure:

```python
{
    'X': np.array([...]),           # Location sequence
    'user_X': np.array([...]),      # User IDs
    'weekday_X': np.array([...]),   # Weekdays (0-6)
    'start_min_X': np.array([...]), # Start minute (0-1439)
    'dur_X': np.array([...]),       # Duration in minutes
    'diff': np.array([...]),        # Time difference (0-7)
    'Y': int                        # Target location
}
```

## Results

Target: **70% Test Acc@1**

The model is designed to achieve this through:
- Deep feature engineering and fusion
- Efficient parameter usage (<1M)
- Advanced training techniques
- Robust generalization strategies

## Metrics

- **Acc@1, Acc@5, Acc@10**: Top-k accuracy
- **MRR**: Mean Reciprocal Rank
- **NDCG**: Normalized Discounted Cumulative Gain
