"""
Pure Neural Next Location Prediction Model with 2M Budget.

Focus on deeper, more powerful neural architecture instead of statistical priors.
Key improvements:
- Deeper Transformer (3-4 layers)
- Larger embeddings
- Better feature engineering
- Advanced training techniques

Target: 70%+ Acc@1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AdvancedNextLocationPredictor(nn.Module):
    """
    Advanced pure neural model optimized for 2M parameter budget.
    """
    
    def __init__(self, num_locations, num_users, hidden_dim=144, num_layers=4, num_heads=4, dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Embeddings with larger capacity
        self.loc_emb = nn.Embedding(num_locations + 1, hidden_dim, padding_idx=0)
        self.user_emb = nn.Embedding(num_users + 1, 32, padding_idx=0)
        
        # Rich temporal encoding
        self.weekday_emb = nn.Embedding(7, 16)
        self.hour_emb = nn.Embedding(24, 16)
        self.time_diff_emb = nn.Embedding(8, 16)
        self.duration_bins_emb = nn.Embedding(20, 16)  # 20 duration bins
        
        # Position encoding for sequence order
        max_len = 150  # Increased from 100
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2) * (-math.log(10000.0) / hidden_dim))
        pe = torch.zeros(1, max_len, hidden_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Input projection
        input_dim = hidden_dim + 32 + 16 + 16 + 16 + 16 + 4  # all features
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Deep Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Shared output via embedding weights
        self.register_buffer('temperature', torch.tensor(1.0))
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param, gain=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, locations, users, weekdays, start_mins, durations, time_diffs, lengths):
        B, L = locations.shape
        device = locations.device
        
        # Embeddings
        loc_emb = self.loc_emb(locations)
        user_emb = self.user_emb(users)
        weekday_emb = self.weekday_emb(weekdays)
        time_diff_emb = self.time_diff_emb(time_diffs)
        
        # Hour and temporal features
        hours = torch.div(start_mins, 60, rounding_mode='floor').clamp(0, 23)
        hour_emb = self.hour_emb(hours)
        
        theta = (start_mins.float() / 1440.0) * 2 * math.pi
        time_sin = torch.sin(theta).unsqueeze(-1)
        time_cos = torch.cos(theta).unsqueeze(-1)
        
        # Duration binning (log-scale)
        log_dur = torch.log1p(durations)
        dur_bins = (log_dur * 2).long().clamp(0, 19)
        dur_emb = self.duration_bins_emb(dur_bins)
        
        dur_log = torch.log1p(durations).unsqueeze(-1)
        dur_sqrt = torch.sqrt(durations.clamp(min=0)).unsqueeze(-1)
        
        # Concatenate all features
        x = torch.cat([
            loc_emb, user_emb, weekday_emb, hour_emb, time_diff_emb, dur_emb,
            time_sin, time_cos, dur_log, dur_sqrt
        ], dim=-1)
        
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pe[:, :L, :]
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(L, L, device=device), diagonal=1).bool()
        
        # Create padding mask
        padding_mask = torch.arange(L, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # Apply transformer
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        # Get last valid position  
        batch_indices = torch.arange(B, device=device)
        last_indices = (lengths - 1).clamp(min=0)
        final_repr = x[batch_indices, last_indices]
        
        # Output
        final_repr = self.output_norm(final_repr)
        final_repr = self.dropout(final_repr)
        
        # Use embedding weights for output (weight tying)
        logits = F.linear(final_repr, self.loc_emb.weight)
        
        return logits / self.temperature
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_locations, num_users, hidden_dim=152, num_layers=3, num_heads=4, dropout=0.2):
    model = AdvancedNextLocationPredictor(
        num_locations=num_locations,
        num_users=num_users,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    )
    return model
