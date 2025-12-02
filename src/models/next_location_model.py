"""
Advanced Next Location Prediction Model with Multi-Feature Fusion.

Efficient architecture optimized for <1M parameters while achieving 70%+ Acc@1:
1. Compact location and user embeddings
2. Efficient multi-scale temporal encoding
3. Lightweight multi-head attention
4. Strategic feature fusion

Target: <1M parameters, 70%+ Acc@1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TemporalEncoder(nn.Module):
    """Lightweight temporal encoder using concatenation and projection."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Compact embeddings
        self.weekday_emb = nn.Embedding(7, 8)
        self.time_diff_emb = nn.Embedding(8, 8)
        
        # Total: 8 (weekday) + 2 (sin/cos time) + 1 (duration) + 8 (time_diff) = 19
        self.proj = nn.Linear(19, hidden_dim)
    
    def forward(self, weekdays, start_mins, durations, time_diffs):
        """
        Args:
            weekdays: [B, L]
            start_mins: [B, L]
            durations: [B, L]
            time_diffs: [B, L]
        Returns:
            temporal_features: [B, L, hidden_dim]
        """
        # Embeddings
        weekday_emb = self.weekday_emb(weekdays)  # [B, L, 8]
        time_diff_emb = self.time_diff_emb(time_diffs)  # [B, L, 8]
        
        # Time of day (sin/cos encoding)
        theta = (start_mins.float() / 1440.0) * 2 * math.pi
        time_sin = torch.sin(theta).unsqueeze(-1)
        time_cos = torch.cos(theta).unsqueeze(-1)
        
        # Duration (log-normalized)
        dur_norm = (torch.log1p(durations) / 10.0).unsqueeze(-1)  # [B, L, 1]
        
        # Concatenate all features
        features = torch.cat([weekday_emb, time_sin, time_cos, dur_norm, time_diff_emb], dim=-1)
        
        # Project to hidden_dim
        return self.proj(features)


class EfficientAttention(nn.Module):
    """Efficient multi-head attention without relative positions."""
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, L, hidden_dim]
            mask: [B, L] - padding mask
        Returns:
            output: [B, L, hidden_dim]
        """
        B, L, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, L, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, L, L]
        
        # Apply mask
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn = attn.masked_fill(~attn_mask.bool(), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # [B, num_heads, L, head_dim]
        out = out.transpose(1, 2).reshape(B, L, self.hidden_dim)
        out = self.out_proj(out)
        
        return out


class NextLocationPredictor(nn.Module):
    """
    Efficient next location prediction model.
    
    Architecture:
    1. Compact embeddings for locations and users
    2. Efficient temporal encoding
    3. Simple feature fusion
    4. Lightweight self-attention
    5. Output prediction
    
    Target: <1M parameters, 70%+ Acc@1
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Compact embeddings - location embedding is the bottleneck
        self.loc_emb = nn.Embedding(num_locations + 1, hidden_dim, padding_idx=0)
        self.user_emb = nn.Embedding(num_users + 1, 16, padding_idx=0)
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(hidden_dim)
        
        # Input fusion: location + user + temporal
        self.input_fusion = nn.Linear(hidden_dim + 16 + hidden_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            EfficientAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Lightweight FFN
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, num_locations + 1)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, locations, users, weekdays, start_mins, durations, time_diffs, lengths):
        """
        Args:
            locations: [B, L]
            users: [B, L]
            weekdays: [B, L]
            start_mins: [B, L]
            durations: [B, L]
            time_diffs: [B, L]
            lengths: [B]
        Returns:
            logits: [B, num_locations]
        """
        B, L = locations.shape
        
        # Create padding mask
        mask = torch.arange(L, device=locations.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # Embeddings
        loc_emb = self.loc_emb(locations)  # [B, L, hidden_dim]
        user_emb = self.user_emb(users)  # [B, L, 16]
        
        # Temporal features
        temp_emb = self.temporal_encoder(weekdays, start_mins, durations, time_diffs)
        
        # Fuse all features
        x = torch.cat([loc_emb, user_emb, temp_emb], dim=-1)
        x = self.input_fusion(x)
        x = self.input_norm(x)
        x = self.dropout(x)
        
        # Apply self-attention layers
        for i in range(len(self.attention_layers)):
            # Self-attention with residual
            attn_out = self.attention_layers[i](x, mask)
            x = self.layer_norms[i](x + attn_out)
            
            # Feed-forward with residual
            ffn_out = self.ffn_layers[i](x)
            x = x + ffn_out
        
        # Get final representation (use last valid token)
        batch_indices = torch.arange(B, device=x.device)
        last_indices = (lengths - 1).clamp(min=0)
        final_repr = x[batch_indices, last_indices]  # [B, hidden_dim]
        
        # Output projection
        final_repr = self.output_norm(final_repr)
        logits = self.output_proj(final_repr)  # [B, num_locations]
        
        return logits
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_locations, num_users, hidden_dim=64, num_heads=4, num_layers=2, dropout=0.2):
    """Factory function to create the model."""
    model = NextLocationPredictor(
        num_locations=num_locations,
        num_users=num_users,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    return model
