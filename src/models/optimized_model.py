"""
Highly Optimized Next Location Prediction Model.

Key optimizations to stay under 1M parameters while maximizing accuracy:
1. Shared embeddings between input and output
2. Lightweight temporal encoding
3. Efficient attention mechanisms
4. Smart parameter allocation

Target: <1M parameters, 70%+ Acc@1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SharedEmbedding(nn.Module):
    """Shared embedding for input and output to reduce parameters."""
    
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        
    def forward(self, x):
        return self.embedding(x)
    
    def compute_logits(self, hidden_states):
        """
        Compute logits by matrix multiplication with embedding weights.
        Args:
            hidden_states: [B, embedding_dim]
        Returns:
            logits: [B, num_embeddings]
        """
        return F.linear(hidden_states, self.embedding.weight)


class CompactTemporalEncoder(nn.Module):
    """Lightweight but effective temporal encoder."""
    
    def __init__(self, hidden_dim):
        super().__init__()
        # Minimal embeddings
        self.weekday_emb = nn.Embedding(7, 8)
        self.time_diff_emb = nn.Embedding(8, 8)
        
        # Compact projection
        self.proj = nn.Linear(20, hidden_dim)  # 8 + 8 + 2 + 2 = 20
        
    def forward(self, weekdays, start_mins, durations, time_diffs):
        """Encode temporal features efficiently."""
        # Weekday and time_diff embeddings
        weekday_emb = self.weekday_emb(weekdays)
        time_diff_emb = self.time_diff_emb(time_diffs)
        
        # Time of day (sin/cos)
        theta = (start_mins.float() / 1440.0) * 2 * math.pi
        time_features = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
        
        # Duration (log + normalized)
        dur_features = torch.stack([
            torch.log1p(durations) / 10.0,
            torch.sqrt(durations) / 50.0
        ], dim=-1)
        
        # Concatenate and project
        features = torch.cat([weekday_emb, time_diff_emb, time_features, dur_features], dim=-1)
        return self.proj(features)


class EfficientTransformerLayer(nn.Module):
    """Efficient Transformer layer with parameter sharing."""
    
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Single projection for Q, K, V
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Lightweight FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, L, _ = x.shape
        
        # Self-attention
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        causal_mask = torch.tril(torch.ones(L, L, device=x.device))
        attn = attn.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0) == 0, float('-inf'))
        
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2).bool(), float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, L, self.hidden_dim)
        out = self.out(out)
        
        # Add & Norm
        x = self.norm1(x + out)
        
        # FFN
        x = self.norm2(x + self.ffn(x))
        
        return x


class OptimizedNextLocationPredictor(nn.Module):
    """
    Optimized model for next location prediction.
    
    Optimizations:
    - Shared embeddings for input/output
    - Efficient attention
    - Minimal parameter count
    """
    
    def __init__(
        self,
        num_locations,
        num_users,
        hidden_dim=88,
        num_heads=4,
        num_layers=2,
        dropout=0.2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Shared location embedding for input and output
        self.loc_emb = SharedEmbedding(num_locations + 1, hidden_dim)
        self.user_emb = nn.Embedding(num_users + 1, 16, padding_idx=0)
        
        # Temporal encoder
        self.temporal_encoder = CompactTemporalEncoder(hidden_dim)
        
        # Input fusion
        self.input_proj = nn.Linear(hidden_dim + 16 + hidden_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            EfficientTransformerLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, locations, users, weekdays, start_mins, durations, time_diffs, lengths):
        B, L = locations.shape
        
        # Create mask
        mask = torch.arange(L, device=locations.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # Embed inputs
        loc_emb = self.loc_emb(locations)
        user_emb = self.user_emb(users)
        temp_emb = self.temporal_encoder(weekdays, start_mins, durations, time_diffs)
        
        # Fuse and project
        x = torch.cat([loc_emb, user_emb, temp_emb], dim=-1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)
        
        # Get final representation
        batch_indices = torch.arange(B, device=x.device)
        last_indices = (lengths - 1).clamp(min=0)
        final_repr = x[batch_indices, last_indices]
        
        # Normalize and compute logits using shared embeddings
        final_repr = self.output_norm(final_repr)
        logits = self.loc_emb.compute_logits(final_repr)
        
        return logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_locations, num_users, hidden_dim=88, num_heads=4, num_layers=2, dropout=0.2):
    model = OptimizedNextLocationPredictor(
        num_locations=num_locations,
        num_users=num_users,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    )
    return model
