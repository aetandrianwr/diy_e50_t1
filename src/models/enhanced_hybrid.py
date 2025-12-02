"""
Enhanced Hybrid Neural-Statistical Next Location Prediction Model.

With 2M parameter budget:
- Larger embeddings
- Deeper networks
- Better statistical integration
- Multi-head attention layers

Target: 70%+ Acc@1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import pickle
import math


class LocationStatistics:
    """Precompute statistical features from training data."""
    
    def __init__(self):
        self.location_freq = defaultdict(int)
        self.user_location_freq = defaultdict(lambda: defaultdict(int))
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        self.user_transition = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.temporal_patterns = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.bigram_transitions = defaultdict(lambda: defaultdict(int))  # Last 2 locations
        
    def fit(self, train_data):
        """Compute statistics from training data."""
        print("Computing enhanced location statistics...")
        
        for sample in train_data:
            locations = sample['X'].tolist()
            user = sample['user_X'][0]
            target = sample['Y']
            weekday = sample['weekday_X'][-1]
            start_min = sample['start_min_X'][-1]
            
            # Global location frequency
            for loc in locations:
                self.location_freq[loc] += 1
            self.location_freq[target] += 1
            
            # User-specific location frequency
            for loc in locations:
                self.user_location_freq[user][loc] += 1
            self.user_location_freq[user][target] += 1
            
            # Transition probabilities
            for i in range(len(locations) - 1):
                self.transition_matrix[locations[i]][locations[i+1]] += 1
            if len(locations) > 0:
                self.transition_matrix[locations[-1]][target] += 1
            
            # Bigram transitions (last 2 -> next)
            if len(locations) >= 2:
                bigram = (locations[-2], locations[-1])
                self.bigram_transitions[bigram][target] += 1
            
            # User-specific transitions
            for i in range(len(locations) - 1):
                self.user_transition[user][locations[i]][locations[i+1]] += 1
            if len(locations) > 0:
                self.user_transition[user][locations[-1]][target] += 1
            
            # Temporal patterns (hour of day)
            hour = start_min // 60
            for loc in locations:
                self.temporal_patterns[weekday][hour][loc] += 1
            self.temporal_patterns[weekday][hour][target] += 1
        
        print(f"Unique locations: {len(self.location_freq)}")
        print(f"Unique users: {len(self.user_location_freq)}")
        print(f"Unique transitions: {len(self.transition_matrix)}")
        print(f"Unique bigrams: {len(self.bigram_transitions)}")
        print("Statistics computed!")
        
    def get_location_prior(self, num_locations):
        """Get global location frequency prior."""
        prior = np.zeros(num_locations + 1)
        total = sum(self.location_freq.values())
        for loc, count in self.location_freq.items():
            if loc < len(prior):
                prior[loc] = count / total
        return prior
    
    def get_user_location_prior(self, user, num_locations):
        """Get user-specific location prior."""
        prior = np.zeros(num_locations + 1)
        user_freqs = self.user_location_freq[user]
        total = sum(user_freqs.values())
        if total > 0:
            for loc, count in user_freqs.items():
                if loc < len(prior):
                    prior[loc] = count / total
        return prior
    
    def get_transition_prob(self, last_loc, num_locations):
        """Get transition probabilities from last location."""
        probs = np.zeros(num_locations + 1)
        transitions = self.transition_matrix[last_loc]
        total = sum(transitions.values())
        if total > 0:
            for loc, count in transitions.items():
                if loc < len(probs):
                    probs[loc] = count / total
        return probs
    
    def get_bigram_transition_prob(self, loc1, loc2, num_locations):
        """Get transition probabilities from last 2 locations."""
        probs = np.zeros(num_locations + 1)
        bigram = (loc1, loc2)
        transitions = self.bigram_transitions[bigram]
        total = sum(transitions.values())
        if total > 0:
            for loc, count in transitions.items():
                if loc < len(probs):
                    probs[loc] = count / total
        return probs
    
    def get_user_transition_prob(self, user, last_loc, num_locations):
        """Get user-specific transition probabilities."""
        probs = np.zeros(num_locations + 1)
        transitions = self.user_transition[user][last_loc]
        total = sum(transitions.values())
        if total > 0:
            for loc, count in transitions.items():
                if loc < len(probs):
                    probs[loc] = count / total
        return probs
    
    def get_temporal_prior(self, weekday, start_min, num_locations):
        """Get temporal location prior."""
        probs = np.zeros(num_locations + 1)
        hour = start_min // 60
        patterns = self.temporal_patterns[weekday][hour]
        total = sum(patterns.values())
        if total > 0:
            for loc, count in patterns.items():
                if loc < len(probs):
                    probs[loc] = count / total
        return probs
    
    def save(self, path):
        """Save statistics to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'location_freq': dict(self.location_freq),
                'user_location_freq': {k: dict(v) for k, v in self.user_location_freq.items()},
                'transition_matrix': {k: dict(v) for k, v in self.transition_matrix.items()},
                'bigram_transitions': {k: dict(v) for k, v in self.bigram_transitions.items()},
                'user_transition': {k: {k2: dict(v2) for k2, v2 in v.items()} 
                                   for k, v in self.user_transition.items()},
                'temporal_patterns': {k: {k2: dict(v2) for k2, v2 in v.items()} 
                                     for k, v in self.temporal_patterns.items()}
            }, f)
    
    def load(self, path):
        """Load statistics from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.location_freq = defaultdict(int, data['location_freq'])
            self.user_location_freq = defaultdict(lambda: defaultdict(int))
            for k, v in data['user_location_freq'].items():
                self.user_location_freq[k] = defaultdict(int, v)
            self.transition_matrix = defaultdict(lambda: defaultdict(int))
            for k, v in data['transition_matrix'].items():
                self.transition_matrix[k] = defaultdict(int, v)
            self.bigram_transitions = defaultdict(lambda: defaultdict(int))
            for k, v in data.get('bigram_transitions', {}).items():
                self.bigram_transitions[k] = defaultdict(int, v)
            self.user_transition = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
            for k, v in data['user_transition'].items():
                self.user_transition[k] = defaultdict(lambda: defaultdict(int))
                for k2, v2 in v.items():
                    self.user_transition[k][k2] = defaultdict(int, v2)
            self.temporal_patterns = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
            for k, v in data['temporal_patterns'].items():
                self.temporal_patterns[k] = defaultdict(lambda: defaultdict(int))
                for k2, v2 in v.items():
                    self.temporal_patterns[k][k2] = defaultdict(int, v2)


class EnhancedNeuralEncoder(nn.Module):
    """Enhanced neural encoder with more capacity."""
    
    def __init__(self, num_locations, num_users, hidden_dim=160, num_layers=3, num_heads=4, dropout=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Embeddings
        self.loc_emb = nn.Embedding(num_locations + 1, hidden_dim, padding_idx=0)
        self.user_emb = nn.Embedding(num_users + 1, 32, padding_idx=0)
        
        # Temporal encoding
        self.weekday_emb = nn.Embedding(7, 16)
        self.hour_emb = nn.Embedding(24, 16)
        self.time_diff_emb = nn.Embedding(8, 16)
        
        # Input projection
        input_dim = hidden_dim + 32 + 16 + 16 + 16 + 4  # loc + user + weekday + hour + time_diff + time_sin_cos + dur
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Transformer layers
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
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, locations, users, weekdays, start_mins, durations, time_diffs, lengths):
        B, L = locations.shape
        device = locations.device
        
        # Embeddings
        loc_emb = self.loc_emb(locations)
        user_emb = self.user_emb(users)
        weekday_emb = self.weekday_emb(weekdays)
        time_diff_emb = self.time_diff_emb(time_diffs)
        
        # Hour of day
        hours = torch.div(start_mins, 60, rounding_mode='floor').clamp(0, 23)
        hour_emb = self.hour_emb(hours)
        
        # Time features
        theta = (start_mins.float() / 1440.0) * 2 * math.pi
        time_sin = torch.sin(theta).unsqueeze(-1)
        time_cos = torch.cos(theta).unsqueeze(-1)
        
        # Duration features
        dur_log = torch.log1p(durations).unsqueeze(-1)
        dur_sqrt = torch.sqrt(durations.clamp(min=0)).unsqueeze(-1)
        
        # Concatenate all features
        x = torch.cat([
            loc_emb, user_emb, weekday_emb, hour_emb, time_diff_emb,
            time_sin, time_cos, dur_log, dur_sqrt
        ], dim=-1)
        
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.dropout(x)
        
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
        
        return self.output_norm(final_repr)


class EnhancedHybridLocationPredictor(nn.Module):
    """
    Enhanced hybrid model with 2M parameter budget.
    """
    
    def __init__(self, num_locations, num_users, hidden_dim=160, num_layers=3, num_heads=4, 
                 dropout=0.2, stats=None):
        super().__init__()
        
        self.num_locations = num_locations
        self.num_users = num_users
        self.stats = stats
        
        # Neural encoder
        self.encoder = EnhancedNeuralEncoder(
            num_locations, num_users, hidden_dim, num_layers, num_heads, dropout
        )
        
        # Use shared embedding for output (weight tying)
        # This saves significant parameters
        self.use_shared_output = True
        
        # Small projection to match embedding dimension
        if hidden_dim != self.encoder.loc_emb.embedding_dim:
            self.hidden_to_emb = nn.Linear(hidden_dim, self.encoder.loc_emb.embedding_dim)
        else:
            self.hidden_to_emb = nn.Identity()
        
        # Learnable ensemble weights
        self.ensemble_weight = nn.Parameter(torch.tensor(0.6))  # Start with 60% neural
        
    def forward(self, locations, users, weekdays, start_mins, durations, time_diffs, lengths, 
                use_stats=True, return_priors=False):
        B = locations.shape[0]
        device = locations.device
        
        # Neural prediction
        hidden = self.encoder(locations, users, weekdays, start_mins, durations, time_diffs, lengths)
        
        # Project to embedding dimension and use shared weights
        hidden_emb = self.hidden_to_emb(hidden)
        neural_logits = F.linear(hidden_emb, self.encoder.loc_emb.weight)
        
        if not use_stats or self.stats is None:
            return neural_logits
        
        # Compute statistical priors (optimized for speed)
        stat_priors = self._compute_statistical_priors(
            locations, users, weekdays, start_mins, lengths, device
        )
        
        if return_priors:
            return neural_logits, stat_priors
        
        # Ensemble with better numerical stability
        alpha = torch.sigmoid(self.ensemble_weight).clamp(0.3, 0.9)
        
        # Normalize statistical priors
        stat_priors = stat_priors + 1e-8
        stat_priors = stat_priors / stat_priors.sum(dim=-1, keepdim=True)
        
        # Convert to logits
        stat_logits = torch.log(stat_priors.clamp(min=1e-8))
        
        # Ensemble
        final_logits = alpha * neural_logits + (1 - alpha) * stat_logits
        
        return final_logits
    
    def _compute_statistical_priors(self, locations, users, weekdays, start_mins, lengths, device):
        """Optimized statistical prior computation."""
        B = locations.shape[0]
        stat_priors = torch.zeros(B, self.num_locations + 1, device=device)
        
        # Global fallback
        global_prior = torch.from_numpy(
            self.stats.get_location_prior(self.num_locations)
        ).float().to(device)
        
        for i in range(B):
            user = users[i, 0].item()
            seq_len = lengths[i].item()
            
            if seq_len == 0:
                stat_priors[i] = global_prior
                continue
                
            last_loc = locations[i, seq_len-1].item()
            wd = weekdays[i, seq_len-1].item()
            sm = start_mins[i, seq_len-1].item()
            
            # Collect priors
            priors = []
            weights = []
            
            # User-specific transitions (highest weight)
            user_trans = self.stats.get_user_transition_prob(user, last_loc, self.num_locations)
            if user_trans.sum() > 0.01:
                priors.append(user_trans)
                weights.append(0.4)
            
            # Bigram transitions (if available)
            if seq_len >= 2:
                loc2 = locations[i, seq_len-2].item()
                bigram_trans = self.stats.get_bigram_transition_prob(loc2, last_loc, self.num_locations)
                if bigram_trans.sum() > 0.01:
                    priors.append(bigram_trans)
                    weights.append(0.25)
            
            # User location frequency
            user_prior = self.stats.get_user_location_prior(user, self.num_locations)
            if user_prior.sum() > 0.01:
                priors.append(user_prior)
                weights.append(0.15)
            
            # Global transitions
            global_trans = self.stats.get_transition_prob(last_loc, self.num_locations)
            if global_trans.sum() > 0.01:
                priors.append(global_trans)
                weights.append(0.1)
            
            # Temporal patterns
            temp_prior = self.stats.get_temporal_prior(wd, sm, self.num_locations)
            if temp_prior.sum() > 0.01:
                priors.append(temp_prior)
                weights.append(0.1)
            
            # Combine
            if priors:
                weights = np.array(weights)
                weights = weights / weights.sum()
                combined = sum(w * p for w, p in zip(weights, priors))
                combined = combined / (combined.sum() + 1e-8)
                stat_priors[i] = torch.from_numpy(combined).float()
            else:
                stat_priors[i] = global_prior
        
        return stat_priors
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_locations, num_users, hidden_dim=144, num_layers=3, num_heads=4, 
                 dropout=0.2, stats=None):
    """Create enhanced hybrid model."""
    model = EnhancedHybridLocationPredictor(
        num_locations=num_locations,
        num_users=num_users,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        stats=stats
    )
    return model
