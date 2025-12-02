"""
Hybrid Neural-Statistical Next Location Prediction Model.

Combines:
1. Neural sequence encoder
2. Location transition statistics (Markov)
3. User-specific location frequency priors
4. Temporal pattern mining
5. Ensemble prediction

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
        
    def fit(self, train_data):
        """Compute statistics from training data."""
        print("Computing location statistics...")
        
        for sample in train_data:
            locations = sample['X'].tolist()
            user = sample['user_X'][0]  # User ID
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
            # Convert defaultdicts to regular dicts for pickling
            pickle.dump({
                'location_freq': dict(self.location_freq),
                'user_location_freq': {k: dict(v) for k, v in self.user_location_freq.items()},
                'transition_matrix': {k: dict(v) for k, v in self.transition_matrix.items()},
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


class CompactNeuralEncoder(nn.Module):
    """Lightweight neural encoder for sequence representation."""
    
    def __init__(self, num_locations, num_users, hidden_dim=88):
        super().__init__()
        
        # Shared location embedding
        self.loc_emb = nn.Embedding(num_locations + 1, hidden_dim, padding_idx=0)
        self.user_emb = nn.Embedding(num_users + 1, 16, padding_idx=0)
        
        # Temporal encoding
        self.weekday_emb = nn.Embedding(7, 8)
        self.time_diff_emb = nn.Embedding(8, 8)
        
        # Simple attention pooling instead of GRU
        self.query = nn.Parameter(torch.randn(1, hidden_dim + 16 + 18))
        self.attention = nn.Linear(hidden_dim + 16 + 18, 1, bias=False)
        
        # Output
        self.output_proj = nn.Linear(hidden_dim + 16 + 18, hidden_dim)
        
    def forward(self, locations, users, weekdays, start_mins, durations, time_diffs, lengths):
        B, L = locations.shape
        
        # Embeddings
        loc_emb = self.loc_emb(locations)
        user_emb = self.user_emb(users)
        weekday_emb = self.weekday_emb(weekdays)
        time_diff_emb = self.time_diff_emb(time_diffs)
        
        # Temporal features
        theta = (start_mins.float() / 1440.0) * 2 * math.pi
        time_features = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
        
        # Concatenate
        x = torch.cat([loc_emb, user_emb, weekday_emb, time_features, time_diff_emb], dim=-1)
        
        # Attention pooling
        mask = torch.arange(L, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        attn_scores = self.attention(x).squeeze(-1)  # [B, L]
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(1)  # [B, 1, L]
        
        # Weighted sum
        pooled = torch.bmm(attn_weights, x).squeeze(1)  # [B, hidden_dim+16+18]
        
        return self.output_proj(pooled)


class HybridLocationPredictor(nn.Module):
    """
    Hybrid model combining neural encoder with statistical priors.
    """
    
    def __init__(self, num_locations, num_users, hidden_dim=88, stats=None):
        super().__init__()
        
        self.num_locations = num_locations
        self.num_users = num_users
        self.stats = stats
        
        # Neural encoder
        self.encoder = CompactNeuralEncoder(num_locations, num_users, hidden_dim)
        
        # Share location embedding for output (weight tying)
        # This saves num_locations * hidden_dim parameters
        self.use_shared_embedding = True
        
        # Learnable ensemble weights
        self.ensemble_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, locations, users, weekdays, start_mins, durations, time_diffs, lengths, 
                use_stats=True, return_priors=False):
        """
        Args:
            All input tensors
            use_stats: Whether to use statistical priors
            return_priors: Return individual prior components
        """
        B = locations.shape[0]
        device = locations.device
        
        # Neural prediction
        hidden = self.encoder(locations, users, weekdays, start_mins, durations, time_diffs, lengths)
        
        # Use shared embedding weights for output (weight tying)
        neural_logits = F.linear(hidden, self.encoder.loc_emb.weight)  # [B, num_locations]
        
        if not use_stats or self.stats is None:
            return neural_logits
        
        # Compute statistical priors
        stat_priors = torch.zeros_like(neural_logits)
        
        # Global location prior as fallback
        global_prior = self.stats.get_location_prior(self.num_locations)
        
        for i in range(B):
            user = users[i, 0].item()
            last_loc = locations[i, lengths[i]-1].item() if lengths[i] > 0 else 0
            wd = weekdays[i, lengths[i]-1].item() if lengths[i] > 0 else 0
            sm = start_mins[i, lengths[i]-1].item() if lengths[i] > 0 else 0
            
            # Combine multiple statistical signals
            priors = []
            weights = []
            
            # User location frequency (weight: 0.3)
            user_prior = self.stats.get_user_location_prior(user, self.num_locations)
            if user_prior.sum() > 0.01:  # Threshold for reliability
                priors.append(user_prior)
                weights.append(0.3)
            
            # User-specific transitions (weight: 0.35)
            user_trans = self.stats.get_user_transition_prob(user, last_loc, self.num_locations)
            if user_trans.sum() > 0.01:
                priors.append(user_trans)
                weights.append(0.35)
            
            # Global transitions (weight: 0.2)
            global_trans = self.stats.get_transition_prob(last_loc, self.num_locations)
            if global_trans.sum() > 0.01:
                priors.append(global_trans)
                weights.append(0.2)
            
            # Temporal patterns (weight: 0.15)
            temp_prior = self.stats.get_temporal_prior(wd, sm, self.num_locations)
            if temp_prior.sum() > 0.01:
                priors.append(temp_prior)
                weights.append(0.15)
            
            # Weighted combination with fallback to global prior
            if priors:
                weights = np.array(weights)
                weights = weights / weights.sum()
                combined = sum(w * p for w, p in zip(weights, priors))
                combined = combined / (combined.sum() + 1e-8)
            else:
                # Fallback to global prior
                combined = global_prior
                combined = combined / (combined.sum() + 1e-8)
            
            stat_priors[i] = torch.from_numpy(combined).float()
        
        stat_priors = stat_priors.to(device)
        
        if return_priors:
            return neural_logits, stat_priors
        
        # Ensemble: adaptive weighting
        alpha = torch.sigmoid(self.ensemble_weight)
        
        # Convert priors to log space with better numerical stability
        # Add smoothing to avoid log(0)
        stat_priors_smooth = stat_priors + 1e-8
        stat_priors_smooth = stat_priors_smooth / stat_priors_smooth.sum(dim=-1, keepdim=True)
        stat_logits = torch.log(stat_priors_smooth)
        
        # Ensemble with temperature scaling
        final_logits = alpha * neural_logits + (1 - alpha) * stat_logits * 0.5  # Temperature scaling
        
        return final_logits
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(num_locations, num_users, hidden_dim=88, stats=None):
    """Create hybrid model."""
    model = HybridLocationPredictor(
        num_locations=num_locations,
        num_users=num_users,
        hidden_dim=hidden_dim,
        stats=stats
    )
    return model
