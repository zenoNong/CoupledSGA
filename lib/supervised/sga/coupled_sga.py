"""
CoupledSGA: Coupled Graph Neural ODE for Scene Graph Anticipation.

Replaces SceneSayer's independent per-relationship Neural ODE with a coupled
Graph Neural ODE that evolves relationship latent trajectories jointly through
a Relationship Interaction Graph (RIG).

Key components:
    1. RIGBuilder: Constructs the Relationship Interaction Graph from pair indices
    2. TimeAwareGATv2Layer: GATv2 attention with time-embedding injection
    3. CoupledGraphODEFunc: ODE derivative function using stacked GATv2 layers
    4. CoupledSGA: Top-level model extending the SceneSayer pattern
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from lib.supervised.sga.base_ldpu_diffeq import BaseLDPU, BaseLDPUDiffEq


# -------------------------------------------------------------------------------------
# -------------------------------- RIG BUILDER ----------------------------------------
# -------------------------------------------------------------------------------------

class RIGBuilder(nn.Module):
    """
    Builds a Relationship Interaction Graph (RIG) from subject-object pair indices.
    
    Nodes are valid subject-object pairs. An edge exists between two pairs
    if they share any entity (subject or object).
    
    Edge weights are learned via a 2-layer MLP over concatenated node embeddings.
    
    Args:
        d_model: Dimensionality of node embeddings (default: 1936)
        edge_hidden_dim: Hidden dimension of edge weight MLP (default: 256)
    """
    
    def __init__(self, d_model=1936, edge_hidden_dim=256):
        super(RIGBuilder, self).__init__()
        
        # 2-layer MLP for learning edge weights from concatenated node embeddings
        self.edge_weight_mlp = nn.Sequential(
            nn.Linear(2 * d_model, edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(edge_hidden_dim, 1),
            nn.Sigmoid()  # Edge weights in [0, 1]
        )
    
    def build_adjacency(self, pair_idx):
        """
        Build the RIG adjacency from pair indices.
        
        Args:
            pair_idx: (N, 2) tensor of subject-object pair indices for a frame batch
        
        Returns:
            edge_index: (2, E) tensor of edge indices in COO format
        """
        N = pair_idx.size(0)
        if N <= 1:
            # No edges possible with 0 or 1 node
            return torch.zeros(2, 0, dtype=torch.long, device=pair_idx.device)
        
        # For each pair of nodes, check if they share any entity
        # pair_idx[i] = [subj_i, obj_i], pair_idx[j] = [subj_j, obj_j]
        # A[i,j] = 1 if subj_i == subj_j or subj_i == obj_j or obj_i == subj_j or obj_i == obj_j
        
        subj = pair_idx[:, 0]  # (N,)
        obj = pair_idx[:, 1]   # (N,)
        
        # Compute sharing matrix efficiently
        # Shape: (N, N) boolean - True if pairs share any entity
        shares_entity = (
            (subj.unsqueeze(1) == subj.unsqueeze(0)) |  # subj_i == subj_j
            (subj.unsqueeze(1) == obj.unsqueeze(0))  |  # subj_i == obj_j
            (obj.unsqueeze(1) == subj.unsqueeze(0))  |  # obj_i == subj_j
            (obj.unsqueeze(1) == obj.unsqueeze(0))      # obj_i == obj_j
        )
        
        # Remove self-loops
        shares_entity.fill_diagonal_(False)
        
        # Convert to COO edge_index
        edge_index = shares_entity.nonzero(as_tuple=False).t()  # (2, E)
        
        return edge_index
    
    def compute_edge_weights(self, h, edge_index):
        """
        Compute learned edge weights from node embeddings.
        
        Args:
            h: (N, d_model) node embeddings
            edge_index: (2, E) edge indices
        
        Returns:
            edge_weight: (E,) learned edge weights
        """
        if edge_index.size(1) == 0:
            return torch.zeros(0, device=h.device)
        
        src, dst = edge_index[0], edge_index[1]
        # Concatenate source and destination embeddings
        edge_features = torch.cat([h[src], h[dst]], dim=1)  # (E, 2*d_model)
        edge_weight = self.edge_weight_mlp(edge_features).squeeze(-1)  # (E,)
        
        return edge_weight
    
    def forward(self, h, pair_idx):
        """
        Build the full RIG with adjacency and learned edge weights.
        
        Args:
            h: (N, d_model) node embeddings
            pair_idx: (N, 2) subject-object pair indices
        
        Returns:
            edge_index: (2, E) edge indices
            edge_weight: (E,) learned edge weights
        """
        edge_index = self.build_adjacency(pair_idx)
        edge_weight = self.compute_edge_weights(h, edge_index)
        return edge_index, edge_weight


# -------------------------------------------------------------------------------------
# ----------------------------- TIME-AWARE GATv2 LAYER --------------------------------
# -------------------------------------------------------------------------------------

class TimeAwareGATv2Layer(nn.Module):
    """
    Single GATv2 attention layer with time-embedding injection.
    
    Implements the derivative formula:
        dh_i/dt = sigma(sum_j alpha_{i,j}(t) W_v h_j(t)) - h_i(t)
    
    Time embedding phi(t) is injected into the attention score computation.
    
    Args:
        d_model: Input/output feature dimension (default: 1936)
        num_heads: Number of attention heads (default: 4)
        dropout: Dropout rate (default: 0.1)
        time_embed_dim: Dimension of time embedding (default: 64)
    """
    
    def __init__(self, d_model=1936, num_heads=4, dropout=0.1, time_embed_dim=64):
        super(TimeAwareGATv2Layer, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        # GATv2 projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Attention vector (per head)
        self.attn_vec = nn.Parameter(torch.zeros(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))
        
        # Time embedding MLP: scalar t -> time_embed_dim -> head_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, self.head_dim)
        )
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        # LayerNorm and residual FFN
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, h, t, edge_index, edge_weight=None):
        """
        Forward pass computing dh/dt via GATv2 attention.
        
        Args:
            h: (N, d_model) node features
            t: scalar time value
            edge_index: (2, E) edge indices
            edge_weight: (E,) optional learned edge weights
        
        Returns:
            dh_dt: (N, d_model) derivative of node features
        """
        N = h.size(0)
        
        if edge_index.size(1) == 0:
            # No edges: derivative is just -h (decay to zero)
            return -h
        
        src, dst = edge_index[0], edge_index[1]
        
        # GATv2: apply linear transforms then LeakyReLU
        q = self.W_q(h)  # (N, d_model)
        k = self.W_k(h)  # (N, d_model)
        v = self.W_v(h)  # (N, d_model)
        
        # Reshape for multi-head: (N, num_heads, head_dim)
        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_heads, self.head_dim)
        v = v.view(N, self.num_heads, self.head_dim)
        
        # Time embedding: inject into query-key attention
        t_tensor = t.float().view(1, 1) if isinstance(t, torch.Tensor) else torch.tensor([[t]], device=h.device, dtype=h.dtype)
        phi_t = self.time_embed(t_tensor)  # (1, head_dim)
        
        # GATv2 attention: e_{ij} = attn_vec^T * LeakyReLU(q_i + k_j + phi_t)
        # Gather source and destination features
        q_src = q[dst]  # (E, num_heads, head_dim) — query from destination (attending node)
        k_dst = k[src]  # (E, num_heads, head_dim) — key from source (neighbor)
        
        # Add time embedding to all edges
        attn_input = self.leaky_relu(q_src + k_dst + phi_t.unsqueeze(0))  # (E, num_heads, head_dim)
        
        # Compute attention logits
        attn_logits = (attn_input * self.attn_vec.unsqueeze(0)).sum(dim=-1)  # (E, num_heads)
        
        # Apply learned edge weights if provided
        if edge_weight is not None:
            attn_logits = attn_logits * edge_weight.unsqueeze(-1)  # (E, num_heads)
        
        # Softmax over neighbors (per destination node, per head)
        # Use scatter-based softmax for sparse graphs
        attn_scores = self._sparse_softmax(attn_logits, dst, N)  # (E, num_heads)
        attn_scores = self.dropout(attn_scores)
        
        # Aggregate: sum_j alpha_{ij} * v_j
        v_src = v[src]  # (E, num_heads, head_dim)
        weighted_v = v_src * attn_scores.unsqueeze(-1)  # (E, num_heads, head_dim)
        
        # Scatter-add to destination nodes
        agg = torch.zeros(N, self.num_heads, self.head_dim, device=h.device, dtype=weighted_v.dtype)
        dst_expanded = dst.unsqueeze(-1).unsqueeze(-1).expand_as(weighted_v)
        agg.scatter_add_(0, dst_expanded, weighted_v)
        agg = agg.to(h.dtype)
        
        # Reshape back to (N, d_model)
        agg = agg.view(N, self.d_model)
        
        # Apply activation
        agg = torch.sigmoid(agg)
        
        # Residual decay term: -h_i(t)
        dh_dt = agg - h
        
        # LayerNorm + Residual FFN (applied to the aggregation, not the derivative)
        out = self.layer_norm1(agg)
        out = out + self.ffn(out)
        out = self.layer_norm2(out)
        
        # Final derivative: transformed aggregation - current state
        dh_dt = out - h
        
        return dh_dt
    
    @staticmethod
    def _sparse_softmax(logits, index, num_nodes):
        """
        Compute softmax over sparse edges grouped by destination node.
        
        Args:
            logits: (E, num_heads) attention logits
            index: (E,) destination node indices
            num_nodes: Total number of nodes
        
        Returns:
            scores: (E, num_heads) normalized attention scores
        """
        # For numerical stability, subtract max per destination
        logits_max = torch.zeros(num_nodes, logits.size(1), device=logits.device, dtype=logits.dtype)
        logits_max.scatter_reduce_(0, index.unsqueeze(-1).expand_as(logits), logits, reduce='amax', include_self=True)
        logits = logits - logits_max[index]
        
        exp_logits = torch.exp(logits)
        
        # Sum of exp per destination node
        exp_sum = torch.zeros(num_nodes, logits.size(1), device=logits.device, dtype=logits.dtype)
        exp_sum.scatter_add_(0, index.unsqueeze(-1).expand_as(exp_logits), exp_logits)
        
        # Normalize
        scores = exp_logits / (exp_sum[index] + 1e-8)
        
        return scores


# -------------------------------------------------------------------------------------
# ----------------------------- COUPLED GRAPH ODE FUNC --------------------------------
# -------------------------------------------------------------------------------------

class CoupledGraphODEFunc(nn.Module):
    """
    ODE derivative function for the Coupled Graph Neural ODE.
    
    Contains 2 stacked TimeAwareGATv2Layer modules. The graph topology
    (edge_index, edge_weight) must be set before calling forward.
    
    Args:
        d_model: Feature dimension (default: 1936)
        num_heads: Number of GATv2 attention heads (default: 4)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self, d_model=1936, num_heads=4, dropout=0.1):
        super(CoupledGraphODEFunc, self).__init__()
        
        self.gat_layer1 = TimeAwareGATv2Layer(d_model, num_heads, dropout)
        self.gat_layer2 = TimeAwareGATv2Layer(d_model, num_heads, dropout)
    
    def forward(self, t, h, edge_index, edge_weight):
        """
        Compute dh/dt given current time t and node states h.
        
        Args:
            t: Scalar time value
            h: (N, d_model) node states
            edge_index: Graph edges
            edge_weight: Graph edge weights
        
        Returns:
            dh_dt: (N, d_model) derivative of node states
        """
        dh1 = self.gat_layer1(h, t, edge_index, edge_weight)
        # Apply first layer's derivative, then compute second layer's
        h_intermediate = h + dh1 * 0.1  # Small residual for stability
        dh2 = self.gat_layer2(h_intermediate, t, edge_index, edge_weight)
        
        # Average the two layers' derivatives
        dh_dt = (dh1 + dh2) / 2.0
        
        return dh_dt


# -------------------------------------------------------------------------------------
# ------------------------------ COUPLED SGA MODEL ------------------------------------
# -------------------------------------------------------------------------------------

class CoupledSGA(BaseLDPUDiffEq):
    """
    CoupledSGA: Coupled Graph Neural ODE for Scene Graph Anticipation.
    
    Extends BaseLDPUDiffEq (the SceneSayer base) by replacing the independent
    per-relationship ODE with a coupled Graph Neural ODE.
    
    The STTran backbone (BaseLDPU) is reused via composition — no backbone changes.
    
    Args:
        mode: Task mode ('sgdet', 'sgcls', 'predcls')
        attention_class_num: Number of attention predicate classes
        spatial_class_num: Number of spatial predicate classes
        contact_class_num: Number of contacting predicate classes
        obj_classes: Object class names
        max_window: Maximum anticipation window
        euler_step_size: Step size for Euler integration (default: 0.25)
        gat_heads: Number of GATv2 attention heads (default: 4)
    """
    
    def __init__(
            self,
            mode,
            attention_class_num=None,
            spatial_class_num=None,
            contact_class_num=None,
            obj_classes=None,
            rel_classes=None,
            enc_layer_num=None,
            dec_layer_num=None,
            max_window=None,
            euler_step_size=0.25,
            gat_heads=4
    ):
        super(CoupledSGA, self).__init__(
            mode=mode,
            attention_class_num=attention_class_num,
            spatial_class_num=spatial_class_num,
            contact_class_num=contact_class_num,
            obj_classes=obj_classes,
            rel_classes=rel_classes,
            max_window=max_window
        )
        
        self.d_model = 1936
        self.euler_step_size = euler_step_size
        
        # STTran backbone (same as SceneSayer)
        self.base_ldpu = BaseLDPU(
            mode=mode,
            attention_class_num=attention_class_num,
            spatial_class_num=spatial_class_num,
            contact_class_num=contact_class_num,
            obj_classes=obj_classes
        )
        # Alias for compatibility with BaseLDPUDiffEq.process_ff_rels_output_entry
        self.dsgdetr = self.base_ldpu
        
        # CoupledSGA-specific components
        self.rig_builder = RIGBuilder(d_model=self.d_model, edge_hidden_dim=256)
        self.coupled_ode_func = CoupledGraphODEFunc(
            d_model=self.d_model,
            num_heads=gat_heads,
            dropout=0.1
        )
    
    def _euler_integrate(self, h0, times, pair_idx):
        """
        Explicit Euler integration of the coupled Graph Neural ODE.
        
        Args:
            h0: (N, d_model) initial node states at t=times[0]
            times: (T,) tensor of time points to evaluate at
            pair_idx: (N, 2) pair indices for RIG construction
        
        Returns:
            trajectory: (T-1, N, d_model) states at times[1:] (excluding initial)
        """
        tau = self.euler_step_size
        device = h0.device
        
        # Build the RIG from current pair indices
        # Wait, rig_builder returns edge_index and edge_weight
        # But wait, in the old code rig_builder(pair_idx, h0.size(0)). Here it was changed to rig_builder(h0, pair_idx). 
        edge_index, edge_weight = self.rig_builder(h0, pair_idx)
        
        # Compute integration grid
        t_start = times[0].item()
        results = []
        
        h = h0
        current_t = t_start
        
        with torch.amp.autocast('cuda', enabled=True):
            for target_t in times[1:]:
                target = target_t.item()
                
                # Euler steps from current_t to target for stable numerical integration
                while current_t < target - 1e-6:
                    step = min(tau, target - current_t)
                    t_tensor = torch.tensor(current_t, device=device, dtype=h.dtype)
                    
                    dh_dt = self.coupled_ode_func(t_tensor, h, edge_index, edge_weight)
                    
                    h = h + step * dh_dt
                    current_t += step
                    
                results.append(h.clone())
        
        if len(results) == 0:
            return torch.zeros(0, h0.size(0), h0.size(1), device=device, dtype=torch.float32)
        
        return torch.stack(results, dim=0).float()  # (T-1, N, d_model) back to FP32
    
    def forward(self, entry, testing=False):
        """
        Forward pass.
        
        Args:
            entry: Dict containing video data (from object detector)
            testing: Whether in test mode
        
        Returns:
            entry: Updated dict with anticipated distributions
        """
        # Process through STTran backbone + build frame ranges/masks
        entry, num_frames, frames_ranges, times_unique, window, global_output, anticipated_vals = \
            self.process_ff_rels(entry, testing)
        
        # Coupled ODE integration: iterate over frames
        curr_id = 0
        pair_idx = entry["pair_idx"]
        
        with torch.amp.autocast('cuda', enabled=True):
            global_output_fp16 = global_output.to(torch.float16)
            for i in range(num_frames - 1):
                end = frames_ranges[i + 1]
                if curr_id == end:
                    continue
                
                batch_y0 = global_output_fp16[curr_id:end]
                batch_times = times_unique[i: i + window + 1]
                batch_pair_idx = pair_idx[curr_id:end]
                
                # Use coupled Euler integration instead of independent odeint
                ret = self._euler_integrate(batch_y0, batch_times, batch_pair_idx)
                
                anticipated_vals = torch.cat((anticipated_vals, ret), dim=1)
                curr_id = end
        
        # Process outputs (same as SceneSayer)
        entry = self.process_ff_rels_output_entry(entry, anticipated_vals.float())
        
        return entry
    
    def forward_single_entry(self, context_fraction, entry):
        """
        Forward pass for context-fraction evaluation (test time).
        
        Args:
            context_fraction: Fraction of frames to use as context (0, 1)
            entry: Dict containing video data
        
        Returns:
            end: Index where context ends
            pred: Dict with predicted distributions
        """
        entry, end, num_frames, frames_ranges, global_output, pair_idx, im_idx, times = \
            self.process_context_fraction_rels(entry, context_fraction)
        
        pred = {}
        
        if end == num_frames - 1 or frames_ranges[end] == frames_ranges[end + 1]:
            return num_frames, pred
        
        batch_y0 = entry["global_output"][frames_ranges[end]:frames_ranges[end + 1]]
        batch_pair_idx = pair_idx[frames_ranges[end]:frames_ranges[end + 1]]
        batch_times = times[end:]
        
        # Use coupled Euler integration
        ret = self._euler_integrate(batch_y0, batch_times, batch_pair_idx)
        
        pred = self.process_context_fraction_output_pred(
            entry, end, num_frames, frames_ranges, pair_idx, im_idx, ret, pred
        )
        
        return end + 1, pred
