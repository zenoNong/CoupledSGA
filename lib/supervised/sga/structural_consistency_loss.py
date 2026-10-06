"""
Structural Consistency Loss for CoupledSGA.

Penalizes mutually exclusive predicate co-activations in the spatial and
contacting predicate heads (multi-label sigmoids).

L_SC = (1/N) * Tr(Y_hat @ M @ Y_hat^T)

where M is a symmetric binary constraint matrix encoding which predicate
pairs are physically mutually exclusive.
"""

import torch
import torch.nn as nn


def build_spatial_constraint_matrix():
    """
    Build constraint matrix for spatial predicates (6 classes).
    
    Action Genome spatial predicates (indices 0-5):
        0: above
        1: beneath
        2: in_front_of
        3: behind
        4: on_the_side_of
        5: in
    
    Mutually exclusive pairs:
        - above (0) <-> beneath (1)
        - in_front_of (2) <-> behind (3)
    """
    num_classes = 6
    M = torch.zeros(num_classes, num_classes)
    
    # above <-> beneath
    M[0, 1] = 1.0
    M[1, 0] = 1.0
    
    # in_front_of <-> behind
    M[2, 3] = 1.0
    M[3, 2] = 1.0
    
    return M


def build_contacting_constraint_matrix():
    """
    Build constraint matrix for contacting predicates (17 classes).
    
    Action Genome contacting predicates (indices 0-16):
        0: carrying
        1: covered_by
        2: drinking_from
        3: eating
        4: have_it_on_the_back
        5: holding
        6: leaning_on
        7: lying_on
        8: not_contacting
        9: other_relationship
        10: sitting_on
        11: standing_on
        12: touching
        13: twisting
        14: wearing
        15: wiping
        16: writing_on
    
    Mutually exclusive pairs:
        - sitting_on (10) <-> standing_on (11)
        - lying_on (7) <-> standing_on (11)
        - lying_on (7) <-> sitting_on (10)
        - not_contacting (8) <-> all other contact predicates (0-7, 9-16)
    """
    num_classes = 17
    M = torch.zeros(num_classes, num_classes)
    
    # sitting_on <-> standing_on
    M[10, 11] = 1.0
    M[11, 10] = 1.0
    
    # lying_on <-> standing_on
    M[7, 11] = 1.0
    M[11, 7] = 1.0
    
    # lying_on <-> sitting_on
    M[7, 10] = 1.0
    M[10, 7] = 1.0
    
    # not_contacting <-> all other contacting predicates
    not_contacting_idx = 8
    for i in range(num_classes):
        if i != not_contacting_idx:
            M[not_contacting_idx, i] = 1.0
            M[i, not_contacting_idx] = 1.0
    
    return M


class StructuralConsistencyLoss(nn.Module):
    """
    Structural Consistency Loss that penalizes co-activation of mutually
    exclusive predicates.
    
    L_SC = lambda_sc * (1/N) * [Tr(Y_spatial @ M_spatial @ Y_spatial^T)
                                + Tr(Y_contact @ M_contact @ Y_contact^T)]
    
    Args:
        spatial_num_classes: Number of spatial predicate classes (default: 6)
        contacting_num_classes: Number of contacting predicate classes (default: 17)
        lambda_sc: Loss weight (default: 0.1)
    """
    
    def __init__(self, spatial_num_classes=6, contacting_num_classes=17, lambda_sc=0.1):
        super(StructuralConsistencyLoss, self).__init__()
        self.lambda_sc = lambda_sc
        
        # Build and register constraint matrices as buffers (not parameters)
        M_spatial = build_spatial_constraint_matrix()
        M_contacting = build_contacting_constraint_matrix()
        
        self.register_buffer('M_spatial', M_spatial)
        self.register_buffer('M_contacting', M_contacting)
    
    def _compute_trace_loss(self, Y_hat, M):
        """
        Compute (1/N) * Tr(Y_hat @ M @ Y_hat^T)
        
        Args:
            Y_hat: (N, C) predicted sigmoid probabilities
            M: (C, C) constraint matrix
        
        Returns:
            Scalar loss value
        """
        if Y_hat.size(0) == 0:
            return torch.tensor(0.0, device=Y_hat.device)
        
        N = Y_hat.size(0)
        # Y_hat @ M -> (N, C), then element-wise multiply with Y_hat and sum
        # This is equivalent to Tr(Y_hat @ M @ Y_hat^T) but more memory efficient
        # Tr(A @ B^T) = sum(A * B)
        YM = Y_hat @ M  # (N, C)
        trace_val = (YM * Y_hat).sum()
        return trace_val / N
    
    def forward(self, spatial_preds, contacting_preds):
        """
        Compute the structural consistency loss.
        
        Args:
            spatial_preds: (N, 6) sigmoid probabilities for spatial predicates
            contacting_preds: (N, 17) sigmoid probabilities for contacting predicates
        
        Returns:
            Scalar loss value weighted by lambda_sc
        """
        loss_spatial = self._compute_trace_loss(spatial_preds, self.M_spatial)
        loss_contacting = self._compute_trace_loss(contacting_preds, self.M_contacting)
        
        return self.lambda_sc * (loss_spatial + loss_contacting)
