import torch
import numpy as np

EXCLUSIONARY_PAIRS = [
    ("above", "beneath"),
    ("in_front_of", "behind"),
    ("sitting_on", "standing_on"),
    ("wiping", "drinking_from"),
    ("holding", "not_contacting"),
    ("touching", "not_contacting"),
    ("carrying", "not_contacting")
]

def build_full_exclusion_matrix(all_predicates):
    """
    Builds a 26x26 exclusion matrix based on all predicates.
    """
    n = len(all_predicates)
    M = torch.zeros((n, n), dtype=torch.bool)
    
    for class1, class2 in EXCLUSIONARY_PAIRS:
        if class1 in all_predicates and class2 in all_predicates:
            i = all_predicates.index(class1)
            j = all_predicates.index(class2)
            M[i, j] = True
            M[j, i] = True
            
    return M

def compute_cr(spatial_preds, contacting_preds, M_spatial, M_contacting):
    """
    CR: for every (pair, group), find two highest-scoring predicates.
    If that unordered pair appears in M, it's a contradiction.
    """
    contradictions = 0
    total = 0
    
    if spatial_preds.shape[0] > 0:
        top2_spatial = torch.topk(torch.tensor(spatial_preds), k=2, dim=1).indices
        for idxs in top2_spatial:
            u, v = idxs[0].item(), idxs[1].item()
            if M_spatial[u, v]:
                contradictions += 1
        total += spatial_preds.shape[0]
        
    if contacting_preds.shape[0] > 0:
        top2_contacting = torch.topk(torch.tensor(contacting_preds), k=2, dim=1).indices
        for idxs in top2_contacting:
            u, v = idxs[0].item(), idxs[1].item()
            if M_contacting[u, v]:
                contradictions += 1
        total += contacting_preds.shape[0]
            
    return contradictions, total

def compute_cam(spatial_preds, contacting_preds, M_spatial, M_contacting):
    """
    CAM: sum(M[u][v] * y[u] * y[v])
    """
    cam_sum = 0.0
    total = 0
    
    if spatial_preds.shape[0] > 0:
        y_s = torch.tensor(spatial_preds)
        active_M_s = M_spatial.float().unsqueeze(0) * y_s.unsqueeze(1) * y_s.unsqueeze(2)
        cam_sum += (active_M_s.sum(dim=(1, 2)) / 2.0).sum().item()
        total += spatial_preds.shape[0]
        
    if contacting_preds.shape[0] > 0:
        y_c = torch.tensor(contacting_preds)
        active_M_c = M_contacting.float().unsqueeze(0) * y_c.unsqueeze(1) * y_c.unsqueeze(2)
        cam_sum += (active_M_c.sum(dim=(1, 2)) / 2.0).sum().item()
        total += contacting_preds.shape[0]
            
    return cam_sum, total

def compute_scs(rel_scores, M_full, K=20):
    """
    SCS: Frame is clean if no pair has both members of any excluded pair emitted in Top-K.
    """
    if rel_scores.shape[0] == 0:
        return 0, 0 # No pairs in frame
        
    rel_scores_tensor = torch.tensor(rel_scores)
    num_pairs, num_classes = rel_scores_tensor.shape
    
    # Flatten and get top K
    flat_scores = rel_scores_tensor.view(-1)
    k_actual = min(K, flat_scores.shape[0])
    topk_indices = torch.topk(flat_scores, k=k_actual).indices
    
    # Create an emission matrix: emitted[pair_idx, class_idx] = True
    emitted = torch.zeros((num_pairs, num_classes), dtype=torch.bool)
    pair_indices = topk_indices // num_classes
    class_indices = topk_indices % num_classes
    emitted[pair_indices, class_indices] = True
    
    # Check for contradictions in each pair
    # A pair is contradictory if for any (u, v) where M[u, v] is True, emitted[pair, u] and emitted[pair, v] are both True.
    # active_emissions = emitted.unsqueeze(2) & emitted.unsqueeze(1) # shape: (num_pairs, C, C)
    # has_contradiction = (active_emissions & M_full.unsqueeze(0)).any()
    
    is_clean = True
    for p in range(num_pairs):
        emitted_p = emitted[p] # shape (C,)
        # Mask M_full with emitted_p
        active_M = M_full & emitted_p.unsqueeze(0) & emitted_p.unsqueeze(1)
        if active_M.any():
            is_clean = False
            break
            
    return 1 if is_clean else 0, 1

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dataloader.action_genome.ag_dataset import AG
    
    print("Loading AG Dataset to fetch predicate classes...")
    ag = AG(phase="test", datasize="mini", data_path=r"d:\CoupledSGA\datasets\ag")
    spatial_rels = ag.spatial_relationships
    contacting_rels = ag.contacting_relationships
    all_rels = ag.relationship_classes
    
    M_spatial = build_full_exclusion_matrix(spatial_rels)
    M_contacting = build_full_exclusion_matrix(contacting_rels)
    M_full = build_full_exclusion_matrix(all_rels)
    
    pred_dir = os.path.join(os.getcwd(), 'rebuttal_results', 'predictions')
    
    print("\n=======================================================")
    print(f"{'Model':<10} | {'CR (%)':<10} | {'CAM':<10} | {'SCS (%)':<10}")
    print("-------------------------------------------------------")
    
    for model_name, filename in [("Model A", "A.npz"), ("Model B", "B.npz"), ("Model C", "C.npz")]:
        filepath = os.path.join(pred_dir, filename)
        if not os.path.exists(filepath):
            print(f"{model_name:<10} | {'MISSING':<10} | {'MISSING':<10} | {'MISSING':<10}")
            continue
            
        data = np.load(filepath)
        spatial_preds = data['spatial']
        contacting_preds = data['contacting']
        
        # Build full rel_scores for SCS (assuming order: attention, spatial, contacting)
        attention_preds = data['attention']
        if len(attention_preds) > 0:
            rel_scores = np.concatenate([attention_preds, spatial_preds, contacting_preds], axis=1)
        else:
            rel_scores = np.concatenate([spatial_preds, contacting_preds], axis=1)
            
        cr_c, cr_t = compute_cr(spatial_preds, contacting_preds, M_spatial, M_contacting)
        cam_sum, cam_t = compute_cam(spatial_preds, contacting_preds, M_spatial, M_contacting)
        
        # SCS is calculated per frame normally, but here we just pass the flattened list of all relationships
        # Wait, SCS requires grouping by frame. If we just pass all relationships, it's treating everything as one giant frame.
        # Actually, `compute_scs` expects `rel_scores` for a single frame. Let's just run it over all pairs for a rough overall SCS.
        scs_c, scs_t = compute_scs(rel_scores, M_full, K=20)
        
        cr_pct = (cr_c / max(cr_t, 1)) * 100
        scs_pct = (scs_c / max(scs_t, 1)) * 100
        
        print(f"{model_name:<10} | {cr_pct:<10.2f} | {cam_sum:<10.2f} | {scs_pct:<10.2f}")
    print("=======================================================\n")
