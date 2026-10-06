import torch
import numpy as np
import structural_metrics

def test_metrics():
    # 1. Build a dummy exclusion matrix (26x26)
    # We will simulate 3 attention, 6 spatial, 17 contacting classes
    # Spatial: above (0), beneath (1), in_front_of (2), behind (3), etc.
    # Let's say above and beneath are mutually exclusive.
    # Wait, we need real names from the dataset.
    
    # We will mock the dataloader names
    attention_classes = ['lookingat', 'notlookingat', 'unsure']
    spatial_classes = ['above', 'beneath', 'infrontof', 'behind', 'on', 'nextto']
    contacting_classes = [
        'sittingon', 'standingon', 'wiping', 'drinkingfrom', 
        'holding', 'notcontacting', 'touching', 'carrying',
        'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'
    ]
    
    all_predicates = attention_classes + spatial_classes + contacting_classes
    M_full = structural_metrics.build_full_exclusion_matrix(all_predicates)
    M_spatial = structural_metrics.build_full_exclusion_matrix(spatial_classes)
    M_contacting = structural_metrics.build_full_exclusion_matrix(contacting_classes)
    
    # Verify M_full has correctly mapped the excluded pairs
    assert M_spatial[0, 1] == True # above <-> beneath
    assert M_contacting[0, 1] == True # sittingon <-> standingon
    
    # 2. Build a dummy prediction Y
    num_pairs = 1
    # spatial_preds: (1, 6)
    spatial_preds = np.zeros((1, 6), dtype=np.float32)
    # Give 'above' and 'beneath' both high scores (a planted contradiction!)
    spatial_preds[0, 0] = 0.9 # above
    spatial_preds[0, 1] = 0.8 # beneath
    spatial_preds[0, 2] = 0.1
    
    # contacting_preds: (1, 17)
    contacting_preds = np.zeros((1, 17), dtype=np.float32)
    # Give 'sittingon' and 'standingon' both high scores (another contradiction!)
    contacting_preds[0, 0] = 0.85 # sittingon
    contacting_preds[0, 1] = 0.95 # standingon
    
    # 3. Test CR (Contradiction Rate)
    cr_c, cr_t = structural_metrics.compute_cr(spatial_preds, contacting_preds, M_spatial, M_contacting)
    # We planted 1 contradiction in spatial and 1 in contacting. Total = 2.
    assert cr_c == 2, f"Expected 2 contradictions, got {cr_c}"
    assert cr_t == 2, f"Expected 2 total pair-groups, got {cr_t}"
    print("CR Test Passed! CR = 100%")
    
    # 4. Test CAM (Co-activation Mass)
    cam_sum, cam_total = structural_metrics.compute_cam(spatial_preds, contacting_preds, M_spatial, M_contacting)
    # Expected CAM for spatial = M[0,1] * 0.9 * 0.8 = 0.72
    # Expected CAM for contacting = M[0,1] * 0.85 * 0.95 = 0.8075
    # Total CAM = 1.5275
    expected_cam = (0.9 * 0.8) + (0.85 * 0.95)
    assert np.isclose(cam_sum, expected_cam), f"Expected CAM {expected_cam}, got {cam_sum}"
    print(f"CAM Test Passed! CAM = {cam_sum:.4f}")
    
    # 5. Test SCS (Structural Consistency Score)
    rel_scores = np.zeros((1, 26), dtype=np.float32)
    rel_scores[0, 3:9] = spatial_preds
    rel_scores[0, 9:26] = contacting_preds
    
    scs_c, scs_t = structural_metrics.compute_scs(rel_scores, M_full, K=20)
    # The frame is dirty (has contradictions), so scs_c should be 0.
    assert scs_c == 0, f"Expected SCS clean count 0, got {scs_c}"
    print("SCS Test Passed! Clean frames = 0")
    
    print("ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    test_metrics()
