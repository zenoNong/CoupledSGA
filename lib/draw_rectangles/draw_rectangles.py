import numpy as np

def minmax(x):
    return min(max(x, 0.0), 1.0)

def draw_union_boxes(box_pairs, pooling_size, padding=0):
    if isinstance(box_pairs, np.ndarray):
        box_pairs_np = box_pairs.astype(np.float32)
    else:
        box_pairs_np = np.asarray(box_pairs, dtype=np.float32)

    N = box_pairs_np.shape[0]
    uboxes = np.zeros((N, 2, pooling_size, pooling_size), dtype=np.float32)
    if N == 0:
        return uboxes

    for n in range(N):
        x1_union = min(box_pairs_np[n, 0], box_pairs_np[n, 4])
        y1_union = min(box_pairs_np[n, 1], box_pairs_np[n, 5])
        x2_union = max(box_pairs_np[n, 2], box_pairs_np[n, 6])
        y2_union = max(box_pairs_np[n, 3], box_pairs_np[n, 7])

        w = max(x2_union - x1_union, 1e-6)
        h = max(y2_union - y1_union, 1e-6)

        for i in range(2):
            x1_box = (box_pairs_np[n, 0 + 4 * i] - x1_union) * pooling_size / w
            y1_box = (box_pairs_np[n, 1 + 4 * i] - y1_union) * pooling_size / h
            x2_box = (box_pairs_np[n, 2 + 4 * i] - x1_union) * pooling_size / w
            y2_box = (box_pairs_np[n, 3 + 4 * i] - y1_union) * pooling_size / h

            for j in range(pooling_size):
                y_contrib = minmax(j + 1 - y1_box) * minmax(y2_box - j)
                for k in range(pooling_size):
                    x_contrib = minmax(k + 1 - x1_box) * minmax(x2_box - k)
                    uboxes[n, i, j, k] = x_contrib * y_contrib

    return uboxes
