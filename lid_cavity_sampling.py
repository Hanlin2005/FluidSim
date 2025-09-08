#THIS CODE IS FOR IMPORTANCE SAMPLING TECHNIQUES AND EXPERIMENTS FOR TRAINING PINNs
import torch
import numpy as np



def sample_points_by_error(pointwise_rel, X, Y, n_samples,
                                uniform_frac=0.2, temperature=1.0, device=None,
                                per_field_reduce="l2"):
    """
    Safe importance sampler over a regular grid. Allows replacement automatically
    if n_samples > number of grid points. Returns (n_samples, 2) tensor on device (optional).
    """
    ny, nx, d = pointwise_rel.shape
    assert d == 3, "pointwise_rel must be (ny, nx, 3)"

    if per_field_reduce == "l2":
        err_map = np.linalg.norm(pointwise_rel, axis=-1)  # (ny, nx)
    elif per_field_reduce == "mean":
        err_map = pointwise_rel.mean(axis=-1)
    else:
        raise ValueError("per_field_reduce must be 'l2' or 'mean'")

    N = nx * ny
    n_imp = int(round(n_samples * (1.0 - uniform_frac)))
    n_uni = n_samples - n_imp

    # Build probabilities
    err = err_map.reshape(-1)
    err = np.maximum(err, 1e-12)
    if temperature != 1.0:
        err = err ** (1.0 / temperature)
    p_imp = err / err.sum()

    # Decide replacement based on requested counts
    imp_replace = n_imp > 0 and n_imp > N
    uni_replace = n_uni > 0 and n_uni > N

    # Draw indices
    imp_idx = np.random.choice(N, size=n_imp, replace=imp_replace, p=p_imp) if n_imp > 0 else np.array([], dtype=int)
    uni_idx = np.random.choice(N, size=n_uni, replace=uni_replace) if n_uni > 0 else np.array([], dtype=int)
    idx = np.concatenate([imp_idx, uni_idx])

    # Map to coords
    Xf = X.reshape(-1)
    Yf = Y.reshape(-1)
    xs = Xf[idx]
    ys = Yf[idx]

    pts = torch.from_numpy(np.stack([xs, ys], axis=1).astype("float32"))
    if device is not None:
        pts = pts.to(device)
    return pts



def sample_boundary_by_error(pointwise_rel, X, Y, n_per_side=200,
                             boundary_frac=0.1, temperature=1.0, device=None):
    """
    Samples boundary points preferentially where the combined relative error is high.
    Returns a (N,2) tensor with points on top/bottom/left/right stripes.
    """
    comb = np.linalg.norm(pointwise_rel, axis=-1)  # (ny, nx)
    x = X[0, :]
    y = Y[:, 0]

    epsx = (x.max() - x.min()) * boundary_frac
    epsy = (y.max() - y.min()) * boundary_frac

    top    = Y >= (y.max() - epsy)
    bottom = Y <= (y.min() + epsy)
    left   = X <= (x.min() + epsx)
    right  = X >= (x.max() - epsx)

    pts_list = []
    for mask in [top, bottom, left, right]:
        err = comb[mask].ravel()
        err = np.maximum(err, 1e-12)
        if temperature != 1.0:
            err = err ** (1.0 / temperature)
        p = err / err.sum()

        Xs = X[mask].ravel()
        Ys = Y[mask].ravel()
        # choose indices within this side
        idx = np.random.choice(len(Xs), size=n_per_side, replace=True, p=p)
        pts_list.append(np.stack([Xs[idx], Ys[idx]], axis=1))

    pts = np.concatenate(pts_list, axis=0).astype("float32")
    pts = torch.from_numpy(pts)
    if device is not None:
        pts = pts.to(device)
    return pts