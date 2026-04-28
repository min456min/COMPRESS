import torch
import math
from sklearn.cluster import KMeans
import pandas as pd
from .grid import Grid


def sample_aa_boundary_dots(AA, n_rays=2048, sigma_scale=0.5, margin=2.0, max_dots=5000):
    """
    Sample points on the surface of the AA molecule using ray-sphere intersection.
    Used to define the boundary constraint for CG site positions.

    Args:
        AA          : AA Grid object
        n_rays      : Number of rays cast per atom (Fibonacci sphere sampling)
        sigma_scale : Surface offset as a fraction of sigma
        margin      : Extra margin beyond the molecule for ray origin
        max_dots    : Max number of surface points (farthest point sampling)
    Returns:
        dots : (max_dots, 3) surface point coordinates
    """
    device = AA.device
    dtype  = AA.dtype

    pos = AA.pos.reshape(-1, 3).to(device=device, dtype=dtype)
    sig = AA.sig.reshape(-1).to(device=device, dtype=dtype)

    dots_list = []
    for pos0 in pos:
        # Effective atom radius for surface detection
        r = (sigma_scale * sig).clamp_min(torch.tensor(1e-8, device=device, dtype=dtype))

        # Fibonacci sphere: uniformly distributed ray directions
        i     = torch.arange(n_rays, device=device, dtype=dtype)
        z     = 1.0 - 2.0 * (i + 0.5) / float(n_rays)
        angle = math.pi * (3.0 - math.sqrt(5.0))
        theta = i * angle
        xy    = torch.sqrt(torch.clamp(1.0 - z * z, min=0.0))
        dirs  = torch.stack([xy * torch.cos(theta), xy * torch.sin(theta), z], dim=1)
        dirs  = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-12)

        # Ray origins: placed outside the molecule, shooting inward
        rel   = pos - pos0[None, :]
        outer = rel.norm(dim=1).max() + r.max() + torch.tensor(float(margin), device=device, dtype=dtype)
        p0    = pos0[None, :] + outer * dirs
        d     = -dirs  # inward direction

        # Vectorized ray-sphere intersection for all rays × all atoms
        # Ray: p(t) = p0 + t*d
        # Sphere j: ||p(t) - c_j||^2 = r_j^2
        oc   = p0[:, None, :] - pos[None, :, :]
        b    = (oc * d[:, None, :]).sum(dim=2)
        c    = (oc * oc).sum(dim=2) - (r[None, :] ** 2)
        disc = b * b - c  # discriminant; >= 0 means intersection

        valid     = disc >= 0.0
        sqrt_disc = torch.zeros_like(disc)
        sqrt_disc[valid] = torch.sqrt(torch.clamp(disc[valid], min=0.0))

        # Two intersection solutions per ray-sphere pair
        t1 = -b - sqrt_disc
        t2 = -b + sqrt_disc

        # Keep only positive t (forward intersections)
        INF     = torch.tensor(1e30, device=device, dtype=dtype)
        t1_pos  = torch.where((t1 > 1e-10) & valid, t1, INF)
        t2_pos  = torch.where((t2 > 1e-10) & valid, t2, INF)
        t_first = torch.minimum(t1_pos, t2_pos)

        # For each ray, pick the nearest atom hit
        t_min, _ = torch.min(t_first, dim=1)
        hit_mask  = t_min < (INF * 0.5)

        # Compute hit coordinates
        dots = p0[hit_mask] + t_min[hit_mask][:, None] * d[hit_mask]
        dots_list.append(dots)

    dots = torch.cat(dots_list, dim=0)

    # Farthest point sampling: select max_dots well-spread surface points
    N        = dots.shape[0]
    idx      = torch.zeros(max_dots, dtype=torch.long, device=device)
    dist     = torch.full((N,), float('inf'), device=device)
    farthest = torch.randint(0, N, (1,), device=device)

    for i in range(max_dots):
        centroid = dots[farthest].view(1, 3)
        d        = torch.sum((dots - centroid) ** 2, dim=1)
        dist     = torch.minimum(dist, d)
        farthest = torch.argmax(dist)
        idx[i]   = farthest

    return dots[idx]


def df_to_tensor(sub_df, device, dtype):
    """Convert param DataFrame columns to tensors."""
    pos = torch.tensor(sub_df[['X', 'Y', 'Z']].values, device=device, dtype=dtype)
    chg = torch.tensor(sub_df['Charge'].values,         device=device, dtype=dtype)
    sig = torch.tensor(sub_df['Sigma'].values,          device=device, dtype=dtype)
    eps = torch.tensor(sub_df['Epsilon'].values,        device=device, dtype=dtype)
    return pos, chg, sig, eps


def set_initial_AA(param_path, device, dtype):
    """Load atomic parameters from CSV and return as tensors."""
    df = pd.read_csv(param_path, sep='\t')
    return df_to_tensor(df, device, dtype)


def get_Grids(param_path, config, N_atom, device, dtype):
    """
    Initialize AA and CG Grid objects.

    AA is built from all-atom parameters and its fields are computed once (no_grad).
    CG is initialized from the N_atom most important AA atoms (by field contribution),
    with log-space sigma/epsilon for unconstrained optimization.
    Boundary dots are sampled from the AA surface for the boundary loss.

    Args:
        param_path : Path to {name}_params.csv
        config     : Hyperparameter dict
        N_atom     : Number of CG sites
        device     : torch device
        dtype      : torch dtype
    Returns:
        AA, CG : initialized Grid objects
    """
    pos_aa, chg_aa, sig_aa, eps_aa = set_initial_AA(param_path, device, dtype)

    aa_center = pos_aa.mean(dim=0)

    # Build AA grid and compute reference molecular fields
    AA = Grid(
            pos           = pos_aa,
            chg           = chg_aa,
            eps           = eps_aa,
            sig           = sig_aa,
            center        = aa_center,
            grid_interval = config['grid_interval'],
            grid_buffer   = config['grid_buffer'],
            taus          = config['taus'],
            )

    with torch.no_grad():
        AA.get_grid()  # Compute and store AA field grids (target)

    # Select initial CG positions from the most field-contributing AA atoms
    remove_order = AA.get_atomic_ranks()  # Sorted least → most important
    keep_n       = int(N_atom)
    keep_idx     = remove_order[-keep_n:].to(device=device, dtype=torch.long)
    keep_idx, _  = torch.sort(keep_idx)

    # Initialize CG parameters (log-space for sigma/epsilon → softplus constraint)
    pos     = AA.pos[keep_idx].clone().detach().requires_grad_(True)
    chg     = AA.chg[keep_idx].clone().detach().requires_grad_(True)
    log_sig = torch.log(torch.expm1(AA.sig[keep_idx] - 0.001)).detach().clone().requires_grad_(True)
    log_eps = torch.log(torch.expm1(AA.eps[keep_idx] - 0.001)).detach().clone().requires_grad_(True)

    # Sample AA surface boundary points for boundary violation loss
    dots = sample_aa_boundary_dots(AA)

    # Build CG grid (shares grid geometry with AA)
    CG = Grid(
         pos           = pos,
         chg           = chg,
         log_sig       = log_sig,
         log_eps       = log_eps,
         center        = aa_center,
         grid_size     = AA.grid_size,
         grid_interval = config['grid_interval'],
         grid_buffer   = config['grid_buffer'],
         grid_coords   = AA.grid_coords,
         taus          = config['taus'],
         )

    CG.softplus_attr()   # Initialize eps/sig from log_eps/log_sig
    CG.boundary = dots   # Attach boundary for loss computation

    return AA, CG
