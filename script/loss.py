import torch
import torch.nn.functional as F


# ── Individual Loss Functions ──────────────────────────────────────────────────

def calc_norm_loss(query, target):
    """Normalized L2 loss between two grids. Range: [0, 1]."""
    return torch.sum((query - target) ** 2 + 1e-12) / (torch.sum(query ** 2 + target ** 2) + 1e-12)


def calc_sig_violation_loss(sig, sig_max=4.0):
    """Penalize sigma values exceeding the upper bound."""
    return (torch.relu(sig - sig_max) ** 2).mean()


def calc_repulsion_loss(G, alpha=2, w=0.4):
    """Penalize COMPRESS sites that overlap each other."""
    pos = G.pos
    sig = G.sig
    N   = pos.size(0)

    idx      = torch.triu_indices(N, N, offset=1, device=pos.device)
    dist     = ((pos[idx[0]] - pos[idx[1]]).pow(2).sum(-1) + 1e-12).sqrt()
    min_dist = (sig[idx[0]] + sig[idx[1]]) / 2.0 * w

    return (torch.relu(min_dist - dist) ** alpha).mean()


def calc_net_charge_loss(AA, CG):
    """Penalize difference in net charge between AA and COMPRESS."""
    return (AA.chg.sum() - CG.chg.sum()) ** 2


def boundary_violation_loss(G, w=0.5, alpha=3.0):
    """Penalize COMPRESS sites that extend outside the AA molecular boundary."""
    pos      = G.pos.view(-1, 3)
    sig      = G.sig.view(-1)
    boundary = G.boundary.view(-1, 3)

    d_min   = torch.cdist(pos, boundary).min(dim=1).values
    sig_cut = sig * w

    return (torch.relu(sig_cut - d_min) ** alpha).mean()


# ── Scheduler Weights ──────────────────────────────────────────────────────────

def get_step_weights(t):
    """
    Return loss weights at step t.
    Overlap and boundary losses are gradually introduced after warmup.
    """
    w_overlap = 0.0 if t < 15 else min(0.005 * (t - 15), 1.0)

    if t < 15:
        w_boundary = 0.0
    elif t < 20:
        w_boundary = 0.2
    else:
        w_boundary = min(1.0, 0.2 * (t - 20))

    return {
        'sig_violation': 1.0,
        'net_charge':    0.1,
        'overlap':       w_overlap,
        'boundary':      w_boundary,
    }


# ── Main Loss Function ─────────────────────────────────────────────────────────
def calc_loss(t, AA, CG):
    """
    Compute total loss at step t.
    Stores loss_dict on COMPRESS for logging in update_features.
    Returns:
        total_loss : scalar tensor for backprop
        loss_dict  : dict of individual loss values for logging
    """
    loss_dict = {}

    # Regularization losses
    loss_sv       = calc_sig_violation_loss(CG.sig)
    loss_net_chg  = calc_net_charge_loss(AA, CG)
    loss_boundary = boundary_violation_loss(CG)
    loss_overlap  = calc_repulsion_loss(CG) if CG.pos.shape[0] >= 2 else torch.tensor(0.0, device=AA.device)

    # Step-dependent weights
    w = get_step_weights(t)

    # Weighted sum of regularization losses
    total_loss = (w['sig_violation'] * loss_sv
                + w['net_charge']    * loss_net_chg
                + w['overlap']       * loss_overlap
                + w['boundary']      * loss_boundary)

    # Grid field losses (density, charge, epsilon, ...)
    G           = CG.get_grid(store=False)
    grid_losses = []

    for field_name in AA.G:
        gtype, tau, g_aa = AA.G[field_name]
        g_cp = G[field_name][2]

        field_w  = min((t + 1) * tau / 2, 1.0)
        field_w *= 1.5 if gtype == 'epsilon' else 1.0

        loss = calc_norm_loss(g_aa, g_cp) * field_w
        total_loss += loss
        grid_losses.append(loss)
        loss_dict[f'loss_{field_name}'] = loss.item()

    avg_grid_loss = torch.stack(grid_losses).mean() if grid_losses else torch.tensor(0.0)

    # Log all losses
    loss_dict['total_loss']    = total_loss.item()
    loss_dict['avg_grid_loss'] = avg_grid_loss.item()
    loss_dict['loss_reg_sig']  = loss_sv.item()
    loss_dict['loss_overlap']  = loss_overlap.item()
    loss_dict['loss_net_chg']  = loss_net_chg.item()
    loss_dict['loss_boundary'] = loss_boundary.item()

    # Store on CG for access in update_features
    CG.loss_dict = loss_dict

    return total_loss, loss_dict
