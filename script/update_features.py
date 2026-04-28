import torch
import torch.optim as optim
from .loss import calc_loss


# ── State Helpers ──────────────────────────────────────────────────────────────

def clone_state(CG):
    """Save current COMPRESS parameters as a detached snapshot."""
    return {
        "pos":         CG.pos.detach().clone(),
        "chg":         CG.chg.detach().clone(),
        "log_sig":     CG.log_sig.detach().clone(),
        "log_eps":     CG.log_eps.detach().clone(),
    }

def load_state(CG, state):
    """Restore COMPRESS parameters from a snapshot."""
    with torch.no_grad():
        CG.pos.copy_(state["pos"])
        CG.chg.copy_(state["chg"])
        CG.log_sig.copy_(state["log_sig"])
        CG.log_eps.copy_(state["log_eps"])
        CG.softplus_attr()


# ── Optimizer ──────────────────────────────────────────────────────────────────

def get_optimizer(CG, config):
    """
    L-BFGS optimizer for fast convergence.
    Preferred over Adam for moderate-sized systems (~300 params)
    as it uses curvature information for better step directions.
    closure() is required by L-BFGS for line search.
    """
    lr = config['learning_rates'].get('lr_lbfgs', 1.0)
    return optim.LBFGS(
        [CG.pos, CG.chg, CG.log_sig, CG.log_eps],
        lr=lr,
        max_iter=10,
        history_size=10,
        line_search_fn="strong_wolfe",
    )


# ── Constraints & Noise ────────────────────────────────────────────────────────

def apply_langevin_noise(CG, config):
    """
    Add Langevin noise for thermal sampling after L-BFGS step.
    std_dev = sqrt(2 * noise_scale * T)
    """
    T           = config['learning_rates']['T']
    noise_scale = config['learning_rates']['noise_scale']

    if T <= 0.0:
        return CG

    with torch.no_grad():
        std_dev = (2.0 * noise_scale * T) ** 0.5
        CG.pos.add_(torch.randn_like(CG.pos)     * std_dev)
        CG.chg.add_(torch.randn_like(CG.chg)     * std_dev)
        CG.log_sig.add_(torch.randn_like(CG.log_sig) * std_dev)
        CG.log_eps.add_(torch.randn_like(CG.log_eps) * std_dev)
        CG.softplus_attr()

    return CG


def decay_temperature(t, config):
    """Decay temperature T every decay_T_interval steps."""
    T        = config['learning_rates']['T']
    decay    = config['decline_learning_rates']['decay_T']
    interval = config['decline_learning_rates']['decay_T_interval']

    if t % interval == 0:
        config['learning_rates']['T'] = max(decay * T, 0.0)

    return config


# ── Single Step ────────────────────────────────────────────────────────────────

def run_step(t, optimizer, AA, CG, config):
    """
    Perform one optimization step with L-BFGS.
    closure() is evaluated multiple times per step during line search.
    """
    def closure():
        optimizer.zero_grad()
        CG.softplus_attr()
        total_loss, _ = calc_loss(t, AA, CG)
        total_loss.backward()
        return total_loss

    optimizer.step(closure)
    apply_langevin_noise(CG, config)
    config = decay_temperature(t, config)

    return CG, optimizer


# ── Main Loop ──────────────────────────────────────────────────────────────────

def update_CG(AA, CG, config):
    """Run full optimization loop and return optimized COMPRESs rperesentation."""
    steps     = config.get("steps", 50)
    optimizer = get_optimizer(CG, config)

    print(f">> Optimizing {steps} steps...")
    for t in range(1, steps + 1):
        CG, optimizer = run_step(t, optimizer, AA, CG, config)
        ld = CG.loss_dict
        print(f">> Step {t:4d} | Grid: {ld['avg_grid_loss']:.4f} | Total: {ld['total_loss']:.4f}")

    return CG

