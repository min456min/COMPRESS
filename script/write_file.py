import os
import torch

def write_result(out_path, AA, CG):
    """Save final AA, COMPRESS parameters and loss to .pt file."""

    save_dict = {
        # AA (all-atom) ground truth
        "AA_pos": AA.pos.detach().cpu(),
        "AA_chg": AA.chg.detach().cpu(),
        "AA_sig": AA.sig.detach().cpu(),
        "AA_eps": AA.eps.detach().cpu(),

        # CG (coarse-grained) result
        "pos": CG.pos.detach().cpu(),
        "chg": CG.chg.detach().cpu(),
        "sig": CG.sig.detach().cpu(),
        "eps": CG.eps.detach().cpu(),

        # Final loss
        "loss":   CG.loss_dict,
    }

    torch.save(save_dict, out_path)
    print(f">> Results saved: {out_path}")
