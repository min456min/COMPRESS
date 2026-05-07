import os
import torch

def write_result(out_path, AA, CG):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "AA": {
            "pos": AA.pos.detach().cpu(),
            "chg": AA.chg.detach().cpu(),
            "sig": AA.sig.detach().cpu(),
            "eps": AA.eps.detach().cpu(),
        },
        "CG": {
            "pos":  CG.pos.detach().cpu(),
            "chg":  CG.chg.detach().cpu(),
            "sig":  CG.sig.detach().cpu(),
            "eps":  CG.eps.detach().cpu(),
            "loss": CG.loss_dict,
        }
    }
    torch.save(save_dict, out_path)
    print(f">> Results saved: {out_path}")


def write_result_all(out_path, AA, results):
    """Save AA and all K results to a single .pt file.
    results: list of (k, CG) tuples
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "AA": {
            "pos": AA.pos.detach().cpu(),
            "chg": AA.chg.detach().cpu(),
            "sig": AA.sig.detach().cpu(),
            "eps": AA.eps.detach().cpu(),
        }
    }

    for k, CG in results:
        save_dict[f"K{k}"] = {
            "pos":  CG.pos.detach().cpu(),
            "chg":  CG.chg.detach().cpu(),
            "sig":  CG.sig.detach().cpu(),
            "eps":  CG.eps.detach().cpu(),
            "loss": CG.loss_dict,
        }

    torch.save(save_dict, out_path)
    print(f">> Results saved: {out_path}")
