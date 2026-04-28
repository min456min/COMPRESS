from pathlib import Path
import argparse
import sys
import torch
from script.extract_params import generate_pdb_from_smiles, run_acpype, extract_params
from script.init import get_Grids
from script.update_features import update_CG
from script.write_file import write_result


def main():
    parser = argparse.ArgumentParser()

    # Basic arguments
    parser.add_argument("-t", "--type",  type=str, default="pdb")
    parser.add_argument("-n", "--name",  type=str, required=True)
    parser.add_argument("-s", "--site",  type=int, default=1)

    # SMILES
    parser.add_argument("-sm", "--smiles", type=str, default=None, help="SMILES string (e.g. 'c1ccccc1')")

    # Hyperparameters
    parser.add_argument("--steps",            type=int,   default=100)
    parser.add_argument("--grid_interval",    type=float, default=0.3)
    parser.add_argument("--grid_buffer",      type=float, default=5.0)

    # Learning rates
    parser.add_argument("--lr_T",             type=float, default=1.0)
    parser.add_argument("--lr_noise_scale",   type=float, default=1e-7)

    # Decay
    parser.add_argument("--decay_T",          type=float, default=0.5)
    parser.add_argument("--decay_T_interval", type=int,   default=3)

    # Taus
    parser.add_argument("--tau_density",      type=float, nargs=2, default=[0.2, 0.5])
    parser.add_argument("--tau_charge",       type=float, nargs=2, default=[0.2, 1.0])
    parser.add_argument("--tau_epsilon",      type=float, nargs=2, default=[0.2, 2.0])

    args = parser.parse_args()

    # Reconstruct config dict
    config = {
        "steps":         args.steps,
        "grid_interval": args.grid_interval,
        "grid_buffer":   args.grid_buffer,
        "learning_rates": {
            "T":           args.lr_T,
            "noise_scale": args.lr_noise_scale,
        },
        "decline_learning_rates": {
            "decay_T":          args.decay_T,
            "decay_T_interval": args.decay_T_interval,
        },
        "taus": {
            "density": args.tau_density,
            "charge":  args.tau_charge,
            "epsilon": args.tau_epsilon,
        }
    }

    work_dir   = Path.cwd()
    name       = args.name
    input_path = work_dir / f"{name}.{args.type}"
    pdb_path   = work_dir / f"{name}.pdb"
    acpype_dir = work_dir / f"{name}.acpype"
    param_path = work_dir / f"{name}_params.csv"
    out_path   = work_dir / f"{name}_s{args.site}_COMPRESS.pt"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype  = torch.float32

    print(f">> ----------------------------------------")
    print(f">> Name      : {name}")
    print(f">> Input     : {input_path}")
    print(f">> Sites     : {args.site}")
    print(f">> Device    : {device}")
    print(f">> Output    : {out_path}")
    print(f">> ----------------------------------------")

    # Skip SMI/PDB/ACPYPE if params already exist
    if param_path.exists():
        print(f">> [SKIP] Params already exists: {param_path}")
        print(f">> [SKIP] SMI/PDB and ACPYPE steps skipped")

    else:
        # Check input file exists
        if not input_path.exists():
            print(f">> Error: Input file not found: {input_path}")
            sys.exit(1)
        else:
            print(f">> Input file found: {input_path}")

        # Convert SMI to PDB if needed
        if args.type == "smi":
            # If SMILES string provided directly, write to .smi file first
            if args.smiles is not None:
                print(f">> Writing SMILES to: {input_path}")
                with open(input_path, 'w') as f:
                    f.write(f"{args.smiles}\n")
    
            if not input_path.exists():
                print(f">> Error: Input file not found: {input_path}")
                sys.exit(1)
    
            if not pdb_path.exists():
                print(f">> Generating PDB from SMILES: {input_path}")
                generate_pdb_from_smiles(input_path, pdb_path)
            else:
                print(f">> [SKIP] PDB already exists: {pdb_path}")
        else:
            pdb_path = input_path
            print(f">> Using PDB: {pdb_path}")

        # Run ACPYPE
        if not acpype_dir.exists():
            print(f">> Running ACPYPE on: {pdb_path}")
            run_acpype(pdb_path)
        else:
            print(f">> [SKIP] ACPYPE already exists: {acpype_dir}")

        # Extract and save params
        print(f">> Extracting params from: {acpype_dir}")
        extract_params(acpype_dir, param_path)

    # Run COMPRESS
    print(f">> ----------------------------------------")
    print(f">> Loading params: {param_path}")
    print(f">> Running COMPRESS...")
    print(f">> ----------------------------------------")
    AA, CG = get_Grids(param_path, config, args.site, device, dtype)
    CG = update_CG(AA, CG, config)
    write_result(out_path, AA, CG)


if __name__ == "__main__":
    main()
