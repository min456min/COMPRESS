from pathlib import Path
import argparse
import sys
import torch
from script.extract_params import generate_pdb_from_smiles, run_acpype, extract_params
from script.init import get_Grids
from script.update_features import update_CG
from script.write_file import write_result, write_result_all


def main():
    parser = argparse.ArgumentParser()

    # Basic arguments
    parser.add_argument("-t", "--type",  type=str, default="pdb")
    parser.add_argument("-n", "--name",  type=str, required=True)
    # parser.add_argument("-s", "--site",  type=int, default=1)
    parser.add_argument("-s", "--site", type=str, default="1", help="Number of CG sites, or 'all' for K=1 to M")

    # SMILES
    parser.add_argument("-sm", "--smiles", type=str, default=None, help="SMILES string (e.g. 'c1ccccc1')")

    # Hyperparameters
    parser.add_argument("--steps",            type=int,   default=50)
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

    # Directory arguments (optional, default to cwd)
    parser.add_argument("--input_dir",  type=Path, default=None, help="Directory containing input files (default: cwd)")
    parser.add_argument("--acpype_dir", type=Path, default=None, help="Directory containing {name}.acpype (default: cwd)")
    parser.add_argument("--param_dir",  type=Path, default=None, help="Directory to save/load {name}_params.csv (default: cwd)")
    parser.add_argument("--out_dir",    type=Path, default=None, help="Output directory (default: cwd)")
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

    input_dir  = args.input_dir  if args.input_dir  else work_dir
    acpype_dir = args.acpype_dir if args.acpype_dir else work_dir
    param_dir  = args.param_dir  if args.param_dir  else work_dir
    out_dir    = args.out_dir    if args.out_dir    else work_dir

    # Fixed filename format
    input_path = input_dir  / f"{name}.{args.type}"   # name.smi or name.pdb
    pdb_path   = input_dir  / f"{name}.pdb"
    acpype_path = acpype_dir / f"{name}.acpype"
    param_path = param_dir  / f"{name}_params.csv"    

    # Output path
    if args.site == "all":
        out_path = out_dir / f"{name}_all_COMPRESS.pt"
    else:
        out_path = out_dir / f"{name}_s{args.site}_COMPRESS.pt"


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
        print(f">> [SKIP] All preprocessing steps skipped")
    elif acpype_path.exists():
        print(f">> [SKIP] ACPYPE already exists: {acpype_path}")
        print(f">> [SKIP] Input/PDB steps skipped")
        # Extract and save params
        print(f">> Extracting params from: {acpype_path}")
        extract_params(acpype_path, param_path)

    else:
        # Check input file exists
        if not input_path.exists():
            print(f">> Error: Input file not found: {input_path}")
            sys.exit(1)
        else:
            print(f">> Input file found: {input_path}")

        # Convert SMI to PDB if needed
        if args.type == "smi":
            if args.smiles is not None:
                print(f">> Writing SMILES to: {input_path}")
                with open(input_path, 'w') as f:
                    f.write(f"{args.smiles}\n")

            if not pdb_path.exists():
                print(f">> Generating PDB from SMILES: {input_path}")
                generate_pdb_from_smiles(input_path, pdb_path)
            else:
                print(f">> [SKIP] PDB already exists: {pdb_path}")
        else:
            pdb_path = input_path
            print(f">> Using PDB: {pdb_path}")

        # Run ACPYPE
        print(f">> Running ACPYPE on: {pdb_path}")
        run_acpype(pdb_path, acpype_dir)

        # Extract and save params
        print(f">> Extracting params from: {acpype_dir}")
        extract_params(acpype_dir, param_path)

    # Run COMPRESS
    print(f">> ----------------------------------------")
    print(f">> Loading params: {param_path}")
    print(f">> Running COMPRESS...")
    print(f">> ----------------------------------------")
    AA, _ = get_Grids(param_path, config, 1, device, dtype)  
    M = AA.pos.shape[0]  # Number of AA atoms

    if args.site == "all":
        print(f">> Running COMPRESS for K=1 to K={M}...")

        results = []
        for k in range(1, M + 1):
            print(f">> ----------------------------------------")
            print(f">> K = {k} / {M}")
            print(f">> ----------------------------------------")
            _, CG = get_Grids(param_path, config, k, device, dtype)
            CG = update_CG(AA, CG, config)
            results.append((k, CG))

        write_result_all(out_path, AA, results)

    else:
        n_sites  = int(args.site)
        print(f">> Running COMPRESS for K={n_sites}...")
        _, CG = get_Grids(param_path, config, n_sites, device, dtype)
        CG = update_CG(AA, CG, config)
        write_result(out_path, AA, CG)

if __name__ == "__main__":
    main()
