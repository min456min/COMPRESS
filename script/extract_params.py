import sys
import shutil
import subprocess
import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def generate_pdb_from_smiles(smi_path, pdb_path):
    """Read SMILES from file and generate 3D PDB using RDKit."""
    with open(smi_path, 'r') as f:
        smiles = f.readline().strip().split()[0]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error: Invalid SMILES string: {smiles}")
        sys.exit(1)

    mol = Chem.AddHs(mol)
    res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if res == -1:
        print("Error: Failed to generate 3D coordinates.")
        sys.exit(1)

    print(f">> Saving PDB file: {pdb_path}")
    Chem.MolToPDBFile(mol, str(pdb_path))


def run_acpype(pdb_path, charge_method='bcc', net_charge=0):
    """Execute acpype in cwd. Results will be in {name}.acpype/"""
    print(">> Running Acpype...")

    cmd = ["acpype", "-i", str(pdb_path), "-c", charge_method, "-n", str(net_charge)]

    try:
        subprocess.run(cmd, check=True, text=True)
        print(">> Acpype finished successfully!")

    except subprocess.CalledProcessError as e:
        print(f"Error: Acpype failed.\n{e}")
        sys.exit(1)

    except FileNotFoundError:
        print("Error: 'acpype' not found. Please ensure it is installed.")
        sys.exit(1)

def extract_params(acpype_dir, param_path):
    """Extract sigma, epsilon, charge, position from acpype output and save as CSV."""
    name = acpype_dir.stem.replace(".acpype", "")

    # Parse ITP file (atom type → sigma, epsilon)
    itp_path  = acpype_dir / f"{name}_GMX.itp"
    atom_dict = {}

    if not itp_path.exists():
        print(f"Error: ITP file not found at {itp_path}")
        sys.exit(1)

    for line in open(itp_path, 'r'):
        parts = line.split()
        if len(parts) < 7: continue
        if line.strip().startswith((';', '#')): continue
        try:
            atom_dict[parts[0]] = {
                'Sigma':   float(parts[5]) * 10,  # nm to Angstrom
                'Epsilon': float(parts[6]) # KJ/mol
            }
        except ValueError:
            continue

    # Parse Mol2 file (position, charge)
    mol2_path = acpype_dir / f"{name}_bcc_gaff2.mol2"
    if not mol2_path.exists():
        mol2_path = acpype_dir / f"{name}_user_gaff2.mol2"
    if not mol2_path.exists():
        print(f"Error: Mol2 file not found in {acpype_dir}")
        sys.exit(1)

    data_rows   = []
    read_switch = False

    with open(mol2_path, 'r') as f:
        for line in f:
            if "<TRIPOS>ATOM" in line:
                read_switch = True;  continue
            if "<TRIPOS>BOND" in line or ("<TRIPOS>" in line and read_switch):
                read_switch = False; continue
            if read_switch:
                parts = line.split()
                if len(parts) < 9: continue
                # Mol2 format: id name x y z type res_id res_name charge
                atom_type = parts[5]
                params    = atom_dict.get(atom_type)
                if params is None:
                    print(f"Warning: atom type '{atom_type}' not found in ITP. Skipping.")
                    continue
                data_rows.append({
                    'Idx':     int(parts[0]),
                    'Atom':    parts[1],
                    'Type':    atom_type,
                    'X':       float(parts[2]),
                    'Y':       float(parts[3]),
                    'Z':       float(parts[4]),
                    'Charge':  float(parts[-1]),
                    'Sigma':   params['Sigma'],
                    'Epsilon': params['Epsilon'],
                })

    pd.DataFrame(data_rows).to_csv(param_path, index=False, sep='\t')
    print(f">> Params saved: {param_path}")
