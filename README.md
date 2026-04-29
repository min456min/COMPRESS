# COMPRESS
**C**ompression **O**f **M**olecular **P**hysical fields into **R**educed **S**patial **S**ites

COMPRESS is an optimization framework that maps an all-atom (AA) molecule of M atoms to K physically parameterized sites (K < M):

```
S ∈ R^{M×6}  →  V ∈ R^{K×6}
  (x, y, z, q, σ, ε)
```

Each site is defined by three spatial coordinates and three non-bonded interaction parameters - partial charge (q), Lennard-Jones radius (σ), and well depth (ε). The K sites are optimized to reproduce the density, electrostatic, and van der Waals (vdW) fields of the original AA molecule on a 3D face-centered cubic (FCC) grid. This yields a fixed-size, directly physically interpretable molecular representation whose compression level is controlled by K.

---

## Directory Structure

```
COMPRESS/
├── COMPRESS.py          # Main entry point
├── README.md
├── pyproject.toml       # Package metadata and dependencies
├── example/
│   └── test.smi         # Example input
└── script/
    ├── __init__.py
    ├── extract_params.py    # SMI/PDB → ACPYPE → params CSV
    ├── init.py              # AA and COMPRESS(CG) grid initialization
    ├── grid.py              # Grid class (field computation)
    ├── loss.py              # Loss functions
    ├── update_features.py   # L-BFGS optimization loop
    └── write_file.py        # Save results
```

---

## Installation

### Option 1: pip (recommended)
All dependencies including OpenBabel are installed automatically:
```bash
git clone https://github.com/username/COMPRESS.git
cd COMPRESS
pip install -e .
```

### Option 2: pip from GitHub (no clone needed)
```bash
# Latest
pip install git+https://github.com/username/COMPRESS.git

# Specific version
pip install git+https://github.com/username/COMPRESS.git@v0.1.0
```

After installation, `compress` command is available anywhere:
```bash
compress -t smi -n benzene -s 12
```

---

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- NumPy >= 1.26.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- RDKit >= 2024.03.0
- acpype >= 2023.10.27
- openbabel-wheel >= 3.1.1

All of the above are installed automatically via `pip install`.

---

## Usage

```bash
compress -t <type> -n <name> -s <n_sites> [options]
```

### Required Arguments

| Argument | Description |
|---|---|
| `-t`, `--type` | Input file type: `smi` or `pdb` |
| `-n`, `--name` | Molecule name (must match filename, e.g. `benzene` → `benzene.smi`) |
| `-s`, `--site` | Number of COMPRESS sites |

### Optional Arguments

| Argument | Default | Description |
|---|---|---|
| `--steps` | 50 | Number of optimization steps |
| `--grid_interval` | 0.3 | Grid spacing (Å) |
| `--grid_buffer` | 5.0 | Grid buffer around molecule (Å) |
| `--lr_T` | 1.0 | Langevin temperature |
| `--lr_noise_scale` | 1e-7 | Langevin noise scale |
| `--decay_T` | 0.5 | Temperature decay factor |
| `--decay_T_interval` | 3 | Temperature decay interval (steps) |
| `--tau_density` | 0.2 0.5 | Tau values for density field |
| `--tau_charge` | 0.2 1.0 | Tau values for charge field |
| `--tau_epsilon` | 0.2 2.0 | Tau values for VDW epsilon field |

---

## Example

An example input is provided in `example/test.smi`:
```
c1ccccc1  benzene
```

### Case 1: SMILES string directly

No input file needed - SMILES is written to `test.smi` automatically:
```bash
compress -t smi -n test -s 12 --smiles "c1ccccc1"
```

### Case 2: SMILES file

Run from the directory containing `test.smi`:
```bash
cd example
compress -t smi -n test -s 12
```

### Case 3: PDB file

If you already have a PDB file, run from the directory containing `test.pdb`:
```bash
compress -t pdb -n test -s 12
```

In all three cases, the pipeline runs automatically:
1. Generate `test.pdb` from SMILES via RDKit (Cases 1 & 2 only)
2. Run ACPYPE → `test.acpype/`
3. Extract atomic parameters → `test_params.csv`
4. Initialize AA and COMPRESS (CG) grids
5. Optimize COMPRESS sites via L-BFGS
6. Save results → `test_s12_COMPRESS.pt`

If `test_params.csv` already exists (e.g. rerunning with a different site count), steps 1–3 are skipped automatically:
```bash
compress -n test -s 6   # reuses test_params.csv
```

Expected output:
```
>> ----------------------------------------
>> Name      : test
>> Input     : /path/to/example/test.smi
>> Sites     : 12
>> Device    : cuda
>> Output    : /path/to/example/test_s12_COMPRESS.pt
>> ----------------------------------------
>> Input file found: test.smi
>> Generating PDB from SMILES: test.smi
>> Saving PDB file: test.pdb
>> Running Acpype...
>> Acpype finished successfully!
>> Extracting params from: test.acpype
>> Params saved: test_params.csv
>> ----------------------------------------
>> Running COMPRESS...
>> ----------------------------------------
>> Optimizing 50 steps...
>> Step    1 | Grid: 0.8241 | Total: 1.2034
>> Step    2 | Grid: 0.7193 | Total: 1.0871
...
>> Step   50 | Grid: 0.1023 | Total: 0.2341
>> Results saved: test_s12_COMPRESS.pt
```

---

## Output

Results are saved as a PyTorch `.pt` file (`{name}_s{n_sites}_COMPRESS.pt`):

```python
import torch

data = torch.load("test_s12_COMPRESS.pt")

data["AA_pos"]   # All-atom positions      (N_aa, 3)
data["AA_chg"]   # All-atom charges        (N_aa,)
data["AA_sig"]   # All-atom sigma          (N_aa,)
data["AA_eps"]   # All-atom epsilon        (N_aa,)

data["pos"]      # COMPRESS site positions       (N_cg, 3)
data["chg"]      # COMPRESS site charges         (N_cg,)
data["sig"]      # COMPRESS site sigma           (N_cg,)
data["eps"]      # COMPRESS site epsilon         (N_cg,)

data["loss"]     # Final loss dict
```

---

## Pipeline Overview

```
example/test.smi
   │
   ▼ RDKit (if smi)
example/test.pdb
   │
   ▼ ACPYPE
example/test.acpype/
   │
   ▼ extract_params
example/test_params.csv
   │
   ▼ COMPRESS (init → optimize)
example/test_s12_COMPRESS.pt
```
