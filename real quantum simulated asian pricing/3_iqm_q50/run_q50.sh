#!/bin/bash
#SBATCH --job-name=asian-pricing
#SBATCH --account=project_465003017
#SBATCH --reservation=JQH2026
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH --output=asian_pricing/asian-pricing-%j.out

# Runs the arithmetic-Asian pricing pipeline (path-dependent, running-weight oracle):
#   - Step 0: print + save the Asian characteristic-function circuit diagram
#   - Step 1: validate G(lambda); price via COS vs exact & MC; MLAE vs sampling
#   - Step 2: shallow pieces (X/Y char circuits) on the real Q50, then compare
# Output: asian_char_circuit.png + asian_query_scaling.png +
#         asian_price_comparison.png + .out log

module load Local-quantum
module load fiqci-vtt-qiskit-JQH

# matplotlib + numpy must be installed on the login node first:
#   pip install --user matplotlib numpy
# Compute nodes have no internet access.

cd asian_pricing

# First: just PRINT the circuit (both measurement bases) for inspection.
echo "######## CIRCUIT DIAGRAMS ########"
python show_circuit.py --basis X --mpl
python show_circuit.py --basis Y

# Then: the full pipeline (validation + COS price + MC + MLAE + Q50 + plots).
echo "######## EXPERIMENT ########"
python asian_pricing.py
