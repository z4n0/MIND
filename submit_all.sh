#!/usr/bin/env bash
# submit_all.sh  –  sbatch every .slurm file under slurm_files/

set -euo pipefail

SCRIPTS_DIR="slurm_files"

echo "→ Submitting all .slurm scripts in $SCRIPTS_DIR ..."
for script in "${SCRIPTS_DIR}"/*.slurm; do
  if [[ -f "$script" ]]; then
    echo "↪ sbatch $script"
    sbatch "$script"
  fi
done

echo "→ Done."

#remember to make this script executable:
#chmod +x submit_all.sh
#./submit_all.sh