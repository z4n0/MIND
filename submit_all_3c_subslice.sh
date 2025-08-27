#!/usr/bin/env bash
# submit_all.sh  –  sbatch every .slurm file under slurm_files/ with delay

set -euo pipefail

SCRIPTS_DIR="slurm_files/subslice/3c"
DELAY_MINUTES=1

# Check if directory exists
if [[ ! -d "$SCRIPTS_DIR" ]]; then
  echo "Error: Directory $SCRIPTS_DIR does not exist"
  exit 1
fi

echo "→ Submitting all .slurm scripts in $SCRIPTS_DIR with ${DELAY_MINUTES} minute delay..."
first_job=true
job_count=0

for script in "${SCRIPTS_DIR}"/*.slurm; do
  if [[ -f "$script" ]]; then
    # Skip delay for the first job
    if [[ "$first_job" = true ]]; then
      first_job=false
    else
      echo "↪ Waiting ${DELAY_MINUTES} minutes before submitting next job..."
      sleep $((DELAY_MINUTES * 60))
    fi
    
    echo "↪ sbatch $script"
    if sbatch "$script"; then
      ((job_count++))
    else
      echo "Error: Failed to submit $script"
    fi
  fi
done

if [[ $job_count -eq 0 ]]; then
  echo "Warning: No .slurm files found in $SCRIPTS_DIR"
else
  echo "→ Done. Submitted $job_count jobs."
fi

#remember to make this script executable:
#chmod +x submit_all_3c_subslice.sh
#./submit_all_3c_subslice.sh