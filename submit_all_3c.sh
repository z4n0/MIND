#!/usr/bin/env bash
# filepath: /home/zano/Documents/TESI/FOLDER_CINECA/submit_all.sh
#!/usr/bin/env bash
# submit_all.sh  –  sbatch every .slurm file under slurm_files/ with delay

set -euo pipefail

SCRIPTS_DIR="slurm_files/3c"
DELAY_MINUTES=45

echo "→ Submitting all .slurm scripts in $SCRIPTS_DIR with ${DELAY_MINUTES} minute delay..."
first_job=true
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
    sbatch "$script"
  fi
done

echo "→ Done."

#remember to make this script executable:
#chmod +x submit_all.sh
#./submit_all.sh