rm -f slurm.err slurm.log
sbatch slurm-rva.sh
echo "wait for 3 sec..."; sleep 3
squeue -u $USER