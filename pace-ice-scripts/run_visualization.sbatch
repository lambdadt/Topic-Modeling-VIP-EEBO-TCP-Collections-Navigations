#!/bin/bash
#SBATCH -J visualization_job       # Job name
#SBATCH --account=coc             # Account (adjust if needed)
#SBATCH -N 1 --ntasks-per-node=1  # 1 node, 1 task per node
#SBATCH --cpus-per-task=12        # Number of CPUs per task
#SBATCH --mem=10G                # Memory per node
#SBATCH -t 1:00:00                # Time limit (1 hour, adjust as needed)
#SBATCH -q coe-ice                # Partition
#SBATCH -o Report-%j.out          # Output file
#SBATCH --mail-type=BEGIN,END,FAIL # Email notifications
#SBATCH --mail-user=tpark309@gatech.edu

# Load Anaconda module
module load anaconda3/2022.05.0.1

# Activate the vip-ted environment
source activate vip-ted

# Set the number of topics and iterations
N_TOPICS=30 
N_ITER=100   
ROOT_FOLDER=..
ARTIFACT_FOLDER=${ROOT_FOLDER}/matrices_and_visuals/vocab100k #  ${ROOT_FOLDER}/out/vocab100k, ${ROOT_FOLDER}/matrices_and_visuals/vocab100k
MATRIX_TYPE=count # count, tfidf, tfidf_reweighted_count_vectors

# Run visualization.py with specified arguments, replacing \r with \n for clean output
python ${ROOT_FOLDER}/visualization.py \
    --matrix_dir ${ARTIFACT_FOLDER} \
    --data_dir ${ARTIFACT_FOLDER} \
    --dz_filename "PLSI_P_dz_${N_TOPICS}topics_${N_ITER}iter_${MATRIX_TYPE}.csv" \
    --zw_filename "PLSI_P_zw_${N_TOPICS}topics_${N_ITER}iter_${MATRIX_TYPE}.csv" \
    --output_dir ${ARTIFACT_FOLDER} \
    2>&1 | sed 's/\r/\n/g' > Report-$SLURM_JOB_ID.out
