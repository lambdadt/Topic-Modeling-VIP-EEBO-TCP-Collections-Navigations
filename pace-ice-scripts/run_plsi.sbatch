#!/bin/bash
#SBATCH -J plsi_job               # Job name
#SBATCH --account=coc             # Account
#SBATCH -N 1                      # 1 node
#SBATCH --ntasks-per-node=1       # 1 task per node
#SBATCH --cpus-per-task=24        # Use all 24 cores
#SBATCH --mem=200G                # Request 700 GB (leaves buffer)
#SBATCH -t 4:00:00                # Time limit (4 hours, adjust after testing)
#SBATCH -q coe-ice                # Partition
#SBATCH -o Report-%j.out          # Output file
#SBATCH --mail-type=BEGIN,END,FAIL # Email notifications
#SBATCH --mail-user=tpark309@gatech.edu

# Load Anaconda module
module load anaconda3/2022.05.0.1

# Activate the vip-ted environment
source activate vip-ted

# Maximize CPU usage for numpy and related libraries
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24
export NUMEXPR_NUM_THREADS=24
export OPENBLAS_NUM_THREADS=24

# Set parameters
N_TOPICS=40
MAX_ITER=100
ROOT_FOLDER=..
ARTIFACT_FOLDER=${ROOT_FOLDER}/matrices_and_visuals/vocab100k #  ${ROOT_FOLDER}/out/vocab100k, ${ROOT_FOLDER}/matrices_and_visuals/vocab100k
MATRIX_TYPE=count # count, tfidf, tfidf_reweighted_count_vectors

# Create wandb project name
WANDB_PROJECT="VIP-Topic-Modelling-pLSI-T${N_TOPICS}-I${MAX_ITER}"

# Optional: Verify CPU usage
echo "Running on $(nproc) cores"
echo "Wandb project: ${WANDB_PROJECT}"

# Run plsi.py with wandb logging
time python ${ROOT_FOLDER}/plsi.py \
    --topics ${N_TOPICS} \
    --input_dir ${ARTIFACT_FOLDER} \
    --output_dir ${ARTIFACT_FOLDER} \
    --max_iter ${MAX_ITER} \
    --tol 1e-5 \
    --pct_docs 100 \
    --matrix_type ${MATRIX_TYPE} \
    --use_wandb \
    --wandb_project "${WANDB_PROJECT}" \
    2>&1 | sed 's/\r/\n/g' > Report-$SLURM_JOB_ID.out
