#!/bin/bash
#SBATCH -J preprocess_job
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

ROOT_FOLDER=..
ARTIFACT_FOLDER=${ROOT_FOLDER}/matrices_and_visuals/vocab100k #  ${ROOT_FOLDER}/out/vocab100k, ${ROOT_FOLDER}/matrices_and_visuals/vocab100k


#   CASE 1: Default case, keep 100k vocab
python ${ROOT_FOLDER}/preprocess.py make_vectors --parsed_texts_dir ${ROOT_FOLDER}/Navigations_headed_xml/Parsed_texts -o ${ARTIFACT_FOLDER} --rare_word_threshold 0.000016


#   CASE 1-1: the --tfidf_reweighted_count_vectors_as_float flag
#     disables rounding and preserve float values in tfidf_reweighted_count_vectors.csv
# python ${ROOT_FOLDER}/preprocess.py make_vectors --parsed_texts_dir ${ROOT_FOLDER}/Navigations_headed_xml/Parsed_texts -o ${ARTIFACT_FOLDER} --rare_word_threshold 0.000016 --tfidf_reweighted_count_vectors_as_float


#   CASE 2: simpler case, keep 5k vocab
# python ${ROOT_FOLDER}/preprocess.py make_vectors --parsed_texts_dir ${ROOT_FOLDER}/Navigations_headed_xml/Parsed_texts -o ${ARTIFACT_FOLDER} --vector_dim_limit 5000