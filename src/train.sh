#!/bin/bash

# Set the handler for job submission system
HANDLER=slurm

# Define the environment for Python
ENV=~/anaconda3/envs/denoise/bin/python

# Job resource allocation parameters
NODES=1
CPUS_PER_GPU=4
MEM='500G'

# Directory containing training data, where logs and models will be stored as well
OUTDIR="/clusterfs/nvme/sayan/AI/training_denoise/"  # Adjust as needed
mkdir -p $OUTDIR

MODEL_ROOT="${OUTDIR}/models"
MODEL_NAME="drunet"  # Adjust as needed

CONFIG_FILE_PATH="src/config.json"  # Path to config file

# Check for jobs in the pending queue. If there are more than 300 pending, wait and check again
while [ $(squeue -u $USER -h -t pending -r | wc -l) -gt 300 ]
do
    sleep 10s  # Wait for 10 seconds before checking again
done

# Define the Python training script command with parameters
j="${ENV} src/train.py"
j="${j} --train_root ${OUTDIR}"
j="${j} --model_root ${MODEL_ROOT}"
j="${j} --model_name ${MODEL_NAME}"
j="${j} --config_file_path ${CONFIG_FILE_PATH}"

# Define the job name
JOB="train-${MODEL_NAME}"

# Define the path to store log files for the job
LOGS="${OUTDIR}/model_logs"
mkdir -p $LOGS

# Set the job submission command for Slurm
task="/usr/bin/sbatch"
task="${task} --qos=abc_high --nice=1111111111"
task="${task} --gres=gpu:4"  # Specify number of GPUs to be used
task="${task} --partition=abc_a100"
#task="${task} --partition=dgx"
#task="${task} --partition=abc"
#task="${task} --constraint='titan'"
task="${task} --nodes=${NODES}"
task="${task} --cpus-per-gpu=${CPUS_PER_GPU}"
task="${task} --mem='${MEM}'"
task="${task} --job-name=${JOB}"
task="${task} --output=${LOGS}/${JOB}.log"
task="${task} --export=ALL"
task="${task} --wrap=\"${j}\""

# Submit the job to the scheduler
echo $task | bash
echo "ABC : R[$(squeue -u $USER -h -t running -r -p abc | wc -l)], P[$(squeue -u $USER -h -t pending -r -p abc | wc -l)]"
