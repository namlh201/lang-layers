#!/bin/bash
#SBATCH -p gpu-ms,gpu-troja,gpu-amd
#SBATCH --job-name=lens_all_eurollm-9b
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=30-0:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="gpuram95G"
#SBATCH --mem=100G

export HF_HOME=/lnet/work/people/luu/hf_home

# export MAX_JOBS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# export VLLM_USE_FLASHINFER_SAMPLER=0

if [ ! -z $(command -v nvidia-smi) ]; then
    module load cuda/13.0
    ENVDIR=/lnet/work/people/luu/miniconda3/envs/lensing
elif [ ! -z $(command -v rocm-smi) ]; then
    GPU_MAPS=$(seq 0 1 $(($SLURM_GPUS - 1)))
    export HIP_VISIBLE_DEVICES=`echo $GPU_MAPS | sed 's/ /,/g'` # this should set $HIP_VISIBLE_DEVICES to 0,1,2,3 regardless of $ROCR_VISIBLE_DEVICES
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    # export VLLM_ATTENTION_BACKEND=TRITON
    ENVDIR=/lnet/work/people/luu/miniconda3/envs/openeurollm_amd
fi

# export CUDA_HOME=$ENVDIR/lib/python3.11/site-packages/nvidia/cu13
# export CUDA_HOME=/opt/cuda/13.1
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LIBRARY_PATH

run $ENVDIR/bin/python lens_all.py \
    --model="utter-project/EuroLLM-9B-2512" \
    --input-dir=translations \
    --batch-size=1 \
    --backend=nnsight \
    --no-generation-only \
    --resume
    # --model="utter-project/EuroLLM-9B-2512" \
    # --model="google/gemma-4-12B" \
    # --no-vllm
