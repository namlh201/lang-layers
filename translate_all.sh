#!/bin/bash
#SBATCH -p gpu-ms,gpu-troja,gpu-amd
#SBATCH --job-name=translate_all
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=7-0:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=4
#SBATCH --constraint="gpuram95G|gpuram40G|gpuram48G"
#SBATCH --mem=64G

export HF_HOME=/lnet/work/people/luu/hf_home

export MAX_JOBS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# export VLLM_ENABLE_V1_MULTIPROCESSING=0
# export VLLM_USE_FLASHINFER_SAMPLER=0

module load cuda/13.0
ENVDIR=/lnet/work/people/luu/miniconda3/envs/lensing

# export CUDA_HOME=$ENVDIR/lib/python3.11/site-packages/nvidia/cu13
# export CUDA_HOME=/opt/cuda/13.1
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# export LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LIBRARY_PATH

# run $ENVDIR/bin/python translate_all.py \
#     --dataset=both \
#     --model="google/gemma-4-12B" \
#     --max-tokens=256 \
#     --n-shot=0
    # --model="utter-project/EuroLLM-9B-2512" \
    # --model="google/gemma-4-12B" \
    # --model="CohereLabs/tiny-aya-base" \
    # --model="Qwen/Qwen3.5-9B-Base" \
    # --no-vllm

run $ENVDIR/bin/python translate_all.py \
    --dataset=both \
    --same-lang \
    --cross-lingual \
    --resume \
    --model="Qwen/Qwen3.5-9B-Base" \
    --max-tokens=256 \
    --n-shot=0

# --cross-lingual \
