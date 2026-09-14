#!/bin/bash
#SBATCH -p cpu-ms,cpu-troja
#SBATCH --job-name=build_token_lang_dist
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=7-0:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G

source .env

export HF_TOKEN=$HF_TOKEN
export HF_HOME=/lnet/work/people/luu/hf_home

if [ ! -z $(command -v rocm-smi) ]; then
    GPU_MAPS=$(seq 0 1 $(($SLURM_GPUS - 1)))
    export HIP_VISIBLE_DEVICES=`echo $GPU_MAPS | sed 's/ /,/g'` # this should set $HIP_VISIBLE_DEVICES to 0,1,2,3 regardless of $ROCR_VISIBLE_DEVICES
    export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    # export VLLM_ATTENTION_BACKEND=TRITON
    ENVDIR=/lnet/work/people/luu/miniconda3/envs/openeurollm_amd
# if [ ! -z $(command -v nvidia-smi) ]; then
else
    # module load cuda/13.0
    ENVDIR=/lnet/work/people/luu/miniconda3/envs/lensing
fi

run $ENVDIR/bin/python build_token_lang_dist_fineweb_dataset.py \
    --model google/gemma-4-12B utter-project/EuroLLM-9B-2512 Qwen/Qwen3.5-9B-Base CohereLabs/tiny-aya-base
