#!/bin/bash
#SBATCH -p cpu-ms,cpu-troja
#SBATCH --job-name=visualize_lens
#SBATCH --output=logs/%x_%j.out
#SBATCH --time=7-0:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

source .env

export HF_TOKEN=$HF_TOKEN
export HF_HOME=/lnet/work/people/luu/hf_home

# export MAX_JOBS=4
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0
# export VLLM_USE_FLASHINFER_SAMPLER=0


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

# TOKENIZER="utter-project/EuroLLM-9B-2512"
# MODEL="utter-project__EuroLLM-9B-2512"

TOKENIZER="Qwen/Qwen3.5-9B-Base"
MODEL="Qwen__Qwen3.5-9B-Base"

# TOKENIZER="CohereLabs/tiny-aya-base"
# MODEL="CohereLabs__tiny-aya-base"

# TOKENIZER="google/gemma-4-12B"
# MODEL="google__gemma-4-12B"

DATASET="flores"
SRC_LANG="ces_Latn"
TGT_LANG="vie_Latn"
# SRC_LANG="eng_Latn"
# TGT_LANG="eng_Latn"
SAMPLE="0"

mkdir -p heatmaps_bak/${MODEL}/${SRC_LANG}-${TGT_LANG}

# run $ENVDIR/bin/python visualize_lens.py \
#     "lens_output_all/${MODEL}/${DATASET}_${SRC_LANG}_${TGT_LANG}/lens_${DATASET}_${SRC_LANG}_${TGT_LANG}_sample${SAMPLE}.json" \
#     --tokenizer $TOKENIZER \
#     --output heatmaps_bak/${MODEL}/${SRC_LANG}-${TGT_LANG}/lens_top1_heatmap.png

# run $ENVDIR/bin/python visualize_combined.py \
#     "lens_output_all/${MODEL}/${DATASET}_${SRC_LANG}_${TGT_LANG}/lens_${DATASET}_${SRC_LANG}_${TGT_LANG}_sample${SAMPLE}.json" \
#     --lang-dist "bak/token_lang_dist_fineweb_dataset_${MODEL}.json" \
#     --output heatmaps_bak/${MODEL}/${SRC_LANG}-${TGT_LANG}/lang_heatmap

# run $ENVDIR/bin/python visualize_english_prob.py \
#     "lens_output_all/${MODEL}/${DATASET}_${SRC_LANG}_${TGT_LANG}/lens_${DATASET}_${SRC_LANG}_${TGT_LANG}_sample${SAMPLE}.json" \
#     --lang-dist "bak/token_lang_dist_fineweb_dataset_${MODEL}.json" \
#     --output heatmaps_bak/${MODEL}/${SRC_LANG}-${TGT_LANG}/lang_heatmap

# run $ENVDIR/bin/python visualize_lang_prob.py \
#     "lens_output_all/${MODEL}/${DATASET}_${SRC_LANG}_${TGT_LANG}/lens_${DATASET}_${SRC_LANG}_${TGT_LANG}_sample${SAMPLE}.json" \
#     --lang-dist "bak/token_lang_dist_fineweb_dataset_${MODEL}.json" \
#     --output heatmaps_bak/${MODEL}/${SRC_LANG}-${TGT_LANG}/lang_heatmap

run $ENVDIR/bin/python visualize_entropy.py \
    "lens_output_all/${MODEL}/${DATASET}_${SRC_LANG}_${TGT_LANG}/lens_${DATASET}_${SRC_LANG}_${TGT_LANG}_sample${SAMPLE}.json" \
    --lang-dist "bak/token_lang_dist_fineweb_dataset_${MODEL}.json" \
    --output heatmaps_bak/${MODEL}/${SRC_LANG}-${TGT_LANG}/lang_heatmap
