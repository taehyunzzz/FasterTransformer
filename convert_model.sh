#https://github.com/NVIDIA/FasterTransformer/blob/main/docs/gpt_guide.md


# model_dir=../models/nlp_gpt3_text-generation_0.35B_MoE-64
# export PYTHONPATH='$PWD':$PYTHONPATH

# python examples/pytorch/gpt/utils/megatron_gpt_moe_ckpt_convert.py \
# --input-dir ${model_dir}/model \
# --saved-dir ${model_dir}/converted_fp32 \
# --infer-gpu-num 1 \
# --weight-data-type fp32 \
# --vocab-path ../models/gpt2config/gpt2-vocab.json \
# --merges-path ../models/gpt2config/gpt2-merges.txt

##################################################

# model_dir=./models/hf_gpt2_small
# mkdir -p $model_dir
# export PYTHONPATH='$PWD':$PYTHONPATH

# python examples/pytorch/gpt/utils/huggingface_gpt_convert.py \
# -in_file=gpt2 \
# -saved_dir=${model_dir}/converted_fp32 \
# -infer_gpu_num 1 \
# -weight_data_type fp32 \

# python examples/pytorch/gpt/utils/huggingface_gpt_convert.py \
# -in_file=gpt2 \
# -saved_dir=${model_dir}/converted_fp16 \
# -infer_gpu_num 1 \
# -weight_data_type fp16 \

##################################################

# model_dir=./models/hf_gpt2_medium
# mkdir -p $model_dir
# export PYTHONPATH='$PWD':$PYTHONPATH

# python examples/pytorch/gpt/utils/huggingface_gpt_convert.py \
# -in_file=gpt2-medium \
# -saved_dir=${model_dir}/converted_fp32 \
# -infer_gpu_num 1 \
# -weight_data_type fp32 \

# python examples/pytorch/gpt/utils/huggingface_gpt_convert.py \
# -in_file=gpt2-medium \
# -saved_dir=${model_dir}/converted_fp16 \
# -infer_gpu_num 1 \
# -weight_data_type fp16 \

##################################################

model_dir="./models/hf_t5_base"
filename="t5-base"
mkdir -p $model_dir
export PYTHONPATH='$PWD':$PYTHONPATH

python examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
-in_file=${filename} \
-saved_dir=${model_dir}/converted_fp32 \
-inference_tensor_para_size 1 \
-weight_data_type fp32 \
-processes 16

python examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
-in_file=${filename} \
-saved_dir=${model_dir}/converted_fp16 \
-inference_tensor_para_size 1 \
-weight_data_type fp16 \
-processes 16

# ##################################################

# model_dir="./models/hf_t5_large"
# filename="t5-large"
# mkdir -p $model_dir
# export PYTHONPATH='$PWD':$PYTHONPATH

# python examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
# -in_file=${filename} \
# -saved_dir=${model_dir}/converted_fp32 \
# -inference_tensor_para_size 1 \
# -weight_data_type fp32 \
# -processes 16

# python examples/pytorch/t5/utils/huggingface_t5_ckpt_convert.py \
# -in_file=${filename} \
# -saved_dir=${model_dir}/converted_fp16 \
# -inference_tensor_para_size 1 \
# -weight_data_type fp16 \
# -processes 16