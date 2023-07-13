#!/bin/bash

cd build

SEQ_LEN=512
INPUT_FILE=../input_${SEQ_LEN}.txt
export CUDA_VISIBLE_DEVICES=1

run_gpt2(){
        # MODEL=gpt2_small
        MODEL=gpt2_medium

        RESULT_DIR=prof_${MODEL}_seq${SEQ_LEN}
        OUT_FILENAME=${MODEL}_batch${batch_size}_type${dtype}

        mkdir -p ${RESULT_DIR}

        CKPT_PATH="../models/hf_${MODEL}/converted_${dtype}/1-gpu/" 
        echo ${CKPT_PATH}

        RUN_CMD="python3 ../examples/pytorch/gpt/multi_gpu_gpt_example.py \
                --tensor_para_size=1 \
                --pipeline_para_size=1 \
                --ckpt_path=${CKPT_PATH} \
                --data_type=${dtype} \
                --vocab_file=../models/jieba_tokenizer.json \
                --vocab_size=51200 \
                --use_jieba_tokenizer
                --max_batch_size=${batch_size} \
                --start_id=7 \
                --end_id=7 \
                --sample_input_file=${INPUT_FILE}
                "
}

run_moe_gpt2(){

        RESULT_DIR=prof_moegpt2m_exp64_seq${SEQ_LEN}
        OUT_FILENAME=moegpt2m_batch${batch_size}_type${dtype}

        mkdir -p ${RESULT_DIR}

        CKPT_PATH="../models/nlp_gpt3_text-generation_0.35B_MoE-64/converted_${dtype}/1-gpu/"
        echo ${CKPT_PATH}

        RUN_CMD="python3 ../examples/pytorch/gpt/multi_gpu_gpt_example.py \
                --tensor_para_size=1 \
                --pipeline_para_size=1 \
                --ckpt_path=${CKPT_PATH} \
                --data_type=${dtype} \
                --vocab_file=../models/jieba_tokenizer.json \
                --vocab_size=51200 \
                --use_jieba_tokenizer
                --max_batch_size=${batch_size} \
                --start_id=7 \
                --end_id=7 \
                --sample_input_file=${INPUT_FILE}
                "
}

for config in 2; do
for dtype in fp32; do
for batch_size in 1 ; do

        run_gpt2
        # run_moe_gpt2

        echo ${RUN_CMD}
        echo ${OUT_FILENAME}
        
        # NSYS PROFILING
        nsys profile \
            -o ${RESULT_DIR}/${OUT_FILENAME} \
            -t cuda,nvtx,cublas,cudnn,osrt \
            --force-overwrite=true \
            --cudabacktrace=true \
            --capture-range=cudaProfilerApi \
            --capture-range-end=stop \
            ${RUN_CMD} 

        cd ${RESULT_DIR}

        nsys stats --force-overwrite=true \
            --force-export=true \
            --output=. \
            --format=csv \
            --timeunit=msec \
            --report=nvtxkernsum \
            --report=nvtxsum \
            ${OUT_FILENAME}.nsys-rep

        sleep 2s

        cd ..

done
done
done

cd ../
