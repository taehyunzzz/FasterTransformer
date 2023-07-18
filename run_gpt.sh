#!/bin/bash

cd build

SEQ_LEN=512
INPUT_FILE=../input_${SEQ_LEN}.txt

NUM_GPUS=1
export WORLD_SIZE=$NUM_GPUS

run_gpt2(){
        # MODEL=gpt2_small
        MODEL=gpt2_medium

        mkdir -p ${RESULT_DIR}

        if [ $MODE -eq "moe" ]; then
                CKPT_PATH="../models/hf_${MODEL}/converted_${dtype}/${NUM_GPUS}-gpu/" 
                RESULT_DIR=prof_${MODEL}_seq${SEQ_LEN}
                OUT_FILENAME=${MODEL}_batch${batch_size}_type${dtype}
        else
                CKPT_PATH="../models/nlp_gpt3_text-generation_0.35B_MoE-64/converted_${dtype}/${NUM_GPUS}-gpu/"
                RESULT_DIR=prof_moegpt2m_exp64_seq${SEQ_LEN}_${NUM_GPUS}gpus
                OUT_FILENAME=moegpt2m_batch${batch_size}_type${dtype}
        fi

        echo "MODEL CKPT PATH:"${CKPT_PATH}

        if [ ${NUM_GPUS} > 1 ]; then
                RUN_CMD="mpirun -n ${NUM_GPUS} --allow-run-as-root"
        fi

        RUN_CMD+="
                python3 ../examples/pytorch/gpt/multi_gpu_gpt_example.py \
                --tensor_para_size=${NUM_GPUS} \
                --pipeline_para_size=1 \
                --ckpt_path=${CKPT_PATH} \
                --data_type=${dtype} \
                --vocab_file=../models/jieba_tokenizer.json \
                --vocab_size=51200 \
                --use_jieba_tokenizer
                --max_batch_size=${batch_size} \
                --start_id=7 \
                --end_id=7 \
                --sample_input_file=${INPUT_FILE} \
                "
}

for dtype in fp32; do
for batch_size in 1 16 32; do
for MODE in "" "moe"; do

        RUN_CMD=""
        run_gpt2

        echo "batch_size:       "${batch_size}
        echo "moe:              "${MODE}
        echo "RUN_CMD:          "${RUN_CMD}
        echo "OUT_FILENAME:     "${OUT_FILENAME}
        
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
