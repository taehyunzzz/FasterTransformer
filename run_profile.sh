#!/bin/bash

cd build

SEQ_LEN=512
INPUT_FILE=../input_${SEQ_LEN}.txt

for config in 0; do
        for dtype in fp16; do
                for batch_size in 1 ; do
# for config in 0 1; do
#         for dtype in fp32 fp16; do
#                 for batch_size in 1 8; do

                        if [[ $config == 0 ]]; then
                                MODEL=gpt2s
                                OUTDIR=prof_mod${MODEL}_seq${SEQ_LEN}
                                OUT_REP=${OUTDIR}/${MODEL}_batch${batch_size}_type${dtype}

                                CKPT_PATH="./models/hf_gpt2_small/converted_${dtype}/1-gpu/" 

                        elif [[ $config == 1 ]]; then
                                MODEL=gpt2m
                                OUTDIR=prof_mod${MODEL}_seq${SEQ_LEN}
                                OUT_REP=${OUTDIR}/${MODEL}_batch${batch_size}_type${dtype}

                                CKPT_PATH="./models/hf_gpt2_medium/converted_${dtype}/1-gpu/" 

                        elif [[ $config == 2 ]]; then
                                OUTDIR=nsys_out_moegpt2m_exp64_seq${SEQ_LEN}
                                OUT_REP=${OUTDIR}/nsys_out_moegpt2m_batch${batch_size}_type${dtype}

                                CKPT_PATH="./models/nlp_gpt3_text-generation_0.35B_MoE-64/converted_${dtype}/1-gpu/"
                        else
                                echo "WRONG CONFIG"
                                break
                        fi

                        mkdir -p ${OUTDIR}


                        PROF_OP1="-t cuda,nvtx,osrt,cublas,cudnn --capture-range=cudaProfilerApi"
                        PROF_OP2="--stats=true --force-overwrite=true --output=${OUT_REP}"
                        # --capture-range=cudaProfilerApi

                        # CMD="python3 ../examples/pytorch/gpt/multi_gpu_gpt_example.py \
                        #         --tensor_para_size=1 \
                        #         --pipeline_para_size=1 \
                        #         --ckpt_path=${CKPT_PATH} \
                        #         --data_type=${dtype} \
                        #         --vocab_file=../models/jieba_tokenizer.json \
                        #         --vocab_size=51200 \
                        #         --use_jieba_tokenizer
                        #         --max_batch_size=${batch_size} \
                        #         --start_id=7 \
                        #         --end_id=7 \
                        #         --sample_input_file=${INPUT_FILE}
                        #         "

                        CMD="
                                python ../examples/pytorch/t5/translate_example.py \
                                --batch_size ${batch_size} \
                                --beam_width 1 \
                                --max_seq_len ${SEQ_LEN} \
                                --data_type ${dtype} \
                                --test_time 3 \
                                --sampling_topk 1 \
                                --model t5-small
                        "

                                # --vocab_file=../models/gpt2config/gpt2-vocab.json \
                                # --merges_file=../models/gpt2config/gpt2-merges.txt \
                                # --vocab_size=50257 \

                                # --vocab_file=../../models/nlp_gpt3_text-generation_0.35B_MoE-64/tokenizer.json \
                                # --vocab_size=51200 \
                                # --use_jieba_tokenizer

                                # --CKPT_PATH=../../models/nlp_gpt3_text-generation_0.35B_MoE-64/converted/1-gpu/ \
                                # --vocab_file=../../models/nlp_gpt3_text-generation_0.35B_MoE-64/tokenizer.json \
                                # --vocab_file=../../models/gpt2config/tokenizer.json \

                        PROF="nsys profile ${PROF_OP1} ${PROF_OP2}"

                        echo $PROF $CMD
                        $PROF $CMD

                        sleep 3s

                done
        done
done

cd ../
