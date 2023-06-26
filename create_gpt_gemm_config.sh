# ./bin/gpt_gemm <batch_size> <beam_width> <max_input_len> <head_number> <size_per_head> <inter_size> <vocab_size> <data_type> <tensor_para_size> <is_append>
# E.g., ./bin/gpt_gemm 8 1 32 12 128 6144 51200 1 1 1


beam_width=1
max_input_len=512
head_number=16
size_per_head=128
inter_size=4096
vocab_size=51200
data_type=fp32
tensor_para_size=1
is_append=1


d_model=1024
d_ff=$((d_model * 4))
size_per_head=$((d_model / head_number))

#for max_input_len in 128 512; do 
    #for batch_size in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do

cd build
for max_input_len in 512; do 
    for batch_size in 1; do
        ./bin/gpt_gemm $batch_size $beam_width $max_input_len $head_number $size_per_head $inter_size $vocab_size $data_type $tensor_para_size $is_append
    done 
done 
cd ..
