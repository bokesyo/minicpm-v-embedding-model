


# python Research/Dataset/reverse_query/main.py \
#     --data_path "/home/jeeves/longdoc-pdf-chunks-limited-1024-2024-03-11-194646" \
#     --output_path "/home/jeeves/gathered_output_zh" \
#     --model_path "/home/jeeves/Yi-34B-Chat" \
#     --n_gpu 1 \
#     --batch_size 32 \
#     --max_new_tokens 512 \
#     --prompt_path "prompt_cn.txt" \



# python Research/Dataset/reverse_query/main.py \
#     --data_path "/home/jeeves/book3.1000k/chunks_test2" \
#     --output_path "/home/jeeves/gathered_output_zh" \
#     --model_path "/home/jeeves/Yi-34B-Chat" \
#     --n_gpu 1 \
#     --batch_size 32 \
#     --max_new_tokens 512 \
#     --prompt_path "prompt_en.txt" \



python Research/Dataset/reverse_query/main.py \
    --data_path "/home/jeeves/book3.1000k/chunks_test2" \
    --output_path "/home/jeeves/gathered_output_en_multi" \
    --model_path "/home/jeeves/Yi-34B-Chat" \
    --n_gpu 1 \
    --batch_size 32 \
    --max_new_tokens 512 \
    --prompt_path "prompt_multi_en.txt" \
