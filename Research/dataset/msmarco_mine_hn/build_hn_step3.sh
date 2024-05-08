WORKDIR=/home/jeeves/mine_marco_2

python Research/Dataset/msmarco/build_hn.py  \
    --tokenizer_name "/home/jeeves/bge-base-en-v1.5"  \
    --hn_file $WORKDIR/results/train.trec  \
    --qrels /home/jeeves/marco/qrels.train.4column.tsv  \
    --queries  /home/jeeves/marco/train.query.txt  \
    --collection /home/jeeves/marco/corpus.tsv  \
    --save_to $WORKDIR/train_hn  \
    --doc_template "<title> <text>" \
    --n_sample 1 \
    --depth_begin 30 \
    --depth_end 50 \

