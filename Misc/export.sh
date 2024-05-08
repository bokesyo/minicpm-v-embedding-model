cd /data/checkpoints/2024-02-04-185120-cpm_d_2b_embedding-lr5e-6-temp0.02-bsz2-8-1/checkpoint-5000

mkdir /data/models/cpm_d_2b_embedding

cp *.safetensors /data/models/cpm_d_2b_embedding
cp *.json /data/models/cpm_d_2b_embedding
cp *.py /data/models/cpm_d_2b_embedding
cp tokenizer.model /data/models/cpm_d_2b_embedding

echo "ok"