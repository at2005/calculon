#!/usr/bin/env bash
set -eux

# 1) Login to Hugging Face using the token from environment
huggingface-cli login --token "$HF_TOKEN" || {
  echo "Hugging Face login failed. Make sure HF_TOKEN is set."
  exit 1
}

# 2) Download the dataset
huggingface-cli download \
  --resume-download \
  --repo-type dataset \
  GAIR/MathPile \
  --local-dir /data/MathPile \
  --local-dir-use-symlinks False


cd /data/MathPile/train
find . -type f -name '*.gz' | parallel gzip -d

cp /app/math_bpe-merges.txt ./
cp /app/math_bpe-vocab.json ./

python /app/data.py

cd /app

exec "$@"
