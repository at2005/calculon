# Use an official PyTorch image with CUDA support as base
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# Set the working directory
WORKDIR /app

# (Optional) Copy a requirements file and install additional Python dependencies.
RUN pip install --upgrade pip && \
    pip install tokenizers && \
    pip install --no-cache-dir huggingface_hub && \
    pip install tqdm && \
    pip install orjson

RUN apt-get update && apt-get install -y parallel gzip findutils

# Copy your code (including entrypoint script)
COPY . .

# Make sure the entrypoint script is executable
RUN chmod +x /app/entrypoint.sh

# By default, the entrypoint script will run. We pass torchrun as CMD.
ENTRYPOINT ["/app/entrypoint.sh"]

# Example default command:
#   --nproc_per_node=1 means "1 process per node (GPU)". Adjust as needed.
CMD ["torchrun", "--nproc_per_node=8", "train.py"]
