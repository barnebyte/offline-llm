#!/bin/bash

# Define directories for GPT-J and evaluation model (BERT)
GPTJ_MODEL_DIR="$(pwd)/gpt-j-6B"
BERT_MODEL_DIR="$(pwd)/bert-base-uncased-SST-2"

# GPT-J model URLs
GPTJ_MODEL_URL="https://huggingface.co/EleutherAI/gpt-j-6B/resolve/main/pytorch_model.bin"
GPTJ_CONFIG_URL="https://huggingface.co/EleutherAI/gpt-j-6B/resolve/main/config.json"
GPTJ_TOKENIZER_URL="https://huggingface.co/EleutherAI/gpt-j-6B/resolve/main/tokenizer.json"

# BERT model URLs for textattack/bert-base-uncased-SST-2
BERT_MODEL_URL="https://huggingface.co/textattack/bert-base-uncased-SST-2/resolve/main/pytorch_model.bin"
BERT_CONFIG_URL="https://huggingface.co/textattack/bert-base-uncased-SST-2/resolve/main/config.json"
BERT_TOKENIZER_URL="https://huggingface.co/textattack/bert-base-uncased-SST-2/resolve/main/vocab.txt"
BERT_TOKENIZER_CONFIG_URL="https://huggingface.co/textattack/bert-base-uncased-SST-2/resolve/main/tokenizer_config.json"

# Create the GPT-J model directory if it doesn't exist
if [ ! -d "$GPTJ_MODEL_DIR" ]; then
    echo "Creating GPT-J model directory: $GPTJ_MODEL_DIR"
    mkdir -p "$GPTJ_MODEL_DIR"
else
    echo "GPT-J model directory already exists: $GPTJ_MODEL_DIR"
fi

# Download the GPT-J model weights if they don't already exist
if [ ! -f "$GPTJ_MODEL_DIR/pytorch_model.bin" ]; then
    echo "Downloading GPT-J model weights..."
    wget -O "$GPTJ_MODEL_DIR/pytorch_model.bin" "$GPTJ_MODEL_URL" --progress=bar:force
else
    echo "GPT-J model weights already exist: $GPTJ_MODEL_DIR/pytorch_model.bin"
fi

# Download the GPT-J config file if it doesn't already exist
if [ ! -f "$GPTJ_MODEL_DIR/config.json" ]; then
    echo "Downloading GPT-J model config..."
    wget -O "$GPTJ_MODEL_DIR/config.json" "$GPTJ_CONFIG_URL" --progress=bar:force
else
    echo "GPT-J config file already exists: $GPTJ_MODEL_DIR/config.json"
fi

# Download the GPT-J tokenizer file if it doesn't already exist
if [ ! -f "$GPTJ_MODEL_DIR/tokenizer.json" ]; then
    echo "Downloading GPT-J tokenizer..."
    wget -O "$GPTJ_MODEL_DIR/tokenizer.json" "$GPTJ_TOKENIZER_URL" --progress=bar:force
else
    echo "GPT-J tokenizer file already exists: $GPTJ_MODEL_DIR/tokenizer.json"
fi

# Create the BERT model directory if it doesn't exist
if [ ! -d "$BERT_MODEL_DIR" ]; then
    echo "Creating BERT model directory: $BERT_MODEL_DIR"
    mkdir -p "$BERT_MODEL_DIR"
else
    echo "BERT model directory already exists: $BERT_MODEL_DIR"
fi

# Download the BERT model weights if they don't already exist
if [ ! -f "$BERT_MODEL_DIR/pytorch_model.bin" ]; then
    echo "Downloading BERT model weights..."
    wget -O "$BERT_MODEL_DIR/pytorch_model.bin" "$BERT_MODEL_URL" --progress=bar:force
else
    echo "BERT model weights already exist: $BERT_MODEL_DIR/pytorch_model.bin"
fi

# Download the BERT config file if it doesn't already exist
if [ ! -f "$BERT_MODEL_DIR/config.json" ]; then
    echo "Downloading BERT model config..."
    wget -O "$BERT_MODEL_DIR/config.json" "$BERT_CONFIG_URL" --progress=bar:force
else
    echo "BERT config file already exists: $BERT_MODEL_DIR/config.json"
fi

# Download the BERT tokenizer vocab file if it doesn't already exist
if [ ! -f "$BERT_MODEL_DIR/vocab.txt" ]; then
    echo "Downloading BERT tokenizer vocab..."
    wget -O "$BERT_MODEL_DIR/vocab.txt" "$BERT_TOKENIZER_URL" --progress=bar:force
else
    echo "BERT tokenizer vocab file already exists: $BERT_MODEL_DIR/vocab.txt"
fi

# Download the BERT tokenizer config file if it doesn't already exist
if [ ! -f "$BERT_MODEL_DIR/tokenizer_config.json" ]; then
    echo "Downloading BERT tokenizer config..."
    wget -O "$BERT_MODEL_DIR/tokenizer_config.json" "$BERT_TOKENIZER_CONFIG_URL" --progress=bar:force
else
    echo "BERT tokenizer config file already exists: $BERT_MODEL_DIR/tokenizer_config.json"
fi

echo "Model files for GPT-J and BERT evaluation model are ready."

# Build the Docker image
echo "Building Docker image..."
docker build -t local-llm-api-fp16 .

echo "Docker image built successfully. You can now run the container."
