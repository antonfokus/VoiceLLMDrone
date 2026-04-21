#!/bin/bash

echo "Installing packages..."
pip install pymavlink huggingface_hub mavsdk llama-cpp-python
echo "Packages installed."

echo "Downloading model ..."
hf download anton105/DroneLlama --local-dir models --include DroneLlama.gguf
echo "Download complete."
