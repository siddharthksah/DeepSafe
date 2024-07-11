#!/bin/bash

# Set environment variables (adjust as needed)
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_CLIENT_SHOWERRORDETAILS=false

# Activate Conda environment
eval "$(conda shell.bash hook)"
conda activate deepsafe

# Run Streamlit
streamlit run main.py \
    --server.port $STREAMLIT_SERVER_PORT \
    --server.address $STREAMLIT_SERVER_ADDRESS \
    --client.showErrorDetails=$STREAMLIT_CLIENT_SHOWERRORDETAILS \
    --server.enableCORS=false \
    --server.enableXsrfProtection=true \
    --server.maxUploadSize=200 \
    --browser.serverAddress=$STREAMLIT_SERVER_ADDRESS \
    --server.maxMessageSize=200

# Deactivate Conda environment
conda deactivate