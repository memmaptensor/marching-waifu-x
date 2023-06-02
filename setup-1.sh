# Setup conda env
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda create -n marching-waifu-x python=3.10
conda activate marching-waifu-x
python -m pip install --upgrade pip

# Install 3rd party packages
pip install gdown ipykernel ipywidgets

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

: '
# Build instant-ngp
# ... (TODO)
    # NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh

    # Setup Optix 7.7
    sudo chmod +x NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh
    sh NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh
    echo "OptiX_INSTALL_DIR=/home/vcg0/NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64" >> ~/.bashrc
    source ~/.bashrc
    cd ~
    rm -rf NVIDIA-OptiX-SDK-7.7.0-linux64-x86_64.sh

    # Setup Vulkan SDK
    wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
    sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list http://packages.lunarg.com/vulkan/lunarg-vulkan-jammy.list
    sudo apt install vulkan-sdk
'