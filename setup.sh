#! /bin/bash

# -----------------------------------------------------
# Reboot after install!
# -----------------------------------------------------
# Make sure to copy over the following files to ~/
# cudnn-local-repo-ubuntu2204-8.9.1.23_1.0-1_amd64.deb
# ----------------------------------------------------- 

# Update submodules
git submodule update --init --recursive

# Setup prerequisites
sudo apt update

# Remove old NVIDIA components
sudo apt remove --purge nvidia-*
sudo apt autoremove --purge

# Setup cuda-toolkit 11.8
cd ~
sudo apt install build-essential cmake unzip
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' >> ~/.bashrc
source ~/.bashrc
sudo nano /etc/ld.so.conf # Append /usr/local/cuda/lib64 to the file here!
sudo ldconfig
rm -rf cuda_11.8.0_520.61.05_linux.run

# Setup cuDNN 8.9.1
mkdir cudnn_install
mv cudnn-local-repo-ubuntu2204-8.9.1.23_1.0-1_amd64.deb cudnn_install
cd cudnn_install
ar -x cudnn-local-repo-ubuntu2204-8.9.1.23_1.0-1_amd64.deb
tar -xvf data.tar.xz
cd var/cudnn-local-repo-ubuntu2204-8.9.1.23/
sudo dpkg -i libcudnn8_8.9.1.23-1+cuda11.8_amd64.deb
sudo dpkg -i libcudnn8-dev_8.9.1.23-1+cuda11.8_amd64.deb
sudo dpkg -i libcudnn8-samples_8.9.1.23-1+cuda11.8_amd64.deb
cd ~
rm -rf cudnn_install

# Setup anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh
sh Anaconda3-2023.03-1-Linux-x86_64.sh
source ~/.bashrc
cd ~
rm -rf Anaconda3-2023.03-1-Linux-x86_64.sh

# Setup conda env
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda create -n marching-waifu-x python=3.10
conda activate marching-waifu-x
python -m pip install --upgrade pip

# Install 3rd party packages
pip install gdown ipykernel ipywidgets

# Install PyTorch
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --index-url https://download.pytorch.org/whl/cu116

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