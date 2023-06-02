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