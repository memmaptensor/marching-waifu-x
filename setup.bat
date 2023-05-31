:: -----------------------------------------------------
:: Make sure anaconda is installed and 
:: to run using conda run (conda env python=3.10)
:: -----------------------------------------------------
:: Make sure to have the following installed:
::  VS2022 build tools + CMake
::  Latest NVIDIA drivers
::  cuda-toolkit 11.8
::  cuDNN 8.9.1
:: -----------------------------------------------------

:: Update submodules
git submodule update --init --recursive

:: Install 3rd party packages
python -m pip install --upgrade pip
pip install gdown ipykernel ipywidgets

:: Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Build instant-ngp
:: ... (TODO)
    ::  Optix 7.7
    ::  Vulkan SDK