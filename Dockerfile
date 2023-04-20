FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Define build arguments
ARG DEBIAN_FRONTEND=noninteractive
ARG DEBCONF_NOWARNINGS="yes"
ARG PIP_ROOT_USER_ACTION=ignore

# install packages
RUN apt-get update && apt-get install -y \
    nano \
    git \
    g++ \
    gcc \
    libgl1 \
    libglib2.0-0

# update pip package
RUN python3 -m pip install --upgrade pip

# install pytorch geometric for GPU
RUN python3 -m pip install --no-cache-dir -f https://data.pyg.org/whl/torch-1.11.0+cu113.html\
    torch-scatter==2.0.9 \
    torch-sparse==0.6.15 \
    torch-cluster==1.6.0 \
    torch-spline-conv==1.2.1 \
    torch-geometric==2.1.0.post1

# copy src code
COPY . /CamRaDepth

# install requirements
RUN python3 -m pip install --no-cache-dir -r /CamRaDepth/requirements.txt

# set working directory and entry point 
WORKDIR /CamRaDepth
ENTRYPOINT [ "/bin/bash"]