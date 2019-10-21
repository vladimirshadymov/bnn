FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
MAINTAINER Vladimir Shadymov <vladimir.shadymov@gmail.com>

# Install build utilities
RUN apt-get update && \
	apt-get install -y gcc make \
 	apt-transport-https ca-certificates \
	build-essential curl

# Install miniconda to /miniconda3
RUN curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda3 -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda3/bin:${PATH}
RUN conda update -y conda

# Creating conda environment
RUN conda create -n bnn python=3.6
RUN echo "source activate bnn" > ~/.bashrc
ENV PATH /miniconda3/envs/bnn/bin:$PATH

# Check Python version
RUN python3 --version

# Install required packages
RUN conda install pytorch=1.3.0 torchvision=0.4.1 cudatoolkit=10.1 -c pytorch
RUN conda install numpy scipy matplotlib seaborn pandas jupyter -c conda-forge

# Create a working directory
RUN mkdir /project
WORKDIR /project

