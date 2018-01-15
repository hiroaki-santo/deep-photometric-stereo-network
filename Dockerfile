FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04
MAINTAINER Hiroaki Santo

RUN apt-get update && apt-get install -y build-essential cmake libgoogle-glog-dev libatlas-base-dev libeigen3-dev libsuitesparse-dev apt-file git vim wget unzip python-dev python-pip python-numpy python-matplotlib libopencv-dev python-opencv libboost-all-dev \
  && apt-get autoclean \
  && apt-get autoremove \
  && rm -rf /var/lib/apt/lists/*
RUN pip install tqdm

WORKDIR /
RUN git clone https://github.com/ndarray/Boost.NumPy
WORKDIR /Boost.NumPy
RUN cmake .
RUN make install
ENV LD_LIBRARY_PATH="/usr/local/lib64:${LD_LIBRARY_PATH}"

RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp27-none-linux_x86_64.whl
