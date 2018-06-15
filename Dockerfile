py		

FROM ubuntu:16.04
MAINTAINER Alberto Soragna alberto dot soragna @gmail.com

# working directory
ENV HOME /root
WORKDIR $HOME

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    nano \
    python-pip python-dev \
    ipython

# pip
RUN pip install --upgrade pip

# tensorflow >= 1.3.0 and tensor2tensor >= 1.4.0
#RUN export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/cuda/extras/CUPTI/lib64
RUN \
    pip install tensorflow && \
    pip install tensor2tensor


RUN \
    git clone https://github.com/dongjun-Lee/rnn-text-classification-tf.git

RUN \
    cd rnn-text-classification-tf/ && \
    pip install -r requirements.txt && \
    cd -

COPY \
    ./script $HOME/script


CMD ["bash"]
