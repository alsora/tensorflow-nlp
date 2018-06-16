
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
    



#RUN export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/local/cuda/extras/CUPTI/lib64
RUN \
    pip install tensorflow && \
    pip install tensor2tensor


COPY \
    ./script $HOME/script


COPY \
    ./tf-helper $HOME/tf-helper

RUN \
    cd tf-helper/ && \
    pip install -r requirements.txt && \
    cd -


CMD ["bash"]
