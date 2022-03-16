FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04
# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8
LABEL com.nvidia.volumes.needed="nvidia_driver"
COPY environment.yaml /environment.yaml
ENV PYTHON_VERSION=3.9
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PATH=$PATH:/opt/conda/bin/
ENV USER nebula

RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    ca-certificates \
    build-essential \
    gcc \
    git \
    zip \
    unzip \
    openssh-client && \
    apt-get install -y --no-install-recommends curl wget && \
    apt-get install -y --no-install-recommends python-dev python3-pip && \
    rm -rf /var/lib/apt/lists/* && \
    curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install conda-build && \
    conda env create -f environment.yaml && \
    conda clean --all -y 
# Create Environment
ENV PATH /opt/conda/bin:$PATH
ENV PATH /opt/conda/envs/nebula/bin:$PATH
COPY run.sh /run.sh
RUN mkdir -p /opt/models

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering
#  our standard output stream, which means that logs can be delivered to the
# user quickly. PYTHONDONTWRITEBYTECODE keeps Python from writing the .pyc
# files which are unnecessary in this case. We also update PATH so that the
# train and serve programs are found when the container is invoked.
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY src /opt/program
RUN chmod +x /opt/program/serve
WORKDIR /opt/program
ENTRYPOINT ["serve"]
#CMD ["/run.sh"]
