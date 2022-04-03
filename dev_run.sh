#!/bin/bash
export SHELL=/bin/bash
rm -rf /storage/lost+found
export PYTHONPATH=/notebooks/:/notebooks/nebula3_vlm/nebula3_database:/notebooks/nebula3_vlm
if [ ! -d "/notebooks/nebula3_vlm" ]
then
  cd /notebooks && git clone https://github.com/dsivov/nebula3_vlm   
  cd /notebooks/nebula3_vlm && git submodule init && git submodule update
  cd /notebooks/nebula3_vlm/nebula3_database && git checkout main  
fi
cd /notebooks
if [ ! -d "/notebooks/conda" ]
then
    curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /notebooks/conda && \
    rm ~/miniconda.sh && \
    /notebooks/conda/bin/conda install conda-build 
fi
chmod -R a+w /notebooks
if [ -z "$JUPYTER_TOKEN" ]; then
    JUPYTER_TOKEN=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 48 | head -n 1)
fi

# Note: print mocked jupyter token so that we can run this container as if it is a notebook within Gradient V1
echo "http://localhost:8888/?token=${JUPYTER_TOKEN}"
echo "http://localhost:8888/\?token\=${JUPYTER_TOKEN}"
export XDG_DATA_HOME=/notebooks
PASSWORD=${JUPYTER_TOKEN} /usr/bin/code-server --bind-addr "0.0.0.0:8888" .
