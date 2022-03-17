#!/bin/bash
source activate nebula_vlm 
export SHELL=/bin/bash
rm -rf /storage/lost+found
export PYTHONPATH=/notebooks/:/notebooks/nebula3_database
cd /notebooks && git clone https://github.com/dsivov/nebula3_vlm.git . &&  git submodule init && git submodule update 
chmod -R a+w /notebooks
jupyter lab --ip=0.0.0.0 --no-browser --allow-root --LabApp.trust_xheaders=True --LabApp.disable_check_xsrf=False --LabApp.allow_remote_access=True --LabApp.allow_origin='*'
