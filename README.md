# nebula3_vlm
Visual Language Models for Nebula3. Implemented in form of microservice
Currently only CLIP API will be implemented. 
Structure:
1. database submodule - submodule for arango/milvus access
2. lvm/clip_api.py - clip API
3. notebooks/ - notebooks for test/debug/play
4. models/ - code for vlm model microservice
5. src/ - infrustructure for microservice
6. config.py - database connection settings
7. Dockerfile - docker image definition
8. run.sh - image entry point
9. environment.yaml - python dependencies
