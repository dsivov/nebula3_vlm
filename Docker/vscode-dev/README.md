# Running VSCode Developent Container

For running vs-code under development container follow the following process:
(Assuming you already cloned your project)
1. create a folder called .devcontainer on the root project folder
2. copy the content of this folder to .devcontainer
3. copy the environment.yml file from the project root to .devcontainer
4. in devcontainer.json:
   - add local mounts to mounts
   - add env params to containerEnv
5. F1 > Remote-Containers: Rebuild and Reopen in Container
6. It will take a few moments
7. If you have changes to devcontainer.json or the Dockerfile choose F1 > Remote-Containers: Rebuild Container

Notes:
- usually the vscode expects the project name to be the same inside the container (not 'notebook') in that case
  it will propmt workspace not found, just choose open folder and you'll see your porject files, just click ok.
- docker will keep the container running even if you close vscode, so if you really finished working:
  1. docker ps
  2. docker fm -f [container id]
  3. docker system prune (this will also clean temp build images and other stuff)
  you'll be surprised how much space you'll free :smile:
