# bnn
## Build environment with docker
```
docker build -t bnn_image .
```
## Run environmnt with docker
```
cd in a project directory and then run following command
docker run -it --runtime=nvidia -v $(pwd):/project bnn_image
```
It will bind project directory with local directory in a container. All changes in the folder are synchronized!!!
