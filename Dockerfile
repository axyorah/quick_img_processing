FROM ubuntu:18.04

LABEL maintainer="axeh (elvira.bozileva@gmail.com)"

# copy contents of the current dir (on host) to container
RUN mkdir /home/img_processing_projects
COPY . /home/img_processing_projects
WORKDIR /home/img_processing_projects

# fix X Error (OpenCV's cv.imshow error)
ENV QT_X11_NO_MITSHM=1

# toggle XLA flag for tensorflow
ENV TF_XLA_FLAGS=--tf_xla_cpu_global_jit

# get python and pip
RUN apt-get update && apt-get install -y python3 python3-pip 

# get everything needed for dlib
RUN apt-get install -y cmake 
# RUN apt-get install -y build-essential libgtk-3-dev libboost-all-dev

# get everything needed for opencv
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# get all python dependencies
RUN pip3 install imutils
RUN pip3 install argparse==1.1
RUN pip3 install matplotlib=3.1.3
RUN pip3 install opencv-python==4.4.0
RUN pip3 install dlib==19.21.0
RUN pip3 install tensorflow==2.3.1
RUN pip3 install numpy==1.17.2



