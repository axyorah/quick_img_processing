FROM ubuntu:18.04

LABEL maintainer="axeh (elvira.bozileva@gmail.com)"

# copy contents of the current dir (on host) to container
RUN mkdir /home/tentacle_beard
COPY . /home/tentacle_beard
WORKDIR /home/tentacle_beard

# fix X Error (OpenCV's cv.imshow error)
ENV QT_X11_NO_MITSHM=1

# get python and pip
RUN apt-get update && apt-get install -y python3 python3-pip 

# get everything needed for dlib
RUN apt-get install -y cmake 
# RUN apt-get install -y build-essential libgtk-3-dev libboost-all-dev

# get everything needed for opencv
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# get all python dependencies
RUN pip3 install imutils
RUN pip3 install argparse
RUN pip3 install matplotlib
RUN pip3 install opencv-python
RUN pip3 install dlib
RUN pip3 install tensorflow==1.14.0
#RUN pip3 install numpy



