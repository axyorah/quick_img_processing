# Everyone is Better Off with a Tentacle Beard!
## 1. What's it all about anyway

Run `tentacle_beard.py` to capture a video stream from your device's camera and give any human-ish individual caught by the camera a fresh new look!

<img src="imgs/tentacle_beard.gif"></img>

Most heavy-lifting is done by the [facial landmark](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/) detector conveniently available from [dlib](https://pypi.org/project/dlib/). Facial landmarks are used to anchor the tentacles, which are animated through the magic of sine-waves and smooth "randomness" of [Perlin noise](https://en.wikipedia.org/wiki/Perlin_noise).

## 2. Requirements
Your device should have a functioning camera and [python 3](https://www.python.org/download/releases/3.0/) installed. </br>
Ideally you'd want to run this (any) project from a [docker](https://www.docker.com/) container (see below the instructions for setting up a container). But if you're feeling adventurous, you can take care of all the dependencies manually.

### Dependencies
```
$ pip3 install imutils
$ pip3 install argparse
$ pip3 install matplotlib
$ pip3 install opencv-python
$ pip3 install dlib
```

To install `dlib` on Windows use this wheel:
```
pip install https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f
```

### Dependencies of dependencies
To compile `dlib` you need to have [cmake](https://cmake.org/) and C++ compiler.</br>
On Linux both are covered by:
```
$ sudo apt-get update && sudo apt-get cmake
```
On Windows you can get `cmake` from https://cmake.org/download/ and `gcc` compiler for C++ from https://osdn.net/projects/mingw/releases/.

To use `OpenCV` on Linux you might need to have the following installed:
```
$ sudo apt-get install -y libsm6 libxext6 libxrender-dev
```

## 3. Installation
Copy this repo to your device by 
```
$ git clone https://github.com/<Oopsy! It's not on github yet!>
```

With all the dependencies taken care of, you can tentaclify yourself by navigating to the directory, to which you have copied the contents of this repo, and running:
```
$ python3 tentacle_beard.py
```

You can regulate the degree of "wiggliness" of the tentacles through `-w` parameter, e.g.:
```
$ python3 tentacle_beard.py -w 0.3
```

### Docker
You can set up a [docker](https://www.docker.com/) container to take care of the environment. </br>
If you don't have `docker`, follow the [instructions](https://docs.docker.com/install/) for your system to install it and add yourself to `docker` group.</br>
With `docker` installed, build `docker` image by running the following command from the directory, where you have copied the contents of this repo:
```
$ docker build -t tentacle_beard .
```

Set up `docker` container by running the following command:
```
$ docker run --rm -it \
  --device=/dev/video0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $XAUTHORITY:/root/.Xauthority \
  -e DISPLAY=$DISPLAY \
  tentacle_beard
```

This should take care about accessing your device camera from the container and displaying the video output. If you're still getting X server related errors, check these resources:
- https://github.com/opencv/gst-video-analytics/wiki/Docker-Run
- https://towardsdatascience.com/real-time-and-video-processing-object-detection-using-tensorflow-opencv-and-docker-2be1694726e5
  