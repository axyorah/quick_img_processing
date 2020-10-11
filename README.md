# Brain Dump for Some Quick Image Processing Projects
## 1. What's it all about anyway

This is a dedicated repo to dump some quick image processing projects inspired by amazing [Adrian Rosebrock's](https://www.pyimagesearch.com/) blog. I'll be updating this repo as more things pile up. Feel free to use the code as a starting point for something more elaborate.<br>
Here's a brief description of the available projects.

<table style="width:100%">
    <tr>
        <th style="text-align:left" width="33%">Demo</th>
        <th style="text-align:left"            >Description</th>
    </tr>
    <tr>
        <td width="33%">
            <img align=left width=300 src="imgs/tentacle_beard.gif"></img>
        </td>
        <td style="text-align:left"> 
            <b>Tentacle Beard!</b>
            <br>
            <br>         
            Everyone is better off with a tentacle beard!<br>
            Run `tentacle_beard.py` to capture a video stream from your machine's camera 
            and give any human-ish individual caught by the camera a fresh new look!
            <br>       
            <ul>
                <li>Most heavy-lifting is done by the 
                    <a href="https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/">facial landmark</a> detector 
                    conveniently available out-of-the-box from <a href="https://pypi.org/project/dlib/"> dlib</a>.</li>
                <li> <a href="https://pypi.org/project/dlib/"> Dlib</a>'s facial landmark detector is based on <a href="https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf"> Histogram </a> 
                    <a href="https://www.learnopencv.com/histogram-of-oriented-gradients/"> 
                        of Oriented Gradients</a> (HOG) detector and a 
                    linear <a href="https://en.wikipedia.org/wiki/Support-vector_machine">SVM</a> classifier.                    
                </li>
                <li>Facial landmarks with indices 3-15 
                    (<a href="https://www.researchgate.net/figure/The-68-points-mark-up-used-for-our-annotations-32_fig2_313584884">img</a>) 
                    serve as the anchor points for 13 beard tentacles<br>
                </li>                    
                <li>Tentacles are animated 
                    through the magic of sine-waves and smooth "randomness" of 
                    <a href="https://en.wikipedia.org/wiki/Perlin_noise"> Perlin noise</a>:
                    <br><br>
                    <ul style="list-style-type:circle;">
                        <li>Each tentacle is composed of a series of straight segments 
                            with exponentially decaying length;
                        </li>
                        <li>
                            For each video frame the angles between the consecutive segments 
                            follow simple sine rule;
                        </li>
                        <li>
                            Amplitude, frequency and phase shift of the sine wave 
                            are sampled from the 2D Perlin matrix;
                        </li>
                        <li>
                            Each tentacle has a dedicated row in the Perlin matrix;
                            and for each new video frame the `next` element of the row is selected
                        </li>
                    </ul>
                </li>
            </ul>  
        </td>
    </tr>
    <tr>
        <td width="33%"> 
            <img align=left width=300 src="imgs/hand_controls.gif"></img> 
        </td>
        <td style="text-align:left"> 
            <b>Hand Controls</b>
            <br><br>           
            Ever felt like controlling the video settings of your camera 
            without actually touching the keyboard?<br>
            Run `hand_controls.py` and toggle custom video switches 
            by waving hands around the control buttons!<br>
            So far I've only added control buttons to resize the video frame 
            and to blur out the faces <br>
           (who knows, might come in handy...).
            <br>         
            <ul>
                <li>The most compute-heavy part of this project is the hand detector 
                    (`dnn/frozen_inference_graph_for_hand_detection.pb`): 
                </li>
                    <ul style="list-style-type:circle;">
                        <li>
                            Base architecture: <a href="https://arxiv.org/abs/1512.02325">SSD</a> 
                            <a href="https://arxiv.org/abs/1704.04861">MobileNet V1</a> trained on 
                            <a href="http://cocodataset.org/#home">COCO dataset</a> 
                            available out-of-the-box from 
                            <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md">tensorflow</a>;                            
                        </li>
                        <li>
                            Assembled with 
                            <a href="https://github.com/tensorflow/models/tree/master/research/object_detection">Tensorflow Object Detection API</a> 
                            and trained on 
                            <a href="http://vision.soic.indiana.edu/projects/egohands/">Egohands dataset</a>;
                        </li>
                    </ul>
                <li>Each video frame is fed to the hand detector, 
                    which in turn produces the bounding boxes for the hands
                    (if present)
                </li>
                <li>
                    If the bounding box of a hand overlaps with the area taken by any of the control buttons,
                    the action corresponding to that button 
                    (e.g. resize video frame or blur all faces)
                    if executed                    
                </li>
                <li>
                    For face detection (needed for bluring the corresponding area)
                    I use simple <a href="https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html">HAAR cascade</a>, 
                    available in <a href="https://github.com/opencv/opencv/tree/master/data/haarcascades">OpenCV</a>.
                </li>
            </ul>
        </td>
    </tr>
</table>





 

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
$ pip3 install tensorflow==1.14.0
$ pip3 install numpy==1.14.0
```

To install `dlib` on Windows use this wheel:
```
pip install https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f
```

Take care to install `tensorflow` version 1.14.0 or above, as the inference graph for the hand detector in `hand_controls.py` will give errors with the earlier versions of tensorflow.

Downgrading `numpy` to version 1.14.0 is not really necessary (yet!), but it will save you a headache of scrolling through the deprecation warnings.

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
$ git clone https://github.com/axyorah/quick_img_processing.git
```

With all the dependencies taken care of, you can tentaclify yourself by navigating to the directory, to which you have copied the contents of this repo, and running:
```
$ python3 tentacle_beard.py
```

You can regulate the degree of "wiggliness" of the tentacles through `-w` parameter, e.g.:
```
$ python3 tentacle_beard.py -w 0.3
```

Similarly, you can invoke the power of hand controls by running:

```
$ python3 hand_controls.py
```

Passing `-b 1` parameter will additionally draw bounding boxes around the hands (just like in the demo):

```
$ python3 hand_controls.py -b 1
```


### Docker
You can set up a [docker](https://www.docker.com/) container to take care of the environment. </br>
If you don't have `docker`, follow the [instructions](https://docs.docker.com/install/) for your system to install it and add yourself to `docker` group.</br>
With `docker` installed, build `docker` image by running the following command from the directory, where you have copied the contents of this repo:
```
$ docker build -t quick_img_processing .
```

Set up `docker` container by running the following command:
```
$ docker run --rm -it \
  --device=/dev/video0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $XAUTHORITY:/root/.Xauthority \
  -e DISPLAY=$DISPLAY \
  quick_img_processing
```

This should take care of accessing your machine's camera from the container and displaying the video output. If you're still getting X server related errors, check these resources:
- https://github.com/opencv/gst-video-analytics/wiki/Docker-Run
- https://towardsdatascience.com/real-time-and-video-processing-object-detection-using-tensorflow-opencv-and-docker-2be1694726e5
  