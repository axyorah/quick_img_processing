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
            <img width=300 src="imgs/tentacle_beard.gif">
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
            <img width=300 src="imgs/hand_controls.gif"> 
        </td>
        <td style="text-align:left"> 
            <b>Hand Controls</b>
            <br><br>
            Ever felt like controlling the video settings of your camera 
            without actually touching the keyboard?<br>
            Run <code>hand_controls.py</code> and toggle custom video switches 
            by waving hands around the control buttons!<br>
            So far I've only added control buttons to resize the video frame 
            and to blur out the faces <br>
           (who knows, might come in handy...).
            <br>
            <ul>
                <li>The most compute-heavy part of this project is the hand detector 
                    (<code>dnn/efficientdet_hand_detector/</code>): 
                </li>
                    <ul style="list-style-type:circle;">
                        <li>
                            Base architecture: <a href="https://arxiv.org/abs/1512.02325">SSD</a> detector with <a href="download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz">EfficientDet D0 512x512</a> backbone trained on 
                            <a href="http://cocodataset.org/#home">COCO dataset</a> 
                            available out-of-the-box from 
                            <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md">tensorflow</a>;
                        </li>
                        <li>
                            Assembled with 
                            <a href="https://github.com/tensorflow/models/tree/master/research/object_detection">Tensorflow Object Detection API</a> 
                            and trained on 
                            <a href="#customdataset">custom dataset</a> with 3907 images of a hand (open palm);
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
                    is executed
                </li>
                <li>
                    For face detection (needed for bluring the corresponding area)
                    I use simple <a href="https://docs.opencv.org/3.4.1/d7/d8b/tutorial_py_face_detection.html">HAAR cascade</a>, 
                    available from <a href="https://github.com/opencv/opencv/tree/master/data/haarcascades">OpenCV</a>.
                </li>
            </ul>
        </td>
    </tr>
    <tr>
        <td width="33%">
            <img width=300 src="imgs/casting_hands.gif">
        </td>
        <td style="text-align:left">
            <b>DnD Enhancer</b>
            <br>
            <br>
            Has 2020  disrupted your DnD sessions and you had to migrate to Skype? 
            Well, at least you use some image processing magic to make DnD magic feel more lively....
            As long as you're ok with casting spells by means of waving hand signs, of course.
            <br>
            <br>
            I trained object classifier on four classes of hand gestures 
            and added some animatted patterns which are triggered by corresponding class detections:
            <ul>
                <li> Open Palm triggers generic spell effect. 
                </li>
                <li> Fist triggers some crowd control: 
                You cast <a href="https://roll20.net/compendium/dnd5e/Dominate%20Person#content">Dominate Person</a>! 
                It's super effective!
                </li>
                <li> <a href="https://naruto.fandom.com/wiki/Body_Flicker_Technique">Teleportation Jutsu</a> 
                creates a puff of smoke: Poof! You cast <a href="https://roll20.net/compendium/dnd5e/Misty%20Step#content">Misty Step</a>!
                </li>
                <li> <a href="https://en.wikipedia.org/wiki/Sign_of_the_horns">Horns</a> invoke a lightning! You cast Lightning Bolt!
                </li>
            </ul>
            <br>
            Here are some technicalities:
            <ul>
                <li> <b>Dataset</b>. For this project I use <a href="#customdataset">custom hand gesture</a> dataset with 11056 images belonging to four classes, as described below.
                </li> 
                <li><b>Model</b>. This project uses <a href="https://arxiv.org/abs/1512.02325">SSD</a> detector with 
                <a href="https://arxiv.org/abs/1704.04861">MobileNet</a> backbone pretrained on <a href="https://cocodataset.org/#home">COCO</a> dataset 
                downloaded from <a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md">
                Tensorflow Detection Model Zoo</a>. The model was assembled with 
                <a href="https://github.com/tensorflow/models/tree/master/research/object_detection">Tensorflow Object Detection API</a> 
                and trained on a dataset described above for about 7000 steps on GPU. 
                Model's <code>.pb</code> file and <code>variables</code> are available in <code>dnn/ssd_mobilenet_gesture_detector/</code>.
                </li>
                <li> <b>Pattern Effects</b>
                    <ul>
                        <li> Open Palm effect is made from several individually rotating polygons. The rotation angle for each polygon is sampled from Perlin matrix. Check the code at <code>utils.pattern_utils.Pattern</code>, <code>utils.pattern_utils.Polygon</code> and <code>utils.pattern_utils.HandPatternEffect</code>. (I will probably add a more detailed description at some point in the future)
                        </li>
                        <li> Fist effect is similar to Open Palm effect, just instead of polygons it uses precalculated vertices. On each frame the pattern for fist effect rotates around its center about the angle sampled from Perlin matrix.
                        Check the code at <code>utils.pattern_utils.FistPatternEffect</code>.
                        </li>
                        <li> Puff of smoke for Teleportation Jutsu is made by following <a href="https://github.com/ssloy/tinykaboom">this</a> tutorial 
                        by <a href="https://github.com/ssloy">Dmitry Sokolov</a>. 
                        In short - explosion is a sphere "wrapped" into a Perlin noise with all the lights and shadows calculated with the help of a 
                        simplified ray tracer. Puffs of smoke are not calculated on the fly (that would be very computationally expensive). Instead some premade  explosion images (<code>imgs/transp_kaboomX.png</code>) are loaded at the beginning of the script and added on top of the current frame whenever 5 out of 10 last frames had teleportation jutsu detections. 
                        Check the code at <code>utils.pattern_utils.JutsuPatternEffect</code>.
                        </li>
                    </ul>
                </li>
            </ul>
        </td>
    </tr>
</table>

### *Custom Dataset <a name="customdataset"></a>
To train hand detectors I used custom dataset floowing the procedure shared by amazing people from <a href="https://www.learnopencv.com/">LearnOpenCV</a>. The exact procedure is described in the beginning of <a href="https://www.learnopencv.com/training-a-custom-object-detector-with-dlib-making-gesture-controlled-applications/?ck_subscriber_id=546165186">this tutorial</a> on HOG-based object detection. 

Hand dataset contains 3907 images of open palm. Hand gensture dataset contains 11056 images of four classes: hand (open palm), fist, "teleportation jutsu", "horns", with approximately 2700 images per class. Each image contains only one object (hand gesture), and both datasetes feature hands that mostly belong to one person (well, me). Additionally, all objects in the dataset are located approximately 1m away from the camera. This drastically narrows down the potential use cases of the detectors trained on these datasets, but for the purposes of this project the datasets are quite sufficient.

## 2. Getting Started

Copy this repo to your device by 
```
$ git clone https://github.com/axyorah/quick_img_processing.git
```

Your device should have a functioning camera and [python 3](https://www.python.org/download/releases/3.0/) installed.

You can run this project from [docker container](#docker), which will take care of all the dependencies, or you can take care of the dependencies [yourself](#manually).

### 2.1 Dependencies <a name="manually"></a>
This project uses the following python packages:
```
imutils
argparse
matplotlib
opencv-python
dlib
tensorflow
numpy
```

Before installing these packages make sure that their dependencies are taken care of. To compile `dlib` you need to have [cmake](https://cmake.org/) and C++ compiler.</br>
On Linux both are covered by:
```
$ sudo apt-get update && sudo apt-get cmake
```
On Windows you can get `cmake` from https://cmake.org/download/ and `gcc` compiler for C++ from https://osdn.net/projects/mingw/releases/.

To use `OpenCV` on Linux you might need to have the following installed:
```
$ sudo apt-get install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx
```

Now, to install the packages run:
```
$ python -m pip install -r requirements.txt
```

If you're having problems installing `dlib` on Windows try this wheel:
```
pip install https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f
```

### 2.2 Use Docker Container <a name="docker"></a>
(This will not work if you are on Windows machine, as accessing your Windows camera from docker container is a bit [complicated](https://medium.com/@jijupax/connect-the-webcam-to-docker-on-mac-or-windows-51d894c44468))</br>
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
  
## 3. Use Example
you can tentaclify yourself by navigating to the directory, to which you have copied the contents of this repo, and running:
```
$ python tentacle_beard.py
```

You can regulate the degree of "wiggliness" of the tentacles through `-w` parameter, e.g.:
```
$ python tentacle_beard.py -w 0.3
```

Similarly, you can invoke the power of hand controls by running:

```
$ python hand_controls.py
```

Passing `-b 1` parameter will additionally draw bounding boxes around the hands (just like in the demo):

```
$ python hand_controls.py -b 1
```

To try the DnD enhancer simply run:
```
$ python casting_hands.py
```