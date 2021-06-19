# Training Own Object Detector 

- [Setup the Environment](#setup)
  - [TnesorFlow](#setup-tf)
  - [YOLOv5-Ultralytics-PyTorch](#setup-yolo)
- [Create Dataset](#ds)
- [Convert Dataset to Correct Format](#format)
  - [TensorFlow](#format-tf)
  - [YOLOv5-Ultralytics-PyTorch](#format-yolo)
- [Adjust Model COnfiguration](#config)
  - [TensorFlow](#config-tf)
  - [YOLOv5-Ultralytics-PyTorch](#config-yolo)
- [Train](#train)
  - [TensorFlow](#train-tf)
  - [YOLOv5-Ultralytics-PyTorch](#train-yolo)

This directory contains scripts that you can use to train your own object detectors using either [TensorfFlow object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) or YOLOv5 helper-tools provided by [Ultralytics](https://github.com/ultralytics/yolov5). This is not a tutorial though. I assume that you already know the theory behind object detection, but got lost somewhere along the implementation line. Below is the list of procedures that you can use to get back on track. 

In short: first we'll create own hand dataset and store it in a required format, then we'll use these records and some supplementary files to train the detector. 

<p style="font-size: 8pt;">YOLOv5 training procedure is very toroughly documented by <a href="https://github.com/ultralytics/yolov5">Ultralytics</a>, so additional comments from my side might seem a bit redundant. Still, I'll leave those here just for the same of completeness, since this project uses YOLOv5.</p>

## Setup the Environment <a name="setup"></a>
### TensorFlow <a name="setup-tf"></a>
First, let's sort the dependencies. To use tensorflow object detection API we'll need:
```
tensorflow==2.4.2
opencv-python==4.4.0.40
argparse==1.1
pillow
```

These can be installed with `pip`:
```bash
$ python -p pip install -r requiremenxt.txt
```

### YOLOv5-Ultralytics-PyTorch <a name="setup-yolo"></a>
To train PyTorch implementation of YOLOv5 as done by [ultralytics](https://github.com/ultralytics/yolov5) we'll need:
```
torch==1.9.0
opencv-python==4.4.0.40
argparse==1.1
pillow
```

These can be installed with `pip`:
```bash
$ python -p pip install -r requiremenxt.txt
```

Additionally, we'll need to clone ultralytics yolov5 repo to our machine:
```bash
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ pip install -r requirements.txt
```

## Create Dataset <a name="ds"></a>
We'll create a very simplictic dataset with only one object per image. This is not particularly good, but using the procedure described in [this blog](https://www.learnopencv.com/training-a-custom-object-detector-with-dlib-making-gesture-controlled-applications/?ck_subscriber_id=546165186) we'll be able to create our dataset very fast (no manual labelling required!). Besides, for all the purposes that we'd want to use this dataset it would be quite sufficient. In short: we'll *first* set the positions of the bounding boxes, and *then* try to fit the hand into box on each frame.

Before creating the dataset, you can run the `camera_test.py` to test your camera and adjust the bounding box size (`winsize` = `width`x`height`) and striding speed (`skip` = number of frames skipped before taking a record):
```bash
$ python vid_stream_blank.py --winsize 130x190 --skip 17
```
Increament `skip` to speed up the sliding bounding box and decrement it to slow the bounding box down.

Once you've adjusted the camera and bounding box settings you start recording frames and box positions:
```bash
$ python mk_dataset.py --datasetroot dataset --class hand1 --clear False --winsize 130x190 --skip 17
```
All arguments are optional; above are the default settings. Use `--clear True` to overwrite the existing records and `--clear False` to append those. Use `--datasetroot <my dataset directory>` to specify the directory in which the records are going to be stored. If you want to record different hand gestures specify `--class <my class>` argument. You can rerun this script for the same or different classes as many times as you want to collect records for training.

If you use default setting `dataset` directory will be created with the folliwing structure:
```
dataset
├── boxes_hand.txt
├── hand1
|   ├── 0.jpeg
|   └── ...
├── hand2
|   ├── 0.jpeg
|   └── ...
└── hand3
    ├── 0.jpeg
    └── ...
```
`dataset/boxes_hand1.txt` would contain bounding box records for each image in the following format `file_name: xmin,ymin,xmax,ymax`, e.g.:
```
0.jpeg:25,10,155,200
1.jpeg:50,10,180,200
2.jpeg:75,10,205,200
...
```

## Convert Dataset to Correct Format <a name="format"></a>
### TensorFlow <a name="format-tf"></a>
Once you've collected enough records, we can split the dataset into train and test sets and write the annotations in `json` for easier parsing later on. To do that run:
```bash
$ python mk_json_anno.py --datasetroot dataset --classes hand1,hand2,hand3
```
Use `datasetroot` to specify your dataset root directory. Use `classes` to specify all the recorded classes separated by commas.
Running `mk_anno_json.py` will create three files (`anno_train.json`, `anno_test.json` and `anno_all.json`) in your dataset root directory. 

Once this is done, we have everything ready to create tensorflow `.record` files required for tensorflow object detection API. To do that run the comman below (Notice, that `mk_tf_record.py` requires tensorflow v2.3 or v2.4!):
```bash
$ python mk_tf_record.py --anno_json ./dataset/anno_train.json --output_path ./dataset/train.record
```
Arguments `anno_json` and `output_path` are required. The former is the path to json annotation file. The latter is the path to the output tensorflow `.record` file. `mk_tf_record.py` will generate tensorflow `.record` file with the name you provided, as well as `label_map.pbtxt` file in the same directory as `.record`. The latter should contain class name-to-index map in a format required by tensorflow API.

### YOLOv5-Ultralytics-PyTorch <a name="format-yolo"></a>
Recall that our origin dataset is stored in a directory with the following structure:
```
dataset
├── boxes_hand.txt
├── hand1
|   ├── 0.jpeg
|   └── ...
├── hand2
|   ├── 0.jpeg
|   └── ...
└── hand3
    ├── 0.jpeg
    └── ...
```

To use Ultralytics YOLOv5 API we need to store our data in a directory with the following structure:
```
dataset_yolov5/
├── images
│   ├── test
│   │   ├── hand1_0.jpeg
|   |   └── ...
│   └── train
│       ├── hand1_1.jpeg
│       ├── hand1_2.jpeg
│       ├── hand2_0.jpeg
|       └── ...
└── labels
│   ├── test
│   │   ├── hand1_0.txt
|   |   └── ...
│   └── train
│       ├── hand1_1.txt
│       ├── hand1_2.txt
│       ├── hand2_0.txt
|       └── ...
```
with textual annotations having the following structure:
```
<class index> <bbox x_center> <bbox y_center> <bbox width> <bbox height>
```
e.g.:
```
0 0.17968 0.21875 0.20312 0.39583
```
All the coordinates/dimensions should relative (relative to image dimensions) and class indices should begin with `0`.

To do the conversion run:
```bash
$ python mk_yolov5_record.py --src dataset --tar dataset_yolov5
```
This will create a new directory (`dataset_yolov5`) with required structure.

## Adjust Model COnfiguration <a name="config"></a>
### TensorFlow <a name="config-tf"></a>
One last thing we need to sort before we can start training the detector is model `.config` file. This directory contains sample config file that you can use to train efficientdet0 (`efficientdet_d0_coco17_tpu-32_pipeline.config`). Alternatively you can pull blank config file from [here](https://github.com/tensorflow/models/tree/master/research/object_detection/configs/tf2), e.g.:
```bash
$ wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/ssd_efficientdet_d0_512x512_coco17_tpu-8.config
```
You'd need to fill the blanks at the end of the file to look something like this:
```python
...
  fine_tune_checkpoint: "./mymodels/efficientdet_d0_coco17_tpu-32/checkpoint/ckpt-0" # path to your model checkpoint (see colab notebook)
  num_steps: 300000
  startup_delay_steps: 0.0
  replicas_to_aggregate: 8
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
  fine_tune_checkpoint_type: "detection" # depending on the model this should be either `detection` or `fine_tune` (for centernet)
  use_bfloat16: true
  fine_tune_checkpoint_version: V2
}
train_input_reader: {
  label_map_path: "./dataset/label_map.pbtxt" # path to your label map
  tf_record_input_reader {
    input_path: "./dataset/train.record" # path to your train.record file
  }
}

eval_config: {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
  batch_size: 1; # high batch size might cause memory allocation issues
}

eval_input_reader: {
  label_map_path: "./dataset/label_map.pbtxt" # path to your label map
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "./dataset/test.record" # path to your test.record file
  }
}
```

### YOLOv5-Ultralytics-PyTorch <a name="config-yolo"></a>
The *actual* model config file used by ultralytics API (one of `yolov5s.yaml`, `yolov5m.yaml` or `yolov5l.yaml`) is stored in `yolov5/models` subdirectory in a direcotory that you've cloned from ultralytics. **Do not edit** this file.

Instead, in `yolov5/data/` subdirecoty create a config file `custom_hands.yaml` with the following contents:
```
train: /path/to/dataset_yolov5/images/train/
val: /path/to/dataset_yolov5/images/test/
nc: 3
names: ['hand1', 'hand2', 'hand3']
```
Replace paths (`train`, `val`), number of classes (`nc`) and class names (`names`).

## Train <a name="train"></a>
Finally, we can start training the detector! 
### TensorFlow <a name="train-tf"></a>
Check [this colab notebook](https://colab.research.google.com/drive/1-iT15Ib5CIFZtNFER8olIIPASHTyZZ0p?usp=sharing) and follow the instructions.

### YOLOv5-Ultralytics-PyTorch <a name="train-yolo"></a>
YOLO is sufficiently lightweight that we can train it locally. To do it in `yolov5` subdirectory run the following:
```bash
$ python3 train.py --img-size 640 --batch 4 --epochs 3 --data custom_hands.yaml --weights yolov5s.pt
```
Adjust the image size, batch size and number of epochs. Additionally, instead of the smallest yolo models (`yolov5s.pt`) you can try larger ones (`yolov5m.pt` or `yolov5l.pt`).

Saved model can be found in `yolov5/runs/train/exp/weights/best.pt`. The model can be easily loaded with `torch.load('/path/to/my/model.pt')`. However, since it was saved using Python's pickle module under the hood, unpickling it would require having the same directory structure as the one that was there during saving. That's quite restrictive. Therefore, we'll load the model **from `volov5` subdir** and save only its sate dictionary, which doesn't pose any restrictions when loading:
```python
import torch 

def load_model(weights, device=None):
    if device is None:
        device = torch.device('cpu')
    with open(weights, 'rb') as f:
        loaded = torch.load(f, map_location=device)
    return loaded['model']

model = load_model('runs/train/exp/weights/best.pt')
torch.save(
  {'model_state_dict': model.state_dict()},
  'my_state_dict.pt'
)
```
