# Training Own Object Detector 

This directory contains scripts that you can use to train your own object detectors. This is not a tutorial though. I assume that you already know the theory behind object detection, but got lost somewhere along the implementation line. Below is the list of procedures that you can use to get back on track. In short: first we'll create own hand dataset and store it as .tfrecord on our local machine, then we'll use these records and some supplementary files to train the detector on colab. 

## Setup the Environment
First, let's sort the dependencies. We'll need:
```
tensorflow==2.3.1
opencv-python==4.4.0.40
argparse==1.1
pillow
```

These can be installed with `pip`:
```bash
$ python -p pip install -r requiremenxt.txt
```

## Creating the Dataset
We'll create a very simplictic dataset with only one object per image. This is not particularly good, but using the procedure described in [this blog](https://www.learnopencv.com/training-a-custom-object-detector-with-dlib-making-gesture-controlled-applications/?ck_subscriber_id=546165186) we'll be able to create our dataset very fast (no manual labelling required!). Besides, for all the purposes that we'd want to use this dataset it would be quite sufficient. In short: we'll *first* set the positions of the bounding boxes, and *then* try to fit the hand into box on each frame.

Before creating the dataset, you can run the `camera_test.py` to test your camera and adjust the bounding box size (`winsize` = `width`x`height`) and striding speed (`skip` = number of frames skipped before taking a record):
```bash
$ python vid_stream_blank.py --winsize 130x190 --skip 17
```
Increament `skip` to speed up the sliding bounding box and decrement it to slow the bounding box down.

Once you've adjusted the camera and bounding box settings you start recording frames and box positions:
```bash
$ python mk_dataset.py --datasetroot dataset --class hand --clear False --winsize 130x190 --skip 17
```
All arguments are optional; above are the default settings. Use `--clear True` to overwrite the existing records and `--clear False` to append those. Use `--datasetroot <my dataset directory>` to specify the directory in which the records are going to be stored. If you want to record different hand gestures specify `--class <my class>` argument. You can rerun this script for the same or different classes as many times as you want to collect records for training.

If you use default setting `dataset` directory will be created with the folliwing structure:
```
dataset
├── boxes_hand.txt
└── hand
    ├── 0.jpeg
    ├── 1.jpeg
    ├── 2.jpeg
    └── ...
```
`dataset/boxes_hand.txt` would contain bounding box records for each image in the following format `file_name: xmin,ymin,xmax,ymax`, e.g.:
```
0.jpeg:25,10,155,200
1.jpeg:50,10,180,200
2.jpeg:75,10,205,200
...
```

## Storing the Dataset as Tensorflow `.record`s
Once you've collected enough records, we can split the dataset into train and test sets and write the annotations in `json` for easier parsing later on. To do that run:
```bash
$ python mk_json_anno.py --datasetroot dataset --classes hand1,hand2,hand3
```
Use `datasetroot` to specify your dataset root directory. Use `classes` to specify all the recorded classes separated by commas.
Running `mk_anno_json.py` will create three files (`anno_train.json`, `anno_test.json` and `anno_all.json`) in your dataset root directory. 

Once this is done, we have everything ready to create tensorflow `.record` files required for tensorflow object detection API. To do that run the comman below (Notice, that `mk_tf_record.py` requires tensorflow v2.3!):
```bash
$ python mk_tf_record.py --anno_json ./dataset/anno_train.json --output_path ./dataset/train.record
```
Arguments `anno_json` and `output_path` are required. The former is the path to json annotation file. The latter is the path to the output tensorflow `.record` file. `mk_tf_record.py` will generate tensorflow `.record` file with the name you provided, as well as `label_map.pbtxt` file in the same directory as `.record`. The latter should contain class name-to-index map in a format required by tensorflow API.

## Adjusting the Model `.config` file
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

## Training
Finally, we can start training the detector! Check [this colab notebook](https://colab.research.google.com/drive/1-iT15Ib5CIFZtNFER8olIIPASHTyZZ0p?usp=sharing) and follow the instructions.