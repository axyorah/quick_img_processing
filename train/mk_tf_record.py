"""
use example:
    python ./mk_tf_record.py \
    --anno_json ./dataset/anno_train.json \
    --output_path ./dataset/train.record

please use tf v2.3.0 or v2.3.1

most code is taken from:
    https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records
"""
import hashlib
import io
import os

import PIL.Image
import tensorflow as tf
import json
import argparse

#from object_detection.utils import dataset_util

def get_args():
    """parse input data"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anno_json",
        help="Path to anno.json file."
    )
    parser.add_argument(
        "--output_path",
        help="Path to output TFRecord"
    )
    parser.add_argument(
        "--classes", 
        default="hand",
        help="recorded classes; if there are several classes,\n" +\
            "they need to be separated with commas, e.g.:\n" +\
            "  `--classes hand1,hand2,hand3`;\n" +\
            "if no classes are specified default `hand` class will be used."
    )

    return vars(parser.parse_args())

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def dict_to_tf_example(data, label_map_dict): 
    """Convert JSON derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding fields from anno.json file for a single image
      label_map_dict: A map from string label names to integers ids.

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """
    full_path = data["filename"]
    with tf.io.gfile.GFile(full_path, "rb") as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != "JPEG":
        raise ValueError("Image format not JPEG")
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data["size"]["width"])
    height = int(data["size"]["height"])

    xmin, ymin, xmax, ymax = [], [], [], []
    classes, classes_text = [], []
    if "object" in data:
        for obj in data["object"]:
    
            x1 = float(obj["bndbox"]["xmin"])
            y1 = float(obj["bndbox"]["ymin"])
            x2 = float(obj["bndbox"]["xmax"])
            y2 = float(obj["bndbox"]["ymax"])
      
            x1, x2 = min([x1,x2]), max([x1,x2])
            y1, y2 = min([y1,y2]), max([y1,y2])

            xmin.append(x1 / width)
            ymin.append(y1 / height)
            xmax.append(x2 / width)
            ymax.append(y2 / height)
            classes_text.append(obj["name"].encode("utf8"))
            classes.append(label_map_dict[obj["name"]])

    example = tf.train.Example(features=tf.train.Features(feature={
        "image/height": int64_feature(height),
        "image/width": int64_feature(width),
        "image/filename": bytes_feature(
            data["filename"].encode("utf8")),
        "image/source_id": bytes_feature(
            data["filename"].encode("utf8")),
        "image/key/sha256": bytes_feature(key.encode("utf8")),
        "image/encoded": bytes_feature(encoded_jpg),
        "image/format": bytes_feature("jpeg".encode("utf8")),
        "image/object/bbox/xmin": float_list_feature(xmin),
        "image/object/bbox/xmax": float_list_feature(xmax),
        "image/object/bbox/ymin": float_list_feature(ymin),
        "image/object/bbox/ymax": float_list_feature(ymax),
        "image/object/class/text": bytes_list_feature(classes_text),
        "image/object/class/label": int64_list_feature(classes)
    }))
    return example


def main(_):

    args = get_args()

    writer = tf.compat.v1.python_io.TFRecordWriter(args["output_path"])
  
    # get data dict from json file
    with open(args["anno_json"], "r") as f:
        anno_json = f.read()
    data = json.loads(anno_json)

    # store label map as a .pbtxt file in the same dir as .record file
    DATASETROOT = os.path.join(*args["output_path"].split(os.path.sep)[:-1])
    CLASSES = {datum["object"][0]["name"] for datum in data}
    label_map_dict = {CLASS: i+1 for i,CLASS in enumerate(CLASSES)}
    with open(os.path.join(DATASETROOT, "label_map.pbtxt"), "w") as f:
        for i,CLASS in enumerate(CLASSES):
            f.write(f"item {{\n    id: {i}\n    name: \'{CLASS}\'\n}}")
  
    # write data examples to .record file
    for datum in data:
        tf_example = dict_to_tf_example(datum, label_map_dict)
        writer.write(tf_example.SerializeToString())

    writer.close()
  
if __name__ == "__main__":
    tf.compat.v1.app.run()
