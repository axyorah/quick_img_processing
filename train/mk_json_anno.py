"""
converts bbox records stored in `boxes_< class >.txt` to json

Recall: format used in `boxes_< class >.txt`:
```
0.jpeg:xmin,ymin,xmax,ymax
1.jpeg:xmin,ymin,xmax,ymax
```

format used in .json file:
```
[
    {
        "filename": "name.jpeg",
        "size": {
            "width": int,
            "height: int,
            "depth": int
        },
        "object": [
            {
                "name": "obj_name",
                "bndbox": {
                    "xmin": int,
                    "ymin": int,
                    "xmax": int,
                    "ymax": int
                }
            }
        ]
    },
    {
        ...
    }   
]
``` 
"""
import os
import json
import argparse
import random

def get_args():
    """parse input data"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasetroot", 
        default="dataset",
        help="root directory containing the recorded class subdirectories;\n" +\
            "if nothing is specified defaults to `dataset`."
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

def check_if_exists(filename):
    if not os.path.exists(filename):
        raise Exception(
            f"\n  File {filename} does not exist! \n" +\
            "  Please check if you correctly specified `datasetroot` and `classes` when using this file.\n" +\
            "  Use example: \n    $ python3 mk_json_anno.py --datasetroot mydataset --classes hand1,hand2,hand3"
        )

def get_annotations(datasetroot, classes, test_ratio=0.3):

    annotations = []
    train_annotations = []
    test_annotations = []

    for clss in classes:
        box_file = os.path.join(datasetroot, f"boxes_{clss}.txt")
        check_if_exists(box_file)

        with open(box_file, "r") as f:
            lines = f.read().splitlines()
        
        for line in lines:
            name,rawcoors = line.split(":")
            fullname = os.path.join(datasetroot, clss, f"{name}")
            xmin,ymin,xmax,ymax = map(int, rawcoors.split(","))

            h,w,d = 480, 640, 3

            anno = dict()
            anno["filename"] = fullname
            anno["size"] = {"width": w, "height": h, "depth": d}
            anno["object"] = [
                {
                    "name": clss,
                    "bndbox": {
                        "xmin":xmin,
                        "ymin":ymin,
                        "xmax":xmax,
                        "ymax":ymax
                    }
                }
            ]

            roll = random.randint(1,100) / 100
            if roll < test_ratio:
                test_annotations.append(anno)
            else:
                train_annotations.append(anno)         
            annotations.append(anno)

    return {
        "all": annotations,
        "train": train_annotations,
        "test": test_annotations
    }

def main():
    args = get_args()
    
    DATASETROOT = args["datasetroot"]
    CLASSES = args["classes"].split(",")

    annotations = get_annotations(DATASETROOT, CLASSES, test_ratio=0.3)

    with open(os.path.join(DATASETROOT, "anno_train.json"), "w") as f:
        json.dump(annotations["train"], f)
    with open(os.path.join(DATASETROOT, "anno_test.json"), "w") as f:
        json.dump(annotations["test"], f)
    with open(os.path.join(DATASETROOT, "anno_all.json"), "w") as f:
        json.dump(annotations["all"], f)


if __name__ == "__main__":
    main()
