import argparse
import os

import pandas as pd

def generate_mapping(args):
    annotations = pd.read_csv(os.path.join(args.dataset_dir, "classnames.csv"))

    class_names = annotations["class_name"].unique()
    class_names.sort()


    with open(args.mapping_output_path, "w") as f:
        f.write('[')
        for i, class_name in enumerate(class_names):
            f.write(""" {{
        "id": {},
        "name": "{}"
    }},
""".format(i + 1, class_name))
        f.write(']')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="/data/home/acw507/mask-OMR/data/tfrecords/")
    # parser.add_argument("--mapping_output_path", default="/data/home/acw507/mask-OMR/data/mapping.json")
    parser.add_argument("--mapping_output_path", default="/data/home/acw507/mask-OMR/data/tfrecords/mapping.txt")
    args = parser.parse_args()

    generate_mapping(args)
