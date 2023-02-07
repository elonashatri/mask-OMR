"""  
Mask R-CNN
Train on the DOREMI Dataset

------------------------------------------------------------

Usage:

    # Train a new model starting from pre-trained COCO weights
    python3 doremi_mrcnn.py train --weights=coco

    Outputting to log file
    python3 doremi_mrcnn.py train --weights=coco &> log
    
    # Resume training a model that you had trained earlier
    python3 doremi_mrcnn.py train --weights=last

    # Train a new model starting from ImageNet weights
    python3 doremi_mrcnn.py train --weights=imagenet

    # Apply color splash to an image
    python3 doremi_mrcnn.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    python3 doremi_mrcnn.py splash --weights=/homes/es314/DOREMI_version_2/MRCNN_DOREMI_20210503/logs/doremi20210504T0325/mask_rcnn_doremi_0029.h5 --image=/homes/es314/DOREMI_version_2/MRCNN_DOREMI_20210503/test-img/accidental-tucking-002.png

    # Evaluate model
    python3 doremi_mrcnn.py evaluate --weights=/homes/es314/DOREMI_version_2/MRCNN_DOREMI_20210503/logs/doremi20210504T0325/mask_rcnn_doremi_0029.h5 --logs=/homes/es314/DOREMI_version_2/MRCNN_DOREMI_20210503/logs/doremi20210504T0325/

"""

import os
import sys
import datetime
import numpy as np
import glob
from tqdm import tqdm
import json
from xml.dom import minidom
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import skimage.draw
from keras import backend as K

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to classnames file
CLASSNAMES_PATH = "/import/c4dm-05/elona/doremi_v5_half/train_validation_test_records/mapping.json"

# Path to XML Train files
XML_DATA_PATH = '/homes/es314/DOREMI_version_2/MRCNN_DOREMI_20210503/1685_images_set/'
# XML_TRAIN_PATH = '/homes/es314/DOREMI_version_2/MRCNN_DOREMI_20210503/dataset/train/'
# Path to XML Val files
# XML_VAL_PATH = '/homes/es314/DOREMI_version_2/MRCNN_DOREMI_20210503/dataset/val/'

# Path to Images 
IMG_PATH = '/homes/es314/DOREMI_version_2/DOREMI_v3/images/'
# Path to trained weights file
# COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_coco.h5")
COCO_WEIGHTS_PATH = '/import/c4dm-05/elona/maskrcnn-logs/1685_dataset_resnet101_coco_april_exp/doremi20220414T1249/mask_rcnn_doremi_0029.h5'
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
# DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
# DEFAULT_LOGS_DIR = '/import/c4dm-05/elona/maskrcnn-logs/logs_all_layers_101/'
DEFAULT_LOGS_DIR = '/import/c4dm-05/elona/maskrcnn-logs/1685_dataset_resnet101_coco_april_exp/'

############################################################
#  Configurations
############################################################


class DoremiConfig(Config):
    """
    Configuration for training on the Doremi dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "doremi"

    # We use a GPU with ??GB memory, which can fit ??? images. (12gb can fit 2 images)
    # Adjust down if you use a smaller/bigger GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 71  # Background + 71 classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.6

    # Our image size 
    # IMAGE_RESIZE_MODE = "none"
    IMAGE_MAX_DIM = 1024
    BACKBONE = "resnet101"

    LEARNING_RATE = 0.0003

############################################################
#  Dataset
############################################################

class DoremiDataset(utils.Dataset):
    def load_doremi(self, subset):
        """
        Load a subset of the Doremi dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        # self.add_class("doremi", class_id, "classname")
        with open(CLASSNAMES_PATH) as json_file:
            data = json.load(json_file)
            for id_class in data:
                self.add_class("doremi", id_class["id"], id_class["name"])

        # Train or validation dataset?
        assert subset in ["train", "val"]
        # dataset_dir = os.path.join(dataset_dir, subset)
        dataset_dir = XML_DATA_PATH+subset+'/*.xml'
        available_xml = glob.glob(dataset_dir)
        # xml_count = len(available_xml)

        # Go through all XML Files
        # Each XML File is 1 Page, each page corresponds to 1 image
        for xml_file in tqdm(available_xml, desc="XML Files"):
            filename = os.path.basename(xml_file)
            # Remove .xml from end of file
            filename = filename[:-4]

            # Parse XML Document
            xmldoc = minidom.parse(xml_file)

            # Get image name from XML file name
            page = xmldoc.getElementsByTagName('Page')
            page_index_str = page[0].attributes['pageIndex'].value

            page_index_int = int(page_index_str) + 1
            # Open image related to XML file
            # /homes/es314/DOREMI_version_2/data_v5/parsed_by_classnames/Parsed_accidental tucking-layout-0-muscima_Page_2.xml
            # Parsed_accidental tucking-layout-0-muscima_Page_2.xml
            # Remove '-layout-0-muscima_Page_' (23 chars) + len of page_index_str

            # Image name
            # /homes/es314/DOREMI_version_2/DOREMI_v3/images/accidental tucking-002.png
            # accidental tucking-002.png

            ending = 23 + len(str(page_index_int))

            start_str = 'Parsed_'
            # If page is 0, we need to add '000'
            leading_zeroes = str(page_index_int).zfill(3)
            img_filename = filename[len(start_str):-ending]+'-'+leading_zeroes
            img_filename = img_filename+'.png'
            # /homes/es314/DOREMI_version_2/DOREMI_v3/images/beam groups 12 demisemiquavers simple-918.png'

            img_path = IMG_PATH + img_filename
            # Hardcoded because our images have the same shape
            img_height = 3504
            img_width = 2474

            mask_arr = []

            nodes = xmldoc.getElementsByTagName('Node')
            # print('nodes len: ', len(nodes))

            instances_count = len(xmldoc.getElementsByTagName('ClassName'))

            # Array containing mask info object that we will use in load_mask
            masks_info = []
            # {
            #     "bbox_top": int
            #     "bbox_left": int
            #     "bbox_width": int
            #     "bbox_height": int
            #     "mask_arr": [int]
            #     "classname": str
            # }
            for node in nodes:
                this_mask_info = {}
                # Classname
                node_classname_el = node.getElementsByTagName('ClassName')[0]
                node_classname = node_classname_el.firstChild.data
                # Top
                node_top = node.getElementsByTagName('Top')[0]
                node_top_int = int(node_top.firstChild.data)
                # Left
                node_left = node.getElementsByTagName('Left')[0]
                node_left_int = int(node_left.firstChild.data)
                # Width
                node_width = node.getElementsByTagName('Width')[0]
                node_width_int = int(node_width.firstChild.data)
                # Height
                node_height = node.getElementsByTagName('Height')[0]
                node_height_int = int(node_height.firstChild.data)

                node_mask = str(node.getElementsByTagName('Mask')[0].firstChild.data)
                # 0: 2 1: 7 0: 9 1: 3
                node_mask = node_mask.replace('0:', '')
                # 2 1: 7 9 1: 3
                node_mask = node_mask.replace('1:', '')
                # 2 7 9 3
                split_mask = node_mask.split(' ')
                # [2, 7, 9, 3]
                split_mask = split_mask[:-1]
                
                notehead_counts = list(map(int, list(split_mask)))

                this_mask_info["classname"] = node_classname
                this_mask_info["bbox_top"] = node_top_int
                this_mask_info["bbox_left"] = node_left_int
                this_mask_info["bbox_width"] = node_width_int
                this_mask_info["bbox_height"] = node_height_int
                this_mask_info["mask_arr"] = notehead_counts

                masks_info.append(this_mask_info)


            # 3 required attributes, rest is kwargs
            # image_info = {
            #     "id": image_id,
            #     "source": source,
            #     "path": path,
            # }
            self.add_image(
                    "doremi",
                    image_id=img_filename,  # use file name as a unique image id
                    path=img_path,
                    img_width=img_width, img_height=img_height,
                    masks_info=masks_info)

    def load_mask(self, image_id):
        """
        Generate instance masks for an image.
        Should returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        info = self.image_info[image_id]
        if info["source"] != "doremi":
            return super(self.__class__, self).load_mask(image_id)
        
        img_height = info["img_height"]
        img_width = info["img_width"]
        mask = np.zeros([img_height, img_width, len(info["masks_info"])],dtype=np.uint8)
        instances_classes = []
        ids_classnames = {}
        with open(CLASSNAMES_PATH) as json_file:
            data = json.load(json_file)
            for id_class in data:
                ids_classnames[id_class["name"]] = id_class["id"]
        for it, info in enumerate(info["masks_info"]):
            class_id = ids_classnames[info["classname"]]
            instances_classes.append(class_id)

            notehead_counts = info["mask_arr"]
            node_top_int = info["bbox_top"]
            node_left_int = info["bbox_left"]
            node_width_int = info["bbox_width"]
            node_height_int = info["bbox_height"]
            # Counts start with Zero
            zero = True
            i = node_top_int
            j = node_left_int

            for count in notehead_counts:
                # If first 0 count is zero, ignore and go to 1
                if count != 0:
                    for _ in range(count):
                        if not zero:
                            mask[i, j, it] = 1

                        j = j + 1
                        if j == img_width or j == node_left_int+node_width_int:
                            j = node_left_int
                            i = i + 1
                zero = not zero
        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), np.array(instances_classes, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "doremi":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

############################################################
#  Training
############################################################


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = DoremiDataset()
    # dataset_train.load_doremi(args. dataset, "train")
    dataset_train.load_doremi("train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DoremiDataset()
    # dataset_val.load_doremi(args. dataset, "val")
    dataset_val.load_doremi("val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    
    # Choose which layers, heads or all 
    # chosen_layers = 'heads'
    chosen_layers = 'all'
    print("Training network ~ Chosen Layers : ", chosen_layers)
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=80,
                layers=chosen_layers)

############################################################
#  Evaluation
############################################################
def evaluate(model, inference_config):

    # Validation dataset
    dataset_val = DoremiDataset()
    # dataset_val.load_doremi(args. dataset, "val")
    dataset_val.load_doremi("val")
    dataset_val.prepare()

    print("Evaluating")

    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = np.random.choice(dataset_val.image_ids, 50)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
                                image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                            r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)
        
    print("mAP: ", np.mean(APs))



############################################################
#  Splash
############################################################


def color_splash(image, mask):
    """
    Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]
    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        print('Post skimage imread')
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        print('Post model detect')
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(
            datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    print("Saved to ", file_name)
############################################################
#  Main
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN with DOREMI.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train', 'evaluate' or 'splash'")
    # parser.add_argument('--dataset', required=False,
    #                     metavar="/path/to/doremi/dataset/",
    #                     help='Directory of the Doremi dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    # if args.command == "train":
    #     assert args. dataset, "Argument --dataset is required for training"
    
    if args.command == "evaluate":
        assert args.weights, "Provide --weights to evaluate"
    elif args.command == "splash":
        assert args.image, "Provide --image to apply color splash"

    # print("Dataset: ", args. dataset)``
    print("Weights: ", args.weights)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = DoremiConfig()
    else:
        # For evaluating or for inference
        class InferenceConfig(DoremiConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(
            mode="training", config=config, model_dir=args.logs)
    else:
        # For evaluating or for inference
        model = modellib.MaskRCNN(
            mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "evaluate":
        evaluate(model, config)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image)
    else:
        print("'{}' is not recognized. Use 'train' or 'splash'".format(args.command))



# get rid of this error https://github.com/tensorflow/tensorflow/issues/3388       
K.clear_session()