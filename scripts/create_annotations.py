import os
import glob
from tqdm import tqdm
import json
from xml.dom import minidom
from PIL import Image
import hashlib
import tensorflow as tf

#this file was last used 5th march 2021, it has the new annotations, lightweight same witht eh ones that worked with the first version of DOREMI


##### FINISH IMPORTS ######
ANNOTATIONS_PATH = "/data/scratch/acw507/DoReMi_v1/Parsed_by_page_omr_xml/*.xml"
IMGS_PATH = "/data/scratch/acw507/DoReMi_v1/Images/"
TF_RECORDS_PATH = "/data/home/acw507/mask-OMR/data/tfrecords/"
# Constants:
CLASSNAMES_PATH = "/data/home/acw507/mask-OMR/data/tfrecords/mapping.json"
TRAIN_SPLIT = 0.8
TEST_SPLIT = 0.9


# From tensorflow.org:
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))



def get_img_annotations(annotations_path, imgs_path):
    print("Getting images and their annotations...")
    imgs_annotations_train = {}
    imgs_annotations_test = {}
    imgs_annotations_validate = {}

    xml_files = glob.glob(annotations_path)
    total_count = len(xml_files)
    current_iteration = 0
    
    ids_classnames = {}
    with open(CLASSNAMES_PATH) as json_file:
        data = json.load(json_file)
        for id_class in data:
            ids_classnames[id_class["name"]] = id_class["id"]

    for xml_file in tqdm(xml_files, desc="XML Files"):
        filename = os.path.basename(xml_file)
        # Remove .xml from end of file
        filename = filename[:-4]

        # Parse XML Document
        xmldoc = minidom.parse(xml_file)

        # Get image name from XML file name
        page = xmldoc.getElementsByTagName("Page")
        page_index_str = page[0].attributes["pageIndex"].value
        page_index_int = int(page_index_str) + 1
        ending = 22 + len(str(page_index_int))
        leading_zeroes = str(page_index_int).zfill(3)
        img_filename = filename[7:-ending]+leading_zeroes
        img_path = imgs_path + "/" + img_filename + ".png"
        img = Image.open(img_path)
        img_width = img.size[0]
        img_height= img.size[1]
        
        nodes = xmldoc.getElementsByTagName("Node")

        y_mins = []
        x_mins = []
        x_maxs = []
        y_maxs = []
        classes = []
        classes_text = []
        truncated = []
        poses = []
        difficult_obj = []
        
        for node in nodes:
            # Classname
            node_classname = node.getElementsByTagName("ClassName")[0]
            node_classname_str = node_classname.firstChild.data 
            # Top we subtract 2 pixels so the bounding box goes up  by 2 pixels to include possible missalignment of objects
            node_top = node.getElementsByTagName('Top')[0]
            node_top_int = int(node_top.firstChild.data) - 2
            # Left
            node_left = node.getElementsByTagName('Left')[0]
            node_left_int = int(node_left.firstChild.data) - 1 
            # Width
            node_width = node.getElementsByTagName('Width')[0]
            node_width_int = int(node_width.firstChild.data) + 2
            # Height, we add 2 pixels to tolarate for objects that fall out of the bounding boxes
            node_height = node.getElementsByTagName('Height')[0]
            node_height_int = int(node_height.firstChild.data) + 2

            if node_classname_str == 'kStaffLine':
                # Top:
                node_top = node.getElementsByTagName("Top")[0]
                node_top_int = int(node_top.firstChild.data) - 5
                # Height 
                node_height = node.getElementsByTagName("Height")[0]
                node_height_int = int(node_height.firstChild.data) + 5


            if node_width_int == 0:
                node_width_int = 2
                node_left_int -= 1
            if node_height_int == 0:
                node_height_int = 2
                node_top_int -= 1
                
            x_min = node_left_int / img_width
            x_max = (node_left_int + node_width_int) / img_width
            y_min = node_top_int / img_height
            y_max = (node_top_int + node_height_int) / img_height
            
            y_mins.append(y_min)
            x_mins.append(x_min)
            x_maxs.append(x_max)
            y_maxs.append(y_max)

            classes_text.append(node_classname_str.encode("utf8"))
            classes.append(ids_classnames[node_classname_str])
            truncated.append(0)
            difficult_obj.append(0)
            poses.append("Unspecified".encode("utf8"))

        annotations = {}
        annotations["img_filename"] = img_filename + ".png"
        annotations["img_height"] = img_height
        annotations["img_width"] = img_width
        
        annotations["y_mins"] = y_mins
        annotations["x_mins"] = x_mins
        annotations["x_maxs"] = x_maxs
        annotations["y_maxs"] = y_maxs

        annotations["classes"] = classes
        annotations["classes_text"] = classes_text

        annotations["poses"] = poses
        annotations["truncated"] = truncated
        annotations["difficult_obj"] = difficult_obj
        
        if current_iteration < int(total_count*TRAIN_SPLIT):
            imgs_annotations_train[img_path] = annotations
        elif current_iteration < int(total_count*TEST_SPLIT):
            imgs_annotations_test[img_path] = annotations
        else:
            imgs_annotations_validate[img_path] = annotations
        current_iteration += 1  
        
    return (imgs_annotations_train,imgs_annotations_test,imgs_annotations_validate)

def transf_img_annotations_into_tfrecords(imgs_annotations: {}):
    print("Transforming annotations into TF Records...")
    tf_examples = []

    for img_path, annotations in tqdm(imgs_annotations.items()):
        image_string = open(img_path, 'rb').read()
        key = hashlib.sha256(image_string).hexdigest()
    
        img_features = {
                "image/height": _int64_feature(annotations["img_height"]),
                "image/width": _int64_feature(annotations["img_width"]),
                "image/filename": _bytes_feature(annotations["img_filename"].encode("utf8")),
                "image/source_id": _bytes_feature(annotations["img_filename"].encode("utf8")),
                "image/key/sha256": _bytes_feature(key.encode("utf8")),
                "image/encoded": _bytes_feature(image_string),
                "image/format": _bytes_feature("png".encode("utf8")),
                "image/object/bbox/xmin": _float_list_feature(annotations["x_mins"]),
                "image/object/bbox/xmax": _float_list_feature(annotations["x_maxs"]), 
                "image/object/bbox/ymin": _float_list_feature(annotations["y_mins"]),
                "image/object/bbox/ymax": _float_list_feature(annotations["y_maxs"]),
                "image/object/class/text": _bytes_list_feature(annotations["classes_text"]),
                "image/object/class/label": _int64_list_feature(annotations["classes"]),
                "image/object/difficult": _int64_list_feature(annotations["difficult_obj"]),
                "image/object/truncated": _int64_list_feature(annotations["truncated"]),
                "image/object/view": _bytes_list_feature(annotations["poses"]),  
        }
            
        tf_example = tf.train.Example(features=tf.train.Features(feature=img_features))
        tf_examples.append(tf_example)
        
    return tf_examples

def main():
    print("Start preparing tf records py")
    imgs_annotations_train, imgs_annotations_test, imgs_annotations_validate = get_img_annotations(ANNOTATIONS_PATH,IMGS_PATH)
    
    # Prepare train TF Records
    train_tf_examples = transf_img_annotations_into_tfrecords(imgs_annotations_train)
    
    writer = tf.python_io.TFRecordWriter(TF_RECORDS_PATH+"train.tfrecords")
    for tf_example in tqdm(train_tf_examples,
                           desc="Serializing train annotations"):
        writer.write(tf_example.SerializeToString())

    writer.close()

    # Prepare test TF Records
    test_tf_examples = transf_img_annotations_into_tfrecords(imgs_annotations_test)
    writer = tf.python_io.TFRecordWriter(TF_RECORDS_PATH+"test.tfrecords")
    for tf_example in tqdm(test_tf_examples,
                           desc="Serializing test annotations"):
        writer.write(tf_example.SerializeToString())

    writer.close()

    # Prepare validate TF Records
    validate_tf_examples = transf_img_annotations_into_tfrecords(imgs_annotations_validate)
    writer = tf.python_io.TFRecordWriter(TF_RECORDS_PATH+"validate.tfrecords")
    for tf_example in tqdm(validate_tf_examples,
                           desc="Serializing validation annotations"):
        writer.write(tf_example.SerializeToString())

    # writer.close()

if __name__ == "__main__":
    main()