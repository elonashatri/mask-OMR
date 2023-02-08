# [new way of trying] pre-prepare data (only 50%)
import glob
import random
import json
from tqdm import tqdm
import shutil

# ALWAYS PUT / IN THE END
XML_DIR = '//data/home/acw507/mask-OMR/data/train/'
XML_PATH = XML_DIR + '*.xml'

# We have 2918 files
# Try only with a few files initially
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.90
TEST_SPLIT = 1.0
# TRAIN_SPLIT = 0.75  # From 0 to 0.75
# VAL_SPLIT = 1.0 # From 0.75 to 1


OUTPUT_XML_DIR = '/data/home/acw507/mask-OMR/data/'


def main():
    # Glob.glob lists all file
    available_xml = glob.glob(XML_PATH)
    xml_count = len(available_xml)
    print('Available XMLs: ', xml_count)
    len_origin_dir = len(XML_DIR)
    copy_count = 0
    for xml_file in tqdm(available_xml, desc='Copying from xml list'):
        # print(xml_file)
        if copy_count < int(xml_count*TRAIN_SPLIT):
            dst = OUTPUT_XML_DIR + 'train/' + xml_file[len_origin_dir:]
            shutil.copyfile(xml_file, dst)
        elif copy_count < int(xml_count*VAL_SPLIT):
            dst = OUTPUT_XML_DIR + 'val/' + xml_file[len_origin_dir:]
            shutil.copyfile(xml_file, dst)
        elif copy_count < int(xml_count*TEST_SPLIT):
            dst = OUTPUT_XML_DIR + 'test/' + xml_file[len_origin_dir:]
            shutil.copyfile(xml_file, dst)
        copy_count = copy_count + 1

if __name__ == '__main__':
    main()
