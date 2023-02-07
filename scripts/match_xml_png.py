# Compare XML List with Image List
# Check which XML need Images
import os
import glob
from tqdm import tqdm
from xml.dom import minidom
import shutil
import os

# ALWAYS PUT / IN THE END
XML_DIR = '/homes/es314/DOREMI_version_2/data_v5/parsed_by_classnames/'
XML_PATH = XML_DIR + '*.xml'
# ALWAYS PUT / IN THE END
IMG_DIR = '/homes/es314/DOREMI_version_2/DOREMI_v3/images/'
IMG_PATH = IMG_DIR + '*.png'

NEW_IMG_DIR = '/homes/es314/DOREMI_version_2/data_v5/images/'


def main():
    # Opening directly from DBF
    available_xml = glob.glob(XML_PATH)
    available_images = glob.glob(IMG_PATH)

    print('Available XMLs: ', len(available_xml))  # 3541
    print('Available Images: ', len(available_images))  # 3504 = 37 difference

    len_img_dirname = len(IMG_DIR)

    def shorten_img(img_name):
        # Remove path from beginning and extension .png from end
        return img_name[len_img_dirname:-4]

    available_images_names = list(map(shorten_img, available_images))

    xml_without_image_count = 0
    xml_without_image = []
    len_xml_dirname = len(XML_DIR)

    for xml_file in tqdm(available_xml, desc='Checking xml list with images'):

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

        if img_filename not in available_images_names:
            xml_without_image_count += 1

            # xml_without_image.append(xml_file[len_xml_dirname:-4])
            xml_without_image.append(img_filename+'.png')
        else:
            # If image is available, copy to new folder
            print('filename: ', filename)
            shutil.copy(IMG_DIR+img_filename+'.png',
                        NEW_IMG_DIR+img_filename+'.png')

    print('xml_without_image_count: ', xml_without_image_count)
    xml_without_image.sort()

    # Optional if we need to save stats about missing images
    # file = open(STATS_PATH+'xml_without_image.txt', 'w')  # write to file
    # file.write('Missing Images:\n')
    # for missing_filename in xml_without_image:
    # file.write(missing_filename)
    # file.write('\n')
    # file.close()

if __name__ == '__main__':
    main()