from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from xml.dom import minidom
import glob
import os


# /data/scratch/acw507/DoReMi_v1/Parsed_by_page_omr_xml/Parsed_Satie - Gnossienne 1-layout-0-muscima_Page_13.xml
# Get files in directory
open_xml_files = '/data/scratch/acw507/DoReMi_v1/Parsed_by_page_omr_xml/Parsed_accidental tucking-layout-0-muscima_Page_17.xml'

# source_img = '/Users/elona/Downloads/OMR_dataset/150dpi/images_dir_150/*.png'
imgs_path = '/data/scratch/acw507/DoReMi_v1/Images/' #imagename.png

print('Beginning...')
# Classnames for ALL pages in ALL files
all_classnames = set()
for xmlfile in glob.glob(open_xml_files):
    filename = os.path.basename(xmlfile)
    # Remove .xml from end of file
    filename = filename[:-4]
    print('Parsing file: ', filename)
    
    # Parse XML Document
    xmldoc = minidom.parse(xmlfile)
    page = xmldoc.getElementsByTagName('Page')
    page_index_str = page[0].attributes['pageIndex'].value
    print('page_index_str = ', page_index_str)
    # Here we add 1 because dorico XML starts pageIndex at 0, but when exporting to image it starts with 1
    # EXCEPTION: Au Tombeau de Rachmanimoff exports a page 0 which is EMPTY and needs to be discarded
    page_index_int = int(page_index_str) + 1
    # Open image related to XML file
    # Filename example: "Parsed_Winstead - Cygnus, The Swan-layout-0-muscima_Page_3.xml"
    # 7: because we remove the "Parsed_" in the beginning of the filename
    # Also remove "layout-0-muscima_Page_" (22 chars) + len of page_index_str
    ending = 22 + len(str(page_index_int))
    # If page is 0, we need to add '000'
    leading_zeroes = str(page_index_int).zfill(3)
    img_filename = filename[7:-ending]+leading_zeroes
    img_path = imgs_path+img_filename+'.png'
    img = Image.open(img_path)
    img = img.convert('RGB')
    #img = img.convert('P', palette=Image.ADAPTIVE, colors=256)
    draw = ImageDraw.Draw(img)
        
    nodes = xmldoc.getElementsByTagName('Node')
    for node in nodes:
        # Classname
        node_classname = node.getElementsByTagName("ClassName")[0]
        node_classname_str = node_classname.firstChild.data 

        # Top we subtract 4 pixels so the bounding box goes up  by 4 pixels to include possible missalignment in staff lines
        node_top = node.getElementsByTagName('Top')[0]
        node_top_int = int(node_top.firstChild.data) - 2
        # Left
        node_left = node.getElementsByTagName('Left')[0]
        node_left_int = int(node_left.firstChild.data) - 1 
        # Width
        node_width = node.getElementsByTagName('Width')[0]
        node_width_int = int(node_width.firstChild.data) + 2
        # Height, we add 4 pixels to tolarate for staff lines that fall out of the bounding boxes
        node_height = node.getElementsByTagName('Height')[0]
        node_height_int = int(node_height.firstChild.data) + 2

        if node_classname_str == 'kStaffLine':
            # Top:
            node_top = node.getElementsByTagName("Top")[0]
            node_top_int = int(node_top.firstChild.data) - 5
            # Height 
            node_height = node.getElementsByTagName("Height")[0]
            node_height_int = int(node_height.firstChild.data) + 5

        # pillow [(x0, y0), (x1, y1)]
        x0 = node_left_int
        y0 = node_top_int
        x1 = node_left_int + node_width_int
        y1 = node_top_int + node_height_int
        
        rect = draw.rectangle(((x0, y0), (x1, y1)), fill=None, outline="green", width=1)
    img.save('/data/home/acw507/mask-OMR/data/test_rectangles/Rects_' + img_filename +'.png')
    
# source_img = Image.open('/Users/elona/Downloads/OMR_dataset/150dpi/images_dir_150/Akinola - Dorico Prelude-001.png').convert("RGBA")
# draw = ImageDraw.Draw(source_img)
# draw.rectangle(((0, 00), (100, 100)), fill="black")
# source_img.save(out_file, "JPEG")

