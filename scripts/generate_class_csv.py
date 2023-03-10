import xml.dom
from xml.dom import minidom
import glob
import os
# Get files in directory
open_files = '/data/scratch/acw507/DoReMi_v1/Parsed_by_page_omr_xml/*.xml'

#xmldoc = minidom.parse('SEP02_300DPI_CLOCKS_MUSCIMA.xml')
print('Beginning...')
# Classnames for ALL pages in ALL files
all_classnames = set()
for xmlfiles in glob.glob(open_files):
    # print('xmlfiles: ', xmlfiles)
    filename = os.path.basename(xmlfiles)
    # Remove .xml from end of file
    filename = filename[:-4]
    # print('Parsing file: ', filename)
    # Parse XML Document
    xmldoc = minidom.parse(xmlfiles)
    root = xmldoc.getElementsByTagName('Pages')
    pages = xmldoc.getElementsByTagName('Page')
    # print('Pages length: ', len(pages))

    # Classnames for ALL pages in each file
    pages_classnames = set()
    for page in pages:
        nodes = page.getElementsByTagName('Node')
        # Add 1 here to match format exported by Dorico
        page_number = int(page.attributes['pageIndex'].value) + 1 
        # print('Page_number = ', page_number)
        # print('Nodes len ', len(nodes))
    
        for node in nodes:
            #node_data = node.getElementsByTagName('Data')[0]
            #node.removeChild(node_data)
            node_classname = node.getElementsByTagName('ClassName')[0]
            node_classname_str = node_classname.firstChild.data
            
            node_mask = node.getElementsByTagName('Mask')[0]
            node_mask_str = node_mask.firstChild.data
            # node_mask_str.replace(': ', ':')
            node_mask.firstChild.data = node_mask_str.replace(': ', ':')
            # Classnames for each file
            pages_classnames.add(node_classname_str)
            # Classnames for all files
            all_classnames.add(node_classname_str)
        # page_str = page.toxml()

    
    allfiles_stats_file = open('/data/home/acw507/mask-OMR/data/classnames.csv', 'w+')
    for classname in all_classnames:
        allfiles_stats_file.write(classname + '\n')
    allfiles_stats_file.close()