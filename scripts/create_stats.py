import xml.dom
from xml.dom import minidom
import glob
import os
import json
# Get files in directory
open_files = '/homes/es314/DOREMI_version_2/DOREMI_v3/parsed_original_double_staffs/*.xml'

print('Beginning...')
# Classnames for ALL pages in ALL files
pages_classnames_count = {}
for xmlfiles in glob.glob(open_files):
    # print('xmlfiles: ', xmlfiles)
    filename = os.path.basename(xmlfiles)
    # Remove .xml from end of file
    filename = filename[:-4]
    #print('Parsing file: ', filename)
    # Parse XML Document
    xmldoc = minidom.parse(xmlfiles)
    root = xmldoc.getElementsByTagName('Pages')
    pages = xmldoc.getElementsByTagName('Page')

    # Classnames for ALL pages in each file
    for page in pages:
        nodes = page.getElementsByTagName('Node')
        #print('Nodes len ', len(nodes))
    
        for node in nodes:
            node_classname = node.getElementsByTagName('ClassName')[0]
            node_classname_str = node_classname.firstChild.data
            if node_classname_str not in pages_classnames_count.keys():
                pages_classnames_count[node_classname_str] = 1
            else:
                pages_classnames_count[node_classname_str] =  pages_classnames_count[node_classname_str] + 1
                
sorted_pages_classnames_count = {k: v for k, v in sorted(pages_classnames_count.items(), key=lambda item: item[1])}

with open('/homes/es314/DOREMI_version_2/data_v3/classnames_90_xclasses.json', 'w') as outfile:
    json.dump(sorted_pages_classnames_count, outfile)