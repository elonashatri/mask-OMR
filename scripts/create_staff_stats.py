import xml.dom
from xml.dom import minidom
import glob
import os
import json
# Get files in directory
open_files = '/homes/es314/DOREMI_version_2/DOREMI_v3/parsed/*.xml'

print('Beginning...')
# Classnames for ALL pages in ALL files
files_staffs_count = {}
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
    
        for node in nodes:
            node_classname = node.getElementsByTagName('ClassName')[0]
            node_classname_str = node_classname.firstChild.data

            if node_classname_str == "kStaffLine":
                if filename not in files_staffs_count.keys():
                    files_staffs_count[filename] = 1
                else:
                    files_staffs_count[filename] =  files_staffs_count[filename] + 1
                
sorted_files_staffs_count = {k: v for k, v in sorted(files_staffs_count.items(), key=lambda item: item[1])}

with open('/homes/es314/DOREMI_version_2/data_v5/staff_stats.json', 'w') as outfile:
    json.dump(sorted_files_staffs_count, outfile)