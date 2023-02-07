import xml.dom
from xml.dom import minidom
import glob
import os
import json
# Get files in directory
open_files = '/homes/es314/DOREMI_version_2/DOREMI_v3/parsed/*.xml'
PATH = '/homes/es314/DOREMI_version_2/data_v5/parsed_by_staff/'
print('Beginning...')
# Classnames for ALL pages in ALL files
files_staffs_count = {}
for xmlfiles in glob.glob(open_files):
    filename = os.path.basename(xmlfiles)
    filename = filename[:-4]
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
                        
    if files_staffs_count[filename] < 30:      
        with open(PATH+filename+'.xml', 'w') as xml_file:
            xml_file.write(xmldoc.toxml())

    
print("finished")
