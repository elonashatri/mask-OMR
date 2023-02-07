from xml.dom import minidom
import glob
import os
import json
from tqdm import tqdm
# Get files in directory
PATH = '/data/scratch/acw507/DoReMi_v1/Parsed_XML/'
open_files = PATH+'*.xml'

for xmlfiles in tqdm(glob.glob(open_files)):
    # print('xmlfiles: ', xmlfiles)
    filename = os.path.basename(xmlfiles)
    # Remove .xml from end of file
    filename = filename[:-4]
    # Parse XML Document
    xmldoc = minidom.parse(xmlfiles)
    nodes = xmldoc.getElementsByTagName('Node')

    existing_kStaffLine_top = []
    for node in nodes:  # O(n)
        node_classname = node.getElementsByTagName('ClassName')[0]
        node_classname_str = node_classname.firstChild.data
        if node_classname_str == 'kStaffLine':
            node_top = node.getElementsByTagName('Top')[0]
            node_top_str = node_top.firstChild.data
            node_top_int = int(node_top_str)

            if node_top_int not in existing_kStaffLine_top:
                existing_kStaffLine_top.append(node_top_int)
                new_len = len(existing_kStaffLine_top)
                #print("New staff!, now we have: ", new_len)
            else:
                # This means this top already exists in this file, so it's repeated and should be deleted
                #print('Node Type: ', type(node))
                node_parent = node.parentNode
                #print('Node parent Type: ', type(node_parent))
                node_parent.removeChild(node)
                # # nodes.removeChild(node_parent)

    with open(PATH+filename+'.xml', 'w') as xml_file:
        xml_file.write(xmldoc.toxml())