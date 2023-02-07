import xml.dom
from xml.dom import minidom
import glob
import os
import json
from tqdm import tqdm
# Get files in directory
open_files = '/homes/es314/DOREMI_version_2/data_v5/parsed_by_staff/*.xml'
PATH = '/homes/es314/DOREMI_version_2/data_v5/parsed_by_classnames/'

print('Beginning...')
# Classnames for ALL pages in ALL files
pages_classnames_count = {}

list_to_exclude = ['accidentalTripleSharp',
                   'accidentalThreeQuarterTonesFlatZimmermann',
                   'accidentalTripleFlat',
                   'timeSig1',
                   'noteheadDoubleWholeSquare',
                   'dynamicFFFF',
                   'rest',
                   'rest64th',
                   'dynamicRinforzando2',
                   'accidentalKomaSharp',
                   'accidentalKomaFlat',
                   'dynamicSforzatoFF',
                   'ornamentTurn',
                   'ornamentMordent',
                   'wiggleTrill',
                   'mensuralNoteheadMinimaWhite',
                   'flag64thUp',
                   'dynamicPPPP']


for xmlfiles in tqdm(glob.glob(open_files)):
    filename = os.path.basename(xmlfiles)
    filename = filename[:-4]
    xmldoc = minidom.parse(xmlfiles)
    root = xmldoc.getElementsByTagName('Pages')
    pages = xmldoc.getElementsByTagName('Page')

    should_save = True
    for page in pages:
        nodes = page.getElementsByTagName('Node')

        for node in nodes:
            node_classname = node.getElementsByTagName('ClassName')[0]
            node_classname_str = node_classname.firstChild.data

            if node_classname_str in list_to_exclude:
                should_save = False
                break
        if should_save == False:
            break

    if should_save:
        with open(PATH+filename+'.xml', 'w') as xml_file:
            xml_file.write(xmldoc.toxml())