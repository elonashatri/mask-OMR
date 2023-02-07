This repository includes work on Instance Segmentation in OMR, namely utilising Mask R-CNNs. 

We mainly use our own dataset DoReMi to train and test but also make use of MUSCIMA++ dataset and COCO and Imagenet weights. 

Given the structure of DoReMi we need to do some pre-processing on the datasetructure, but also filtering based on the task. 

Here is a guide on the tasks that are done to bring DoReMi to a TF record fit for Mask R-CNN. 

## Prettify files and clean filenames

    python /data/home/acw507/mask-OMR/scripts/parsing_xml.py

or alternatively you can use the already parsed files in /data/scratch/acw507/DoReMi_v1/Parsed_by_page_omr_xml

## There is some error when data is generated and stafflines are double generated, which is why we need to clean the doubles using: 
    python /data/home/acw507/mask-OMR/scripts/clean_double_stafflines.py

no need to do these if using the parsed files /data/scratch/acw507/DoReMi_v1/Parsed_by_page_omr_xml 

## To make sure we have the right images and the matching XMLs we run:

    python mask-OMR/scripts/match_xml_png.py

## Create classnames CSV
    python mask-OMR/scripts/generate_class_csv.py

## Create json mapping

    python /data/home/acw507/mask-OMR/scripts/create_mappings.py

## Check the stats


