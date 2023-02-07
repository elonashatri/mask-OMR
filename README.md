This repository includes work on Instance Segmentation in OMR, namely utilising Mask R-CNNs. 

We mainly use our own dataset DoReMi to train and test but also make use of MUSCIMA++ dataset and COCO and Imagenet weights. 

Given the structure of DoReMi we need to do some pre-processing on the datasetructure, but also filtering based on the task. 

Here is a guide on the tasks that are done to bring DoReMi to a TF record fit for Mask R-CNN.

## Create classnames CSV
    python mask-OMR/scripts/generate_class_csv.py

## There is some error when data is generated and stafflines are double generated, which is why we need to clean the doubles using: 
    python /data/home/acw507/mask-OMR/scripts/clean_double_stafflines.py

    

