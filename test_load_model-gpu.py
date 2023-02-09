#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
from pathlib import Path


# In[3]:


import tensorflow as tf


# In[4]:


tf.__version__


# In[5]:


tf.test.is_built_with_cuda()


# In[6]:


tf.test.is_gpu_available()


# In[7]:


tf.test.gpu_device_name()


# In[ ]:





# In[8]:


from mrcnn.model import MaskRCNN
from mrcnn.config import Config


# In[9]:


class DoremiConfig(Config):
    """
    Configuration for training on the Doremi dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "doremi"

    # We use a GPU with ??GB memory, which can fit ??? images. (12gb can fit 2 images)
    # Adjust down if you use a smaller/bigger GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 71  # Background + 71 classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.6

    # Our image size 
    # IMAGE_RESIZE_MODE = "none"
    IMAGE_MAX_DIM = 1024
    BACKBONE = "resnet101"

    LEARNING_RATE = 0.0003
    
class InferenceConfig(DoremiConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# In[10]:


mode = 'inference'
config = InferenceConfig()
model_dir = Path('logs')


# In[11]:


model = MaskRCNN(mode, config, model_dir)


# In[18]:


weights_path = Path('/data/home/acw507/mask-OMR/logs/pre-trained/mask_rcnn_doremi_0018.h5')


# In[19]:


model.load_weights(weights_path, by_name=True)

