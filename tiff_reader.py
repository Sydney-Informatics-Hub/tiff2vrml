#!/usr/bin/env python
# coding: utf-8
"""
 __                 
/  \        _______________________
|  |       /                        \
@  @       | It looks like you      |
|| ||      | are you writing        |
|| ||   <--| Python code to process |
|\_/|      | a series of TIFFs and  |
|   |      | create a 3d volume     |
\___/      \________________________/
"""
# In[6]:


import numpy as np
import tifffile as tiff
import os
import scipy
import matplotlib.pyplot as plt


# In[8]:


# Read the tiff files, assume have sorteable filenames 001.tif, 002.tif etc.
tiff_dir = "data/"
tiff_files = [f for f in os.listdir(tiff_dir) if f.endswith(".tif")]
tiff_files.sort()

# Load the first TIFF file to get its dimensions
first_tiff = tiff.imread(os.path.join(tiff_dir, tiff_files[0]))
height, width = first_tiff.shape
num_slices = len(tiff_files)

# Create a numpy array to store the volume
volume = np.zeros((height, width, num_slices), dtype=np.uint8)

# Iterate through TIFF files and populate a volume
for i, tiff_file in enumerate(tiff_files):
  print(tiff_file)
  tiff_data = tiff.imread(os.path.join(tiff_dir, tiff_file))
  volume[:, :, i] = tiff_data


# In[9]:


for i in range(volume.shape[2]):
    slice_2d = volume[:, :, i]
    plt.imshow(slice_2d, cmap='viridis')
    plt.title(f"Slice {i}")
    plt.colorbar()
    plt.show()


# In[ ]:




