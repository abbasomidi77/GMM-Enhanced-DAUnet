import os
import nibabel as nib
from matplotlib import pyplot as plt

example_filename = os.path.join("data", '011.nii.gz')
data = nib.load(example_filename)

plt.imshow(data.get_fdata()[:,:,128])
plt.show()

example_filename = os.path.join("result", 'image_0.nii.gz')
data = nib.load(example_filename)
plt.imshow(data.get_fdata()[:,:,128])
plt.show()