import matplotlib.pyplot as plt
import numpy as np
import torchio as tio

from gmm2io import Subject
import matplotlib as pyplot
subject_ = Subject('data/training_label_maps/training_seg_01.nii.gz')
prior_means = np.load('data/labels_classes_priors/prior_means_t1.npy')
# same as for prior_means, but for the standard deviations of the GMM.
prior_stds = np.load('data/labels_classes_priors/prior_stds_t1.npy')
n_classes = 25
generation_classes = np.load('data/labels_classes_priors/generation_classes_contrast_specific.npy')
# tmp_classes_means = utils.draw_value_from_distribution(prior_means, n_classes, prior_distributions,
#                                                        125., 125., positive_only=True)
# tmp_classes_stds = utils.draw_value_from_distribution(prior_stds, n_classes, prior_distributions,
#                                                       15., 15., positive_only=True)
# random_coef = npr.uniform()
# if random_coef > 0.95:  # reset the background to 0 in 5% of cases
#     tmp_classes_means[0] = 0
#     tmp_classes_stds[0] = 0
# elif random_coef > 0.7:  # reset the background to low Gaussian in 25% of cases
#     tmp_classes_means[0] = npr.uniform(0, 15)
#     tmp_classes_stds[0] = npr.uniform(0, 5)
tmp_means = prior_means[0][generation_classes]
tmp_stds = prior_stds[0][generation_classes]
simulation_transform = tio.RandomLabelsToImage(
    label_key='tissues', mean=tmp_means, std=tmp_stds, discretize=False
)
norm_transform = tio.transforms.RescaleIntensity(out_min_max=(0, 255))
transform = tio.Compose([simulation_transform, norm_transform])
transformed = transform(subject_)

plt.imshow(transformed['image_from_labels'].data[0,:,100],cmap='gray')
plt.colorbar()
plt.show()
# subject = tio.datasets.ICBM2009CNonlinearSymmetric()
# Using the default parameters
transform = tio.RandomLabelsToImage(label_key='tissues')
# Using custom mean and std
transform = tio.RandomLabelsToImage(
    label_key='tissues', mean=prior_means.flatten(), std=prior_stds.flatten()
)
# Discretizing the partial volume maps and blurring the result
simulation_transform = tio.RandomLabelsToImage(
    label_key='tissues', mean=[0.33, 0.66, 1.], std=[0, 0, 0], discretize=True
)
blurring_transform = tio.RandomBlur(std=0.3)
transform = tio.Compose([simulation_transform, blurring_transform])
transformed = transform(subject)  # subject has a new key 'image_from_labels' with the simulated image
# Filling holes of the simulated image with the original T1 image
rescale_transform = tio.RescaleIntensity(
    out_min_max=(0, 1), percentiles=(1, 99))   # Rescale intensity before filling holes
simulation_transform = tio.RandomLabelsToImage(
    label_key='tissues',
    image_key='t1',
    used_labels=[0, 1]
)
transform = tio.Compose([rescale_transform, simulation_transform])
transformed = transform(subject)  # subject's key 't1' has been replaced with the simulated image

plt.imshow(transformed['t1'].data[0,80])
plt.show()