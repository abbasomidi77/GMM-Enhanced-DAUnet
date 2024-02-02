import monai
import sys
import os

# Get the path to the 'synthseg' folder
synthseg_path = os.path.abspath('GMM_MRI_COMPLEX')  # Replace with the actual path

# Add the 'synthseg' folder to the sys.path
sys.path.append(synthseg_path)

# Now you should be able to import modules from 'synthseg'

from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.data import Dataset, ArrayDataset, DataLoader
from monai.transforms import (LoadImaged, EnsureChannelFirstd, ScaleIntensityd, RandCropByPosNegLabeld,\
                             RandAxisFlipd, RandGaussianNoised, RandGibbsNoised, RandSpatialCropd, Compose, \
                             CropForegroundd,AdjustContrastd)
import pandas as pd
import numpy as np
from monai.data.utils import pad_list_data_collate
from multiprocessing import freeze_support
import random
import torch
from torch.utils.data import Dataset, DataLoader
from RL2I import RandomLabelsToImage
from gmm2io import Subject
import torchio as tio
from torchio import RandomElasticDeformation
from monai.transforms import (
    apply_transform,
)
class BrainGenerator():
    def __init__(self,labels_dir,generation_classes,generation_labels,output_labels,prior_means,prior_stds,discretize=False):
        self.subjects = self.get_subjects(labels_dir)
        self.prior_means = np.load(prior_means)
        self.prior_stds = np.load(prior_stds)
        self.n_classes = 25
        self.generation_classes = np.load(generation_classes)
        self.generation_labels = np.load(generation_labels)
        self.output_labels = np.load(output_labels)
        self.tmp_means = self.prior_means[0][self.generation_classes]
        self.tmp_stds = self.prior_stds[0][self.generation_classes]
        self.discretize = discretize
        
        # Deleting one of the Blur transforms
        self.Affine_transform = tio.RandomAffine(scales=(0.9, 1.2),degrees=15)
        self.Elastic_transform = RandomElasticDeformation(num_control_points=(7, 7, 7),locked_borders=2)
        self.Random_transform = tio.transforms.RandomBlur()
        self.Motion_transform = tio.transforms.RandomMotion()
        self.blurring_transform = tio.RandomBlur()
        self.transform = tio.Compose([self.Affine_transform, self.Elastic_transform, self.Random_transform, self.Motion_transform, self.blurring_transform])

    def filter_labels(self,subj, boolean_array):
        """
        Retain labels in the image that correspond to True in the boolean_array,
        and set the rest to zero.

        Args:
        image (numpy.ndarray): A 2D array representing the labeled image.
        boolean_array (numpy.ndarray): A 1D array of boolean values (True/False).

        Returns:
        numpy.ndarray: The modified image with filtered labels.
        """
        label_unique = np.unique(subj.tissues.data)
        label_size = len(boolean_array)
        # mask[mask == -1] = 0
        for label in label_unique:
            if label not in boolean_array:
                subj.tissues.data[subj.tissues.data == label] = 0


        # subj['mask'] = mask
        return subj
    def get_subjects(self, labels_dir):
        subjs = []
        for file in os.listdir(labels_dir):
            if file.endswith('.nii.gz'):
                subjs.append(Subject(os.path.join(labels_dir,file)))
        return subjs


    def generate_brain(self):
        subject_ = random.choice(self.subjects)
        # self.simulation_transform = tio.RandomLabelsToImage(
        #     label_key='tissues', used_labels=[int(item) for item in self.generation_labels[:int(subject_.tissues.data.max())+1]],
        #     mean=self.tmp_means[:int(subject_.tissues.data.max())+1], std=self.tmp_stds[:int(subject_.tissues.data.max())+1], discretize=self.discretize
        # )

        np_means = np.zeros(int(np.unique(subject_.tissues.data).max())+1)
        np_stds = np.zeros(int(np.unique(subject_.tissues.data).max())+1)
        for label in np.unique(subject_.tissues.data):
            id_ = self.generation_classes[np.argwhere(self.generation_labels==label)[0,0]]
            np_means[int(label)]  =  self.prior_means[0][id_]
            np_stds[int(label)] = self.prior_stds[0][id_]

        self.simulation_transform = RandomLabelsToImage(
            label_key='tissues',
            mean=np_means,
            std=np_stds
        )


        transformed = self.simulation_transform(subject_)
        transformed = self.filter_labels(transformed, self.output_labels)
        transformed = self.transform(transformed)

        im = transformed['image_from_labels']['data']
        label = transformed['tissues']['data']
        return im,label
# https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
class SynthSegDataset(Dataset):
    def __init__(self, config,transform):
        self.config = config
        self.brain_generator = BrainGenerator(
            labels_dir=config['path_label_map'],
            generation_classes=config['generation_classes'],
            generation_labels = config['generation_labels'],
            output_labels=config['output_labels'],
            prior_means=config['prior_means'],
            prior_stds=config['prior_stds'],
        )
        self.domain = config['domain']
        self.transform = transform




    def __len__(self):
        return self.config['n_examples']

    def __getitem__(self, idx):
        im, lab = self.brain_generator.generate_brain()
        lab = lab.to(torch.bool)
        sample = {"img": im.squeeze(0), "brain_mask": lab.squeeze(0), "domain_label": self.domain}
        return apply_transform(self.transform, sample) if self.transform is not None else sample
        # return sample
    
config_train = {
    'n_examples': 1024,
    'path_label_map': 'GMM_MRI_COMPLEX/data/training_label_maps',
    'generation_labels': 'GMM_MRI_COMPLEX/data/labels_classes_priors/generation_labels.npy',
    'output_labels': 'GMM_MRI_COMPLEX/data/labels_classes_priors/synthseg_segmentation_labels_amir.npy',
    'generation_classes': 'GMM_MRI_COMPLEX/data/labels_classes_priors/generation_classes_contrast_specific.npy',
    'prior_means':'GMM_MRI_COMPLEX/data/labels_classes_priors/prior_means_t1.npy',
    'prior_stds':'GMM_MRI_COMPLEX/data/labels_classes_priors/prior_stds_t1.npy',
    'domain':  0.0
}

config_val = {
    'n_examples': 128,
    'path_label_map': 'GMM_MRI_COMPLEX/data/training_label_maps',
    'generation_labels': 'GMM_MRI_COMPLEX/data/labels_classes_priors/generation_labels.npy',
    'output_labels': 'GMM_MRI_COMPLEX/data/labels_classes_priors/synthseg_segmentation_labels_amir.npy',
    'generation_classes': 'GMM_MRI_COMPLEX/data/labels_classes_priors/generation_classes_contrast_specific.npy',
    'prior_means':'GMM_MRI_COMPLEX/data/labels_classes_priors/prior_means_t1.npy',
    'prior_stds':'GMM_MRI_COMPLEX/data/labels_classes_priors/prior_stds_t1.npy',
    'domain':  0.0
}



source_transforms = Compose(
    [
        # LoadImaged(keys=["img", "brain_mask"]),
        EnsureChannelFirstd(keys=["img",  "brain_mask"]),
        ScaleIntensityd(
            keys=["img"],
            minv=0.0,
            maxv=1.0
        ),
        RandSpatialCropd(keys=["img","brain_mask"], roi_size=(112, 112, 112), random_size=False),
        #RandCropByPosNegLabeld(
        #    keys=["img", "brain_mask"],
        #    spatial_size=(64, 64, 64),
        #    label_key="brain_mask",
        #    pos = 0.9,
        #    neg=0.1,
        #    num_samples=1,
        #    image_key="img",
        #    image_threshold=-0.1
        #),
        #AdjustContrastd(keys=["img"], gamma=2.0),
        RandAxisFlipd(keys=["img", "brain_mask"], prob = 0.2),
        RandGaussianNoised(keys = ["img"], prob=0.2, mean=0.0, std=0.05),
        RandGibbsNoised(keys=["img"], prob = 0.2, alpha = (0.1,0.6))
    ]
)

def threshold(x):
    # threshold at 1
    return x > 0.015


target_transforms = Compose(
    [
        LoadImaged(keys=["img"]),
        EnsureChannelFirstd(keys=["img"]),
        ScaleIntensityd(keys=["img"], minv=0.0, maxv=1.0),
        CropForegroundd(keys=["img"], source_key = "img", select_fn=threshold, margin=3),
        RandSpatialCropd(keys=["img"], roi_size=(112, 112, 112), random_size=False),
        RandGaussianNoised(keys = ["img"], prob=0.2, mean=0.0, std=0.05),
        RandGibbsNoised(keys=["img"], prob = 0.2, alpha = (0.1,0.6)),
        RandAxisFlipd(keys=["img"], prob = 0.2)
    ]
)


def load_data(source_dev_images_csv, source_dev_masks_csv,
              target_dev_images_csv = None, batch_size = 1, val_split = 0.2, verbose = False):

    '''
    source_dev_images = pd.read_csv(source_dev_images_csv)
    source_dev_masks = pd.read_csv(source_dev_masks_csv)

    assert source_dev_images.size == source_dev_masks.size
    '''
    if target_dev_images_csv:
        target_dev_images = pd.read_csv(target_dev_images_csv)
    '''
    if verbose:
        print("Shape source images:", source_dev_images.shape)
        print("Shape source masks:",  source_dev_masks.shape)
        if target_dev_images_csv:
            print("Shape target images:", target_dev_images.shape)
        else:
            print("Target images CSV file path not provided")    
    
    
    indexes_source = np.arange(source_dev_images.shape[0])
    
    np.random.seed(100)  
    np.random.shuffle(indexes_source)
    
  
    source_dev_images = np.array(source_dev_images["filename"])[indexes_source]
    source_dev_masks = np.array(source_dev_masks["filename"])[indexes_source]
    
    ntrain_samples = int((1 - val_split)*indexes_source.size)
    source_train_images = source_dev_images[:ntrain_samples]
    source_train_masks = source_dev_masks[:ntrain_samples]

    source_val_images = source_dev_images[ntrain_samples:]
    source_val_masks = source_dev_masks[ntrain_samples:]

    if verbose:
        print("Source train set size:", source_train_images.size)
        print("Source val set size:", source_val_images.size)

    '''
    # Putting the filenames in the MONAI expected format - source train set

    source_ds_train = SynthSegDataset(config_train,source_transforms)

    source_ds_val = SynthSegDataset(config_val,source_transforms)

    # filenames_train_source = [{"img": x, "brain_mask": y, "domain_label": 0.0}\
    #                           for (x,y) in zip(source_train_images, source_train_masks)]
       
    # source_ds_train = monai.data.Dataset(filenames_train_source,
    #                                      source_transforms)

    source_train_loader = DataLoader(source_ds_train, 
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=0, 
                                    pin_memory=True, 
                                    collate_fn=pad_list_data_collate,
                                    drop_last=True) # add drop_last argument here


    # Putting the filenames in the MONAI expected format - source val set
    #filenames_val_source = [{"img": x, "brain_mask": y, "domain_label": 0.0}\
    #                          for (x,y) in zip(source_val_images, source_val_masks)]
       
    #source_ds_val = monai.data.Dataset(filenames_val_source,
    #                                     source_transforms)
                                         
    source_val_loader = DataLoader(source_ds_val, 
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=0, 
                                    pin_memory=True, 
                                    collate_fn=pad_list_data_collate,
                                    drop_last=True) # add drop_last argument here



    # If there is not target domain data - return the source domain train and val datasets and loaders
    if not target_dev_images_csv:
        return source_ds_train, source_train_loader, source_ds_val, source_val_loader

    
    
    indexes_target = np.arange(target_dev_images.shape[0])
    np.random.seed(100)  
    np.random.shuffle(indexes_target)

    target_dev_images = np.array(target_dev_images["filename"])[indexes_target]
    
    ntrain_samples_target = int((1 - val_split)*indexes_target.size)
    target_train_images = target_dev_images[:ntrain_samples_target]
    
    target_val_images = target_dev_images[ntrain_samples_target:]

    if verbose:
        print("Traget train set size:", target_train_images.size)
        print("Target val set size:", target_val_images.size)


    # Putting the filenames in the MONAI expected format - target train set
    filenames_train_target = [{"img": x, "domain_label": 1.0}\
                              for x in target_train_images]
       
    target_ds_train = monai.data.Dataset(filenames_train_target,
                                         target_transforms)

    target_train_loader = DataLoader(target_ds_train, 
                                    batch_size=batch_size, 
                                    shuffle = True, 
                                    num_workers=0, 
                                    pin_memory=True, 
                                    collate_fn=pad_list_data_collate,
                                    drop_last=True) # add drop_last argument here

    # Putting the filenames in the MONAI expected format - target val set
    filenames_val_target = [{"img": x, "domain_label": 1.0}\
                              for x in target_val_images]


    target_ds_val = monai.data.Dataset(filenames_val_target,
                                         target_transforms)
                                         
    target_val_loader = DataLoader(target_ds_val, 
                                   batch_size=batch_size, 
                                   shuffle = True, 
                                   num_workers=0, 
                                   pin_memory=True, 
                                   collate_fn=pad_list_data_collate,
                                   drop_last=True) # add drop_last argument here

    return source_ds_train, source_train_loader, source_ds_val, source_val_loader,\
           target_ds_train, target_train_loader, target_ds_val, target_val_loader

