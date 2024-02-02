import os
import random
import matplotlib.pyplot as plt
import nibabel
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchio as tio
from torchio import RandomElasticDeformation

from RL2I import RandomLabelsToImage
from gmm2io import Subject


# https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT

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

        self.Affine_transform = tio.RandomAffine(scales=(0.9, 1.2),degrees=15)
        self.Elastic_transform = RandomElasticDeformation(num_control_points=(5, 5, 5),max_displacement=4,locked_borders=2)
        self.Blur_transform = tio.transforms.RandomBlur()
        self.Motion_transform = tio.transforms.RandomMotion()
        self.RandomNoise = tio.transforms.RandomNoise(std=0.1)
        self.RandomBiasField = tio.transforms.RandomBiasField()
        self.transform = tio.Compose([ self.Affine_transform, self.Blur_transform,
                                       self.Elastic_transform,self.Motion_transform,
                                      self.RandomBiasField,self.RandomNoise])



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
        # plt.imshow(subj.tissues.data[0, :,128].type(torch.bool))
        # plt.show()
        # boolean_array=[ 506, 507, 509, 511, 512, 515, 516, 518, 530]
        for label in label_unique:
            if label not in boolean_array:
                subj.tissues.data[subj.tissues.data == label] = 0
                # plt.imshow(subj.tissues.data[0, :, :, 128].type(torch.bool))
                # plt.title(str(label))
                # plt.show()

        # subj.tissues.data = process_brain_mask(subj.tissues.data)
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
            std=np_stds,
            ignore_background=True
        )

        transformed = self.simulation_transform(subject_)
        transformed = self.filter_labels(transformed, self.output_labels)
        transformed = self.transform(transformed)

        im = transformed['image_from_labels']['data']
        label = transformed['tissues']['data']
        return im,label


class SynthSegDataset(Dataset):
    def __init__(self, config):
        self.config = config

        self.brain_generator = BrainGenerator(
            labels_dir=config['path_label_map'],
            generation_classes=config['generation_classes'],
            generation_labels = config['generation_labels'],
            output_labels=config['output_labels'],
            prior_means=config['prior_means'],
            prior_stds=config['prior_stds'],
        )
        # self.blurring_transform = tio.RandomBlur(std=0.3)
        # self.coils = mri.sim.birdcage_maps((8, 256, 256, 256))
    def __len__(self):
        return self.config['n_examples']

    def __getitem__(self, idx):
        im, label = self.brain_generator.generate_brain()
        # subj['image_from_labels']['data'] = subj['image_from_labels']['data'] * (self.coils)
        return im.squeeze().permute(2,1,0).flip([0]), label.squeeze().permute(2,1,0).flip([0])







def get_dataloader(config):
    dataset = SynthSegDataset(config)

    dataloader = DataLoader(dataset,
                            batch_size=config['batch_size'],
                            shuffle=True,
                            num_workers=0)

    return dataloader


# config.py

config = {
    'batch_size': 1,
    'n_examples': 128,
    'path_label_map': 'data/training_label_maps',
    'generation_labels': 'data/labels_classes_priors/generation_labels.npy',
    'output_labels': 'data/labels_classes_priors/synthseg_segmentation_labels_amir.npy',
    'generation_classes': 'data/labels_classes_priors/generation_classes_contrast_specific.npy',
    'prior_means':'data/labels_classes_priors/prior_means_t1.npy',
    'prior_stds':'data/labels_classes_priors/prior_stds_t1.npy'
}

def main():
    dataloader = get_dataloader(config=config)

    for i_batch, sample_batch in enumerate(dataloader):
        # training code here

        fig,ax = plt.subplots(1,2)
        ax[0].imshow((sample_batch[0])[0, :,128], cmap='gray',vmin=0)
        ax[1].imshow((sample_batch[1]).numpy().astype(bool)[ 0, :, 128],vmin=0)
        plt.show()
        result_dir = "result/"
        affine_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        nibabel.Nifti1Image(sample_batch[0].squeeze().permute(2,1,0).flip([0]).numpy(),affine=affine_matrix ).to_filename(
                          os.path.join(result_dir, 'image_%s.nii.gz' % i_batch))
        nibabel.Nifti1Image((sample_batch[1].squeeze().numpy() != 0).astype(int),affine=affine_matrix ).to_filename(
                          os.path.join(result_dir, 'label_%s.nii.gz' % i_batch))
        # nibabel.Nifti1Image(sample_batch['mask'].squeeze().numpy(),
        #                     sample_batch['tissues']['affine'].squeeze().numpy()).to_filename(
        #                   os.path.join(result_dir, 'mask_%s.nii.gz' % i_batch))
        pass


if __name__ == '__main__':
    # freeze_support() #here if program needs to be frozen
    main()