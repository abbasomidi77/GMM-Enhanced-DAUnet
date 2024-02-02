import urllib.parse

import torch
from torchio import LabelMap
from torchio.datasets.mni.mni import SubjectMNI
from torchio.utils import get_torchio_cache_dir


class Subject(SubjectMNI):
    r"""ICBM template.

    More information can be found in the `website
    <http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009>`_.

    .. image:: http://www.bic.mni.mcgill.ca/uploads/ServicesAtlases/mni_icbm152_sym_09c_small.jpg
        :alt: ICBM 2009c Nonlinear Symmetric

    Args:
        load_4d_tissues: If ``True``, the tissue probability maps will be loaded
            together into a 4D image. Otherwise, they will be loaded into
            independent images.

    Example:
        >>> import torchio as tio
        >>> icbm = tio.datasets.ICBM2009CNonlinearSymmetric()
        >>> icbm
        ICBM2009CNonlinearSymmetric(Keys: ('t1', 'eyes', 'face', 'brain', 't2', 'pd', 'tissues'); images: 7)
        >>> icbm = tio.datasets.ICBM2009CNonlinearSymmetric(load_4d_tissues=False)
        >>> icbm
        ICBM2009CNonlinearSymmetric(Keys: ('t1', 'eyes', 'face', 'brain', 't2', 'pd', 'gm', 'wm', 'csf'); images: 9)
    """  # noqa: B950

    def __init__(self, files_dir, load_4d_tissues: bool = True):


        subject_dict = {

        }
        if load_4d_tissues:
            subject_dict['tissues'] = LabelMap(
                files_dir,
                channels_last=True,
            )
        subject_dict['folder'] = files_dir
        # original_labels = subject_dict['tissues'].data.unique() # Add all your original labels here
        # new_labels = list(range(len(original_labels)))  # [0, 1, 2, 3, ...]
        #
        # # Create a mapping dictionary
        # label_mapping = {original: new for original, new in zip(original_labels, new_labels)}
        # remapped_image = subject_dict['tissues'].data.clone()  # Clone the image tensor
        # for original, new in label_mapping.items():
        #     remapped_image[subject_dict['tissues'].data == original] = new
        # subject_dict['tissues'].data = remapped_image
        super().__init__(subject_dict)
