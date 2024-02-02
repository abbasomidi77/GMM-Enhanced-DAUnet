# GMM-Enhanced DAUnet

Welcome to GMM-Enhanced DAUnet, an unsupervised domain adaptation tool for performing MRI skull-stripping on newborn data using a model trained on adult data. This repository allows you to seamlessly adapt your model for this challenging task.

![midf](https://github.com/abbasomidi77/GMM-Enhanced-DAUnet/assets/61683254/510045b2-fad9-44c4-afd3-880a8c9ace0f)

# GMM Data
The code for creating synthetic data can be found in the GMM folder. You can utilize this code to generate data and subsequently incorporate it into your source dataset directory. 

# Data
We also have leveraged the CALGARY-CAMPINAS PUBLIC BRAIN MR DATASET, available at [this link](https://sites.google.com/view/calgary-campinas-dataset/home). Please download the dataset from the provided link to proceed. Additionally, we have included sample data-split CSV files in the Data-split folder to help you organize your data.

# Configuration
The current patch size for both adult and newborn data is set to 112x112x122. However, you can easily customize the patch sizes for both datasets by modifying the parameters in the data_loader_da.py script.

# Getting Started
To run the code, execute the following command, making sure to adjust the paths to your source and target data splits and your desired output folder:

`python main.py --batch_size 2 --epochs 400 --source_dev_images Data-split/source_train_set_neg.csv --source_dev_images Data-split/source_train_set_masks_neg.csv --source_dev_images Data-split/target_train_set.csv --results_dir Outputs`

# Resuming Training
If you need to resume your training, simply include the `--resume_training True` argument in your command.

Feel free to reach out if you have any questions or need assistance.
