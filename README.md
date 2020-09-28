# README 






1. Nuclei Segmentation
    
    - 1.1 Setup
        
        - Folder setup
        - Environment setup
    
    - 1.2 Load images

    - 1.3 Annotate images using cvat

    - 1.4 Preprocessing of images

         - Convert images to png
         - Create training/validation/test fraction text files
         - Normalization
         - Creating border labels
         - Augmentation (affine transformation)
         - Preprocessing of image-sets for Model 5

    - 1.5 Training

    - 1.6 Prediction

    - 1.7 Evaluation
    
    - 1.8 Models

    - 1.9 Scripts

         - format_convertion.py
         - config.py
         - 00-load-and-reformat-dataset.py
         - 01-Augmentation.py
         - 02-training.py
         - 03-prediction.py
         - 04-evaluation.ipynb
         - 06-resize-images.py
         - Relabeling_1.py
         - Relabeling_2.py
         - Utils scripts
             - augmentation.py 
             - data_provider.py
             - dirtools.py
             - evaluation.py
             - experiment.py
             - metrics.py
             - model_builder.py
             - objectives.py
         - Preprocessing of datasets for Model 5
             - Preprocessing_and_merging_annotations_BBBC038.py
             - Preprocessing_BBBC020.py
             - Preprocessing_celltracking_images.py

    - 1.10 Docs

      - filelist_wrong_images.txt
      - 1-2_training.txt
      - 3_training.txt
      - 4_training.txt
      - 5_training_500.txt
      - 6_training.txt
      - 7_training.txt
      - 8_training.txt
      - 9_training.txt



# 1 Nuclei Segmentation



I have with help from the script by Broad Bioimage Benchmarc Collection (BBBC), found at https://github.com/carpenterlab/unet4nuclei, recreated their experiment with the use of images from Aits lab, and additional images from other open source datasets.

In the BBBC repository, the full script can be found for their experiments, but for our purpose the script has been modified.

Some modifications had to be done due to outdated versions of python packages, and some modifications because of different image-formats. In our experiment it was also of importance to include micronucleis in the modelprediction, so the scripts are modified to fit that purpose.


This README will give instructions of how to recreate the experiments I've made.

All scripts and textfiles are provided in this README, but can also be downloaded from www.github.com/Malou-A/Nuclei_Segmentation 

## 1.1 Setup

### Folder setup

To be able to run the scripts without doing major changes in the script, the folder structure should look the same as in this project.

In the script **config.py**, the variable 
```python
config_vars["home_folder"] = '/home/maloua/Malou_Master/5_Models'
```
must be changed to your absolute pathway to where you will store this project. The subfolders should have the following structure:

Folder structure:

- **2_Final_Models**
    - **1_Model1**
        - 02-training.py
        - 03-prediction.py
        - 04-evaluation.ipynb
        - config.py
        - **utils**
            - augmentation.py
            - data_provider.py
            - dirtools.py
            - evaluation.py
            - experiment.py
            - metrics.py
            - model_builder.py
            - objectives.py

    - **data**
        - **1_raw_annotations**
        - **2_raw_images**
        - **3_preprocessing_of_data**
            - 00-load-and-reformat-dataset.py
            - 01-Augmentation.py
            - 05-Augment-validation-set.py
            - 06-resize-images.py
            - Relabeling_1.py
            - Relabeling_2.py
            - config.py
        - **4_filelists**
            - 1-2_training.txt
            - 3_training.txt
            - 4_training.txt
            - 5_training.txt
            - 6_training.txt
            - 7_training.txt
            - 8_training.txt
            - TEST.txt
            - VALIDATION.txt
        - **utils**
            - augmentation.py
            - data_provider.py
            - dirtools.py
            - evaluation.py
            - experiment.py
            - metrics.py
            - model_builder.py
            - objectives.py
- **3_data**
    - **2_additional_datasets**
        - **1_celltracking_challenge_data**
        - **2_BBBC_image_sets**


In the experiment several models are created. The folder structure for each model is exactly the same as **1_Model1**, and for each model added, an identical folder is added named **#_Model#** where "#" is replaced with the model number. Each model has some modifications done in the scripts which is specified in section **1.7 - Models**

### Environment setup

## 1.2 Load images

 
### Download BBBC images

The images from the bbbc experiment can be found here:

https://data.broadinstitute.org/bbbc/BBBC039/images.zip

The images are extracted and put in the folder **2_Final_Models/data/2_raw_images**

And the annotations for the images:

https://data.broadinstitute.org/bbbc/BBBC039/masks.zip

The mask images are extracted and put in the folder **2_Final_Models/data/1_raw_annotations**
Downloading image sets for Model 5

### Additional datasets



**Celltracking challenge data:**

<u> GFP-GOWT1 mouse stem cells:</u>

http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip

<u> HeLa cells stably expressing H2b-GFP: </u>

http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DL-HeLa.zip

<u> Simulated nuclei of HL60 cells stained with Hoescht:</u>

http://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-SIM+.zip

**BBBC image sets:**

<u> kaggle-dsbowl-2018-dataset:</u>

https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes/archive/master.zip

<u> BBBC Murine bone-marrow derived macrophages: </u>

https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_images.zip

https://data.broadinstitute.org/bbbc/BBBC020/BBBC020_v1_outlines_nuclei.zip

The zipfiles should be extracted to the following folderstructure:

- **3_data**
    - **2_additional_datasets**
        - **1_celltracking_challenge_data**
            - Fluo-N2DL-HeLa
            - Fluo-N2DH-SIM+
            - Fluo-N2DH-GOWT1
        - **2_BBBC_image_sets**
            - kaggle-dsbowl-2018-dataset-fixes
            - BBBC020_v1_images
            - BBBC020_v1_outlines_nuclei

    


**Aits lab images**

We have worked with 100 randomly selected images from the full data set consisting of approx. 6 million images, these images are of the format .C01 which is a microscopy format. The input images in the BBBC scripts are expected to be .tiff or .png, so all 100 images are first converted to png.

For this I have created a script that will take a directory as input, and output the converted images to a new selected directory.

The script is done with argparse, and can be used just by downloading the script and following these steps:


**Downloading and installing bftools:**
```bash
cd ~/bin
wget http://downloads.openmicroscopy.org/latest/bio-formats/artifacts/bftools.zip
unzip bftools.zip
rm bftools.zip
export PATH=$PATH:~/bin/bftools
```
**Installing required python packages:**

```bash
pip install argparse
pip install os
pip install subprocess
pip install tqdm
pip install pathlib
```


**The program is run like:**

```bash
python3 format_convertion.py -i INDIR -o OUTDIR -ift IN_FILETYPE -oft OUT_FILETYPE
```

The script **format_convertion.py** is found in section **1.9 - Scripts**



## 1.3 Annotate images using cvat

We have annotated 50 images to use in the experiment, we have used the annotation program cvat. Information about the program and installation instructions are found on their github page, https://github.com/opencv/cvat.

Only one label is used for annotation, nucleus, and each nucleus is drawn with the polygon tool.

The work is saved using the option “DUMP ANNOTATION” -> “MASK ZIP 1.1”

That will create and download a zip file with one folder of images only with the class (nucleus), showing all nucleus in the same color, and one folder with annotations of the objects, each image will be an RGB image, with all objects being different colors to distinguish between them.

In the creation of our labels, the object images was used. The images should be extracted to the folder These images are extracted to the folder **2_Final_Models/data/1_raw_annotations**

These images are not of the same format as the bbbc institute’s, so the script had to be modified to fit these images.

The images we have annotated are the ones with filenames found in **1-2_training.txt**, **VALIDATION.txt** and **TEST.txt**, which can be found in section **1.10 - Docs**

## 1.4 Preprocessing of images

### Normalization and creation of boundary labels

First the images are normalized to have pixel values between 0-1 instead of 0-255, and converted to png if that is not already done.

Then boundary labels are created. Objects are found in the annotation image using the skimage module, both for finding distinguished objects, and for finding boundaries of the objects. 

These steps are done in the script **00-load-and-reformat-dataset.py**

The annotations are expected to be one image with all annotations, where each object is of different pixelvalue. for the images downloaded for Model 5, some additional preprocessing was needed.

### Preprocessing of the images for Model 5

#### kaggle-dsbowl-2018-dataset:

This datasets consists of different images, many images that is not similar to our dataset, and not wanted in our model. It is no specific structure of the directories and where to find the similar images, but the images similar to ours are all grey scale, and can be distinguished in that way.

Another difference with this dataset is that the annotations are separated per object, so that it exists one image per object instead of one image with all objects.

To extract the images a script was created, it goes through the directories, it is one image per directory. The image is checked if it is grey scale or not. The image type is RGB-D even if it is grey scale, so to control if it is grey scale the pixel values are compared. If the image is grey scale, the first 3 values of each pixel are expected to be the same (the fourth value is the value of the image transparancy).

If the image is gray scale then the masks are extracted and combined to form one image. Each mask image is black and white, the object is white. The mask images are combined, with each mask given a different pixel value.

In the same script the normalization and boundarylabeling are done in the same way as the previous datasets.

The script is named **Preprocessing_and_merging_annotations_BBBC038.py** and can be found under **1.9 - Scripts**

After running the script and looking through the images, some images were not suited for our model, so they were removed. It was done by creating a list with all the images to be removed and the following bash command:

Create a doc folder:
```bash
mkdir 3_data/2_additional_datasets/2_BBBC_image_sets/doc
cd 3_data/2_additional_datasets/2_BBBC_image_sets/doc
```

the content of **filelist_wrong_images.txt** is found under **1.10 - Docs**:

Put the textfile in the created doc folder, and execute the following command:
```bash
cat filelist_wrong_images.txt | while read line; do rm ../BBBC038_*/$line; done
```


#### BBBC Murine bone-marrow derived macrophages:

Each folder in **3_data/2_additional_datasets/2_BBBC_image_sets/BBBC020_v1_images** consists of 3 subfolders with images, one with the cells, one with the nucleus, and one with the combined images. We are only interested in the images of the nucleus, and need to extract those. These images are the ones with the ending _c5.TIF in their names.

The images needs to be converted to grey images, and the mask images needs to be combined as in the BBBC038 dataset.

This is all done with the script **Preprocessing_BBBC020.py** which is found under **1.9 - Scripts**

#### Celltracking challenge images (GFP-GOWT1 mouse stem cells, HeLa cells stably expressing H2b-GFP and Simulated nuclei of HL60 cells stained with Hoescht)

These images need no specific preprocessing. A script was created to extract, normalize and create boundary labels for all the images in one go.

The script is named **Preprocessing_celltracking_images.py** and is found under **1.9 - Scripts**.

Creating 5_training.txt:

For Model 5, all these images should be used in the training set, together with the images used in Model 4. The below bash commands will create the file 5_training.txt with all the images, and place it in the folder **2_Final_Models/data/4_filelists**.
```bash
ls 3_data/2_additional_datasets/2_BBBC_image_sets/BBBC038_normalized_images/ > 2_Final_Models/data/4_filelists/5_training.txt && 
ls 3_data/2_additional_datasets/2_BBBC_image_sets/BBBC020_normalized_images/ >> 2_Final_Models/data/4_filelists/5_training.txt && 
ls 3_data/2_additional_datasets/1_celltracking_challenge_data/normalized_images/ >> 2_Final_Models/data/4_filelists/5_training.txt &&
```
For Model 5 we want 500 additional images, but we have 1368 images. To use only 500 images the file 5_training.txt is randomly sorted and the 500 first lines are used and put in a new textfile using bashcommand:

```bash
sort -R 2_Final_Models/data/4_filelists/5_training.txt | head -n 500 > 2_Final_Models/data/4_filelists/5_training_500.txt
```
The list we got and used for this can be found under **1.10 - Docs**

Model 5 should also include the images used in Model 4, so the lines from **4_training.txt** should be added to **5_training_500.txt**. It is done with the bash command:

```bash
cat 2_Final_Models/data/4_filelists/4_training.txt >> 2_Final_Models/data/4_filelists/5_training_500.txt
```

#### Moving all data to the same folder:

After doing the above steps with the preprocessing, all images are ready to use in the model for training. The images should be moved to **2_Final_Models/data/boundary_labels** and **2_Final_Models/data/norm_images**. These folders were created during the preprocessing step when running the script **00-load-and-reformat-dataset.py**

Moving the files is done using bash command:
```bash
mv 3_data/2_additional_datasets/1_celltracking_challenge_data/boundary_labels/* 2_Final_Models/data/boundary_labels/

mv 3_data/2_additional_datasets/1_celltracking_challenge_data/normalized_images/* 2_Final_Models/data/norm_images/

mv 3_data/2_additional_datasets/2_BBBC_image_sets/BBBC020_boundary_labels/* 2_Final_Models/data/boundary_labels/

mv 3_data/2_additional_datasets/2_BBBC_image_sets/BBBC020_normalized_images/* 2_Final_Models/data/norm_images/

mv 3_data/2_additional_datasets/2_BBBC_image_sets/BBBC038_boundary_labels/* 2_Final_Models/data/boundary_labels/

mv 3_data/2_additional_datasets/2_BBBC_image_sets/BBBC038_normalized_images/* 2_Final_Models/data/norm_images/
```

### Augmentation

To increase the size of the dataset, we have done augmentations on the images. Some augmentations are implemented in the training step. In some models we have used affine transformation on the images, which is not done in the training step, which creates an image that is slightly transformed but is interpreted as a different image to the neural network. The original script was provided from the BBBC group, but needed some adjustments.

For Model 1, the 30 images will be augmented with affine transformation, which is done using script **01-Augmentation.py**.

For Model 2.2 augmentation was also done for the images in the validation set, which was not done in **01-Augmentation.py**, the script for the validation set is the same with some small changes, the full script is **05-Augment-validation-set.py**. Both scripts are found in section **1.9 - Scripts**


## 1.5 Training



## 1.6 Prediction

In the prediction step, the model is loaded and predicts the images in the validation set. The images generated from the prediction are red, green and blue images, as the boundary label images. From the prediction images objects are identified by the green area. A True and False array is created with the value True for all green pixels, and False everywhere else. The skimage.morphology.label module will then create a label image from this, that will be stored and later used in the evaluation step.

This is done using script **03-prediction.py** which is found in section **1.9 -Scripts**


## 1.7 Evaluation

## 1.8 Models

* **Model 1** - Using 40 images from Aits lab. 30 images for training, and 10 for validation.
* **Model 2** - Using 40 images from Aits lab, with elastic transformation done on the training images so a total of 300 images is used as the training set.
* **Model 3** - The same images as Model 2, with 100 additional images from the BBBC039 image set.
* **Model 4** - The same images as Model 2, with 200 additional images from the BBBC039 image set.
* **Model 5** - The same as Model 4, with additional 500 randomly selected images from the image collections BBBC038, BBBC020, "GFP-GOWT1 mouse stem cells", "HeLa cells stably expressing H2b-GFP" and "Simulated nuclei of HL60 cells stained with Hoescht"
* **Model 6** - Using 2 images from Aits lab
* **Model 7** - Using 5 images from Aits lab
* **Model 8** - Using 15 images from Aits lab
* **Model 9** - Model 1 trained for additional 15 epochs
* **Model 10** - Same images as Model 1 and 29 additional images from the BBBC038 image set, handpicked based on similarity to aits images.
* **Model 11** - Using only the 29 handpicked images used in Model 10.
* **Model 12** - Same as Model 10, with the 29 additional images resized to the same size as the images from Aits lab (1104x1104).
* **Model 13** - 40 Aits lab images with affine transformation + 333 images from BBBC038
* **Model 14** - Model 3 trained for 15 additional epochs
* **Model 15** - Model 5 trained for 15 additional epochs
* **Model 16** - Model 10 trained for 15 additional epochs
* **Model 17** - Model 1, with the weight parameter of the boundary label changed from 10 to 5. No dilation in the evaluation step
* **Model 18** - Model 1, with modification to the weight parameter of the boundary label. backround, interior and boundary now has the same value instead of 1/1/10. No dilation done in the evaluation step now.
* **Model 19** - Model 13 trained for 15 additional epochs
* **Model 20** - Same images as Model 1, but boundary labels created differently, objects are increased in size and then applied 2 pixel boundaries, one inside of object, and one outside.
* **Model 21** - Same images as Model 1, but 2 pixel boundaries.
* **Model 22** - Model 20 trained for 5 additional epochs
* **Model 24** - Model created with same images as Model 1, but with boundaries created as the original BBBC model (2 pixels boundary). Objects less than 40 pixels removed before training.
* **Model 25** - Model created with image set 2, and objects less than 25 pixels removed before training.
* **Model 26** - Model created with image set 2, and objects less than 100 pixels removed before training.


For all models, the variable "experiment_name" needs to be changed to the corresponding model name. e.g. for Model 5 ```experiment_name = 'Model_5'```. The variable is found in script **02-training.py**, **03-prediction.py** and **04-evaluation.ipynb**


For model 3, 4, 5 and 13, where affine transformation is done on the images, following line in the script **02-training.py** needs to be changed:
```python
data_partitions = utils.dirtools.read_data_partitions(config_vars, load_augmented = False)
```
changes to :
```python
data_partitions = utils.dirtools.read_data_partitions(config_vars)
```

### Additional changes in corresponding model :

#### Model 3:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/3_training.txt'
```

#### Model 4:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/4_training.txt'
```

#### Model 5:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/5_training_500.txt'
```

#### Model 6:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/6_training.txt'
```

#### Model 7:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/7_training.txt'
```

#### Model 8:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/8_training.txt'
```

#### Model 9:
in script **02-training.py**
under section "# build model" where this line is found:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
```
Additional line is added below:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
model.load_weights(config_vars["home_folder"] + '/2_Final_Models/data/experiments/Model_1/model.hdf5')
```


#### Model 10
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/10_training.txt'
```

#### Model 11
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/5_Models/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/11_training.txt'
```
#### Model 12
For this model some images were resized, which was done using the script **06-resize-images.py**, found in section **1.9 - Scripts**.


in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/12_training.txt'
```

#### Model 13
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/13_training.txt'
```


#### Model 14:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/3_training.txt'
```
and under section "# build model" where this line is found:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
```
Additional line is added below:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
model.load_weights(config_vars["home_folder"] + '/2_Final_Models/data/experiments/Model_3/model.hdf5')
```
#### Model 15:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + /2_Final_Models/data/4_filelists/5_training.txt'
```

and under section "# build model" where this line is found:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
```
Additional line is added below:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
model.load_weights(config_vars["home_folder"] + '/2_Final_Models/data/experiments/Model_5/model.hdf5')
```

#### Model 16:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/10_training.txt'
```

and under section "# build model" where this line is found:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
```
Additional line is added below:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
model.load_weights(config_vars["home_folder"] + '/2_Final_Models/data/experiments/Model_10/model.hdf5')
```

#### Model 17

In the utils script **objectives.py** the following line is changed:

```python
class_weights = tf.constant([[[[1., 1., 10.]]]])
```
to:

```python
class_weights = tf.constant([[[[1., 1., 5.]]]])
```
and in the script **04-evaluation.ipynb** the following section is removed:

```python
# Apply object dilation
if config_vars["object_dilation"] > 0:
    struct = skimage.morphology.square(config_vars["object_dilation"])
    prediction = skimage.morphology.dilation(prediction, struct)
elif config_vars["object_dilation"] < 0:
    struct = skimage.morphology.square(-config_vars["object_dilation"])
    prediction = skimage.morphology.erosion(prediction, struct)
```

#### Model 18

In the utils script **objectives.py** the following line is changed:

```python
class_weights = tf.constant([[[[1., 1., 10.]]]])
```
to:

```python
class_weights = tf.constant([[[[1., 1., 1.]]]])
```

and in the script **04-evaluation.ipynb** the following section is removed:

```python
# Apply object dilation
if config_vars["object_dilation"] > 0:
    struct = skimage.morphology.square(config_vars["object_dilation"])
    prediction = skimage.morphology.dilation(prediction, struct)
elif config_vars["object_dilation"] < 0:
    struct = skimage.morphology.square(-config_vars["object_dilation"])
    prediction = skimage.morphology.erosion(prediction, struct)
```

#### Model 19:
in script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/13_training.txt'
```

and under section "# build model" where this line is found:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
```
Additional line is added below:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
model.load_weights(config_vars["home_folder"] + '/2_Final_Models/data/experiments/Model_13/model.hdf5')
```

#### Model 20:
For this model new boundary labels were created using the script **Relabeling_1.py**

In script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/relabel_1-2_training.txt'
```
#### Model 21:
For this model new boundary labels were created using the script **Relabeling_2.py**

In script **02-training.py**

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
```
changes to:

```python
config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/relabel2_1-2_training.txt'
```

#### Model 22:
In **config.py**

the variable 
```python
config_vars["epochs"] = 15
```
is changed to:

```python
config_vars["epochs"] = 5
```


In script **02-training.py**

Under section "# build model" where this line is found:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
```
Additional line is added below:

```python
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
model.load_weights(config_vars["home_folder"] + '/2_Final_Models/data/experiments/Model_20/model.hdf5')
```



## 1.9 Scripts

### 00-load-and-reformat-dataset.py

```python
#!/usr/bin/env python
# coding: utf-8

import glob
import os
import shutil
import zipfile
import requests
from config import config_vars
import random
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from tqdm.notebook import tqdm
import skimage.io
import skimage.segmentation
import utils.dirtools
import utils.augmentation
from skimage.util import img_as_ubyte
from skimage.color import rgb2lab


# Create output directories for transformed data

os.makedirs(config_vars["normalized_images_dir"], exist_ok=True)
os.makedirs(config_vars["boundary_labels_dir"], exist_ok=True)

config_vars["raw_images_dir"]= config_vars["home_folder"] + '/2_Final_Models/data/2_raw_images/'
config_vars["raw_annotations_dir"]=config_vars["home_folder"] + '/2_Final_Models/data/1_raw_annotations/'
# ## Create file-lists

# ## Normalize images


if config_vars["transform_images_to_PNG"]:
    
    filelist = sorted(os.listdir(config_vars["raw_images_dir"]))

    # run over all raw images
    for filename in tqdm(filelist):

        # load image and its annotation
        orig_img = skimage.io.imread(config_vars["raw_images_dir"] + filename)       

        # IMAGE

        # normalize to [0,1]
        percentile = 99.9
        high = np.percentile(orig_img, percentile)
        low = np.percentile(orig_img, 100-percentile)

        img = np.minimum(high, orig_img)
        img = np.maximum(low, img)

        img = (img - low) / (high - low) # gives float64, thus cast to 8 bit later
        img = skimage.img_as_ubyte(img) 

        skimage.io.imsave(config_vars["normalized_images_dir"] + filename[:-3] + 'png', img)    
else:
    config_vars["normalized_images_dir"] = config_vars["raw_images_dir"]


# ## Create boundary labels


filelist = sorted(os.listdir(config_vars["raw_annotations_dir"]))
from skimage.util import img_as_ubyte
from skimage.color import rgb2lab
total_objects = 0

# run over all raw images
for filename in tqdm(filelist):
    
    # GET ANNOTATION
    annot = skimage.io.imread(config_vars["raw_annotations_dir"] + filename)

    # strip the first channel
    if annot.shape[2]!=3:
        annot = annot[:,:,0]
    else:
        annot = rgb2lab(annot)
        annot = annot[:,:,0]
    # label the annotations nicely to prepare for future filtering operation
    
    annot = skimage.morphology.label(annot)
    total_objects += len(np.unique(annot)) - 1
      
    # find boundaries
    boundaries = skimage.segmentation.find_boundaries(annot, mode = 'outer')

    # BINARY LABEL
    
    # prepare buffer for binary label
    label_binary = np.zeros((annot.shape + (3,)))
    
    # write binary label
    label_binary[(annot == 0) & (boundaries == 0), 0] = 1
    label_binary[(annot != 0) & (boundaries == 0), 1] = 1
    label_binary[boundaries == 1, 2] = 1
    
    label_binary = img_as_ubyte(label_binary)
    # save it - converts image to range from 0 to 255
    skimage.io.imsave(config_vars["boundary_labels_dir"] + filename, label_binary)
    
print("Total objects: ",total_objects)
```

### 02-training.py

```python
#!/usr/bin/env python
# coding: utf-8

# # Step 02
# # Training a U-Net model


import sys
import os

import numpy as np
import skimage.io

import tensorflow as tf

import keras.backend
import keras.callbacks
import keras.layers
import keras.models
import keras.optimizers

import utils.model_builder
import utils.data_provider
import utils.metrics
import utils.objectives
import utils.dirtools

# Uncomment the following line if you don't have a GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ''


from config import config_vars

config_vars['path_files_training'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/1-2_training.txt'
config_vars['path_files_validation'] =config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/VALIDATION.txt'
config_vars['path_files_test'] = config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/TEST.txt'


# # Configuration

experiment_name = 'Model_1'

config_vars = utils.dirtools.setup_experiment(config_vars, experiment_name)

data_partitions = utils.dirtools.read_data_partitions(config_vars, load_augmented = False)


# # Initiate data generators


# build session running on GPU 1
configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "1"
session = tf.compat.v1.Session(config = configuration)

# apply session
tf.compat.v1.keras.backend.set_session(session)

train_gen = utils.data_provider.random_sample_generator(
    config_vars["normalized_images_dir"],
    config_vars["boundary_labels_dir"],
    data_partitions["training"],
    config_vars["batch_size"],
    config_vars["pixel_depth"],
    config_vars["crop_size"],
    config_vars["crop_size"],
    config_vars["rescale_labels"]
)

val_gen = utils.data_provider.single_data_from_images(
     config_vars["normalized_images_dir"],
     config_vars["boundary_labels_dir"],
     data_partitions["validation"],
     config_vars["val_batch_size"],
     config_vars["pixel_depth"],
     config_vars["crop_size"],
     config_vars["crop_size"],
     config_vars["rescale_labels"]
)



# build model
model = utils.model_builder.get_model_3_class(config_vars["crop_size"], config_vars["crop_size"], activation=None)
model.summary()

#loss = "categorical_crossentropy"
loss = utils.objectives.weighted_crossentropy

metrics = [keras.metrics.categorical_accuracy, 
           utils.metrics.channel_recall(channel=0, name="background_recall"), 
           utils.metrics.channel_precision(channel=0, name="background_precision"),
           utils.metrics.channel_recall(channel=1, name="interior_recall"), 
           utils.metrics.channel_precision(channel=1, name="interior_precision"),
           utils.metrics.channel_recall(channel=2, name="boundary_recall"), 
           utils.metrics.channel_precision(channel=2, name="boundary_precision"),
          ]

optimizer = keras.optimizers.RMSprop(lr=config_vars["learning_rate"])

model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

# Performance logging
callback_csv = keras.callbacks.CSVLogger(filename=config_vars["csv_log_file"])

callbacks=[callback_csv]



# TRAIN
statistics = model.fit_generator(
    generator=train_gen,
    steps_per_epoch=config_vars["steps_per_epoch"],
    epochs=config_vars["epochs"],
    validation_data=val_gen,
    validation_steps=int(len(data_partitions["validation"])/config_vars["val_batch_size"]),
    callbacks=callbacks,
    verbose = 1
)

model.save_weights(config_vars["model_file"])

print('Done! :)')

```

### 03-prediction.py

```python
import os
import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import skimage.io
import skimage.morphology

import tensorflow as tf
import keras

import utils.metrics
import utils.model_builder
print(skimage.__version__)


# # Configuration

from config import config_vars

# Partition of the data to make predictions (test or validation)

config_vars['path_files_training'] = '/home/maloua/Malou_Master/5_Models/2_Final_Models/data/4_filelists/1-2_training.txt'
config_vars['path_files_validation'] ='/home/maloua/Malou_Master/5_Models/2_Final_Models/data/4_filelists/VALIDATION.txt'
config_vars['path_files_test'] = '/home/maloua/Malou_Master/5_Models/2_Final_Models/data/4_filelists/TEST.txt'

partition = "validation"

experiment_name = 'Model_1'

config_vars = utils.dirtools.setup_experiment(config_vars, experiment_name)

data_partitions = utils.dirtools.read_data_partitions(config_vars)


# Configuration to run on GPU
configuration = tf.compat.v1.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.visible_device_list = "0"

session = tf.compat.v1.Session(config = configuration)

# apply session
tf.compat.v1.keras.backend.set_session(session)


# # Load images and run predictions

image_names = [os.path.join(config_vars["normalized_images_dir"], f) for f in data_partitions[partition]]

imagebuffer = skimage.io.imread_collection(image_names)

images = imagebuffer.concatenate()

dim1 = images.shape[1]
dim2 = images.shape[2]

images = images.reshape((-1, dim1, dim2, 1))

# preprocess (assuming images are encoded as 8-bits in the preprocessing step)
images = images / 255

# build model and load weights
model = utils.model_builder.get_model_3_class(dim1, dim2)
model.load_weights(config_vars["model_file"])

# Normal prediction time
predictions = model.predict(images, batch_size=1)

# # Transform predictions to label matrices

for i in range(len(images)):

    filename = imagebuffer.files[i]
    filename = os.path.basename(filename)

    probmap = predictions[i].squeeze()
    

    skimage.io.imsave(config_vars["probmap_out_dir"] + filename, probmap)
    
    pred = utils.metrics.probmap_to_pred(probmap, config_vars["boundary_boost_factor"])

    
    label = utils.metrics.pred_to_label(pred,config_vars['cell_min_size'] )
 
    
    skimage.io.imsave(config_vars["labels_out_dir"] + filename, label)
```

### 04-evaluation.ipynb
This script is run as a notebook in jupyter notebook. Each section should be run separately.

```python
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

import skimage.io
import skimage.morphology
import skimage.segmentation

import utils.evaluation
from config import config_vars
```

#### Partition of the data to make predictions (test or validation)

```python
config_vars['path_files_training'] = '/home/maloua/Malou_Master/5_Models/2_Final_Models/data/4_filelists/1-2_training.txt'
config_vars['path_files_validation'] ='/home/maloua/Malou_Master/5_Models/2_Final_Models/data/4_filelists/VALIDATION.txt'
config_vars['path_files_test'] = '/home/maloua/Malou_Master/5_Models/2_Final_Models/data/4_filelists/TEST.txt'

partition = "validation"

experiment_name = 'Model_1'

config_vars = utils.dirtools.setup_experiment(config_vars, experiment_name)

data_partitions = utils.dirtools.read_data_partitions(config_vars)

```
#### Display prediction along with segmentation to visualize errors
```python

def show(ground_truth, prediction, threshold=0.5, image_name="N"):

    # Compute Intersection over Union
    IOU = utils.evaluation.intersection_over_union(ground_truth, prediction)

    # Create diff map
    diff = np.zeros(ground_truth.shape + (3,))
    A = ground_truth.copy()
    B = prediction.copy()
    A[A > 0] = 1
    B[B > 0] = 1
    D = A - B
    #diff[D > 0,:2] = 1
    #diff[D < 0,1:] = 1

    # Object-level errors
    C = IOU.copy()
    C[C>=threshold] = 1
    C[C<threshold] = 0
    missed = np.where(np.sum(C,axis=1) == 0)[0]
    extra = np.where(np.sum(C,axis=0) == 0)[0]

    for m in missed:
        diff[ground_truth == m+1, 0] = 1
    for e in extra:
        diff[prediction == e+1, 2] = 1

    # Display figures
    fig, ax = plt.subplots(1, 4, figsize=(18,6))
    ax[0].imshow(ground_truth)
    ax[0].set_title("True objects:"+str(len(np.unique(ground_truth))))
    ax[1].imshow(diff)
    ax[1].set_title("Segmentation errors:"+str(len(missed)))
    ax[2].imshow(prediction)
    ax[2].set_title("Predicted objects:"+str(len(np.unique(prediction))))
    ax[3].imshow(IOU)
    ax[3].set_title(image_name)
```


#### Run the evaluation
```python
all_images = data_partitions[partition]
from skimage.color import rgb2gray,rgb2lab

results = pd.DataFrame(columns=["Image", "Threshold", "F1", "Jaccard", "TP", "FP", "FN"])
false_negatives = pd.DataFrame(columns=["False_Negative", "Area"])
false_positives = pd.DataFrame(columns=["False_Positive", "Area"])
splits_merges = pd.DataFrame(columns=["Image_Name", "Merges", "Splits"])

for image_name in all_images:
    # Load ground truth data
    img_filename = os.path.join(config_vars["raw_annotations_dir"], image_name)
    ground_truth = skimage.io.imread(img_filename)
    #ground_truth = ground_truth.squeeze()
    if len(ground_truth.shape) == 3:
        ground_truth = rgb2lab(ground_truth)
        ground_truth = ground_truth[:,:,0]
    
    ground_truth = skimage.morphology.label(ground_truth)
    
    # Transform to label matrix
    #ground_truth = skimage.morphology.label(ground_truth)
    
    # Load predictions
    pred_filename = os.path.join(config_vars["labels_out_dir"], image_name)
    prediction = skimage.io.imread(pred_filename)
    
    # Apply object dilation
    if config_vars["object_dilation"] > 0:
        struct = skimage.morphology.square(config_vars["object_dilation"])
        prediction = skimage.morphology.dilation(prediction, struct)
    elif config_vars["object_dilation"] < 0:
        struct = skimage.morphology.square(-config_vars["object_dilation"])
        prediction = skimage.morphology.erosion(prediction, struct)
        
    ####################################################################################    
    #### Testing prediction with no small objects on annot and prediction #####
    #ground_truth = skimage.morphology.remove_small_objects(ground_truth, min_size=100) 
    #prediction = skimage.morphology.remove_small_objects(prediction, min_size=50)
    #####################################################################################
    
    # Relabel objects (cut margin of 30 pixels to make a fair comparison with DeepCell)
    ground_truth = skimage.segmentation.relabel_sequential(ground_truth)[0] #[30:-30,30:-30])[0]
    prediction = skimage.segmentation.relabel_sequential(prediction)[0] #[30:-30,30:-30])[0]
    
    # Compute evaluation metrics
    results = utils.evaluation.compute_af1_results(
        ground_truth, 
        prediction, 
        results, 
        image_name
    )
    
    false_negatives = utils.evaluation.get_false_negatives(
        ground_truth, 
        prediction, 
        false_negatives, 
        image_name
    )
    
    false_positives = utils.evaluation.get_false_positives(
        ground_truth, 
        prediction, 
        false_positives, 
        image_name
    )
    
    splits_merges = utils.evaluation.get_splits_and_merges(
        ground_truth, 
        prediction, 
        splits_merges, 
        image_name
    )
    
    # Display an example image
    #if image_name == all_images[0]:
    show(ground_truth, prediction, image_name=image_name)

```

#### Display accuracy results

```python
average_performance = results.groupby("Threshold").mean().reset_index()

R = results.groupby("Image").mean().reset_index()
g = sb.jointplot(data=R[R["F1"] > 0.4], x="Jaccard", y="F1")

average_performance
R.sort_values(by="F1",ascending=False)
```


#### Plot accuracy results
```python
sb.regplot(data=average_performance, x="Threshold", y="F1", order=3, ci=None)
average_performance
```


#### Compute and print Average F1
```python
average_F1_score = average_performance["F1"].mean()
jaccard_index = average_performance["Jaccard"].mean()
print("Average F1 score:", average_F1_score)
print("Jaccard index:", jaccard_index)
```
#### Summarize False Negatives by area
```python
false_negatives = false_negatives[false_negatives["False_Negative"] == 1]

false_negatives.groupby(
    pd.cut(
        false_negatives["Area"], 
        [0,250,625,900,10000], # Area intervals
        labels=["Tiny nuclei","Small nuclei","Normal nuclei","Large nuclei"],
    )
)["False_Negative"].sum()
```

#### Summarize splits and merges
```python
print("Splits:",np.sum(splits_merges["Splits"]))
print("Merges:",np.sum(splits_merges["Merges"]))
```
#### Report false positives
```python
print("Extra objects (false postives):",results[results["Threshold"].round(3) == 0.7].sum()["FP"])
```

### config.py

```python
import os
import utils.dirtools

config_vars = {}

# ************ 01 ************ #
# ****** PREPROCESSING ******* #
# **************************** #

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 01.01 INPUT DIRECTORIES AND FILES

config_vars["home_folder"] = '/home/maloua/Malou_Master/5_Models'

config_vars["root_directory"] = config_vars["home_folder"] + '/2_Final_Models/data/'

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 01.02 DATA PARTITION INFO

## Maximum number of training images (use 0 for all)
config_vars["max_training_images"] = 0

## Generate partitions?
## If False, load predefined partitions (training.txt, validation.txt and test.txt)
config_vars["create_split_files"] = False

## Randomly choose training and validation images.
## The remaining fraction is reserved for test images.
config_vars["training_fraction"] = 0.5
config_vars["validation_fraction"] = 0.25

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 01.03 IMAGE STORAGE OPTIONS

## Transform gray scale TIF images to PNG
config_vars["transform_images_to_PNG"] = True
config_vars["pixel_depth"] = 8

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 01.04 PRE-PROCESSING OF ANNOTATIONS

## Area of minimun object in pixels
config_vars["min_nucleus_size"] = 10

## Pixels of the boundary (min 2 pixels)
config_vars["boundary_size"] = 2

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 01.05 DATA AUGMENTATION USING ELASTIC DEFORMATIONS

## Elastic deformation takes a lot of times to compute. 
## It is computed only once in the preprocessing. 
config_vars["augment_images"] =  False

## Augmentation parameters. 
## Calibrate parameters using the 00-elastic-deformation.ipynb
config_vars["elastic_points"] = 16
config_vars["elastic_distortion"] = 5

## Number of augmented images
config_vars["elastic_augmentations"] = 10


# ************ 02 ************ #
# ********* TRAINING ********* #
# **************************** #

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 02.01 OPTIMIZATION

config_vars["learning_rate"] = 1e-4

config_vars["epochs"] = 15

config_vars["steps_per_epoch"] = 500

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 02.02 BATCHES

config_vars["batch_size"] = 10

config_vars["val_batch_size"] = 10

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# 02.03 DATA NORMALIZATION

config_vars["rescale_labels"] = True

config_vars["crop_size"] = 256

# ************ 03 ************ #
# ******** PREDICTION ******** #
# **************************** #

config_vars["cell_min_size"] = 16

config_vars["boundary_boost_factor"] = 1

# ************ 04 ************ #
# ******** EVALUATION ******** #
# **************************** #

config_vars["object_dilation"] = 3

# **************************** #
# ******** FINAL SETUP ******* #
# **************************** #

config_vars = utils.dirtools.setup_working_directories(config_vars)

```

### 06-resize-images.py

```python
import skimage
from skimage import io
from matplotlib import pyplot as plt
from skimage.transform import resize
import os
from skimage import img_as_ubyte
import skimage.segmentation
import numpy as np

def create_boundary_label(im):
    
    # strip the first channel
    #print(len(im.shape))
    if len(im.shape)>2:
        if im.shape[2]!=3:
            annot = im[:,:,0]
        else:
            im = rgb2lab(annot)
            im = annot[:,:,0]
    # label the annotations nicely to prepare for future filtering operation
    
    im = skimage.morphology.label(im)
    #print(np.unique(im))
    # find boundaries
    boundaries = skimage.segmentation.find_boundaries(im, mode = 'outer')

    
    label_binary = np.zeros((im.shape + (3,)))
    # write binary label
    label_binary[(im == 0) & (boundaries == 0), 0] = 1
    label_binary[(im != 0) & (boundaries == 0), 1] = 1
    label_binary[boundaries == 1, 2] = 1
    #print(np.unique(label_binary.reshape(-1, merged.shape[2]), axis=0))
    label_binary = img_as_ubyte(label_binary)
    return(label_binary)

imagefile = open(config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/11_training.txt')
filelist = []
for line in imagefile:
    line = line.rstrip()
    filelist.append(line)
imagefile.close()
imagepath = config_vars["home_folder"] + '/2_Final_Models/data/norm_images/'
labelpath = config_vars["home_folder"] + '/3_data/2_additional_datasets/2_BBBC_image_sets/BBBC038_raw_annotations/'
boundarypath = config_vars["home_folder"] + '/2_Final_Models/data/boundary_labels/'
for image in filelist:
    im = skimage.io.imread(imagepath + image)
    labelim = skimage.io.imread(labelpath + image)
    im = resize(im, (1104,1104))
    im = img_as_ubyte(im)
    labelim = resize(labelim, (1104,1104))
    labelim = img_as_ubyte(labelim)
    boundaryim = (create_boundary_label(labelim))
    plt.imshow(im)
    plt.show()
    plt.imshow(boundaryim)
    plt.show()
    skimage.io.imsave(imagepath + 'resized_' + image , im)  
    skimage.io.imsave(boundarypath + 'resized_'+ image, boundaryim)

```

### Relabeling_1.py
```python

import numpy as np
import pandas as pd
import skimage
from skimage import io
import skimage.segmentation
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from skimage import img_as_ubyte
import numpy as np
from scipy.ndimage import distance_transform_edt

## Source code from module
def expand_labels(label_image, distance=1):
    """Expand labels in label image by ``distance`` pixels without overlapping.
    Given a label image, ``expand_labels`` grows label regions (connected components)
    outwards by up to ``distance`` pixels without overflowing into neighboring regions.
    More specifically, each background pixel that is within Euclidean distance
    of <= ``distance`` pixels of a connected component is assigned the label of that
    connected component.
    Where multiple connected components are within ``distance`` pixels of a background
    pixel, the label value of the closest connected component will be assigned (see
    Notes for the case of multiple labels at equal distance).
    Parameters
    ----------
    label_image : ndarray of dtype int
        label image
    distance : float
        Euclidean distance in pixels by which to grow the labels. Default is one.
    Returns
    -------
    enlarged_labels : ndarray of dtype int
        Labeled array, where all connected regions have been enlarged
    Notes
    -----
    Where labels are spaced more than ``distance`` pixels are apart, this is
    equivalent to a morphological dilation with a disc or hyperball of radius ``distance``.
    However, in contrast to a morphological dilation, ``expand_labels`` will
    not expand a label region into a neighboring region.  
    This implementation of ``expand_labels`` is derived from CellProfiler [1]_, where
    it is known as module "IdentifySecondaryObjects (Distance-N)" [2]_.
    There is an important edge case when a pixel has the same distance to
    multiple regions, as it is not defined which region expands into that
    space. Here, the exact behavior depends on the upstream implementation
    of ``scipy.ndimage.distance_transform_edt``.
    See Also
    --------
    :func:`skimage.measure.label`, :func:`skimage.segmentation.watershed`, :func:`skimage.morphology.dilation`
    References
    ----------
    .. [1] https://cellprofiler.org
    .. [2] https://github.com/CellProfiler/CellProfiler/blob/082930ea95add7b72243a4fa3d39ae5145995e9c/cellprofiler/modules/identifysecondaryobjects.py#L559
    Examples
    --------
    >>> labels = np.array([0, 1, 0, 0, 0, 0, 2])
    >>> expand_labels(labels, distance=1)
    array([1, 1, 1, 0, 0, 2, 2])
    Labels will not overwrite each other:
    >>> expand_labels(labels, distance=3)
    array([1, 1, 1, 1, 2, 2, 2])
    In case of ties, behavior is undefined, but currently resolves to the
    label closest to ``(0,) * ndim`` in lexicographical order.
    >>> labels_tied = np.array([0, 1, 0, 2, 0])
    >>> expand_labels(labels_tied, 1)
    array([1, 1, 1, 2, 2])
    >>> labels2d = np.array(
    ...     [[0, 1, 0, 0],
    ...      [2, 0, 0, 0],
    ...      [0, 3, 0, 0]]
    ... )
    >>> expand_labels(labels2d, 1)
    array([[2, 1, 1, 0],
           [2, 2, 0, 0],
           [2, 3, 3, 0]])
    """

    distances, nearest_label_coords = distance_transform_edt(
        label_image == 0, return_indices=True
    )
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out


import sys
np.set_printoptions(threshold=sys.maxsize)

# GET ANNOTATION
with open(config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/filelist_all_aits_annotated_images.txt','r') as O1:
    
    for image in O1:
        image = image.rstrip()
        annot = skimage.io.imread(config_vars["home_folder"] + '/2_Final_Models/data/1_raw_annotations/' + image)

        # strip the first channel
        if annot.shape[2]!=3:
            annot = annot[:,:,0]
        else:
            annot = rgb2lab(annot)
            annot = annot[:,:,0]
        # label the annotations nicely to prepare for future filtering operation

        annot = skimage.morphology.label(annot)
        annot = annot.astype(np.uint8)
        annot = expand_labels(annot)
        # find boundaries
        boundaries = skimage.segmentation.find_boundaries(annot, background = 0).astype(np.uint8)

        # BINARY LABEL
        # prepare buffer for binary label
        label_binary = np.zeros((annot.shape + (3,)))

        # write binary label
        label_binary[(annot == 0) & (boundaries == 0), 0] = 1
        label_binary[(annot != 0) & (boundaries == 0), 1] = 1
        label_binary[boundaries == 1, 2] = 1

        label_binary = img_as_ubyte(label_binary)
        
        # save it - converts image to range from 0 to 255
        skimage.io.imsave( config_vars["home_folder"] + '/2_Final_Models/data/boundary_labels/'+ 'relabel_' + image, label_binary)
        
```

### Relabeling_2.py
```python

import numpy as np
import pandas as pd
import skimage
from skimage import io
import skimage.segmentation
import matplotlib.pyplot as plt
from skimage.color import rgb2lab
from skimage import img_as_ubyte
import numpy as np
from scipy.ndimage import distance_transform_edt

import sys
np.set_printoptions(threshold=sys.maxsize)

# GET ANNOTATION
with open(config_vars["home_folder"] + '/2_Final_Models/data/4_filelists/filelist_all_aits_annotated_images.txt','r') as O1:
    
    for image in O1:
        image = image.rstrip()
        annot = skimage.io.imread(config_vars["home_folder"] + '/2_Final_Models/data/1_raw_annotations/' + image)

        # strip the first channel
        if annot.shape[2]!=3:
            annot = annot[:,:,0]
        else:
            annot = rgb2lab(annot)
            annot = annot[:,:,0]
        # label the annotations nicely to prepare for future filtering operation

        annot = skimage.morphology.label(annot)
        annot = annot.astype(np.uint8)
        # find boundaries
        boundaries = skimage.segmentation.find_boundaries(annot, background = 0).astype(np.uint8)

        # BINARY LABEL
        # prepare buffer for binary label
        label_binary = np.zeros((annot.shape + (3,)))

        # write binary label
        label_binary[(annot == 0) & (boundaries == 0), 0] = 1
        label_binary[(annot != 0) & (boundaries == 0), 1] = 1
        label_binary[boundaries == 1, 2] = 1

        label_binary = img_as_ubyte(label_binary)
        
        # save it - converts image to range from 0 to 255
        skimage.io.imsave(config_vars["home_folder"] + '/2_Final_Models/data/boundary_labels/'+ 'relabel2_' + image, label_binary)
        
```


```python

```
