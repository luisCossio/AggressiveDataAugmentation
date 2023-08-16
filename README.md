
# Aggressive Data Augmentation 

This repo serve as an all in place tool for offline data augmentation of three popular methods of Data Augmentation , 
[Copy-Paste](https://arxiv.org/pdf/2012.07177.pdf), [Mosaic](https://arxiv.org/pdf/2004.10934.pdf) and Self-train. 


Each augmentation method can be used by providing a .json file with a coco-like format and the corresponding images. 
This algorithm were implemented to allow easy augmentation of existing dataset. This repository provides a samples dataset for testing porpoises, using images from the COCO dataset. 


# Mosaic Augmentation
This method propose in the [YOLOv4](https://arxiv.org/pdf/2004.10934.pdf) paper, and consist of combining images and their respective labels
by making a mosaic of 4 different images.
![Mosaic example](resources/pixel_transplant_mosaic.jpg)

Labels are accordingly placed in the resulting positions. 


## Usage

To generate a simple dataset from an existing one you can use the next command:
```sh
--path-dataset sample_data/images --path-annotations sample_data/labels.json --path-to-save new_images --ratio-dataset 0.20 --img-size 480 480 --ext .jpg
```

There are three relevant parameters to customize the a dataset. These parameters are:
- ratio-dataset: this allows to define the number of images to be generated, and its propotional to eh size of the original dataset. A ratio-dataset of 1.0 means to generate as many copy-pasted images as the images in the original dataset
- img-size: Size of images to generate
- ext: name of the extension of the images to be used (Ex: .jpg, .png)


Its worth pointing out that it should be avoided the usage of this method in datasets with big objects, since slicing through objects is not implemented. Objects are avoided to be cut in the mosaic process as to not lose data for small size objects. Further problems can be caused by using an image size too big in comparison with the size found in the original dataset, since very few images will match with each other.

# Copy Paste usage
Copy-Paste augmentation is an augmentation method that add's objects from a donor image to a random location in a donor 
image, thus generating new images. Since COCO labels define segments of objects new annotated data
can be generated combining images in this manner. 

![](resources/copy-paste sample.jpg)

Using  copy-paste augmentation we can take an existing dataset and generate new annotations and images to train
object detection models. 


## Usage
To generate a simple dataset from an existing one you can use the next command:
```sh
python CopyPaste/copy_paste.py --path-dataset sample_data/images --path-annotations sample_data/labels.json --path-to-save new_images --ratio-dataset 1.0 --n-samples 1 3 --ext .jpg
```
Notice that this code will generate a folder with images and annotations in the form of a .json file. 
This new file will contain all the original annotations plus the newly generated images so it can be used to train
a model with the full set of images. 




There are multiples parameters that allow to customize the generation of augmented dataset. These parameters are:
- modify-base: To rescale and translate the base image and the donor samples
- ratio-dataset: this allows to define the number of images to be generated, and its propotional to eh size of the original dataset. A ratio-dataset of 1.0 means to generate as many copy-pasted images as the images in the original dataset 
- n-samples: range of donor objects to be pasted in each image. 
- ext: name of the extension of the images to be used (Ex: .jpg, .png)
- relative-augment: If True the augmentation methods are defined relative to the original image. For instance n-samples will paste new object in a donor image relative to the number of objects already in the image. Also the location of the pasted objects will be close to already existing objects in the image. This is done by selecting 2 existing objects and pasting an object in the vicinity of the average position between those objects. 

![](resources/copy-paste sample_relative.jpg)
The next command is an example of how to define a relative augmentation.  
```sh
python CopyPaste/copy_paste.py --path-dataset sample_data/images --path-annotations sample_data/labels.json --path-to-save new_images --modify-base --ratio-dataset 1.0 --relative-augment --n-samples 0.1 0.2 --ext .jpg
```

