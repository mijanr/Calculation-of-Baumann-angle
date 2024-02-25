# Calculation of Baumann angle using deep learning methods.

## Description
Baumann angle, also known
as the humeral-capitellar angle, is measured for the evaluation of
the displacement of the humerus. An angle between  150-165° is considered normal. 
A deep learning method is used to calculate the Baumann angle from X-ray images.


## Dataset
The dataset used is publicly available [Musculoskeletal Radiographs (MURA)](https://stanfordmlgroup.github.io/competitions/mura/)  dataset from Standford ML group. The MURA dataset consist of X-ray images of elbows. The objective is to calculate the Baumann angle from the X-ray images. The images are not annotated by default and looks like this:
![MURA dataset](imgs/image1.png)
![MURA dataset](imgs/image2.png)
So, the images are labelled with [Labelme](https://github.com/wkentaro/labelme) for 8 distinct points. 
Those are: Humerus, Posterior Border Line, Anterior Border Line, Proximal Shaft Intersection, Distal Shaft Intersection, Shaft Centerline, Tangent, Articular Block. 
Once the images are annotated, and two lines are draw, and the angle between them is calculated. The angle is the Baumann angle. The annotated images look like this:

![Labeled images](imgs/labeled_image.PNG)
![MURA dataset](imgs/labeled_img.png)

## Model
A regression based approach is taken to accomplish the task. A vanilla CNN network is used to perform some experiments, and them a [ResNet50](https://arxiv.org/abs/1512.03385) is used for final training and prediction purpose. 

## Results
The predictions made by the model are as follows:

![Predictions](imgs/test_1.png)
![Predictions](imgs/test_5.png)
![Predictions](imgs/test_6.png)
![Predictions](imgs/test_7.png)



## Libraries used
- numpy
- pandas
- matplotlib
- PyTorch
- torchvision
- PIL
- sklearn
- albumentations

## Authors
- [Md Mijanur Rahman](https://github.com/mijanr)



