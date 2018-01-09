Deep photometric stereo network
===============================

This repository is an implementation of Deep Photometric Stereo Network.
(http://openaccess.thecvf.com/content_ICCV_2017_workshops/w9/html/Santo_Deep_Photometric_Stereo_ICCV_2017_paper.html)

How to Train
============
We use the deep learning framework **Tensorflow** with following libraries:
 * Numpy
 * cv2
 * tqdm
 * Boost.Numpy (https://github.com/ndarray/Boost.NumPy)

We use python 2.7 on Ubuntu 14.04. You can use our ``Dockerfile``.


Download datasets
-----------------
We use following dataset for the training and evaluation.
* [Blobby Shape Dataset](http://www.mit.edu/~kimo/blobs/)
* [MERL BRDF Database](https://www.merl.com/brdf/)
* [DiLiGenT Photometric Stereo Dataset](https://sites.google.com/site/photometricstereodata/) (Optional)

You can download each file by ``download_*.sh``.
DiLiGenT is only used for evaluation.


``params.py``
-------------
This file defines paths of each dataset and the light source directions.
Now the light source directions are fit to DiLiGenT dataset. You can modify this values for your setup.

Also, the path to save the training images are defined here.

Rendering training data
-----------------------
First, you need to build:
```
$ cd ./merl_brdf_database
$ cmake .
$ make
```
This is because we use ``BRDFRead.cpp`` to read MERL BRDF Database, which is the sample code in that project.

You can render synthetic training and test data by:
```
$ python renderin_with_merl.py
```
The training and test data are output to the specified path in ``params.py``.

Preparing training data
-----------------------
We use ``TFRecord`` format for training data.
You can convert rendered images to the ``TFRecord`` file by:

```
$ python dataset.py
```

Training
--------
```
$ python train.py --output_path PATH_TO_SAVE_MODEL --gpu GPU_ID
```
Other arguments can be confirmed by ``--help`` option.


Directory tree of Model
-----------------------
``PATH_TO_SAVE_MODEL`` has following directories:

### ``summary``
Summary for tensorboard
 * ``{train|test}/cost`` : Output of loss function
 * ``{train|test}/RMSE`` : Root Mean Squared Error between ground truth and predicted normal vector

### ``checkpoint``
Checkpoint files

### ``best_checkpoint``
Best checkpoint file.
"Best" means that minimize the L_2 loss for synthetic test data.

### ``eval``
Estimated images for synthetic test data.

Testing for DiLiGenT
====================

We will add testing code for DiLiGenT dataset.