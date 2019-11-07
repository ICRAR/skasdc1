# The ML Solution to SKA SDC1
This repository shares **icrar** team's machine learning solution to the [SKA Science Data Challenge 1](https://astronomers.skatelescope.org/ska-science-data-challenge-1/). The ML solution has earned the team [a second place](https://astronomers.skatelescope.org/ska-science-data-challenge-1-results/) in this data challenge.

## Pre-processing
+ Convert the **raw** catalogues to CSV files
+ Split the entire image into a set _I_ of small (205 by 205 pixel) cutouts
+ Spatially index each image cutout, and manage all indexes in the PostgreSQL database _D_
+ Go througth each "ground-truth" source _S_ in the CSV catalogue
    - Find the cutout _C_ that contains _S_ using its index in _D_
    - Calculate the background noise level _rms_ of _S_
    - Check if the flux of _S_ is greater than _k_ (_k_ = [0.5 to 3]) sigma above _rms_
        - If So, keep _S_ in the **training** catalogue _T_
        - Else, discard _S_
+ Go through each valid source _V_ in _T_
    - Calculate the pix coordinates of its bounding box _B_ based on its sky coordinates encoded in the catalogue
    - Obtain the class label _CL_ for _V_
    - Assemble _B_ and _CL_, together with some other identifiers (e.g. source id)as a valid source record _R_
+ Create the final JSON file _J_ that contains
    - names of all cutout images, each of which has at least one valid source
    - a set of valid source records (many _Rs_)
+ Pass on both _I_ and _J_ to the following machine learning pipeline (see the section below)

## Machine learning
Given _I_ and _J_ for each dataset (e.g. 1000h and B1), we trained [ClaRAN - Classifying Radio Galaxies Automatically with Neural Networks](https://academic.oup.com/mnras/article/482/1/1211/5142869) to detect sources in all cutout images. Particurly, we used [ClaRAN V0.2](https://github.com/chenwuperth/claran), which requires _I_ and _J_ to be organised as in the following directories:
```
SKASDC1/DATA_DIR/
  annotations/
    instances_train_B1_1000h.json
    instances_test_B1_1000h.json
    ...
  train_B1_1000h/
    SKAMid_B1_1000h_v3_train_image*.png
    ...
  val_B1_1000h/
    SKAMid_B1_1000h_v3_train_image*.png
    ...
  
```
All the above data is [publicaly available](https://drive.google.com/open?id=1MV4G0-yOiWNST7D2bw5EwUDvfzCWEiPW). For detailed description of ClaRAN's detection algorithms, please refer to [our paper](https://academic.oup.com/mnras/article/482/1/1211/5142869).

We have also prepared a [Python notebook](https://github.com/ICRAR/skasdc1/blob/master/claran_skasdc1_example.ipynb) that shows the basic steps to get started with training SDC1 datasets (B1, 1000 hours) with ClaRAN v0.2.