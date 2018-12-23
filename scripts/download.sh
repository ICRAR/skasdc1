#!/bin/bash
# credit - https://gist.github.com/nudomarinero/95e903c8f79d02aceed8c1b2c05222d8
# Images
# changed with V2 released by SKAO
wget https://owncloud.ia2.inaf.it/index.php/s/AKM2CUAn1hQhtvO/download -O SKAMid_B1_8h_v2.fits
wget https://owncloud.ia2.inaf.it/index.php/s/7ihk8de9ILYY2dc/download -O SKAMid_B1_100h_v2.fits
wget https://owncloud.ia2.inaf.it/index.php/s/P9YaJVOROTqdKkF/download -O SKAMid_B1_1000h_v2.fits
wget https://owncloud.ia2.inaf.it/index.php/s/BUfFtgQVwfVsvnt/download -O SKAMid_B2_8h_v2.fits
wget https://owncloud.ia2.inaf.it/index.php/s/aASQCoevae5W6aI/download -O SKAMid_B2_100h_v2.fits
wget https://owncloud.ia2.inaf.it/index.php/s/pgEhVTTVs0rkVoV/download -O SKAMid_B2_1000h_v2.fits
wget https://owncloud.ia2.inaf.it/index.php/s/UTlfPUCddWYlZFB/download -O SKAMid_B5_8h_v2.fits
wget https://owncloud.ia2.inaf.it/index.php/s/t7aoXFbkH4fsmy1/download -O SKAMid_B5_100h_v2.fits
wget https://owncloud.ia2.inaf.it/index.php/s/ldjHXaZToppJLdh/download -O SKAMid_B5_1000h_v2.fits

# Ancillary data
wget https://owncloud.ia2.inaf.it/index.php/s/ZbaSDe7zGBYgxL1/download -O PrimaryBeam_B1.fits
wget https://owncloud.ia2.inaf.it/index.php/s/cwzf1BO2pyg9TVv/download -O SynthesBeam_B1.fits
wget https://owncloud.ia2.inaf.it/index.php/s/tVGse9GaLBQmntc/download -O PrimaryBeam_B2.fits
wget https://owncloud.ia2.inaf.it/index.php/s/tAxdh3x57JLMPam/download -O SynthesBeam_B2.fits
wget https://owncloud.ia2.inaf.it/index.php/s/HlEJNsN2Vd4RL9W/download -O PrimaryBeam_B5.fits
wget https://owncloud.ia2.inaf.it/index.php/s/zppNCe5SM3PPkh9/download -O SynthesBeam_B5.fits

# Training sets
wget https://owncloud.ia2.inaf.it/index.php/s/iF4fNZxQcPSKPyk/download -O TrainingSet_B1.txt
wget https://owncloud.ia2.inaf.it/index.php/s/R4Kw8RYFoOR5AfM/download -O TrainingSet_B2.txt
wget https://owncloud.ia2.inaf.it/index.php/s/agY9Vindrm07rJe/download -O TrainingSet_B5.txt
