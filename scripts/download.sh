#!/bin/bash
# credit - https://gist.github.com/nudomarinero/95e903c8f79d02aceed8c1b2c05222d8
# Images
wget https://owncloud.ia2.inaf.it/index.php/s/UDiKqvuscDQViEt/download -O SKAMid_B1_8h.fits
wget https://owncloud.ia2.inaf.it/index.php/s/2nTtGBWpzYJ5Ghr/download -O SKAMid_B1_100h.fits
wget https://owncloud.ia2.inaf.it/index.php/s/DUNztYKW0PSlNH5/download -O SKAMid_B1_1000h.fits
wget https://owncloud.ia2.inaf.it/index.php/s/zEW4JsefeMycR8p/download -O SKAMid_B2_8h.fits
wget https://owncloud.ia2.inaf.it/index.php/s/8K750vop1yjZZXE/download -O SKAMid_B2_100h.fits
wget https://owncloud.ia2.inaf.it/index.php/s/A7XjSZ3n1Kr57TV/download -O SKAMid_B2_1000h.fits
wget https://owncloud.ia2.inaf.it/index.php/s/Uu7yhRjCYknle54/download -O SKAMid_B5_8h.fits
wget https://owncloud.ia2.inaf.it/index.php/s/7lEeXlNvoYFmEfG/download -O SKAMid_B5_100h.fits
wget https://owncloud.ia2.inaf.it/index.php/s/Wo7gqyfgpCqj2XY/download -O SKAMid_B5_1000h.fits

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
