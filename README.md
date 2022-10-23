# IMVDA
This is the demo for IMVDA via CEKT.

# Datasets
[Office-31](https://faculty.cc.gatech.edu/~judy/domainadapt/) and [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html) are popular datasets for domain adaptation, which only includes RGB images.

RGB-D and B3DO are public available datasets with RGB and Depth images. Note that the original depth images should be processed and then are fed into the network.

# Dependencies
python==3.6.2  
pytorch==1.0.0  
torchvision==0.2.2  
numpy==1.16.2  
scikit-learn==0.20.2  
scipy==1.1.0  
opencv-python==4.0.0  

# Acknowledgements
The implementation of channel exchange is mainly based on [CEN](https://github.com/yikaiw/CEN). Many thanks to Yikai for previous work.