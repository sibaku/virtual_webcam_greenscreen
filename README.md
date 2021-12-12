# Virtual Webcam Greenscreen

This project is based on https://github.com/ZHKKKe/MODNet

It will provide you with a virtual webcam, that can be used in other tools, if they do not yet include any background removal functionality. The virtual webcam includes a few features:

* Background blur
* Replace background with color
* Replace background with image
* Replace background with video (not super performant)
  
These are controlled via GUI with a preview image.

This is a quick weekend project, so do not expect very high coding standards.

# Installation

1. Follow the instructions at https://github.com/ZHKKKe/MODNet/tree/master/demo/video_matting/webcam
   1. It is recommended to install torch beforehand with the proper GPU support manually from https://pytorch.org/get-started/locally/. Otherwise the network may run on the CPU, which is probably way too slow
   2. Make sure that you have downloaded the pretrained networks and put it in the pretrained directory
   3. Running the script from this repository works the same way as running the webcam demo from the original repository. You can either just copy the script webcam_virtual_greenscreen.py to the webcam demo folder of MODNet and run it there or go the other way around and copy the contents of the MODNet repository next to this script.
2. Install the requirements listed in requirements_virtual_greenscreen.txt: ```pip install -r requirements_virtual_greenscreen.txt```
3. Follow the instructions listed for the pyvirtualcam package at https://github.com/letmaik/pyvirtualcam under the heading **Supported virtual cameras**

Once everything is installed, you can start the script with ```python webcam_virtual_greenscreen.py``` (assuming the script lies next to the MODNET src directory)

# License

As this code is based on the MODNet code, it is also licensed under the [Creative Commons Attribution NonCommercial ShareAlike 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license.