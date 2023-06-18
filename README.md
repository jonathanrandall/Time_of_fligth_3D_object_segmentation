# Arducam Time of Flight Camera for 3d object segmentation
 3d object segmentation with arducam time of flight

This project combines an Arducam Time of flight depth camera with an esp32-cam to do 3-dimensional segmentation and distance estimation. I've used a raspberry pi 4B to capture and process the images from the depth and the esp32-cam. <br>
The Arducam depth camera can be bought at the below site (also available on Amazon).<br>
https://www.arducam.com/product/time-of-flight-camera-for-raspberry-pi/


#### To setup the arducam camera follow the instruction here:
https://github.com/ArduCAM/Arducam_tof_camera 

Note: setting up on bullseye, you can run:<br>
`sudo pip install opencv-python ArducamDepthCamera`

 The **.py** files are as follows:<br>
 **preview_jr.py**: file for previewing arducam output.<br>
 **overlay_esp32_pi.py**:file for doing naive overlay of depth image onto esp32-cam image. <br>
 **overlay_esp32_pi_v4**: file for doing overlay of depth image with shift, lebeling and bounding boxes onto esp32-cam image. May need to adjust the 
 **tantheta**

 The **notebooks** are:<br>
 **pre_processing_test_v3.ipynb**: is the development notebook for processing the gray-level output of the depth camera.
 **overlay_images_v3.ipynb**: is the development notebook for applying the segmentation mask and shift to the esp32-cam image.
