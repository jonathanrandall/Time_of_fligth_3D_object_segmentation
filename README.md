# Arducam Time of Flight Camera for 3d object segmentation
 3d object segmentation with arducam time of flight

 The **.py** files are as follows:<br>
 `preview_jr.py`: file for previewing arducam output.<br>
 `overlay_esp32_pi.py`:file for doing naive overlay of depth image onto esp32-cam image. <br>
 `overlay_esp32_pi_v4`: file for doing overlay of depth image with shift, lebeling and bounding boxes onto esp32-cam image. May need to adjust the 
 **tantheta**

 The **notebooks** are:<br>
 `pre_processing_test_v3.ipynb`: is the development notebook for processing the gray-level output of the depth camera.
 `overlay_images_v3.ipynb`: is the development notebook for applying the segmentation mask and shift to the esp32-cam image.
