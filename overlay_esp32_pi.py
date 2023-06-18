import ArducamDepthCamera as ac
import requests

import numpy as np
import cv2
import threading
import queue
import time

from time import sleep

AWB = True

MAX_DISTANCE = 4

def process_frame(depth_buf: np.ndarray, amplitude_buf: np.ndarray) -> np.ndarray:
        
    depth_buf = np.nan_to_num(depth_buf)

    amplitude_buf[amplitude_buf<=7] = 0
    amplitude_buf[amplitude_buf>7] = 255

    depth_buf =(1 - (depth_buf/MAX_DISTANCE)) * 255
    depth_buf = np.clip(depth_buf, 0, 255)
    result_frame = depth_buf.astype(np.uint8)  & amplitude_buf.astype(np.uint8)
    return result_frame 

class UserRect():
    def __init__(self) -> None:
        self.start_x = 0
        self.start_y = 0
        self.end_x = 0
        self.end_y = 0

selectRect = UserRect()

followRect = UserRect()

def on_mouse(event, x, y, flags, param):
    global selectRect,followRect
    
    if event == cv2.EVENT_LBUTTONDOWN:
        pass

    elif event == cv2.EVENT_LBUTTONUP:
        selectRect.start_x = x - 4 if x - 4 > 0 else 0
        selectRect.start_y = y - 4 if y - 4 > 0 else 0
        selectRect.end_x = x + 4 if x + 4 < 240 else 240
        selectRect.end_y=  y + 4 if y + 4 < 180 else 180
    else:
        followRect.start_x = x - 4 if x - 4 > 0 else 0
        followRect.start_y = y - 4 if y - 4 > 0 else 0
        followRect.end_x = x + 4 if x + 4 < 240 else 240
        followRect.end_y = y + 4 if y + 4 < 180 else 180
        
def usage(argv0):
    print("Usage: python "+argv0+" [options]")
    print("Available options are:")
    print(" -d        Choose the video to use")


class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()
#functions for the command handler

def set_resolution(url: str, index: int=1, verbose: bool=False):
    try:
        if verbose:
            resolutions = "10: UXGA(1600x1200)\n9: SXGA(1280x1024)\n8: XGA(1024x768)\n7: SVGA(800x600)\n6: VGA(640x480)\n5: CIF(400x296)\n4: QVGA(320x240)\n3: HQVGA(240x176)\n0: QQVGA(160x120)"
            print("available resolutions\n{}".format(resolutions))

        if index in [10, 9, 8, 7, 6, 5, 4, 3, 0]:
            requests.get(url + "/control?var=framesize&val={}".format(index))
        else:
            print("Wrong index")
    except:
        print("SET_RESOLUTION: something went wrong")

def set_quality(url: str, value: int=1, verbose: bool=False):
    try:
        if value >= 10 and value <=63:
            requests.get(url + "/control?var=quality&val={}".format(value))
    except:
        print("SET_QUALITY: something went wrong")

def set_awb(url: str, awb: int=1):
    try:
        awb = not awb
        requests.get(url + "/control?var=awb&val={}".format(1 if awb else 0))
    except:
        print("SET_QUALITY: something went wrong")
    return awb

#focal length. Pre-calibrated in stereo_image_v6 notebook
if __name__ == "__main__":
    cam = ac.ArducamCamera()
    if cam.open(ac.TOFConnect.CSI,0) != 0 :
        print("initialization failed")
    if cam.start(ac.TOFOutput.DEPTH) != 0 :
        print("Failed to start camera")
    cam.setControl(ac.TOFControl.RANG,MAX_DISTANCE)
    cv2.namedWindow("preview", cv2.WINDOW_AUTOSIZE)
    #cv2.setMouseCallback("preview",on_mouse)
    
    URL_cam = "http://192.168.1.181"

    fl = 2.043636363636363
    tantheta = 0.7648732789907391-0.1
    set_resolution(URL_cam, index=5)
    cap_esp32 = cv2.VideoCapture(URL_cam + ":81/stream")
    cap_esp32.release()
    set_resolution(URL_cam, index=5)
    cap_esp32 = VideoCapture(URL_cam + ":81/stream")
    i=20
    while True:
        #frame_esp32 = cap_esp32.read()
        if cap_esp32.cap.isOpened():
            #ret_esp32, frame_esp32 = cap_esp32.read()
            frame_esp32 = cap_esp32.read()
            cv2.imshow("frame_esp32", frame_esp32)



            #if ret_esp32:
            #    cv2.imshow("frame_esp32", frame_esp32) 
            #else:
            #    frame_esp32.release()
            #    cap_esp32 = cv2.VideoCapture(URL_cam + ":81/stream")
        
        
        frame = cam.requestFrame(200)
        if frame != None:
            depth_buf = frame.getDepthData()
            amplitude_buf = frame.getAmplitudeData()
            cam.releaseFrame(frame)
            amplitude_buf*=(255/1024)
            amplitude_buf = np.clip(amplitude_buf, 0, 255)

            cv2.imshow("preview_amplitude", amplitude_buf.astype(np.uint8))
            
            result_image = process_frame(depth_buf,amplitude_buf)
            #result_image = cv2.applyColorMap(result_image, cv2.COLORMAP_JET)
            cv2.rectangle(result_image,(selectRect.start_x,selectRect.start_y),(selectRect.end_x,selectRect.end_y),(128,128,128), 1)
            cv2.rectangle(result_image,(followRect.start_x,followRect.start_y),(followRect.end_x,followRect.end_y),(255,255,255), 1)
    
            cv2.imshow("preview",result_image)
            xx = frame_esp32.shape[1]
            yy = frame_esp32.shape[0]
            
            resized_image = cv2.resize(result_image, (xx, yy),interpolation=cv2.INTER_CUBIC)

            # Convert grayscale image to color
            alpha = 0.5
            beta = 0.5
            color_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2BGR)
            
            combined_image = cv2.addWeighted(frame_esp32, alpha, color_image, beta, 0)
            
            cv2.imshow("combined", combined_image) 

        key = cv2.waitKey(1)
        if key == ord("q"):
            exit_ = True
            cam.stop()
            cam.close()
            
            cv2.destroyAllWindows()
            cap_esp32.cap.release()
            sys.exit(0)
        if key == ord("s"):
            i=i+1
            fname = 'outputs/tof_'+str(i) + '.jpg'
            fname2 = 'outputs/esp32_'+str(i) + '.jpg'
            fname3 = 'outputs/combined_'+str(i) + '.jpg'
            cv2.imwrite(fname, result_image)
            cv2.imwrite(fname2, frame_esp32)
            cv2.imwrite(fname3, combined_image)
        
        key = cv2.waitKey(1) & 0xFF  # Wait for a key press (1ms delay)

        if key == ord('q'):  # If the 'q' key is pressed
            cv2.destroyAllWindows()
            cap_esp32.release()
            break  # Break out of the loop

    cv2.destroyAllWindows()
    cap_esp32.cap.release()
