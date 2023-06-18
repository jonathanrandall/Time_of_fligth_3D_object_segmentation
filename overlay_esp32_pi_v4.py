# before you run this need to install the below
#pip3 install -U scikit-learn
#pip install matplotlib

import ArducamDepthCamera as ac
import requests

import numpy as np
import cv2
import threading
import queue
import time

from time import sleep
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


AWB = True

MAX_DISTANCE = 4


COLOURS = [
    tuple(int(colour_hex.strip('#')[i:i+2], 16) for i in (0, 2, 4))
    for colour_hex in plt.rcParams['axes.prop_cycle'].by_key()['color']
]

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


def do_kmeans(filtered_image, n_samples = 5):
    pixels = filtered_image.reshape(-1, 1)

    np.random.seed(42)

    # Define the size of the random sample
    sample_size = 3000

    # Take a random sample from the pixels array
    random_sample = np.random.choice(pixels.flatten(), size=sample_size)
    random_sample=random_sample.reshape(-1,1)


    # Perform k-means clustering
    k = n_samples  # Number of clusters
    kmeans = KMeans(n_clusters=k, random_state=0,n_init=10)
    kmeans.fit(random_sample)#pixels)
    
    cluster_centers = kmeans.cluster_centers_
    clusters = kmeans.predict(pixels)
    

    clustered_image = cluster_centers[clusters].reshape(filtered_image.shape)
    # Convert the spread image to 8-bit for visualization
    clustered_image = np.uint8(clustered_image)
    dists = ((255-cluster_centers)*4/255)
    return clustered_image, cluster_centers


def get_segemntation(clustered_image, cluster_centers,min_region_size=1000):
    num_grps = len(cluster_centers)
    gray_level_ranges = [(*i-1, *i+1) for i in cluster_centers]
    
    segmentation_masks = [] #np.zeros_like(clustered_image, dtype=np.uint8)
    stats = []
    lbls = []
    for i, (min_level, max_level) in enumerate(gray_level_ranges):
        segmentation_mask = np.zeros_like(clustered_image, dtype=bool)#np.uint8)
        # Apply thresholding to create a binary image for the current region
        _, binary_image = cv2.threshold(clustered_image, min_level, 255, cv2.THRESH_BINARY)
        _, upper_thresholded = cv2.threshold(clustered_image, max_level, 255, cv2.THRESH_BINARY_INV)
        binary_image = cv2.bitwise_and(binary_image, upper_thresholded)

        # Apply connected component labeling
        
        num_labels_, labels_, stats_, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        

        # Assign a unique label to the corresponding pixels in the segmentation masks
        p = 1
        for p, st in enumerate(stats_):# range(q, q+num_labels_-1):
            if p==0:
                continue        
            elif st[cv2.CC_STAT_AREA] < min_region_size:
                segmentation_mask[labels_ == p] = 0
                
            else:
                stats.append(st)
                segmentation_mask[labels_ == p] =True #cluster_centers[i]
                segmentation_masks.append(segmentation_mask)
                lbls.append(cluster_centers[i]) #(255.0-cluster_centers[i])*400.0/255.0)

    sorted_index = sorted(range(len(lbls)), key=lambda k: lbls[k])

    # Sort l1
    lbls = sorted(lbls)

    # Reorder l2 and l3 using the sorted index
    stats = [stats[i] for i in sorted_index]
    segmentation_masks = [segmentation_masks[i] for i in sorted_index]    
    return lbls,stats,segmentation_masks
        
    
def draw_instance_segmentation_masks(img, masks):
    ''' Draws coloured polygons masks over img '''
    filled = np.zeros_like(img)
    # image_ret = img.copy()
    
    for i, mask in enumerate(masks):
        if i==(len(masks) -1):
            filled[mask] = COLOURS[i]
    # Blend original and filled into a composite image
    cv2.addWeighted(img, 0.75, filled, 0.45, 0.0, dst=img)
    
def draw_detections(img, stts, colours=COLOURS):
    for i, (tlx, tly, w, h, d) in enumerate(stts):
        if i== len(stts)-1:
            i %= len(colours)
            cv2.rectangle(img, (tlx, tly), (tlx+w, tly+h), color=colours[i], thickness=2)
            
def annotate_distance(img, dist, stts, colours=COLOURS):
    for i, (tlx, tly, w, h, d) in enumerate(stts):
        if i == len(stts)-1:
            
            txt = f'{dist[i][0]:1.1f} cm'

            offset = 1

            # cv2.rectangle(img, 
            #               (tlx+offset, tly+offset+12),
            #               (tlx+offset+len(txt)*12, tly),
            #               color=colours[i],
            #               thickness=cv2.FILLED)

            ff = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(img, txt, (tlx, tly+12), fontFace=ff, fontScale=1.0, color=(255,)*3)
        

selectRect = UserRect()

followRect = UserRect()




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
    #adjust tantheta to shift either way.
    tantheta = 0.51#648732789907391
    dc = 4.0 #cm distance between cameras
    set_resolution(URL_cam, index=5)
    cap_esp32 = cv2.VideoCapture(URL_cam + ":81/stream")
    cap_esp32.release()
    set_resolution(URL_cam, index=5)
    cap_esp32 = VideoCapture(URL_cam + ":81/stream")
    i=40
    while True:
        #frame_esp32 = cap_esp32.read()
        if cap_esp32.cap.isOpened():
            #ret_esp32, frame_esp32 = cap_esp32.read()
            frame_esp32 = cap_esp32.read()
            cv2.imshow("frame_esp32", frame_esp32)
            
            xx = frame_esp32.shape[1]
            yy = frame_esp32.shape[0]
            #print(xx, yy)



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
            #do a median filter. 
            result_image = cv2.medianBlur(result_image, 5) 
            #result_image = cv2.applyColorMap(result_image, cv2.COLORMAP_JET)
            clustered_image, cluster_centers = do_kmeans(result_image, n_samples = 5)
            lbls,stats,segmentation_masks = get_segemntation(clustered_image, cluster_centers)
            
            distance = ((255.0-np.array(lbls))*400.0/255.0)#-2.0) #add 0.001 to hack divide by zero error
            P = (240*dc/(2*tantheta))/distance
    
            cv2.imshow("preview",clustered_image)
            
            
            frame_esp32 = cv2.resize(frame_esp32, (240, 180),interpolation=cv2.INTER_CUBIC)
            print(xx, yy)
            shifted_masks = []
            shifted_stats = []
            yn = 180
            xn = 240
            for i, p_val in enumerate(P):
                st = np.copy(stats[i])
                st[0] = st[0]+int(p_val)
                st[2] = min(xx, st[2])
                #tack this onto the left side of the mask
                left_side = np.zeros((yn,int(p_val)), dtype=bool)
                s_mask = np.concatenate((left_side,segmentation_masks[i]),axis=1)
                s_mask = s_mask[:yn,:xn]
                shifted_masks.append(s_mask)
                shifted_stats.append(st)

            img_out = np.copy(frame_esp32)
            draw_instance_segmentation_masks(img_out, shifted_masks)
            draw_detections(img_out,shifted_stats)
            annotate_distance(img_out, distance, shifted_stats)
            combined_image = cv2.resize(img_out, (xx*2, yy*2),interpolation=cv2.INTER_CUBIC)
            cv2.imshow("combined", combined_image) 

        key = cv2.waitKey(1)
        if key == ord("q"):
            exit_ = True
            cam.stop()
            cam.close()
            
            cv2.destroyAllWindows()
            cap_esp32.cap.release()
            #sys.exit(0)
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
