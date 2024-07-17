# Trackers
This repository includes three types of object trackers, providing a comprehensive suite of tools for various tracking needs.
# Features
- OpenCV Trackers
  - Two distinct object tracking algorithms implemented using OpenCV. BotSort and ByteTrack
- ByteTrack
  - A highly accurate and efficient tracking algorithm.
  - Detailed setup instructions included.
- Deployment
  - Comprehensive guidelines for deploying ByteTrack on ncnn/CPP.
  - Optimized for various resource-constrained devices
# Python OpenCV Trackers
This repository contains Python scripts that demonstrate the use of OpenCV trackers for object tracking in videos. It includes implementations of various tracking algorithms available in OpenCV, showcasing their usage in different scenarios.
# Installation
To use the scripts in this repository, follow these steps:
1. **Copy the ultralytics github repository:**
   `git clone https://github.com/ultralytics/ultralytics`
3. **Create a Conda environment**:
   ```
   conda create --name ultralytics-env python=3.8 -y
   conda activate ultralytics-env
   ```
4. **Install required packages:**
   ```
   conda install -c conda-forge ultralytics
   conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
   ```
5. **Once you have set up the environment, you can use the scripts to experiment with different OpenCV trackers:**
   
   **What you need to modify**:
     - VideoPath (line 10 for PlottingTracksOverTime, line 8 for PersistingTracksLoop):
       ```
       video_path = "video.mp4" #set path to your video file
       cap = cv2.VideoCapture(video_path)
       ```
       - Also you can configure output_path (line 24 for PlottingTracksOverTime, line 13 for PersistingTracksLoop)
         
     - Trackers and Imgsz (line 22 for PersistingTracksLoop, line 36 for PlottingTracksOverTime):
       ```
       results = model.track(frame, persist=True, tracker = "bytetrack.yaml", imgsz = [1080,1920]) #you can choose "botsort.yaml" or "bytetrack.yaml" and different image size
       ```
**PlottingTracksOverTime ByteTrack**
Speed: 6.5ms preprocess, 5.3ms inference, 0.8ms postprocess per image at shape (1, 3, 1088, 1920)
**PersistingTracksLoop ByteTrack**
Speed: 6.4ms preprocess, 5.3ms inference, 0.7ms postprocess per image at shape (1, 3, 1088, 1920)

# ByteTrack

**ByteTrack** is a high-speed and accurate object tracker. You can run inference on your system by following the instructions in the ByteTrack/README.

**To deploy** this tracker on low-resource devices, navigate to ByteTrack/Deploy/ncnn/cpp, where you will also find detailed instructions for running on various types of devices.

You can view the list of supported devices at this link: `https://github.com/Tencent/ncnn/wiki/how-to-build`

