# PPT Paper and Dataset Research

Source PPT:
- `C:\Users\aaron\Desktop\URAP\URAP-UAV-to-UAV-Detection-and-Tracking\doc\Copy of Moving Object Detection from Moving Platform.pptx`
- Extracted text: `C:\Users\aaron\Desktop\URAP\artifacts\ppt_extract\slides_text.txt`

## Papers explicitly mentioned in the PPT

### 1. Advanced Aerial Monitoring and Tracking System: YOLOv4 and DeepSORT Integration for Drone-Based Surveillance
- PPT location: slide 2-4
- What the slide says: YOLOv4 + DeepSORT, classes include light motor vehicles and heavy motor vehicles.
- Dataset used:
  - Public metadata confirms the paper exists, but the accessible abstract/metadata I could retrieve does not expose a precise dataset name.
  - Based on the class names shown in the PPT (`Light Motor Vehicles`, `Heavy Motor Vehicles`), this appears to be a custom aerial traffic / surveillance dataset rather than a standard anti-UAV benchmark.
- Dataset characteristics:
  - Aerial-view surveillance setting.
  - At least two vehicle classes.
  - Likely evaluated on real video rather than still-image-only data because the paper integrates DeepSORT tracking.
- Confidence: low for exact dataset identity; the public abstract is insufficient.

### 2. Real-Time Airborne Target Tracking using DeepSort Algorithm and Yolov7 Model
- PPT location: slide 5-6
- Dataset used:
  - Training images were gathered mainly from:
    - Roboflow Universe
    - CUB-200-2011
    - Real World Object Detection Dataset for Quadcopter UAV Detection
  - Tracking videos were taken from:
    - Drone-vs-Bird Detection challenge data / related videos
- Evidence:
  - The paper states the training data were collected mainly from refs [15], [16], [17], and videos from [18]. In the references these map to Roboflow, CUB-200-2011, the Real World Object Detection Dataset, and the Drone-vs-Bird challenge paper.
- Dataset characteristics:
  - This is not one clean benchmark; it is a stitched multi-source dataset.
  - It mixes drones, birds, airplanes, "dayframes", and buildings.
  - Good for broad class coverage, but weaker as a scientific benchmark because collection protocols differ across sources.
  - CUB-200-2011 is a fine-grained bird dataset, so it helps bird-vs-drone discrimination.
  - The Real World Object Detection Dataset emphasizes real RGB quadcopter imagery across different sizes, backgrounds, and viewpoints.
  - Drone-vs-Bird videos are hard because drones are small, birds are distracting negatives, and both static and moving cameras appear.

### 3. An Unsupervised Moving Object Detection Network for UAV Videos
- PPT location: slide 8
- Dataset used:
  - Main dataset introduced by the paper: UAVMD
  - Additional evaluation benchmark: UAVDT test set
  - Additional generalization benchmark: BMS
- Dataset characteristics:
  - UAVMD is a UAV-to-ground moving-object dataset collected with DJI drones.
  - 70 sequences, 31,536 images, 1920x1080, with 37 representative sequences annotated every 10 frames.
  - Targets include motorcycles, tricycles, passenger cars, trucks, and other moving road objects.
  - The paper stresses UAV-specific challenges: sparse foreground, small targets, changing scale, partial occlusion, and camera/platform motion.
  - UAVDT is harder for moving-platform detection because it is a drone-view benchmark with vehicle targets and complex motion / weather / viewpoint variation.
  - BMS is a more generic moving-object benchmark with larger, more centered objects than UAVMD.

### 4. Fast and Robust UAV to UAV Detection and Tracking From Video
- PPT location: slide 9
- Dataset used:
  - U2U-D&TD (their own public UAV-to-UAV dataset)
- Dataset characteristics:
  - 50 video sequences, 70,250 frames, 30 fps.
  - Recorded by a GoPro camera mounted on a custom delta-wing airframe.
  - HD resolution around 1920x1080 or 1280x1060.
  - Up to 8 UAVs may appear in one frame.
  - Manually annotated ground truth.
  - Designed specifically for UAV-to-UAV detection, so the targets are tiny, faint, and often hard even for humans to see.
  - Strong match for your project because both sensor platform and target UAV are moving independently.

### 5. Detecting Flying Objects Using a Single Moving Camera
- PPT location: slide 10
- Dataset used:
  - Two datasets collected by the authors:
    - UAV dataset
    - Aircraft dataset
- Dataset characteristics:
  - The EPFL page says they collected two challenging datasets for UAVs and aircraft.
  - The search snippet for the PAMI paper states the UAV dataset has 20 video sequences of roughly 4000 frames each on average, at 752x480 resolution.
  - Captured by a camera mounted on a drone filming similar drones, both indoors and outdoors.
  - Includes strong real-world variation: lighting changes, weather, moving backgrounds, and tiny targets.
  - The aircraft dataset is aimed at collision-avoidance style detection with large scale variation and motion compensation demands.
  - This benchmark is important because it explicitly targets tiny flying objects viewed from a moving camera.

### 6. Moving Object Detection in Freely Moving Camera via Global Motion Compensation and Local Spatial Information Fusion
- PPT location: slide 11
- Dataset used:
  - CDNET2014
  - FBMS-59
  - CBD
- Dataset characteristics:
  - CDNET2014: broad foreground-background benchmark, mainly for fixed or mildly jittery cameras, includes illumination changes, shadows, dynamic backgrounds, and camera jitter.
  - FBMS-59: 59 motion-segmentation videos with moving cameras, including translation, rotation, and scaling; common test clips include cars, dogs, and people.
  - CBD: complex dynamic-background videos used for moving-camera segmentation.
  - This paper is less UAV-specific and more about general moving-camera foreground segmentation robustness.

### 7. Drone-vs-Bird Detection Grand Challenge at ICASSP 2023: A Review of Methods and Results
- PPT location: slide 17
- Dataset used:
  - Drone-vs-Bird Detection Challenge dataset
- Dataset characteristics:
  - 77 training video sequences and 30 test sequences according to the review.
  - Mix of static-camera and moving-camera recordings.
  - Resolutions range from 720x576 to 3840x2160.
  - Average sequence length is about 1,384 frames.
  - Drones are often very small; many annotated drones are under 322 pixels, and birds are frequent distractors.
  - Useful specifically for reducing false alarms between drones and birds.

## Most relevant datasets for your project

### Best match for UAV-to-UAV detection from a moving platform
1. U2U-D&TD
- Best domain match.
- Same core geometry: moving camera on UAV, moving aerial target, tiny objects.

2. EPFL UAV / Aircraft datasets
- Also directly about detecting flying objects from a moving camera.
- Good for motion-compensation ideas and tiny-target appearance variation.

3. Drone-vs-Bird Challenge dataset
- Valuable if bird false positives are a concern.
- Less directly UAV-to-UAV than Purdue, but strong for realistic distractors.

### More indirect but still useful
1. UAVMD / UAVDT
- Good for moving-platform motion segmentation and small targets.
- But they are mostly UAV-to-ground / vehicle-centric rather than UAV-to-UAV.

2. CDNET2014 / FBMS-59 / CBD
- Useful for generic motion-compensation and segmentation benchmarking.
- Weak domain fit for airborne tiny-object detection.

## Gaps / caveats
- The slide-2 paper (`Advanced Aerial Monitoring ...`) is real, but I could not verify the exact dataset name from accessible public metadata. The PPT evidence suggests a custom aerial vehicle dataset.
- The `Real-Time Airborne Target Tracking ...` paper uses a composite dataset assembled from multiple sources, so its reported results are less cleanly comparable to benchmark-driven papers.

## Sources
- Purdue UAV dataset page: https://engineering.purdue.edu/~bouman/UAV_Dataset/
- Li et al., `Fast and Robust UAV to UAV Detection and Tracking From Video`: https://engineering.purdue.edu/~bouman/UAV_Dataset/pubs/tetc01.pdf
- EPFL project page: https://www.epfl.ch/labs/cvlab/research/uav/research-unmanned-detection/
- PubMed record for `Detecting Flying Objects Using a Single Moving Camera`: https://pubmed.ncbi.nlm.nih.gov/28113698/
- `An Unsupervised Moving Object Detection Network for UAV Videos`: https://www.mdpi.com/2504-446X/9/2/150
- `Moving Object Detection in Freely Moving Camera via Global Motion Compensation and Local Spatial Information Fusion`: https://www.mdpi.com/1424-8220/24/9/2859
- `Real-Time Airborne Target Tracking using DeepSort Algorithm and Yolov7 Model`: https://thesai.org/Downloads/Volume15No2/Paper_48-Real_Time_Airborne_Target_Tracking.pdf
- `Drone-vs-Bird Detection Grand Challenge at ICASSP 2023: A Review of Methods and Results`: https://vcl.iti.gr/media/documents/the-drone-vs-bird-detection-grand-challenge-at-icassp-2023-a-review-of-methods-and-results.pdf
- WOSDETC challenge page: https://signalprocessingsociety.org/publications-resources/data-challenges/wosdetc-drone-vs-bird-detection-challenge-ieee-icassp-2023
