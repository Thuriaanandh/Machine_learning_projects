# **Facial Emotion Recognition Using Deep Learning**  

## **Overview**  
This project implements a **real-time facial emotion recognition system** using **DeepFace, OpenCV, and Python**. It detects emotions such as **happiness, fear, anger, and neutrality** from both **static images and live webcam feeds**. The model leverages **deep learning-based facial analysis** to ensure accurate recognition of facial expressions.  

## **Features**  
- ✅ Emotion detection from images and live webcam feeds  
- ✅ Uses **DeepFace** for deep learning-based analysis  
- ✅ **OpenCV** for image processing and real-time video capture  
- ✅ Bounding boxes around detected faces with **labeled emotions**  
- ✅ Supports multiple **deep learning models** for emotion classification  

## **Technologies Used**  
- **Python**  
- **OpenCV** (for image processing and face detection)  
- **DeepFace** (for emotion recognition)  
- **Matplotlib** (for visualization)  
- **Google Colab** (for execution and testing)  

## **Installation**  
To run this project locally, install the required dependencies:  

```bash
pip install deepface opencv-python matplotlib
```

## **Usage**  
### **1. Emotion Detection from an Image**  
Run the following code to analyze emotions from an image:  

```python
import cv2
from deepface import DeepFace

img = cv2.imread("your_image.jpg")  
predictions = DeepFace.analyze(img, actions=['emotion'])  
print(predictions)
```

### **2. Real-Time Emotion Detection Using Webcam**  
This script captures an image from the webcam, detects faces, and classifies emotions:  

```python
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)  # Open webcam
ret, frame = cap.read()  # Capture frame
cap.release()  # Release webcam

analysis = DeepFace.analyze(frame, actions=['emotion'])
print("Detected Emotion:", analysis[0]['dominant_emotion'])
```

## **Demo**  
![Emotion Detection](demo.gif)  

## **Challenges & Future Enhancements**  
### **Challenges**  
- Accuracy may be affected by **lighting conditions** and **facial occlusions** (glasses, masks, etc.).  
- **Real-time processing** requires optimization for better speed.  
- Dataset biases may impact model performance.  

### **Future Enhancements**  
- **Multi-modal analysis** (combining facial expressions with voice recognition).  
- **Edge AI optimization** for lightweight deployment on mobile/embedded devices.  
- **Enhanced GUI** for a better user interface.  


