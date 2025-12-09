# ECE-4990-Final-Project
This project implements a real-time facial emotion detection system using a Raspberry Pi, PiCamera2, OpenCV, and the FER deep-learning model. The system captures live video frames, detects faces, classifies emotions, and overlays confidence scores directly onto the video stream. It serves as a complete implementation testbed for embedded emotion recognition.

Features:

- Real-time video capture using Picamera2
- Fast face detection using lightweight OpenCV models
- Emotion classification using the FER deep learning model
- Overlays bounding boxes, emotion labels, and confidence scores
- Fully optimized for Raspberry Pi hardware (low latency, low compute overhead)
- Modular design—easily extendable to additional models or preprocessing steps

Hardware requirements:

- Raspberry Pi 4 or 5
- Raspberry Pi Camera Module 3
- MicroSD card (32GB or higher recommended)
- External display (HDMI)
- Keyboard and mouse

Dataset used (FER-2013)

This project utilizes the FER-2013 (Facial Expression Recognition) dataset, a widely used benchmark for emotion classification.

Dataset Link: 
[FER-2013 Dataset (Kaggle)](https://www.kaggle.com/datasets/msambare/fer2013)

The FER model used in this project is pre-trained on FER-2013 and does not require additional training to run on the Raspberry Pi.

How to get started: 

1. Instal virtual environment tools:

sudo apt update
sudo apt install python3-venv -y

2. Create a virtual environment:

python3 -m venv emotion-env

3. Activate the environment:

source emotion-env/bin/activate

4. Install all project dependencies from requirements.txt:

pip install --upgrade pip
pip install -r requirements.txt

5. Create the code environment and paste main_emotion_script.py:

nano main_emotion_script.py

Press ctrl + O, ENTER, then ctrl + X to close.

6. Run the script:
   
python3 main_emotion_script.py

7. To exit environment:
   
deactivate

Implementation notes: 

- PiCamera2 is configured at 640×480 resolution for optimal speed vs. accuracy
- FER is run with mtcnn=False to use OpenCV’s faster face detector, improving performance on limited hardware.
- The script includes preprocessing steps such as:
  - BGR → RGB conversion
  - Confidence score extraction
  - Emotion probability selection
