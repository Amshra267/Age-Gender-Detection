#! /bin/bash

## creating virtual environment
## Installing requirements 
pip3 install -r requirements-gpu.txt

## downloading yolov4 darknet weights and save in data folder
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -P data/

## making directory and adding one by one
mkdir model_data
wget https://github.com/Amshra267/BOSCH_A-G_INTERIIT/releases/download/v1.0.0/age.h5 -P model_data/
wget https://github.com/Amshra267/BOSCH_A-G_INTERIIT/releases/download/v1.0.0/gender.h5 -P model_data/
wget https://github.com/Amshra267/BOSCH_A-G_INTERIIT/releases/download/v1.0.0/unet.onnx -P model_data/
wget https://github.com/Amshra267/BOSCH_A-G_INTERIIT/releases/download/v1.0.0/mars-small128.pb -P model_data/

## converting darknet weights to tensorflow model
python3 save_model.py --model yolov4
