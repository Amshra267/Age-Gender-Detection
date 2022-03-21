#! /bin/bash

## creating virtual environment
python -3 m venv base_Env
source base_Env/bin/activate

## Installing requirements 
pip install -r requirements-cpu.txt

## downloading yolov4 darknet weights and save in data folder
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -P data/

## making directory and adding one by one
mkdir model_data
wget https://github.com/Amshra267/BOSCH_A-G_INTERIIT/releases/download/v1.0.0/age.h5 -p model_data/
wget https://github.com/Amshra267/BOSCH_A-G_INTERIIT/releases/download/v1.0.0/gender.h5 -p model_data/
wget https://github.com/Amshra267/BOSCH_A-G_INTERIIT/releases/download/v1.0.0/unet.onnx -p model_data/
wget https://github.com/Amshra267/BOSCH_A-G_INTERIIT/releases/download/v1.0.0/mars-small128.pb -p model_data/
wget https://github.com/Amshra267/BOSCH_A-G_INTERIIT/releases/download/v1.0.0/New_32CL_5LR_43Epoc.zip -p model_data/
cd model_data
unzip New_32CL_5LR_43Epoc.zip
cd ..

## converting darknet weights to tensorflow model
python save_model.py --model yolov4

## making model_data directory and installing our weights into it
mkdir model_data

## making outputs folder
mkdir outputs