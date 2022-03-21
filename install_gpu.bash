#! /bin/bash

## creating virtual environment
python -3 m venv base_Env
source base_Env/bin/activate

## Installing requirements 
pip install -r requirements-gpu.txt

## downloading yolov4 darknet weights and save in data folder
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights -P data/

## converting darknet weights to tensorflow model
python save_model.py --model yolov4

## making model_data directory and installing our weights into it
mkdir model_data


## making outputs folder
mkdir outputs