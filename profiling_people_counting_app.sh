#File for profile the application with VTune
source /opt/intel/openvino/bin/setupvars.sh
python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m Model/TensorFlow/ssd_mobilenet_v2_coco_2018_03_29/FP16/TF_ssd_mobilenet_v2_coco_2018_03_29.xml --stopSending True
