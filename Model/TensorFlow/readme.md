# Install the prerequisites for Caffe via the installation of OpenVINO.
```bash
/opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites_tf.sh
```

# Outline
You can find the model ins the following [TensorFlow's Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
, more information and documentation about how to transform TensorFlow Models [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html)

<https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html>

# Instructions
_Instructions working at June 2021_
Download every model and unzip it inside the TensorFlow folder.
Execute the following commands in your terminal, inside the TensorFlow folder. You may have to check the *DIR* and *NAME_OF_MODEL* variables to ensure that the names are the same as it's shown in your downloaded files.
Setup the address of the MO and the name of the Framework
```bash
MODIR=/opt/intel/openvino/deployment_tools/model_optimizer
FRAMEWORK=TF
```
Path for the configuration file with custom operation description of TensorFlow
```bash
/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf
```


## Convert Faster RCNNN
# Faster_rcnn
Download the model form [here](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
## Data type: FP16
```bash
DIR=faster_rcnn_inception_v2_coco_2018_01_28
NAME_OF_MODEL=frozen_inference_graph.pb
TYPE=FP16

python3 ${MODIR}/mo_tf.py --input_model=${DIR}/${NAME_OF_MODEL} --tensorflow_use_custom_operations_config ${MODIR}/extensions/front/tf/faster_rcnn_support.json --reverse_input_channels --output=detection_classes,detection_scores,detection_boxes,num_detections --tensorflow_object_detection_api_pipeline_config ${DIR}/pipeline.config --input=image_tensor --data_type ${TYPE}  --progress --output_dir ${DIR}/${TYPE} --model_name ${FRAMEWORK}_${DIR} --input_shape=[1,600,1024,3]
```

## Data type: half
```bash
TYPE=half
python3 ${MODIR}/mo_tf.py --input_model=${DIR}/${NAME_OF_MODEL} --transformations_config ${MODIR}/extensions/front/tf/faster_rcnn_support.json --reverse_input_channels --output=detection_classes,detection_scores,detection_boxes,num_detections --tensorflow_object_detection_api_pipeline_config ${DIR}/pipeline.config --input=image_tensor --data_type ${TYPE}  --progress --output_dir ${DIR}/${TYPE} --model_name ${FRAMEWORK}_${DIR} --input_shape=[1,600,1024,3]
```
# SSD
Download the model form [here](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)
## Data type: FP16
```bash
DIR=ssd_inception_v2_coco_2018_01_28
NAME_OF_MODEL=frozen_inference_graph.pb
TYPE=FP16
python3 ${MODIR}/mo_tf.py --input_model=${DIR}/${NAME_OF_MODEL} --transformations_config ${MODIR}/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ${DIR}/pipeline.config --reverse_input_channels --progress --data_type ${TYPE} --output_dir ${DIR}/${TYPE} --model_name ${FRAMEWORK}_${DIR}
```
## Data type: Half
```bash
TYPE=half
python3 ${MODIR}/mo_tf.py --input_model=${DIR}/${NAME_OF_MODEL} --transformations_config ${MODIR}/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ${DIR}/pipeline.config --reverse_input_channels --progress --data_type ${TYPE} --output_dir ${DIR}/${TYPE} --model_name ${FRAMEWORK}_${DIR}
```
# ssd_mobilenet_v2_coco_2018_03_29
Download the model form [here](http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz)

## Data type: FP16
```bash
DIR=ssd_mobilenet_v2_coco_2018_03_29
NAME_OF_MODEL=frozen_inference_graph.pb
TYPE=FP16

python3 ${MODIR}/mo_tf.py --input_model=${DIR}/${NAME_OF_MODEL} --data_type ${TYPE} --tensorflow_use_custom_operations_config ${MODIR}/extensions/front/tf/ssd_v2_support.json --output_dir ${DIR}/${TYPE} --progress --tensorflow_object_detection_api_pipeline_config ${DIR}/pipeline.config --model_name ${FRAMEWORK}_${DIR} --output=detection_classes,detection_scores,detection_boxes,num_detections --reverse_input_channels --input_shape=[1,300,300,3] --input=image_tensor
```
## Data type: half
```bash
TYPE=half

python3 ${MODIR}/mo_tf.py --input_model=${DIR}/${NAME_OF_MODEL} --data_type ${TYPE} --tensorflow_use_custom_operations_config ${MODIR}/extensions/front/tf/ssd_v2_support.json --output_dir ${DIR}/${TYPE} --progress --tensorflow_object_detection_api_pipeline_config ${DIR}/pipeline.config --model_name ${FRAMEWORK}_${DIR} --output=detection_classes,detection_scores,detection_boxes,num_detections --reverse_input_channels --input_shape=[1,300,300,3] --input=image_tensor
```

# YOLO-V3
## Convert to pb file
Get the files from this: [GitHub repo](https://github.com/mystic123/tensorflow-yolo-v3).
First, convert the YOLOV3 Darknet model to TensorFlow Model as show [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html)
```bash
python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights
```
_Then, check the name of your files!_
## Data type: FP16
```bash
TYPE=FP16
DIR=YOLO-v3
NAME_OF_MODEL=frozen_darknet_yolov3_model.pb
TYPE=FP16
python3 ${MODIR}/mo_tf.py --input_model=${DIR}/${NAME_OF_MODEL} --data_type ${TYPE} --output_dir ${DIR}/${TYPE} --progress --tensorflow_use_custom_operations_config ${MODIR}/extensions/front/tf/yolo_v3.json --batch 1 --model_name ${FRAMEWORK}_${DIR}
```
## Data type: half
```bash
TYPE=half
python3 ${MODIR}/mo_tf.py --input_model=${DIR}/${NAME_OF_MODEL} --data_type ${TYPE} --output_dir ${DIR}/${TYPE} --progress --tensorflow_use_custom_operations_config ${MODIR}/extensions/front/tf/yolo_v3.json --batch 1 --model_name ${FRAMEWORK}_${DIR}
```
# YOLO-V3 tiny
## Convert to pb file
Get the files from this: [GitHub repo](https://github.com/mystic123/tensorflow-yolo-v3).
First, convert the YOLOV3 Darknet model to TensorFlow Model as show [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html)
```bash
python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3-tiny.weights --tiny
```

## Data type: FP16
```bash
DIR=YOLOv3-tiny
NAME_OF_MODEL=frozen_darknet_yolov3_model.pb
TYPE=FP16

python3 ${MODIR}/mo_tf.py --input_model=${DIR}/frozen_darknet_yolov3_model.pb --data_type ${TYPE} --output_dir ${DIR}/${TYPE} --progress --tensorflow_use_custom_operations_config ${MODIR}/extensions/front/tf/yolo_v3_tiny.json --batch 1 --model_name ${FRAMEWORK}_${DIR}
```
## Data type: half
```bash
TYPE=half

python3 ${MODIR}/mo_tf.py --input_model=${DIR}/frozen_darknet_yolov3_model.pb --data_type ${TYPE} --output_dir ${DIR}/${TYPE} --progress --tensorflow_use_custom_operations_config ${MODIR}/extensions/front/tf/yolo_v3_tiny.json --batch 1 --model_name ${FRAMEWORK}_${DIR}
```
