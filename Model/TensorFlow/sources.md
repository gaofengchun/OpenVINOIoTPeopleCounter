
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html

/opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites_tf.sh

##Convert Faster RCNNN

MODIR=/opt/intel/openvino/deployment_tools/model_optimizer
FRAMEWORK=TF

#Faster_rcnn
##Data type: FP16
DIR=faster_rcnn_inception_v2_coco_2018_01_28
NAME_OF_MODEL=frozen_inference_graph.pb
TYPE=FP16

python3 ${MODIR}/mo_tf.py --input_model=${DIR}/${NAME_OF_MODEL} --transformations_config ${MODIR}/extensions/front/tf/faster_rcnn_support.json --reverse_input_channels --tensorflow_object_detection_api_pipeline_config ${DIR}/pipeline.config --input_shape=[1,480,720,3] --input=image_tensor --output=detection_scores,detection_boxes,num_detections --data_type ${TYPE}  --progress --output_dir ${DIR}/${TYPE} --model_name ${FRAMEWORK}_${DIR}


##Data type: half
TYPE=half
python3 ${MODIR}/mo_tf.py --input_model=${DIR}/${NAME_OF_MODEL} --transformations_config ${MODIR}/extensions/front/tf/faster_rcnn_support.json --reverse_input_channels --tensorflow_object_detection_api_pipeline_config ${DIR}/pipeline.config --input_shape=[1,480,720,3] --input=image_tensor --output=detection_scores,detection_boxes,num_detections --data_type ${TYPE}  --progress --output_dir ${DIR}/${TYPE} --model_name ${FRAMEWORK}_${DIR}

#SSD
##Data type: FP16
DIR=ssd_inception_v2_coco_2018_01_28
NAME_OF_MODEL=frozen_inference_graph.pb
TYPE=FP16
python3 ${MODIR}/mo_tf.py --input_model=${DIR}/${NAME_OF_MODEL} --transformations_config ${MODIR}/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ${DIR}/pipeline.config --reverse_input_channels --progress --data_type ${TYPE} --output_dir ${DIR}/${TYPE} --model_name ${FRAMEWORK}_${DIR}
##Data type: Half
TYPE=half
python3 ${MODIR}/mo_tf.py --input_model=${DIR}/${NAME_OF_MODEL} --transformations_config ${MODIR}/extensions/front/tf/ssd_v2_support.json --tensorflow_object_detection_api_pipeline_config ${DIR}/pipeline.config --reverse_input_channels --progress --data_type ${TYPE} --output_dir ${DIR}/${TYPE} --model_name ${FRAMEWORK}_${DIR}

#YOLO-V3
python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights
##Data type: FP16
TYPE=FP16
DIR=YOLO-v3
NAME_OF_MODEL=frozen_darknet_yolov3_model.pb
TYPE=FP16
python3 ${MODIR}/mo_tf.py --input_model=${DIR}/${NAME_OF_MODEL} --data_type ${TYPE} --output_dir ${DIR}/${TYPE} --progress --tensorflow_use_custom_operations_config ${MODIR}/extensions/front/tf/yolo_v3.json --batch 1 --model_name ${FRAMEWORK}_${DIR}

##Data type: half
TYPE=half
python3 ${MODIR}/mo_tf.py --input_model=${DIR}/${NAME_OF_MODEL} --data_type ${TYPE} --output_dir ${DIR}/${TYPE} --progress --tensorflow_use_custom_operations_config ${MODIR}/extensions/front/tf/yolo_v3.json --batch 1 --model_name ${FRAMEWORK}_${DIR}


#YOLO-V3 tiny
#Convert to pb file
python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3-tiny.weights --tiny

##Data type: FP16
DIR=YOLOv3-tiny
NAME_OF_MODEL=frozen_darknet_yolov3_model.pb
TYPE=FP16

python3 ${MODIR}/mo_tf.py --input_model=${DIR}/frozen_darknet_yolov3_model.pb --data_type ${TYPE} --output_dir ${DIR}/${TYPE} --progress --tensorflow_use_custom_operations_config ${MODIR}/extensions/front/tf/yolo_v3_tiny.json --batch 1 --model_name ${FRAMEWORK}_${DIR}
##Data type: half
TYPE=half

python3 ${MODIR}/mo_tf.py --input_model=${DIR}/frozen_darknet_yolov3_model.pb --data_type ${TYPE} --output_dir ${DIR}/${TYPE} --progress --tensorflow_use_custom_operations_config ${MODIR}/extensions/front/tf/yolo_v3_tiny.json --batch 1 --model_name ${FRAMEWORK}_${DIR}

#ssd_mobilenet_v2_coco_2018_03_29
DIR=ssd_mobilenet_v2_coco_2018_03_29
NAME_OF_MODEL=frozen_darknet_yolov3_model.pb
##Data type: FP16
TYPE=FP16

python3 ${MODIR}/mo_tf.py --input_meta_graph=${DIR}/model.ckpt.meta --data_type ${TYPE} --tensorflow_use_custom_operations_config ${MODIR}/extensions/front/tf/ssd_v2_support.json --output_dir ${DIR}/${TYPE} --progress --tensorflow_object_detection_api_pipeline_config ${DIR}/pipeline.config --model_name ${FRAMEWORK}_${DIR}
##Data type: half
TYPE=half

python3 ${MODIR}/mo_tf.py --input_meta_graph=${DIR}/model.ckpt.meta --data_type ${TYPE} --tensorflow_use_custom_operations_config ${MODIR}/extensions/front/tf/ssd_v2_support.json --output_dir ${DIR}/${TYPE} --progress --tensorflow_object_detection_api_pipeline_config ${DIR}/pipeline.config --model_name ${FRAMEWORK}_${DIR}
