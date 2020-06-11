https://github.com/BVLC/caffe/wiki/Model-Zoo
https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe.html#Convert_From_Caffe

/opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites_caffe.sh.sh
MODIR=/opt/intel/openvino/deployment_tools/model_optimizer
FRAMEWORK=CAFFE

#faster_rcnn_models
##Data type: FP16
DIR=faster_rcnn_models
NAME_OF_MODEL=VGG16_faster_rcnn_final.caffemodel
PROTO=test.prototxt
##Data type: FP16
TYPE=FP16
python3 ${MODIR}/mo_caffe.py --input_model ${DIR}/${NAME_OF_MODEL} --input_proto ${DIR}/${PROTO} --data_type FP16 --output_dir ${DIR}/${TYPE} --progress --model_name ${FRAMEWORK}_${DIR}

##Data type: half
TYPE=half
python3 ${MODIR}/mo_caffe.py --input_model ${DIR}/${NAME_OF_MODEL} --input_proto ${DIR}/${PROTO} --data_type FP16 --output_dir ${DIR}/${TYPE} --progress --model_name ${FRAMEWORK}_${DIR}
