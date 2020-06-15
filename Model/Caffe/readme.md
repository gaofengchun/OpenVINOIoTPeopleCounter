# Install the prerequisites for Caffe via the installation of OpenVINO.
```bash
/opt/intel/openvino/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites_caffe.sh
```
# Outline
Download the model from the faster_rcnn model from [Caffe's Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
, more information and documentation about how to transform Caffe Models [here](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe.html#Convert_From_Caffe)

# Instructions
_Instructions working at June 2021_
Download the [prototxt](https://raw.githubusercontent.com/rbgirshick/py-faster-rcnn/master/models/pascal_voc/ZF/faster_rcnn_end2end/test.prototxt) and [caffemodel](https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0), more info [here](https://docs.openvinotoolkit.org/latest/_demos_object_detection_demo_faster_rcnn_README.html).
Execute the following commands in your terminal, inside the Caffe folder. You may have to check the *DIR* and *NAME_OF_MODEL* variables to ensure that the names are the same as it's shown in your downloaded files.
Setup the adress of the MO and the name of the Framework
```bash
MODIR=/opt/intel/openvino/deployment_tools/model_optimizer
FRAMEWORK=CAFFE
```

# faster_rcnn_models
## Data type: FP16
```bash
DIR=faster_rcnn_models
NAME_OF_MODEL=VGG16_faster_rcnn_final.caffemodel
PROTO=test.prototxt
TYPE=FP16
python3 ${MODIR}/mo_caffe.py --input_model ${DIR}/${NAME_OF_MODEL} --input_proto ${DIR}/${PROTO} --data_type FP16 --output_dir ${DIR}/${TYPE} --progress --model_name ${FRAMEWORK}_${DIR}
```

## Data type: half
```bash
TYPE=half
python3 ${MODIR}/mo_caffe.py --input_model ${DIR}/${NAME_OF_MODEL} --input_proto ${DIR}/${PROTO} --data_type FP16 --output_dir ${DIR}/${TYPE} --progress --model_name ${FRAMEWORK}_${DIR}
```
