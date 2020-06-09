# People Counting Project

## Description
This project is intended to develop a counting algorithm to know how many people is in Room.. The algorithm detect and track the pedestrians via an ID and calculate the direction of every ID with the past positions of the subjects. Some useful applications of the algorithm may be:
- Total Passengers count in buses, metros, etc.
- Calculate the total of people assisting to an event
- Analytics
- Automatize the entry, exit doors to allow people to respect the social distancing.


## Motivation
My motivation behind this project is because in my country Ecuador is very common that Buses that connect cities are exceeding the maximum capacity of its units allowed by law, so people have to travel standing. My country has a important rate of accidents in highways, so passengers in overloaded buses might suffer more injuries even death. The Police Control is not enough, so I think that Artificial Vision Algorithms may help to control the maximum allowed number of passengers in a bus and surely, the data gathered by the algorithm could help to improve the transportation system in the future by taking better decisions in the schedules of the transport.

### Software Used:
- OpenVINO™ Toolkit and  [person-detection-retail-0013 model](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)
- OpenCV
- Paho-mqtt
- Mosquitto
- termcolor
- NodeJS
- Javascript
- Bootstrap
- React

### Features
- Using the Intel's People Counting Template made in React
- Select the max number of people inside the room.
- Show/Hide the Stats data of the video
- Start/Stop transmission of the video stream
- Custom people tracking algorithm.

### Dependencies
Install [OpenVINO Toolkit](https://docs.openvinotoolkit.org/latest/index.html) for your distribution

Dependencies for Linux/Raspberry Pi

```
sudo apt-get install python3-opencv nodejs mosquitto python3-pip libzmq3-dev libkrb5-dev ffmpeg cmake npm
sudo pip3 install numpy paho-mqtt termcolor py-cpuinfo
```

### Instructions
- Clone the repository: `git clone https://github.com/josejacomeb/OpenVINOIoTPeopleCounter.git`
- Change your current directory to the project directory *OpenVINOIoTPeopleCounter*
- Install the dependencies for the Mosca Server


```
cd webservice/server
npm install
```

- Install the dependencies for the Web server as well

```
cd ../ui
npm install
```
- Setup your OpenVINO environment `source /opt/intel/openvino/bin/setupvars.sh`
- Start the Mosca Server

```
cd webservice/server/node-server
node ./server.js
```

- Start the NodeJS Server

```sh
cd webservice/ui
npm run dev
```

 (After you can access to the main page in a Web Browser with the adress *device_ip*:3000, for example 0.0.0.0:3000)
- Execute the python script `python3 app.py -i ` (You must include a video or the id of a camera, my test files: [Video Test Files](https://drive.google.com/open?id=1RkcITNEsRpw5I01vvI9t7Pj0hgRV_GZF)

**Video Demonstration**

#Test in Raspberry Pi
Unfortunately, the FFMPEG server(ffserver) is deprecated, so until now, I didn't make it works, I connected it to the server of my laptop to test.

![Test in RPI][resources/TestRPI.png]

#Test in the PC
Click on the following link to open the video Demonstration

[![Test in Youtube](http://img.youtube.com/vi/8lGoEMAfvX8/0.jpg)](https://youtu.be/8lGoEMAfvX8)

###Usage

```sh
python3 app.py [arguments]
```

Examples:

-Execute the script, load the resources/Pedestrian_Detect_2_1_1.mp4 file, the program will run with the default data.

```
python3 main.py -i --i resources/Pedestrian_Detect_2_1_1.mp4
```

-Execute the script, load the file from the folder Videos/pedestrian.mp4, and send the data to the FFMPEG Server

```
python3 main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m Model/person-detection-retail-0013.xml -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```


#### **Arguments**
```
  -i or --input (String) The location of the input file(Video or camera index) (Required)
  -d or --device (String) The device name, default: 'CPU'
  -l or --cpu_extension (String) Load the correct extension for your device, default:
  -o (String) Save the video processing file with the name given, default: "libcpu_extension_sse4.so"
  -m or model (String) The location the the OpenVINO™ Model, default: Model/person-detection-retail-0013.xml
  -pt or --prob_threshold (Float) Minimum confidence threshold[0-1.0], default=0.6
  --ip (String) IP Address of the MQTT Server, default: localhost
  --port (Integer) Port of the Mosca Server, default: 3002
  --debug (Boolean Enable debug information, default: False
  --headless (Boolean) Enable headless mode, default: False
  --stopSending or -s (Boolean) Allow to the user to send or not send the stream via FFMPEG, default False
  --showInferenceStats or -t (Boolean) Show the probability of the inference, the ID of the person, default True
```
### Models used
The first both models were converted with the model optimizer and they're in the Model's folders of the project. It was converted to FP16 and the instructions to convert are inside the Model/Tensorflow folder.

- [Tensorflow ssd_inception_v2_coco_2018_01_28](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

- [Tensorflow ssd_mobilenet_v2_coco_2018_03_29](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

- [OpenVINO person-detection-retail-0013](https://docs.openvinotoolkit.org/2018_R5/_docs_Retail_object_detection_pedestrian_rmnet_ssd_0013_caffe_desc_person_detection_retail_0013.html)

###Performance

Two devices where tested in this experiment, the data is under the ExperimentalData Folder, a Intel(R) Core(TM) i7-6700HQ CPU @ 2.60GHz and a Raspberry Pi with it's MYRIAD Neural Compute Stick.
####Test in the Laptop

| Model                            | People Counted| AVG FPS | AVG Inference Time[ms]   |
| -------------                    |:-------------:| :-----:         |   :--------------:    |
| person-detection-retail-0013     |   6           |  35             | 24                    |
| ssd_inception_v2_coco_2018_01_28 | 12            |  42             | 20                    |   
| ssd_mobilenet_v2_coco_2018_03_29 | 12            |  44             | 19                    |

####Test in the Laptop with NCS

| Model                            | People Counted| AVG FPS | AVG Inference Time[ms]   |
| -------------                    |:-------------:| :-----:         |   :--------------:    |
| person-detection-retail-0013     |   6           |  7             | 132                    |
| ssd_inception_v2_coco_2018_01_28 | 12            |  14             | 65                    |   
| ssd_mobilenet_v2_coco_2018_03_29 | 12            |  14             | 65                    |
####Test in the RPI with NCS
* It was not possible to do with the another networks, I'm still investigating.

| Model                            | People Counted| AVG FPS | AVG Inference Time[ms]   |
| -------------                    |:-------------:| :-----:         |   :--------------:    |
| person-detection-retail-0013     |   6           |  4             | 208                   |
