# People Counting Project

Part of the **#30_days_udacity Challenge**

## Description
This project is intended to develop a counting algorithm to know how many people have left a  scene(for instance: A Bus Door, a crosswalk, a shop, etc.) via an exit and enter area, from a High-Angle camera. The algorithm detect and track the pedestrians via an ID and calculate the direction of every ID with the past positions of the subjects. The entry and exit bounding boxes are defined by the user. The direction of exiting and entering are necessary due to in some cases(refer to Concepts), the entry and exit are in the same area(for example a rear bus door). Some useful applications of the algorithm may be:
- Total Passengers count in buses, metros, etc.
- Calculate the total of people assisting to an event
- Analytics


## Motivation
My motivation behind this project is because in my country Ecuador is very common that Buses that connect cities are exceeding the maximum capacity of its units allowed by law, so people have to travel standing. My country has a important rate of accidents in highways, so passengers in overloaded buses might suffer more injuries even death. The Police Control is not enough, so I think that Artificial Vision Algorithms may help to control the maximum allowed number of passengers in a bus and surely, the data gathered by the algorithm could help to improve the transportation system in the future by taking better decisions in the schedules of the transport.

### Software Used:
- OpenVINO™ Toolkit and  [person-detection-retail-0013 model](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)
- OpenCV
- Paho-mqtt
- Mosquitto
- termcolor

### Dependencies
Install [OpenVINO Toolkit](https://docs.openvinotoolkit.org/latest/index.html) for your distribution

Dependencies for Linux/Raspberry Pi

```
sudo apt-get install python3-opencv nodejs mosquitto python3-pip libzmq3-dev libkrb5-dev ffmpeg cmake npm
sudo pip3 install numpy paho-mqtt termcolor py-cpuinfo
```

### Instructions
- Clone the repository: `git clone https://gitlab.com/josejacomeb/openvino-peoplecounter.git`
- Change your current directory to the project directory *openvino-peoplecounter*
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
- Setup your OpenVINO environment `source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6`
- Start the Mosquito Server `mosquitto -c websockets.conf`
- Start the NodeJS Server `./WebServer/openVINOPeopleCounting/bin` (After you can access to the main page in a Web Browser with the adress *device_ip*:3000, for example localhost:3000)
- Execute the python script `python3 app.py -i ` (You must include a video or the id of a camera, my test files: [Video Test Files](https://drive.google.com/open?id=1RkcITNEsRpw5I01vvI9t7Pj0hgRV_GZF)

**Note: Every Video Test file includes a configured entry and exit box, you can use it to quick test the algorithm!**

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/nNR66ZM1jaU/0.jpg)](https://youtu.be/nNR66ZM1jaU)

###Usage

```sh
python3 app.py [arguments]
```

Examples:

-Execute the script, load the Videos/factory.mp4 file, as I don't specify the bounding boxes file, the program will ask for it processing window.

```
./app.py -i Videos/factory.mp4
```

-Execute the script, load the file from the folder Videos/pedestrian.mp4, with the entry and exit bounding given by the user and with headless operation.

```
./app.py -i Videos/pedestrian.mp4 -n Videos/pedestrian_entry.json -x Videos/pedestrian_exit.json --headless True
```


#### **Arguments**
```
  -i (String) The location of the input file(Video or camera index) (Required)
  -d (String)The device name, default: 'CPU'
  --cpu_extension (String) Load the correct extension for your device, default:
  -o (String) Save the video processing file with the name given, default: "libcpu_extension_sse4.so"
  -m (String) The location the the OpenVINO™ Model
  -p (Float) Minimum confidence threshold[0-1.0], default=0.5
  --ip (String) IP Address of the Mosquitto Server, default: localhost
  --port (Integer) Port of the Mosquitto Server, default: 1883
  -n (String) File containing data of the entry bounding box of the video file, default: entry.son
  -x (String) File containing data of the exit bounding box of the video file, default: exit.son
  --debug (Boolean Enable debug information, default: False
  --headless (Boolean) Enable headless mode, default: False
```
### Graphical
Press

  x: To select the exit bounding box (Also to reset the last exit bounding box)

  n: To select the entry bounding box (Also to reset the last entry bounding box)

  Enter: Accept the selected box

  Space Bar: Toggle to start/stop the video and processing

###Concepts
#### Person
A person, is a bounding box result of the inference with OpenVINO™ Toolkit and has some  properties such as a numeric ID and a vector of direction shown in the upper part of the box.

![alt text](WebServer/openVINOPeopleCounting/public/images/id.png "ID image")

#### Entry/Exit Bounding box
Entry (green) and Exit(red) bounding boxes can be configured horizontally or vertically. They define a naturally entry and exit area (such as the end of a street, a door, etc) to start to count people who left inside each described area. Example:  
![alt text](WebServer/openVINOPeopleCounting/public/images/boundinboxes.png "ID image")
#### U Vectors
Useful to define the expected directions in which people will leave scene, if the person leaves in another direction, he/she isn't added to the counting. In the case such as below, we can define a single entry and exit box, with the direction the algorithm can differentiate between people exiting and entering of a door.
![alt text](WebServer/openVINOPeopleCounting/public/images/bothsides.png "ID image")

###Performance
The tests were conducted in the videos uploaded in my Cloud, the entry/exit boxes were loaded and I tested the Average FPS of every video in the CPU of my Laptop(Intel Core i7-6700HQ) but I'd like to test it in a Intel Neural Stick :D.

| File                  | Resolution    | GUI Average FPS | Headless Average FPS  |
| -------------         |:-------------:| :-----:         |   :--------------:    |
| aus_pedestrian.webm   | 1280x720      |  12             | 50                    |
| both_sides.mp4        | 1280x720      |  12             | 47                    |   
| chinapedestrians.mp4  | 1280x720      |  12             | 42                    |
| factory.mp4           | 1920x1080     |  13             | 41                    |
| pedestrian.mp4        | 640x360       |  11             | 35                    |
### Road Map
- [x]Include instructions to configure the system
- [x]Include the Model in Python
- [x]Pre-Process and Proccess the images on Python
- [x]Get the outputs of the inference and graph on it
- [x]Use a tracking algorithm to detect if the person is entering or exiting the Bus
- [x]Send information to a server
