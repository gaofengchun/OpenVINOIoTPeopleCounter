"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import socket
import json #Convert data
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

from termcolor import colored #Color the output
import signal, sys #Get Control + C Data, send data
import numpy as np #Transpose function
from random import uniform #Get a new combination of colors very time
from platform import processor, system
from os.path import split, basename, splitext
import cpuinfo
import mimetypes
mimetypes.init()
import faulthandler; faulthandler.enable()

# MQTT server environment variables
HOSTNAME = socket.gethostname()
MQTT_KEEPALIVE_INTERVAL = 60

terminate = False
showInferenceStats = True
stopSending = False
mqttActive = True
def signal_handling(signum,frame):
    global terminate
    terminate = True

def update_existing(detection, location, maximum_locations = 5):
    detection['past_locations'].append(detection['current_location'])
    detection['current_location'] = location
    detection['frame_number'] = location['frame']
    if len(detection['past_locations']) > maximum_locations :
        detection['past_locations'].pop(0) #Delete first element
    location_x = location['chest'][0]
    location_y = location['chest'][1]
    sum_location_x = 0
    sum_location_y = 0
    for i in reversed(detection['past_locations']):
        sum_location_x+= (location_x - i['chest'][0])
        sum_location_y+= (location_y - i['chest'][1])
        location_x = i['chest'][0]
        location_y = i['chest'][1]
    sum_location_x/= (len(detection['past_locations']))
    sum_location_y/= (len(detection['past_locations']))
    detection['tendency'] = (sum_location_x, sum_location_y)
    return detection

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-i", '--input', required=True,
                        help="Input file(number if camera, or file path to a videofile)")
    parser.add_argument("-d", '--device', default='CPU',
                        help="Specify the target device to infer on: "
                        "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                        "will look for a suitable plugin for device "
                        "specified (CPU by default)")
    parser.add_argument("-o", default="", help="If given, it will save a videofile with the result of the detection.")
    parser.add_argument("-m", '--model', default="Model/person-detection-retail-0013.xml")
    parser.add_argument("-pt",'--prob_threshold', type=float, default=0.6,
                        help="Probability threshold for detections filtering(0.5 by default)")
    parser.add_argument("--ip", type=str, default= socket.gethostbyname(HOSTNAME))
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None, help="MKLDNN (CPU)-targeted custom layers."
                         "Absolute path to a shared library with the"
                         "kernels impl.")
    parser.add_argument("--port", help="", type=int, default=3002)
    parser.add_argument("-n", help="", type=str, default="entry.json")
    parser.add_argument("-x", help="", type=str, default="exit.json")
    parser.add_argument("--debug", help="", type=bool, default=False)
    parser.add_argument("--headless", help="headless_desc", type=bool, default=True)
    parser.add_argument("--stopSending" ,"-s", help="Choose to send or not video", type=bool, default=False)
    parser.add_argument("--showInferenceStats" ,"-t", help="Show the inference stats and video files", type=bool, default=True)
    return parser

def on_message(client, userdata, msg):
    global stopSending, showInferenceStats
    jsonpayload = json.loads(msg.payload)
    if str(msg.topic) == str('stopSending'):
        stopSending = jsonpayload['value']
    elif msg.topic == 'showInferenceStats':
        showInferenceStats = jsonpayload['value']

def connect_mqtt(args):
    ### TODO: Connect to the MQTT client ###
    try:
        client = mqtt.Client(transport="websockets")
        client.on_message = on_message
        client.subscribe("stopSending")
        client.subscribe("showInferenceStats")
        client.connect(args.ip, port=args.port, keepalive=MQTT_KEEPALIVE_INTERVAL)
        connected = True
    except:
        print(colored("Can\'t connect to MQTT Server, please start the next time the program Mosquitto with mosquitto -c websockets.conf", 'yellow'))
        print(colored("Please read the Readme for more information",'yellow'))
        connected = False
    return client, connected


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    detections = []
    maximum_ratio_difference = 0.07
    maximum_frame_difference = 10 #10 frames
    ids = list(range(0,100))
    out = None
    if args.o:
        out = cv2.VideoWriter(args.o, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frame_rate, (frame_width, frame_height))
    entry_people = 0
    exit_people = 0
    process = True
    help = False
    scale = 5
    action = ""
    color_help = (251,36,11)
    color_warning = (102,255,255)
    color_informative = (25,255,255)
    total_fps = 0
    total_fps_measurements = 0
    frame_counter = 0
    fps = 0

    if not args.headless:
        cv2.namedWindow("Output Video")
    if args.debug:
        cv2.namedWindow("Original")
        cv2.namedWindow("Past detection")
    global entry_parameters, exit_parameters, stopSending, showInferenceStats
    global noEntryBox, noExitBox, mqttActive
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model, device=args.device, cpu_extension=args.cpu_extension)

    ### TODO: Handle the input stream ###
    if args.input.isnumeric():
        dataorFile = int(args.input)
        print(colored("Using the camera id {} to get image!".format(args.input),'green'))

    else:
        type_of_data = mimetypes.guess_type(args.input)[0]
        if "image" in type_of_data:
            print(colored("Sorry, it's not possible to run the program in a image, please choose an video or specify an id of a camera as input!",'red'))
            return -1
        else:
            dataorFile = args.input
            print(colored("Using the videofile {} to get image!".format(args.input),'green'))

    cap = cv2.VideoCapture(dataorFile)
    ret,frame = cap.read()
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    ### TODO: Pre-process the image as needed ###
    nw_shape = infer_network.get_input_shape()
    infer_network.inference_parameters(frame_height, frame_width, prob_threshold )

    if not ret:
        print(colored("No image found, please check the videofile!",'red'))
        return
    ### TODO: Loop until stream is over ###
    if args.headless:
        '''print(colored("The app is running in headless mode!",'magenta'))
        print(colored("Please press Control + C to terminate",'magenta'))'''
        pass
    signal.signal(signal.SIGINT,signal_handling)
    total_number_people = 0
    info_system = "OS: " + system() + " arch: " + processor()
    info_model = "Model: {}".format(split(args.model)[1])
    time_end_inference = 0
    avg_time_inference = 0
    start = time.time() #Start time to calculate FPS
    stopSending = args.stopSending
    showInferenceStats = args.showInferenceStats

    while cap.isOpened():

        if terminate:
            print(colored("Finishing the cycle", 'green'))
            break
        number_of_people = 0
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ### TODO: Read from the video capture ###
        if process:
            ret,frame = cap.read()
            if not ret:
                break
            output_frame = frame.copy()
            pp_frame = cv2.resize(frame, (nw_shape[3],  nw_shape[2]))
            pp_frame = np.transpose(pp_frame, (2,0,1))
            pp_frame = np.reshape(pp_frame, (1,nw_shape[1],nw_shape[2],nw_shape[3]))
            ### TODO: Start asynchronous inference for specified request ###
            time_inference = int(round(time.time() * 1000)) #Start time to calculate FPS
            infer_network.exec_net(pp_frame)
            ### TODO: Wait for the result ###
            while infer_network.wait():
                pass
            ### TODO: Get the results of the inference request ###
            time_end_inference = int(round(time.time() * 1000))
            time_end_inference = (time_end_inference - time_inference)
            output = infer_network.get_output()

            if args.debug:
                print("--------------------------------")
                print("Frame number: " + str(frame_number))
                print("--------------------------------")
            draw_color = (0, 0, 0)
            new_detections = []
            index = 0
            if args.debug:
                for x in detections:
                    cv2.circle(past_frame, x['current_location']['chest'], int(x['current_location']['width']/2), x['color'], 1)
                    cv2.putText(past_frame, str(x['id']), (x['current_location']['corner'][0], int(x['current_location']['corner'][1] + x['current_location']['height']/2) ), cv2.FONT_HERSHEY_SIMPLEX ,1, x['color'], 2, cv2.LINE_AA)
            for i in output:
                location = {'corner':(0,0),'height':0, 'width':0, 'area':0, 'chest':(0,0), 'frame': 0, 'prob': i['confidence']}
                p1x = i['xmin']
                p1y = i['ymin']
                p2x = i['xmax']
                p2y = i['ymax']
                location['corner'] = (p1x, p1y)
                location['height'] = abs(p2y - p1y)
                location['width'] = abs(p2x - p1x)
                location['chest'] = (int(p1x + (p2x-p1x)/2), int(p1y + (p2y-p1y)/2))
                location['area'] = (p2x - p1x)*(p2y-p1y)
                location['frame'] = frame_number
                distancep2_chest = cv2.norm(location['corner'], location['chest'], normType=cv2.NORM_L2)
                sort_detection = []

                for x in detections:
                    calculated_distance = cv2.norm(x['current_location']['chest'], location['chest'], normType=cv2.NORM_L2)
                    speed_distance = cv2.norm((int(x['current_location']['chest'][0] + x['tendency'][0]),int(x['current_location']['chest'][1] + x['tendency'][1])), location['chest'], normType=cv2.NORM_L2)
                    calculated_area_ratio =  0#abs(1.0 - area/x['location'][len(x['location'])-1]['area'])
                    calculated_difference_frames = abs(location['frame'] - x['frame_number'])
                    future_frame_distance = cv2.norm((int(x['current_location']['chest'][0] + x['tendency'][0]*calculated_difference_frames),int(x['current_location']['chest'][1] + x['tendency'][1]*calculated_difference_frames)), location['chest'], normType=cv2.NORM_L2)

                    if calculated_difference_frames > 0:
                        ratio = (detections.index(x), calculated_distance, calculated_area_ratio, calculated_difference_frames, future_frame_distance, speed_distance)
                        sort_detection.append(ratio)
                sort_detection_meaning = sorted(sort_detection, key=lambda sort_detection: sort_detection[1])
                sort_detection_future = sorted(sort_detection, key=lambda sort_detection: sort_detection[4])
                if args.debug:
                    for lines in sort_detection_meaning[0:1]:
                        id = lines[0]
                        if len(detections[id]['past_locations']) > 0:
                            cv2.line(past_frame, location['chest'], detections[id]['past_locations'][-1]['chest'], detections[id]['color'],2)
                if len(sort_detection_future) > 0:
                    id = sort_detection_future[0][0]
                    df = sort_detection_future[0][3]
                    length_line = 10
                    future_point = (int(detections[id]['current_location']['chest'][0] + detections[id]['tendency'][0]*df),int(detections[id]['current_location']['chest'][1] + detections[id]['tendency'][1]*df))
                    if args.debug:
                        cv2.line(past_frame, (future_point[0] + length_line, future_point[1]),(future_point[0] - length_line, future_point[1]), detections[id]['color'],2)
                        cv2.line(past_frame, (future_point[0], future_point[1] + length_line),(future_point[0], future_point[1]- length_line), detections[id]['color'],2)
                if args.debug:
                    for lines in sort_detection_future[0:1]:
                        id = lines[0]
                        if len(detections[id]['past_locations']) > 0:
                            cv2.line(past_frame, location['chest'], detections[id]['past_locations'][-1]['chest'], detections[id]['color'],2)

                if len(sort_detection_meaning)>0 and sort_detection_meaning[0][1] <= distancep2_chest and sort_detection_meaning[0][3] < maximum_frame_difference and sort_detection_meaning[0][3] > 0:
                    index = sort_detection_meaning[0][0]
                    id = detections[index]['id']
                    if args.debug:
                        print("¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡")
                        print("Distance checking")
                        print("Detected ID: " + str(id))
                        print(sort_detection_meaning[0:1])
                    detections[index] = update_existing(detections[index], location)
                    draw_color = detections[index]['color']
                    number_of_people+=1
                elif len(sort_detection_future)>0 and sort_detection_future[0][4] <= distancep2_chest and sort_detection_future[0][3] < maximum_frame_difference and sort_detection_future[0][3] > 0:
                    if args.debug:
                        print("<><><><><><><><><><><><><><><><>")
                        print("Future distance checking!")
                        print(sort_detection_future[0:1])
                        print("Detected ID: " + str(id))
                    index = sort_detection_future[0][0]
                    id = detections[index]['id']
                    detections[index] = update_existing(detections[index], location)
                    draw_color = detections[index]['color']
                    number_of_people+=1
                else:
                    #Detected a new person
                    if args.debug:
                        print("%%%%%%%%%%%%%%%%%%%%%")
                        print("New person detected!")
                        print("Detected ID: " + str(id))

                    tracker = {'id':0, 'current_location':{}, 'past_locations': [] , 'frame_number':0, 'color':(255,255,255), 'tendency': (0,0), 'vector': (0,0)}
                    tracker['current_location'] = location
                    tracker['color'] =  (int(uniform(125, 255)), int(uniform(125, 255)), int(uniform(125, 255)))
                    tracker['frame_number'] =  frame_number
                    tracker['frame_begin'] =  frame_number

                    tracker['id'] =  ids[0]
                    ids.pop(0)
                    detections.append(tracker)
                    id = tracker['id']
                    index = len(detections) - 1
                    draw_color = tracker['color']
                    number_of_people+=1
                    total_number_people += 1

                cv2.circle(output_frame, location['chest'], 3, draw_color, -1)
                cv2.rectangle(output_frame,(p1x,p1y), (p2x, p2y), draw_color, 3)
                cv2.putText(output_frame, str(id), (location['corner'][0], location['corner'][1]+30) , cv2.FONT_HERSHEY_SIMPLEX ,
                           1, draw_color, 2, cv2.LINE_AA)
                cv2.putText(output_frame, "{0:.2f}".format(float(location['prob'])), (location['corner'][0] + location['width'], location['corner'][1] + 30) , cv2.FONT_HERSHEY_SIMPLEX ,
                           1, draw_color, 2, cv2.LINE_AA)

                if len(detections) > 0:
                    if len(detections[index]['past_locations']) > 0:
                        cv2.line(output_frame, (int(location['corner'][0] + location['width']/2), location['corner'][1]), \
                        (int((location['corner'][0] + location['width']/2 + scale*detections[index]['tendency'][0])),\
                        int((location['corner'][1] + scale*detections[index]['tendency'][1]))), \
                         detections[index]['color'],3)
#####
            for x in detections:
                if abs(x['frame_number'] - frame_number) > maximum_frame_difference:
                    uvx = 0
                    if x['tendency'][0] > 0 or x['tendency'][0] < 0:
                        uvx = int(x['tendency'][0]/abs(x['tendency'][0]))
                    uvy = 0
                    if x['tendency'][1] > 0 or x['tendency'][1] < 0:
                        uvy = int(x['tendency'][1]/abs(x['tendency'][1]))
                    tuv = (uvx, uvy)
                    if args.debug:
                        print("//////Deleting detected person///////////////")
                        print("ID: " + str(x['id']))
                        print("******************************")
                        print("TUV: " + str(tuv))
                        print(x['tendency'])
                        print("******************************")
                    ### Topic "person/duration": key of "duration" ###
                    if mqttActive:
                        person_duration = json.dumps({'duration':int((frame_number - x['frame_begin'])/fps)})
                        client.publish("person/duration", payload=person_duration)
                    ids.append(x['id'])
                    detections.remove(x)

            frame_counter+=1
            if(time.time() - start) >= 1:
                fps = frame_counter
                total_fps += fps
                total_fps_measurements+=1
                start = time.time()
                frame_counter = 0
                avg_time_inference += time_end_inference

        else:
            output_frame = frame.copy()
        if args.o:
            out.write(output_frame)

        if help:
            cv2.putText(output_frame, "Press: ", (int(3*output_frame.shape[1]/7),int(4.25*output_frame.shape[0]/7) ) , cv2.FONT_HERSHEY_SIMPLEX,0.78, color_help, 2, cv2.LINE_AA)
            cv2.putText(output_frame, "Space bar: Toggle to start/stop processing", (int(3*output_frame.shape[1]/7),int(5*output_frame.shape[0]/7) ) , cv2.FONT_HERSHEY_SIMPLEX,0.88, color_help, 2, cv2.LINE_AA)
            cv2.putText(output_frame, "Esc: Exit", (int(3*output_frame.shape[1]/7),int(5.25*output_frame.shape[0]/7) ) , cv2.FONT_HERSHEY_SIMPLEX,0.88, color_help, 2, cv2.LINE_AA)
        if mqttActive:
            person_data = json.dumps({'count':number_of_people,'total':total_number_people})
            client.publish("person", payload=person_data)
        if showInferenceStats:
            cv2.rectangle(output_frame,(0, 30),  (100, 0), (0,0,0), -1)
            cv2.putText(output_frame, 'People: {}'.format(str(number_of_people)), (0, 20) , cv2.FONT_HERSHEY_SIMPLEX,0.5, (44,245,131), 2, cv2.LINE_AA)
            cv2.rectangle(output_frame,(0,30), (250,60), (0,0,0), -1)
            cv2.putText(output_frame, 'Inference {} time: {} ms'.format(args.device, str(time_end_inference)) , (0, 50) , cv2.FONT_HERSHEY_SIMPLEX,0.5, (44,245,131), 2, cv2.LINE_AA)
            cv2.rectangle(output_frame,(int(output_frame.shape[1]/2-30), output_frame.shape[0] - 40), (int(output_frame.shape[1]/2+40), output_frame.shape[0]), (0,0,0), -1)
            cv2.putText(output_frame, 'FPS: ' + str(fps), (int(output_frame.shape[1]/2  - 20), int(output_frame.shape[0] - 10)) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (44,235,131), 2, cv2.LINE_AA)
            cv2.putText(output_frame, info_system, (0, int(output_frame.shape[0] - 10)) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (44,235,131), 2, cv2.LINE_AA)
            #cv2.putText(output_frame, info_model, (0, int(output_frame.shape[0] - 30)) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (44,235,131), 2, cv2.LINE_AA)
        if not args.headless:
            cv2.imshow("Output Video", output_frame)
            if args.debug:
                cv2.imshow("Original", frame)
                cv2.imshow("Past detection", past_frame)
            k = cv2.waitKey(1)
            #24fps
            if k == 27:
                break
            elif k == 32: #Space
                process = not process
        ### TODO: Send the frame to the FFMPEG server ###
        if not stopSending:
            output_resized = cv2.resize(output_frame, (768,432))
            sys.stdout.buffer.write(output_resized)
            sys.stdout.flush()

    print(colored("=================People counted======================", 'green'))
    print(colored("People Counted: {}".format(total_number_people), 'green'))
    print(colored("=====================================================", 'green'))
    print(colored("~~~~~~~~~~~~~~~~~~~~~~~~Performance~~~~~~~~~~~~~~~~~~~~~~~~", 'green'))
    if total_fps_measurements > 0:
        print(colored("Average FPS: " + str(int(total_fps/total_fps_measurements)), 'green'))
        print(colored("Average Inference Time: {} ms".format(str(int(avg_time_inference/total_fps_measurements))), 'green'))
        print(colored("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", 'green'))
    info = cpuinfo.get_cpu_info()
    finaldata = {
        "videofile": split(args.input)[1],
        "model": split(args.model)[1],
        "device": args.device,
        "processor": info['brand'],
        "people_counted": total_number_people,
        "avgFPS": int(total_fps/total_fps_measurements),
        "avgInferenceTimems": int(avg_time_inference/total_fps_measurements)
    }
    with open('ExperimentalData/{}_{}_{}.json'.format(finaldata['device'],splitext(basename(finaldata['model']))[0], splitext(basename(finaldata['videofile']))[0]), 'w') as outfile:
        json.dump(finaldata, outfile)

    cv2.destroyAllWindows()
    cap.release()
    client.disconnect()
    if args.o:
        out.release()



            ### TODO: Extract any desired stats from the results ###

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###




        ### TODO: Write an output image if `single_image_mode` ###


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    global mqttActive
    client, mqttActive = connect_mqtt(args)
    client.on_message = on_message
    client.subscribe("stopSending")
    client.subscribe("showInferenceStats")
    client.loop_start()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
