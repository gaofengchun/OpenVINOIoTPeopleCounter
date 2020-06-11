#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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

import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore
from YoloParams import YoloParams                       #Import Yolo Parser
from math import exp as exp
from platform import processor, system


class Network:
    """
    Load and configure inference plugins for the specified target devices
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        ### TODO: Initialize any class variables desired ###
        self.net = None
        self.model = None
        self.core = None
        self.input_layer = None
        self.output_layer = None
        self.device = ""
        self.model_xml = ""
        self.model_bin = ""
        self.infer_request = None
        self.unsupported_layers = ""
        self.camera_height = 0
        self.camera_width = 0
        self.input_height = 0
        self.input_width = 0
        self.prob_threshold = 0.7
        self.anchors = [10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326]
        self.yolo_scale_13 = 13
        self.yolo_scale_26 = 26
        self.yolo_scale_52 = 52
        self.num = 3
        self.coords = 4
        self.classes = 80
    def load_model(self, model_location, device="CPU", cpu_extension=None):
        ### TODO: Load the model ###
        ### TODO: Check for supported layers ###
        ### TODO: Add any necessary extensions ###
        ### TODO: Return the loaded inference plugin ###
        ### Note: You may need to update the function parameters. ###
        self.model_xml = os.path.splitext(model_location)[0] + ".xml"
        self.model_bin = os.path.splitext(model_location)[0] + ".bin"
        self.device = device
        self.core = IECore()
        self.net = self.core.load_network(self.model, device_name = self.device, num_requests = 1)
        if 'arm' in platform():
            self.model = self.core.load_network(network=self.net, device_name=self.device, num_requests=1)
        else:
            self.model = self.core.read_network(model=self.model_xml, weights=self.model_bin) #Changing because deprecated
        #Check for incompatible layers
        layers_map = self.core.query_network(network=self.model, device_name=self.device)
        for key, value in layers_map.items():
            if not value == self.device:
                self.unsupported_layers += key
                self.unsupported_layers += "\n"

        if not self.unsupported_layers == "":
            print("The following layers are not supported: {}".format(self.unsupported_layers))
            print("Please, load the appropiate extension of your device, in order to attempt to fix that issue or contact to the person who converted the model")
        else:
            print("All the layers of the network are supported in this {} device".format(self.device))
        if cpu_extension:
            self.core.add_extension(cpu_extension, device_name=self.device)
        self.input_layer = next(iter(self.model.inputs))
        self.output_layer = next(iter(self.model.outputs))
        self.infer_request = 0
        return self.net
    def inference_parameters(self, height, width, confidence):
        self.camera_width = width
        self.camera_height = height
        self.prob_threshold = confidence
    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        self.input_height = self.model.inputs[self.input_layer].shape[2]
        self.input_width = self.model.inputs[self.input_layer].shape[3]
        return self.model.inputs[self.input_layer].shape

    def exec_net(self, image):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.net.start_async(request_id=self.infer_request, inputs={self.input_layer: image})
        return

    def wait(self):
        ### TODO: Wait for the request to be complete. ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        status = self.net.requests[self.infer_request].wait(-1)
        return status

    def get_output(self):
        ### TODO: Extract and return the output results
        ### Note: You may need to update the function parameters. ###
        ### Return a vector of bounding boxes representing of the object.
        ##Type {'xmin': 297, 'xmax': 328, 'ymin': 612, 'ymax': 804, 'class_id': 0, 'confidence': 0.85626584}
        outputs = self.net.requests[self.infer_request].outputs
        print(outputs)
        objects = list()
        if 'yolo' in self.model_xml.lower():
            for layer_name, out_blob in outputs.items():
                out_blob = out_blob.reshape(self.net.requests[self.infer_request].outputs[layer_name].shape)
                print(self.net.requests[self.infer_request].outputs[layer_name].shape)
                layer_params = YoloParams(self.model.layers[layer_name].params, out_blob.shape[2])
                objects += self.parse_yolo_region(out_blob, (self.input_height, self.input_width),
                                             (self.camera_height, self.camera_width), layer_params,
                                             self.prob_threshold)
            objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
            objects_to_delete = []
            for i in range(len(objects)):
                if objects[i]['confidence'] == 0:
                    continue
                for j in range(i + 1, len(objects)):
                    if self.intersection_over_union(objects[i], objects[j]) > self.prob_threshold:
                        objects[j]['confidence'] = 0
                        objects_to_delete.append(objects[j])
            for i in objects_to_delete:
                objects.remove(i)
        else:
            for i in outputs[self.output_layer][0][0]:
                if i[2] >= self.prob_threshold:
                    objects.push({'xmin': int(i[3]*self.camera_width), 'xmax': int(i[5]*self.camera_width), 'ymin': int(i[4]*self.camera_height), 'ymax': int(i[6]*self.camera_height), 'class_id': i[1], 'confidence': i[2]})
        return objects

    #Define a Yolo Parser Region Taken from OpenVINO examples
    def parse_yolo_region(self, blob, resized_image_shape, original_im_shape, params, threshold):
    # ------------------------------------------ Validating output parameters ------------------------------------------
        _, _, out_blob_h, out_blob_w = blob.shape
        assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                         "be equal to width. Current height = {}, current width = {}" \
                                         "".format(out_blob_h, out_blob_w)

        # ------------------------------------------ Extracting layer parameters -------------------------------------------
        orig_im_h, orig_im_w = original_im_shape
        resized_image_h, resized_image_w = resized_image_shape
        objects = list()
        predictions = blob.flatten()
        side_square = params.side * params.side

        # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
        for i in range(side_square):
            row = i // params.side
            col = i % params.side
            for n in range(params.num):
                obj_index = self.entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
                scale = predictions[obj_index]
                if scale < threshold:
                    continue
                box_index = self.entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
                # Network produces location predictions in absolute coordinates of feature maps.
                # Scale it to relative coordinates.
                x = (col + predictions[box_index + 0 * side_square]) / params.side
                y = (row + predictions[box_index + 1 * side_square]) / params.side
                # Value for exp is very big number in some cases so following construction is using here
                try:
                    w_exp = exp(predictions[box_index + 2 * side_square])
                    h_exp = exp(predictions[box_index + 3 * side_square])
                except OverflowError:
                    continue
                # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
                w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
                h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
                for j in range(params.classes):
                    class_index = self.entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                              params.coords + 1 + j)
                    confidence = scale * predictions[class_index]
                    if confidence < threshold:
                        continue
                    objects.append(self.scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                              h_scale=orig_im_h, w_scale=orig_im_w))
        return objects
    def entry_index(self, side, coord, classes, location, entry):
        side_power_2 = side ** 2
        n = location // side_power_2
        loc = location % side_power_2
        return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)
    def scale_bbox(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        xmin = int((x - w / 2) * w_scale)
        ymin = int((y - h / 2) * h_scale)
        xmax = int(xmin + w * w_scale)
        ymax = int(ymin + h * h_scale)
        return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)
    def intersection_over_union(self, box_1, box_2):
        width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
        height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
        if width_of_overlap_area < 0 or height_of_overlap_area < 0:
            area_of_overlap = 0
        else:
            area_of_overlap = width_of_overlap_area * height_of_overlap_area
        box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
        box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
        area_of_union = box_1_area + box_2_area - area_of_overlap
        if area_of_union == 0:
            return 0
        return area_of_overlap / area_of_union
