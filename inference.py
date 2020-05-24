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
        self.model = self.core.read_network(model=self.model_xml, weights=self.model_bin) #Changing because deprecated
        self.net = self.core.load_network(self.model, device_name = self.device, num_requests = 1)
        if cpu_extension:
            self.core.add_extension(cpu_extension, device_name=self.device)
        self.input_layer = next(iter(self.model.inputs))
        self.output_layer = next(iter(self.model.outputs))
        self.infer_request = 0
        return self.net

    def get_input_shape(self):
        ### TODO: Return the shape of the input layer ###
        return self.model.inputs[self.input_layer].shape

    def exec_net(self, image, ):
        ### TODO: Start an asynchronous request ###
        ### TODO: Return any necessary information ###
        ### Note: You may need to update the function parameters. ###
        self.net.start_async(request_id=self.infer_request, inputs={self.input_layer: image})
        while True:
            if self.wait() == 0:
                break
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
        return self.net.requests[self.infer_request].outputs[self.output_layer]
