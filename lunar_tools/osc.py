#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pythonosc import udp_client 
from threading import Thread
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server

#%% 

class OSCSender():
    def __init__(self,
                 ip_server = None,
                 port_server = 8003,
                 start_thread = False,
                 verbose_high = False,
                 ):
        
        self.ip_server = ip_server
        self.port_server = port_server
        self.client = udp_client.SimpleUDPClient(self.ip_server, self.port_server)
        self.DELIM = " "
        self.verbose_high = verbose_high
        
    def send_message(self, identifier, message):
        self.client.send_message(identifier, message)
        if self.verbose_high:
            print(f"OSCSender: {identifier} {message} to {self.ip_server}:{self.port_server}")
    
    def test_message(self):
        for x in range(10):
            self.send_message("/filter", random.random())
            print(f"send message to: {self.ip_server}")
            time.sleep(1)


class OSCReceiver():
    def __init__(self,
                 ip_server = None, 
                 start_thread = True,
                 BUFFER_SIZE = 500,
                 force_fract = False, # forcing adaptive rescaling of values between [0, 1]
                 dt_timeout = 3, # if after this dt nothing new arrived, send back le default
                 port_server = 8003,
                 verbose_high = False,
                 ):
        
        # Start a new thread to read asynchronously
        self.ip_server = ip_server
        self.port_server = port_server
        self.force_fract = force_fract
        if start_thread:
            self.thread = Thread(target=self.runfunc_thread)
            self.thread.start()
        
        self.dict_messages = {}
        self.dict_time = {}
        self.dt_timeout = dt_timeout # if after this dt nothing new arrived, send back le default
        self.BUFFER_SIZE = BUFFER_SIZE
        self.verbose_high = verbose_high
        self.filter_identifiers = []
            
        
    def runfunc_thread(self):
        dispatcher = Dispatcher()
        dispatcher.map('/*', self.process_incoming)
        server = osc_server.ThreadingOSCUDPServer((self.ip_server, self.port_server), dispatcher)
        print("Serving on {}".format(server.server_address))
        server.serve_forever()


    def process_incoming(self, *args):
        identifier = args[0]
        message = args[1]
        # print(f"process_incoming: {identifier} {message}")
        
        if identifier not in self.dict_messages.keys():
            self.dict_messages[identifier] = []
            
            
        if identifier not in self.dict_time.keys():
            self.dict_time[identifier] = 0
        self.dict_time[identifier] = time.time()
            
        
        # buffer length
        if len(self.dict_messages[identifier]) >= self.BUFFER_SIZE:
            self.dict_messages[identifier].pop(0)
        
        self.dict_messages[identifier].append(message)
        
        if self.verbose_high:
            # if identifier in self.filter_identifiers:
            print(f"OSCReceiver: {identifier} {message} from {self.ip_server}:{self.port_server}")
        
        
    def get_last_value(self, 
                       identifier, 
                       val_min=0, 
                       val_max=1, 
                       val_default=None, 
                       force_fract_this_var=False
                       ):
        
        if val_default is None:
            val_default = 0.5*(val_min+val_max)
            
        if identifier in self.dict_messages.keys():
            # Check if timeout applies
            
            if time.time() - self.dict_time[identifier] > self.dt_timeout:
                return val_default
            
            value = self.dict_messages[identifier][-1]
            if self.force_fract or force_fract_this_var:
                
                minval = np.min(self.dict_messages[identifier])
                maxval = np.max(self.dict_messages[identifier])
                
                if maxval - minval == 0:
                    fract = 1
                else:
                    fract = (value - minval) / (maxval - minval)
                    fract = np.clip(fract, 0, 1)
                value = val_min + (val_max-val_min)*fract
                
            return value
        else:
            if self.verbose_high:
                print(f"ERROR get_last_value: identifier {identifier} was never received!")
            return val_default
        
    def get_all_values(self, identifier):
        if identifier in self.dict_messages.keys():
            if len(self.dict_messages[identifier]) >= 2:
                
                return self.dict_messages[identifier]
            else:
                return [0,0]
        else:
            if self.verbose_high:
                print(f"ERROR get_last_value: identifier {identifier} was never received!")
            return [0,0]
        
    def plot_nice(self):
        for j, identifier in enumerate(self.dict_messages.keys()):
            data = self.get_all_values(identifier)
            plt.plot(data)
            plt.title(identifier)
            plt.show()


if __name__ == "__main__":
    
    sender = OSCSender('localhost')
    receiver = OSCReceiver('localhost')
    
    for i in range(10):
        time.sleep(0.1)
        # sends two sinewaves to the respective osc variables
        val1 = (np.sin(0.5*time.time())+1)*0.5
        val2 = (np.cos(0.5*time.time())+1)*0.5
        sender.send_message("/env1", val1)
        sender.send_message("/env2", val2)
    
    
    receiver.get_all_values("/env1")
    
    
    
    # test_clientserver = True
    # generate_signal = False
    
    
    # if test_clientserver:
    #     # Get a sender
    #     fs = OSCSender('johannes_mac')

    #     # Get a receiver
    #     fl = OSCReceiver('borneo')
        
    #     # Here you receive messages
    #     fl.get_all_values("/brightness")
    #     fl.get_last_value("/buba_420")
        
    # if generate_signal:
    #     fs = OSCSender(ip_server="192.168.0.54")
        
    #     while True:
    #         time.sleep(0.1)
    #         # sends two sinewaves to the respective osc variables
    #         val1 = (np.sin(0.5*time.time())+1)*0.5
    #         val2 = (np.cos(0.5*time.time())+1)*0.5
    #         fs.send_message("/env1", val1)
    #         fs.send_message("/env2", val2)
    
    
