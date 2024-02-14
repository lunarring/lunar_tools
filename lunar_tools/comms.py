import zmq
import json
import threading
import cv2
import numpy as np
from queue import Queue
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pythonosc import udp_client 
from threading import Thread
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
from lunar_tools.logprint import LogPrint

class ZMQPairEndpoint:
    def __init__(self, is_server, ip='localhost', port='5555', timeout=2, logger=None):
        self.address = f"tcp://{ip}:{port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.messages = Queue()
        self.last_image = None
        self.logger = logger if logger else LogPrint()
        self.running = False

        if is_server:
            self.socket.bind(self.address)
        else:
            self.socket.connect(self.address)
            self.socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)
            self.socket.setsockopt(zmq.SNDTIMEO, timeout * 1000)

        # Default encoding properties
        self.format = '.jpg'
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, 95]

        # Start listening thread
        self.thread = threading.Thread(target=self.listen, daemon=True)
        self.start()

    def start(self):
        self.thread.start()

    def listen(self):
        self.running = True
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while self.running:
            # Wait for a message for a short time
            socks = dict(poller.poll(1000))  # timeout in milliseconds
            if self.socket in socks:
                try:
                    message = self.socket.recv(zmq.NOBLOCK)
                    if message.startswith(b'img:'):
                        nparr = np.frombuffer(message[4:], np.uint8)
                        self.last_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        self.messages.put(('img', self.last_image))
                    else:
                        json_data = json.loads(message.decode('utf-8'))
                        self.messages.put(('json', json_data))
                    
                    
                except zmq.Again:
                    continue
                except Exception as e:
                    self.logger.error(f"Exception in listen thread: {e}")
                    break
    

    def stop(self):
        # Signal the thread to stop
        self.running = False

        # Wait for the listening thread to finish
        if self.thread.is_alive():
            self.thread.join()

        # Now safely close the socket
        self.socket.close()


    def get_messages(self):
        messages = []
        while not self.messages.empty():
            message_type, message_data = self.messages.get()
            if message_type == 'json':
                messages.append(message_data)
        return messages

    def get_img(self):
        while not self.messages.empty():
            message_type, message_data = self.messages.get()
            if message_type == 'img':
                return message_data
        return None

    def configure_image_encoding(self, format='.jpg', **params):
        self.format = format
        self.encode_params = []
        for key, value in params.items():
            if key == 'jpeg_quality':
                self.encode_params.extend([cv2.IMWRITE_JPEG_QUALITY, value])
            elif key == 'png_compression':
                self.encode_params.extend([cv2.IMWRITE_PNG_COMPRESSION, value])
            elif key == 'webp_quality':
                self.encode_params.extend([cv2.IMWRITE_WEBP_QUALITY, value])

    def send_json(self, data):
        json_data = json.dumps(data).encode('utf-8')
        self.socket.send(json_data)

    def send_img(self, img):
        _, buffer = cv2.imencode(self.format, img, self.encode_params)
        self.socket.send(b'img:' + buffer.tobytes())


#%% 

class OSCSender():
    def __init__(self,
                 ip_receiver = None,
                 port_receiver = 8003,
                 start_thread = False,
                 verbose_high = False,
                 ):
        
        self.ip_receiver = ip_receiver
        self.port_receiver = port_receiver
        self.client = udp_client.SimpleUDPClient(self.ip_receiver, self.port_receiver)
        self.DELIM = " "
        self.verbose_high = verbose_high
        
    def send_message(self, identifier, message):
        self.client.send_message(identifier, message)
        if self.verbose_high:
            print(f"OSCSender: {identifier} {message} to {self.ip_receiver}:{self.port_receiver}")
    

class OSCReceiver():
    def __init__(self,
                 ip_receiver = None, 
                 start = True,
                 BUFFER_SIZE = 500,
                 rescale_all_input = False, # forcing adaptive rescaling of values between [0, 1]
                 dt_timeout = 3, # if after this dt nothing new arrived, send back le default
                 port_receiver = 8003,
                 verbose_high = False,
                 ):
        
        # Start a new thread to read asynchronously
        self.ip_receiver = ip_receiver
        self.port_receiver = port_receiver
        self.rescale_all_input = rescale_all_input
        self.thread_osc = Thread(target=self.runfunc_thread_osc)
        self.running = False

        self.dict_messages = {}
        self.dict_time = {}
        self.dt_timeout = dt_timeout # if after this dt nothing new arrived, send back le default
        self.BUFFER_SIZE = BUFFER_SIZE
        self.verbose_high = verbose_high
        self.filter_identifiers = []
            
        if start:
            self.start()
    
    def runfunc_thread_osc(self):
        self.running = True
        dispatcher = Dispatcher()
        dispatcher.map('/*', self.process_incoming)
        self.server = osc_server.ThreadingOSCUDPServer((self.ip_receiver, self.port_receiver), dispatcher)
        print("Serving on {}".format(self.server.server_address))
        self.server.serve_forever()

    def start(self):
        self.thread_osc.start()

    def start_visualization(self, shape_hw_vis=(200, 300), nmb_cols_vis=3, nmb_rows_vis=3):
        self.shape_hw_vis = shape_hw_vis
        self.nmb_cols_vis = nmb_cols_vis
        self.nmb_rows_vis = nmb_rows_vis
        self.list_images = []
        self.renderer = lt.GridRenderer(nmb_cols=self.nmb_cols_vis, nmb_rows=self.nmb_rows_vis, shape_hw=self.shape_hw_vis)
        self.low_val_vis = 0
        self.high_val_vis = 30
        self.running_vis = True
        self.thread_vis = Thread(target=self.runfunc_thread_vis)
        self.thread_vis.start()

    def runfunc_thread_vis(self):
        while self.running_vis:
            time.sleep(0.01)
            list_images = []
            # Sort the keys of dict_messages alphabetically
            sorted_keys = sorted(self.dict_messages.keys())
            # Truncate the sorted keys to limit the number of items processed
            max_items = self.nmb_cols_vis * self.nmb_rows_vis
            for key in sorted_keys[:max_items]:
                values = self.get_all_values(key)
                # Limit the number of values to the width of the image to avoid out-of-bounds errors
                values = values[-self.shape_hw_vis[1]:]  # Only keep the last 'shape_hw[1]' number of values
                # Determine the background color based on the last value received
                last_value = values[-1] if values else 0  # Use the last value or 0 if no values
                gray_value = np.interp(last_value, (min(values), max(values)), (self.low_val_vis, self.high_val_vis))
                gray_value = int(gray_value)
                # Create an image with a uniform gray background
                image = np.ones((self.shape_hw_vis[0], self.shape_hw_vis[1], 3), dtype=np.uint8) * gray_value
                # Normalize the values to fit in the image height
                normalized_values = np.interp(values, (min(values), max(values)), (0, self.shape_hw_vis[0]-1))
                # Convert to integer for pixel indices
                normalized_values = normalized_values.astype(np.uint8)
                # Draw the curve in red
                image = lt.add_text_to_image(image, key, y_pos=0.01, font_color=(255,255,255))
                image = np.copy(np.asarray(image))
                for i in range(1, len(normalized_values)):
                    image[self.shape_hw_vis[0] - 1 - normalized_values[i], i] = (0, 255, 0)  
                list_images.append(image)
                # Stop filling the list if it reaches the maximum number of items
                if len(list_images) >= max_items:
                    break
            self.list_images = list_images 

    def show_visualization(self): 
        if self.running_vis:
            self.renderer.update(self.list_images)
            self.renderer.render()
        else:
            return
        

    def stop(self):
        if self.running:
            self.server.shutdown()
            self.server.server_close()
            print("OSC server stopped")
            self.running = False

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
            print(f"OSCReceiver: {identifier} {message} from {self.ip_receiver}:{self.port_receiver}")
        
        
    def get_last_value(self, 
                       identifier, 
                       val_min=0, 
                       val_max=1, 
                       val_default=None, 
                       rescale_this_input=False
                       ):
        
        if val_default is None:
            val_default = 0.5*(val_min+val_max)
            
        if identifier in self.dict_messages.keys():
            # Check if timeout applies
            
            if time.time() - self.dict_time[identifier] > self.dt_timeout:
                return val_default
            
            value = self.dict_messages[identifier][-1]
            if self.rescale_all_input or rescale_this_input:
                
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
            if len(self.dict_messages[identifier]) >= 1:
                
                return self.dict_messages[identifier]
        
    def plot_nice(self):
        for j, identifier in enumerate(self.dict_messages.keys()):
            data = self.get_all_values(identifier)
            plt.plot(data)
            plt.title(identifier)
            plt.show()

    def show_last_received(self):
        if not self.dict_time:
            print("Nothing was received since you started the receiver.")
        else:
            current_time = time.time()
            for identifier, timestamp in self.dict_time.items():
                time_since_received = current_time - timestamp
                print(f"Signal '{identifier}' was last received {time_since_received:.2f} seconds ago.")



# Example usage ZMQ
if __name__ == "__main__xxx":
    import numpy as np
    import time

    # Create server and client
    server = ZMQPairEndpoint(is_server=True, ip='127.0.0.1', port='5556')
    client = ZMQPairEndpoint(is_server=False, ip='127.0.0.1', port='5556')
    
    # Client: Send JSON to Server
    client.send_json({"message": "Hello from Client!"})
    time.sleep(0.01)
    # Server: Check for received messages
    server_msgs = server.get_messages()
    print("Messages received by server:", server_msgs)
    
    # Server: Send JSON to Client
    server.send_json({"response": "Hello from Server!"})
    time.sleep(0.01)

    # Client: Check for received messages
    client_msgs = client.get_messages()
    print("Messages received by client:", client_msgs)

    # Bidirectional Image Sending
    sz = (800, 800)
    client_image = np.random.randint(0, 256, (sz[0], sz[1], 3), dtype=np.uint8)
    server_image = np.random.randint(0, 256, (sz[0], sz[1], 3), dtype=np.uint8)

    # Client sends image to Server
    client.send_img(client_image)
    time.sleep(0.01)
    server_received_image = server.get_img()
    if server_received_image is not None:
        print("Server received image from Client")

    # Server sends image to Client
    server.send_img(server_image)
    time.sleep(0.01)
    client_received_image = client.get_img()
    if client_received_image is not None:
        print("Client received image from Server")
        
if __name__ == "__main__":
    # import lunar_tools as lt
    import numpy as np
    import time
    import lunar_tools as lt
    
    receiver = OSCReceiver('10.40.48.97')
    receiver.start_visualization()
    while True:
        receiver.show_visualization()
 