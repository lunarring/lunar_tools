import zmq
import json
import threading
import cv2
import numpy as np
from queue import Queue, Empty, Full
import time
from pythonosc import udp_client 
from threading import Thread
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
from lunar_tools.logprint import create_logger
from lunar_tools.fontrender import add_text_to_image


def get_local_ip():
    """
    Get the local IP address on Ubuntu/Linux systems.
    
    This function intelligently finds the local IP address by:
    1. First trying to parse ifconfig output to find active interfaces
    2. Looking for interfaces that are UP and RUNNING (not just UP)
    3. Prioritizing private network IP ranges (10.x, 192.168.x, 172.x)
    4. Falling back to socket-based detection if ifconfig fails
    
    Returns:
        str: The local IP address, or None if not found
    
    Example:
        >>> ip = get_local_ip()
        >>> print(f"Local IP: {ip}")
        Local IP: 10.40.49.109
    """
    import subprocess
    import re
    import socket
    
    # Method 1: Parse ifconfig output (most accurate for Linux)
    try:
        result = subprocess.run(['ifconfig'], capture_output=True,
                                text=True, check=True)
        output = result.stdout
        
        # Split by interface blocks (starts with interface name + colon)
        interface_blocks = re.split(r'\n(?=\w+:)', output)
        
        candidate_ips = []
        
        for block in interface_blocks:
            if not block.strip():
                continue
                
            # Check if interface is UP and RUNNING (active interface)
            if 'UP' in block and 'RUNNING' in block:
                # Find inet addresses in this interface block
                inet_pattern = r'inet (\d+\.\d+\.\d+\.\d+)'
                inet_matches = re.findall(inet_pattern, block)
                
                for ip in inet_matches:
                    # Skip localhost
                    if ip.startswith('127.'):
                        continue
                    
                    # Prioritize common private network ranges
                    if ip.startswith('10.'):
                        candidate_ips.insert(0, ip)  # Highest priority
                    elif ip.startswith('192.168.'):
                        candidate_ips.append(ip)     # Medium priority  
                    elif (ip.startswith('172.') and
                          16 <= int(ip.split('.')[1]) <= 31):
                        candidate_ips.append(ip)     # Medium priority
                    else:
                        candidate_ips.append(ip)     # Lowest priority
        
        if candidate_ips:
            return candidate_ips[0]
            
    except (subprocess.CalledProcessError, FileNotFoundError,
            IndexError, ValueError):
        pass
    
    # Method 2: Socket-based fallback (works on most systems)
    try:
        # Create socket and connect to remote address to get local IP
        # We use Google's DNS server, but don't actually send data
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            local_ip = s.getsockname()[0]
            
            # Verify it's not localhost
            if not local_ip.startswith('127.'):
                return local_ip
                
    except (socket.error, OSError):
        pass
    
    # Method 3: Last resort - get hostname IP
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        if not local_ip.startswith('127.'):
            return local_ip
    except (socket.error, OSError):
        pass
    
    return None


class ZMQPairEndpoint:
    def __init__(self, is_server, ip='localhost', port='5555', timeout=2, jpeg_quality=99, logger=None):
        self.address = f"tcp://{ip}:{port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.messages = Queue()
        self.send_queue = Queue()
        self.last_image = None
        self.last_audio = None
        self.logger = logger if logger else create_logger(__name__ + ".ZMQPairEndpoint")
        self.running = False
        timeout_seconds = float(timeout)
        self.timeout_ms = max(int(timeout_seconds * 1000), 1)
        self.poll_interval_ms = max(5, min(self.timeout_ms, 50))
        self._send_block_timeout = timeout_seconds if timeout_seconds > 0 else None

        # Set socket timeouts for both server and client
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        
        # Note: Heartbeat options are not supported for PAIR sockets
        # They are only available for DEALER/ROUTER and some other socket types

        if is_server:
            self.socket.bind(self.address)
        else:
            self.socket.connect(self.address)

        # Default encoding properties
        self.format = '.jpg'
        self.encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

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
            self._flush_outgoing()
            try:
                socks = dict(poller.poll(self.poll_interval_ms))
            except zmq.ZMQError as exc:
                if self.running:
                    self.logger.error(f"ZMQ poll error: {exc}")
                break

            if not self.running:
                break

            if self.socket in socks and socks[self.socket] & zmq.POLLIN:
                try:
                    self._handle_incoming()
                except Exception as exc:
                    self.logger.error(f"Exception in listen thread: {exc}")
                    break

            # Attempt to send any messages that arrived while processing input
            self._flush_outgoing()

        # Drain any remaining outgoing messages without sending once stopped
        while True:
            try:
                _, response_queue = self.send_queue.get_nowait()
            except Empty:
                break

            if response_queue is not None:
                response_queue.put_nowait(("error", zmq.Again("Endpoint stopped")))

    def _handle_incoming(self):
        try:
            message = self.socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return

        if message.startswith(b'img:'):
            nparr = np.frombuffer(message[4:], np.uint8)
            self.last_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            self.messages.put(('img', self.last_image))
        elif message.startswith(b'audio:'):
            # Parse audio message: audio:sample_rate:channels:dtype:data
            header_end = message.find(b':', 6)  # Find end of sample_rate
            if header_end == -1:
                return
            sample_rate = int(message[6:header_end])

            channels_end = message.find(b':', header_end + 1)  # Find end of channels
            if channels_end == -1:
                return
            channels = int(message[header_end + 1:channels_end])

            dtype_end = message.find(b':', channels_end + 1)  # Find end of dtype
            if dtype_end == -1:
                return
            dtype_str = message[channels_end + 1:dtype_end].decode('utf-8')

            # Extract audio data
            audio_data = message[dtype_end + 1:]
            audio_array = np.frombuffer(audio_data, dtype=np.dtype(dtype_str))

            # Reshape for stereo if needed
            if channels == 2:
                audio_array = audio_array.reshape(-1, 2)

            self.last_audio = {
                'data': audio_array,
                'sample_rate': sample_rate,
                'channels': channels,
                'dtype': dtype_str
            }
            self.messages.put(('audio', self.last_audio))
        else:
            json_data = json.loads(message.decode('utf-8'))
            self.messages.put(('json', json_data))

    def _flush_outgoing(self):
        while self.running:
            try:
                payload, response_queue = self.send_queue.get_nowait()
            except Empty:
                break

            if payload is None:
                if response_queue is not None:
                    response_queue.put_nowait(("ok", None))
                continue

            try:
                self.socket.send(payload)
            except zmq.Again as exc:
                if response_queue is not None:
                    response_queue.put_nowait(("error", exc))
                else:
                    self.logger.error("Send failed with EAGAIN and no response handle.")
            except zmq.ZMQError as exc:
                if response_queue is not None:
                    response_queue.put_nowait(("error", exc))
                else:
                    self.logger.error(f"Exception while sending message: {exc}")
            else:
                if response_queue is not None:
                    response_queue.put_nowait(("ok", None))

    def _queue_send(self, payload):
        if payload is None:
            self.send_queue.put_nowait((None, None))
            return

        response_queue = Queue(maxsize=1)
        try:
            if self._send_block_timeout is None:
                self.send_queue.put((payload, response_queue))
            else:
                self.send_queue.put((payload, response_queue), timeout=self._send_block_timeout)
        except Full as exc:
            raise zmq.Again("Send queue full: peer not accepting data before timeout") from exc

        try:
            status, exc = response_queue.get(timeout=self._send_block_timeout or None)
        except Empty as exc:
            raise zmq.Again("Send acknowledgement timed out") from exc

        if status == "error":
            raise exc
    

    def stop(self):
        # Signal the thread to stop
        self.running = False
        self._queue_send(None)

        # Wait for the listening thread to finish
        if (
            hasattr(self, "thread")
            and self.thread.is_alive()
            and threading.current_thread() is not self.thread
        ):
            self.thread.join()

        # Now safely close the socket
        if hasattr(self, "socket"):
            try:
                self.socket.close(linger=0)
            finally:
                self.socket = None

        if getattr(self, "context", None) is not None:
            try:
                self.context.term()
            finally:
                self.context = None


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

    def get_audio(self):
        """
        Retrieve the most recent audio data from the message queue.
        
        Returns:
            dict or None: Dictionary containing audio data and metadata:
                - 'data': numpy array with audio samples (1D for mono, 2D for stereo)
                - 'sample_rate': int, sample rate in Hz
                - 'channels': int, number of channels (1 for mono, 2 for stereo)
                - 'dtype': str, numpy dtype of the audio data
        """
        while not self.messages.empty():
            message_type, message_data = self.messages.get()
            if message_type == 'audio':
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
        self._queue_send(json_data)

    def send_img(self, img):
        success, buffer = cv2.imencode(self.format, img, self.encode_params)
        if not success:
            raise ValueError("Failed to encode image for transmission")
        self._queue_send(b'img:' + buffer.tobytes())
        # self.socket.send(b'img:' + buffer.tobytes(), zmq.NOBLOCK)
        
        # # Wait for up to 2 seconds for the message to be sent
        # poller = zmq.Poller()
        # poller.register(self.socket, zmq.POLLOUT)
        # socks = dict(poller.poll(2000))  # 2000 milliseconds = 2 seconds
        
        # if self.socket not in socks:
        #     print("Timeout occurred while sending image.")

    def send_audio(self, audio_data, sample_rate, channels=None):
        """
        Send audio data over ZMQ without any encoding/compression.
        
        Args:
            audio_data (numpy.ndarray): Audio samples. 
                - For mono: 1D array of shape (n_samples,)
                - For stereo: 2D array of shape (n_samples, 2) or 1D interleaved
            sample_rate (int): Sample rate in Hz (e.g., 44100, 48000)
            channels (int, optional): Number of channels (1 for mono, 2 for stereo).
                If None, inferred from audio_data shape.
        
        Example:
            # Mono audio
            mono_audio = np.random.randn(44100).astype(np.float32)  # 1 second at 44.1kHz
            endpoint.send_audio(mono_audio, 44100)
            
            # Stereo audio 
            stereo_audio = np.random.randn(44100, 2).astype(np.float32)
            endpoint.send_audio(stereo_audio, 44100)
        """
        # Convert to numpy array if not already
        audio_data = np.asarray(audio_data)
        
        # Infer channels from shape if not provided
        if channels is None:
            if audio_data.ndim == 1:
                channels = 1
            elif audio_data.ndim == 2:
                channels = audio_data.shape[1]
            else:
                raise ValueError("Audio data must be 1D (mono) or 2D (stereo)")
        
        # Validate channels
        if channels not in [1, 2]:
            raise ValueError("Only mono (1) and stereo (2) audio are supported")
        
        # Ensure correct shape for stereo
        if channels == 2 and audio_data.ndim == 2:
            # Flatten stereo data to interleaved format for transmission
            audio_data = audio_data.flatten()
        elif channels == 1 and audio_data.ndim == 2:
            raise ValueError("Mono audio should be 1D array")
        
        # Get dtype string
        dtype_str = str(audio_data.dtype)
        
        # Create message: audio:sample_rate:channels:dtype:data
        header = f"audio:{sample_rate}:{channels}:{dtype_str}:".encode('utf-8')
        message = header + audio_data.tobytes()
        
        self._queue_send(message)


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


    def process_incoming(self, *args):
        identifier = args[0]
        message = args[1]
        # print(f"process_incoming: {identifier} {message}")
        
        if identifier not in self.dict_messages.keys():
            self.dict_messages[identifier] = []
            
        if identifier not in self.dict_time.keys():
            self.dict_time[identifier] = []
            
        
        # buffer length
        if len(self.dict_messages[identifier]) >= self.BUFFER_SIZE:
            self.dict_messages[identifier].pop(0)
            self.dict_time[identifier].pop(0)
        
        try:
            message = float(message)
        except ValueError:
            print(f"Received non-numerical message: {message}")
            return
        self.dict_messages[identifier].append(message)
        self.dict_time[identifier].append(time.time())
        
        if self.verbose_high:
            # if identifier in self.filter_identifiers:
            print(f"OSCReceiver: {identifier} {message} from {self.ip_receiver}:{self.port_receiver}")
        

    def start_visualization(self, shape_hw_vis=(200, 300), nmb_cols_vis=3, nmb_rows_vis=3, backend=None):
        from lunar_tools.display_window import GridRenderer
        self.shape_hw_vis = shape_hw_vis
        self.nmb_cols_vis = nmb_cols_vis
        self.nmb_rows_vis = nmb_rows_vis
        self.list_images = []
        self.renderer = GridRenderer(nmb_cols=self.nmb_cols_vis, nmb_rows=self.nmb_rows_vis, shape_hw=self.shape_hw_vis, backend=backend, window_title='osc monitor')
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
            for identifier in sorted_keys[:max_items]:
                values = self.get_all_values(identifier)
                # Limit the number of values to the width of the image to avoid out-of-bounds errors
                values = np.array(values[-self.shape_hw_vis[1]:])
                # Determine the background color based on the last value received
                if len(values) > 0:
                    min_val = min(values)
                    max_val = max(values)
                    if max_val - min_val == 0:
                        grey_value = int(0)
                        values = [0]
                    else:
                        grey_value = int(round(self.low_val_vis + (self.high_val_vis - self.low_val_vis) * (values[-1] - min_val) / (max_val - min_val)))
                        grey_value = np.clip(grey_value, self.low_val_vis, self.high_val_vis)
                        rescaled_values = (values - min_val) / (max_val - min_val) * (self.shape_hw_vis[0]-1) + 1
                        values = rescaled_values
                else:
                    continue
                

                values = np.asarray(np.floor(values), dtype=np.int16)
                values = self.shape_hw_vis[0] - values
                valid_indices = (0 <= values) & (values < self.shape_hw_vis[0])
                curve_array = grey_value*np.ones((*self.shape_hw_vis, 3), dtype=np.uint8)  # Adding a third dimension for RGB channels
                curve_array[values[valid_indices].astype(int), np.arange(len(values))[valid_indices], 1] = 255  # Setting the green channel
                
                if len(self.dict_time[identifier]) > 1:
                    dt = np.mean(np.diff(self.dict_time[identifier][-10:]))
                    dt = int(1000*dt)
                else:
                    dt = 0
                text = f"{identifier} {dt}ms"
                image = add_text_to_image(curve_array, text, y_pos=0.01, font_color=(255,255,255))
                image = add_text_to_image(image, f"{max_val:.2e}", y_pos=0.01, align='left', font_color=(0,255,0), font_size=15)
                image = add_text_to_image(image, f"{min_val:.2e}", y_pos=0.99, align='left', font_color=(0,255,0), font_size=15)
                
                image = np.copy(np.asarray(image))

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
            
            if time.time() - self.dict_time[identifier][-1] > self.dt_timeout:
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
        

    def show_last_received(self):
        if not self.dict_time:
            print("Nothing was received since you started the receiver.")
        else:
            current_time = time.time()
            for identifier, timestamp_all in self.dict_time.items():
                time_since_received = current_time - timestamp_all[-1]
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

    # Bidirectional Audio Sending
    sample_rate = 44100
    duration = 1.0  # 1 second
    n_samples = int(sample_rate * duration)
    
    # Client sends mono audio to Server
    client_mono_audio = np.random.randn(n_samples).astype(np.float32) * 0.1
    client.send_audio(client_mono_audio, sample_rate)
    time.sleep(0.01)
    server_received_audio = server.get_audio()
    if server_received_audio is not None:
        print(f"Server received mono audio: {server_received_audio['data'].shape}, "
              f"sample_rate={server_received_audio['sample_rate']}, "
              f"channels={server_received_audio['channels']}")

    # Server sends stereo audio to Client  
    server_stereo_audio = np.random.randn(n_samples, 2).astype(np.float32) * 0.1
    server.send_audio(server_stereo_audio, sample_rate)
    time.sleep(0.01)
    client_received_audio = client.get_audio()
    if client_received_audio is not None:
        print(f"Client received stereo audio: {client_received_audio['data'].shape}, "
              f"sample_rate={client_received_audio['sample_rate']}, "
              f"channels={client_received_audio['channels']}")
        
if __name__ == "__main__":
    # import lunar_tools as lt
    import numpy as np
    import time
    import lunar_tools as lt
    
    receiver = OSCReceiver('10.40.48.97')
    receiver.start_visualization(shape_hw_vis=(300, 500), nmb_cols_vis=3, nmb_rows_vis=2)
    while True:
        receiver.show_visualization()
 
