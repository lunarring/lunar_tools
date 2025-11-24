import time
from threading import Thread

import numpy as np
from pythonosc import osc_server, udp_client
from pythonosc.dispatcher import Dispatcher

class OSCSender:
    def __init__(self, ip_receiver="127.0.0.1", port_receiver=8003, start_thread=False, verbose_high=False):
        if ip_receiver is None:
            raise ValueError("OSCSender requires a valid ip_receiver")
        if port_receiver is None:
            raise ValueError("OSCSender requires a valid port_receiver")
        self.ip_receiver = ip_receiver
        self.port_receiver = port_receiver
        try:
            self.client = udp_client.SimpleUDPClient(self.ip_receiver, self.port_receiver)
        except OSError as exc:
            raise OSError(f"Could not create OSC client for {self.ip_receiver}:{self.port_receiver}") from exc
        self.DELIM = " "
        self.verbose_high = verbose_high
        self._unused_start_thread = start_thread

    def send_message(self, identifier, message):
        self.client.send_message(identifier, message)
        if self.verbose_high:
            print(f"OSCSender: {identifier} {message} to {self.ip_receiver}:{self.port_receiver}")


class OSCReceiver:
    def __init__(
        self,
        ip_receiver=None,
        start=True,
        BUFFER_SIZE=500,
        rescale_all_input=False,
        dt_timeout=3,
        port_receiver=8003,
        verbose_high=False,
    ):
        self.ip_receiver = ip_receiver
        self.port_receiver = port_receiver
        self.rescale_all_input = rescale_all_input
        self.thread_osc = Thread(target=self.runfunc_thread_osc)
        self.running = False
        self.running_vis = False
        self.thread_vis = None
        self.renderer = None
        self.server = None
        self._add_text_to_image = None

        self.dict_messages = {}
        self.dict_time = {}
        self.dt_timeout = dt_timeout
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
        if len(args) < 2:
            if self.verbose_high:
                print("OSCReceiver: ignoring message without payload")
            return
        identifier = args[0]
        message = args[1]

        try:
            message = float(message)
        except (ValueError, TypeError):
            if self.verbose_high:
                print(f"Received non-numerical message on {identifier}: {message!r}")
            return
        if identifier not in self.dict_messages:
            self.dict_messages[identifier] = []
        if identifier not in self.dict_time:
            self.dict_time[identifier] = []

        if len(self.dict_messages[identifier]) >= self.BUFFER_SIZE:
            self.dict_messages[identifier].pop(0)
            self.dict_time[identifier].pop(0)
        self.dict_messages[identifier].append(message)
        self.dict_time[identifier].append(time.time())

        if self.verbose_high:
            print(f"OSCReceiver: {identifier} {message} from {self.ip_receiver}:{self.port_receiver}")

    def start_visualization(self, shape_hw_vis=(200, 300), nmb_cols_vis=3, nmb_rows_vis=3, backend=None):
        from lunar_tools.display_window import GridRenderer
        from lunar_tools.fontrender import add_text_to_image

        if self.running_vis:
            return
        self.shape_hw_vis = shape_hw_vis
        self.nmb_cols_vis = nmb_cols_vis
        self.nmb_rows_vis = nmb_rows_vis
        self.list_images = []
        self.renderer = GridRenderer(
            nmb_cols=self.nmb_cols_vis,
            nmb_rows=self.nmb_rows_vis,
            shape_hw=self.shape_hw_vis,
            backend=backend,
            window_title='osc monitor',
        )
        self.low_val_vis = 0
        self.high_val_vis = 30
        self.running_vis = True
        self.thread_vis = Thread(target=self.runfunc_thread_vis)
        self.thread_vis.start()
        self._add_text_to_image = add_text_to_image

    def runfunc_thread_vis(self):
        while self.running_vis:
            time.sleep(0.01)
            list_images = []
            sorted_keys = sorted(self.dict_messages.keys())
            max_items = self.nmb_cols_vis * self.nmb_rows_vis
            for identifier in sorted_keys[:max_items]:
                values = self.get_all_values(identifier)
                values = np.array(values[-self.shape_hw_vis[1]:])
                if len(values) > 0:
                    min_val = min(values)
                    max_val = max(values)
                    if max_val - min_val == 0:
                        grey_value = int(0)
                        values = [0]
                    else:
                        grey_value = int(
                            round(
                                self.low_val_vis
                                + (self.high_val_vis - self.low_val_vis) * (values[-1] - min_val) / (max_val - min_val)
                            )
                        )
                        grey_value = np.clip(grey_value, self.low_val_vis, self.high_val_vis)
                        rescaled_values = (values - min_val) / (max_val - min_val) * (self.shape_hw_vis[0] - 1) + 1
                        values = rescaled_values
                else:
                    continue

                values = np.asarray(np.floor(values), dtype=np.int16)
                values = self.shape_hw_vis[0] - values
                valid_indices = (0 <= values) & (values < self.shape_hw_vis[0])
                curve_array = grey_value * np.ones((*self.shape_hw_vis, 3), dtype=np.uint8)
                curve_array[values[valid_indices].astype(int), np.arange(len(values))[valid_indices], 1] = 255

                if len(self.dict_time[identifier]) > 1:
                    dt = np.mean(np.diff(self.dict_time[identifier][-10:]))
                    dt = int(1000 * dt)
                else:
                    dt = 0
                text = f"{identifier} {dt}ms"
                add_text = self._add_text_to_image
                if add_text is None:
                    raise RuntimeError("Visualization support requires lunar_tools.fontrender.")
                image = add_text(curve_array, text, y_pos=0.01, font_color=(255, 255, 255))
                image = add_text(image, f"{max_val:.2e}", y_pos=0.01, align='left', font_color=(0, 255, 0), font_size=15)
                image = add_text(image, f"{min_val:.2e}", y_pos=0.99, align='left', font_color=(0, 255, 0), font_size=15)

                image = np.copy(np.asarray(image))

                list_images.append(image)
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
        if self.running and self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.running = False
            if self.thread_osc.is_alive():
                self.thread_osc.join(timeout=1.0)
            print("OSC server stopped")
            self.server = None
        if self.running_vis:
            self.running_vis = False
            if self.thread_vis and self.thread_vis.is_alive():
                self.thread_vis.join(timeout=1.0)
            if self.renderer:
                try:
                    self.renderer.close()
                except AttributeError:
                    pass

    def get_last_value(self, identifier, val_min=0, val_max=1, val_default=None, rescale_this_input=False):
        if val_default is None:
            val_default = 0.5 * (val_min + val_max)

        if identifier in self.dict_messages.keys():
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
                value = val_min + (val_max - val_min) * fract

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


__all__ = ["OSCSender", "OSCReceiver"]
