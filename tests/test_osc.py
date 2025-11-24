import importlib.util
import os
import socket
import string
import sys
import threading
import time
import types
import unittest
from pathlib import Path

import numpy as np

if "elevenlabs.client" not in sys.modules:
    eleven_mod = types.ModuleType("elevenlabs")
    eleven_client_mod = types.ModuleType("elevenlabs.client")

    class _DummyElevenLabs:
        pass

    eleven_client_mod.ElevenLabs = _DummyElevenLabs
    eleven_mod.Voice = type("Voice", (), {})
    eleven_mod.VoiceSettings = type("VoiceSettings", (), {})

    def _noop(*_, **__):
        return None

    eleven_mod.play = _noop
    eleven_mod.save = _noop
    sys.modules["elevenlabs"] = eleven_mod
    sys.modules["elevenlabs.client"] = eleven_client_mod
    eleven_mod.client = eleven_client_mod

OSC_MODULE = Path(__file__).resolve().parents[1] / "lunar_tools" / "comms" / "osc.py"
spec = importlib.util.spec_from_file_location("lt_osc", OSC_MODULE)
osc_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(osc_module)
OSCSender = osc_module.OSCSender
OSCReceiver = osc_module.OSCReceiver


class TestOSCReceiverClient(unittest.TestCase):
    def setUp(self):
        port = self._reserve_port()
        if port is None:
            self.skipTest("UDP sockets are not permitted in this environment.")
        self.port = port
        self.sender = OSCSender('127.0.0.1', port_receiver=self.port)
        self.receiver = OSCReceiver('127.0.0.1', port_receiver=self.port)

    def tearDown(self):
        if hasattr(self, "receiver"):
            self.receiver.stop()

    @staticmethod
    def _reserve_port():
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.bind(("127.0.0.1", 0))
                return sock.getsockname()[1]
        except PermissionError:
            return None

    def test_received_short(self):
        time.sleep(0.1)
        self.sender.send_message("/env1", 42)
        time.sleep(0.1)
        data = self.receiver.get_all_values("/env1")
        if data is not None:
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0], 42)
        else:
            self.fail("No data received from OSC receiver.")

    def test_received_long(self):
        time.sleep(0.1)
        for i in range(10):
            self.sender.send_message("/env1", i)
            time.sleep(0.1)
        data = self.receiver.get_all_values("/env1")
        if data is not None:
            self.assertEqual(len(data), 10)
            for i in range(10):
                self.assertEqual(data[i], i)
        else:
            self.fail("No data received from OSC receiver.")


class TestOSCReceiverUnit(unittest.TestCase):
    def setUp(self):
        self.receiver = OSCReceiver('127.0.0.1', start=False)

    def test_sender_requires_ip(self):
        with self.assertRaises(ValueError):
            OSCSender(ip_receiver=None)

    def test_process_incoming_requires_payload(self):
        self.receiver.process_incoming("/env")
        self.assertNotIn("/env", self.receiver.dict_messages)

    def test_process_incoming_non_numeric(self):
        self.receiver.process_incoming("/env", "not-a-number")
        self.assertNotIn("/env", self.receiver.dict_messages)

    def test_process_incoming_numeric(self):
        self.receiver.process_incoming("/env", 10)
        self.assertIn("/env", self.receiver.dict_messages)
        self.assertEqual(self.receiver.dict_messages["/env"][-1], 10.0)

    def test_stop_shuts_down_visualization(self):
        class DummyRenderer:
            def __init__(self):
                self.closed = False

            def close(self):
                self.closed = True

        self.receiver.running_vis = True
        stopper = threading.Event()

        def _worker():
            stopper.wait(0.1)

        thread = threading.Thread(target=_worker)
        thread.start()
        self.receiver.thread_vis = thread
        renderer = DummyRenderer()
        self.receiver.renderer = renderer
        self.receiver.stop()
        self.assertFalse(self.receiver.running_vis)
        self.assertTrue(renderer.closed)


if __name__ == "__main__":
    unittest.main()
