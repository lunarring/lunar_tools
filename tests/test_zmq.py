import socket
import time

import numpy as np
import pytest
import zmq

from lunar_tools.comms import ZMQPairEndpoint


def _pick_free_port() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return str(sock.getsockname()[1])


@pytest.fixture
def zmq_pair():
    port = _pick_free_port()
    server = ZMQPairEndpoint(is_server=True, ip="127.0.0.1", port=port, timeout=0.5)
    client = ZMQPairEndpoint(is_server=False, ip="127.0.0.1", port=port, timeout=0.5)

    try:
        yield server, client
    finally:
        client.stop()
        server.stop()


def _wait_for_json(endpoint, timeout=1.0):
    deadline = time.time() + timeout
    collected = []
    while time.time() < deadline and not collected:
        collected = endpoint.get_messages()
        if collected:
            break
        time.sleep(0.01)
    return collected


def _wait_for_img(endpoint, timeout=1.0):
    deadline = time.time() + timeout
    img = None
    while time.time() < deadline and img is None:
        img = endpoint.get_img()
        if img is not None:
            break
        time.sleep(0.01)
    return img


def test_addresses_match_configuration(zmq_pair):
    server, client = zmq_pair
    assert server.address == client.address
    assert server.address.startswith("tcp://127.0.0.1")


def test_client_to_server_json(zmq_pair):
    server, client = zmq_pair
    client.send_json({"message": "Hello, server!"})
    messages = _wait_for_json(server)

    assert messages == [{"message": "Hello, server!"}]


def test_server_to_client_json(zmq_pair):
    server, client = zmq_pair
    server.send_json({"response": "Hello, client!"})
    messages = _wait_for_json(client)

    assert messages == [{"response": "Hello, client!"}]


def test_bidirectional_json(zmq_pair):
    server, client = zmq_pair
    client.send_json({"message": "Client to Server"})
    server.send_json({"response": "Server to Client"})

    server_msgs = _wait_for_json(server)
    client_msgs = _wait_for_json(client)

    assert {"message": "Client to Server"} in server_msgs
    assert {"response": "Server to Client"} in client_msgs


def test_image_round_trip(zmq_pair):
    server, client = zmq_pair
    client_image = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    server_image = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)

    client.send_img(client_image)
    server.send_img(server_image)

    received_server = _wait_for_img(server)
    received_client = _wait_for_img(client)

    assert received_server is not None and received_server.shape == (32, 32, 3)
    assert received_client is not None and received_client.shape == (16, 16, 3)


def test_send_blocks_when_peer_unavailable():
    port = _pick_free_port()
    server = ZMQPairEndpoint(is_server=True, ip="127.0.0.1", port=port, timeout=0.2)
    try:
        start = time.perf_counter()
        with pytest.raises(zmq.Again):
            server.send_json({"message": "should timeout"})
        elapsed = time.perf_counter() - start
        assert elapsed >= 0.18
    finally:
        server.stop()
