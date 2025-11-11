from __future__ import annotations

from lunar_tools.comms import CommunicationServices, MessageBusConfig, create_message_bus
from lunar_tools.services.comms.message_bus import MessageBusService
from tests.fakes import FakeMessageReceiver, FakeMessageSender


def test_message_bus_send_uses_default_address():
    sender = FakeMessageSender()
    bus = MessageBusService()
    bus.register_sender("osc", sender, default_address="/lunar")

    bus.send("osc", {"payload": 1.0})

    assert sender.sent == [("/lunar", {"payload": 1.0})]


def test_message_bus_broadcast_routes_to_all_senders():
    first = FakeMessageSender()
    second = FakeMessageSender()
    bus = MessageBusService()
    bus.register_sender("first", first, default_address="addr1")
    bus.register_sender("second", second, default_address="addr2")

    bus.broadcast("ping")

    assert first.sent == [("addr1", "ping")]
    assert second.sent == [("addr2", "ping")]


def test_message_bus_register_receiver_auto_starts_and_polls():
    receiver = FakeMessageReceiver()
    receiver.inject("addr", {"value": 42})

    bus = MessageBusService()
    bus.register_receiver("osc", receiver, default_address="addr", auto_start=True)

    message = bus.poll("osc")

    assert receiver.start_calls == 1
    assert message == {"address": "addr", "payload": {"value": 42}}


def test_message_bus_wait_for_times_out():
    receiver = FakeMessageReceiver()
    bus = MessageBusService()
    bus.register_receiver("osc", receiver, auto_start=False)

    result = bus.wait_for("osc", timeout=0.05, poll_interval=0.01)

    assert result is None


def test_create_message_bus_without_extras():
    services = create_message_bus()

    assert isinstance(services, CommunicationServices)
    assert isinstance(services.message_bus, MessageBusService)
    assert services.osc_sender is None
    assert services.zmq_endpoint is None

