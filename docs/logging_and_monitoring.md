# Logging and Monitoring

These utilities help observe the state of a running installation—whether you need a colour-coded console log, frame timing metrics, or remote health pings.

## Logging

`create_logger` wraps Python’s logging module with ANSI-coloured output and optional file handlers.

```python
import lunar_tools as lt

import logging

logger = lt.create_logger(
    "lunar.demo",
    level=logging.INFO,
    file_path="logs/demo.log",
    file_level=logging.DEBUG,
)

logger.info("Boot sequence started")
logger.warning("Projector link is unstable")
```

Use `dynamic_print` for live status lines that update in place (progress bars, FPS counters, etc.).

```python
from lunar_tools import dynamic_print

dynamic_print("Rendering frame 120/240…")
```

## FPS tracking

`FPSTracker` maintains a rolling average of frame times and breaks down per-segment timings. Call `start_segment` each time you enter a new block of work, then `print_fps()` once per frame.

```python
import time
import lunar_tools as lt

tracker = lt.FPSTracker(update_interval=0.25)

while True:
    tracker.start_segment("simulate")
    time.sleep(0.01)

    tracker.start_segment("render")
    time.sleep(0.005)

    tracker.print_fps()
```

Output example:

```
FPS: 58.9 | simulate: 9.9ms | render: 5.0ms
```

## Health reporting (Telegram)

`HealthReporter` periodically checks that your process is still calling `report_alive()`. If the heartbeat stalls it sends a Telegram alert. Configure the bot token and chat ID via environment variables (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`).

```python
import time
import lunar_tools as lt

reporter = lt.HealthReporter("Immersive Room", alive_status_enable_after=120)

try:
    while True:
        reporter.report_alive()
        time.sleep(5)
except Exception as exc:
    reporter.report_exception(exc)
```

Send on-demand updates with `reporter.report_message("Projector restarted")`. When you intentionally shut the system down call `reporter.report_message("Shutting down")` so operators know it was deliberate.
