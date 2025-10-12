import logging

import pytest

from lunar_tools.logprint import create_logger


def _flush_handlers(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        flush = getattr(handler, "flush", None)
        if callable(flush):
            flush()


def test_create_logger_writes_to_file(tmp_path):
    log_path = tmp_path / "logs" / "custom.log"
    logger = create_logger("tests.loggers.file", console=False, file_path=str(log_path))
    logger.info("Test log with specific filename.")
    _flush_handlers(logger)

    assert log_path.exists()
    assert "Test log with specific filename." in log_path.read_text()


def test_create_logger_without_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger = create_logger("tests.loggers.console-only")
    logger.info("Default log message.")

    assert not (tmp_path / "logs").exists()


def test_console_respects_level(capsys):
    logger = create_logger(
        "tests.loggers.console-level",
        level=logging.INFO,
        console_level=logging.INFO,
        console_color=False,
        console=True,
        file_path=None,
    )

    logger.debug("debug message")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""

    logger.info("info message")
    captured = capsys.readouterr()
    assert "info message" in captured.err


def test_nested_directories_are_created(tmp_path):
    nested_path = tmp_path / "nested" / "logs" / "deep.log"

    logger = create_logger("tests.loggers.nested", console=False, file_path=str(nested_path))
    logger.warning("Deep log message")
    _flush_handlers(logger)

    assert nested_path.exists()
    assert "Deep log message" in nested_path.read_text()
