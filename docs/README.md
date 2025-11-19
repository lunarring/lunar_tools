# Lunar Tools Documentation

This folder breaks the monolithic README into smaller guides. Each page focuses on a toolkit area so you can jump straight to the features you care about.

## Layout
- [`inputs.md`](inputs.md) – capture devices, MIDI, keyboard helpers, and controller abstractions.
- [`audio_and_voice.md`](audio_and_voice.md) – speech recording, transcription, text-to-speech, and realtime voice conversations.
- [`vision_and_display.md`](vision_and_display.md) – rendering windows, GPU pipelines, movie compositing, and generative imagery.
- [`communication.md`](communication.md) – OSC, ZeroMQ, and remote streaming patterns.
- [`logging_and_monitoring.md`](logging_and_monitoring.md) – FPS tracking, logging, and health reporting.
- [`development.md`](development.md) – environment setup, tests, optional dependencies, and API keys.
- [`examples.md`](examples.md) – narrative index for the runnable scripts in [`examples/`](../examples).
- [`configuration.md`](configuration.md) – config file schemas and CLI entry points for presentation stacks.
- [`migration.md`](migration.md) – Phase E deprecations plus before/after import guidance.

## Suggested reading order
1. Start with the main [README](../README.md) for installation instructions.
2. Skim [`inputs.md`](inputs.md) and [`audio_and_voice.md`](audio_and_voice.md) if you are wiring sensors or microphones.
3. Jump to [`vision_and_display.md`](vision_and_display.md) to render results or package them as movies.
4. Use [`communication.md`](communication.md) and [`logging_and_monitoring.md`](logging_and_monitoring.md) to connect systems and observe behaviour.

## Copy-friendly examples
Every guide contains runnable snippets. They are intentionally minimal and highlight required extras. Look at [`examples.md`](examples.md) when you want end-to-end demos or reference implementations.
