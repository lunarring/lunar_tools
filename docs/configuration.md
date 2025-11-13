# Configuration & CLI Guide

Phase D introduces config-driven entry points so you can wire stacks together
without writing bespoke scripts. Each CLI consumes JSON or YAML files that map
directly to the dataclasses under `lunar_tools.presentation`.

## Audio & Realtime Voice

- CLI: `python -m lunar_tools.presentation.realtime_voice --config examples/configs/realtime_voice.yaml`
- Sections:
  - `audio_stack`: arguments for `AudioStackConfig` (e.g. `enable_playback`, `preferred_tts`).
  - `realtime_voice`: constructor kwargs for `RealTimeVoice` (e.g. `instructions`, `model`, `voice`).
- Command-line overrides: `--model`, `--voice`, `--enable-playback`, `--no-audio-stack`, etc.

## Webcam Display

- CLI: `python -m lunar_tools.presentation.webcam_display --config examples/configs/webcam_display.yaml`
- Sections:
  - `camera`: mirrors `lt.WebCam` constructor (`cam_id`, `shape_hw`, `mirror`, `shift_colors`).
  - `display_stack`: plugs into `DisplayStackConfig` (`backend`, `window_title`, `use_grid`, ...).
  - `loop`: optional runtime tweaks (`print_fps`, `fps_interval`, `loop_sleep`).
- Overrides: `--cam-id`, `--backend`, `--mirror`, `--print-fps`, `--run-seconds`.

## Movie Writer

- Script: `python examples/movie_example.py --config examples/configs/movie_writer.yaml`
- Sections:
  - `movie_stack`: settings for `MovieStackConfig` (`output_path`, `fps`, `threaded`).
  - `demo`: width/height/duration/star count used by the gallery generator.
- Overrides: `--output`, `--fps`, `--seconds`, `--width`, `--height`, `--stars`.

## MIDI & Control Inputs

- Script: `python examples/midi_meta_example.py --config examples/configs/midi_input.yaml`
- Sections:
  - `control_input`: maps to `ControlInputStackConfig` (e.g. `use_meta`, `force_device`).
  - `controls`: dictionary passed to `poll_and_broadcast`; keys are printed in the console.
  - `loop`: Optional `sleep` override.
- Overrides: `--keyboard-only`, `--force-device`, `--run-seconds`, `--sleep`.

## Tips

- All config files can be JSON or YAML. YAML parsing requires `lunar_tools[presentation]` (PyYAML).
- CLI flags always win over configuration values.
- Use `--run-seconds` on long-running demos when scripting CI or automated smoke tests.
