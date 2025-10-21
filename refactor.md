# Lunar Tools Modernisation Blueprint

## 1. Objectives

- **Clarify structure** so engineers can identify where a capability lives and how to extend it.
- **Isolate third-party SDKs** behind thin adapters to keep optional dependencies optional.
- **Improve testability** by separating orchestration logic from device/network integrations.
- **Lay groundwork** for future CLI/tooling built on a consistent service layer.

## Status Snapshot

- Phase A – Platform Foundation: completed; package layout, logging/config consolidation, and service contracts are in place.
- Phase B – Audio Stack Pilot: initiated; adapters, services, and presentation wiring are the new focus.

---

## 2. Architectural Model

Adopt a service-based layering inspired by well-run platform teams:

| Layer | Responsibility | Example contents | Dependencies |
| ----- | -------------- | ---------------- | ------------ |
| `platform/` | Pure utilities, configuration, logging, type definitions | `logging.py`, `config.py`, `exceptions.py`, data containers | Standard library only |
| `services/` | Use-case orchestration expressed in terms of abstract ports | `audio/recorder_service.py`, `audio/tts_service.py`, `comms/message_router.py`, `vision/image_service.py`, `llm/conversation_service.py` | `platform/` |
| `adapters/` | Concrete implementations talking to SDKs or hardware | `audio/openai_tts_adapter.py`, `audio/elevenlabs_adapter.py`, `audio/sounddevice_recorder.py`, `comms/zmq_adapter.py`, `vision/replicate_adapter.py` | `platform/`, third-party libs |
| `presentation/` | UI, CLI, sample apps wiring services together | `realtime_voice_app.py`, `display_ui/`, `examples/` | `platform/`, `services/`, optional adapters |

**Key practices**

- All service modules depend on abstract interfaces; adapters register/instantiate those interfaces.
- `presentation` chooses adapters via configuration and injects them into services.
- No module should import from both `services` and `adapters` unless it is composition/bootstrap code.
- `platform/__init__.py` exports only approved utilities; package root re-exports curated entry points (e.g., factories).

---

## 3. Module Inventory (Current vs. Target)

| Current module | Current role | Suggested new home |
| -------------- | ------------ | ------------------ |
| `logprint.py` | logging helper | `platform/logging.py` |
| `utils.py` (API key helpers, etc.) | Misc utils | `platform/config.py`, `platform/io.py` |
| `audio.py` | Recorder, TTS, transcription, playback | Split into `services/audio/*.py` and `adapters/audio/*.py` |
| `image_gen.py` | Multiple generator classes | `services/vision/image_service.py` (public API) + `adapters/vision/*` |
| `comms.py` | OSC & ZMQ | `services/comms/message_bus.py` + `adapters/comms/*` |
| `llm.py` | OpenAI/Gemini/DeepSeek wrappers | `services/llm/conversation_service.py` + `adapters/llm/*` |
| `display_window.py`, `realtime_voice.py`, `control_input.py` | UI orchestration | `presentation/` package |
| `torch_utils.py`, `fps_tracker.py`, `health_reporting.py` | Cross-cutting util | Evaluate: either stay in `platform/` or move under a dedicated `support/` namespace |

---

## 4. Migration Roadmap

### Phase A – Platform Foundation (Completed)

Phase A established the foundational layout, centralized logging/config helpers, and introduced the initial service contracts plus testing scaffolding.

- [x] Create packages `platform/`, `services/`, `adapters/`, `presentation/`; relocate logging/config utilities; add optional dependency guards.
- [x] Define service contracts via protocols/ABCs under `services/*/contracts.py`.
- [x] Bootstrap tests with fake adapters and pytest markers to prepare for feature slices.

### Phase B – Audio Stack Pilot (In Progress)

1. **Adapters**
   - Split `audio.py` responsibilities:
     - `adapters/audio/sounddevice_recorder.py`
     - `adapters/audio/openai_tts.py`
     - `adapters/audio/elevenlabs_tts.py`
     - `adapters/audio/deepgram_transcriber.py`
   - Each adapter implements contracts defined in Phase A and handles missing SDKs gracefully.
   - [x] Optional dependency guards now funnel through `_optional.require_extra` for clearer guidance.

2. **Services**
   - `services/audio/recorder_service.py` orchestrates recording lifecycle.
   - `services/audio/tts_service.py` handles TTS; configurable to pick desired adapter.
   - `services/audio/transcription_service.py` for streaming/offline transcripts.
   - [x] Added `SpeechToTextService` and extended TTS service to handle playback adapters.

3. **Presentation wiring**
   - Update `realtime_voice.py` (temporarily move to `presentation/`) to build services via a factory (e.g., `bootstrap_audio_stack(config)`).
   - Replace direct `LogPrint` usage with injected loggers (obtained from `platform.logging`).
   - [x] Introduced `presentation/audio_stack.py` with `AudioStackConfig`, `bootstrap_audio_stack`, and an `AudioConversationController` orchestrating services.
   - [x] Migrated the realtime voice client to `presentation/realtime_voice.py` with lazy adapter bootstrapping plus a shim module for backwards compatibility.

4. **Testing & docs**
   - Add unit tests for services using fake adapters.
   - Document new audio APIs and migration steps for downstream users.
   - [x] Added service-layer tests (`tests/test_audio_services.py`, `tests/test_audio_module.py`) leveraging the fake adapters.
   - [x] Introduced lightweight stubs/pytest markers so CI can exercise services without audio hardware or third-party SDKs.
   - [ ] Publish user-facing guide for the audio conversation controller and bootstrap helper.

### Phase C – Communications & Vision

1. **Communications**
   - Carve out `services/comms/message_bus.py` exposing send/receive queues.
   - Create `adapters/comms/zmq_adapter.py` (wraps ZeroMQ) and `adapters/comms/osc_adapter.py`.
   - Adapt ZMQ tests to mock adapter; ensure timeout/backpressure covered.

2. **Vision / Image generation**
   - `services/vision/image_service.py` defines an interface for generation/editing.
   - Provider adapters: `adapters/vision/openai_dalle.py`, `replicate_sdxl.py`, `fal_flux.py`.
   - Provide strategy selection (based on config or argument).

3. **LLM**
   - Mirror structure for OpenAI/Gemini/DeepSeek.
   - Offer service for text completion/conversation with pluggable adapters.

4. **Documentation & tests**
   - Update README/examples to use new service factories.
   - Expand pytest coverage for messaging and image generation flows.

### Phase D – Presentation & Tooling

1. **Presentation layer cleanup**
   - Move `display_window.py`, `control_input.py`, `realtime_voice.py`, `movie.py` orchestrations under `presentation/`.
   - Replace module-level instantiation with dependency injection from a `bootstrap` module.

2. **CLI / Example consolidation**
   - Provide sample scripts invoking services via config (YAML/CLI flags).
   - Document entry points (e.g., `python -m lunar_tools.presentation.realtime_voice --config config.yml`).

3. **Packaging extras**
   - Update `pyproject.toml` or `setup.cfg` with extras: `audio`, `vision`, `llm`, `full`.
   - Add installation guidance for each extra in README.

### Phase E – Hardening & Deprecation

1. **Compatibility layer**
   - Maintain shims in `lunar_tools/__init__.py` mapping old names to new constructors with `DeprecationWarning`.
   - Provide migration guide (before/after snippets).

2. **Static analysis**
   - Enforce `ruff`, `mypy` (optional), and coverage thresholds in CI.
   - Introduce `tox` environments for extras to ensure optional dependency imports remain healthy.

3. **Cleanup**
   - After a release cycle, remove deprecated symbols and obsolete files.
   - Prune leftover generated assets (`output_speech.mp3`, etc.) from repository.

---

## 5. Immediate Tasks (Phase B Kickoff)

1. Polish documentation for the audio bootstrap/controller APIs and publish a migration note for downstream users.
2. Extend adapter coverage (Deepgram/OpenAI transcription) to align with the new service contracts and ensure graceful SDK handling.
3. Update the realtime voice sample in `examples/` to demonstrate the new bootstrap + controller flow end-to-end.
4. Enable optional extras (e.g., `lunar-tools[audio]`) in packaging metadata so users can install the full stack in one step.

Latest iteration:
- Hardened adapters with lazy optional imports and playback integration hooks.
- Added `SpeechToTextService`, playback-aware `TextToSpeechService`, and presentation helpers (`AudioStackConfig`, `AudioConversationController`, `RealTimeVoice`).
- Reworked audio tests (with stubs for heavy deps) to exercise the new layering without requiring hardware access.

With Phase A complete, Phase B is now the proving ground for the architecture. Patterns from the audio stack will guide the remaining roadmap stages.
