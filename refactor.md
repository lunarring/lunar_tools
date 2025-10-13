# Lunar Tools Modernisation Blueprint

## 1. Objectives

- **Clarify structure** so engineers can identify where a capability lives and how to extend it.
- **Isolate third-party SDKs** behind thin adapters to keep optional dependencies optional.
- **Improve testability** by separating orchestration logic from device/network integrations.
- **Lay groundwork** for future CLI/tooling built on a consistent service layer.

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

### Phase A – Platform Foundation

1. **Create packages**
   - Add `platform/`, `services/`, `adapters/`, `presentation/` with `__init__.py`.
   - Move `create_logger` & `dynamic_print` to `platform/logging.py`; update imports.
   - Extract configuration helpers (`read_api_key`, env loaders) into `platform/config.py`.
   - Tag optional dependency imports with clear error messaging (e.g., `raise RuntimeError("Install lunar-tools[audio]")`).

2. **Define service contracts**
   - Introduce lightweight protocols/ABCs in `services/audio/contracts.py`, `services/comms/contracts.py`, etc.
   - Document expected methods (e.g., `generate(text: str) -> str`, `send(payload: bytes) -> None`).

3. **Bootstrap tests**
   - Provide fake adapter implementations under `tests/fakes/` for use in unit tests.
   - Configure pytest markers for optional features (`@pytest.mark.audio`).

### Phase B – Audio Stack Pilot

1. **Adapters**
   - Split `audio.py` responsibilities:
     - `adapters/audio/sounddevice_recorder.py`
     - `adapters/audio/openai_tts.py`
     - `adapters/audio/elevenlabs_tts.py`
     - `adapters/audio/deepgram_transcriber.py`
   - Each adapter implements contracts defined in Phase A and handles missing SDKs gracefully.

2. **Services**
   - `services/audio/recorder_service.py` orchestrates recording lifecycle.
   - `services/audio/tts_service.py` handles TTS; configurable to pick desired adapter.
   - `services/audio/transcription_service.py` for streaming/offline transcripts.

3. **Presentation wiring**
   - Update `realtime_voice.py` (temporarily move to `presentation/`) to build services via a factory (e.g., `bootstrap_audio_stack(config)`).
   - Replace direct `LogPrint` usage with injected loggers (obtained from `platform.logging`).

4. **Testing & docs**
   - Add unit tests for services using fake adapters.
   - Document new audio APIs and migration steps for downstream users.

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

## 5. Immediate Tasks (Iteration 0)

1. Introduce `platform/`, `services/`, `adapters/`, `presentation/` folders; move `logging` and config helpers under `platform/`.
2. Draft interface contracts for audio services and outline expected adapter methods.
3. Prepare issue tracker / task board reflecting the roadmap phases.
4. Align team on dependency strategy (extras vs. optional imports) before deep refactors.

Once the foundation is merged, the Audio Pilot (Phase B) becomes the proving ground for the new architecture. Subsequent phases can borrow patterns from that pilot to bring the rest of the toolkit up to the same standard.
