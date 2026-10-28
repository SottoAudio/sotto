# Sotto Voce: notes for contributors and coding agents

- Python 3.10+ for `failsafe/` and `prototype/`; numpy, scipy, onnxruntime, sounddevice. No other
  dependencies without a good reason.
- Tests use the standard library `unittest`: `python -m unittest discover -s tests -v`.
  Speech-model tests are skipped until `python prototype/tools/get_models.py` has been run.
- Never commit model files or recordings (see .gitignore). Models are downloaded and converted
  by `prototype/tools/get_models.py` (checksum-pinned).
- The fail-safe rules are fixed requirements, not preferences:
  - starts disarmed;
  - HEARTBEAT only while armed and every check is green;
  - any fault latches;
  - re-arming is manual only, and needs 5 s of green;
  - sidetone never passes through the app.
- Audio-thread code must not block, allocate large buffers, or touch the network. RRCS and HTTP
  run on their own threads.
- The soundcard's real-time callback is native C (`engine/audio-io`): it only copies audio through
  lock-free rings, plays the talkers dry and latches HEARTBEAT off if the engine is late. Python
  never runs on it; the engine runs on its own thread a fixed safety margin behind.
- Write user-facing text plainly, for comms engineers.
