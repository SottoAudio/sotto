# Sotto design notes

These are the decisions so far and the reasons for them. Riedel details come from the RRCS 9.0.1
interface specification and the Artist 9.0.1 release notes.

## Scope

| | |
| --- | --- |
| Phase 1 | 4 talkers: show caller, creative director, technical manager, stage manager (2 on SmartPanels, 2 on Bolero) |
| End goal | Up to 64 channels in and out on one laptop |
| Reference | One mono feed for everyone: the PGM feed that drives the PA |
| Computer | One laptop at the comms engineer's station: MacBook Pro M2 or newer, on the network via an Ethernet adapter; Windows supported too |
| Audio route | Dante Virtual Soundcard (Dante card in the frame) or Merging VAD (AES67) |
| Clock | PTP grandmaster stays the Master Antenna |
| Latency | About 20 ms of processing plus the soundcard buffers; to be reduced later |

## Fail-safe

The rule is that a listener never loses a talker for longer than the switch-over. 250–500 ms is
acceptable; anything shorter risks false trips.

1. **Per talker, inside the app.** Watchdogs catch invalid output, a diverging echo canceller,
   output silent while the talker is speaking, and missed deadlines. That one talker goes to dry
   inside the app within about 20 ms, and stays there until re-armed.
2. **Engine checks.** HEARTBEAT is sent only while every PROBE is returning, deadlines are met
   and the audio device is healthy. Any failure disarms everything, and it stays disarmed.
3. **Restart.** A supervisor (LaunchAgent on Mac, a service on Windows) restarts a crashed
   engine. It always comes back disarmed.
4. **The Riedel has the last word.** A VOX on the HEARTBEAT 4-wire, with a 200 or 400 ms hold,
   drives a virtual key whose function is Logic. That key sets CLEAN_OK. The Design D logic plays
   listeners the clean return only while CLEAN_OK (and ARMED) are on, otherwise the dry mic.
   Anything that silences the laptop (crash, freeze, soundcard, adapter, cable, switch, Dante
   card) therefore lands listeners on dry.

PROBE is looped laptop → frame → laptop, so a break in either direction is noticed. This
matters because panel mics are digital silence between calls, which would otherwise look the
same as a dead input. Use one PROBE per stream.

## Re-arming

Never automatic. Requests come from the console, Companion (HTTP) or a panel key (RRCS
`GetAllLogicSources_v2`, polled). The app accepts a request only after 5 s of green checks.
Arming everyone turns HEARTBEAT on, so CLEAN_OK comes on, and sets ARMED through RRCS
`SetLogicSourceState`. Arming one talker is refused while everyone is disarmed. RRCS is used
only for re-arming and status, never for falling back.

## Channel budget, phase 1

| Into the laptop | Out of the laptop |
| --- | --- |
| 4 talkers, PGM, PROBE return(s) | 4 clean returns, HEARTBEAT, PROBE, engineer's monitor |

At 64 talkers, a 64 × 64 soundcard holds about 54 talkers once references, probes and the
heartbeat are counted. Merging VAD Premium (136 × 136) holds 64.

## Processing cost

Measured in Python on one core of the development machine:
- **Speech model, one talker at a time:** 2.0 ms per 10 ms frame.
- **Speech model, batched:** 0.35 ms per talker for a batch of 16, and 25 ms for a batch of 64.
- **Echo canceller and suppressor:** about 1.2 ms per talker with 2 references.

Four talkers fit easily on an M2. 64 talkers needs parallel batches across several cores (Max
chip), or the speech model running on the Neural Engine.

## Toolkit and licence

MIT licence. The engine will be C++ on permissive libraries: RtAudio or miniaudio for Core Audio
and ASIO, ONNX Runtime for the speech model, and Accelerate on Mac. The console is served by the
engine to a browser, so it also works from an iPad or phone, and can be wrapped in Tauri or
Electron if a standalone window is wanted. GitHub Actions builds signed Mac and Windows
installers.

## Open questions for Riedel

- Do VOX and VOX-type virtual keys with a Logic function work on Dante-card and AES67 4-wires
  on an Artist-1024? Is the hold time purely a release delay, and how quickly does logic act
  after it?
- Can Director logic release a latching virtual key (ARMED) when CLEAN_OK drops?
- In Artist mode, is a Bolero's mic sent to the matrix only while a talk key is held, and is its
  sidetone a matrix crosspoint?
- When an incoming stream stops, does a 4-wire output go silent straight away?
- Can a logic action light a key on the engineer's panel to show ARMED?
