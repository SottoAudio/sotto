"""Where the problem is: one status for each link in the chain, and where to start looking.

Sotto Voce sits at the end of a chain:

  Dante network port -> Dante Virtual Soundcard -> this Mac -> audio from the soundcard
  -> Sotto Voce's engine -> processing -> the Riedel loop (PROBE) -> RRCS

Each link gets a state (ok, note, warn, fault, off), a plain sentence saying what is wrong, and
where to look first. The most upstream link that isn't ok is where to start: when the network
port loses its link, PROBE goes missing as well, and the console says "start at the network
port" rather than listing every symptom.

Two parts:
  SystemWatch  a background thread that reads, every second or two, what macOS says about the
               network ports (ifconfig, networksetup) and which programs are using the CPU (ps).
               It never runs on an audio thread and never blocks one.
  FaultFinder  turns that, and Sotto Voce's own counters, into the link states. Pure Python, no
               system calls, so every case can be tested anywhere.

A lost network link, or Dante Virtual Soundcard stopping, is a fault for the fail-safe: Sotto
Voce disarms, and stays disarmed until someone re-arms after 5 s of green.
"""
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque

OK, NOTE, WARN, FAULT, OFF = "ok", "note", "warn", "fault", "off"
LINKS = (
    ("network", "Dante network port"),
    ("dvs", "Dante Virtual Soundcard"),
    ("mac", "This Mac"),
    ("audio", "Audio from the soundcard"),
    ("engine", "Sotto Voce engine"),
    ("processing", "Processing"),
    ("loop", "Riedel loop (PROBE)"),
    ("rrcs", "RRCS"),
)
LABEL = dict(LINKS)
WINDOW_S = 5.0                 # symptoms (skipped cycles, late blocks) count for this long
NOT_WIRED = ("wi-fi", "airport", "bluetooth", "iphone", "ipad", "modem")
OWN_NAMES = ("Sotto Voce", "sotto-voce", "Python")


# ---------------------------------------------------------------- reading what macOS says
def parse_hardware_ports(text):
    """`networksetup -listallhardwareports` -> {device: port name}, e.g. {"en7": "USB 10/100/1G/2.5G LAN"}."""
    out, port = {}, None
    for line in (text or "").splitlines():
        line = line.strip()
        if line.startswith("Hardware Port:"):
            port = line.split(":", 1)[1].strip()
        elif line.startswith("Device:") and port:
            out[line.split(":", 1)[1].strip()] = port
            port = None
    return out


_MEDIA = re.compile(r"\((\d+(?:\.\d+)?)(G?)base", re.I)


def parse_ifconfig(text):
    """`ifconfig -a` -> {iface: {"up", "active", "ipv4": [...], "speed_mbps", "full_duplex", "members"}}.
    "active" is the link (carrier): True, False, or None when macOS reports no status line."""
    out, cur = {}, None
    for line in (text or "").splitlines():
        if line and not line[0].isspace() and ":" in line:
            name, rest = line.split(":", 1)
            flags = re.search(r"<([^>]*)>", rest)
            fl = flags.group(1).split(",") if flags else []
            cur = out[name] = {"up": "UP" in fl, "active": None, "ipv4": [], "speed_mbps": None,
                               "full_duplex": None, "members": []}
            continue
        if cur is None:
            continue
        s = line.strip()
        if s.startswith("inet "):
            cur["ipv4"].append(s.split()[1])
        elif s.startswith("status:"):
            cur["active"] = s.split(":", 1)[1].strip() == "active"
        elif s.startswith("media:"):
            m = _MEDIA.search(s)
            if m:
                cur["speed_mbps"] = int(float(m.group(1)) * (1000 if m.group(2) else 1))
            cur["full_duplex"] = "full-duplex" in s if "duplex" in s else None
        elif s.startswith("member:"):
            cur["members"].append(s.split()[1])
    return out


def _cpu_seconds(s):
    """ps's TIME column: [dd-]hh:mm:ss, mm:ss.ss or hh:mm.ss -> seconds."""
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [float(p) for p in s.split(":")]
    sec = 0.0
    for p in parts:
        sec = sec * 60 + p
    return days * 86400 + sec


def parse_ps(text):
    """`ps -axc -o pid=,time=,comm=` -> {pid: (cpu seconds, name)}."""
    out = {}
    for line in (text or "").splitlines():
        bits = line.split(None, 2)
        if len(bits) < 3 or not bits[0].isdigit():
            continue
        try:
            out[int(bits[0])] = (_cpu_seconds(bits[1]), bits[2].strip())
        except ValueError:
            continue
    return out


def cpu_by_program(before, after, seconds, own_pid=None):
    """CPU % per program name between two ps readings (100 % = one core), busiest first."""
    if not before or seconds <= 0:
        return []
    use = {}
    for pid, (cpu, name) in after.items():
        if pid == own_pid:
            continue
        prev = before.get(pid)
        d = cpu - prev[0] if prev and prev[1] == name else cpu if pid not in before else 0.0
        if d > 0:
            use[name] = use.get(name, 0.0) + 100.0 * d / seconds
    return sorted(use.items(), key=lambda x: -x[1])


def wired_ports(hw, ifs):
    """The wired network ports macOS has: [{"dev", "name", "bridge", ...ifconfig fields}]."""
    out = []
    for dev, name in hw.items():
        if any(w in name.lower() for w in NOT_WIRED) or dev not in ifs:
            continue
        if not (dev.startswith("en") or dev.startswith("bridge")):
            continue
        out.append({"dev": dev, "name": name, "bridge": dev.startswith("bridge") or "bridge" in name.lower(),
                    **ifs[dev]})
    return out


def wifi_on(hw, ifs):
    return any(ifs.get(dev, {}).get("active") for dev, name in hw.items() if "wi-fi" in name.lower() or "airport" in name.lower())


def choose_port(ports, wanted="auto", locked=None):
    """Which port Dante Virtual Soundcard is on. wanted: "auto", "none" (don't check), or a device
    name such as "en7". Returns (port dict or None, note). In auto, the one wired port with a link
    and an address; once found it is kept, so a port that later loses its link is still watched."""
    if wanted == "none":
        return None, "not checked (Settings)"
    by_dev = {p["dev"]: p for p in ports}
    if wanted and wanted != "auto":
        return by_dev.get(wanted) or {"dev": wanted, "name": wanted, "missing": True}, ""
    if locked:
        return by_dev.get(locked) or {"dev": locked, "name": locked, "missing": True}, ""
    live = [p for p in ports if p.get("active") and p.get("ipv4")]
    if not live:
        return None, ""
    plain = [p for p in live if not p["bridge"]] or live
    if len(plain) > 1:
        return plain[0], (f"Several wired ports are up ({', '.join(p['dev'] for p in plain)}); watching "
                          f"{plain[0]['dev']}. Pick the one Dante Virtual Soundcard uses in Settings.")
    return plain[0], ""


def _port_label(p):
    return f"{p['name']} ({p['dev']})" if p.get("name") and p["name"] != p["dev"] else p["dev"]


# ---------------------------------------------------------------- the background reader
class SystemWatch:
    """Reads the network ports every second and CPU use every two, on its own thread."""

    def __init__(self, interval=1.0, run=None):
        self.interval = interval
        self._run = run or self._cmd
        self.lock = threading.Lock()
        self.snap = {"available": sys.platform == "darwin", "t": None, "hw": {}, "ifs": {}, "procs": None,
                     "top": [], "dvsd": None, "dvsd_cpu": None, "others_cpu": None}
        self._stop = threading.Event()
        self._thread = None
        self._ps_prev = None
        self._ps_t = None
        self._hw_t = -1e9
        self._ps_due = 0.0
        self.own_pid = os.getpid()

    @staticmethod
    def _cmd(argv):
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=3)
            return r.stdout if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    def start(self):
        if self._thread is None and self.snap["available"]:
            self._thread = threading.Thread(target=self._loop, name="system-watch", daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()

    def get(self):
        with self.lock:
            return dict(self.snap)

    def _loop(self):
        from . import macos
        try:
            macos.set_thread_qos("utility")            # never competes with the audio threads
        except Exception:
            pass
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def poll(self, now=None):
        now = time.monotonic() if now is None else now
        ift = self._run(["/sbin/ifconfig", "-a"])
        ifs = parse_ifconfig(ift) if ift is not None else None
        hw = None
        if ifs is not None and (now - self._hw_t > 30 or set(ifs) != set(self.snap.get("ifs") or {})):
            hwt = self._run(["/usr/sbin/networksetup", "-listallhardwareports"])
            if hwt is not None:
                hw, self._hw_t = parse_hardware_ports(hwt), now
        upd = {"t": now}
        if ifs is not None:
            upd["ifs"] = ifs
        if hw is not None:
            upd["hw"] = hw
        if now >= self._ps_due:
            self._ps_due = now + 2.0
            pst = self._run(["/bin/ps", "-axc", "-o", "pid=,time=,comm="])
            if pst is not None:
                procs = parse_ps(pst)
                names = {n for _, n in procs.values()}
                upd["dvsd"] = "dvsd" in names
                if self._ps_prev is not None:
                    top = cpu_by_program(self._ps_prev, procs, now - self._ps_t, self.own_pid)
                    upd["dvsd_cpu"] = round(dict(top).get("dvsd", 0.0), 1)
                    others = [(n, c) for n, c in top if not any(o.lower() in n.lower() for o in OWN_NAMES)]
                    upd["top"] = [(n, round(c)) for n, c in others[:5]]
                    upd["others_cpu"] = round(sum(c for _, c in others))
                self._ps_prev, self._ps_t = procs, now
        with self.lock:
            self.snap.update(upd)


# ---------------------------------------------------------------- the link states
class FaultFinder:
    def __init__(self, port_setting="auto"):
        self.port_setting = port_setting or "auto"
        self.locked_port = None
        self.links = {k: {"key": k, "label": v, "state": OFF, "what": "", "look": "", "since": None, "because": None}
                      for k, v in LINKS}
        self.root = None
        self.hist = deque()                      # (t, counters) for the symptom window
        self._bad = {}                           # debounce: consecutive bad readings per check
        self._silent_since = None
        self.history = deque(maxlen=200)         # (wall clock, key, state, what) for the diagnostics file
        self.ports = []
        self._names = {}                         # port names seen, for a port that has since gone

    def set_port(self, wanted):
        self.port_setting = wanted or "auto"
        self.locked_port = None

    # -- helpers
    def _debounced(self, key, bad, n=2):
        self._bad[key] = self._bad.get(key, 0) + 1 if bad else 0
        return self._bad[key] >= n

    def _window(self, t, counters):
        self.hist.append((t, counters))
        while len(self.hist) > 2 and self.hist[1][0] <= t - WINDOW_S:
            self.hist.popleft()
        t0, c0 = self.hist[0]
        return {k: (v - c0.get(k, v)) if v is not None and c0.get(k) is not None else None for k, v in counters.items()}

    # -- the evaluation
    def update(self, t, sysnap, app):
        """sysnap: SystemWatch.get(). app: Sotto Voce's own view (see app._find_faults).
        Returns a list of (kind, text) events for the links that changed."""
        d = self._window(t, {"skipped": app.get("skipped"), "late": app.get("late_cycles"),
                             "blocks": app.get("blocks"), "proc_late": app.get("proc_late")})
        new = {}
        dante = app.get("is_dante", False)
        sim = app.get("simulate", False)
        avail = bool(sysnap.get("available")) and not sim

        # network port
        hw, ifs = sysnap.get("hw") or {}, sysnap.get("ifs") or {}
        self.ports = wired_ports(hw, ifs) if avail else []
        for p in self.ports:
            self._names[p["dev"]] = p["name"]
        if not dante or not avail or not ifs:
            new["network"] = (OFF, "not watched" if dante or sim else "the soundcard isn't Dante", "")
        else:
            port, note = choose_port(self.ports, self.port_setting, self.locked_port)
            if self.port_setting == "none":
                new["network"] = (OFF, note, "")
            elif port is None:
                bad = self._debounced("net", True)
                new["network"] = (FAULT if bad else WARN, "No wired network port has a link.",
                                  "Plug the Dante network into the Mac's Ethernet adaptor and check its link light. "
                                  "Dante Virtual Soundcard only works over wired Ethernet.")
            else:
                if self.port_setting == "auto" and not port.get("missing") and port.get("active"):
                    self.locked_port = port["dev"]
                if port.get("missing") and port["dev"] in self._names:
                    port = {**port, "name": self._names[port["dev"]]}
                lab = _port_label(port)
                gone = port.get("missing")
                down = not gone and port.get("active") is False
                noip = not gone and not down and not port.get("ipv4")
                bad = self._debounced("net", gone or down or noip)
                ch = app.get("channels") or 0
                if gone:
                    new["network"] = (FAULT if bad else WARN, f"{lab} has gone: the adaptor was unplugged or lost power.",
                                      "The USB or Thunderbolt Ethernet adaptor and its connection to the Mac. Replug it, "
                                      "then check Dante Virtual Soundcard is using that port again.")
                elif down:
                    new["network"] = (FAULT if bad else WARN, f"{lab} has no link.",
                                      "The network cable and the switch port it goes to (link light), then the adaptor.")
                elif noip:
                    new["network"] = (FAULT if bad else WARN, f"{lab} has a link but no IP address.",
                                      "DHCP on the Dante network, or give the port a fixed address. Dante Controller shows "
                                      "the address Dante Virtual Soundcard has.")
                elif port.get("speed_mbps") and port["speed_mbps"] < 1000 and ch > 32:
                    new["network"] = (WARN, f"{lab} is running at {port['speed_mbps']} Mb/s.",
                                      "A Gigabit switch port and a good cable: Dante Virtual Soundcard needs Gigabit above "
                                      "32 x 32 channels.")
                elif port.get("bridge"):
                    new["network"] = (NOTE, f"{lab}: Dante Virtual Soundcard on a bridged port.",
                                      "Audinate don't support bridged ports (Thunderbolt Bridge) for Dante Virtual "
                                      "Soundcard. Use a USB or Thunderbolt Ethernet adaptor for shows.")
                elif note:
                    new["network"] = (NOTE, note, "Settings, Dante network port.")
                elif wifi_on(hw, ifs):
                    new["network"] = (NOTE, f"{lab}: link up ({port.get('speed_mbps') or '?'} Mb/s). Wi-Fi is on.",
                                      "Turn Wi-Fi off for shows: Audinate recommend it for Dante Virtual Soundcard.")
                else:
                    sp = port.get("speed_mbps")
                    new["network"] = (OK, f"{lab}: link up{f' at {sp} Mb/s' if sp else ''}, {port['ipv4'][0]}.", "")

        # Dante Virtual Soundcard (or whichever soundcard)
        name = app.get("device_name") or "the soundcard"
        no_dvsd = bool(dante and avail and sysnap.get("dvsd") is False)
        no_dvsd_2 = self._debounced("dvsd", no_dvsd)
        if sim:
            new["dvs"] = (OFF, "simulated frame", "")
        elif app.get("soundcard_down"):
            new["dvs"] = (FAULT, f"{name} stopped delivering audio.",
                          ("Dante Virtual Soundcard: was it stopped, restarted or its settings changed? Sotto Voce "
                           "reconnects by itself when it's back, disarmed.") if dante else
                          "The soundcard's connection and power. Sotto Voce reconnects by itself when it's back, disarmed.")
        elif no_dvsd:
            new["dvs"] = (FAULT if no_dvsd_2 else WARN, "Dante Virtual Soundcard isn't running (no dvsd).",
                          "Open Dante Virtual Soundcard and press Start. Check its licence and its network port.")
        elif dante and (sysnap.get("dvsd_cpu") or 0) >= 70:
            new["dvs"] = (WARN, f"Dante Virtual Soundcard is working very hard (dvsd {sysnap['dvsd_cpu']:.0f} % CPU).",
                          "Check no other app is using the Dante Virtual Soundcard, and its latency setting (10 ms or "
                          "more at 64 channels). Restart it when it's safe to.")
        else:
            new["dvs"] = (OK, f"{name}: running." if dante else f"Soundcard: {name}, running.", "")

        # this Mac
        symptoms = (d.get("skipped") or 0) > 0 or (d.get("late") or 0) > 0
        others, top = sysnap.get("others_cpu"), sysnap.get("top") or []
        busy = others is not None and (others >= 300 or (top and top[0][1] >= 90))
        pw = app.get("power") or {}
        busy_txt = ", ".join(f"{n} {c} %" for n, c in top[:3])
        if pw.get("low_power"):
            new["mac"] = (WARN, "Low Power Mode is on.", "Turn Low Power Mode off (System Settings, Battery).")
        elif pw.get("thermal") and not str(pw["thermal"]).startswith("normal"):
            new["mac"] = (WARN, f"The Mac is {pw['thermal']}: macOS is slowing it down.",
                          "Give it air; take it off soft surfaces and out of the sun.")
        elif symptoms and busy:
            new["mac"] = (WARN, f"The Mac was busy when audio was lost: {busy_txt}.",
                          "Quit or pause those programs for the show (software updates, backups, builds, Photos, "
                          "Spotlight indexing, browser tabs playing video).")
        elif pw.get("power") == "battery":
            new["mac"] = (NOTE, "Running on battery.", "Plug the Mac in for shows: on battery macOS saves power at the "
                                                       "cost of timing.")
        else:
            new["mac"] = (OK, f"Not busy{f' (busiest: {busy_txt})' if busy_txt else ''}." if others is not None
                          else "OK.", "")

        # audio from the soundcard
        sk = d.get("skipped")
        silent = app.get("inputs_silent")
        if silent and not app.get("soundcard_down"):
            self._silent_since = self._silent_since or t
        else:
            self._silent_since = None
        if sim:
            new["audio"] = (OFF, "simulated frame", "")
        elif app.get("soundcard_down"):
            new["audio"] = (FAULT, "No audio: the soundcard has stopped.", "")
        elif sk:
            new["audio"] = (WARN, f"The soundcard skipped {sk} cycle{'s' if sk != 1 else ''} in the last "
                                  f"{WINDOW_S:.0f} s ({sk * app.get('cycle_ms', 10):.0f} ms of audio lost).",
                            "Usually the Mac or Dante Virtual Soundcard being held up: check This Mac, then the "
                            "DVS latency (10 ms or more at 64 channels).")
        elif self._silent_since is not None and t - self._silent_since >= 2.0:
            new["audio"] = (WARN, "Every input is digital silence: no Dante audio is arriving.",
                            "Dante Controller: the subscriptions to Sotto Voce's inputs, and that Dante Virtual "
                            "Soundcard is locked to the clock leader." if dante else
                            "The routing into the soundcard's inputs.")
        else:
            new["audio"] = (OK, "Arriving on time.", "")

        # Sotto Voce's engine
        lt = d.get("late")
        if sim or app.get("late_cycles") is None:
            new["engine"] = (OK if not sim else OFF, "running" if not sim else "simulated frame", "")
        elif app.get("soundcard_down"):
            new["engine"] = (WARN, "Waiting for the soundcard.", "")
        elif lt:
            new["engine"] = (WARN, f"Late: {lt} soundcard cycle{'s' if lt != 1 else ''} played dry in the last "
                                   f"{WINDOW_S:.0f} s. HEARTBEAT held off until re-armed.",
                             "If This Mac shows busy, start there. Otherwise try a bigger engine safety margin "
                             "(Settings) and send a diagnostics file.")
        else:
            new["engine"] = (OK, f"On time ({app.get('margin_ms', '?')} ms margin).", "")

        # processing
        if not app.get("processing_on"):
            new["processing"] = (OFF, app.get("processing_why") or "off", "")
        else:
            blocks, late = d.get("blocks"), d.get("proc_late")
            frac = (late / blocks) if blocks and late is not None else 0.0
            hr = app.get("headroom_ms")
            if frac >= 0.02:
                new["processing"] = (WARN, f"Late on about {100 * min(frac, 1):.0f} % of blocks in the last "
                                           f"{WINDOW_S:.0f} s; those talkers got dry audio.",
                                     ("Settings, Processing headroom: 20 ms. " if hr and hr < 20 else "") +
                                     "Then fewer talkers on Full clean, or check This Mac.")
            elif app.get("processing_ok") is False:
                new["processing"] = (WARN, app.get("processing_why") or "not keeping up", "Settings, Processing.")
            else:
                new["processing"] = (OK, f"Keeping up ({hr} ms headroom)." if hr else "Keeping up.", "")

        # the Riedel loop
        if app.get("starting"):
            new["loop"] = (OFF, "starting up", "")
        elif app.get("probe_ok") is False:
            ms = app.get("probe_missing_ms")
            new["loop"] = (FAULT, "PROBE isn't coming back" + (f" (missing {ms / 1000:.1f} s)." if ms and ms < 98000
                                                              else "."),
                           f"The loop: Sotto Voce out {app.get('probe_out')} -> Dante -> Riedel -> back to Sotto Voce "
                           f"in {app.get('probe_in')}. Check those Dante subscriptions, then the frame's PROBE route.")
        else:
            new["loop"] = (OK, "PROBE coming back.", "")

        # RRCS
        r = app.get("rrcs")
        if not r or not r.get("configured"):
            new["rrcs"] = (OFF, "not set up", "")
        elif not r.get("ok"):
            new["rrcs"] = (WARN, f"No answer from the frame ({r.get('url', '')}): {r.get('error') or 'no reply'}.",
                           "The frame's control network: its IP address, the RRCS licence and port 8193, and the "
                           "cable to the Mac.")
        else:
            new["rrcs"] = (OK, f"Answering ({r.get('url', '')}).", "")

        return self._apply(t, new)

    def _apply(self, t, new):
        events = []
        wall = time.strftime("%H:%M:%S")
        for k, _ in LINKS:
            st, what, look = new[k]
            L = self.links[k]
            changed = st != L["state"]
            L.update(state=st, what=what, look=look)
            if changed:
                L["since"] = wall
                self.history.append((wall, k, st, what))
                if st in (WARN, FAULT):
                    events.append(("link_" + st, f"{L['label']}: {what}" + (f" Look at: {look}" if look else "")))
                elif st == OK and self._was_bad(k):
                    events.append(("link_ok", f"{L['label']}: back to normal. {what}"))
            L["_bad"] = st in (WARN, FAULT)
        root = next((k for k, _ in LINKS if self.links[k]["state"] in (WARN, FAULT)), None)
        for k, _ in LINKS:
            L = self.links[k]
            L["because"] = root if root and k != root and L["state"] in (WARN, FAULT) and \
                [x for x, _ in LINKS].index(k) > [x for x, _ in LINKS].index(root) else None
        if root != self.root and root is not None:
            R = self.links[root]
            events.append(("start_here", f"Start at {R['label']}: {R['what']}" + (f" Look at: {R['look']}" if R["look"] else "")))
        self.root = root
        return events

    def _was_bad(self, k):
        return self.links[k].get("_bad", False)

    def latched(self):
        """A fault the fail-safe acts on (disarm, not green): the network port or DVS itself."""
        for k in ("network", "dvs"):
            L = self.links[k]
            if L["state"] == FAULT:
                return f"{L['label']}: {L['what']}"
        return None

    def state(self):
        links = [{x: v for x, v in self.links[k].items() if not x.startswith("_")} for k, _ in LINKS]
        R = self.links.get(self.root) if self.root else None
        return {"links": links, "root": self.root,
                "headline": (f"Start at {R['label']}: {R['what']}" if R else "Every link is working."),
                "look": R["look"] if R else "",
                "port_setting": self.port_setting,
                "port": self.locked_port,
                "ports": [{"dev": p["dev"], "name": p["name"], "active": p.get("active"), "ipv4": p.get("ipv4"),
                           "speed_mbps": p.get("speed_mbps"), "bridge": p.get("bridge")} for p in self.ports]}

    def report(self):
        """Plain text for the diagnostics file."""
        lines = []
        st = self.state()
        lines.append(st["headline"] + (f"\n  Look at: {st['look']}" if st["look"] else ""))
        lines.append("")
        for L in st["links"]:
            lines.append(f"{L['label']:<26} {L['state'].upper():<6} {L['what']}"
                         + (f" (follows from {LABEL[L['because']]})" if L.get("because") else ""))
        lines.append("")
        lines.append(f"Dante network port setting: {self.port_setting}; watching: {self.locked_port or '-'}")
        for p in st["ports"]:
            lines.append(f"  {p['dev']:<8} {p['name']:<34} link {'up' if p['active'] else 'down' if p['active'] is False else '?'}"
                         f"  {p['speed_mbps'] or '?'} Mb/s  {', '.join(p['ipv4'] or []) or 'no IPv4'}"
                         + ("  (bridge)" if p["bridge"] else ""))
        if self.history:
            lines.append("")
            lines.append("Changes (most recent last):")
            lines += [f"  {w}  {LABEL[k]:<26} {s.upper():<6} {what}" for w, k, s, what in self.history]
        return "\n".join(lines)
