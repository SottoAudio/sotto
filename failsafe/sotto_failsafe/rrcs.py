"""Optional link to Riedel RRCS (XML-RPC, TCP 8193 by default), per the RRCS 9.0.1 interface spec.

Used for two things only, never for falling back:
  * reading the ARM_REQUEST logic source (set by a key on the engineer's panel) by polling
    GetAllLogicSources_v2, and
  * setting the ARMED logic source with SetLogicSourceState when the app arms or disarms.
If RRCS is unreachable the app carries on; the console and Companion can still arm it.
"""
import itertools
import threading
import time
import xmlrpc.client


class _TimeoutTransport(xmlrpc.client.Transport):
    """Per-call timeout without touching the process-wide socket default."""

    def __init__(self, timeout):
        super().__init__()
        self.timeout = timeout

    def make_connection(self, host):
        conn = super().make_connection(host)
        conn.timeout = self.timeout
        return conn


class RRCSLink:
    def __init__(self, host, port=8193, armed_name=None, arm_request_name=None,
                 on_arm_request=None, poll_s=0.25, timeout_s=1.0):
        self.url = f"http://{host}:{port}"
        self.armed_name = armed_name
        self.arm_request_name = arm_request_name
        self.on_arm_request = on_arm_request
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self._key = itertools.count(1)
        self.ids = {}                 # logic source name (long or 8-char label) -> object id
        self.states = {}
        self.ok = False
        self.last_error = None
        self._prev_request = None
        self._stop = threading.Event()
        self._th = None

    def _proxy(self):
        return xmlrpc.client.ServerProxy(self.url, allow_none=True,
                                         transport=_TimeoutTransport(self.timeout_s))

    def _tk(self):
        return f"sotto-{next(self._key)}"

    # ------------------------------------------------------------ calls
    def read_logic_sources(self):
        res = self._proxy().GetAllLogicSources_v2(self._tk())
        n = int(res.get("LogicSourceCount", 0))
        ids, states = {}, {}
        for i in range(1, n + 1):
            item = res.get(f"LogicSource#{i}") or res.get(f" LogicSource#{i}")
            if not item:
                continue
            longname, label, oid, state = item[0], item[1], int(item[2]), bool(item[3])
            for nm in (longname, label):
                if nm:
                    ids[nm.strip()] = oid
                    states[nm.strip()] = state
        self.ids, self.states = ids, states
        return states

    def set_armed(self, state):
        if not self.armed_name:
            return None
        oid = self.ids.get(self.armed_name)
        if oid is None:
            self.read_logic_sources()
            oid = self.ids.get(self.armed_name)
        if oid is None:
            raise RuntimeError(f"logic source '{self.armed_name}' not found in RRCS")
        res = self._proxy().SetLogicSourceState(self._tk(), oid, bool(state))
        err = res[1] if isinstance(res, (list, tuple)) and len(res) > 1 else 0
        if err:
            raise RuntimeError(f"SetLogicSourceState error code {err}")
        return True

    def set_armed_safe(self, state):
        try:
            self.set_armed(state)
            return True
        except Exception as e:      # never let RRCS trouble affect audio or the fallback
            self.last_error = repr(e)
            return False

    # ------------------------------------------------------------ polling
    def start(self):
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.wait(self.poll_s):
            try:
                states = self.read_logic_sources()
                self.ok = True
                self.last_error = None
                if self.arm_request_name:
                    cur = states.get(self.arm_request_name)
                    if cur and self._prev_request is False and self.on_arm_request:
                        self.on_arm_request()            # rising edge = the panel key was pressed
                    if cur is not None:
                        self._prev_request = cur
            except Exception as e:
                self.ok = False
                self.last_error = repr(e)
                time.sleep(1.0)

    def status(self):
        return {"configured": True, "ok": self.ok, "url": self.url, "error": self.last_error,
                "armed_source": self.armed_name, "arm_request_source": self.arm_request_name,
                "armed_state": self.states.get(self.armed_name) if self.armed_name else None}
