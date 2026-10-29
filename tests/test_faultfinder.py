"""The fault finder: reading what macOS says, and naming the first link that failed."""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "failsafe"))

from sotto_failsafe import faultfinder as F  # noqa: E402

HW = """
Hardware Port: Thunderbolt Bridge
Device: bridge0
Ethernet Address: N/A

Hardware Port: Wi-Fi
Device: en0
Ethernet Address: aa:bb:cc:dd:ee:00

Hardware Port: USB 10/100/1G/2.5G LAN
Device: en7
Ethernet Address: aa:bb:cc:dd:ee:07

Hardware Port: Thunderbolt 1
Device: en1
Ethernet Address: aa:bb:cc:dd:ee:01

VLAN Configurations
===================
"""


def ifconfig(en7_status="active", en7_ip="169.254.10.20", media="autoselect (1000baseT <full-duplex>)",
             wifi="inactive", with_en7=True):
    en7 = (f"""en7: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\toptions=6464<VLAN_MTU,TSO4,TSO6,CHANNEL_IO,PARTIAL_CSUM,ZEROINVERT_CSUM>
\tether aa:bb:cc:dd:ee:07
""" + (f"\tinet {en7_ip} netmask 0xffff0000 broadcast 169.254.255.255\n" if en7_ip else "") + f"""\tmedia: {media}
\tstatus: {en7_status}
""") if with_en7 else ""
    return f"""lo0: flags=8049<UP,LOOPBACK,RUNNING,MULTICAST> mtu 16384
\tinet 127.0.0.1 netmask 0xff000000
en0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tether aa:bb:cc:dd:ee:00
\tmedia: autoselect
\tstatus: {wifi}
en1: flags=8963<UP,BROADCAST,SMART,RUNNING,PROMISC,SIMPLEX,MULTICAST> mtu 1500
\tmedia: autoselect <full-duplex>
\tstatus: inactive
bridge0: flags=8863<UP,BROADCAST,SMART,RUNNING,SIMPLEX,MULTICAST> mtu 1500
\tmember: en1 flags=3<LEARNING,DISCOVER>
\tmedia: <unknown type>
\tstatus: inactive
{en7}"""


def snap(**kw):
    return {"available": True, "hw": F.parse_hardware_ports(HW), "ifs": F.parse_ifconfig(ifconfig(**kw)),
            "dvsd": True, "dvsd_cpu": 8.0, "others_cpu": 60, "top": [("WindowServer", 20)]}


def app(**kw):
    a = {"device_name": "Dante Virtual Soundcard", "is_dante": True, "simulate": False, "soundcard_down": False,
         "skipped": 0, "late_cycles": 0, "blocks": 0, "proc_late": 0, "inputs_silent": False, "probe_ok": True,
         "probe_missing_ms": 0, "probe_in": "61", "probe_out": 62, "starting": False, "rrcs": None,
         "power": {"power": "mains"}, "channels": 64, "cycle_ms": 10.0, "margin_ms": 20.0, "processing_on": True,
         "processing_ok": True, "processing_why": "", "headroom_ms": 20}
    a.update(kw)
    return a


def run(ff, steps):
    """steps: [(sysnap, app)] one second apart. Returns all events."""
    ev = []
    for i, (s, a) in enumerate(steps):
        ev += ff.update(float(i), s, a)
    return ev


class ParseTests(unittest.TestCase):
    def test_ports(self):
        hw = F.parse_hardware_ports(HW)
        self.assertEqual(hw["en7"], "USB 10/100/1G/2.5G LAN")
        self.assertEqual(hw["bridge0"], "Thunderbolt Bridge")

    def test_ifconfig(self):
        ifs = F.parse_ifconfig(ifconfig())
        self.assertTrue(ifs["en7"]["active"])
        self.assertEqual(ifs["en7"]["ipv4"], ["169.254.10.20"])
        self.assertEqual(ifs["en7"]["speed_mbps"], 1000)
        self.assertFalse(ifs["en0"]["active"])
        self.assertEqual(ifs["bridge0"]["members"], ["en1"])
        self.assertIsNone(ifs["lo0"]["active"])
        self.assertEqual(F.parse_ifconfig(ifconfig(media="autoselect (2500Base-T <full-duplex>)"))["en7"]["speed_mbps"], 2500)
        self.assertEqual(F.parse_ifconfig(ifconfig(media="autoselect (10GbaseT <full-duplex>)"))["en7"]["speed_mbps"], 10000)

    def test_wired_ports_leave_out_wifi(self):
        s = snap()
        devs = [p["dev"] for p in F.wired_ports(s["hw"], s["ifs"])]
        self.assertIn("en7", devs)
        self.assertIn("bridge0", devs)
        self.assertNotIn("en0", devs)

    def test_ps_and_cpu(self):
        a = F.parse_ps("  1 1:00.00 launchd\n 50 0:10.00 dvsd\n 60 0:01.00 clang\n 99 0:05.00 Sotto Voce\n")
        b = F.parse_ps("  1 1:00.10 launchd\n 50 0:10.40 dvsd\n 60 0:04.00 clang\n 61 0:01.50 clang\n 99 0:09.00 Sotto Voce\n")
        self.assertEqual(a[50], (10.0, "dvsd"))
        top = dict(F.cpu_by_program(a, b, 2.0, own_pid=99))
        self.assertAlmostEqual(top["clang"], 225.0)                 # 3 s + a new 1.5 s process over 2 s
        self.assertAlmostEqual(top["dvsd"], 20.0)
        self.assertNotIn("Sotto Voce", top)
        self.assertEqual(F._cpu_seconds("1-02:00:00"), 86400 + 7200)


class ChainTests(unittest.TestCase):
    def test_all_green(self):
        ff = F.FaultFinder()
        run(ff, [(snap(), app())])
        st = ff.state()
        self.assertIsNone(st["root"])
        self.assertEqual(st["headline"], "Every link is working.")
        net = st["links"][0]
        self.assertEqual(net["state"], F.OK)
        self.assertIn("en7", net["what"])
        self.assertIsNone(ff.latched())

    def test_cable_pulled_is_the_place_to_start(self):
        ff = F.FaultFinder()
        down = snap(en7_status="inactive", en7_ip=None)
        ev = run(ff, [(snap(), app()), (down, app()), (down, app(probe_ok=False, probe_missing_ms=2000,
                                                                   inputs_silent=True))])
        st = ff.state()
        self.assertEqual(st["root"], "network")
        self.assertIn("has no link", st["headline"])
        self.assertIn("cable", st["look"])
        loop = next(L for L in st["links"] if L["key"] == "loop")
        self.assertEqual(loop["state"], F.FAULT)
        self.assertEqual(loop["because"], "network")                # a symptom, not where to start
        self.assertIn("Dante network port", ff.latched())          # the fail-safe acts on it
        self.assertTrue(any(k == "start_here" and "network port" in t for k, t in ev))

    def test_one_bad_reading_does_not_latch(self):
        ff = F.FaultFinder()
        run(ff, [(snap(), app()), (snap(en7_status="inactive"), app())])
        self.assertIsNone(ff.latched())
        self.assertEqual(ff.state()["links"][0]["state"], F.WARN)

    def test_adaptor_unplugged(self):
        ff = F.FaultFinder()
        gone = snap(with_en7=False)
        run(ff, [(snap(), app()), (gone, app()), (gone, app())])
        self.assertIn("has gone", ff.state()["headline"])
        self.assertIn("USB 10/100/1G/2.5G LAN (en7)", ff.state()["headline"])
        self.assertTrue(ff.latched())

    def test_back_to_normal(self):
        ff = F.FaultFinder()
        down = snap(en7_status="inactive")
        ev = run(ff, [(snap(), app()), (down, app()), (down, app()), (snap(), app())])
        self.assertIsNone(ff.latched())
        self.assertTrue(any(k == "link_ok" for k, _ in ev))

    def test_no_ip_address(self):
        ff = F.FaultFinder(port_setting="en7")
        run(ff, [(snap(en7_ip=None), app())] * 2)
        self.assertIn("no IP address", ff.state()["headline"])

    def test_slow_link(self):
        ff = F.FaultFinder()
        run(ff, [(snap(media="autoselect (100baseTX <full-duplex>)"), app())])
        self.assertIn("100 Mb/s", ff.state()["headline"])
        self.assertIsNone(ff.latched())

    def test_wifi_and_bridge_are_notes(self):
        ff = F.FaultFinder()
        run(ff, [(snap(wifi="active"), app())])
        self.assertEqual(ff.state()["links"][0]["state"], F.NOTE)
        self.assertIsNone(ff.state()["root"])

    def test_dvs_stopped(self):
        ff = F.FaultFinder()
        s = snap()
        s["dvsd"] = False
        run(ff, [(s, app()), (s, app())])
        self.assertEqual(ff.state()["root"], "dvs")
        self.assertIn("isn't running", ff.state()["headline"])
        self.assertTrue(ff.latched())

    def test_soundcard_stopped(self):
        ff = F.FaultFinder()
        run(ff, [(snap(), app(soundcard_down=True))])
        self.assertEqual(ff.state()["root"], "dvs")
        self.assertTrue(ff.latched())

    def test_busy_mac_explains_skips_and_late_engine(self):
        ff = F.FaultFinder()
        busy = snap()
        busy.update(others_cpu=900, top=[("clang", 600), ("ld", 200), ("Runner.Worker", 100)])
        run(ff, [(snap(), app()), (busy, app(skipped=30, late_cycles=12))])
        st = ff.state()
        self.assertEqual(st["root"], "mac")
        self.assertIn("clang 600 %", st["headline"])
        eng = next(L for L in st["links"] if L["key"] == "engine")
        self.assertEqual(eng["because"], "mac")
        self.assertIsNone(ff.latched())                              # engine late has its own fail-safe path

    def test_skips_without_load_point_at_the_soundcard(self):
        ff = F.FaultFinder()
        run(ff, [(snap(), app()), (snap(), app(skipped=3))])
        self.assertEqual(ff.state()["root"], "audio")
        self.assertIn("skipped 3 cycles", ff.state()["headline"])

    def test_symptoms_clear_after_the_window(self):
        ff = F.FaultFinder()
        steps = [(snap(), app())] + [(snap(), app(skipped=3))] * 14
        run(ff, steps)
        self.assertIsNone(ff.state()["root"])

    def test_a_standing_fault_is_not_blamed_on_a_later_warning(self):
        ff = F.FaultFinder()
        lost = dict(probe_ok=False, probe_missing_ms=99000)
        run(ff, [(snap(), app(**lost)), (snap(), app(**lost)), (snap(), app(skipped=1, **lost))])
        st = ff.state()
        self.assertEqual(st["root"], "loop")                       # faults come first
        audio = next(L for L in st["links"] if L["key"] == "audio")
        self.assertEqual(audio["state"], F.WARN)
        loop = next(L for L in st["links"] if L["key"] == "loop")
        self.assertIsNone(loop["because"])

    def test_busy_stays_shown_through_the_window(self):
        ff = F.FaultFinder()
        busy = snap()
        busy.update(others_cpu=500, top=[("Google Chrome Helper", 300)])
        quiet = snap()
        quiet.update(others_cpu=50, top=[("WindowServer", 20)])
        run(ff, [(snap(), app()), (busy, app(late_cycles=2)), (quiet, app(late_cycles=2)), (quiet, app(late_cycles=2))])
        self.assertIn("Google Chrome Helper", ff.state()["headline"])

    def test_dvs_helpers_count_as_dvs_not_the_mac(self):
        outputs = {"/sbin/ifconfig": ifconfig(), "/usr/sbin/networksetup": HW,
                   "/bin/ps": " 50 0:10.00 dvsd\n 51 0:01.00 dvs_ape\n 52 0:01.00 conmon_cmm\n"}
        w = F.SystemWatch(run=lambda argv: outputs[argv[0]])
        w.poll(now=0.0)
        outputs["/bin/ps"] = " 50 0:11.80 dvsd\n 51 0:02.00 dvs_ape\n 52 0:01.20 conmon_cmm\n"
        w.poll(now=2.0)
        s = w.get()
        self.assertEqual(s["dvsd_cpu"], 150.0)
        self.assertEqual(s["others_cpu"], 0)

    def test_two_ports_on_one_subnet(self):
        s = snap()
        s["ifs"]["en1"].update(active=True, ipv4=["169.254.10.21"])
        port, note = F.choose_port(F.wired_ports(s["hw"], s["ifs"]))
        self.assertIn("169.254.10.x", note)

    def test_digital_silence(self):
        ff = F.FaultFinder()
        run(ff, [(snap(), app(inputs_silent=True))] * 4)
        self.assertEqual(ff.state()["root"], "audio")
        self.assertIn("digital silence", ff.state()["headline"])
        self.assertIn("subscriptions", ff.state()["look"])

    def test_processing_late_names_the_headroom(self):
        ff = F.FaultFinder()
        run(ff, [(snap(), app(headroom_ms=10)), (snap(), app(headroom_ms=10, blocks=500, proc_late=80))])
        st = ff.state()
        self.assertEqual(st["root"], "processing")
        self.assertIn("16 %", st["headline"])
        self.assertIn("Processing headroom: 20 ms", st["look"])

    def test_probe_missing(self):
        ff = F.FaultFinder()
        run(ff, [(snap(), app(probe_ok=False, probe_missing_ms=99000))])
        st = ff.state()
        self.assertEqual(st["root"], "loop")
        self.assertIn("out 62", st["look"])

    def test_rrcs(self):
        ff = F.FaultFinder()
        run(ff, [(snap(), app(rrcs={"configured": True, "ok": False, "url": "http://10.0.0.10:8193",
                                     "error": "timed out"}))])
        self.assertEqual(ff.state()["root"], "rrcs")

    def test_not_dante_and_simulated(self):
        ff = F.FaultFinder()
        run(ff, [(snap(), app(is_dante=False, device_name="MacBook Pro Speakers"))])
        self.assertEqual(ff.state()["links"][0]["state"], F.OFF)
        ff = F.FaultFinder()
        run(ff, [({"available": False}, app(simulate=True, late_cycles=None, skipped=None))])
        self.assertIsNone(ff.latched())

    def test_port_choice(self):
        s = snap()
        ports = F.wired_ports(s["hw"], s["ifs"])
        self.assertEqual(F.choose_port(ports)[0]["dev"], "en7")
        self.assertIsNone(F.choose_port(ports, "none")[0])
        self.assertTrue(F.choose_port(ports, "en9")[0]["missing"])
        ff = F.FaultFinder(port_setting="none")
        run(ff, [(snap(en7_status="inactive"), app())] * 3)
        self.assertIsNone(ff.latched())

    def test_report(self):
        ff = F.FaultFinder()
        run(ff, [(snap(), app()), (snap(en7_status="inactive"), app()), (snap(en7_status="inactive"), app())])
        txt = ff.report()
        self.assertIn("Start at Dante network port", txt)
        self.assertIn("Changes", txt)


class WatchTests(unittest.TestCase):
    def test_poll_with_fake_commands(self):
        outputs = {"/sbin/ifconfig": ifconfig(), "/usr/sbin/networksetup": HW,
                   "/bin/ps": " 50 0:10.00 dvsd\n 60 0:01.00 mds_stores\n"}
        w = F.SystemWatch(run=lambda argv: outputs[argv[0]])
        w.snap["available"] = True
        w.poll(now=0.0)
        outputs["/bin/ps"] = " 50 0:10.20 dvsd\n 60 0:03.00 mds_stores\n"
        w.poll(now=2.0)
        s = w.get()
        self.assertTrue(s["dvsd"])
        self.assertEqual(s["top"][0], ("mds_stores", 100))
        self.assertEqual(s["dvsd_cpu"], 10.0)
        self.assertIn("en7", s["ifs"])

    def test_failed_commands_leave_unknown(self):
        w = F.SystemWatch(run=lambda argv: None)
        w.poll(now=0.0)
        self.assertEqual(w.get()["ifs"], {})
        self.assertIsNone(w.get()["dvsd"])


if __name__ == "__main__":
    unittest.main()
