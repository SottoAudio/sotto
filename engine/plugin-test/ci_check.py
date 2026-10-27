"""CI check for sotto-plugin-test runs against the VST3 SDK's ADelay example plugin."""
import json
import sys

a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
assert a["mono_native"] and a["processed_ok"] and a["output_finite"], a
assert a["state_restores"] and a["state_params_match"] and a["restored_copies_diff_db"] < -120, a
delay = {c["name"]: c["value"] for c in b["controls"]}["Delay"]
assert delay.startswith("0.25"), f"restored Delay is {delay}, expected 0.25"
print("ok:", a["plugin"], "latency", a["latency_samples"], "| restored Delay =", delay,
      "| 16 instances:", round(b["cpu_ms_per_instance_per_block"], 4), "ms each per block")
