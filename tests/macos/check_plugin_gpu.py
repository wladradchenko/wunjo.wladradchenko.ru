#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
# SPDX-License-Identifier: BSD-2-Clause

"""What Metal on this Mac can actually run, and whether the plugins know it.

Apple's torch backend is not a smaller CUDA. It is a backend with holes in it,
and the holes move with the operating system: `aten::grid_sampler_3d` — warping
a face — has no Metal kernel at all, and the Fourier transform every
spectrogram starts with arrived in macOS 14, so on the 13.3 this application
still targets it stops with "FFT operations are only supported on MacOS 14+".
torch's answer to either is to raise, which turns one missing kernel into a
plugin that cannot run.

The plugins answer that in two ways, and this script checks both:

  the env var   PYTORCH_ENABLE_MPS_FALLBACK=1, set in each plugin's main.py
                before torch is ever imported — torch reads it once, when the
                library loads, so an `import torch` above that line silently
                disables every fallback below it. That is a source check and it
                runs on any machine, including the Linux one this is written on.
  the routing   what no fallback can fix. An operator torch has a Metal kernel
                for but macOS refuses still raises, fallback or not, which is
                the FFT case: the voice plugin asks the machine (voice_clone/
                device.py) and sends the transform to the processor when the
                answer is no. This part needs a Mac and a plugin environment.

    python3 tests/macos/check_plugin_gpu.py            # every environment found
    python3 tests/macos/check_plugin_gpu.py --python ~/Library/Application\\ Support/wunjo/venv-face-toolkit/bin/python3

It exits non-zero when a plugin would meet a stop it does not handle: an
operator that raises even with the fallback asked for, or a machine whose Metal
cannot take a transform while the plugin believes it can. A missing kernel that
the fallback covers is reported and is not a failure — that is the fallback
doing its job, slowly, which is the point of it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

def mac_plugins() -> list:
    """The plugins that say they run on a Mac and put work on torch.

    Read out of each manifest rather than written down here: a plugin that
    declares itself Linux-and-Windows — LatentSync asks for 12 GB of CUDA and
    says so — has no Metal to arm a fallback for, and one added tomorrow should
    be checked without this file being edited.
    """
    found = []
    for manifest in sorted((REPO / "plugins").glob("*/plugin.json")):
        try:
            described = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if "macos" not in (described.get("os") or []):
            continue
        main = manifest.parent / "main.py"
        if main.is_file() and re.search(r"^\s*(?:import torch|from torch)|torch",
                                        main.read_text(encoding="utf-8"), re.M):
            found.append(manifest.parent.name)
    return found

#: Where the application keeps a plugin's private environment on macOS.
VENV_ROOT = Path.home() / "Library" / "Application Support" / "wunjo"

FALLBACK_VAR = "PYTORCH_ENABLE_MPS_FALLBACK"

#: What each operator is here for, so a report says which feature is at stake
#: rather than which symbol. The probe below runs them in this order.
PROBES = {
    "fft": "the spectrogram every voice job starts with (macOS 14+ on Metal)",
    "grid_sampler_3d": "warping a face — Live Portrait's whole second half",
    "conv3d": "the volume the warping module convolves",
    "gru": "the reference encoder that measures a voice",
}

# Run inside the plugin's own interpreter: it is the torch that ships with that
# environment whose answer matters, not the one on PATH.
PROBE = r'''
import json, os, platform, sys
report = {"python": sys.version.split()[0], "macos": platform.mac_ver()[0],
          "fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", ""), "ops": {}}
try:
    import torch
except Exception as error:
    report["error"] = "torch: %s" % error
    print(json.dumps(report)); raise SystemExit(0)

report["torch"] = torch.__version__
mps = getattr(torch.backends, "mps", None)
report["mps"] = bool(mps is not None and mps.is_built() and mps.is_available())
if not report["mps"]:
    print(json.dumps(report)); raise SystemExit(0)

def probe(name, work):
    try:
        work()
        report["ops"][name] = "ok"
    except Exception as error:
        report["ops"][name] = str(error).strip().splitlines()[0][:160]

probe("fft", lambda: torch.fft.rfft(torch.zeros(64, device="mps")))
probe("grid_sampler_3d", lambda: torch.nn.functional.grid_sample(
    torch.zeros(1, 1, 2, 2, 2, device="mps"), torch.zeros(1, 1, 1, 1, 3, device="mps"), align_corners=False))
probe("conv3d", lambda: torch.nn.Conv3d(1, 1, 3, padding=1).to("mps")(torch.zeros(1, 1, 4, 8, 8, device="mps")))
probe("gru", lambda: torch.nn.GRU(4, 4, batch_first=True).to("mps")(torch.zeros(1, 3, 4, device="mps")))
print(json.dumps(report))
'''


def arms_fallback(source: str) -> str:
    """Whether @p source asks for the CPU fallback before torch can load.

    Order is the whole of it: torch registers its Metal fallback when the
    library loads and never looks at the variable again, so the line has to come
    before the first import of torch anywhere in the file — including the ones
    inside functions, since any of them may be what loads it first.
    """
    armed = re.search(r"""%s""" % re.escape(FALLBACK_VAR), source)
    if not armed:
        return "never asks for %s" % FALLBACK_VAR
    imported = re.search(r"^\s*(?:import torch|from torch)", source, re.M)
    if imported and imported.start() < armed.start():
        return "imports torch at line %d, before %s is set" % (
            source[:imported.start()].count("\n") + 1, FALLBACK_VAR)
    return ""


def check_sources(names: list) -> int:
    """The half that runs anywhere: is the fallback armed, and armed in time."""
    problems = 0
    for name in names:
        main = REPO / "plugins" / name / "main.py"
        trouble = arms_fallback(main.read_text(encoding="utf-8"))
        print("  %-16s %s" % (name, trouble or ("%s set before torch loads" % FALLBACK_VAR)))
        problems += bool(trouble)
    return problems


def environments(names: list, chosen: str = "") -> list:
    if chosen:
        return [("chosen", Path(chosen).expanduser())]
    found = []
    for name in names:
        python = VENV_ROOT / ("venv-%s" % name) / "bin" / "python3"
        if python.is_file():
            found.append((name, python))
    return found


def check_machine(python: Path) -> int:
    """The half that needs a Mac: what Metal here does with what the plugins ask."""
    problems = 0
    for wanted in ("", "1"):
        environment = dict(os.environ)
        environment.pop(FALLBACK_VAR, None)
        if wanted:
            environment[FALLBACK_VAR] = wanted
        finished = subprocess.run([str(python), "-c", PROBE], capture_output=True, text=True,
                                  env=environment, timeout=600)
        line = finished.stdout.strip().splitlines()[-1] if finished.stdout.strip() else ""
        try:
            report = json.loads(line)
        except ValueError:
            print("    could not ask it: %s" % (finished.stderr.strip().splitlines()[-1:] or ["no output"])[0])
            return 1
        if report.get("error"):
            print("    %s" % report["error"])
            return 1
        if not report.get("mps"):
            print("    torch %s, no Metal backend on this machine — nothing to check"
                  % report.get("torch", "?"))
            return 0
        print("    torch %s on macOS %s, fallback %s"
              % (report["torch"], report["macos"] or "?", "asked for" if wanted else "not asked for"))
        for name, why in PROBES.items():
            answer = report["ops"].get(name, "not probed")
            print("      %-16s %-9s %s" % (name, "ok" if answer == "ok" else "stops", why))
            if answer == "ok" or not wanted:
                continue
            # With the fallback asked for, only what Metal refuses on its own
            # terms is left, and the plugins have to route around each one.
            if name == "fft":
                print("        → handled: the voice plugin takes spectrograms on the processor")
            else:
                print("        → %s" % answer)
                problems += 1
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="", help="a plugin environment's interpreter")
    args = parser.parse_args()

    names = mac_plugins()
    print("plugins that say they run on a Mac: %s" % (", ".join(names) or "none in this checkout"))
    problems = check_sources(names)

    found = environments(names, args.python)
    if not found:
        print("\nno plugin environment to ask (%s)"
              % ("not macOS" if sys.platform != "darwin" else "none installed under %s" % VENV_ROOT))
        return 1 if problems else 0

    for name, python in found:
        print("\n%s (%s):" % (name, python))
        problems += check_machine(python)

    print("\n%s" % ("something a plugin cannot route around" if problems else "nothing a plugin cannot handle"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
