#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024 Jean-Baptiste Mardelle <jb@kdenlive.org>
# SPDX-FileCopyrightText: 2022 Julius Künzel <julius.kuenzel@kde.org>
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL

import sys
import os
import subprocess
import importlib.metadata
import importlib.util
from pathlib import Path


def print_help():
    print("""
    THIS SCRIPT IS PART OF WUNJO (www.wunjo.online)

    Usage: python3 checkpackages.py [mode] [packages]

    Where [packages] is a list of python package names separated by blank space

    And [mode] one of the following:

    --help     print this help
    --install  install missing packages
    --upgrade  upgrade the packages
    --details  show details about the packages like eg. version
    --check    show which of the given packages are not yet installed
    """)


if '--help' in sys.argv:
    print_help()
    sys.exit()


def install_command(targets, extra=()):
    """How to install these requirements into the environment we are running in.

    The editor hands us uv through WUNJO_UV when it has one. uv unpacks every
    wheel once into a cache and hard-links it into each environment, so a
    second plugin wanting the same torch costs neither the download nor the
    gigabytes; pip copies, and each environment paid in full. Without the
    variable this is the pip call it always was.
    """
    uv = os.environ.get("WUNJO_UV", "")
    args = []
    for target in targets:
        args += ['-r', target] if target.endswith(".txt") else [target]
    if uv:
        # --python ties the install to this interpreter: uv would otherwise
        # look for an environment of its own making in the working directory.
        return [uv, 'pip', 'install', '--python', sys.executable, *extra, *args]
    return [sys.executable, '-m', 'pip', 'install', *extra, *args, '--no-cache-dir']


def install_order(targets):
    """The requirements in two rounds: the CUDA file first, then everything else.

    It used to be one call per file in whatever order a set happened to yield,
    and that quietly cost a second copy of torch. The CUDA file is the one
    carrying "--index-url .../whl/cu126"; a plain requirements file next to it
    names torch again with no index, so whichever ran first decided which build
    was downloaded — and both did, in turn. Now the pinned index wins because it
    goes first, and what follows resolves against a torch that is already there.

    The second round is asked not to build in isolation: a package built from
    source (SAM-2 is a GitHub archive) declares torch among its build
    requirements, and an isolated build downloads a whole torch of its own —
    from PyPI, so not even the CUDA build we just installed. It falls back to
    the isolated build if that fails, since a package may legitimately need
    something at build time this environment has not got.
    """
    files = [t for t in targets if t.endswith(".txt")]
    names = [t for t in targets if not t.endswith(".txt")]
    cuda = sorted(f for f in files if "cuda" in os.path.basename(f))
    rest = sorted(f for f in files if f not in cuda) + sorted(names)
    rounds = [r for r in (cuda, rest) if r]
    if len(rounds) > 1:
        # A build without isolation uses this environment's build tools, and a
        # venv seeded for python 3.13 has only pip. They get a round of their
        # own: the CUDA file carries "--index-url .../whl/cu126", which applies
        # to the whole call, and setuptools does not exist on that index.
        rounds.insert(0, ['setuptools', 'wheel'])
    return rounds


def run_install(targets, extra=(), env=None):
    """Install one round, retrying without --no-build-isolation if it fails."""
    rounds = install_order(targets)
    for index, round_targets in enumerate(rounds):
        # Only the last round may skip build isolation: by then torch and the
        # build tools are installed, which is exactly what a source package
        # would otherwise download a second copy of.
        last = index == len(rounds) - 1
        isolated_off = ('--no-build-isolation',) if last and index > 0 and os.environ.get("WUNJO_UV") else ()
        try:
            subprocess.check_call(install_command(round_targets, (*extra, *isolated_off)), env=env)
        except Exception:
            if not isolated_off:
                print("failed installing ", round_targets, flush=True)
                continue
            print("build needs its own environment, retrying: ", round_targets, flush=True)
            try:
                subprocess.check_call(install_command(round_targets, extra), env=env)
            except Exception:
                print("failed installing ", round_targets, flush=True)

required = set()
missing = set()

for arg in sys.argv[1:]:
    if not arg.startswith("--"):
        if arg.endswith(".txt"):
            required.add(arg)
        else:
            required.add(arg.lower())

if len(required) == 0:
    print_help()
    sys.exit("Error: You need to provide at least one package name")

installed = {pkg.metadata['Name'] for pkg in importlib.metadata.distributions()}
normalizedInstalled = set()
for i in installed:
    if i is None:
        continue
    normalizedInstalled.add(i.lower())

missing = required - normalizedInstalled

if '--check' in sys.argv:
    for m in missing:
        print("Missing: ", m)
elif '--install' in sys.argv and len(sys.argv) > 1:
    # install missing modules
    if len(missing) > 0:
        print("Installing missing packages: ", missing, flush=True)
        tmpFolder = os.path.join(Path.home(), ".cache/pip-wunjo-tmp-folder")
        print("Using tmp folder: ", tmpFolder, flush=True)
        os.makedirs(tmpFolder, exist_ok=True)
        my_env = os.environ.copy()
        my_env["TMPDIR"] = tmpFolder
        run_install(missing, env=my_env)
elif '--force-install' in sys.argv and len(sys.argv) > 1:
    # install missing modules
    if len(missing) > 0:
        print("Installing missing packages: ", missing, flush=True)
        tmpFolder = os.path.join(Path.home(), ".cache/pip-wunjo-tmp-folder")
        print("Using tmp folder: ", tmpFolder, flush=True)
        os.makedirs(tmpFolder, exist_ok=True)
        my_env = os.environ.copy()
        my_env["TMPDIR"] = tmpFolder
        run_install(missing, ('--force-reinstall',), env=my_env)
elif '--upgrade' in sys.argv:
    # update modules
    # print("Updating packages: ", required)
    upgradable = normalizedInstalled - required
    if upgradable:
        run_install(upgradable, ('--upgrade',))
    run_install(required, ('--upgrade',))
elif '--details' in sys.argv:
    # check modules version
    python = sys.executable
    for m in missing:
        print(m, "==missing", file=sys.stdout,flush=True)
    subprocess.check_call([python, '-m', 'pip', 'freeze'])
else:
    print_help()
    sys.exit("Error: You need to provide a mode")
