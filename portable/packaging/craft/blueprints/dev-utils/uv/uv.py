# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
# SPDX-License-Identifier: BSD-2-Clause

"""Craft blueprint for uv, the package manager the plugin environments use.

The flatpak manifest installs uv as a module of its own; nothing equivalent
existed for the Craft builds, so the packaged application fell back to pip
without saying anything — AbstractPythonInterface::uvExec() returns an empty
string when the binary is absent, by design.

What uv buys is not speed but disk: it keeps one cache of unpacked wheels and
hard-links them into each environment, so a second plugin wanting the same
torch costs nothing. With pip every environment carries its own copy.

Shipped as the upstream prebuilt binary rather than built here: it is written
in Rust, and compiling it would drag a whole toolchain into the image for one
executable. Same reasoning, and the same version, as the flatpak manifest.

It installs into bin/ because that is where the application looks for it:
uvExec() checks QCoreApplication::applicationDirPath() + "/uv" first, and in
every package the editor and this binary end up in the same directory.
"""

import info
from CraftCompiler import CraftCompiler
from CraftCore import CraftCore
from Package.BinaryPackageBase import BinaryPackageBase
from Utils import CraftHash

# Digests are upstream's own, from the .sha256 file published beside each
# archive in the release. Keep them in step with the version.
DIGESTS = {
    "0.12.3": {
        "x86_64-unknown-linux-gnu": "600cf9a742aca00d292673b16b5acffaa7b8c269a364ad0c2e79498dcb1fe101",
        "aarch64-apple-darwin": "546f7f8a6c70ff13a3a9d2bc958db3427298cebf3e0cb756f9177133b7068843",
        "x86_64-pc-windows-msvc": "b23350c79e8ad0192b8124af13a0f17e8d4e4549524785e1aef389ae5a06990e",
    },
}


class subinfo(info.infoclass):
    def setTargets(self):
        for ver, digests in DIGESTS.items():
            if CraftCore.compiler.isWindows:
                triple, ext = "x86_64-pc-windows-msvc", "zip"
            elif CraftCore.compiler.isMacOS:
                arm = CraftCore.compiler.architecture == CraftCompiler.Architecture.arm64
                triple, ext = ("aarch64-apple-darwin" if arm else "x86_64-apple-darwin"), "tar.gz"
            else:
                triple, ext = "x86_64-unknown-linux-gnu", "tar.gz"

            self.targets[ver] = f"https://github.com/astral-sh/uv/releases/download/{ver}/uv-{triple}.{ext}"
            # The two archive kinds are laid out differently, and upstream does
            # not document it: the tarballs carry a top-level directory named
            # after the triple, the Windows zip puts uv.exe, uvx.exe and uvw.exe
            # straight at the root. Naming a source directory that is not there
            # fails the install with "copyDir called. srcdir: ... does not
            # exists", so only the tarballs get one.
            if ext == "tar.gz":
                self.targetInstSrc[ver] = f"uv-{triple}"
            self.targetInstallPath[ver] = "bin"
            if triple in digests:
                self.targetDigests[ver] = ([digests[triple]], CraftHash.HashAlgorithm.SHA256)

        self.defaultTarget = "0.12.3"
        self.description = "An extremely fast Python package and project manager, written in Rust"
        self.webpage = "https://github.com/astral-sh/uv"


class Package(BinaryPackageBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
