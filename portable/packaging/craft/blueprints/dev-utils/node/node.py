# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
# SPDX-License-Identifier: BSD-2-Clause

"""Craft blueprint for Node.js, the runtime of the chat's agents.

The agents the chat talks to over the Agent Client Protocol — Claude, Codex,
Gemini and the rest of the ACP registry — are mostly npm packages, and
claude-agent-acp needs Node 22 or later. The agents themselves are downloaded
the first time the user picks one; only the runtime ships with the editor. The
flatpak manifest installs the same version as a module of its own.

Shipped as the upstream prebuilt binary, like uv: building Node here would pull
a whole toolchain into the image for one executable.

It installs where the application's own data is, because that is where it
looks (QStandardPaths::locate under AppDataLocation, the way it finds its MCP
server): share/wunjo/node in the AppImage and on Windows, share/node on macOS,
which Craft moves into Contents/Resources. The Windows zip
puts node.exe at its root; the tarballs carry bin/node — the application checks
both.
"""

import info
from CraftCompiler import CraftCompiler
from CraftCore import CraftCore
from Package.BinaryPackageBase import BinaryPackageBase
from Utils import CraftHash

# Digests are upstream's own, from SHASUMS256.txt beside each release. Keep them
# in step with the version and with the flatpak manifest.
DIGESTS = {
    "22.23.3": {
        "linux-x64.tar.xz": "df450af89261115ef9f9e3830c3eeb2cc9213b63c720b1af623cb5dcbe2e02de",
        "darwin-arm64.tar.gz": "23b25245dcfb9af7262f8ff142e9e2e0af025368117329e7a7458a51e5922f53",
        "darwin-x64.tar.gz": "8a677b0219178efd6eb0e475457c4afb452b521a92f6e67845a73bd85727f2a8",
        "win-x64.zip": "2b0ff57b049cda1bbcea2240eec20467018713c1efe1f7360c2681859b90ed71",
    },
}


class subinfo(info.infoclass):
    def setTargets(self):
        for ver, digests in DIGESTS.items():
            if CraftCore.compiler.isWindows:
                platform = "win-x64.zip"
            elif CraftCore.compiler.isMacOS:
                arm = CraftCore.compiler.architecture == CraftCompiler.Architecture.arm64
                platform = "darwin-arm64.tar.gz" if arm else "darwin-x64.tar.gz"
            else:
                platform = "linux-x64.tar.xz"
            stem = f"node-v{ver}-{platform.split('.')[0]}"  # node-v22.23.3-linux-x64

            self.targets[ver] = f"https://nodejs.org/dist/v{ver}/node-v{ver}-{platform}"
            # every archive carries one top-level directory named after itself
            self.targetInstSrc[ver] = stem
            # beside the application's other data: share/wunjo everywhere but
            # macOS, where it has no "wunjo" level (DATA_INSTALL_PREFIX is
            # empty there, CMakeLists.txt) and Craft moves share/ into
            # Contents/Resources — so Contents/Resources/node
            self.targetInstallPath[ver] = "share/node" if CraftCore.compiler.isMacOS else "share/wunjo/node"
            self.targetDigests[ver] = ([digests[platform]], CraftHash.HashAlgorithm.SHA256)

        self.defaultTarget = "22.23.3"
        self.description = "JavaScript runtime the chat's ACP agents run on"
        self.webpage = "https://nodejs.org"


class Package(BinaryPackageBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
