# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
# SPDX-License-Identifier: BSD-2-Clause

"""Craft blueprint for Wunjo Make.

Craft is KDE's build system for Windows and macOS: it knows how to build Qt, the
KDE Frameworks and their dependencies, and pulls most of them from a prebuilt
binary cache instead of compiling them. This file is the recipe for *this*
application on top of that.

The dependency list mirrors `find_package` in `portable/CMakeLists.txt` and
`portable/src/CMakeLists.txt`. When one changes, so must the other, or the build
fails on the runner with a missing package rather than here.

The source directory is passed in by the workflow with
`--options wunjo.srcDir=<checkout>/portable`, so this blueprint never fetches
anything itself.

The repeated name in the path is Craft's rule, not a choice: it finds blueprints
by looking for `<category>/<name>/<name>.py`, so the file has to be named after
its directory. `apps` is the category, the same way KDE's own repository uses
`kde/kdemultimedia/kdenlive/kdenlive.py`.
"""

import os
import shutil
import subprocess
from pathlib import Path

import info
from CraftCore import CraftCore
from Package.CMakePackageBase import CMakePackageBase
from Packager.AppImagePackager import AppImagePackager
from Utils import CodeSign


class subinfo(info.infoclass):
    def setTargets(self):
        self.svnTargets["master"] = ""
        self.defaultTarget = "master"
        self.description = "Nonlinear video editor with Artificial Intelligence tools"
        self.displayName = "Wunjo Make"
        self.webpage = "https://wunjo.online"

    def setDependencies(self):
        self.runtimeDependencies["virtual/base"] = None

        # Build-only
        self.buildDependencies["kde/frameworks/extra-cmake-modules"] = None
        self.buildDependencies["dev-utils/pkgconf"] = None

        # qttools is a build tool here, not a runtime library: ki18n_install(po)
        # needs lrelease and nothing in this application links Qt Help, UiTools
        # or Designer. Every KDE framework we depend on declares it the same way.
        #
        # As a runtime dependency it also drags Assistant.app, Designer.app and
        # Linguist.app into the package, and on macOS that is fatal rather than
        # merely wasteful: the packager rewrites their rpaths, which invalidates
        # the signatures they arrive with, and the ad-hoc `codesign --deep` that
        # follows walks into those nested bundles and fails.
        self.buildDependencies["libs/qt6/qttools"] = None
        if CraftCore.compiler.isLinux:
            # AppImagePackager shells out to linuxdeploy; without it the package
            # step aborts with "Craft requires linuxdeploy to create an AppImage".
            self.buildDependencies["dev-utils/linuxdeploy"] = None

        # Qt 6 — see find_package(Qt6 …) in CMakeLists.txt.
        # qtimageformats is not in that list and is still required: it carries
        # the WebP decoder, and every transition preview under
        # data/transitions/previews plus both splash backgrounds are .webp.
        # Without it they load as nothing at all, with no error anywhere.
        for module in (
            "qtbase",
            "qtdeclarative",
            "qtsvg",
            "qtmultimedia",
            "qtnetworkauth",
            "qt5compat",
            "qtimageformats",
        ):
            self.runtimeDependencies[f"libs/qt6/{module}"] = None

        # KDE Frameworks 6 — see find_package(KF6 …) in CMakeLists.txt.
        # Craft resolves dependencies by exact path, and its KF6 recipes are
        # filed under the framework's tier, so the tier is part of the name.
        for framework in (
            "tier1/ki18n",
            "tier1/karchive",
            "tier1/kcodecs",
            "tier1/kcoreaddons",
            "tier1/kconfig",
            "tier1/kwidgetsaddons",
            "tier1/kguiaddons",
            "tier1/solid",
            "tier1/breeze-icons",
            "tier2/kfilemetadata",
            "tier2/kcrash",
            "tier2/kdoctools",
            "tier3/kbookmarks",
            "tier3/kconfigwidgets",
            "tier3/kio",
            "tier3/knotifyconfig",
            "tier3/knewstuff",
            "tier3/kxmlgui",
            "tier3/knotifications",
            "tier3/ktextwidgets",
            "tier3/kiconthemes",
            "tier3/purpose",
        ):
            self.runtimeDependencies[f"kde/frameworks/{framework}"] = None

        # The Python plugins run against an interpreter found on PATH; the
        # manifest of the built-in ones asks for python3.11 or newer. The
        # flatpak gets one from the KDE SDK, and inside an AppImage there is
        # only what this package pulls in. Craft has libs/python as a *build*
        # dependency of virtual/base, and the packager collects the runtime
        # closure only — so without naming it here the image ships no
        # interpreter at all and every plugin reports "Cannot find a compatible
        # python version". Craft's build is 3.11, which satisfies that manifest.
        self.runtimeDependencies["libs/python"] = None

        # uv builds those plugin environments in pip's place, and the flatpak
        # manifest installs it as a module of its own. Craft has no recipe for
        # it, so this repository carries one — without it the packaged editor
        # silently falls back to pip and every plugin environment keeps its own
        # copy of every wheel.
        self.runtimeDependencies["dev-utils/uv"] = None

        # Everything else the editor links
        self.runtimeDependencies["qt-libs/kddockwidgets"] = None
        self.runtimeDependencies["libs/mlt"] = None
        self.runtimeDependencies["libs/ffmpeg"] = None
        self.runtimeDependencies["libs/opencv/opencv"] = None
        self.runtimeDependencies["libs/opentimelineio"] = None
        # find_package(Imath REQUIRED) in CMakeLists.txt, for the OTIO header workaround
        self.runtimeDependencies["libs/imath"] = None
        self.runtimeDependencies["libs/frei0r-plugins"] = None

        # The widget style the brand stylesheet is written against.
        #
        # Without it macOS runs on the native style — the bundle's
        # PlugIns/styles held libqmacstyle and nothing else — and a native style
        # takes its metrics from AppKit rather than from src/assets/style.qss.
        # Combo boxes put their text in a corner with no padding, the welcome
        # screen drew one icon several times its size, and AppKit's own geometry
        # crashed the application from that same drawing path.
        #
        # Craft does build it for macOS: kde/plasma/breeze drops
        # frameworkintegration and kdecoration there, which are the only pieces
        # that want a Linux desktop. The Breeze colour schemes come with it.
        #
        # macOS only on purpose. Linux resolves Breeze through its own packaging
        # already, and adding it here would change a dependency closure that
        # works.
        if CraftCore.compiler.isMacOS:
            self.runtimeDependencies["kde/plasma/breeze"] = None


class Package(CMakePackageBase):
    # Craft instantiates recipes as `Package(package=<CraftPackageObject>)`, so
    # the constructor has to pass its keyword arguments through.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # RELEASE_BUILD strips the git revision from the version string; tests
        # are not run on the packaging runner, and they pull extra dependencies.
        self.subinfo.options.configure.args += [
            "-DRELEASE_BUILD=ON",
            "-DBUILD_TESTING=OFF",
        ]
        # AppImagePackager looks for <appname>.desktop unless told otherwise, and
        # data/CMakeLists.txt installs the file under its reverse-DNS name.
        # PackagerBase.getMacAppPath globs for "<appname>.app", and on macOS the
        # bundle is built as "Wunjo Make.app" (see src/CMakeLists.txt) because
        # that filename is what Finder shows. Everywhere else appname is the
        # executable's own name and the AppImage packager needs it lowercase.
        self.defines["appname"] = "Wunjo Make" if CraftCore.compiler.isMacOS else "wunjo"
        self.defines["desktopFile"] = "online.wunjo.make"

    def internalCreatePackage(self, defines=None, **kwargs) -> bool:
        """Make the interpreter in the bundle runnable, after it is in there.

        Neither copy of Python works as packaged, and they fail for opposite
        reasons. The one inside the framework records its library as
        ``@executable_path/../Frameworks/Python.framework/.../Python``, which
        only resolves when the running executable sits in ``Contents/MacOS`` —
        from its own ``bin`` directory it points at a Frameworks folder that
        does not exist, and dyld aborts. What does sit in ``Contents/MacOS`` is
        a Craft shim of the right name that redirects to
        ``../lib/Python.framework``, a directory the packager never creates:
        it puts the framework in ``Contents/Frameworks``. So the two halves
        point past each other and the bundle carries a Python that cannot start.

        Nothing about that is visible from outside. Every plugin needing an
        interpreter fell back to whatever was on PATH, and macOS has shipped no
        Python since 12.3 — ``/usr/bin/python3`` is a stub that offers to install
        the Command Line Tools. On a machine without them the plugins simply do
        not work.

        The fix is the copy itself: put the framework's real binary where the
        shim was. Run from ``Contents/MacOS`` its own load command resolves, and
        it needs no further patching.

        This hangs off ``internalCreatePackage`` and not off ``preArchive``,
        which is where it lived first and never once ran. ``preArchive`` is
        called by ``CollectionPackagerBase`` at the very top of
        ``MacBasePackager.internalCreatePackage``, and the framework does not
        arrive until ``MacDylibBundler`` runs some forty seconds later — so the
        copy looked for a Python.framework that was not there yet, logged that
        it had found none, and left. Craft offers no hook between the bundling
        and the end, so the whole of the parent runs first and this comes after.

        Which means the bundle has already been signed by the time anything is
        copied into it, and a changed binary voids that signature. So it is
        signed again here. Nothing on a CI machine without a Developer ID would
        ever show that omission: an unsigned build packages and tests exactly
        the same, and only a user's Gatekeeper would refuse it.
        """
        if not super().internalCreatePackage(defines, **kwargs):
            return False
        if not CraftCore.compiler.isMacOS:
            return True
        status = True

        # Found by looking rather than through getMacAppPath: that reads
        # defines["apppath"] with a plain subscript, and the key is put there by
        # the packager itself, not by the recipe — asking for it here raises
        # KeyError and takes the whole package step with it.
        name = f"{self.defines['appname']}.app"
        apps = [p for p in Path(self.archiveDir()).glob(f"**/{name}") if p.is_dir()]
        if not apps:
            CraftCore.log.warning(f"no {name} under {self.archiveDir()}; leaving Python alone")
            return status
        app = apps[0]
        macos = app / "Contents" / "MacOS"
        versions = app / "Contents" / "Frameworks" / "Python.framework" / "Versions"
        if not versions.is_dir():
            CraftCore.log.warning(f"no Python.framework in {app}; leaving its interpreters alone")
            return status

        fixed = self._repointPythonLibrary(versions)
        if not fixed:
            CraftCore.log.warning(f"nothing under {versions} referenced the framework's library; leaving it alone")

        replaced = False
        for real in sorted(versions.glob("*/bin/python3*")):
            if real.name.endswith("-config") or not real.is_file():
                continue
            target = macos / real.name
            CraftCore.log.info(f"replacing {target} with the interpreter from {real.parent}")
            target.unlink(missing_ok=True)
            shutil.copy2(real, target)
            target.chmod(0o755)
            replaced = True

        if not replaced:
            CraftCore.log.warning(f"no interpreter under {versions}; the bundle's Python will not start")
            return status
        # The parent signed the bundle before returning, and what was just
        # copied in is unsigned — sign the whole thing again rather than the
        # new files alone, because the seal covers the bundle as a whole.
        return CodeSign.signMacApp(app)


    @staticmethod
    def _repointPythonLibrary(versions: Path) -> int:
        """Make every Mach-O in the framework find ``Python`` from where it sits.

        The framework records its library as
        ``@executable_path/../Frameworks/Python.framework/Versions/X/Python``.
        ``@executable_path`` is the path of whatever process is running, not of
        the file holding the load command, so the reference only ever resolved
        for a binary sitting in ``Contents/MacOS`` — and not even then.

        ``bin/python3`` is not the interpreter. It is a stub that execs
        ``Resources/Python.app/Contents/MacOS/Python``, which python.org ships
        so the process gets a bundle identity. Copying the stub into
        ``Contents/MacOS`` therefore moved nothing that mattered: the exec
        handed control to a binary two directories deeper whose own load command
        then resolved against *its* location and found nothing. That is what the
        package test reported as
        ``tried: .../Resources/Python.app/Contents/Frameworks//Python.framework/...``.

        ``@loader_path`` is measured from the file that carries the load command,
        so it is right wherever the binary is run from and whatever exec'd it.
        Each binary needs its own number of ``..`` — four for the one inside
        Python.app, one for the stub in ``bin`` — so the path is computed rather
        than written down.

        Rewriting a Mach-O voids its signature, and on Apple Silicon an invalid
        one is refused outright, so each file is signed again as it is changed.
        Ad-hoc (``-``) because CI has no Developer ID; the bundle is signed
        properly afterwards where there is one.
        """
        changed = 0
        for version in sorted(versions.iterdir()):
            if version.is_symlink() or not version.is_dir():
                continue  # "Current" points at a real version already handled
            library = version / "Python"
            if not library.is_file():
                continue
            for binary in version.rglob("*"):
                if binary.is_symlink() or not binary.is_file() or binary == library:
                    continue
                if not os.access(binary, os.X_OK):
                    continue
                listing = subprocess.run(["otool", "-L", str(binary)], capture_output=True, text=True)
                if listing.returncode != 0:
                    continue  # not a Mach-O; otool says so on stderr
                for line in listing.stdout.splitlines():
                    reference = line.strip().split(" (")[0]
                    if not reference.startswith("@executable_path/") or not reference.endswith("/Python"):
                        continue
                    relative = os.path.relpath(library, binary.parent)
                    replacement = f"@loader_path/{relative}"
                    CraftCore.log.info(f"repointing {binary} at {replacement}")
                    subprocess.run(["install_name_tool", "-change", reference, replacement, str(binary)], check=True)
                    subprocess.run(["codesign", "--force", "--sign", "-", str(binary)], check=False)
                    changed += 1
                    break
        return changed

    def createPackage(self):
        if CraftCore.compiler.isMacOS:
            self.blacklist_file.append(self.blueprintDir() / "exclude_macos.list")
            # Craft's macOS blacklist throws away share/icons wholesale, which
            # takes this application's entire icon theme with it. A whitelisted
            # path outranks a blacklisted one, so this is what carries the theme
            # into the bundle — see keep_macos.list for what breaks without it.
            self.whitelist_file.append(self.blueprintDir() / "keep_macos.list")
        return super().createPackage()

    def setDefaults(self, defines: dict) -> dict:
        defines = super().setDefaults(defines)
        if CraftCore.compiler.isLinux and isinstance(self, AppImagePackager):
            # MLT resolves its modules, profiles and presets through the
            # environment. Inside an AppImage the build-time paths do not exist,
            # so point them back into the bundle or the app starts with no
            # producers and no consumers. Mirrors kdenlive's recipe.
            defines["runenv"] += [
                "PACKAGE_TYPE=appimage",
                "MLT_REPOSITORY=$this_dir/usr/lib/mlt-7/",
                "MLT_DATA=$this_dir/usr/share/mlt-7/",
                "MLT_ROOT_DIR=$this_dir/usr/",
                "MLT_APPDIR=$this_dir/usr/",
                "MLT_PROFILES_PATH=$this_dir/usr/share/mlt-7/profiles/",
                "MLT_PRESETS_PATH=$this_dir/usr/share/mlt-7/presets/",
                "LADSPA_PATH=$this_dir/usr/lib/ladspa",
                "FREI0R_PATH=$this_dir/usr/lib/frei0r-1",
                "SDL_AUDIODRIVER=pulseaudio",
                "ALSA_CONFIG_DIR=/usr/share/alsa",
                "ALSA_PLUGIN_DIR=/usr/lib/x86_64-linux-gnu/alsa-lib",
                "LIBVA_DRIVERS_PATH=/usr/lib/dri:/usr/lib64/dri:/usr/lib/x86_64-linux-gnu/dri:/usr/lib/aarch64-linux-gnu/dri",
            ]
        return defines
