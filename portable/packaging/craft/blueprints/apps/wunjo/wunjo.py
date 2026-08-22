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

import info
from CraftCore import CraftCore
from Package.CMakePackageBase import CMakePackageBase
from Packager.AppImagePackager import AppImagePackager


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
