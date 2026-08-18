# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
# SPDX-License-Identifier: BSD-2-Clause

"""Craft blueprint for Wunjo Make.

Craft is KDE's build system for Windows and macOS: it knows how to build Qt, the
KDE Frameworks and their dependencies, and pulls most of them from a prebuilt
binary cache instead of compiling them. This file is the recipe for *this*
application on top of that.

The dependency list mirrors `find_package` in `desktop/CMakeLists.txt` and
`desktop/src/CMakeLists.txt`. When one changes, so must the other, or the build
fails on the runner with a missing package rather than here.

The source directory is passed in by the workflow with
`--options wunjo.srcDir=<checkout>/desktop`, so this blueprint never fetches
anything itself.

The repeated name in the path is Craft's rule, not a choice: it finds blueprints
by looking for `<category>/<name>/<name>.py`, so the file has to be named after
its directory. `apps` is the category, the same way KDE's own repository uses
`kde/kdemultimedia/kdenlive/kdenlive.py`.
"""

import info
from Package.CMakePackageBase import CMakePackageBase


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
        self.buildDependencies["dev-utils/pkg-config"] = None

        # Qt 6 — see find_package(Qt6 …) in CMakeLists.txt
        for module in (
            "qtbase",
            "qtdeclarative",
            "qtsvg",
            "qtmultimedia",
            "qtnetworkauth",
            "qttools",
            "qt5compat",
        ):
            self.runtimeDependencies[f"libs/qt6/{module}"] = None

        # KDE Frameworks 6 — see find_package(KF6 …) in CMakeLists.txt
        for framework in (
            "ki18n",
            "karchive",
            "kbookmarks",
            "kcodecs",
            "kcoreaddons",
            "kconfig",
            "kconfigwidgets",
            "kio",
            "kwidgetsaddons",
            "knotifyconfig",
            "knewstuff",
            "kxmlgui",
            "knotifications",
            "kguiaddons",
            "ktextwidgets",
            "kiconthemes",
            "solid",
            "kfilemetadata",
            "purpose",
            "kcrash",
            "kdoctools",
            "breeze-icons",
        ):
            self.runtimeDependencies[f"kde/frameworks/{framework}"] = None

        # Everything else the editor links
        self.runtimeDependencies["kde/thirdparty/kddockwidgets"] = None
        self.runtimeDependencies["libs/mlt"] = None
        self.runtimeDependencies["libs/ffmpeg"] = None
        self.runtimeDependencies["libs/opencv"] = None
        self.runtimeDependencies["libs/opentimelineio"] = None
        self.runtimeDependencies["libs/frei0r-plugins"] = None


class Package(CMakePackageBase):
    def __init__(self):
        CMakePackageBase.__init__(self)
        # RELEASE_BUILD strips the git revision from the version string; tests
        # are not run on the packaging runner, and they pull extra dependencies.
        self.subinfo.options.configure.args += [
            "-DRELEASE_BUILD=ON",
            "-DBUILD_TESTING=OFF",
        ]
