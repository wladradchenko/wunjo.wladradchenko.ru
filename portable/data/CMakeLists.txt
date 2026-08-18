# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: Vincent PINON <vincent.pinon@laposte.net>, Jean-Baptiste Mardelle <jb@kdenlive.org>, Julius Künzel <julius.kuenzel@kde.org>

add_subdirectory(ai)
add_subdirectory(effects)
add_subdirectory(generators)
add_subdirectory(layouts)
add_subdirectory(transitions)
add_subdirectory(icons)
add_subdirectory(icons-wunjo)
add_subdirectory(knewstuff)
add_subdirectory(lumas)
add_subdirectory(scripts)
add_subdirectory(man)
add_subdirectory(titles)
add_subdirectory(profiles)
add_subdirectory(resourceproviders)
add_subdirectory(shortcuts)
add_subdirectory(luts)

install(FILES
  meta_magiclantern.png
  encodingprofiles.rc
  externalproxies.rc
  effectscategory.rc
  transcodingrc
  slideanimations.rc
  clipjobsettings.rc

  DESTINATION ${KDE_INSTALL_DATADIR}${DATA_INSTALL_PREFIX})

install(FILES wunjo.notifyrc DESTINATION ${KDE_INSTALL_KNOTIFYRCDIR})

# Wunjo brand color schemes (dark default + light; mint accent)
install(FILES color-schemes/Wunjo.colors color-schemes/WunjoLight.colors DESTINATION ${KDE_INSTALL_DATADIR}/color-schemes)

# Wunjo brand fonts (loaded at runtime via QFontDatabase in main.cpp)
install(DIRECTORY fonts DESTINATION ${KDE_INSTALL_DATADIR}${DATA_INSTALL_PREFIX})

# Built-in AI plugins shipped out of the box. They install to
# share/wunjo/plugins/<id>/, a read-only AppLocalData root PluginManager::rescan()
# scans as "bundled" — optional (the app runs without them), not compiled in.
# In directory data/plugins put default plugiins
install(DIRECTORY plugins DESTINATION ${KDE_INSTALL_DATADIR}${DATA_INSTALL_PREFIX})

# The MCP server the assistant speaks through — the same one whether the driver
# is the built-in model or Claude Code in a terminal. It lives at the top of the
# repository (it is used standalone during development too) and is copied into
# the build by the flatpak manifest, so a plain source checkout has no such
# folder and simply ships without it.
if(EXISTS ${CMAKE_SOURCE_DIR}/mcp)
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/mcp DESTINATION ${KDE_INSTALL_DATADIR}${DATA_INSTALL_PREFIX}
            PATTERN "__pycache__" EXCLUDE
            PATTERN ".venv" EXCLUDE
            PATTERN "tests" EXCLUDE)
    # The launcher an outside agent names in .mcp.json goes inside that folder,
    # as "mcp/start": everything about the server then sits in one place, and
    # the name says what it does without repeating the application's own.
    install(PROGRAMS mcp_start DESTINATION ${KDE_INSTALL_DATADIR}${DATA_INSTALL_PREFIX}/mcp RENAME start)
endif()

install(FILES profiles.xml DESTINATION ${KDE_INSTALL_DATADIR}${DATA_INSTALL_PREFIX}/export)
# The metainfo is what software centres and Flathub read, so it ships by default.
# It is opt-out only because the local dev box runs flatpak-builder 1.2.2, which calls
# the removed `appstream-compose` (the KDE 6.10 SDK ships only `appstreamcli compose`)
# as soon as it sees a metainfo file. That build passes -DINSTALL_METAINFO=OFF; every
# real package (distro, snap, Flathub, CI) must leave it ON.
option(INSTALL_METAINFO "Install the AppStream metainfo file" ON)
if(INSTALL_METAINFO)
    install(FILES online.wunjo.make.appdata.xml DESTINATION ${KDE_INSTALL_METAINFODIR})
endif()
install(FILES online.wunjo.make.desktop DESTINATION ${KDE_INSTALL_APPDIR})

find_package(SharedMimeInfo REQUIRED)
install(FILES online.wunjo.make.xml westley.xml DESTINATION ${KDE_INSTALL_MIMEDIR})
update_xdg_mimetypes(${KDE_INSTALL_MIMEDIR})

