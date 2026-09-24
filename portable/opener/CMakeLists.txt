# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>

# A one-call helper that opens a file or folder with the desktop's own handler.
# See main.cpp for why it exists and why it is installed under KDE's name.
add_executable(wunjo_open main.cpp)

target_link_libraries(wunjo_open
    Qt${QT_MAJOR_VERSION}::Core
    Qt${QT_MAJOR_VERSION}::Gui)

set_target_properties(wunjo_open PROPERTIES
    # The name KIO looks for. The target keeps ours so it reads as what it is
    # in build output and in this file.
    OUTPUT_NAME kde-open
    # No console window flashing over the editor on Windows when KIO starts it,
    # and a plain binary beside wunjo_render on macOS rather than an .app.
    WIN32_EXECUTABLE TRUE
    MACOSX_BUNDLE FALSE)

install(TARGETS wunjo_open DESTINATION ${KDE_INSTALL_BINDIR})
