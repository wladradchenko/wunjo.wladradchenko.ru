# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: Vincent Pinon <vpinon@kde.org>, Jean-Baptiste Mardelle <jb@kdenlive.org>

set(QT_DONT_USE_QTGUI 1)

set(wunjo_render_SRCS
  main.cpp
  renderjob.cpp
  ../src/lib/localeHandling.cpp
)

ecm_qt_declare_logging_category(wunjo_render_SRCS
    HEADER wunjo_renderer_debug.h
    IDENTIFIER WUNJO_RENDERER_LOG
    CATEGORY_NAME online.wunjo.make.render
    DESCRIPTION "wunjo"
    EXPORT WUNJO
)

add_executable(wunjo_render ${wunjo_render_SRCS})
ecm_mark_nongui_executable(wunjo_render)

target_link_libraries(wunjo_render
    Qt${QT_MAJOR_VERSION}::Core
    Qt${QT_MAJOR_VERSION}::Widgets
    Qt${QT_MAJOR_VERSION}::Xml
    Mlt7::mlt++)
# The render process reports progress over a QLocalSocket to RenderServer —
# renderjob.cpp has never used D-Bus, only the link was conditional on it.
target_link_libraries(wunjo_render Qt${QT_MAJOR_VERSION}::Network)

install(TARGETS wunjo_render DESTINATION ${KDE_INSTALL_BINDIR})
