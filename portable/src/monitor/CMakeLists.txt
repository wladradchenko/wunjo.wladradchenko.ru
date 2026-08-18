# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: Jean-Baptiste Mardelle <jb@kdenlive.org>, Vincent PINON <vincent.pinon@laposte.net>

add_subdirectory(scopes)

if (APPLE)
    list(APPEND wunjo_SRCS
        monitor/videowidget.cpp
        monitor/metalvideowidget.mm
    )
elseif (WIN32)
    list(APPEND wunjo_SRCS
        monitor/videowidget.cpp
        monitor/openglvideowidget.cpp
        monitor/d3dvideowidget.cpp
    )
else()
    list(APPEND wunjo_SRCS
        monitor/videowidget.cpp
        monitor/openglvideowidget.cpp
    )
endif()

set(wunjo_SRCS
    ${wunjo_SRCS}
    monitor/abstractmonitor.cpp
    monitor/monitor.cpp
    monitor/monitormanager.cpp
    monitor/recmanager.cpp
    monitor/qmlmanager.cpp
    monitor/monitorproxy.cpp
    monitor/view/qmliconprovider.cpp
    PARENT_SCOPE
)
