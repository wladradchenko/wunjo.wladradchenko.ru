# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>

set(wunjo_SRCS
  ${wunjo_SRCS}
    plugins/filedownloader.cpp
    plugins/filedownloadjob.cpp
    plugins/pluginmanifest.cpp
    plugins/pluginmanager.cpp
    plugins/plugineffects.cpp
    plugins/pluginpythonenv.cpp
    plugins/pluginmanagerwidget.cpp
    plugins/pluginsetpanel.cpp
    plugins/pluginsetstore.cpp
    plugins/pluginsettingstab.cpp
    plugins/pluginaboutdialog.cpp
  PARENT_SCOPE)
