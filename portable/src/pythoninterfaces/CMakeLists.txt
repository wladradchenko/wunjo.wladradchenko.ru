# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: Julius Künzel <julius.kuenzel@kde.org>

add_subdirectory(dialogs)
set(wunjo_SRCS
  ${wunjo_SRCS}
  pythoninterfaces/saminterface.cpp
  pythoninterfaces/speechtotext.cpp
  pythoninterfaces/speechtotextvosk.cpp
  pythoninterfaces/speechtotextwhisper.cpp
  pythoninterfaces/abstractpythoninterface.cpp
  PARENT_SCOPE
)
