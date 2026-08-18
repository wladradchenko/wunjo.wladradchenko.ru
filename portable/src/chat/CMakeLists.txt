# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>

set(wunjo_SRCS
  ${wunjo_SRCS}
    chat/chatbackend.cpp
    chat/chatbrain.cpp
    chat/pluginchatbackend.cpp
    chat/chatguidancestore.cpp
    chat/chatmessagemodel.cpp
    chat/chathistorystore.cpp
    chat/chatvoiceinput.cpp
    chat/chatwidget.cpp
  PARENT_SCOPE)
