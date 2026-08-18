# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: Julius Künzel <julius.kuenzel@kde.org>

list(APPEND wunjo_SRCS
  render/renderserver.cpp)

set(wunjo_SRCS
  ${wunjo_SRCS}
  render/renderrequest.cpp
  PARENT_SCOPE)
