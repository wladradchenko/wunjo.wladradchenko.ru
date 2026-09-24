# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: Nicolas Carion <french.ebook.lover@gmail.com>

set(wunjo_SRCS
  ${wunjo_SRCS}
  transitions/transitionsrepository.cpp
  transitions/transitionlist/view/transitionlistwidget.cpp
  transitions/transitionlist/view/transitionicondelegate.cpp
  transitions/transitionlist/model/transitiontreemodel.cpp
  transitions/transitionlist/model/transitionfilter.cpp
  transitions/view/transitionstackview.cpp
  transitions/view/mixstackview.cpp
  PARENT_SCOPE)

