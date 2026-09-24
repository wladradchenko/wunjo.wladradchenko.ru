# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>

# Built-in assistant guidance: the editing-craft skills every agent driving the
# editor starts with, and the loops that string them into whole jobs. Read-only
# here — ChatGuidanceStore merges these directories with the user's own library
# and writes only to the user's, so an app update ships better guidance without
# touching anything the user edited.
install(DIRECTORY skills loops DESTINATION ${KDE_INSTALL_DATADIR}${DATA_INSTALL_PREFIX}/guidance)
