# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>

# Bundled AI models (small, CPU-friendly, no downloads needed):
# mnet.25_v2.simplify.onnx — RetinaFace-MobileNet0.25 face detector (~2 MB)
install(FILES
  mnet.25_v2.simplify.onnx

  DESTINATION ${KDE_INSTALL_DATADIR}${DATA_INSTALL_PREFIX}/ai)
