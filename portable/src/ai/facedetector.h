/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QImage>
#include <QList>
#include <QRectF>

/** @class FaceDetector
    @brief CPU face detection on a single frame using the bundled
    RetinaFace-MobileNet0.25 ONNX model (~2 MB, ships with the app in
    share/wunjo/ai/) through OpenCV DNN — no downloads, no venv.

    Port of the reference implementation from the Wunjo v2 web app
    (portable/src/wunjo/static/modules/js/mnet/detect.dev.js).
 */
class FaceDetector
{
public:
    /** @brief True when the bundled model file is present and loadable. */
    static bool isAvailable();

    /** @brief Detect faces on @p image.
        @returns bounding boxes normalized to 0..1 of the frame size. */
    static QList<QRectF> detect(const QImage &image, float probThreshold = 0.75f, float nmsThreshold = 0.5f);
};
