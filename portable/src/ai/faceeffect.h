/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QSize>
#include <QString>

class QRectF;

/** @namespace FaceEffect
    @brief Shared face-tracking → keyframe logic for the Motion Tracker
    (opencv.tracker) effect. Used by both the monitor's "Hide Face" menu and
    the scripting/MCP entry point so there is a SINGLE source for the approved
    keyframe shape (movement-only keyframes; per-frame dumps freeze MLT — see
    desktop/CLAUDE.md, do not "optimize"). */
namespace FaceEffect {

/** @brief Build the opencv.tracker `results` keyframe string tracking @p face
    (a normalized 0..1 rect detected at source frame @p sourcePos) across the
    clip, offset by the clip in-point @p clipIn and scaled to @p profile.
    @return the keyframe string, or empty if the track yields no points. */
QString buildTrackerResults(const QString &binId, int sourcePos, int clipIn,
                            const QRectF &face, const QSize &profile);

/** @brief Auto blur/pixelate intensity scaled to the face size (in @p profile
    pixels), so small and large faces are obscured to the same degree. */
int autoStrength(const QRectF &face, const QSize &profile);

} // namespace FaceEffect
