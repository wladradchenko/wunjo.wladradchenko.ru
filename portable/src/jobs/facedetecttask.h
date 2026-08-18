/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "abstracttask.h"

#include <QList>
#include <QPair>

/** @class FaceDetectTask
    @brief Background analysis started when face detection is enabled on a
    clip: runs the bundled face detector over every frame and fills
    FaceDataStore, which persists to <projectDataFolder>/faces/<binId>.json.
    The monitors draw their face overlays from that data, so scrubbing and
    re-enabling never re-detect.

    Switched on from a timeline clip, it analyses that clip's own range before
    the rest of the file. A ten-minute source cut down to five seconds used to
    be analysed from its first frame regardless, so the piece the user was
    actually looking at got its boxes last — after minutes of work on footage
    that is not in the edit. The whole clip is still covered, just not first.
 */
class FaceDetectTask : public AbstractTask
{
public:
    /** @param from,to the part to analyse before anything else, in frames of
     *  the bin clip; -1 means "no part is more urgent than another". */
    static void start(const QString &binId, QObject *object = nullptr, int from = -1, int to = -1);

protected:
    void run() override;

private:
    FaceDetectTask(const ObjectId &owner, const QString &binId, const QString &saveFolder, QObject *object, int from, int to);
    /** @brief Walk the urgent range first, then everything else. Frames already
     *  in the store are skipped either way, so the two passes never overlap. */
    QList<QPair<int, int>> passes(int duration) const;
    QString m_binId;
    QString m_saveFolder;
    int m_urgentFrom;
    int m_urgentTo;
};
