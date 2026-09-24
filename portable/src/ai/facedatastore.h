/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QHash>
#include <QMap>
#include <QMutex>
#include <QRectF>
#include <QSet>
#include <QString>

/** @class FaceDataStore
    @brief Session store for face detection. "Detect faces" is a per-clip
    property: while it is on, every previewed frame of that content runs
    through the detector and the face positions collected here stay
    available for future face-based features (blur, mask seeding, chat
    tools), even though the user only sees the monitor overlay.

    Everything the detector produced is kept under the **hash of the clip's
    file**, never under its bin id: ids are per-project counters, and an unsaved
    project stores its data in a folder shared with every other unsaved project,
    so "clip 4" would otherwise inherit the boxes of a completely different
    clip 4. The hash also means the same footage analysed once is recognised
    when it comes back under another name. Whether detection is *on* stays keyed
    by bin id — that is a property of this clip in this project, not of the
    file.
 */
class FaceDataStore
{
public:
    static FaceDataStore &instance();

    /** @brief Analysis grid: the detector runs every Nth frame and positions
     *  in between are interpolated — faces do not teleport within 200 ms, so
     *  this cuts decoding + inference cost by the same factor. */
    static constexpr int kAnalyseStep = 5;

    /** @brief Stored results carry the version of the detection that produced
     *  them. Bump it whenever detection changes, so a project analysed by an
     *  older build is analysed again instead of showing its stale boxes. */
    static constexpr int kDataVersion = 2;

    /** @brief Whether face detection is enabled for this bin clip. */
    bool isEnabled(const QString &id) const;
    void setEnabled(const QString &id, bool enabled);
    /** @brief True when at least one clip has detection enabled. */
    bool hasAnyEnabled() const;

    void addFrameData(const QString &id, int position, const QList<QRectF> &faces);
    bool hasFrameData(const QString &id, int position) const;
    QList<QRectF> frameData(const QString &id, int position) const;
    /** @brief Face rects at @p position, lerped between the two nearest
     *  analysed frames when the position itself was not analysed. */
    QList<QRectF> interpolatedFrameData(const QString &id, int position) const;
    /** @brief All collected positions for a clip: frame -> face rects (0..1). */
    QMap<int, QList<QRectF>> clipData(const QString &id) const;
    /** @brief Follow one face across frames starting from @p startRect at
     *  @p startPosition, associating by nearest center (both directions).
     *  @returns frame -> rect of that face. */
    QMap<int, QRectF> buildFaceTrack(const QString &id, int startPosition, const QRectF &startRect) const;
    void clearData(const QString &id);

    /** @brief Full clip analysed — no background job needed anymore. */
    bool isComplete(const QString &id) const;
    void markComplete(const QString &id);
    /** @brief Load <folder>/faces/<hash>.json once per session (no-op after). */
    void ensureLoaded(const QString &id, const QString &projectDataFolder);
    /** @brief Persist collected data to <folder>/faces/<hash>.json. */
    void saveClip(const QString &id, const QString &projectDataFolder) const;
    /** @brief Forget everything held for the project that is closing — bin ids
     *  are about to be handed out again to different clips. */
    void clearSession();

private:
    FaceDataStore() = default;
    /** @brief Hash of the clip's file, or empty while the clip is still
     *  loading: with no identity there is nothing safe to read or write, so
     *  every operation simply does nothing until the clip is ready. */
    QString keyFor(const QString &binId) const;

    mutable QMutex m_mutex;
    QSet<QString> m_enabled;
    QSet<QString> m_complete;
    QSet<QString> m_loaded;
    QHash<QString, QMap<int, QList<QRectF>>> m_data;
    /** @brief bin id -> file hash, valid for the current project only. */
    mutable QHash<QString, QString> m_keys;
};
