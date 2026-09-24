/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "faceeffect.h"
#include "facedatastore.h"

#include <QRectF>
#include <QStringList>

namespace FaceEffect {

QString buildTrackerResults(const QString &binId, int sourcePos, int clipIn, const QRectF &face, const QSize &profile)
{
    const QMap<int, QRectF> track = FaceDataStore::instance().buildFaceTrack(binId, sourcePos, face);
    QList<QPair<int, QRectF>> points;
    points.reserve(track.size());
    for (auto it = track.constBegin(); it != track.constEnd(); ++it) {
        const int local = it.key() - clipIn;
        if (local < 0) {
            continue;
        }
        // RetinaFace boxes are tight (brow→lip). Expand asymmetrically to cover
        // the whole head: extra top for hair/forehead, MORE at the bottom for
        // the mouth/chin/jaw the raw box cuts off, moderate on the sides.
        QRectF rect = it.value();
        const qreal padX = rect.width() * 0.20;
        const qreal padTop = rect.height() * 0.30;
        const qreal padBottom = rect.height() * 0.45;
        rect.adjust(-padX, -padTop, padX, padBottom);
        points.append({local, rect.intersected(QRectF(0, 0, 1, 1))});
    }
    if (points.isEmpty()) {
        return QString();
    }
    // The raw track holds one rect per analysed frame; MLT's animation parser
    // and the keyframe model are quadratic in keyframe count, so on long clips
    // a full dump freezes the UI for minutes. Keep a keyframe only when the
    // face actually moves, anchoring the frame before each move so the linear
    // interpolation cannot drift across still spans — the 15% padding above
    // absorbs whatever remains.
    const qreal moveEps = 0.008;
    QList<QPair<int, QRectF>> kept;
    kept.append(points.constFirst());
    for (int i = 1; i < points.size(); ++i) {
        const QRectF &last = kept.constLast().second;
        const QRectF &cur = points.at(i).second;
        const bool moved = qAbs(cur.center().x() - last.center().x()) > moveEps || qAbs(cur.center().y() - last.center().y()) > moveEps ||
                           qAbs(cur.width() - last.width()) > 2 * moveEps || qAbs(cur.height() - last.height()) > 2 * moveEps;
        if (moved || i == points.size() - 1) {
            if (points.at(i - 1).first > kept.constLast().first) {
                kept.append(points.at(i - 1));
            }
            if (points.at(i).first > kept.constLast().first) {
                kept.append(points.at(i));
            }
        }
    }
    // hard ceiling for pathological cases (continuous motion on hour-long clips)
    const int maxKeyframes = 2000;
    if (kept.size() > maxKeyframes) {
        QList<QPair<int, QRectF>> sampled;
        const int step = (kept.size() + maxKeyframes - 1) / maxKeyframes;
        for (int i = 0; i < kept.size(); i += step) {
            sampled.append(kept.at(i));
        }
        if (sampled.constLast().first != kept.constLast().first) {
            sampled.append(kept.constLast());
        }
        kept = sampled;
    }
    QStringList keyframes;
    for (const auto &point : std::as_const(kept)) {
        keyframes << QStringLiteral("%1=%2 %3 %4 %5")
                         .arg(point.first)
                         .arg(qRound(point.second.x() * profile.width()))
                         .arg(qRound(point.second.y() * profile.height()))
                         .arg(qMax(1, qRound(point.second.width() * profile.width())))
                         .arg(qMax(1, qRound(point.second.height() * profile.height())));
    }
    // collapse the region right outside the tracked range so nothing stays
    // blurred while the face is off screen
    const int firstLocal = kept.constFirst().first;
    const int lastLocal = kept.constLast().first;
    const QString firstRect = keyframes.constFirst().section(QLatin1Char('='), 1);
    const QString lastRect = keyframes.constLast().section(QLatin1Char('='), 1);
    if (firstLocal > 0) {
        keyframes.prepend(QStringLiteral("%1=%2 %3 0 0")
                              .arg(firstLocal - 1)
                              .arg(firstRect.section(QLatin1Char(' '), 0, 0), firstRect.section(QLatin1Char(' '), 1, 1)));
    }
    keyframes.append(QStringLiteral("%1=%2 %3 0 0")
                         .arg(lastLocal + 1)
                         .arg(lastRect.section(QLatin1Char(' '), 0, 0), lastRect.section(QLatin1Char(' '), 1, 1)));
    return keyframes.join(QLatin1Char(';'));
}

int autoStrength(const QRectF &face, const QSize &profile)
{
    // Pixelate/blur intensity ~ face_width / 7 → roughly a 7-block mosaic across
    // the face (unrecognizable but clearly a face), clamped to the filter's sane
    // range so tiny faces still blur and huge ones don't max out.
    const int facePx = qRound(face.width() * profile.width());
    return qBound(15, facePx / 7, 120);
}

} // namespace FaceEffect
