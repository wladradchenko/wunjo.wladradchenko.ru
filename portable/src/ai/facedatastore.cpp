/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "facedatastore.h"

#include "bin/projectclip.h"
#include "bin/projectitemmodel.h"
#include "core.h"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMutexLocker>

#include <cmath>

FaceDataStore &FaceDataStore::instance()
{
    static FaceDataStore store;
    return store;
}

QString FaceDataStore::keyFor(const QString &binId) const
{
    {
        QMutexLocker locker(&m_mutex);
        const auto cached = m_keys.constFind(binId);
        if (cached != m_keys.constEnd()) {
            return cached.value();
        }
    }
    QString key;
    if (std::shared_ptr<ProjectClip> clip = pCore->projectItemModel()->getClipByBinID(binId)) {
        // empty while the clip is still loading — its hash is computed with the
        // producer, so there is simply no identity to file results under yet
        key = clip->hash();
    }
    if (key.isEmpty()) {
        return key;
    }
    QMutexLocker locker(&m_mutex);
    m_keys.insert(binId, key);
    return key;
}

void FaceDataStore::clearSession()
{
    QMutexLocker locker(&m_mutex);
    m_keys.clear();
    m_enabled.clear();
    m_loaded.clear();
    m_complete.clear();
    m_data.clear();
}

bool FaceDataStore::isEnabled(const QString &id) const
{
    QMutexLocker locker(&m_mutex);
    return m_enabled.contains(id);
}

void FaceDataStore::setEnabled(const QString &id, bool enabled)
{
    QMutexLocker locker(&m_mutex);
    if (enabled) {
        m_enabled.insert(id);
    } else {
        m_enabled.remove(id);
    }
}

bool FaceDataStore::hasAnyEnabled() const
{
    QMutexLocker locker(&m_mutex);
    return !m_enabled.isEmpty();
}

void FaceDataStore::addFrameData(const QString &id, int position, const QList<QRectF> &faces)
{
    const QString key = keyFor(id);
    if (key.isEmpty()) {
        return;
    }
    QMutexLocker locker(&m_mutex);
    m_data[key][position] = faces;
}

QList<QRectF> FaceDataStore::frameData(const QString &id, int position) const
{
    const QString key = keyFor(id);
    QMutexLocker locker(&m_mutex);
    return m_data.value(key).value(position);
}

QMap<int, QList<QRectF>> FaceDataStore::clipData(const QString &id) const
{
    const QString key = keyFor(id);
    QMutexLocker locker(&m_mutex);
    return m_data.value(key);
}

QList<QRectF> FaceDataStore::interpolatedFrameData(const QString &id, int position) const
{
    const QString key = keyFor(id);
    QMutexLocker locker(&m_mutex);
    const auto clip = m_data.constFind(key);
    if (clip == m_data.constEnd() || clip->isEmpty()) {
        return {};
    }
    const QMap<int, QList<QRectF>> &data = *clip;
    const auto exact = data.constFind(position);
    if (exact != data.constEnd()) {
        return exact.value();
    }
    auto after = data.upperBound(position);
    if (after == data.constBegin()) {
        return (after.key() - position) <= kAnalyseStep ? after.value() : QList<QRectF>();
    }
    const auto before = std::prev(after);
    if (after == data.constEnd() || (after.key() - before.key()) > kAnalyseStep * 3) {
        // trailing edge or a hole in the data — hold the last analysed frame
        // briefly instead of interpolating across an unknown span
        return (position - before.key()) <= kAnalyseStep ? before.value() : QList<QRectF>();
    }
    const qreal t = qreal(position - before.key()) / (after.key() - before.key());
    QList<QRectF> result;
    for (const QRectF &from : before.value()) {
        // pair with the nearest face on the other side of the gap
        QRectF to;
        qreal bestDist = 0.2;
        for (const QRectF &candidate : after.value()) {
            const QPointF delta = candidate.center() - from.center();
            const qreal dist = std::sqrt(delta.x() * delta.x() + delta.y() * delta.y());
            if (dist < bestDist) {
                bestDist = dist;
                to = candidate;
            }
        }
        if (to.isNull()) {
            result.append(from);
            continue;
        }
        result.append(QRectF(from.x() + (to.x() - from.x()) * t, from.y() + (to.y() - from.y()) * t, from.width() + (to.width() - from.width()) * t,
                             from.height() + (to.height() - from.height()) * t));
    }
    return result;
}

void FaceDataStore::clearData(const QString &id)
{
    const QString key = keyFor(id);
    QMutexLocker locker(&m_mutex);
    m_data.remove(key);
}

bool FaceDataStore::hasFrameData(const QString &id, int position) const
{
    const QString key = keyFor(id);
    QMutexLocker locker(&m_mutex);
    const auto it = m_data.constFind(key);
    return it != m_data.constEnd() && it->contains(position);
}

bool FaceDataStore::isComplete(const QString &id) const
{
    const QString key = keyFor(id);
    QMutexLocker locker(&m_mutex);
    return !key.isEmpty() && m_complete.contains(key);
}

void FaceDataStore::markComplete(const QString &id)
{
    const QString key = keyFor(id);
    if (key.isEmpty()) {
        return;
    }
    QMutexLocker locker(&m_mutex);
    m_complete.insert(key);
}

void FaceDataStore::ensureLoaded(const QString &id, const QString &projectDataFolder)
{
    const QString key = keyFor(id);
    if (projectDataFolder.isEmpty() || key.isEmpty()) {
        return;
    }
    {
        QMutexLocker locker(&m_mutex);
        if (m_loaded.contains(key)) {
            return;
        }
        m_loaded.insert(key);
    }
    QFile file(projectDataFolder + QStringLiteral("/faces/") + key + QStringLiteral(".json"));
    if (!file.open(QIODevice::ReadOnly)) {
        return;
    }
    const QJsonObject root = QJsonDocument::fromJson(file.readAll()).object();
    if (root.value(QStringLiteral("version")).toInt() != kDataVersion) {
        // analysed by an older detector — ignore it and let the clip be analysed again
        return;
    }
    const QJsonObject frames = root.value(QStringLiteral("frames")).toObject();
    QMap<int, QList<QRectF>> data;
    for (auto it = frames.constBegin(); it != frames.constEnd(); ++it) {
        QList<QRectF> rects;
        const QJsonArray list = it.value().toArray();
        for (const QJsonValue &value : list) {
            const QJsonArray r = value.toArray();
            if (r.size() == 4) {
                rects.append(QRectF(r.at(0).toDouble(), r.at(1).toDouble(), r.at(2).toDouble(), r.at(3).toDouble()));
            }
        }
        data.insert(it.key().toInt(), rects);
    }
    QMutexLocker locker(&m_mutex);
    // keep freshly detected frames over stored ones
    QMap<int, QList<QRectF>> &existing = m_data[key];
    for (auto it = data.constBegin(); it != data.constEnd(); ++it) {
        if (!existing.contains(it.key())) {
            existing.insert(it.key(), it.value());
        }
    }
    if (root.value(QStringLiteral("complete")).toBool()) {
        m_complete.insert(key);
    }
}

void FaceDataStore::saveClip(const QString &id, const QString &projectDataFolder) const
{
    const QString key = keyFor(id);
    if (projectDataFolder.isEmpty() || key.isEmpty()) {
        return;
    }
    QMap<int, QList<QRectF>> data;
    bool complete;
    {
        QMutexLocker locker(&m_mutex);
        data = m_data.value(key);
        complete = m_complete.contains(key);
    }
    if (data.isEmpty()) {
        return;
    }
    QDir dir(projectDataFolder);
    dir.mkpath(QStringLiteral("faces"));
    QJsonObject frames;
    for (auto it = data.constBegin(); it != data.constEnd(); ++it) {
        QJsonArray rects;
        for (const QRectF &r : it.value()) {
            rects.append(QJsonArray{r.x(), r.y(), r.width(), r.height()});
        }
        frames.insert(QString::number(it.key()), rects);
    }
    QJsonObject root;
    root.insert(QStringLiteral("version"), kDataVersion);
    root.insert(QStringLiteral("frames"), frames);
    root.insert(QStringLiteral("complete"), complete);
    QFile file(dir.absoluteFilePath(QStringLiteral("faces/") + key + QStringLiteral(".json")));
    if (file.open(QIODevice::WriteOnly)) {
        file.write(QJsonDocument(root).toJson(QJsonDocument::Compact));
    }
}

QMap<int, QRectF> FaceDataStore::buildFaceTrack(const QString &id, int startPosition, const QRectF &startRect) const
{
    QMap<int, QList<QRectF>> data;
    {
        const QString key = keyFor(id);
        QMutexLocker locker(&m_mutex);
        data = m_data.value(key);
    }
    QMap<int, QRectF> track;
    if (data.isEmpty()) {
        track.insert(startPosition, startRect);
        return track;
    }
    // association by nearest center within a sane jump distance
    const auto nearest = [](const QList<QRectF> &candidates, const QRectF &reference) -> QRectF {
        QRectF best;
        qreal bestDist = 0.2; // normalized units; larger jumps break the track
        for (const QRectF &rect : candidates) {
            const QPointF delta = rect.center() - reference.center();
            const qreal dist = std::sqrt(delta.x() * delta.x() + delta.y() * delta.y());
            if (dist < bestDist) {
                bestDist = dist;
                best = rect;
            }
        }
        return best;
    };
    track.insert(startPosition, startRect);
    // forward
    QRectF previous = startRect;
    for (auto it = data.upperBound(startPosition); it != data.constEnd(); ++it) {
        const QRectF match = nearest(it.value(), previous);
        if (match.isNull()) {
            break;
        }
        track.insert(it.key(), match);
        previous = match;
    }
    // backward
    previous = startRect;
    auto it = data.lowerBound(startPosition);
    while (it != data.constBegin()) {
        --it;
        if (it.key() >= startPosition) {
            continue;
        }
        const QRectF match = nearest(it.value(), previous);
        if (match.isNull()) {
            break;
        }
        track.insert(it.key(), match);
        previous = match;
    }
    return track;
}
