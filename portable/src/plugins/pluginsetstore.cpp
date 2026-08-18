/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "pluginsetstore.h"

#include "bin/projectclip.h"
#include "core.h"
#include "doc/wunjodoc.h"

#include <KLocalizedString>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QCryptographicHash>
#include <QRegularExpression>

namespace {

QJsonObject readJson(const QString &file)
{
    QFile handle(file);
    if (!handle.open(QIODevice::ReadOnly)) {
        return {};
    }
    return QJsonDocument::fromJson(handle.readAll()).object();
}

QString displayName(const QString &wanted)
{
    QString name = wanted.simplified();
    name.replace(QRegularExpression(QStringLiteral("[/\\\\:*?\"<>|]")), QStringLiteral("_"));
    return name.isEmpty() ? i18n("set") : name;
}

/** @brief Write @p set as the file of @p key, replacing what was there. */
bool writeSet(const QDir &dir, const QString &key, QJsonObject set, QString *errorOut)
{
    const QString destination = dir.absoluteFilePath(key + QStringLiteral(".json"));
    QFile file(destination);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (errorOut) {
            *errorOut = i18n("Could not write %1", destination);
        }
        return false;
    }
    file.write(QJsonDocument(set).toJson(QJsonDocument::Compact));
    return true;
}

} // namespace

namespace PluginSets {

QString hashOf(const QString &path)
{
    if (path.isEmpty() || !QFileInfo::exists(path)) {
        return {};
    }
    // the editor's own clip hash: a digest of the file's head, tail and size
    return QString::fromLatin1(ProjectClip::calculateHash(path).first.toHex());
}

QString folder(const QString &pluginId)
{
    if (pluginId.isEmpty() || pCore->currentDoc() == nullptr) {
        return {};
    }
    const QString dataFolder = pCore->currentDoc()->projectDataFolder();
    if (dataFolder.isEmpty()) {
        return {};
    }
    const QString path = dataFolder + QStringLiteral("/plugin-sets/") + pluginId;
    QDir().mkpath(path);
    return path;
}

Set read(const QString &file)
{
    Set set;
    const QJsonObject root = readJson(file);
    if (root.isEmpty()) {
        return set;
    }
    set.file = file;
    set.name = root.value(QStringLiteral("name")).toString(QFileInfo(file).completeBaseName());
    set.kind = root.value(QStringLiteral("kind")).toString();
    set.source = root.value(QStringLiteral("source")).toString();
    set.sourceHash = root.value(QStringLiteral("source_hash")).toString(QFileInfo(file).completeBaseName());
    set.fps = root.value(QStringLiteral("fps")).toDouble();
    set.count = root.value(QStringLiteral("count")).toInt();
    if (set.count == 0) {
        // count is a convenience, the values decide
        const QJsonObject values = root.value(QStringLiteral("values")).toObject();
        for (auto it = values.constBegin(); it != values.constEnd(); ++it) {
            set.count = qMax(set.count, it.value().toArray().size());
        }
    }
    return set;
}

QVector<Set> sets(const QString &pluginId, const QString &kind)
{
    QVector<Set> result;
    const QString path = folder(pluginId);
    if (path.isEmpty()) {
        return result;
    }
    QDir dir(path);
    const QStringList files = dir.entryList({QStringLiteral("*.json")}, QDir::Files, QDir::Name);
    for (const QString &file : files) {
        const Set set = read(dir.absoluteFilePath(file));
        if (set.isValid() && (kind.isEmpty() || set.kind == kind)) {
            result.append(set);
        }
    }
    return result;
}

Set store(const QString &pluginId, const QString &name, const QString &kind, const QString &resultFile, QString *errorOut)
{
    const QString path = folder(pluginId);
    if (path.isEmpty()) {
        if (errorOut) {
            *errorOut = i18n("Save the project first — sets are stored next to it.");
        }
        return {};
    }
    QJsonObject root = readJson(resultFile);
    // Either a value per frame (a recorded performance) or a plain description
    // of what was analysed (a face embedding, an audio track): both are sets.
    if (root.value(QStringLiteral("values")).toObject().isEmpty() && root.value(QStringLiteral("data")).toObject().isEmpty()) {
        if (errorOut) {
            *errorOut = i18n("The analysis returned nothing.");
        }
        return {};
    }
    // Filed under the hash of what was analysed, not under a name: analysing the
    // same performance again updates that one set instead of leaving "take 2"
    // behind, and a set keeps working when the source is renamed or moved.
    const QString source = root.value(QStringLiteral("source")).toString();
    QString key = hashOf(source);
    if (key.isEmpty()) {
        // source already gone (a temporary export?) — fall back to the content
        key = QString::fromLatin1(QCryptographicHash::hash(QJsonDocument(root).toJson(QJsonDocument::Compact), QCryptographicHash::Md5).toHex());
    }
    QDir dir(path);
    root.insert(QStringLiteral("plugin"), pluginId);
    root.insert(QStringLiteral("name"), displayName(name));
    root.insert(QStringLiteral("source_hash"), key);
    // The plugin may say what it produced; otherwise it is what was asked for.
    if (!kind.isEmpty() && root.value(QStringLiteral("kind")).toString().isEmpty()) {
        root.insert(QStringLiteral("kind"), kind);
    }
    if (!writeSet(dir, key, root, errorOut)) {
        return {};
    }
    return read(dir.absoluteFilePath(key + QStringLiteral(".json")));
}

bool remove(const QString &file)
{
    return QFile::remove(file);
}

QString importSet(const QString &pluginId, const QString &file, QString *errorOut)
{
    const QString path = folder(pluginId);
    if (path.isEmpty()) {
        if (errorOut) {
            *errorOut = i18n("Save the project first — sets are stored next to it.");
        }
        return {};
    }
    const QJsonObject root = readJson(file);
    if (root.value(QStringLiteral("values")).toObject().isEmpty() && root.value(QStringLiteral("data")).toObject().isEmpty()) {
        if (errorOut) {
            *errorOut = i18n("%1 is not a set file.", QFileInfo(file).fileName());
        }
        return {};
    }
    // Same identity as a freshly recorded set, so importing what another project
    // recorded from the same video lands on the same file instead of doubling it.
    QJsonObject copy = root;
    QString key = copy.value(QStringLiteral("source_hash")).toString();
    if (key.isEmpty()) {
        key = hashOf(copy.value(QStringLiteral("source")).toString());
    }
    if (key.isEmpty()) {
        key = QString::fromLatin1(QCryptographicHash::hash(QJsonDocument(root).toJson(QJsonDocument::Compact), QCryptographicHash::Md5).toHex());
    }
    QDir dir(path);
    copy.insert(QStringLiteral("plugin"), pluginId);
    copy.insert(QStringLiteral("name"), displayName(copy.value(QStringLiteral("name")).toString(QFileInfo(file).completeBaseName())));
    copy.insert(QStringLiteral("source_hash"), key);
    if (!writeSet(dir, key, copy, errorOut)) {
        return {};
    }
    return dir.absoluteFilePath(key + QStringLiteral(".json"));
}

bool exportSet(const QString &file, const QString &destination, QString *errorOut)
{
    QFile::remove(destination);
    if (QFile::copy(file, destination)) {
        return true;
    }
    if (errorOut) {
        *errorOut = i18n("Could not write %1", destination);
    }
    return false;
}

} // namespace PluginSets
