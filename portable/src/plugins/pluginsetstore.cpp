/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "pluginsetstore.h"

#include "bin/projectclip.h"
#include "core.h"
#include "doc/kthumb.h"
#include "doc/wunjodoc.h"

#include <KLocalizedString>

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QImage>
#include <QImageReader>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <QSet>

#include <mlt++/Mlt.h>

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

/** @brief The file that goes with a set's json: same name, another suffix. */
QString sidecar(const QString &jsonFile, const QString &suffix)
{
    const QFileInfo info(jsonFile);
    return info.dir().absoluteFilePath(info.completeBaseName() + QLatin1Char('.') + suffix);
}

/** @brief Put a picture next to the set as a png, whatever it arrived as. */
void keepPicture(const QString &from, const QString &to)
{
    QFile::remove(to);
    if (from.isEmpty() || !QFileInfo::exists(from)) {
        return;
    }
    const QImage image(from);
    if (!image.isNull()) {
        image.save(to, "PNG");
    }
}

/** @brief Copy a moving preview next to the set, as it is. */
void keepPreview(const QString &from, const QString &to)
{
    QFile::remove(to);
    if (!from.isEmpty() && QFileInfo::exists(from)) {
        QFile::copy(from, to);
    }
}

/** @brief Move the pictures of a set along with its json — for import, export. */
void carryPictures(const QString &fromJson, const QString &toJson)
{
    for (const char *suffix : {"png", "gif"}) {
        const QString from = sidecar(fromJson, QLatin1String(suffix));
        const QString to = sidecar(toJson, QLatin1String(suffix));
        QFile::remove(to);
        if (QFileInfo::exists(from)) {
            QFile::copy(from, to);
        }
    }
}

bool looksLikeImage(const QString &path)
{
    static const QStringList suffixes = {QStringLiteral("png"), QStringLiteral("jpg"), QStringLiteral("jpeg"), QStringLiteral("webp"),
                                         QStringLiteral("bmp"), QStringLiteral("tif"), QStringLiteral("tiff"), QStringLiteral("gif")};
    return suffixes.contains(QFileInfo(path).suffix().toLower()) || !QImageReader::imageFormat(path).isEmpty();
}

/** @brief The first frame of a video, through the same producer the bin uses. */
QImage firstFrame(const QString &path, int size)
{
    Mlt::Profile profile(pCore->getCurrentProfilePath().toUtf8().constData());
    Mlt::Producer producer(profile, path.toUtf8().constData());
    if (!producer.is_valid()) {
        return {};
    }
    const int width = size;
    const int height = qMax(1, int(size * double(profile.height()) / qMax(1, profile.width())));
    return KThumb::getFrame(producer, 0, width, height);
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
    // the pictures are optional and live next to the json under its own name
    const QString thumb = sidecar(file, QStringLiteral("png"));
    if (QFileInfo::exists(thumb)) {
        set.thumb = thumb;
    }
    const QString preview = sidecar(file, QStringLiteral("gif"));
    if (QFileInfo::exists(preview)) {
        set.preview = preview;
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

namespace {

/** @brief @p wanted, or "wanted (2)", "wanted (3)"… when another set of this
 *  kind already goes by it. Two photos both called portrait.jpg would
 *  otherwise be one word twice in the list. The set being written (@p key)
 *  is not counted: analysing the same file again updates it in place and
 *  must keep its name. */
QString uniqueName(const QString &pluginId, const QString &kind, const QString &wanted, const QString &key)
{
    QSet<QString> taken;
    const QVector<Set> others = sets(pluginId, kind);
    for (const Set &other : others) {
        if (other.sourceHash != key) {
            taken.insert(other.name.toLower());
        }
    }
    QString name = wanted;
    for (int number = 2; taken.contains(name.toLower()); ++number) {
        name = QStringLiteral("%1 (%2)").arg(wanted).arg(number);
    }
    return name;
}

} // namespace

Set store(const QString &pluginId, const QString &name, const QString &kind, const QJsonArray &outputs, QString *errorOut)
{
    const QString path = folder(pluginId);
    if (path.isEmpty()) {
        if (errorOut) {
            *errorOut = i18n("Save the project first — sets are stored next to it.");
        }
        return {};
    }
    // What the plugin handed back: the set itself, and the pictures of it. The
    // json is the output that says so, or failing that the first one — the
    // plugins written before pictures existed sent the json alone.
    QString resultFile;
    QString picture;
    QString moving;
    for (const QJsonValue &value : outputs) {
        const QJsonObject output = value.toObject();
        const QString type = output.value(QStringLiteral("type")).toString();
        const QString file = output.value(QStringLiteral("path")).toString();
        if (file.isEmpty()) {
            continue;
        }
        if (type == QLatin1String("image") && picture.isEmpty()) {
            picture = file;
        } else if (type == QLatin1String("animation") && moving.isEmpty()) {
            moving = file;
        } else if (resultFile.isEmpty() && (type == QLatin1String("data") || type.isEmpty() || file.endsWith(QLatin1String(".json")))) {
            resultFile = file;
        }
    }
    if (resultFile.isEmpty() && !outputs.isEmpty()) {
        resultFile = outputs.first().toObject().value(QStringLiteral("path")).toString();
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
    // The plugin may say what it produced; otherwise it is what was asked for.
    if (!kind.isEmpty() && root.value(QStringLiteral("kind")).toString().isEmpty()) {
        root.insert(QStringLiteral("kind"), kind);
    }
    root.insert(QStringLiteral("name"), uniqueName(pluginId, root.value(QStringLiteral("kind")).toString(), displayName(name), key));
    root.insert(QStringLiteral("source_hash"), key);
    if (!writeSet(dir, key, root, errorOut)) {
        return {};
    }
    const QString file = dir.absoluteFilePath(key + QStringLiteral(".json"));
    keepPicture(picture, sidecar(file, QStringLiteral("png")));
    keepPreview(moving, sidecar(file, QStringLiteral("gif")));
    return read(file);
}

bool rename(const QString &file, const QString &name, QString *errorOut)
{
    QJsonObject root = readJson(file);
    if (root.isEmpty()) {
        if (errorOut) {
            *errorOut = i18n("%1 is not a set file.", QFileInfo(file).fileName());
        }
        return false;
    }
    const QFileInfo info(file);
    const QString kind = root.value(QStringLiteral("kind")).toString();
    const QString key = root.value(QStringLiteral("source_hash")).toString(info.completeBaseName());
    root.insert(QStringLiteral("name"), uniqueName(root.value(QStringLiteral("plugin")).toString(), kind, displayName(name), key));
    return writeSet(info.dir(), info.completeBaseName(), root, errorOut);
}

QString makeThumbnail(const Set &set, int size)
{
    if (!set.thumb.isEmpty() && QFileInfo::exists(set.thumb)) {
        return set.thumb;
    }
    if (!set.isValid() || set.source.isEmpty() || !QFileInfo::exists(set.source)) {
        return {};
    }
    QImage image;
    if (looksLikeImage(set.source)) {
        image = QImage(set.source);
        if (!image.isNull()) {
            image = image.scaled(size, size, Qt::KeepAspectRatio, Qt::SmoothTransformation);
        }
    } else {
        image = firstFrame(set.source, size);
    }
    if (image.isNull()) {
        return {};
    }
    const QString thumb = sidecar(set.file, QStringLiteral("png"));
    return image.save(thumb, "PNG") ? thumb : QString();
}

bool remove(const QString &file)
{
    QFile::remove(sidecar(file, QStringLiteral("png")));
    QFile::remove(sidecar(file, QStringLiteral("gif")));
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
    const QString kind = copy.value(QStringLiteral("kind")).toString();
    copy.insert(QStringLiteral("name"),
                uniqueName(pluginId, kind, displayName(copy.value(QStringLiteral("name")).toString(QFileInfo(file).completeBaseName())), key));
    copy.insert(QStringLiteral("source_hash"), key);
    if (!writeSet(dir, key, copy, errorOut)) {
        return {};
    }
    const QString destination = dir.absoluteFilePath(key + QStringLiteral(".json"));
    carryPictures(file, destination);
    return destination;
}

bool exportSet(const QString &file, const QString &destination, QString *errorOut)
{
    QFile::remove(destination);
    if (QFile::copy(file, destination)) {
        carryPictures(file, destination);
        return true;
    }
    if (errorOut) {
        *errorOut = i18n("Could not write %1", destination);
    }
    return false;
}

} // namespace PluginSets
