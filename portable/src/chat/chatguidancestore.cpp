/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "chatguidancestore.h"

#include "core.h"
#include "doc/wunjodoc.h"

#include <KConfigGroup>
#include <KSharedConfig>

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QStandardPaths>

namespace {

const char SKILLS_PROPERTY[] = "wunjo.ai.skills"; // JSON array of names
const char LOOP_PROPERTY[] = "wunjo.ai.loop";     // single name
const char CONFIG_GROUP[] = "Chat Guidance";      // hidden built-ins live here

QString subdir(ChatGuidanceStore::Kind kind)
{
    return kind == ChatGuidanceStore::Kind::Skill ? QStringLiteral("skills") : QStringLiteral("loops");
}

/** Document names become file names — forbid separators and hidden files. */
bool isValidName(const QString &name)
{
    return !name.isEmpty() && !name.startsWith(QLatin1Char('.')) && !name.contains(QLatin1Char('/')) && !name.contains(QLatin1Char('\\'));
}

QString userFilePath(ChatGuidanceStore::Kind kind, const QString &name)
{
    return ChatGuidanceStore::directory(kind) + QLatin1Char('/') + name + QStringLiteral(".md");
}

/** The documents shipped with the app. AppDataLocation is share/wunjo on Linux
    and the bundle's Resources on macOS — the same lookup the bundled face model
    uses. Empty when running uninstalled, which simply means no built-ins. */
QString builtinDirectory(ChatGuidanceStore::Kind kind)
{
    return QStandardPaths::locate(QStandardPaths::AppDataLocation, QStringLiteral("guidance/") + subdir(kind), QStandardPaths::LocateDirectory);
}

/** Path of built-in document @p name, or empty when the app ships no such document. */
QString builtinFilePath(ChatGuidanceStore::Kind kind, const QString &name)
{
    const QString dir = builtinDirectory(kind);
    if (dir.isEmpty()) {
        return QString();
    }
    const QString path = dir + QLatin1Char('/') + name + QStringLiteral(".md");
    return QFile::exists(path) ? path : QString();
}

QString hiddenKey(ChatGuidanceStore::Kind kind)
{
    return kind == ChatGuidanceStore::Kind::Skill ? QStringLiteral("hiddenSkills") : QStringLiteral("hiddenLoops");
}

/** Built-ins the user deleted. Global, like the library itself. */
QStringList hiddenBuiltins(ChatGuidanceStore::Kind kind)
{
    KConfigGroup group(KSharedConfig::openConfig(), QLatin1String(CONFIG_GROUP));
    return group.readEntry(hiddenKey(kind), QStringList());
}

void setHiddenBuiltins(ChatGuidanceStore::Kind kind, const QStringList &names)
{
    KConfigGroup group(KSharedConfig::openConfig(), QLatin1String(CONFIG_GROUP));
    group.writeEntry(hiddenKey(kind), names);
    group.sync();
}

QStringList documentNamesIn(const QString &dir)
{
    if (dir.isEmpty()) {
        return {};
    }
    QStringList names;
    const QStringList entries = QDir(dir).entryList({QStringLiteral("*.md")}, QDir::Files, QDir::Name);
    for (const QString &entry : entries) {
        names << entry.chopped(3); // strip ".md"
    }
    return names;
}

} // namespace

namespace ChatGuidanceStore {

QString directory(Kind kind)
{
    const QString dir = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + QLatin1Char('/') + subdir(kind);
    QDir().mkpath(dir);
    return dir;
}

QStringList list(Kind kind)
{
    QStringList names = documentNamesIn(directory(kind));
    const QStringList suppressed = hiddenBuiltins(kind);
    const QStringList builtins = documentNamesIn(builtinDirectory(kind));
    for (const QString &name : builtins) {
        // A user document of the same name shadows the built-in, so it is
        // already listed; a deleted built-in stays gone.
        if (!names.contains(name) && !suppressed.contains(name)) {
            names << name;
        }
    }
    names.sort(Qt::CaseInsensitive);
    return names;
}

QString read(Kind kind, const QString &name)
{
    if (!isValidName(name)) {
        return QString();
    }
    QString path = userFilePath(kind, name);
    if (!QFile::exists(path)) {
        path = builtinFilePath(kind, name);
    }
    if (path.isEmpty()) {
        return QString();
    }
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return QString();
    }
    return QString::fromUtf8(file.readAll());
}

bool write(Kind kind, const QString &name, const QString &content)
{
    if (!isValidName(name)) {
        return false;
    }
    QFile file(userFilePath(kind, name));
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        return false;
    }
    if (file.write(content.toUtf8()) < 0) {
        return false;
    }
    // Writing a document the user had deleted brings it back into the library.
    QStringList suppressed = hiddenBuiltins(kind);
    if (suppressed.removeAll(name) > 0) {
        setHiddenBuiltins(kind, suppressed);
    }
    return true;
}

bool remove(Kind kind, const QString &name)
{
    if (!isValidName(name)) {
        return false;
    }
    // Read the selection before the document goes: selectedSkills() and
    // selectedLoop() filter themselves against the library, so afterwards they
    // no longer mention @p name and the stale entry would stay in the project
    // file for good.
    QStringList selected = kind == Kind::Skill ? selectedSkills() : QStringList();
    const bool wasSelectedLoop = kind == Kind::Loop && selectedLoop() == name;

    const QString userPath = userFilePath(kind, name);
    const bool hadUserCopy = QFile::exists(userPath);
    if (hadUserCopy && !QFile::remove(userPath)) {
        return false;
    }
    if (!builtinFilePath(kind, name).isEmpty()) {
        // Built-ins are read-only; remember the deletion instead of failing,
        // otherwise deleting one would appear to work and come back next launch.
        QStringList suppressed = hiddenBuiltins(kind);
        if (!suppressed.contains(name)) {
            suppressed << name;
            setHiddenBuiltins(kind, suppressed);
        }
    } else if (!hadUserCopy) {
        return false; // no such document
    }
    // Keep the project selection consistent with the library
    if (kind == Kind::Skill) {
        if (selected.removeAll(name) > 0) {
            setSelectedSkills(selected);
        }
    } else if (wasSelectedLoop) {
        setSelectedLoop(QString());
    }
    return true;
}

Origin origin(Kind kind, const QString &name)
{
    if (!isValidName(name) || builtinFilePath(kind, name).isEmpty()) {
        return Origin::User;
    }
    return QFile::exists(userFilePath(kind, name)) ? Origin::BuiltinEdited : Origin::Builtin;
}

QStringList selectedSkills()
{
    auto doc = pCore ? pCore->currentDoc() : nullptr;
    if (!doc) {
        return {};
    }
    const QJsonArray array = QJsonDocument::fromJson(doc->getDocumentProperty(QLatin1String(SKILLS_PROPERTY)).toUtf8()).array();
    QStringList names;
    const QStringList library = list(Kind::Skill);
    for (const auto &value : array) {
        const QString name = value.toString();
        if (library.contains(name)) { // silently drop deleted documents
            names << name;
        }
    }
    return names;
}

void setSelectedSkills(const QStringList &names)
{
    auto doc = pCore ? pCore->currentDoc() : nullptr;
    if (!doc) {
        return;
    }
    QJsonArray array;
    for (const QString &name : names) {
        array.append(name);
    }
    doc->setDocumentProperty(QLatin1String(SKILLS_PROPERTY), QString::fromUtf8(QJsonDocument(array).toJson(QJsonDocument::Compact)));
}

QString selectedLoop()
{
    auto doc = pCore ? pCore->currentDoc() : nullptr;
    if (!doc) {
        return QString();
    }
    const QString name = doc->getDocumentProperty(QLatin1String(LOOP_PROPERTY));
    return list(Kind::Loop).contains(name) ? name : QString();
}

void setSelectedLoop(const QString &name)
{
    auto doc = pCore ? pCore->currentDoc() : nullptr;
    if (!doc) {
        return;
    }
    doc->setDocumentProperty(QLatin1String(LOOP_PROPERTY), name);
}

} // namespace ChatGuidanceStore
