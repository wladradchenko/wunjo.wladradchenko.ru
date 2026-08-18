/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "chatguidancestore.h"

#include "core.h"
#include "doc/wunjodoc.h"

#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QStandardPaths>

namespace {

const char SKILLS_PROPERTY[] = "wunjo.ai.skills"; // JSON array of names
const char LOOP_PROPERTY[] = "wunjo.ai.loop";     // single name

QString subdir(ChatGuidanceStore::Kind kind)
{
    return kind == ChatGuidanceStore::Kind::Skill ? QStringLiteral("skills") : QStringLiteral("loops");
}

/** Document names become file names — forbid separators and hidden files. */
bool isValidName(const QString &name)
{
    return !name.isEmpty() && !name.startsWith(QLatin1Char('.')) && !name.contains(QLatin1Char('/')) && !name.contains(QLatin1Char('\\'));
}

QString filePath(ChatGuidanceStore::Kind kind, const QString &name)
{
    return ChatGuidanceStore::directory(kind) + QLatin1Char('/') + name + QStringLiteral(".md");
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
    QDir dir(directory(kind));
    QStringList names;
    const QStringList entries = dir.entryList({QStringLiteral("*.md")}, QDir::Files, QDir::Name);
    for (const QString &entry : entries) {
        names << entry.chopped(3); // strip ".md"
    }
    return names;
}

QString read(Kind kind, const QString &name)
{
    if (!isValidName(name)) {
        return QString();
    }
    QFile file(filePath(kind, name));
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
    QFile file(filePath(kind, name));
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) {
        return false;
    }
    return file.write(content.toUtf8()) >= 0;
}

bool remove(Kind kind, const QString &name)
{
    if (!isValidName(name) || !QFile::remove(filePath(kind, name))) {
        return false;
    }
    // Keep the project selection consistent with the library
    if (kind == Kind::Skill) {
        QStringList selected = selectedSkills();
        if (selected.removeAll(name) > 0) {
            setSelectedSkills(selected);
        }
    } else if (selectedLoop() == name) {
        setSelectedLoop(QString());
    }
    return true;
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
