/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QMap>
#include <QString>
#include <QStringList>
#include <QVector>

/** @namespace PluginSets
    @brief The parameter sets a plugin recorded by analysing a media file.

    One set is one analysis pass — a value per frame for every parameter the
    plugin measured (a head turning, a mouth opening…). Sets live next to the
    project, in `<projectDataFolder>/plugin-sets/<pluginId>/<name>.json`, so they
    travel with it like masks do; @ref importSet and @ref exportSet move one
    between projects instead of analysing the same video twice.

    An effect points at a set through a `urllist` parameter with
    `paramlist="%pluginSets"`: the list offers what was recorded, the effect
    stores the chosen file, and the plugin reads it frame by frame when it
    renders. A set is never folded into the effect's keyframes — those stay the
    user's own, for motion added on top of the recording.
 */
namespace PluginSets {

struct Set
{
    QString name;       ///< what the list shows
    QString kind;       ///< which sort of preset this is, when a plugin has several
    QString file;       ///< absolute path of the json
    QString source;     ///< the media it was recorded from
    QString sourceHash; ///< identity of that media, and of this set
    double fps{0};
    int count{0}; ///< recorded frames
    bool isValid() const { return !file.isEmpty(); }
};

/** @brief Identity of a media file — the same hash the editor gives its clips.
 *  A set is named after it, so analysing one performance twice updates one set
 *  instead of piling up copies, and renaming or moving the file changes
 *  nothing. */
QString hashOf(const QString &path);

/** @brief Folder holding @p pluginId's sets for the current project, created on
 *  demand. Empty when there is no project to store them next to. */
QString folder(const QString &pluginId);

/** @brief Every set recorded for @p pluginId in this project, by name. When
 *  @p kind is given, only presets of that sort — a plugin that does three
 *  different things keeps three kinds side by side. */
QVector<Set> sets(const QString &pluginId, const QString &kind = QString());

/** @brief Read one set file (values are not loaded, only its description). */
Set read(const QString &file);

/** @brief Take the json an analysis produced and keep it as a set, filed under
 *  the hash of the media it was recorded from. @p name is only what the list
 *  shows. Returns an invalid set on failure. */
Set store(const QString &pluginId, const QString &name, const QString &kind, const QString &resultFile, QString *errorOut = nullptr);

bool remove(const QString &file);
/** @brief Copy a set json from anywhere into this project. Returns its new path. */
QString importSet(const QString &pluginId, const QString &file, QString *errorOut = nullptr);
bool exportSet(const QString &file, const QString &destination, QString *errorOut = nullptr);

} // namespace PluginSets
