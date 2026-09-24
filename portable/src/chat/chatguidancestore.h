/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QString>
#include <QStringList>

/** @namespace ChatGuidanceStore
    @brief Storage for assistant guidance documents: skills and loops.

    Skills are reusable "how to work" notes (several may be active per
    project); a loop is a start-to-finish pipeline scenario (at most one
    active per project). The document library is global (app data dir,
    shared by every project); WHICH documents are active is stored per
    project in document properties, so both the chat UI and the MCP
    scripting API always see the same state.

    The library has two layers. Built-in documents ship with the app
    (read-only, `share/wunjo/guidance/`) and give every agent the editing
    craft to start from; the user's own documents live in the writable app
    data dir. Reads prefer the user's copy, writes always land there, so
    editing a built-in document shadows it instead of altering it and an
    app update ships better built-ins without touching the user's work.
    Deleting a built-in hides it (global config) rather than failing.
 */
namespace ChatGuidanceStore {

/** @brief Kind of guidance document. */
enum class Kind { Skill, Loop };

/** @brief Where a document's text currently comes from. */
enum class Origin {
    User, ///< Written by the user; no built-in of that name.
    Builtin, ///< Shipped with the app, untouched.
    BuiltinEdited ///< Shipped with the app, overridden by the user's copy.
};

/** @brief Names (file basenames) of all visible documents of @p kind, sorted. */
QStringList list(Kind kind);
/** @brief Content of document @p name (user copy first, then built-in). */
QString read(Kind kind, const QString &name);
/** @brief Create or overwrite @p name in the user's library. False on bad name/IO. */
bool write(Kind kind, const QString &name, const QString &content);
/** @brief Delete @p name: drops the user's copy, hides a built-in, deselects it. */
bool remove(Kind kind, const QString &name);
/** @brief Where @p name comes from, for the UI to mark built-ins. */
Origin origin(Kind kind, const QString &name);

/** @brief Skill names selected in the current project (may be empty). */
QStringList selectedSkills();
/** @brief Persist the per-project skill selection (document property). */
void setSelectedSkills(const QStringList &names);
/** @brief Loop selected in the current project, or empty. */
QString selectedLoop();
/** @brief Persist the per-project loop selection; empty clears it. */
void setSelectedLoop(const QString &name);

/** @brief Absolute writable library directory for @p kind (created on demand). */
QString directory(Kind kind);

} // namespace ChatGuidanceStore
