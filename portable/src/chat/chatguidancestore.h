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
 */
namespace ChatGuidanceStore {

/** @brief Kind of guidance document. */
enum class Kind { Skill, Loop };

/** @brief Names (file basenames) of all documents of @p kind, sorted. */
QStringList list(Kind kind);
/** @brief Content of document @p name, or empty if missing. */
QString read(Kind kind, const QString &name);
/** @brief Create or overwrite document @p name. Returns false on bad name/IO. */
bool write(Kind kind, const QString &name, const QString &content);
/** @brief Delete document @p name (also deselects it in the open project). */
bool remove(Kind kind, const QString &name);

/** @brief Skill names selected in the current project (may be empty). */
QStringList selectedSkills();
/** @brief Persist the per-project skill selection (document property). */
void setSelectedSkills(const QStringList &names);
/** @brief Loop selected in the current project, or empty. */
QString selectedLoop();
/** @brief Persist the per-project loop selection; empty clears it. */
void setSelectedLoop(const QString &name);

/** @brief Absolute library directory for @p kind (created on demand). */
QString directory(Kind kind);

} // namespace ChatGuidanceStore
