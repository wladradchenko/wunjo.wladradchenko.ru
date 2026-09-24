/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QList>
#include <QString>

/** @namespace ChatBrain
    @brief Who is answering in the Chat panel.

    The editor is driven the same way whoever is at the other end: through the
    MCP server, with the same tools and the same running commentary in the chat.
    What this picks is only the driver — an agent the user runs themselves in a
    terminal, or an assistant plugin that runs one here without them seeing it.

    A plugin declaring `target: agent` is offered as a way of talking, which is
    also how a third way (a paid API behind a key) will arrive: as another such
    plugin, not as another mechanism.
 */
namespace ChatBrain {

/** @brief The mode where the user drives from Claude Code, Cursor or the like
 *  and the chat only mirrors what happens. */
inline const QString External() { return QStringLiteral("external"); }

/** @brief One way of talking, as the picker shows it. */
struct Option {
    QString id;          ///< @ref External, or the id of an assistant plugin
    QString name;
    QString description;
    /** @brief False while it still needs setting up (environment, weights).
     *  Picking it is still allowed — that is what opens the page to set it up. */
    bool ready = false;
};

/** @brief Every way of talking available on this machine, external first. */
QList<Option> options();

/** @brief True once the MCP server can actually be started: it is installed and
 *  its own environment (`venv-mcp`) is built. Nothing here depends on a model
 *  plugin — an agent in a terminal is a complete way of working on its own. */
bool serverReady();

/** @brief The chosen mode, or empty when the user has not chosen yet — which
 *  is what makes the chat show the picker instead of an input field. */
QString current();
void setCurrent(const QString &mode);

/** @brief Where the folder for an outside agent belongs for the open project:
 *  next to the project file, or in the app's data folder while it is unsaved. */
QString agentFolder();

/** @brief Write that folder: the MCP registration, the instructions an agent
 *  reads, and a note for the human. Returns the folder, or empty with a reason
 *  in @p errorOut. Safe to call again — it overwrites its own files and leaves
 *  anything else in there alone. */
QString prepareAgentFolder(QString *errorOut = nullptr);

/** @brief The line to paste in a terminal to start working there. */
QString launchCommand();

} // namespace ChatBrain
