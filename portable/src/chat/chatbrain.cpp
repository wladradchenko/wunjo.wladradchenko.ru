/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "chatbrain.h"

#include "core.h"
#include "doc/wunjodoc.h"
#include "mcppythonenv.h"
#include "plugins/pluginmanager.h"
#include "wunjosettings.h"

#include <KLocalizedString>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QStandardPaths>

namespace {

const char FOLDER_NAME[] = "agent";

/** @brief True while a plugin still has to build its environment.
 *
 * Asked of the folder rather than of the interpreter: PluginManager falls back
 * to the system Python so a plugin without dependencies still runs, and that
 * fallback would read as "ready" for one that does have them.
 */
bool environmentBuilt(const PluginManifest &manifest)
{
    const QString venv = manifest.venvName();
    if (venv.isEmpty()) {
        return true; // nothing to build
    }
    return !PluginManager::venvPython(venv).isEmpty();
}

/** @brief True once the MCP server has somewhere to run.
 *
 * The server used to be launched with the interpreter of whichever plugin
 * declared `target: agent`. That made driving the editor from Claude Code
 * depend on having installed a local model the outside agent never touches,
 * and "whichever plugin" became meaningless as soon as there could be several
 * of them. It has its own environment now, holding the one package it imports.
 */
bool serverInstalled()
{
    return !PluginManager::mcpServerDir().isEmpty() && !McpPythonEnv::python().isEmpty();
}

/** @brief Running inside the flatpak sandbox, where an agent on the host cannot
 *  use our interpreter and has to come in through `flatpak run` instead. */
bool sandboxed()
{
    return qEnvironmentVariable("PACKAGE_TYPE") == QLatin1String("flatpak");
}

bool writeFile(const QString &path, const QString &content, QString *errorOut)
{
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text) || file.write(content.toUtf8()) < 0) {
        if (errorOut) {
            *errorOut = i18n("Could not write %1.", QFileInfo(path).fileName());
        }
        return false;
    }
    return true;
}

/** @brief How an outside agent should start the MCP server.
 *
 * Inside the sandbox this re-enters it. That looks like a detour and is not:
 * the environment the server needs is built by the app, in the sandbox, against
 * the runtime's interpreter — a path that does not exist on the host at all, so
 * naming it directly in .mcp.json would fail on the first call.
 */
QJsonObject serverInvocation()
{
    QJsonObject server;
    const QString launcher = QStandardPaths::locate(QStandardPaths::AppDataLocation, QStringLiteral("mcp/start"));
    QJsonArray arguments;
    if (sandboxed() && !launcher.isEmpty()) {
        arguments.append(QStringLiteral("run"));
        arguments.append(QString(QStringLiteral("--command=") + launcher));
        arguments.append(QStringLiteral("online.wunjo.make"));
        server.insert(QStringLiteral("command"), QStringLiteral("flatpak"));
        server.insert(QStringLiteral("args"), arguments);
        return server;
    }
    const QString python = McpPythonEnv::python();
    arguments.append(QString(PluginManager::mcpServerDir() + QStringLiteral("/run.py")));
    server.insert(QStringLiteral("command"), python.isEmpty() ? QStringLiteral("python3") : python);
    server.insert(QStringLiteral("args"), arguments);
    return server;
}

/** @brief The briefing an agent reads when it opens the folder.
 *
 * Deliberately about this editor and not about this codebase: the agent is
 * being handed a running application to drive, and everything it needs to know
 * is which tools exist, what to read first, and what the user will see.
 */
QString instructionsDocument()
{
    return QStringLiteral(R"(# Driving Wunjo Make

This folder connects you to a **running Wunjo Make** — a video editor — through
the `wunjo-make` MCP server. The user has it open on screen right now, and they
will watch what you do to their project.

## Start here

1. `get_project_info` and `get_timeline_summary` — see what is actually open.
2. `get_selected_skills` and `get_selected_loop` — the user's own standing
   instructions for this project, written by them in the app's Chat panel.
   A *skill* is how they want you to work; a *loop* is a pipeline they want
   followed from source material to finished video. If either has content it
   outranks your own judgement. Do this once per session.

## Tell them what you are doing

The user is looking at the app, not at your terminal. Mirror the conversation
into its Chat panel, in **their language**:

- `chat_user` — echo their request when you start on it
- `chat_thinking(True/False)` — while you work things out
- `chat_assistant` / `chat_assistant_stream` — your reply
- `chat_tool_start` / `chat_tool_progress` / `chat_tool_end` — a progress card
  around anything slow: a plugin, speech recognition, a render

None of this changes the project; it is what makes you present in the app
instead of a black box.

## Working on the project

- Frames go in, timecodes come out.
- Material has to be in the media pool before it can go on a track:
  `import_media` with an absolute path, then place it.
- `build_timeline` assembles a whole sequence in one call — prefer it to a
  dozen inserts when starting from scratch.
- After an assembly or a replacement, call `render_frame` and **look** at the
  result before reporting success. State is textual *and* visual here.
- `undo` exists. Use it when you get something wrong, and say that you did.
- The editor's own AI plugins are tools too: `list_plugins`, then
  `plugin_status` before `run_plugin`. If one is not set up, tell the user what
  is missing (an API key, an environment, a model) in their language.

## Ground rules

- The project belongs to the user. Do not open or replace it without being
  asked, and save only when they want it saved.
- Prefer the composite tools; they carry a whole workflow and fail more clearly.
- When unsure what state the project is in, call `get_timeline_summary` again
  rather than assuming.
)");
}

QString readmeDocument(const QString &folder)
{
    return i18n(R"(# Drive Wunjo Make from your own agent

This folder was written by Wunjo Make. It contains everything an agent needs to
control the editor that is running right now.

## Three steps

1. Open a terminal in this folder:

       cd "%1"

2. Start your agent there:

       claude

   Cursor, Codex and anything else that reads `.mcp.json` and `AGENTS.md` work
   the same way — open this folder as the working directory.

3. Ask for what you want: "cut this on the beats", "put the takes in order and
   cross-dissolve them", "burn in subtitles".

The agent's replies also appear in the app's Chat panel, so you can keep your
eyes on the timeline.

## Files here

- `.mcp.json` — registers the `wunjo-make` tool server
- `CLAUDE.md`, `AGENTS.md` — how to drive this editor (read by the agent)

Keep Wunjo Make open: the tools talk to the live application, not to files.
)",
                folder);
}

} // namespace

namespace ChatBrain {

QList<Option> options()
{
    QList<Option> list;
    Option external;
    external.id = External();
    external.name = i18n("External MCP");
    external.description = i18n("Keep your own agent. Wunjo hands it the keys to the editor in a folder you open in a terminal; "
                                "this panel then shows what it does.");
    // Only the tool server has to be there. This way of working is complete
    // without a model on this machine — that is the point of it.
    external.ready = serverInstalled();
    list << external;

    const QList<PluginManifest> agents = PluginManager::instance().pluginsForTarget(QStringLiteral("agent"));
    for (const PluginManifest &manifest : agents) {
        Option option;
        option.id = manifest.id();
        option.name = manifest.name();
        option.description = manifest.description();
        option.ready = environmentBuilt(manifest) && PluginManager::instance().runBlocker(manifest.id()).isEmpty();
        list << option;
    }
    return list;
}

bool serverReady()
{
    return serverInstalled();
}

QString current()
{
    return WunjoSettings::chatBrain();
}

void setCurrent(const QString &mode)
{
    WunjoSettings::setChatBrain(mode);
    WunjoSettings::self()->save();
}

QString agentFolder()
{
    // Next to the project, so the folder travels with the work and a second
    // project does not inherit the first one's conversation.
    if (auto *doc = pCore ? pCore->currentDoc() : nullptr) {
        const QString url = doc->url().toLocalFile();
        if (!url.isEmpty()) {
            return QFileInfo(url).absolutePath() + QLatin1Char('/') + QLatin1String(FOLDER_NAME);
        }
    }
    return QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + QLatin1Char('/') + QLatin1String(FOLDER_NAME);
}

QString prepareAgentFolder(QString *errorOut)
{
    const QString folder = agentFolder();
    if (!QDir().mkpath(folder)) {
        if (errorOut) {
            *errorOut = i18n("Could not create %1.", folder);
        }
        return {};
    }
    if (PluginManager::mcpServerDir().isEmpty()) {
        if (errorOut) {
            *errorOut = i18n("This build does not ship the MCP server an outside agent needs.");
        }
        return {};
    }

    QJsonObject servers;
    servers.insert(QStringLiteral("wunjo-make"), serverInvocation());
    QJsonObject root;
    root.insert(QStringLiteral("mcpServers"), servers);
    const QString registration = QString::fromUtf8(QJsonDocument(root).toJson(QJsonDocument::Indented));

    const QString instructions = instructionsDocument();
    // Two names for one document: agents disagree about which to read, and a
    // user who switches from one to the other should not have to notice.
    if (!writeFile(folder + QStringLiteral("/.mcp.json"), registration, errorOut) ||
        !writeFile(folder + QStringLiteral("/CLAUDE.md"), instructions, errorOut) ||
        !writeFile(folder + QStringLiteral("/AGENTS.md"), instructions, errorOut) ||
        !writeFile(folder + QStringLiteral("/README.md"), readmeDocument(folder), errorOut)) {
        return {};
    }
    return folder;
}

QString launchCommand()
{
    return QStringLiteral("cd \"%1\" && claude").arg(agentFolder());
}

} // namespace ChatBrain
