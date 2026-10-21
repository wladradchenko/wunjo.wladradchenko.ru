/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QJsonArray>
#include <QJsonObject>
#include <QObject>
#include <QPointer>
#include <QProcess>
#include <QProcessEnvironment>
#include <QString>
#include <QStringList>

#include <functional>

class FileDownloader;

/** @brief The agents the chat can talk to over the Agent Client Protocol.

    The list is the public ACP registry (agentclientprotocol.com/registry) —
    Claude, Codex, Gemini, Copilot, Grok and the rest — cached in the app's data
    folder. An agent is installed the first time it is picked, into
    `<AppLocalData>/acp/<id>/<version>`, from the distribution its entry names:
    an npm package run on the Node.js that ships with the editor, or an archive
    with its checksum.

    Two adapters wrap a program the user may already have: claude-agent-acp
    runs `claude` and codex-acp runs `codex`. Both come with the Claude Code and
    ChatGPT extensions of VS Code and Cursor, and with the CLIs. When one is
    found the adapter is installed without its own copy (a few dozen megabytes
    instead of several hundred) and pointed at the one already there — which
    also brings the user's existing sign-in with it.
 */
namespace AcpAgents {

/** @brief The Node.js shipped with the editor (share/wunjo/node), or one on
 *  PATH when running from a build tree; empty when there is none. */
QString nodeExecutable();

/** @brief Every agent of the cached registry, as the registry writes them. */
QJsonArray registry();
/** @brief One agent of the cached registry, empty when unknown. */
QJsonObject agent(const QString &id);
/** @brief Fetch the registry again; @p done runs on the GUI thread either way. */
void refreshRegistry(QObject *context, const std::function<void(bool ok)> &done);

/** @brief Installed in the version the registry names now. */
bool isInstalled(const QString &id);

/** @brief A program of the user's this agent can run instead of its own copy
 *  (`claude`, `codex`), or empty. */
QString existingBinary(const QString &id);

/** @brief How to start an installed agent. */
struct Launch {
    QString program;
    QStringList arguments;
    QProcessEnvironment environment;
    QString error; ///< why it cannot start, when program is empty
};
Launch launchFor(const QString &id);

/** @brief One installation, reported as it goes. */
class Installer : public QObject
{
    Q_OBJECT
public:
    explicit Installer(const QString &id, QObject *parent = nullptr);
    void start();

Q_SIGNALS:
    /** @brief @p percent -1 when unknown; @p line what is happening. */
    void progress(int percent, const QString &line);
    void finished(bool ok, const QString &error);

private:
    void installPackage(const QJsonObject &npx);
    void installArchive(const QJsonObject &binary);
    void finish(bool ok, const QString &error);

    QString m_id;
    QJsonObject m_agent;
    QString m_dir;
    QPointer<QProcess> m_process;
    QPointer<FileDownloader> m_download;
};

} // namespace AcpAgents
