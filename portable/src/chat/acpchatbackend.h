/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "chatbackend.h"

#include <QHash>
#include <QJsonObject>
#include <QJsonValue>
#include <QPointer>
#include <QProcess>
#include <QStringList>

#include <functional>

namespace AcpAgents {
class Installer;
}

/** @class AcpChatBackend
    @brief The chat, answered by an agent over the Agent Client Protocol —
    Claude, Codex, Gemini or any other agent of the ACP registry.

    The agent runs as a child process speaking JSON-RPC over its stdin and
    stdout; the editor is the client. Each session is created with the editor's
    own MCP server (wunjo-make) attached, so the agent drives the timeline with
    the same tools an agent in a terminal has. Unlike the assistant plugin,
    everything comes back through this object: the reply as it streams, a card
    per tool call, and the questions only the user can answer — run this
    command? sign in how? — as buttons in the chat.

    The agent is installed the first time a message is sent to it (a card shows
    the download), and a conversation picks up the agent's own session again
    when the agent can load one.
 */
class AcpChatBackend : public AbstractChatBackend
{
    Q_OBJECT
public:
    explicit AcpChatBackend(const QString &agentId, QObject *parent = nullptr);
    ~AcpChatBackend() override;

    void sendMessage(const QString &text) override;
    void cancel() override;
    void setSession(const QString &sessionId) override;
    void answerQuestion(const QString &questionId, const QString &optionId) override;
    void release() override;
    QJsonArray modes() const override;
    QString currentMode() const override;
    void setMode(const QString &id) override;

private:
    using Reply = std::function<void(const QJsonObject &result, const QJsonObject &error)>;

    // getting to a session the prompt can go to
    void ensureSession(const std::function<void()> &then);
    void install(const std::function<void()> &then);
    void launch(const std::function<void()> &then);
    void openSession(const std::function<void()> &then);
    void askToSignIn(const std::function<void()> &then);
    /** @brief Sign in by running the agent's own login command (an auth method
     *  with "terminal-auth"): its output goes into the chat, the first link in
     *  it opens in the browser, and the session is opened once it succeeds. */
    void runLogin(const QString &methodId, const QJsonObject &terminal, const std::function<void()> &then);
    /** @brief What went wrong, from the end of the agent's stderr: the error
     *  line rather than the runtime's version printed under it. */
    QString lastError(const QString &fallback) const;

    // the turn
    void prompt(const QString &text);
    void endTurn();
    void sendPending();

    // JSON-RPC
    void request(const QString &method, const QJsonObject &params, const Reply &reply);
    void notify(const QString &method, const QJsonObject &params);
    void respond(const QJsonValue &id, const QJsonObject &result);
    void respondError(const QJsonValue &id, int code, const QString &message);
    void write(const QJsonObject &message);
    void readOutput();
    void handle(const QJsonObject &message);
    void onUpdate(const QJsonObject &update);
    void onPermission(const QJsonValue &id, const QJsonObject &params);
    void stopProcess();

    /** @brief Take the modes an opened session reports, and switch to the one
     *  the user chose for this agent if it is among them. */
    void takeModes(const QJsonObject &sessionResult);
    /** @brief Per agent, the modes it reported last and the one chosen, so the
     *  menu has them before the first message of a run. */
    void saveModes() const;
    QString storedSession() const;
    void storeSession(const QString &acpSessionId);

    QString m_agentId;
    QString m_agentName;
    QString m_chatSession;
    QPointer<QProcess> m_process;
    QPointer<AcpAgents::Installer> m_installer;
    QPointer<QProcess> m_login;
    QByteArray m_buffer;
    int m_nextId{1};
    QHash<int, Reply> m_replies;

    bool m_initialized{false};
    QJsonObject m_capabilities;
    QJsonArray m_authMethods;
    QString m_session;  ///< the agent's own session id, empty until one is open
    bool m_loading{false}; ///< session/load replays history; it is not news

    bool m_busy{false};
    bool m_stopped{false};
    QString m_stream;   ///< the reply so far, since the last tool call
    QStringList m_pending;
    QStringList m_stderr;

    /** @brief Questions waiting for the user: question id -> what answers it. */
    QHash<QString, std::function<void(const QString &optionId)>> m_questions;
    int m_nextQuestion{1};

    QJsonArray m_modes;
    QString m_currentMode;
    QString m_chosenMode; ///< what the user picked; applied to every new session
};
