/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "acpchatbackend.h"

#include "acpagents.h"
#include "chatbrain.h"

#include <KLocalizedString>

#include <QCoreApplication>
#include <QDesktopServices>
#include <QDir>
#include <QRegularExpression>
#include <QUrl>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QSaveFile>
#include <QStandardPaths>
#include <QTimer>

#include <memory>

#if defined(Q_OS_UNIX)
#include <csignal>
#include <unistd.h>
#endif

namespace {

/** @brief The ACP version this client speaks. */
constexpr int PROTOCOL_VERSION = 1;

QString modesPath()
{
    return QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + QStringLiteral("/acp/modes.json");
}

QJsonObject readJson(const QString &path)
{
    QFile file(path);
    return file.open(QIODevice::ReadOnly) ? QJsonDocument::fromJson(file.readAll()).object() : QJsonObject();
}

void writeJson(const QString &path, const QJsonObject &content)
{
    QDir().mkpath(QFileInfo(path).absolutePath());
    QSaveFile file(path);
    if (file.open(QIODevice::WriteOnly)) {
        file.write(QJsonDocument(content).toJson());
        file.commit();
    }
}

QString sessionsPath()
{
    return QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + QStringLiteral("/acp/sessions.json");
}

QJsonObject readSessions()
{
    QFile file(sessionsPath());
    return file.open(QIODevice::ReadOnly) ? QJsonDocument::fromJson(file.readAll()).object() : QJsonObject();
}

/** @brief The words of a tool call's content, for the card's last line. */
QString contentText(const QJsonArray &content)
{
    for (const QJsonValue &value : content) {
        const QJsonObject inner = value.toObject().value(QStringLiteral("content")).toObject();
        const QString text = inner.value(QStringLiteral("text")).toString().simplified();
        if (!text.isEmpty()) {
            return text.left(200);
        }
    }
    return {};
}

} // namespace

AcpChatBackend::AcpChatBackend(const QString &agentId, QObject *parent)
    : AbstractChatBackend(parent)
    , m_agentId(agentId)
{
    const QJsonObject entry = AcpAgents::agent(agentId);
    m_agentName = entry.value(QStringLiteral("name")).toString(agentId);
    const QJsonObject saved = readJson(modesPath()).value(agentId).toObject();
    m_modes = saved.value(QStringLiteral("modes")).toArray();
    m_chosenMode = saved.value(QStringLiteral("chosen")).toString();
    m_currentMode = m_chosenMode.isEmpty() ? saved.value(QStringLiteral("current")).toString() : m_chosenMode;
}

QJsonArray AcpChatBackend::modes() const
{
    return m_modes;
}

QString AcpChatBackend::currentMode() const
{
    return m_currentMode;
}

void AcpChatBackend::setMode(const QString &id)
{
    m_chosenMode = id;
    m_currentMode = id;
    saveModes();
    if (!m_session.isEmpty() && m_process) {
        request(QStringLiteral("session/set_mode"), QJsonObject{{QStringLiteral("sessionId"), m_session}, {QStringLiteral("modeId"), id}},
                [this](const QJsonObject &, const QJsonObject &error) {
                    if (!error.isEmpty()) {
                        Q_EMIT errorOccurred(i18n("%1 did not switch mode: %2", m_agentName, error.value(QStringLiteral("message")).toString()));
                    }
                });
    }
    // otherwise the next session opens in it
}

void AcpChatBackend::takeModes(const QJsonObject &sessionResult)
{
    const QJsonObject modes = sessionResult.value(QStringLiteral("modes")).toObject();
    if (modes.isEmpty()) {
        return;
    }
    m_modes = modes.value(QStringLiteral("availableModes")).toArray();
    m_currentMode = modes.value(QStringLiteral("currentModeId")).toString();
    bool offered = false;
    for (const QJsonValue &value : std::as_const(m_modes)) {
        offered = offered || value.toObject().value(QStringLiteral("id")).toString() == m_chosenMode;
    }
    if (offered && m_chosenMode != m_currentMode) {
        setMode(m_chosenMode);
    }
    saveModes();
}

void AcpChatBackend::saveModes() const
{
    QJsonObject all = readJson(modesPath());
    all.insert(m_agentId, QJsonObject{{QStringLiteral("modes"), m_modes}, {QStringLiteral("current"), m_currentMode}, {QStringLiteral("chosen"), m_chosenMode}});
    writeJson(modesPath(), all);
}

AcpChatBackend::~AcpChatBackend()
{
    stopProcess();
}

void AcpChatBackend::sendMessage(const QString &text)
{
    // said while the agent is still answering: the next thing it is asked
    if (m_busy) {
        m_pending << text;
        return;
    }
    m_busy = true;
    m_stopped = false;
    Q_EMIT busyChanged(true);
    ensureSession([this, text]() { prompt(text); });
}

void AcpChatBackend::cancel()
{
    m_stopped = true;
    m_pending.clear();
    // a question the user has not answered is answered "no" by stopping
    const QStringList open = m_questions.keys();
    for (const QString &question : open) {
        answerQuestion(question, QString());
    }
    if (m_installer) {
        m_installer->deleteLater();
    }
    if (!m_session.isEmpty() && m_process) {
        notify(QStringLiteral("session/cancel"), QJsonObject{{QStringLiteral("sessionId"), m_session}});
        // The agent ends the turn itself, with stopReason "cancelled". One that
        // does not within a few seconds is stopped, with everything it started.
        QTimer::singleShot(4000, this, [this]() {
            if (m_busy && m_stopped) {
                stopProcess();
                endTurn();
            }
        });
        return;
    }
    stopProcess();
    endTurn();
}

void AcpChatBackend::setSession(const QString &sessionId)
{
    if (sessionId == m_chatSession) {
        return;
    }
    m_chatSession = sessionId;
    // another conversation: the agent's session for it is found again on the
    // next message, or a new one is opened
    m_session.clear();
}

void AcpChatBackend::answerQuestion(const QString &questionId, const QString &optionId)
{
    const auto answer = m_questions.take(questionId);
    if (answer) {
        answer(optionId);
    }
}

void AcpChatBackend::release()
{
    stopProcess();
}

// ── getting to a session ────────────────────────────────────────────────────

void AcpChatBackend::ensureSession(const std::function<void()> &then)
{
    if (!m_session.isEmpty() && m_process && m_process->state() == QProcess::Running) {
        then();
        return;
    }
    if (!AcpAgents::isInstalled(m_agentId)) {
        install([this, then]() { launch(then); });
        return;
    }
    launch(then);
}

void AcpChatBackend::install(const std::function<void()> &then)
{
    const QString card = QStringLiteral("acp-install:") + m_agentId;
    Q_EMIT toolStarted(card, i18n("Installing %1", m_agentName));
    m_installer = new AcpAgents::Installer(m_agentId, this);
    connect(m_installer, &AcpAgents::Installer::progress, this,
            [this, card](int percent, const QString &line) { Q_EMIT toolProgress(card, percent, line); });
    connect(m_installer, &AcpAgents::Installer::finished, this, [this, card, then](bool ok, const QString &error) {
        m_installer->deleteLater();
        Q_EMIT toolFinished(card, !ok, ok ? i18n("%1 is installed.", m_agentName) : error);
        if (!ok) {
            Q_EMIT errorOccurred(error);
            endTurn();
            return;
        }
        then();
    });
    m_installer->start();
}

void AcpChatBackend::launch(const std::function<void()> &then)
{
    if (m_process && m_process->state() == QProcess::Running) {
        openSession(then);
        return;
    }
    AcpAgents::Launch how = AcpAgents::launchFor(m_agentId);
    if (how.program.isEmpty()) {
        // installed without its own copy of the program it wraps, and that
        // program has gone: launchFor() forgot the install, so do it again
        if (!AcpAgents::isInstalled(m_agentId)) {
            install([this, then]() { launch(then); });
            return;
        }
        Q_EMIT errorOccurred(how.error);
        endTurn();
        return;
    }
    QString error;
    const QString folder = ChatBrain::prepareChatAgentFolder(&error);
    if (folder.isEmpty()) {
        Q_EMIT errorOccurred(error);
        endTurn();
        return;
    }

    m_process = new QProcess(this);
    m_process->setProgram(how.program);
    m_process->setArguments(how.arguments);
    m_process->setProcessEnvironment(how.environment);
    m_process->setWorkingDirectory(folder);
#if defined(Q_OS_UNIX)
    // its own process group, so stopping it stops what it started as well
    m_process->setChildProcessModifier([]() { ::setsid(); });
#endif
    connect(m_process, &QProcess::readyReadStandardOutput, this, &AcpChatBackend::readOutput);
    connect(m_process, &QProcess::readyReadStandardError, this, [this]() {
        const QStringList lines = QString::fromUtf8(m_process->readAllStandardError()).split(QLatin1Char('\n'), Qt::SkipEmptyParts);
        m_stderr << lines;
        while (m_stderr.size() > 20) {
            m_stderr.removeFirst();
        }
    });
    connect(m_process, &QProcess::finished, this, [this](int exitCode, QProcess::ExitStatus) {
        const bool expected = m_stopped;
        m_initialized = false;
        m_session.clear();
        m_replies.clear();
        m_questions.clear();
        if (m_process) {
            m_process->deleteLater();
        }
        if (m_busy && !expected) {
            Q_EMIT errorOccurred(i18n("%1 stopped: %2", m_agentName, lastError(i18n("exit code %1", exitCode))));
        }
        if (m_busy) {
            endTurn();
        }
    });
    m_process->start();
    if (!m_process->waitForStarted(10000)) {
        Q_EMIT errorOccurred(i18n("Could not start %1: %2", m_agentName, m_process->errorString()));
        endTurn();
        return;
    }

    const QJsonObject capabilities{
        // the agent reads, writes and runs commands with its own tools
        {QStringLiteral("fs"), QJsonObject{{QStringLiteral("readTextFile"), false}, {QStringLiteral("writeTextFile"), false}}},
        {QStringLiteral("terminal"), false},
    };
    request(QStringLiteral("initialize"),
            QJsonObject{{QStringLiteral("protocolVersion"), PROTOCOL_VERSION}, {QStringLiteral("clientCapabilities"), capabilities},
                        // name and version are both required: Gemini and Cursor refuse one without the other
                        {QStringLiteral("clientInfo"), QJsonObject{{QStringLiteral("name"), QStringLiteral("wunjo-make")},
                                                                   {QStringLiteral("title"), QStringLiteral("Wunjo Make")},
                                                                   {QStringLiteral("version"), QCoreApplication::applicationVersion()}}}},
            [this, then](const QJsonObject &result, const QJsonObject &error) {
                if (!error.isEmpty()) {
                    Q_EMIT errorOccurred(i18n("%1 did not start: %2", m_agentName, error.value(QStringLiteral("message")).toString()));
                    endTurn();
                    return;
                }
                m_initialized = true;
                m_capabilities = result.value(QStringLiteral("agentCapabilities")).toObject();
                m_authMethods = result.value(QStringLiteral("authMethods")).toArray();
                openSession(then);
            });
}

void AcpChatBackend::openSession(const std::function<void()> &then)
{
    QString error;
    const QString folder = ChatBrain::prepareChatAgentFolder(&error);
    const QJsonArray servers{ChatBrain::acpMcpServer()};
    const QString previous = storedSession();
    if (!previous.isEmpty() && m_capabilities.value(QStringLiteral("loadSession")).toBool()) {
        // The agent replays the conversation as updates; the chat already
        // shows it, so they are let past without being drawn again.
        m_loading = true;
        request(QStringLiteral("session/load"),
                QJsonObject{{QStringLiteral("sessionId"), previous}, {QStringLiteral("cwd"), folder}, {QStringLiteral("mcpServers"), servers}},
                [this, previous, then](const QJsonObject &result, const QJsonObject &error) {
                    m_loading = false;
                    if (error.isEmpty()) {
                        m_session = previous;
                        takeModes(result);
                        then();
                        return;
                    }
                    storeSession(QString()); // gone at the agent's end: start again
                    openSession(then);
                });
        return;
    }
    request(QStringLiteral("session/new"), QJsonObject{{QStringLiteral("cwd"), folder}, {QStringLiteral("mcpServers"), servers}},
            [this, then](const QJsonObject &result, const QJsonObject &error) {
                if (!error.isEmpty()) {
                    // Not signed in is what this usually is; the agent said at
                    // the start which ways of signing in it takes.
                    if (!m_authMethods.isEmpty()) {
                        askToSignIn(then);
                        return;
                    }
                    Q_EMIT errorOccurred(i18n("%1: %2", m_agentName, error.value(QStringLiteral("message")).toString()));
                    endTurn();
                    return;
                }
                m_session = result.value(QStringLiteral("sessionId")).toString();
                storeSession(m_session);
                takeModes(result);
                then();
            });
}

void AcpChatBackend::askToSignIn(const std::function<void()> &then)
{
    QJsonArray options;
    for (const QJsonValue &value : std::as_const(m_authMethods)) {
        const QJsonObject method = value.toObject();
        options.append(QJsonObject{{QStringLiteral("id"), method.value(QStringLiteral("id"))}, {QStringLiteral("label"), method.value(QStringLiteral("name"))}});
    }
    const QString question = QStringLiteral("acp-auth-%1").arg(m_nextQuestion++);
    m_questions.insert(question, [this, then](const QString &methodId) {
        if (methodId.isEmpty()) {
            endTurn();
            return;
        }
        QJsonObject method;
        for (const QJsonValue &value : std::as_const(m_authMethods)) {
            if (value.toObject().value(QStringLiteral("id")).toString() == methodId) {
                method = value.toObject();
            }
        }
        const QJsonObject terminal = method.value(QStringLiteral("_meta")).toObject().value(QStringLiteral("terminal-auth")).toObject();
        if (!terminal.value(QStringLiteral("command")).toString().isEmpty()) {
            runLogin(methodId, terminal, then);
            return;
        }
        // the agent does the signing in itself — a browser page, a key prompt
        request(QStringLiteral("authenticate"), QJsonObject{{QStringLiteral("methodId"), methodId}},
                [this, then, methodId](const QJsonObject &, const QJsonObject &error) {
                    if (!error.isEmpty()) {
                        // the method's own description says what it needs, e.g. "run 'agent login' first"
                        QString how;
                        for (const QJsonValue &value : std::as_const(m_authMethods)) {
                            if (value.toObject().value(QStringLiteral("id")).toString() == methodId) {
                                how = value.toObject().value(QStringLiteral("description")).toString();
                            }
                        }
                        const QString why = error.value(QStringLiteral("message")).toString();
                        Q_EMIT errorOccurred(how.isEmpty() ? i18n("Signing in to %1 failed: %2", m_agentName, why)
                                                           : i18n("Signing in to %1 failed: %2. %3", m_agentName, why, how));
                        endTurn();
                        return;
                    }
                    m_authMethods = QJsonArray(); // signed in: no second question
                    openSession(then);
                });
    });
    Q_EMIT questionAsked(question, i18n("%1 needs you to sign in.", m_agentName), options);
}

void AcpChatBackend::runLogin(const QString &methodId, const QJsonObject &terminal, const std::function<void()> &then)
{
    const QString card = QStringLiteral("acp-login:") + m_agentId;
    Q_EMIT toolStarted(card, terminal.value(QStringLiteral("label")).toString(i18n("Signing in to %1", m_agentName)));
    m_login = new QProcess(this);
    m_login->setProcessChannelMode(QProcess::MergedChannels);
    QStringList arguments;
    for (const QJsonValue &argument : terminal.value(QStringLiteral("args")).toArray()) {
        arguments << argument.toString();
    }
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    const QJsonObject extra = terminal.value(QStringLiteral("env")).toObject();
    for (auto it = extra.begin(); it != extra.end(); ++it) {
        env.insert(it.key(), it.value().toString());
    }
    m_login->setProcessEnvironment(env);
    auto output = std::make_shared<QString>();
    auto opened = std::make_shared<bool>(false);
    connect(m_login, &QProcess::readyReadStandardOutput, this, [this, output, opened]() {
        *output += QString::fromUtf8(m_login->readAllStandardOutput());
        // the command talks to a terminal: its device code and link are what
        // the user needs, shown as they arrive
        static const QRegularExpression ansi(QStringLiteral("\\x1b\\[[0-9;?]*[A-Za-z]"));
        QString shown = *output;
        shown.remove(ansi);
        Q_EMIT assistantStream(shown.trimmed());
        static const QRegularExpression link(QStringLiteral("https?://[^\\s\"'<>]+"));
        const QRegularExpressionMatch match = link.match(shown);
        if (!*opened && match.hasMatch()) {
            *opened = true;
            QDesktopServices::openUrl(QUrl(match.captured(0)));
        }
    });
    connect(m_login, &QProcess::finished, this, [this, card, methodId, then](int exitCode, QProcess::ExitStatus status) {
        m_login->deleteLater();
        const bool ok = status == QProcess::NormalExit && exitCode == 0;
        Q_EMIT toolFinished(card, !ok, ok ? i18n("Signed in.") : i18n("The sign-in did not finish."));
        if (!ok) {
            endTurn();
            return;
        }
        // tell the agent the method is done now; then the session it refused
        request(QStringLiteral("authenticate"), QJsonObject{{QStringLiteral("methodId"), methodId}}, [this, then](const QJsonObject &, const QJsonObject &) {
            m_authMethods = QJsonArray();
            openSession(then);
        });
    });
    m_login->start(terminal.value(QStringLiteral("command")).toString(), arguments);
}

QString AcpChatBackend::lastError(const QString &fallback) const
{
    // From the end back: the first line that names an error, skipping the
    // "Node.js v22…" Node prints under every crash.
    for (int i = m_stderr.size() - 1; i >= 0; --i) {
        const QString line = m_stderr.at(i).trimmed();
        if (line.contains(QLatin1String("Error"), Qt::CaseInsensitive) && !line.startsWith(QLatin1String("at "))) {
            return line.left(300);
        }
    }
    for (int i = m_stderr.size() - 1; i >= 0; --i) {
        const QString line = m_stderr.at(i).trimmed();
        if (!line.isEmpty() && !line.startsWith(QLatin1String("Node.js")) && !line.startsWith(QLatin1String("at "))) {
            return line.left(300);
        }
    }
    return fallback;
}

// ── the turn ────────────────────────────────────────────────────────────────

void AcpChatBackend::prompt(const QString &text)
{
    m_stream.clear();
    const QJsonArray content{QJsonObject{{QStringLiteral("type"), QStringLiteral("text")}, {QStringLiteral("text"), text}}};
    request(QStringLiteral("session/prompt"), QJsonObject{{QStringLiteral("sessionId"), m_session}, {QStringLiteral("prompt"), content}},
            [this](const QJsonObject &, const QJsonObject &error) {
                if (!error.isEmpty() && !m_stopped) {
                    Q_EMIT errorOccurred(i18n("%1: %2", m_agentName, error.value(QStringLiteral("message")).toString()));
                }
                endTurn();
            });
}

void AcpChatBackend::endTurn()
{
    if (!m_busy) {
        return;
    }
    m_busy = false;
    m_stream.clear();
    Q_EMIT busyChanged(false);
    sendPending();
}

void AcpChatBackend::sendPending()
{
    if (m_pending.isEmpty() || m_stopped) {
        return;
    }
    const QString text = m_pending.join(QLatin1Char('\n'));
    m_pending.clear();
    sendMessage(text);
}

// ── JSON-RPC ────────────────────────────────────────────────────────────────

void AcpChatBackend::request(const QString &method, const QJsonObject &params, const Reply &reply)
{
    const int id = m_nextId++;
    m_replies.insert(id, reply);
    write(QJsonObject{{QStringLiteral("jsonrpc"), QStringLiteral("2.0")}, {QStringLiteral("id"), id}, {QStringLiteral("method"), method},
                      {QStringLiteral("params"), params}});
}

void AcpChatBackend::notify(const QString &method, const QJsonObject &params)
{
    write(QJsonObject{{QStringLiteral("jsonrpc"), QStringLiteral("2.0")}, {QStringLiteral("method"), method}, {QStringLiteral("params"), params}});
}

void AcpChatBackend::respond(const QJsonValue &id, const QJsonObject &result)
{
    write(QJsonObject{{QStringLiteral("jsonrpc"), QStringLiteral("2.0")}, {QStringLiteral("id"), id}, {QStringLiteral("result"), result}});
}

void AcpChatBackend::respondError(const QJsonValue &id, int code, const QString &message)
{
    write(QJsonObject{{QStringLiteral("jsonrpc"), QStringLiteral("2.0")},
                      {QStringLiteral("id"), id},
                      {QStringLiteral("error"), QJsonObject{{QStringLiteral("code"), code}, {QStringLiteral("message"), message}}}});
}

void AcpChatBackend::write(const QJsonObject &message)
{
    if (m_process && m_process->state() == QProcess::Running) {
        m_process->write(QJsonDocument(message).toJson(QJsonDocument::Compact) + '\n');
    }
}

void AcpChatBackend::readOutput()
{
    m_buffer += m_process->readAllStandardOutput();
    int end = -1;
    while ((end = m_buffer.indexOf('\n')) >= 0) {
        const QByteArray line = m_buffer.left(end).trimmed();
        m_buffer.remove(0, end + 1);
        if (line.isEmpty()) {
            continue;
        }
        const QJsonDocument document = QJsonDocument::fromJson(line);
        if (document.isObject()) {
            handle(document.object());
        }
    }
}

void AcpChatBackend::handle(const QJsonObject &message)
{
    const QString method = message.value(QStringLiteral("method")).toString();
    const QJsonValue id = message.value(QStringLiteral("id"));
    if (method.isEmpty()) {
        // a reply to one of ours
        const Reply reply = m_replies.take(id.toInt());
        if (reply) {
            reply(message.value(QStringLiteral("result")).toObject(), message.value(QStringLiteral("error")).toObject());
        }
        return;
    }
    if (method == QLatin1String("session/update")) {
        if (!m_loading) {
            onUpdate(message.value(QStringLiteral("params")).toObject().value(QStringLiteral("update")).toObject());
        }
        return;
    }
    if (id.isUndefined()) {
        return; // a notification this client has no use for
    }
    if (method == QLatin1String("session/request_permission")) {
        onPermission(id, message.value(QStringLiteral("params")).toObject());
        return;
    }
    // fs/* and terminal/*: offered as off in initialize; anything else unknown
    respondError(id, -32601, QStringLiteral("Not supported by Wunjo Make"));
}

void AcpChatBackend::onUpdate(const QJsonObject &update)
{
    const QString kind = update.value(QStringLiteral("sessionUpdate")).toString();
    if (kind == QLatin1String("agent_message_chunk")) {
        const QJsonObject content = update.value(QStringLiteral("content")).toObject();
        if (content.value(QStringLiteral("type")).toString() == QLatin1String("text")) {
            m_stream += content.value(QStringLiteral("text")).toString();
            Q_EMIT assistantStream(m_stream);
        }
    } else if (kind == QLatin1String("tool_call")) {
        // text after a tool call is a new paragraph under the card, not more
        // of the one above it
        m_stream.clear();
        const QString card = QStringLiteral("acp-tool:") + update.value(QStringLiteral("toolCallId")).toString();
        Q_EMIT toolStarted(card, update.value(QStringLiteral("title")).toString());
        const QString status = update.value(QStringLiteral("status")).toString();
        if (status == QLatin1String("completed") || status == QLatin1String("failed")) {
            Q_EMIT toolFinished(card, status == QLatin1String("failed"), contentText(update.value(QStringLiteral("content")).toArray()));
        }
    } else if (kind == QLatin1String("tool_call_update")) {
        const QString card = QStringLiteral("acp-tool:") + update.value(QStringLiteral("toolCallId")).toString();
        const QString status = update.value(QStringLiteral("status")).toString();
        const QString text = contentText(update.value(QStringLiteral("content")).toArray());
        if (status == QLatin1String("completed") || status == QLatin1String("failed")) {
            Q_EMIT toolFinished(card, status == QLatin1String("failed"), text);
        } else {
            Q_EMIT toolProgress(card, -1, text);
        }
    } else if (kind == QLatin1String("current_mode_update")) {
        // the agent changed mode itself, e.g. leaving Plan once the plan is agreed
        m_currentMode = update.value(QStringLiteral("currentModeId")).toString();
        saveModes();
    }
    // agent_thought_chunk, plan, available_commands_update: the "thinking"
    // line already covers the wait
}

void AcpChatBackend::onPermission(const QJsonValue &id, const QJsonObject &params)
{
    const QJsonObject toolCall = params.value(QStringLiteral("toolCall")).toObject();
    const QString title = toolCall.value(QStringLiteral("title")).toString();
    const QJsonArray options = params.value(QStringLiteral("options")).toArray();

    auto select = [this, id](const QString &optionId) {
        respond(id, optionId.isEmpty() ? QJsonObject{{QStringLiteral("outcome"), QJsonObject{{QStringLiteral("outcome"), QStringLiteral("cancelled")}}}}
                                       : QJsonObject{{QStringLiteral("outcome"), QJsonObject{{QStringLiteral("outcome"), QStringLiteral("selected")},
                                                                                               {QStringLiteral("optionId"), optionId}}}});
    };

    // The editor's own tools are what the user picked this agent to use; a
    // paid one asks the price in its own window anyway. Asking before every
    // cut would make the chat unusable, so they are allowed for the session.
    if (title.contains(QLatin1String("wunjo-make")) || title.contains(QLatin1String("wunjo_make"))) {
        QString chosen;
        for (const QString &wanted : {QStringLiteral("allow_always"), QStringLiteral("allow_once")}) {
            for (const QJsonValue &value : options) {
                if (chosen.isEmpty() && value.toObject().value(QStringLiteral("kind")).toString() == wanted) {
                    chosen = value.toObject().value(QStringLiteral("optionId")).toString();
                }
            }
        }
        if (!chosen.isEmpty()) {
            select(chosen);
            return;
        }
    }

    QJsonArray choices;
    for (const QJsonValue &value : options) {
        const QJsonObject option = value.toObject();
        choices.append(QJsonObject{{QStringLiteral("id"), option.value(QStringLiteral("optionId"))}, {QStringLiteral("label"), option.value(QStringLiteral("name"))}});
    }
    const QString question = QStringLiteral("acp-permission-%1").arg(m_nextQuestion++);
    m_questions.insert(question, select);
    Q_EMIT questionAsked(question, i18n("%1 asks permission: %2", m_agentName, title), choices);
}

void AcpChatBackend::stopProcess()
{
    if (m_login) {
        m_login->kill();
    }
    if (!m_process || m_process->state() == QProcess::NotRunning) {
        return;
    }
    m_stopped = true;
    const qint64 pid = m_process->processId();
#if defined(Q_OS_UNIX)
    // the whole group: the adapter and the agent program it started
    ::kill(-pid_t(pid), SIGTERM);
    if (!m_process->waitForFinished(2000)) {
        ::kill(-pid_t(pid), SIGKILL);
        m_process->waitForFinished(1000);
    }
#elif defined(Q_OS_WIN)
    // /T takes the children with it; a plain kill leaves claude.exe running
    QProcess::execute(QStringLiteral("taskkill"), {QStringLiteral("/PID"), QString::number(pid), QStringLiteral("/T"), QStringLiteral("/F")});
    m_process->waitForFinished(2000);
#else
    m_process->kill();
    m_process->waitForFinished(2000);
#endif
}

QString AcpChatBackend::storedSession() const
{
    if (m_chatSession.isEmpty()) {
        return {};
    }
    return readSessions().value(m_agentId + QLatin1Char('|') + m_chatSession).toString();
}

void AcpChatBackend::storeSession(const QString &acpSessionId)
{
    if (m_chatSession.isEmpty()) {
        return;
    }
    QJsonObject sessions = readSessions();
    const QString key = m_agentId + QLatin1Char('|') + m_chatSession;
    if (acpSessionId.isEmpty()) {
        sessions.remove(key);
    } else {
        sessions.insert(key, acpSessionId);
    }
    QDir().mkpath(QFileInfo(sessionsPath()).absolutePath());
    QSaveFile file(sessionsPath());
    if (file.open(QIODevice::WriteOnly)) {
        file.write(QJsonDocument(sessions).toJson());
        file.commit();
    }
}
