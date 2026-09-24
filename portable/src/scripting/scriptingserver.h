/*
    SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QLocalServer>
#include <QLocalSocket>
#include <QObject>
#include <QVariant>

class QJsonObject;

/** @class ScriptingServer
    @brief Exposes the application's scriptable methods over a local socket.

    This is what the assistant drives the editor through. It used to be D-Bus,
    which only exists on Linux; a `QLocalServer` is a named pipe on Windows and
    a unix socket everywhere else, needs no daemon, no port and no firewall
    permission, and is already how the render process reports its progress
    (@ref RenderServer).

    Nothing is registered method by method. The methods are the `Q_SCRIPTABLE`
    slots of the target object — the same set D-Bus used to export — and they
    are found through `QMetaObject` at call time. Adding a scriptable method to
    MainWindow is therefore all it takes to make it callable from outside, as
    before.

    ## Protocol

    One JSON object per line, in both directions.

        → {"id": 7, "method": "scriptImportMedia", "args": ["/a.mp4", "-1"]}
        ← {"id": 7, "ok": true, "result": ["3"]}
        ← {"id": 7, "ok": false, "error": "no such method: scriptFoo"}

    `id` is echoed back and is the caller's business; anything else about the
    request is ignored. Two names are answered by the server itself rather than
    the target: `__methods__` lists every callable signature, and `__ping__`
    answers `true` — between them a client can tell "the application is not
    running" from "the application is running and does not have that method".

    ## Several instances

    The first instance takes the plain socket name and later ones get their pid
    appended, so a client that knows nothing finds the first one and a client
    that wants a particular copy can name it. This mirrors what KDBusService did
    with `Multiple` (its default, and what this application used): running two
    copies has always been allowed, and this does not change that.
 */
class ScriptingServer : public QObject
{
    Q_OBJECT
public:
    /** @brief Serve @p target's scriptable slots. Call @ref start afterwards. */
    explicit ScriptingServer(QObject *target, QObject *parent = nullptr);
    ~ScriptingServer() override;

    /** @brief Begin listening. False means no socket could be claimed at all —
     *  the application still runs, it just cannot be scripted. */
    bool start();

    /** @brief The name this instance actually listens on, empty before
     *  @ref start succeeds. */
    QString name() const { return m_name; }

    /** @brief The name a client should try first. Under `XDG_RUNTIME_DIR` when
     *  there is one, so it dies with the login session rather than lingering in
     *  a world-writable /tmp. */
    static QString defaultSocketName();
    /** @brief The name of the instance with process id @p pid, which is what a
     *  second copy of the application ends up on. */
    static QString socketNameForPid(qint64 pid);

private:
    void newConnection();
    void readFrom(QLocalSocket *socket);
    /** @brief Run one request against the target and produce its reply. */
    QJsonObject dispatch(const QJsonObject &request);
    /** @brief Every scriptable slot of the target, as readable signatures. */
    QStringList methodList() const;

    QObject *m_target;
    QLocalServer m_server;
    QString m_name;
};
