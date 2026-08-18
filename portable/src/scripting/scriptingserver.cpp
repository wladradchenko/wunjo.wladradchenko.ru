/*
    SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "scriptingserver.h"

#include <QCoreApplication>
#include <QDir>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>
#include <QMetaMethod>
#include <QMetaObject>
#include <QMetaType>
#include <QStandardPaths>

namespace {

/** @brief Convert one JSON argument to the type the method declares.
 *
 *  The interface only uses six argument types, so this is a closed list rather
 *  than a general converter: int, QString, bool, QStringList, double and
 *  QList<int>. Anything else is a method that was written after this function
 *  and needs a line adding here — which is why an unknown type is an error
 *  rather than a silent default.
 */
bool toArgument(const QJsonValue &value, int typeId, QVariant &out, QString *errorOut)
{
    switch (typeId) {
    case QMetaType::Int:
        out = value.toInt();
        return true;
    case QMetaType::Double:
        out = value.toDouble();
        return true;
    case QMetaType::Bool:
        out = value.toBool();
        return true;
    case QMetaType::QString:
        out = value.toString();
        return true;
    case QMetaType::QStringList: {
        QStringList list;
        const QJsonArray array = value.toArray();
        for (const QJsonValue &item : array) {
            list << item.toString();
        }
        out = list;
        return true;
    }
    default:
        break;
    }
    if (typeId == QMetaType::fromType<QList<int>>().id()) {
        QList<int> list;
        const QJsonArray array = value.toArray();
        for (const QJsonValue &item : array) {
            list << item.toInt();
        }
        out = QVariant::fromValue(list);
        return true;
    }
    if (errorOut) {
        *errorOut = QStringLiteral("unsupported argument type '%1'").arg(QLatin1String(QMetaType(typeId).name()));
    }
    return false;
}

/** @brief Convert a returned QVariant to JSON.
 *
 *  QVariantMap and QVariantList are already JSON-shaped and go through
 *  QJsonValue::fromVariant; QList<int> is the one type that is not, because it
 *  is not a QVariant container.
 */
QJsonValue fromReturn(const QVariant &value)
{
    if (value.metaType() == QMetaType::fromType<QList<int>>()) {
        QJsonArray array;
        const QList<int> list = value.value<QList<int>>();
        for (int item : list) {
            array.append(item);
        }
        return array;
    }
    return QJsonValue::fromVariant(value);
}

} // namespace

ScriptingServer::ScriptingServer(QObject *target, QObject *parent)
    : QObject(parent)
    , m_target(target)
{
    connect(&m_server, &QLocalServer::newConnection, this, &ScriptingServer::newConnection);
}

ScriptingServer::~ScriptingServer()
{
    m_server.close();
}

namespace {
/** @brief True when something is listening on @p name right now.
 *
 *  A socket file left behind by a process that died looks exactly like a live
 *  one until somebody tries to connect, so this is the only way to tell a
 *  second instance from a crash.
 */
bool socketAlive(const QString &name)
{
    QLocalSocket probe;
    probe.connectToServer(name);
    const bool alive = probe.waitForConnected(300);
    probe.abort();
    return alive;
}
} // namespace

QString ScriptingServer::defaultSocketName()
{
    // On Unix QLocalServer turns a bare name into a file in a temp directory.
    // XDG_RUNTIME_DIR is per user and cleaned up when the session ends, which
    // is what a socket that means "this user's editor" should follow.
    const QString runtime = QStandardPaths::writableLocation(QStandardPaths::RuntimeLocation);
    const QString name = QStringLiteral("online.wunjo.make.scripting");
    return runtime.isEmpty() ? name : runtime + QLatin1Char('/') + name;
}

QString ScriptingServer::socketNameForPid(qint64 pid)
{
    return defaultSocketName() + QLatin1Char('-') + QString::number(pid);
}

bool ScriptingServer::start()
{
    const QString preferred = defaultSocketName();
    if (m_server.listen(preferred)) {
        m_name = preferred;
        return true;
    }
    // Taken. If nobody is behind it, a previous run died without cleaning up
    // and the name is ours to reclaim; otherwise another copy is running and
    // this one goes on its own name, as KDBusService's Multiple did.
    if (!socketAlive(preferred)) {
        QLocalServer::removeServer(preferred);
        if (m_server.listen(preferred)) {
            m_name = preferred;
            qWarning() << "ScriptingServer: reclaimed a stale socket at" << preferred;
            return true;
        }
    }
    const QString own = socketNameForPid(QCoreApplication::applicationPid());
    QLocalServer::removeServer(own);
    if (m_server.listen(own)) {
        m_name = own;
        qWarning() << "ScriptingServer: another instance holds" << preferred << "— listening on" << own;
        return true;
    }
    qWarning() << "ScriptingServer: cannot listen on" << preferred << "or" << own << m_server.errorString();
    return false;
}

void ScriptingServer::newConnection()
{
    while (QLocalSocket *socket = m_server.nextPendingConnection()) {
        connect(socket, &QLocalSocket::readyRead, this, [this, socket]() { readFrom(socket); });
        connect(socket, &QLocalSocket::disconnected, socket, &QLocalSocket::deleteLater);
    }
}

void ScriptingServer::readFrom(QLocalSocket *socket)
{
    // One request per line. Requests can be pipelined, and a read can stop in
    // the middle of one, so only whole lines are taken and the rest is left in
    // the socket's buffer for the next readyRead.
    while (socket->canReadLine()) {
        const QByteArray line = socket->readLine().trimmed();
        if (line.isEmpty()) {
            continue;
        }
        QJsonParseError error;
        const QJsonObject request = QJsonDocument::fromJson(line, &error).object();
        QJsonObject reply;
        if (error.error != QJsonParseError::NoError) {
            reply.insert(QStringLiteral("id"), 0);
            reply.insert(QStringLiteral("ok"), false);
            reply.insert(QStringLiteral("error"), QStringLiteral("malformed request: %1").arg(error.errorString()));
        } else {
            reply = dispatch(request);
        }
        socket->write(QJsonDocument(reply).toJson(QJsonDocument::Compact) + '\n');
        socket->flush();
    }
}

QStringList ScriptingServer::methodList() const
{
    QStringList names;
    const QMetaObject *mo = m_target->metaObject();
    for (int i = 0; i < mo->methodCount(); ++i) {
        const QMetaMethod method = mo->method(i);
        if (method.methodType() != QMetaMethod::Slot) {
            continue;
        }
        if (!(method.attributes() & QMetaMethod::Scriptable)) {
            continue;
        }
        names << QString::fromLatin1(method.methodSignature());
    }
    return names;
}

QJsonObject ScriptingServer::dispatch(const QJsonObject &request)
{
    QJsonObject reply;
    reply.insert(QStringLiteral("id"), request.value(QStringLiteral("id")));

    const QString name = request.value(QStringLiteral("method")).toString();
    const QJsonArray args = request.value(QStringLiteral("args")).toArray();

    const auto fail = [&reply](const QString &message) {
        reply.insert(QStringLiteral("ok"), false);
        reply.insert(QStringLiteral("error"), message);
        return reply;
    };
    const auto succeed = [&reply](const QJsonValue &result) {
        reply.insert(QStringLiteral("ok"), true);
        reply.insert(QStringLiteral("result"), result);
        return reply;
    };

    // Served here rather than by the target: they are about the connection, and
    // the application should not have to grow slots to describe itself.
    if (name == QLatin1String("__ping__")) {
        return succeed(true);
    }
    if (name == QLatin1String("__methods__")) {
        return succeed(QJsonArray::fromStringList(methodList()));
    }

    // Find the scriptable slot with this name and this many arguments. Names
    // are unique in practice, but overloads are legal, so the count decides.
    const QMetaObject *mo = m_target->metaObject();
    QMetaMethod method;
    bool nameSeen = false;
    for (int i = 0; i < mo->methodCount(); ++i) {
        const QMetaMethod candidate = mo->method(i);
        if (candidate.methodType() != QMetaMethod::Slot || !(candidate.attributes() & QMetaMethod::Scriptable)) {
            continue;
        }
        if (QString::fromLatin1(candidate.name()) != name) {
            continue;
        }
        nameSeen = true;
        if (candidate.parameterCount() == args.size()) {
            method = candidate;
            break;
        }
    }
    if (!method.isValid()) {
        return fail(nameSeen ? QStringLiteral("'%1' takes a different number of arguments than the %2 given").arg(name).arg(args.size())
                             : QStringLiteral("no such method: %1").arg(name));
    }

    // Convert first, invoke second: a half-converted call must not reach the
    // editor and change something before failing.
    QList<QVariant> values;
    values.reserve(method.parameterCount());
    for (int i = 0; i < method.parameterCount(); ++i) {
        QVariant value;
        QString error;
        if (!toArgument(args.at(i), method.parameterType(i), value, &error)) {
            return fail(QStringLiteral("argument %1 of %2: %3").arg(i + 1).arg(name, error));
        }
        values.append(value);
    }

    // QGenericArgument points into the variants, so they have to outlive the
    // call — hence the list above rather than converting inline.
    QGenericArgument a[10];
    for (int i = 0; i < values.size() && i < 10; ++i) {
        a[i] = QGenericArgument(QMetaType(method.parameterType(i)).name(), values.at(i).constData());
    }

    const QMetaType returnType = method.returnMetaType();
    if (returnType.id() == QMetaType::Void) {
        const bool ok = method.invoke(m_target, Qt::DirectConnection, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
        return ok ? succeed(QJsonValue::Null) : fail(QStringLiteral("could not invoke %1").arg(name));
    }

    QVariant result(returnType);
    QGenericReturnArgument returnArg(returnType.name(), result.data());
    const bool ok = method.invoke(m_target, Qt::DirectConnection, returnArg, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9]);
    if (!ok) {
        return fail(QStringLiteral("could not invoke %1").arg(name));
    }
    return succeed(fromReturn(result));
}
