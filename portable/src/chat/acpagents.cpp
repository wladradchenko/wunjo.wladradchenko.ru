/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "acpagents.h"

#include "plugins/filedownloader.h"

#include <KArchiveDirectory>
#include <KLocalizedString>
#include <KTar>
#include <KZip>

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QSaveFile>
#include <QStandardPaths>
#include <QSysInfo>
#include <QUrl>

#include <memory>

namespace AcpAgents {

namespace {

const char REGISTRY_URL[] = "https://cdn.agentclientprotocol.com/registry/v1/latest/registry.json";

QString dataRoot()
{
    return QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + QStringLiteral("/acp");
}

QString registryPath()
{
    return dataRoot() + QStringLiteral("/registry.json");
}

QString installDir(const QJsonObject &agent)
{
    return dataRoot() + QLatin1Char('/') + agent.value(QStringLiteral("id")).toString() + QLatin1Char('/') +
        agent.value(QStringLiteral("version")).toString();
}

QString markerPath(const QJsonObject &agent)
{
    return installDir(agent) + QStringLiteral("/.installed");
}

/** @brief The registry's name for this machine: linux-x86_64, darwin-aarch64,
 *  windows-x86_64 … */
QString platformKey()
{
    QString os = QStringLiteral("linux");
#if defined(Q_OS_WIN)
    os = QStringLiteral("windows");
#elif defined(Q_OS_MACOS)
    os = QStringLiteral("darwin");
#endif
    const QString cpu = QSysInfo::currentCpuArchitecture();
    const QString arch = (cpu == QLatin1String("arm64") || cpu == QLatin1String("aarch64")) ? QStringLiteral("aarch64") : QStringLiteral("x86_64");
    return os + QLatin1Char('-') + arch;
}

/** @brief The adapters that run a program the user may already have, and the
 *  variable that tells them where it is. */
struct Wrapped {
    const char *agent;
    const char *program;
    const char *variable;
};
constexpr Wrapped WRAPPED[] = {
    {"claude-acp", "claude", "CLAUDE_CODE_EXECUTABLE"},
    {"codex-acp", "codex", "CODEX_PATH"},
};

const Wrapped *wrapped(const QString &id)
{
    for (const Wrapped &w : WRAPPED) {
        if (id == QLatin1String(w.agent)) {
            return &w;
        }
    }
    return nullptr;
}

/** @brief `@scope/name@1.2.3` → `@scope/name`. */
QString packageName(const QString &spec)
{
    const int at = spec.lastIndexOf(QLatin1Char('@'));
    return at > 0 ? spec.left(at) : spec;
}

/** @brief npm, next to the Node it belongs to: lib/node_modules/npm on Unix,
 *  node_modules/npm beside node.exe on Windows. */
QString npmCli(const QString &node)
{
    QDir dir = QFileInfo(node).dir();
    const QStringList candidates = {dir.filePath(QStringLiteral("node_modules/npm/bin/npm-cli.js")),
                                    dir.filePath(QStringLiteral("../lib/node_modules/npm/bin/npm-cli.js"))};
    for (const QString &candidate : candidates) {
        if (QFileInfo::exists(candidate)) {
            return QFileInfo(candidate).canonicalFilePath();
        }
    }
    return {};
}

/** @brief Whether an npm package's "bin" is JavaScript for Node, or a program
 *  of its own (Grok ships a native binary there) to be started directly. */
bool needsNode(const QString &path)
{
    QFile file(QFileInfo(path).canonicalFilePath());
    if (!file.open(QIODevice::ReadOnly)) {
        return true;
    }
    const QByteArray head = file.read(256);
    if (head.startsWith("\x7f" "ELF") || head.startsWith("MZ")) {
        return false; // Linux, Windows
    }
    static const QList<QByteArray> machO = {QByteArray("\xfe\xed\xfa\xce", 4), QByteArray("\xfe\xed\xfa\xcf", 4), QByteArray("\xce\xfa\xed\xfe", 4),
                                            QByteArray("\xcf\xfa\xed\xfe", 4), QByteArray("\xca\xfe\xba\xbe", 4)};
    for (const QByteArray &magic : machO) {
        if (head.startsWith(magic)) {
            return false; // macOS
        }
    }
    if (head.startsWith("#!")) {
        return head.left(head.indexOf('\n')).contains("node"); // a shell script runs itself
    }
    return true;
}

QJsonObject readMarker(const QJsonObject &agent)
{
    QFile file(markerPath(agent));
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    return QJsonDocument::fromJson(file.readAll()).object();
}

bool writeMarker(const QJsonObject &agent, const QJsonObject &content)
{
    QSaveFile file(markerPath(agent));
    return file.open(QIODevice::WriteOnly) && file.write(QJsonDocument(content).toJson()) >= 0 && file.commit();
}

} // namespace

QString nodeExecutable()
{
    const QString dir = QStandardPaths::locate(QStandardPaths::AppDataLocation, QStringLiteral("node"), QStandardPaths::LocateDirectory);
    if (!dir.isEmpty()) {
        for (const QString &candidate : {QStringLiteral("/bin/node"), QStringLiteral("/node.exe"), QStringLiteral("/bin/node.exe")}) {
            if (QFileInfo(dir + candidate).isExecutable()) {
                return dir + candidate;
            }
        }
    }
    // a build tree runs without the packaged runtime; a system Node 22 will do
    return QStandardPaths::findExecutable(QStringLiteral("node"));
}

QJsonArray registry()
{
    QFile file(registryPath());
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    return QJsonDocument::fromJson(file.readAll()).object().value(QStringLiteral("agents")).toArray();
}

QJsonObject agent(const QString &id)
{
    const QJsonArray agents = registry();
    for (const QJsonValue &value : agents) {
        if (value.toObject().value(QStringLiteral("id")).toString() == id) {
            return value.toObject();
        }
    }
    return {};
}

void refreshRegistry(QObject *context, const std::function<void(bool ok)> &done)
{
    auto *network = new QNetworkAccessManager(context);
    QNetworkRequest request{QUrl(QString::fromLatin1(REGISTRY_URL))};
    request.setTransferTimeout(20000);
    QNetworkReply *reply = network->get(request);
    QObject::connect(reply, &QNetworkReply::finished, context, [reply, network, done]() {
        bool ok = false;
        if (reply->error() == QNetworkReply::NoError) {
            const QByteArray body = reply->readAll();
            // only a list that parses replaces the one we have
            if (QJsonDocument::fromJson(body).object().value(QStringLiteral("agents")).isArray()) {
                QDir().mkpath(dataRoot());
                QSaveFile file(registryPath());
                ok = file.open(QIODevice::WriteOnly) && file.write(body) == body.size() && file.commit();
            }
        }
        reply->deleteLater();
        network->deleteLater();
        if (done) {
            done(ok);
        }
    });
}

bool isInstalled(const QString &id)
{
    const QJsonObject entry = agent(id);
    return !entry.isEmpty() && QFileInfo::exists(markerPath(entry));
}

QString existingBinary(const QString &id)
{
    const Wrapped *w = wrapped(id);
    if (w == nullptr) {
        return {};
    }
    const QString name = QString::fromLatin1(w->program);
#if defined(Q_OS_WIN)
    const QString exe = name + QStringLiteral(".exe");
#else
    const QString exe = name;
#endif
    const QString onPath = QStandardPaths::findExecutable(name);
    if (!onPath.isEmpty()) {
        return onPath;
    }
    const QString home = QDir::homePath();
    const QStringList usual = {home + QStringLiteral("/.claude/local/") + exe, home + QStringLiteral("/.local/bin/") + exe};
    for (const QString &candidate : usual) {
        if (QFileInfo(candidate).isExecutable()) {
            return candidate;
        }
    }
    // The extensions of VS Code and its forks carry the program inside; the
    // newest version found wins.
    const QString prefix = name == QLatin1String("claude") ? QStringLiteral("anthropic.claude-code-") : QStringLiteral("openai.chatgpt-");
    for (const QString &editor : {QStringLiteral(".vscode"), QStringLiteral(".vscode-insiders"), QStringLiteral(".cursor"), QStringLiteral(".windsurf"),
                                  QStringLiteral(".vscode-oss"), QStringLiteral(".vscode-server")}) {
        QDir extensions(home + QLatin1Char('/') + editor + QStringLiteral("/extensions"));
        QStringList found = extensions.entryList({prefix + QLatin1Char('*')}, QDir::Dirs, QDir::Name | QDir::Reversed);
        for (const QString &extension : std::as_const(found)) {
            QDir root(extensions.filePath(extension));
            QStringList inside = {root.filePath(QStringLiteral("resources/native-binary/") + exe)};
            const QStringList platforms = QDir(root.filePath(QStringLiteral("bin"))).entryList(QDir::Dirs | QDir::NoDotAndDotDot);
            for (const QString &platform : platforms) {
                inside << root.filePath(QStringLiteral("bin/") + platform + QLatin1Char('/') + exe);
            }
            for (const QString &candidate : std::as_const(inside)) {
                if (QFileInfo(candidate).isExecutable()) {
                    return candidate;
                }
            }
        }
    }
    return {};
}

Launch launchFor(const QString &id)
{
    Launch launch;
    const QJsonObject entry = agent(id);
    if (entry.isEmpty()) {
        launch.error = i18n("The agent '%1' is not in the ACP registry.", id);
        return launch;
    }
    const QString dir = installDir(entry);
    const QJsonObject distribution = entry.value(QStringLiteral("distribution")).toObject();
    launch.environment = QProcessEnvironment::systemEnvironment();

    if (distribution.contains(QStringLiteral("npx"))) {
        const QJsonObject npx = distribution.value(QStringLiteral("npx")).toObject();
        const QString node = nodeExecutable();
        if (node.isEmpty()) {
            launch.error = i18n("Node.js is missing from this installation; the agent cannot start.");
            return launch;
        }
        // Installed without its own copy of the program it wraps, and that
        // program is gone since: install it again, whole.
        const Wrapped *w = wrapped(id);
        const QString binary = existingBinary(id);
        if (readMarker(entry).value(QStringLiteral("omitOptional")).toBool() && binary.isEmpty()) {
            QFile::remove(markerPath(entry));
            launch.error = i18n("%1 has to be installed again.", entry.value(QStringLiteral("name")).toString());
            return launch;
        }
        if (w != nullptr && !binary.isEmpty()) {
            launch.environment.insert(QString::fromLatin1(w->variable), binary);
        }
        // Run the package's entry script with this Node, not through npx or the
        // .bin shims: a shim stores the path of the Node it was made with, and
        // inside an AppImage that path changes on every start.
        const QString name = packageName(npx.value(QStringLiteral("package")).toString());
        QFile manifest(dir + QStringLiteral("/node_modules/") + name + QStringLiteral("/package.json"));
        if (!manifest.open(QIODevice::ReadOnly)) {
            launch.error = i18n("%1 is not installed.", entry.value(QStringLiteral("name")).toString());
            return launch;
        }
        const QJsonValue bin = QJsonDocument::fromJson(manifest.readAll()).object().value(QStringLiteral("bin"));
        QString script = bin.toString();
        if (bin.isObject()) {
            const QJsonObject scripts = bin.toObject();
            const QString shortName = name.section(QLatin1Char('/'), -1);
            script = scripts.contains(shortName) ? scripts.value(shortName).toString() : scripts.begin().value().toString();
        }
        const QString entryPath = QDir(dir + QStringLiteral("/node_modules/") + name).absoluteFilePath(script);
        if (needsNode(entryPath)) {
            launch.program = node;
            launch.arguments << entryPath;
        } else {
            launch.program = entryPath;
        }
        for (const QJsonValue &argument : npx.value(QStringLiteral("args")).toArray()) {
            launch.arguments << argument.toString();
        }
        const QJsonObject env = npx.value(QStringLiteral("env")).toObject();
        for (auto it = env.begin(); it != env.end(); ++it) {
            launch.environment.insert(it.key(), it.value().toString());
        }
        // whatever the agent starts itself finds this Node first
        launch.environment.insert(QStringLiteral("PATH"),
                                  QFileInfo(node).absolutePath() + QDir::listSeparator() + launch.environment.value(QStringLiteral("PATH")));
        return launch;
    }

    if (distribution.contains(QStringLiteral("binary"))) {
        const QJsonObject target = distribution.value(QStringLiteral("binary")).toObject().value(platformKey()).toObject();
        if (target.isEmpty()) {
            launch.error = i18n("%1 has no build for this system.", entry.value(QStringLiteral("name")).toString());
            return launch;
        }
        QString cmd = target.value(QStringLiteral("cmd")).toString();
        if (cmd.startsWith(QLatin1String("./"))) {
            cmd = cmd.mid(2);
        }
        launch.program = QDir(dir).absoluteFilePath(cmd);
        for (const QJsonValue &argument : target.value(QStringLiteral("args")).toArray()) {
            launch.arguments << argument.toString();
        }
        const QJsonObject env = target.value(QStringLiteral("env")).toObject();
        for (auto it = env.begin(); it != env.end(); ++it) {
            launch.environment.insert(it.key(), it.value().toString());
        }
        return launch;
    }

    launch.error = i18n("%1 is distributed in a way this version cannot install yet.", entry.value(QStringLiteral("name")).toString());
    return launch;
}

Installer::Installer(const QString &id, QObject *parent)
    : QObject(parent)
    , m_id(id)
{
}

void Installer::start()
{
    m_agent = agent(m_id);
    if (m_agent.isEmpty()) {
        finish(false, i18n("The agent '%1' is not in the ACP registry.", m_id));
        return;
    }
    m_dir = installDir(m_agent);
    if (!QDir().mkpath(m_dir)) {
        finish(false, i18n("Could not create %1.", m_dir));
        return;
    }
    const QJsonObject distribution = m_agent.value(QStringLiteral("distribution")).toObject();
    if (distribution.contains(QStringLiteral("npx"))) {
        installPackage(distribution.value(QStringLiteral("npx")).toObject());
    } else if (distribution.contains(QStringLiteral("binary"))) {
        installArchive(distribution.value(QStringLiteral("binary")).toObject());
    } else {
        finish(false, i18n("%1 is distributed in a way this version cannot install yet.", m_agent.value(QStringLiteral("name")).toString()));
    }
}

void Installer::installPackage(const QJsonObject &npx)
{
    const QString node = nodeExecutable();
    const QString npm = node.isEmpty() ? QString() : npmCli(node);
    if (npm.isEmpty()) {
        finish(false, i18n("Node.js is missing from this installation; the agent cannot be installed."));
        return;
    }
    // A program of the user's the adapter can run instead of its own copy: the
    // copy is the optional platform package, hundreds of megabytes.
    const bool omitOptional = !existingBinary(m_id).isEmpty();
    QStringList arguments = {npm, QStringLiteral("install"), QStringLiteral("--prefix"), m_dir, QStringLiteral("--no-audit"),
                             QStringLiteral("--no-fund"), QStringLiteral("--loglevel=error")};
    if (omitOptional) {
        arguments << QStringLiteral("--omit=optional");
    }
    arguments << npx.value(QStringLiteral("package")).toString();

    m_process = new QProcess(this);
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    // npm's cache in the app's data folder, so nothing is written elsewhere
    env.insert(QStringLiteral("npm_config_cache"), dataRoot() + QStringLiteral("/npm-cache"));
    // the Node that runs npm is also the one its install scripts should find
    env.insert(QStringLiteral("PATH"), QFileInfo(node).absolutePath() + QDir::listSeparator() + env.value(QStringLiteral("PATH")));
    m_process->setProcessEnvironment(env);
    m_process->setProcessChannelMode(QProcess::MergedChannels);
    connect(m_process, &QProcess::readyReadStandardOutput, this, [this]() {
        const QString text = QString::fromUtf8(m_process->readAllStandardOutput()).trimmed();
        if (!text.isEmpty()) {
            Q_EMIT progress(-1, text.section(QLatin1Char('\n'), -1));
        }
    });
    connect(m_process, &QProcess::finished, this, [this, omitOptional](int exitCode, QProcess::ExitStatus status) {
        if (status != QProcess::NormalExit || exitCode != 0) {
            finish(false, i18n("npm could not install %1 (exit code %2).", m_agent.value(QStringLiteral("name")).toString(), exitCode));
            return;
        }
        writeMarker(m_agent, QJsonObject{{QStringLiteral("omitOptional"), omitOptional}});
        // what npm downloaded is installed now; its cache is only disk
        QDir(dataRoot() + QStringLiteral("/npm-cache")).removeRecursively();
        finish(true, QString());
    });
    Q_EMIT progress(-1, omitOptional ? i18n("Installing %1, using the copy already on this computer", m_agent.value(QStringLiteral("name")).toString())
                                     : i18n("Installing %1", m_agent.value(QStringLiteral("name")).toString()));
    m_process->start(node, arguments);
}

void Installer::installArchive(const QJsonObject &binary)
{
    const QJsonObject target = binary.value(platformKey()).toObject();
    const QString url = target.value(QStringLiteral("archive")).toString();
    if (url.isEmpty()) {
        finish(false, i18n("%1 has no build for this system.", m_agent.value(QStringLiteral("name")).toString()));
        return;
    }
    const QString archive = m_dir + QLatin1Char('/') + QUrl(url).fileName();
    const QString expected = target.value(QStringLiteral("sha256")).toString().toLower();
    m_download = new FileDownloader(this);
    connect(m_download, &FileDownloader::progress, this, [this](qint64 received, qint64 total) {
        Q_EMIT progress(total > 0 ? int(received * 100 / total) : -1, i18n("Downloading %1", m_agent.value(QStringLiteral("name")).toString()));
    });
    connect(m_download, &FileDownloader::finished, this, [this, archive, expected, target](bool ok, const QString &error) {
        if (!ok) {
            finish(false, error.isEmpty() ? i18n("The download was stopped.") : error);
            return;
        }
        QFile file(archive);
        QCryptographicHash hash(QCryptographicHash::Sha256);
        if (!file.open(QIODevice::ReadOnly) || !hash.addData(&file) || (!expected.isEmpty() && QString::fromLatin1(hash.result().toHex()) != expected)) {
            file.remove();
            finish(false, i18n("The downloaded %1 does not match its checksum.", m_agent.value(QStringLiteral("name")).toString()));
            return;
        }
        file.close();
        std::unique_ptr<KArchive> unpacker;
        if (archive.endsWith(QLatin1String(".zip"))) {
            unpacker = std::make_unique<KZip>(archive);
        } else {
            unpacker = std::make_unique<KTar>(archive); // .tar.gz, .tgz, .tar.xz, .tar.bz2
        }
        if (!unpacker->open(QIODevice::ReadOnly) || !unpacker->directory()->copyTo(m_dir, true)) {
            finish(false, i18n("Could not unpack %1.", QFileInfo(archive).fileName()));
            return;
        }
        unpacker->close();
        QFile::remove(archive);
        QString cmd = target.value(QStringLiteral("cmd")).toString();
        if (cmd.startsWith(QLatin1String("./"))) {
            cmd = cmd.mid(2);
        }
        QFile program(QDir(m_dir).absoluteFilePath(cmd));
        program.setPermissions(program.permissions() | QFileDevice::ExeOwner | QFileDevice::ExeGroup | QFileDevice::ExeOther);
        writeMarker(m_agent, QJsonObject());
        finish(true, QString());
    });
    m_download->start(QUrl(url), archive);
}

void Installer::finish(bool ok, const QString &error)
{
    Q_EMIT finished(ok, error);
}

} // namespace AcpAgents
