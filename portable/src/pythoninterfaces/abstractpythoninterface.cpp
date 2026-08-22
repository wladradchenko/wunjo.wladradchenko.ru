/*
    SPDX-FileCopyrightText: 2022 Julius Künzel <julius.kuenzel@kde.org>

    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "abstractpythoninterface.h"
#include "core.h"
#include "mainwindow.h"
#include "plugins/pluginmanager.h"

#include <KLocalizedString>
#include <KMessageBox>
#include <QAction>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QDebug>
#include <QGuiApplication>
#include <QMutex>
#include <QProcess>
#include <QStandardPaths>
#include <QtConcurrent/QtConcurrentRun>

static QMutex mutex;
static bool installInProgress;

PythonDependencyMessage::PythonDependencyMessage(QWidget *parent, AbstractPythonInterface *interface, bool setupErrorOnly)
    : KMessageWidget(parent)
    , m_interface(interface)
{
    setWordWrap(true);
    setCloseButtonVisible(false);
    if (m_interface->status() == AbstractPythonInterface::Installed) {
        hide();
    }
    setText(i18n("Check configuration"));
    m_installAction = new QAction(QIcon::fromTheme(QStringLiteral("download")), i18n("Install"), this);
    m_abortAction = new QAction(i18n("Abort installation"), this);
    addAction(m_installAction);
    connect(m_abortAction, &QAction::triggered, m_interface, &AbstractPythonInterface::abortScript);
    connect(m_interface, &AbstractPythonInterface::setupError, this, [&](const QString &message) {
        if (m_interface->m_installStatus == AbstractPythonInterface::NotInstalled) {
            removeAction(m_abortAction);
            m_installAction->setEnabled(true);
        }
        doShowMessage(message, KMessageWidget::Warning);
    });
    connect(m_interface, &AbstractPythonInterface::installStatusChanged, this, [&]() {
        switch (m_interface->status()) {
        case AbstractPythonInterface::Installed:
            hide();
            break;
        case AbstractPythonInterface::InProgress:
            setMessageType(KMessageWidget::Information);
            setText(i18n("Installing… please be patient, this can take several minutes"));
            m_installAction->setEnabled(false);
            addAction(m_abortAction);
            show();
            break;
        default:
            setMessageType(KMessageWidget::Warning);
            setText(i18n("Install plugin to use"));
            m_installAction->setEnabled(true);
            removeAction(m_abortAction);
            show();
            break;
        }
    });

    connect(m_interface, &AbstractPythonInterface::setupMessage, this,
            [&](const QString &message, int messageType) { doShowMessage(message, KMessageWidget::MessageType(messageType)); });

    if (!setupErrorOnly) {
        connect(m_interface, &AbstractPythonInterface::checkVersionsResult, this, [&](const QStringList &list) {
            removeAction(m_abortAction);
            if (list.isEmpty()) {
                if (m_interface->featureName().isEmpty()) {
                    doShowMessage(i18n("Everything is properly configured."), KMessageWidget::Positive);
                } else {
                    doShowMessage(i18n("%1 is properly configured.", m_interface->featureName()), KMessageWidget::Positive);
                }
            } else {
                if (m_interface->featureName().isEmpty()) {
                    doShowMessage(i18n("Everything is configured:<br>%1", list.join(QStringLiteral(", "))), KMessageWidget::Positive);
                } else {
                    doShowMessage(i18n("%1 is configured:<br>%2", m_interface->featureName(), list.join(QStringLiteral(", "))), KMessageWidget::Positive);
                }
            }
            m_installAction->setEnabled(false);
            Q_EMIT m_interface->venvSetupChanged();
        });

        connect(m_interface, &AbstractPythonInterface::dependenciesMissing, this, [&](const QStringList &messages) {
            m_installAction->setEnabled(true);
            removeAction(m_abortAction);
            m_installAction->setText(m_interface->installMessage());
            doShowMessage(messages.join(QStringLiteral("\n")), KMessageWidget::Warning);
        });

        connect(m_interface, &AbstractPythonInterface::proposeUpdate, this, [&](const QString &message) {
            // only allow upgrading python modules once
            m_installAction->setText(i18n("Check for update"));
            m_installAction->setEnabled(true);
            removeAction(m_abortAction);
            doShowMessage(message, KMessageWidget::Warning);
        });

        connect(m_interface, &AbstractPythonInterface::dependenciesAvailable, this, [&]() {
            if (!m_updated) {
                // only allow upgrading python modules once
                m_installAction->setEnabled(false);
                removeAction(m_abortAction);
            }
            if (text().isEmpty() && m_interface->status() == AbstractPythonInterface::Installed) {
                hide();
            }
        });

        connect(m_installAction, &QAction::triggered, this, [&]() {
            switch (m_interface->status()) {
            case AbstractPythonInterface::Unknown:
                m_interface->checkVenvConcurrently(false);
                break;
            case AbstractPythonInterface::NotInstalled:
            case AbstractPythonInterface::Broken:
                // Ask first, then leave: building the environment and fetching
                // the packages both take minutes, and neither belongs on the
                // thread that is supposed to be drawing the progress.
                if (m_interface->confirmInstall()) {
                    doShowMessage(i18n("Setting up the environment — this takes a few minutes…"), KMessageWidget::Information);
                    m_interface->checkVenvConcurrently(true);
                }
                break;
            case AbstractPythonInterface::MissingDependencies:
                if (m_interface->confirmInstall()) {
                    doShowMessage(i18n("Downloading the dependencies — this takes a few minutes…"), KMessageWidget::Information);
                    m_interface->installMissingDependencies();
                }
                break;
            case AbstractPythonInterface::InProgress:
                return;
            default:
                // upgrade
                if (m_updated) {
                    return;
                }
                m_updated = true;
                m_interface->updateDependencies();
                break;
            }
        });
    }
}

void PythonDependencyMessage::doShowMessage(const QString &message, KMessageWidget::MessageType messageType)
{
    if (message.isEmpty() && m_interface->status() == AbstractPythonInterface::Installed) {
        hide();
    } else {
        setMessageType(messageType);
        setText(message);
        show();
    }
}

void PythonDependencyMessage::checkAfterInstall()
{
    doShowMessage(i18n("Checking configuration…"), KMessageWidget::Information);
    // Listing what pip has installed shells out once per package; asking for it
    // synchronously froze the window right after a long install, which reads as
    // the application having died at the finish line.
    m_interface->checkDependenciesConcurrently();
}

QString AbstractPythonInterface::cudaRequirementsFor(const QString &baseRequirements)
{
    const double driver = PluginManager::driverCudaVersion();
    if (driver <= 0 || baseRequirements.isEmpty()) {
        return {}; // no usable card: the plain requirements bring a CPU torch
    }
    // Highest line the driver can still run. A wheel built for a newer CUDA
    // than the driver loads without complaint and then finds no device.
    const QDir folder = QFileInfo(baseRequirements).absoluteDir();
    const QStringList variants = folder.entryList({QStringLiteral("requirements-cuda*.txt")}, QDir::Files, QDir::Name);
    QString best;
    double bestVersion = 0;
    const QString prefix = QStringLiteral("requirements-cuda");
    double newestShipped = 0;
    for (const QString &variant : variants) {
        // requirements-cuda126.txt -> 126 -> 12.6
        QString digits = variant;
        digits.remove(0, prefix.length());
        digits.chop(4); // ".txt"
        bool ok = false;
        const int packed = digits.toInt(&ok);
        if (!ok || packed < 100) {
            continue;
        }
        const double version = packed / 10.0;
        newestShipped = qMax(newestShipped, version);
        if (version <= driver && version > bestVersion) {
            bestVersion = version;
            best = folder.absoluteFilePath(variant);
        }
    }

    // A driver newer than anything pinned here needs no pin at all: what pip
    // installs by default is built for the current CUDA, and forcing an older
    // line onto it would swap a working wheel for a slower one. The pins exist
    // for the machines the default has left behind, not for the ones ahead of
    // it.
    if (driver >= newestShipped && newestShipped > 0) {
        return {};
    }
    if (best.isEmpty()) {
        qWarning() << "no CUDA build here matches the driver's" << driver << "— falling back to the processor";
    }
    return best;
}

AbstractPythonInterface::AbstractPythonInterface(QObject *parent)
    : QObject{parent}
    , m_dependencies()
{
    addScript(QStringLiteral("checkpackages.py"));
    addScript(QStringLiteral("checkgpu.py"));
}

AbstractPythonInterface::~AbstractPythonInterface()
{
    qDebug() << ":::: DELETING ABSTRACT PYTHON INTERFACE.....";
    if (m_watcher.isRunning()) {
        m_watcher.waitForFinished();
    }
    if (m_depsWatcher.isRunning()) {
        m_depsWatcher.waitForFinished();
    }
    if (m_versionWatcher.isRunning()) {
        m_versionWatcher.waitForFinished();
    }
}

const QString AbstractPythonInterface::getVenvPath()
{
    return QStringLiteral("venv");
}

const QString AbstractPythonInterface::getVenvBinPath()
{
#ifdef Q_OS_WIN
    const QString pythonPath = QStringLiteral("%1/Scripts/").arg(getVenvPath());
#else
    const QString pythonPath = QStringLiteral("%1/bin/").arg(getVenvPath());
#endif
    return pythonPath;
}

void AbstractPythonInterface::rebuildVenv()
{
#ifdef Q_OS_WIN
    const QString pythonName = QStringLiteral("python");
#else
    const QString pythonName = QStringLiteral("python3");
#endif
    QDir pluginDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation));
    QFileInfo pyPath(pluginDir.absoluteFilePath(pythonName));
    if (!pyPath.exists()) { // && pyPath.isSymLink()) {
        // Recreate venv with updated python path
        QFile::remove(pyPath.absoluteFilePath());
        QProcess envProcess;
        if (pluginDir.cd(getVenvPath())) {
            QStringList args = {QStringLiteral("-m"), QStringLiteral("venv"), pluginDir.absolutePath(), QStringLiteral("--upgrade")};
            const QString pythonExec = systemPythonExec();
            envProcess.start(pythonExec, args);
            envProcess.waitForStarted();
            envProcess.waitForFinished(-1);
            if (envProcess.exitStatus() != QProcess::NormalExit) {
                Q_EMIT setupMessage(envProcess.readAllStandardError(), KMessageWidget::Warning);
            } else {
                Q_EMIT setupMessage(i18n("Successfully rebuilt environment"), KMessageWidget::Positive);
            }
        } else {
            qDebug() << ":::: CANNOT ENTER PLUGIN DIR: " << pluginDir.absoluteFilePath(getVenvPath());
        }
    }
}

void AbstractPythonInterface::deleteVenv()
{
    QDir pluginDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation));
    const QString binaryPath = getVenvPath();
    if (pluginDir.cd(binaryPath)) {
        if (pluginDir.dirName().contains(QLatin1String("venv")) && pluginDir.absolutePath().contains(QLatin1String("wunjo"))) {
            pluginDir.removeRecursively();
        }
        setStatus(NotInstalled);
        calculateVenvSize();
        Q_EMIT venvSetupChanged();
    }
}

AbstractPythonInterface::PythonExec AbstractPythonInterface::venvPythonExecs(bool checkPip)
{
    QDir pluginDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation));
    const QString binaryPath = getVenvBinPath();

    if (!pluginDir.cd(binaryPath)) {
        qDebug() << "Python venv binary folder" << binaryPath << "does not exist in" << pluginDir.absolutePath();
        return {};
    }

#ifdef Q_OS_WIN
    const QString pythonName = QStringLiteral("python");
    const QString pipName = QStringLiteral("pip");
#else
    const QString pythonName = QStringLiteral("python3");
    const QString pipName = QStringLiteral("pip3");
#endif

    const QStringList pythonPaths = {pluginDir.absolutePath()};

    QString pythonExe = QStandardPaths::findExecutable(pythonName, pythonPaths);
    if (pythonExe.isEmpty()) {
        // Try reinstalling the venv
        QFileInfo pyPath(pluginDir.absoluteFilePath(pythonName));
        if (pyPath.isSymLink()) {
            // To recreate the venv, python3 must be removed
            QFile::remove(pyPath.absoluteFilePath());
        }
        pluginDir.cdUp();
        QStringList args = {QStringLiteral("-m"), QStringLiteral("venv"), pluginDir.absolutePath(), QStringLiteral("--upgrade")};
        const QString sPython = systemPythonExec();
        Q_EMIT setupMessage(i18n("Corrupted plugin, trying to reinstall…"), KMessageWidget::Warning);
        QProcess envProcess;
        QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
        envProcess.setProcessEnvironment(env);
        envProcess.start(sPython, args);
        envProcess.waitForStarted();
        envProcess.waitForFinished(-1);
        if (envProcess.exitStatus() != QProcess::NormalExit) {
            Q_EMIT setupMessage(envProcess.readAllStandardError(), KMessageWidget::Warning);
        } else {
            Q_EMIT setupMessage(QString());
        }
    }
    pythonExe = QStandardPaths::findExecutable(pythonName, pythonPaths);
    QString pipExe;
    if (checkPip) {
        pipExe = QStandardPaths::findExecutable(pipName, pythonPaths);
    }

    return {pythonExe, pipExe};
}

QString AbstractPythonInterface::systemPythonExec()
{
#ifdef Q_OS_WIN
    const QString pythonName = QStringLiteral("python");
#else
    QString pythonName = QStringLiteral("python3");
    for (auto i = m_dependencies.cbegin(), end = m_dependencies.cend(); i != end; ++i) {
        QFile deps(i.key());
        if (deps.open(QIODevice::ReadOnly)) {
            QTextStream in(&deps);
            if (!in.atEnd()) {
                const QString line = in.readLine();
                if (line.startsWith(QStringLiteral("#python"))) {
                    QStringList compatiblePython = line.section(QLatin1Char('#'), 1).split(QLatin1Char(','), Qt::SkipEmptyParts);
                    for (auto &p : compatiblePython) {
                        // Next to the application first. On macOS the interpreter
                        // Craft ships sits in Contents/MacOS and is never on PATH,
                        // so a bundle carrying python3.11 still answered "Cannot
                        // find a compatible python version" and every plugin that
                        // needs Python was dead on arrival. It is also the build
                        // the wheels were resolved against, which makes it the
                        // right one to prefer even where PATH offers another.
                        QString compatPath = QStandardPaths::findExecutable(p, {QCoreApplication::applicationDirPath()});
                        if (compatPath.isEmpty()) {
                            compatPath = QStandardPaths::findExecutable(p);
                        }
                        if (!compatPath.isEmpty()) {
                            return compatPath;
                        }
                    }
                    setStatus(Broken);
                    Q_EMIT setupError(
                        i18n("Cannot find a compatible python version:\n%1\nPlease install it on your system.\n", line.section(QLatin1Char('#'), 1)));
                    return QString();
                    break;
                }
            }
        }
        deps.close();
    }
#endif
    const QString path = QStandardPaths::findExecutable(pythonName);
    if (path.isEmpty()) {
        setStatus(Broken);
        Q_EMIT setupError(i18n("Cannot find %1, please install it on your system.\n"
                               "If already installed, check it is installed in a directory "
                               "listed in PATH environment variable",
                               pythonName));
    }
    return path;
}

QString AbstractPythonInterface::uvExec()
{
    // Shipped beside the application binary; the PATH lookup covers the builds
    // that install it as an ordinary system package.
    static const QString path = []() {
        const QString bundled = QCoreApplication::applicationDirPath() + QStringLiteral("/uv")
#ifdef Q_OS_WIN
                                + QStringLiteral(".exe")
#endif
            ;
        if (QFileInfo::exists(bundled)) {
            return bundled;
        }
        return QStandardPaths::findExecutable(QStringLiteral("uv"));
    }();
    return path;
}

QString AbstractPythonInterface::uvCacheDir()
{
    const QString dir = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + QStringLiteral("/uv-cache");
    QDir().mkpath(dir);
    return dir;
}

bool AbstractPythonInterface::useSystemPython()
{
    return false;
}

bool AbstractPythonInterface::checkVenv(bool calculateSize, bool forceInstall)
{
    if (useSystemPython()) {
        return true;
    }
    if (installInProgress) {
        qDebug() << "...... ANOTHER INSTALL IN PROGRESS..";
        return false;
    }

    QMutexLocker bk(&mutex);

    QDir pluginDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation));
    QString binPath = getVenvBinPath();

    qDebug() << "++++++ \n\nCHECKING PYTHON PATH FROM: " << pluginDir.absolutePath() << ", FOLDER: " << binPath << ", FORCING INSTALL: " << forceInstall;

    if (pluginDir.exists(binPath)) {
        PythonExec execs = venvPythonExecs(true);
        if (execs.python.isEmpty() || !QFile::exists(execs.python)) {
            setStatus(Broken);
            Q_EMIT setupError(i18n("Cannot find python3 in the wunjo venv.\n"
                                   "Try to uninstall and then reinstall the plugin."));
            if (calculateSize) {
                Q_EMIT gotPythonSize(QString());
            }
            return false;
        }
        if (execs.pip.isEmpty() || !QFile::exists(execs.pip)) {
            Q_EMIT setupError(i18n("Cannot find pip3 in the wunjo venv.\n"
                                   "Try to uninstall and then reinstall the plugin."));
            if (calculateSize) {
                Q_EMIT gotPythonSize(QString());
            }
            setStatus(Broken);
            return false;
        }

        // Everything ok: calculate the size if requested and quit
        if (calculateSize) {
            calculateVenvSize();
        }
        if (!forceInstall) {
            setStatus(Installed);
            return true;
        }
    } else {
        // Venv folder not found
        setStatus(NotInstalled);
        if (calculateSize) {
            Q_EMIT gotPythonSize(QString());
        }
    }

    if (!forceInstall) {
        return false;
    }
    qDebug() << "================\n\nSTARTING INSTALL\n\n=========================";
    // Setup venv
    if (!setupVenv()) {
        // setup failed
        return false;
    }
    setStatus(InProgress);
    bk.unlock();
    return installMissingDependencies();
}

void AbstractPythonInterface::calculateVenvSize()
{
    QDir pluginDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation));
    if (pluginDir.cd(getVenvPath())) {
        QFuture<KIO::filesize_t> future = QtConcurrent::run(&MainWindow::fetchFolderSize, pCore->window(), pluginDir.absolutePath());
        QFutureWatcher<KIO::filesize_t> *watcher = new QFutureWatcher<KIO::filesize_t>(this);
        watcher->setFuture(future);
        connect(watcher, &QFutureWatcherBase::finished, this, [this, watcher] {
            KIO::filesize_t size = watcher->result();
            Q_EMIT gotPythonSize(KIO::convertSize(size));
            watcher->deleteLater();
        });
    } else {
        Q_EMIT gotPythonSize(QString());
    }
}

bool AbstractPythonInterface::checkSetup(bool requestInstall, bool *newInstall)
{
    PythonExec exes = venvPythonExecs(true);
    qDebug() << "::::: FOUND PYTHON EXECS: " << exes.python << exes.pip;
    if (!exes.python.isEmpty() && !exes.pip.isEmpty() && std::find(m_scripts.cbegin(), m_scripts.cend(), QString()) == m_scripts.cend()) {
        qDebug() << "//// SCRIPT VALUES: " << m_scripts.values();
        if (m_installStatus == Unknown) {
            setStatus(Installed);
        }
        return true;
    }
    if (!checkVenv(false, requestInstall)) {
        return false;
    }
    if (requestInstall) {
        *newInstall = true;
    }

    QMapIterator<QString, QString> i(m_scripts);
    while (i.hasNext()) {
        i.next();
        if (i.value().isEmpty()) {
            return false;
        }
    }
    return true;
}

bool AbstractPythonInterface::setupVenv()
{
    // First check if python and venv are available
    QString pythonExec = systemPythonExec();
    // Check that the system python is found
    if (pythonExec.isEmpty() || installInProgress) {
        if (m_installStatus != Broken) {
            Q_EMIT setupError(i18n("Cannot find system python"));
        }
        return false;
    }
    // Use system python to check for venv
    installInProgress = true;
    // Ensure the message is displayed before starting the busy work
    qApp->processEvents();
    QDir pluginDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation));
    pluginDir.mkpath(QStringLiteral("."));

    QProcess envProcess;
    // For some reason, this fails in AppImage, but when extracting the Appimage it works...
    // No workaround found yet for AppImage
    QStringList args = {QStringLiteral("-m"), QStringLiteral("venv"), pluginDir.absoluteFilePath(getVenvPath())};
    QString creator = pythonExec;
    const QString uv = uvExec();
    if (!uv.isEmpty()) {
        // Same environment, made by uv against the very same interpreter. It is
        // asked for the seed packages because everything downstream — the
        // dependency check, the settings pages — looks for a pip inside the
        // environment and would call it broken without one.
        creator = uv;
        args = {QStringLiteral("venv"), QStringLiteral("--seed"), QStringLiteral("--python"), pythonExec, pluginDir.absoluteFilePath(getVenvPath())};
        QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
        env.insert(QStringLiteral("UV_CACHE_DIR"), uvCacheDir());
        envProcess.setProcessEnvironment(env);
    }
    envProcess.start(creator, args);
    envProcess.waitForStarted();
    envProcess.waitForFinished(-1);
    if (envProcess.exitStatus() == QProcess::NormalExit) {
        PythonExec exes = venvPythonExecs(true);
        if (!exes.python.isEmpty() && !exes.pip.isEmpty()) {
            installPackage({QStringLiteral("importlib")});
            installInProgress = false;
            return true;
        }
    }
    m_installStatus = NotInstalled;
    // ERROR READ JOB OUTPUT
    QString errorLog = envProcess.readAllStandardOutput();
    errorLog.append(QStringLiteral("\n%1").arg(envProcess.readAllStandardError()));
    Q_EMIT setupError(i18n("Cannot create the python virtual environment:\n%1", errorLog));

    // Install failed, remove venv
    const QString venvPath = getVenvPath();
    if (pluginDir.cd(venvPath)) {
        if (pluginDir.dirName() == venvPath) {
            pluginDir.removeRecursively();
        }
    }
    installInProgress = false;
    Q_EMIT venvSetupChanged();
    return false;
}

const QString AbstractPythonInterface::locateScript(const QString &script)
{
    const QString path = QStandardPaths::locate(QStandardPaths::AppDataLocation, QStringLiteral("scripts/%1").arg(script));
    if (path.isEmpty()) {
        Q_EMIT setupError(i18n("The %1 script was not found, check your install.", script));
    }
    return path;
}

void AbstractPythonInterface::addDependency(const QString &pipname, const QString &purpose, bool optional)
{
    m_dependencies.insert(pipname, purpose);
    if (optional) {
        m_optionalDeps << pipname;
    }
}

void AbstractPythonInterface::addScript(const QString &script)
{
    m_scripts.insert(script, locateScript(script));
}

const QString AbstractPythonInterface::getScript(const QString &scriptName) const
{
    if (!m_scripts.contains(scriptName)) {
        return {};
    }
    return m_scripts.value(scriptName);
}

void AbstractPythonInterface::checkDependenciesConcurrently()
{
    m_depsJob = QtConcurrent::run(&AbstractPythonInterface::checkDependencies, this, false, false);
    m_depsWatcher.setFuture(m_depsJob);
}

void AbstractPythonInterface::checkVersionsConcurrently()
{
    m_versionJob = QtConcurrent::run(&AbstractPythonInterface::checkVersions, this, true);
    m_versionWatcher.setFuture(m_versionJob);
}

bool AbstractPythonInterface::checkDependencies(bool force, bool async)
{
    if (m_installStatus == InProgress || (!force && m_dependenciesChecked)) {
        // Don't check twice if dependencies are satisfied
        return true;
    }
    // Force check, reset flag
    m_missing.clear();
    m_optionalMissing.clear();
    // Check if venv is correctly installed
    PythonExec exes = venvPythonExecs(true);
    if (exes.python.isEmpty() || exes.pip.isEmpty()) {
        // Check if venv folder exists
        QDir pluginDir(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation));
        const QString binaryPath = getVenvPath();
        if (!pluginDir.exists(binaryPath)) {
            setStatus(NotInstalled);
            return false;
        }
        setStatus(Broken);
        return false;
    }
    const QString output = runPackageScript(QStringLiteral("--check"), false, false, force);
    QStringList missingDeps = output.split(QStringLiteral("Missing: "), Qt::SkipEmptyParts);
    QStringList outputMissing;
    for (auto &m : missingDeps) {
        outputMissing << m.simplified();
    }
    const QStringList deps = parseDependencies(m_dependencies.keys(), true);
    QStringList messages;
    if (!output.isEmpty()) {
        // We have missing dependencies
        for (const auto &i : deps) {
            if (outputMissing.contains(i)) {
                if (m_optionalDeps.contains(i)) {
                    m_optionalMissing << i;
                    continue;
                }
                m_missing.append(i);
                messages.append(xi18n("The <application>%1</application> python module is required.", i));
            }
        }
    }
    m_dependenciesChecked = true;
    if (messages.isEmpty()) {
        if (async) {
            checkVersionsConcurrently();
        } else {
            checkVersions(true);
        }
        Q_EMIT dependenciesAvailable();
    } else {
        messages.prepend(i18n("Using python from %1", exes.python));
        Q_EMIT dependenciesMissing(messages);
    }
    return false;
}

QStringList AbstractPythonInterface::missingDependencies(const QStringList &filter)
{
    if (filter.isEmpty()) {
        return m_missing;
    }
    QStringList filtered;
    for (const auto &item : filter) {
        if (m_missing.contains(item)) {
            filtered.append(item);
        }
    }
    return filtered;
};

bool AbstractPythonInterface::confirmInstall()
{
    if (KMessageBox::warningContinueCancel(pCore->window(),
                                           i18n("This requires an internet connection and will take several minutes\nto download all necessary "
                                                "dependencies. After that all processing will happen offline.")) != KMessageBox::Continue) {
        setStatus(NotInstalled);
        return false;
    }
    return true;
}

void AbstractPythonInterface::checkVenvConcurrently(bool forceInstall)
{
    (void)QtConcurrent::run([this, forceInstall]() { checkVenv(false, forceInstall); });
}

bool AbstractPythonInterface::installMissingDependencies()
{
    setStatus(InProgress);
    runPackageScript(QStringLiteral("--install"), true);
    return true;
}

void AbstractPythonInterface::updateDependencies()
{
    runPackageScript(QStringLiteral("--upgrade"), true);
}

void AbstractPythonInterface::runConcurrentScript(const QString &script, QStringList args, bool feedback)
{
    if (m_dependencies.keys().isEmpty()) {
        qWarning() << "No dependencies specified";
        Q_EMIT setupError(i18n("Installation Issue: Cannot find dependency list for %1", featureName()));
        return;
    }
    if (!checkSetup()) {
        qWarning() << "setup error for script: " << script;
        return;
    }
    m_scriptJob = QtConcurrent::run(&AbstractPythonInterface::runScript, this, script, args, QString(), true, feedback);
    m_watcher.setFuture(m_scriptJob);
}

void AbstractPythonInterface::proposeMaybeUpdate(const QString &dependency, const QString &minVersion)
{
    checkVersions(false);
    const QString currentVersion = m_versions.value(dependency);
    if (currentVersion.isEmpty()) {
        Q_EMIT setupError(i18n("Error while checking version of module %1", dependency));
        return;
    }
    if (versionToInt(currentVersion) < versionToInt(minVersion)) {
        Q_EMIT proposeUpdate(i18n("At least version %1 of module %2 is required, "
                                  "but your current version is %3",
                                  minVersion, dependency, currentVersion));
    } else {
        Q_EMIT proposeUpdate(i18n("Please consider to update your setup."));
    }
}

int AbstractPythonInterface::versionToInt(const QString &version)
{
    QStringList v = version.split(QStringLiteral("."));
    return QT_VERSION_CHECK(v.length() > 0 ? v.at(0).toInt() : 0, v.length() > 1 ? v.at(1).toInt() : 0, v.length() > 2 ? v.at(2).toInt() : 0);
}

void AbstractPythonInterface::checkVersions(bool signalOnResult)
{
    QStringList deps = parseDependencies(m_dependencies.keys(), true);
    const QStringList output = runPackageScript(QStringLiteral("--details"), false, false).split(QLatin1Char('\n'));
    QMutexLocker locker(&m_versionsMutex);
    QStringList versionsText;
    for (auto &o : output) {
        if (o.contains(QLatin1String("=="))) {
            const QString package = o.section(QLatin1String("=="), 0, 0).toLower();
            if (deps.contains(package)) {
                QString version = o.section(QLatin1String("=="), 1);
                if (version == QLatin1String("missing")) {
                    version = i18nc("@item:intext indicates a missing dependency", "missing (optional)");
                }
                m_versions.insert(package, version);
                versionsText.append(QStringLiteral("<b>%1</b> %2").arg(package, version));
            }
        }
    }
    if (m_versions.isEmpty()) {
        Q_EMIT setupMessage(i18nc("@label:textbox", "No version information available."), KMessageWidget::Warning);
        qDebug() << "::: CHECKING DEPENDENCIES... NO VERSION INFO AVAILABLE";
        return;
    }
    if (signalOnResult) {
        Q_EMIT checkVersionsResult(versionsText);
    }
}

QStringList AbstractPythonInterface::parseDependencies(const QStringList deps, bool split)
{
    // Extract requirements files
    if (split == false) {
        return deps;
    }
    QStringList packages;
    QStringList reqFiles;
    for (auto &d : deps) {
        if (d.endsWith(QLatin1String(".txt"))) {
            reqFiles << d;
        } else {
            packages << d;
        }
    }
    for (auto &r : reqFiles) {
        // This is a requirements.txt file, read content
        QFile textFile(r);
        if (textFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
            QTextStream textStream(&textFile);
            while (!textStream.atEnd()) {
                QString line = textStream.readLine();
                if (line.simplified().isEmpty())
                    continue;
                else if (line.contains(QLatin1Char(';')))
                    continue;
                else if (line.simplified().startsWith(QLatin1Char('-')))
                    // pip options (--index-url, --extra-index-url, -r …) steer the
                    // install; they are not packages and must never be checked for
                    continue;
                else if (!line.startsWith(QLatin1Char('#'))) {
                    if (line.contains(QLatin1Char('>'))) {
                        line = line.section(QLatin1Char('>'), 0, 0);
                    } else if (line.contains(QLatin1Char('<'))) {
                        line = line.section(QLatin1Char('<'), 0, 0);
                    } else if (line.contains(QLatin1Char('='))) {
                        line = line.section(QLatin1Char('='), 0, 0);
                    }
                    if (line.contains(QLatin1Char('#'))) {
                        line = line.section(QLatin1Char('#'), 1).simplified();
                    }
                    packages.append(line.simplified());
                }
            }
        }
    }
    return packages;
}

QString AbstractPythonInterface::runPackageScript(QString mode, bool concurrent, bool displayFeedback, bool forceInstall)
{
    if (m_dependencies.keys().isEmpty()) {
        qWarning() << "No dependencies specified";
        Q_EMIT setupError(i18n("Installation Issue: Cannot find dependency list for %1", featureName()));
        return {};
    }
    qDebug() << "=== CHECKING SETUP...";
    bool newInstall = false;
    if (!checkSetup(forceInstall, &newInstall)) {
        qDebug() << "=== CHECKING SETUP...NO!!!";
        return {};
    }
    qDebug() << "=== CHECKING SETUP...OK";
    bool installAction = mode.contains(QLatin1String("install")) || mode == QLatin1String("--upgrade");
    QStringList deps = parseDependencies(m_dependencies.keys(), !installAction);

    if (concurrent) {
        (void)QtConcurrent::run(&AbstractPythonInterface::runScript, this, QStringLiteral("checkpackages.py"), deps, mode, concurrent, displayFeedback);
        return {};
    } else {
        return runScript(QStringLiteral("checkpackages.py"), deps, mode, concurrent, displayFeedback);
    }
}

bool AbstractPythonInterface::installRequirements(QString reqFile)
{
    if (!QFile::exists(reqFile)) {
        Q_EMIT setupError(i18n("Installation issue: Cannot find %1 for %2", reqFile, featureName()));
        return false;
    }

    bool newInstall = false;
    if (!checkSetup(false, &newInstall)) {
        qDebug() << "=== CHECKING SETUP...NO!!!";
        return {};
    }
    qDebug() << "=== CHECKING SETUP...OK";

    const QStringList deps = {reqFile};
    (void)QtConcurrent::run(&AbstractPythonInterface::runScript, this, QStringLiteral("checkpackages.py"), deps, QStringLiteral("--force-install"), true, true);
    return true;
}

QString AbstractPythonInterface::installPackage(const QStringList packageNames)
{
    if (!checkSetup()) {
        return {};
    }
    return runScript(QStringLiteral("checkpackages.py"), packageNames, "--install", false, true);
}

const QStringList AbstractPythonInterface::listDependencies()
{
    return parseDependencies(m_dependencies.keys(), true);
}

QString AbstractPythonInterface::runScript(const QString &script, QStringList args, const QString &firstarg, bool concurrent, bool packageFeedback)
{
    const QString scriptpath = m_scripts.value(script);
    qDebug() << "=== CHECKING RUNNING SCRIPT: " << scriptpath;
    const QString pythonExe = venvPythonExecs().python;
    if (pythonExe.isEmpty()) {
        Q_EMIT setupError(i18n("Python exec not found"));
        return {};
    }
    if (scriptpath.isEmpty()) {
        Q_EMIT setupError(i18n("Failed to find script file %1", script));
        return {};
    }

    bool installAction = firstarg.contains(QLatin1String("install")) || firstarg == QLatin1String("--upgrade");

    if (concurrent && installAction) {
        Q_EMIT scriptStarted();
    }
    args = parseDependencies(args, !installAction);
    if (!firstarg.isEmpty()) {
        args.prepend(firstarg);
    }
    args.prepend(scriptpath);
    QProcess scriptJob;
    connect(this, &AbstractPythonInterface::abortScript, &scriptJob, &QProcess::kill, Qt::DirectConnection);
    if (installAction) {
        setStatus(InProgress);
        // checkpackages.py installs with whichever of the two it is handed. uv
        // is passed by path rather than found by the script: it belongs to this
        // application, not to whatever happens to be on the user's PATH.
        const QString uv = uvExec();
        if (!uv.isEmpty()) {
            QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
            env.insert(QStringLiteral("WUNJO_UV"), uv);
            env.insert(QStringLiteral("UV_CACHE_DIR"), uvCacheDir());
            scriptJob.setProcessEnvironment(env);
        }
    }

    if (packageFeedback) {
        if (concurrent) {
            scriptJob.setProcessChannelMode(QProcess::MergedChannels);
        }
        // The context is the process, not this: this object lives in the main
        // thread while the process lives in whichever thread runs the script, so
        // naming `this` made the handler run in the main thread and read the
        // same QProcess the worker was about to read below. Two threads in one
        // QIODevice buffer is undefined behaviour, and it showed: the ring
        // buffer would spin forever with the window frozen, or fall over
        // outright. It survived pip only because output arrived rarely enough
        // for the two rarely to meet; uv writes often, and it met every time.
        connect(&scriptJob, &QProcess::readyReadStandardOutput, &scriptJob, [this, &scriptJob]() {
            const QString processData = QString::fromUtf8(scriptJob.readAllStandardOutput());
            if (!processData.isEmpty()) {
                Q_EMIT installFeedback(processData.simplified());
            }
        });
    }

    scriptJob.start(pythonExe, args);
    // Don't timeout
    qDebug() << "::: RUNNING SCRIPT: " << pythonExe << " = " << args;
    scriptJob.waitForFinished(-1);

    if (scriptJob.exitStatus() != QProcess::NormalExit || scriptJob.exitCode() != 0) {
        const QString errorMessage = concurrent ? scriptJob.readAllStandardOutput() : scriptJob.readAllStandardError();
        Q_EMIT setupError(i18n("Error while running python3 script:\n %1\n%2", scriptpath, errorMessage));
        if (installAction) {
            setStatus(Broken);
        }
        return {};
    }

    if (concurrent && !packageFeedback) {
        const QString processData = QString::fromUtf8(scriptJob.readAllStandardOutput());
        Q_EMIT scriptFeedback(script, args, processData.split(QLatin1Char('\n'), Qt::SkipEmptyParts));
    }
    qDebug() << "// RAN CONCURRENT SCRIPT: " << concurrent << ", SCRIPT: " << script << ", firstARG: " << firstarg;
    if (installAction) {
        setStatus(Installed);
    }
    if (script.contains(QLatin1String("checkgpu.py")) || script.contains(QLatin1String("whisperquery")) || script.contains(QLatin1String("seamlessquery"))) {
        Q_EMIT concurrentScriptFinished(script, args);
    } else if (concurrent && installAction) {
        Q_EMIT scriptFinished(args);
    }
    return scriptJob.readAllStandardOutput();
}

bool AbstractPythonInterface::installInProcess() const
{
    return installInProgress;
}

bool AbstractPythonInterface::optionalDependencyAvailable(const QString &dependency) const
{
    return !m_optionalMissing.contains(dependency);
}

const QString AbstractPythonInterface::installMessage() const
{
    return i18n("Install missing dependencies");
}

void AbstractPythonInterface::setStatus(AbstractPythonInterface::InstallStatus status)
{
    if (m_installStatus != status) {
        m_installStatus = status;
        Q_EMIT installStatusChanged();
    }
}

AbstractPythonInterface::InstallStatus AbstractPythonInterface::status() const
{
    return m_installStatus;
}
