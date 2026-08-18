/*
    SPDX-FileCopyrightText: 2024 Jean-Baptiste Mardelle <jb@kdenlive.org>

    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "saminterface.h"
#include "wunjosettings.h"

#include <KIO/Global>
#include <KLocalizedString>

#include <QApplication>
#include <QDebug>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QLabel>
#include <QListWidget>
#include <QListWidgetItem>
#include <QPushButton>
#include <QStandardPaths>
#include <QVBoxLayout>

SamInterface::SamInterface(QObject *parent)
    : AbstractPythonInterface(parent)
{
    QString scriptPath = QStandardPaths::locate(QStandardPaths::AppDataLocation, QStringLiteral("scripts/automask/requirements-sam.txt"));
    if (!scriptPath.isEmpty()) {
        m_dependencies.insert(scriptPath, QString());
        // The torch this driver can actually run. Segmentation on the processor
        // is the difference between a wait and an afternoon, and picking the
        // wrong wheel fails silently — see cudaRequirementsFor.
        const QString cuda = cudaRequirementsFor(scriptPath);
        if (!cuda.isEmpty()) {
            m_dependencies.insert(cuda, QString());
        }
    }
    addScript(QStringLiteral("automask/sam-objectmask.py"));
}

const QString SamInterface::modelFolder(bool)
{
    // Object Detection is a plugin in everything but the manifest, so its
    // weights live where a plugin's do — plugins/<id>/models — instead of a
    // "sam2models" folder of its own beside them. Its environment already
    // follows the same rule: venv-sam is venv-<id>.
    const QString dir = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + QStringLiteral("/plugins/sam/models");
    QDir().mkpath(dir);
    // Weights fetched by an earlier version sit in the old folder; move them
    // over instead of asking for the same gigabytes again.
    const QString legacy = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + QStringLiteral("/sam2models");
    QDir legacyDir(legacy);
    if (legacyDir.exists()) {
        const QStringList models = legacyDir.entryList({QStringLiteral("*.pt")}, QDir::Files);
        for (const QString &model : models) {
            QFile::rename(legacyDir.absoluteFilePath(model), dir + QLatin1Char('/') + model);
        }
        // only when nothing of ours is left behind — never take a folder with it
        legacyDir.rmdir(legacy);
    }
    return dir;
}

const QStringList SamInterface::getInstalledModels()
{
    QString modelDirectory = modelFolder();
    if (modelDirectory.isEmpty()) {
        qDebug() << "=== /// CANNOT ACCESS SPEECH DICTIONARIES FOLDER";
        return {};
    }
    QDir modelsFolder(modelDirectory);
    QStringList installedModels;
    QStringList files = modelsFolder.entryList({QStringLiteral("*.pt")}, QDir::Files);
    for (auto &f : files) {
        installedModels << modelsFolder.absoluteFilePath(f);
    }
    return installedModels;
}

bool SamInterface::installNewModel(const QString &)
{
    return false;
}

QString SamInterface::featureName()
{
    return i18n("Object Segmentation (SAM2)");
}

QString SamInterface::subtitleScript()
{
    return QString();
}

QString SamInterface::speechScript()
{
    return QString();
}

const QString SamInterface::getVenvPath()
{
    return QStringLiteral("venv-sam");
}

const QString SamInterface::configForModel()
{
    KConfig conf(QStringLiteral("sammodelsinfo.rc"), KConfig::CascadeConfig, QStandardPaths::AppDataLocation);
    KConfigGroup group(&conf, QStringLiteral("models"));
    QMap<QString, QString> values = group.entryMap();
    QMapIterator<QString, QString> i(values);
    while (i.hasNext()) {
        i.next();
        if (QFileInfo(i.value()).completeBaseName() == QFileInfo(WunjoSettings::samModelFile()).completeBaseName()) {
            return i.key();
        }
    }
    return QString();
}

AbstractPythonInterface::PythonExec SamInterface::venvPythonExecs(bool checkPip)
{
    if (WunjoSettings::sam_system_python()) {
        // Use system python for SAM plugin
#ifdef Q_OS_WIN
        const QString pythonName = QStringLiteral("python");
        const QString pipName = QStringLiteral("pip");
#else
        const QString pythonName = QStringLiteral("python3");
        const QString pipName = QStringLiteral("pip3");
#endif
        const QStringList pythonPaths = {QFileInfo(WunjoSettings::sam_system_python_path()).dir().absolutePath()};
        const QString pythonExe = QStandardPaths::findExecutable(pythonName, pythonPaths);
        QString pipExe;
        if (checkPip) {
            pipExe = QStandardPaths::findExecutable(pipName, pythonPaths);
        }
        return {pythonExe, pipExe};
    }
    return AbstractPythonInterface::venvPythonExecs(checkPip);
}

bool SamInterface::useSystemPython()
{
    return WunjoSettings::sam_system_python();
}

bool SamInterface::installRequirements(QString reqFile)
{
    QString scriptPath = QStandardPaths::locate(QStandardPaths::AppDataLocation, QStringLiteral("scripts/automask/%1").arg(reqFile));
    qDebug() << "::: FOUND REQPATH: " << scriptPath;
    if (!scriptPath.isEmpty()) {
        return AbstractPythonInterface::installRequirements(scriptPath);
    }
    return false;
}
