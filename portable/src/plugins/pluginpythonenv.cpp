/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "pluginpythonenv.h"

#include "pluginmanager.h"

PluginPythonEnv::PluginPythonEnv(const PluginManifest &manifest, QObject *parent)
    : AbstractPythonInterface(parent)
{
    m_featureName = manifest.name();
    // Private by default; a plugin uses the shared env only if it says so.
    m_venvPath = manifest.venv() == QLatin1String("shared") ? QStringLiteral("venv") : QStringLiteral("venv-") + manifest.id();
    if (manifest.hasDependencies()) {
        // requirements-file-as-dependency-key: the base class parses the file
        // and honours a leading "#python3.x,..." interpreter-pin line, exactly
        // like SamInterface. Which file, though, depends on the machine: a
        // plugin pins the versions it works with per CUDA line, and the driver
        // decides which of them can actually run here.
        const QString file = manifest.requirementsFor(PluginManager::driverCudaVersion());
        m_dependencies.insert(manifest.rootDir() + QLatin1Char('/') + file, QString());
    }
}

const QString PluginPythonEnv::getVenvPath()
{
    return m_venvPath;
}

QString PluginPythonEnv::featureName()
{
    return m_featureName;
}
