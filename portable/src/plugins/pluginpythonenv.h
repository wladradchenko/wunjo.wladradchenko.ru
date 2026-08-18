/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "pluginmanifest.h"
#include "pythoninterfaces/abstractpythoninterface.h"

/** @class PluginPythonEnv
    @brief The Python environment of one installed plugin.

    A thin AbstractPythonInterface driven by the plugin's manifest, so a plugin
    reuses the exact venv machinery the built-in Speech/SAM tabs use: the
    PythonDependencyMessage banner, venv size, install and delete. The venv is
    private (`venv-<id>`) unless the manifest opts into the shared `venv`.
 */
class PluginPythonEnv : public AbstractPythonInterface
{
    Q_OBJECT
public:
    explicit PluginPythonEnv(const PluginManifest &manifest, QObject *parent = nullptr);
    const QString getVenvPath() override;
    QString featureName() override;

private:
    QString m_venvPath;
    QString m_featureName;
};
