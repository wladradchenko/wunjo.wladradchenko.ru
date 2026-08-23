/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "mcppythonenv.h"

#include "plugins/pluginmanager.h"

#include <KLocalizedString>

McpPythonEnv::McpPythonEnv(QObject *parent)
    : AbstractPythonInterface(parent)
{
    // requirements-file-as-dependency-key, as PluginPythonEnv does it: the base
    // class parses the file itself. There is only one line in it to parse.
    const QString server = PluginManager::mcpServerDir();
    if (!server.isEmpty()) {
        m_dependencies.insert(server + QStringLiteral("/requirements.txt"), QString());
    }
}

const QString McpPythonEnv::getVenvPath()
{
    return venvName();
}

QString McpPythonEnv::featureName()
{
    return i18n("MCP");
}

QString McpPythonEnv::venvName()
{
    return QStringLiteral("venv-mcp");
}

QString McpPythonEnv::python()
{
    return PluginManager::venvPython(venvName());
}
