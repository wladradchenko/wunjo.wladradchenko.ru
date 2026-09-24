/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "pythoninterfaces/abstractpythoninterface.h"

/** @class McpPythonEnv
    @brief The environment the MCP server runs in.

    Its own (`venv-mcp`), and deliberately not a plugin's. The server used to
    borrow the interpreter of whichever plugin declared `target: agent`, which
    tied driving the editor from Claude Code to having installed a local model —
    a model the outside agent never uses. Worse, "whichever plugin" stopped
    meaning anything once there could be several: the server would have taken
    the environment of an arbitrary one.

    It declares one dependency, `mcp`, which brings a modest tree behind it
    (httpx, jsonschema, cryptography and the compiled `cffi` backend among
    them) — tens of megabytes, not the gigabytes of a model, and nothing
    platform-specific to decide. It is built on its own settings tab like every
    other environment here. A local model is a separate plugin with a separate
    environment, and either end can be absent.
 */
class McpPythonEnv : public AbstractPythonInterface
{
    Q_OBJECT
public:
    explicit McpPythonEnv(QObject *parent = nullptr);
    const QString getVenvPath() override;
    QString featureName() override;

    /** @brief venv-mcp's interpreter, or empty while it has not been built. */
    static QString python();
    /** @brief The name of the environment, shared with the launcher script. */
    static QString venvName();
};
