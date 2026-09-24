/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "pluginmanifest.h"

#include <QList>
#include <QPair>
#include <QString>

class QWidget;

/** @class PluginAboutDialog
    @brief "About <plugin>" — KDE's own About window, filled from a manifest.

    A plugin's settings page is for settings; what the plugin is, how it is used
    and who wrote it belong behind the information button. Rather than imitating
    the application's About box, this builds a KAboutData out of the manifest and
    hands it to KAboutApplicationDialog, so the two windows are the same window
    with different contents — same header, same author row with its mail button.
 */
namespace PluginAboutDialog
{
/** @brief Open it, modally, for @p manifest. */
void show(const PluginManifest &manifest, QWidget *parent = nullptr);

/** @brief The instructions, as title/detail pairs.
 *
 * Written from the manifest rather than by each author: the editor is what
 * decides how a plugin is reached — a clip's context menu, an effect, the chat
 * — so it is also what can describe it without every plugin repeating it (and
 * getting it wrong when the editor changes).
 */
QList<QPair<QString, QString>> usageSteps(const PluginManifest &manifest);
} // namespace PluginAboutDialog
