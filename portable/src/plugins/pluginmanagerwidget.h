/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "pluginmanager.h"

#include <QWidget>

class KMessageWidget;
class QTableWidget;
class QPushButton;

/** @class PluginManagerWidget
    @brief Settings page for loading user plugins.

    Lets the user pick a folder or a `.wmplugin` archive; validates it and
    shows the metadata in a green banner — or the reasons it was rejected in a
    red one — before an Import button commits it. Below, the installed plugins
    are shown in a name/version table with Uninstall.
 */
class PluginManagerWidget : public QWidget
{
    Q_OBJECT
public:
    explicit PluginManagerWidget(QWidget *parent = nullptr);

private Q_SLOTS:
    void chooseArchive();
    void chooseFolder();
    void doImport();
    void cancelPreview();
    void uninstallSelected();
    void refreshList();

private:
    void preview(const QString &path);

    KMessageWidget *m_info = nullptr;
    QPushButton *m_importButton = nullptr;
    QPushButton *m_cancelButton = nullptr;
    QTableWidget *m_table = nullptr;
    QPushButton *m_uninstallButton = nullptr;
    PluginManager::ImportCandidate m_candidate;
};
