/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "pluginmanifest.h"

#include <QElapsedTimer>
#include <QList>
#include <QWidget>

class KMessageWidget;
class FileDownloadJob;
class PluginPythonEnv;
class QLabel;
class QComboBox;
class QLineEdit;
class QPlainTextEdit;
class QPushButton;
class QTimer;

/** @class PluginSettingsTab
    @brief Settings page for a single installed plugin.

    Mirrors the built-in Speech To Text / Object Detection tabs: it exposes the
    setup a plugin declares in its manifest — the Python environment (install /
    size / delete, via PluginPythonEnv + the shared PythonDependencyMessage
    banner), model downloads, and the API key for `api` plugins — plus Uninstall.
    Sections render only when the manifest declares them.
 */
class PluginSettingsTab : public QWidget
{
    Q_OBJECT
public:
    explicit PluginSettingsTab(const PluginManifest &manifest, QWidget *parent = nullptr);

private Q_SLOTS:
    void saveKey();
    void downloadModel(int index);
    void refreshModels();
    void deleteAllModels();
    /** @brief Remove only the installed environment (venv), like the built-in
     *  Object Detection tab's "Uninstall plugin". The plugin files stay; the
     *  whole plugin is removed via "Delete plugin" in the Load Plugins list. */
    void uninstallEnvironment();
    /** @brief Pick the CUDA line by hand and reinstall against it — for the
     *  machine whose card the automatic choice failed to use. */
    void chooseCudaVariant();
    /** @brief Fill the device list from what torch reports inside this
     *  plugin's environment. */
    void gotDeviceList(const QString &script, const QStringList &args, const QStringList &jobData);
    /** @brief Keep the last line pip printed, so a ten-minute install shows what
     *  it is doing instead of one frozen sentence. */
    void showInstallFeedback(const QString &feedback);
    /** @brief Redraw "3:41 · Downloading torch…" — also on the ticking timer, so
     *  the elapsed time moves even while pip is silent. */
    void updateInstallLine();

private:
    /** @brief Write one chunk of pip output into the install log, the way a
     *  terminal would: package lines stack up, the download progress overwrites
     *  itself on a single line. */
    void appendInstallLog(const QString &chunk);
    /** @brief Add one row to the log, replacing the previous one when that was a
     *  progress reading the new row supersedes. */
    void appendLogRow(const QString &text, bool isProgress);

    PluginManifest m_manifest;
    PluginPythonEnv *m_env = nullptr;
    QLineEdit *m_keyEdit = nullptr;
    /** @brief Which device the plugin should run its model on; empty data means
     *  "decide at run time". */
    QComboBox *m_deviceCombo = nullptr;
    QLabel *m_venvSize = nullptr;
    QPushButton *m_uninstallEnv = nullptr;
    /** @brief The environment banner — during an install it carries the live
     *  progress instead of one static sentence. */
    KMessageWidget *m_installBanner = nullptr;
    /** @brief Everything pip said, like the built-in Speech To Text / Object
     *  Detection tabs show it: hidden until an install writes its first line. */
    QPlainTextEdit *m_installLog = nullptr;
    /** @brief The last log row is a download progress reading, so the next one
     *  replaces it instead of stacking below it. */
    bool m_logEndsWithProgress = false;
    QTimer *m_installTimer = nullptr;
    QElapsedTimer m_installElapsed;
    QString m_lastInstallLine;
    struct ModelRow {
        QLabel *status = nullptr;
        QPushButton *button = nullptr;
        /** @brief The download this row started, while it runs: the button
         *  cancels it, the row shows a short reading of it, and the periodic
         *  refresh keeps its hands off. The window belongs to the job tracker. */
        FileDownloadJob *download = nullptr;
    };
    QList<ModelRow> m_modelRows;
    /** @brief Anything too long for a row: why a download will not start, what
     *  an unpack choked on, which weight arrived damaged. */
    KMessageWidget *m_modelsMessage = nullptr;
};
