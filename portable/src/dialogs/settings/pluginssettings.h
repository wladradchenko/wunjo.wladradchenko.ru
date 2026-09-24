/*
 *    SPDX-FileCopyrightText: 2024 Jean-Baptiste Mardelle <jb@kdenlive.org>
 *
 * SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
 */

#pragma once

#include "pythoninterfaces/speechtotext.h"

#include "ui_configspeech_ui.h"
#include <QListWidget>
#include <QWidget>

class SamInterface;
class McpPythonEnv;
class PythonDependencyMessage;
class KJob;
class QLineEdit;
class QListWidget;

class SpeechList : public QListWidget
{
    Q_OBJECT

public:
    SpeechList(QWidget *parent = nullptr);

protected:
    QStringList mimeTypes() const override;
    void dropEvent(QDropEvent *event) override;

Q_SIGNALS:
    void getDictionary(const QUrl url);
};

class PluginsSettings : public QWidget, public Ui::ConfigSpeech_UI
{
    Q_OBJECT

public:
    PluginsSettings(QWidget *parent = nullptr);
    ~PluginsSettings() override;
    /** @brief Launch pytonh scripts to check speech engine dependencies */
    void checkSpeechDependencies();
    void applySettings();
    void setActiveTab(int index);
    /** @brief Raise the tab of plugin @p pluginId, if it is installed. Sending
     *  someone to "the plugins page" and letting them hunt for the right tab is
     *  the sort of instruction that gets abandoned halfway. */
    void showPluginTab(const QString &pluginId);

private:
    SpeechToText *m_sttVosk;
    SpeechToText *m_sttWhisper;
    PythonDependencyMessage *m_msgWhisper;
    PythonDependencyMessage *m_msgVosk;
    PythonDependencyMessage *m_pythonSamLabel;
    SamInterface *m_samInterface;
    /** @brief The MCP server's environment. Its own, and not a model plugin's:
     *  driving the editor from an outside agent must not wait on weights that
     *  agent never loads. Every way of talking goes through this one server. */
    McpPythonEnv *m_mcpEnv;
    PythonDependencyMessage *m_msgMcp;
    SpeechList *m_speechListWidget;
    QAction *m_downloadModelAction;
    /** @brief The model offered when none is installed: turbo on a card, small
     *  on a processor, where anything bigger is slower than realtime. */
    QString m_recommendedModel;
    /** @brief One settings tab per installed plugin, rebuilt on change. */
    QList<QWidget *> m_pluginTabs;
    void rebuildPluginTabs();

    // ── navigation ──────────────────────────────────────────────────────
    // The pages still live in `tabWidget`; what changed is how one is chosen.
    // Its tab bar is hidden and this list drives it instead, because the bar
    // held two different kinds of thing — sections and plugins — and overflowed
    // once there were more than a handful.
    QListWidget *m_navList{nullptr};
    QLineEdit *m_navSearch{nullptr};
    QWidget *m_chipRow{nullptr};
    /** @brief The chosen `target` filter, empty for "All". */
    QString m_navFilter;
    /** @brief Ask for a plugin file, show what is in it, and install it if the
     *  person says so. There is no page behind this: picking a file and
     *  approving what was found are two moments, not a place to be. */
    void addPluginFromFile();
    /** @brief Build the search field, filters, import button and list. */
    void buildNavigation();
    /** @brief Refill the list from whatever pages `tabWidget` now holds. */
    void rebuildPluginList();
    /** @brief Show or hide rows to match the search text and the filter. */
    void applyNavFilter();
    /** @brief Right-click on an installed plugin: open its folder, or delete
     *  it. Both live here and nowhere else — the plugin's own page is about
     *  setting it up, not about getting rid of it. */
    void showPluginRowMenu(const QPoint &pos);

    /** @brief Check folder size */
    void checkWhisperFolderSize();
    /** @brief Refresh the list of available models in combobox */
    void reloadWhisperModels();
    /** @brief Check folder size */
    void checkSamFolderSize();
    /** @brief Allow installing specific cuda version */
    void checkCuda(bool isSam);

private Q_SLOTS:
    void slotParseVoskDictionaries();
    void getDictionary(const QUrl &sourceUrl = QUrl());
    void removeDictionary();
    void downloadModelFinished(KJob *job);
    void processArchive(const QString &path);
    void doShowSpeechMessage(const QString &message, int messageType);
    /** @brief Check required python dependencies for speech engine */
    void slotCheckSttConfig();
    /** @brief Display the python job output */
    void showSpeechLog(const QString &jobData);
    void showSamLog(const QString &jobData);
    /** @brief A download job is finished  */
    void downloadJobDone(bool success);
    /** @brief Start downloading a model */
    void downloadSamModel(const QString &url);
    /** @brief Show a model download dialog */
    void downloadSamModels();
    /** @brief Install a model if none, and refresh the list of available SAM models in combobox */
    void installSamModelIfEmpty();
    /** @brief Refresh the list of available SAM models in combobox */
    void reloadSamModels();
    /** @brief Get ready to delete the venv */
    void doDeleteSamVenv();
    void doDeleteWrVenv();
    /** @brief Get ready to delete the tool server's venv */
    void doDeleteMcpVenv();
    /** @brief Get ready to delete the models */
    void doDeleteSamModels();
    /** @brief Check if SAM is correctly setup */
    void checkSamEnvironement(bool afterInstall = true);
    void gotWhisperFeedback(const QString &scriptName, const QStringList args, const QStringList jobData);
    void whisperFinished(const QString &scriptName, const QStringList &args);
    void whisperAvailable();
    void whisperMissing();
    void gotSamFeedback(const QString &scriptName, const QStringList args, const QStringList jobData);
    void samFinished(const QString &scriptName, const QStringList &args);
    void samMissing(const QStringList &);
    void samDependenciesChecked();

Q_SIGNALS:
    void openBrowserUrl(const QString &url);
    /** @brief Trigger parsing of the speech models folder */
    void parseDictionaries();
};
