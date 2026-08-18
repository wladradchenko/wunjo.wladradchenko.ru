/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QString>
#include <QWidget>

class KMessageWidget;
class QAudioOutput;
class QLabel;
class QMediaPlayer;
class QProgressBar;
class QPushButton;
class QToolButton;
class QTreeWidget;

/** @class PluginSetPanel
    @brief Records the parameter sets an effect picks from — the pen next to
    "Expression source" opens this.

    It sits where the effect stack is, like the mask panel: choose a video or a
    photo, analyse it (the plugin measures a value per frame), and the result is
    kept as a named set next to the project (see PluginSets). Sets can be
    deleted, imported from another project and exported, so the same performance
    never has to be analysed twice.
 */
class PluginSetPanel : public QWidget
{
    Q_OBJECT
public:
    explicit PluginSetPanel(QWidget *parent = nullptr);

    /** @brief Show the sets of @p pluginId — the plugin that owns the effect
     *  being edited. */
    void setPlugin(const QString &pluginId, const QString &kind = QString());
    QString pluginId() const { return m_pluginId; }
    /** @brief Nothing here holds the interface hostage: the job belongs to the
     *  plugin manager and keeps running when this panel is hidden, so the user
     *  is free to walk to another clip or effect meanwhile. */
    bool isLocked() const { return false; }

Q_SIGNALS:
    /** @brief The recorded sets changed; parameter lists showing them are stale. */
    void setsChanged();
    /** @brief The user is done — go back to the effect stack. */
    void closeRequested();

private Q_SLOTS:
    void chooseSource();
    void analyse();
    void deleteSet();
    void importSet();
    void exportSet();

private:
    void refresh();
    void updateButtons();
    void showStatus(const QString &message, bool error);
    /** @brief True when the presets here are sound — a track cannot be told
     *  apart from another by its name alone, so those get a listen button. */
    bool isAudioKind() const;
    /** @brief Play @p path, or stop if it is already the one playing. */
    void togglePlay(const QString &path);
    void stopPlayback();

    QString m_pluginId;
    /** @brief Which sort of preset this panel is recording right now. */
    QString m_kind;
    QString m_source;
    bool m_running{false};
    QLabel *m_title;
    QLabel *m_sourceLabel;
    QPushButton *m_choose;
    /** @brief File dialog filter the plugin asked for. */
    QString m_sourceFilter;
    QPushButton *m_analyse;
    QProgressBar *m_progress;
    KMessageWidget *m_status;
    QTreeWidget *m_sets;
    QToolButton *m_delete;
    QToolButton *m_import;
    QToolButton *m_playSource{nullptr};
    QToolButton *m_playSet{nullptr};
    /** @brief Built on first use — most plugins never play anything. */
    QMediaPlayer *m_player{nullptr};
    QAudioOutput *m_audioOut{nullptr};
    QString m_playing;
};
