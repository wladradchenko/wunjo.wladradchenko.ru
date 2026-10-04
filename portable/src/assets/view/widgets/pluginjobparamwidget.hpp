/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "abstractparamwidget.hpp"

#include "definitions.h"

class QLabel;
class QProgressBar;
class QPushButton;

/** @class PluginJobParamWidget
    @brief The button that makes a plugin effect produce its result.

    Same shape as the Analyse button of Motion Tracker or Loudness
    (ButtonParamWidget / `filterjob`): a button with a thin progress bar under
    it, one label before the work is done and another after. The difference is
    what runs — not an MLT filter through melt, but the plugin that brought the
    effect, with everything the effect holds handed over as the job.

    An effect whose model cannot play while the timeline does is otherwise mute:
    the parameters sit there and nothing happens. This is where the user says
    "now", and where the wait is visible.

    With `<jobparam name="action">price</jobparam>` the button asks the plugin
    a question instead (what the render would cost) and shows the answer under
    itself. With `<jobparam name="gate">price</jobparam>` a render button stays
    shut until that question has been answered yes for the effect as it is now.
 */
class PluginJobParamWidget : public AbstractParamWidget
{
    Q_OBJECT
public:
    PluginJobParamWidget(std::shared_ptr<AssetParameterModel> model, QModelIndex index, QWidget *parent);

    /** @brief No caption on the left: the button says what it does itself,
     *  exactly like the Analyse button of Motion Tracker. */
    QLabel *createLabel() override;

public Q_SLOTS:
    void slotShowComment(bool show) override;
    void slotRefresh() override;

private:
    void runJob();
    void ask();
    /** @brief Ask the panel for room for the answer under the button. */
    void fitHeight();
    void resizeEvent(QResizeEvent *event) override;
    /** @brief The effect's job as it stands, to tell whether an answer still holds. */
    QByteArray asked() const;
    /** @brief Label, tooltip and progress: whether a render is in flight is
     *  asked of the manager, never remembered here — this widget is destroyed
     *  and rebuilt every time the playhead moves or the clip is reselected. */
    void updateState();

    QPushButton *m_button;
    QProgressBar *m_progress;
    QLabel *m_answer{nullptr};
    /** @brief The question this button asks, when it is not a render. */
    QString m_action;
    /** @brief The question that must have been answered yes before this render. */
    QString m_gate;
    QString m_pluginId;
    /** @brief Parameter that receives the produced file, so the effect knows
     *  whether it has a result at all. */
    QString m_resultParam;
    QString m_runLabel;
    QString m_rerunLabel;
    QString m_pendingHint;
    ObjectId m_owner;
    int m_effectItemId{-1};
};
