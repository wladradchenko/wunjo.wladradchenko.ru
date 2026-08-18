/*
SPDX-FileCopyrightText: 2016 Jean-Baptiste Mardelle <jb@kdenlive.org>
This file is part of Wunjo. See www.wunjo.online.

SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QElapsedTimer>
#include <QStyleOptionToolButton>
#include <QToolButton>

class QAction;

/** @class ProgressButton
    @brief A Toolbar button with a small progress bar.
 */
class ProgressButton : public QToolButton
{
    Q_PROPERTY(int progress READ progress WRITE setProgress NOTIFY progressChanged)
    Q_OBJECT
public:
    explicit ProgressButton(const QString &text, double max = 100, QWidget *parent = nullptr);
    ~ProgressButton() override;
    int progress() const;
    void setProgress(int);
    void defineDefaultAction(QAction *action, QAction *actionInProgress);

protected:
    void paintEvent(QPaintEvent *event) override;
    /** @brief Whenever the default action changes, Qt copies its text back
        onto the button; keep the button icon-only. */
    void actionEvent(QActionEvent *event) override;

private:
    /** @brief setDefaultAction + clear the copied text: the button shows the
        icon only, action names stay in the drop-down menu. */
    void applyDefaultAction(QAction *action);
    QAction *m_defaultAction;
    int m_max;
    int m_progress;
    QElapsedTimer m_timer;
    QString m_remainingTime;
    int m_iconSize;
    QFont m_progressFont;
    QStyleOptionToolButton m_buttonStyle;
    /** @brief While rendering, replace real action by a fake on so that rendering is not triggered when clicking again. */
    QAction *m_dummyAction;

Q_SIGNALS:
    void progressChanged();
};
