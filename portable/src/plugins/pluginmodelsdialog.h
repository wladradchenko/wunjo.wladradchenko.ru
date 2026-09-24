/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "pluginmanifest.h"

#include <QDialog>
#include <QList>

class FileDownloadJob;
class KMessageWidget;
class QGroupBox;
class QListWidget;
class QListWidgetItem;
class QProgressBar;
class QPushButton;

/** @class PluginModelsDialog
    @brief The window behind "Manage models" on a plugin's settings page, for a
    plugin whose model comes in several sizes: install one, remove one.

    Shaped like the speech models' window, because it answers the same
    question: every size on one list, the installed ones marked, the ones this
    machine cannot run greyed out with the reason, and one button that installs
    or removes whichever is selected. A size is all of its weights: they are
    fetched one after another and it counts as installed only when the last of
    them is on disk and whole.
 */
class PluginModelsDialog : public QDialog
{
    Q_OBJECT
public:
    enum Role { VariantRole = Qt::UserRole, InstalledRole };
    explicit PluginModelsDialog(const PluginManifest &manifest, QWidget *parent = nullptr);
    /** @brief True when a size was installed or removed here. */
    bool modelsChanged() const { return m_changed; }

private Q_SLOTS:
    void updateButton(int row);
    void act();
    void queryClose();

private:
    void fill();
    int rowOf(const QString &variantId) const;
    /** @brief The weights of one size, as this machine needs them. */
    QList<PluginModel> weightsOf(const QString &variantId) const;
    /** @brief Fetch the next weight in the queue, or finish when it is empty. */
    void startNext();
    void finishDownload(bool ok, const QString &error);
    void removeVariant(int row);

    PluginManifest m_manifest;
    QListWidget *m_list;
    KMessageWidget *m_message;
    QGroupBox *m_downloadGroup;
    QProgressBar *m_progress;
    QPushButton *m_button;
    /** @brief The size being fetched: its row, its weights, how many are done,
     *  and the job on the current one. */
    int m_downloadingRow = -1;
    QList<PluginModel> m_queue;
    int m_queueDone = 0;
    FileDownloadJob *m_job = nullptr;
    bool m_aborting = false;
    bool m_changed = false;
};
