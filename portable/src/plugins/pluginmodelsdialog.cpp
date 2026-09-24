/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "pluginmodelsdialog.h"

#include "filedownloadjob.h"
#include "pluginmanager.h"

#include <KIO/Global>
#include <KJob>
#include <KLocalizedString>
#include <KMessageBox>
#include <KMessageWidget>

#include <QDialogButtonBox>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QListWidgetItem>
#include <QProgressBar>
#include <QPushButton>
#include <QVBoxLayout>

PluginModelsDialog::PluginModelsDialog(const PluginManifest &manifest, QWidget *parent)
    : QDialog(parent)
    , m_manifest(manifest)
{
    setWindowTitle(i18n("%1 models", manifest.name()));
    auto *l = new QVBoxLayout;
    setLayout(l);
    l->addWidget(new QLabel(i18n("Select new models to download"), this));
    m_list = new QListWidget(this);
    m_list->setAlternatingRowColors(true);
    if (PluginManager::gpuVramGb() <= 0) {
        auto *cpuNote = new KMessageWidget(this);
        cpuNote->setCloseButtonVisible(false);
        cpuNote->setWordWrap(true);
        cpuNote->setMessageType(KMessageWidget::Information);
        cpuNote->setText(i18n("No graphics card was found, so only the light models are offered. Bigger ones run far slower on the processor."));
        l->addWidget(cpuNote);
    }
    l->addWidget(m_list);
    m_message = new KMessageWidget(this);
    m_message->setCloseButtonVisible(false);
    m_message->setWordWrap(true);
    m_message->setMessageType(KMessageWidget::Information);
    m_message->hide();
    l->addWidget(m_message);
    m_downloadGroup = new QGroupBox(this);
    auto *downloadLayout = new QHBoxLayout;
    downloadLayout->addWidget(new QLabel(i18n("Downloading"), this));
    m_progress = new QProgressBar(this);
    m_progress->setRange(0, 100);
    downloadLayout->addWidget(m_progress);
    m_downloadGroup->setLayout(downloadLayout);
    m_downloadGroup->setVisible(false);
    l->addWidget(m_downloadGroup);
    m_button = new QPushButton(i18n("Install model"), this);
    m_button->setEnabled(false);
    l->addWidget(m_button);
    auto *buttonBox = new QDialogButtonBox(QDialogButtonBox::Close);
    l->addWidget(buttonBox);
    connect(m_list, &QListWidget::currentRowChanged, this, &PluginModelsDialog::updateButton);
    connect(m_button, &QPushButton::clicked, this, &PluginModelsDialog::act);
    connect(buttonBox->button(QDialogButtonBox::Close), &QPushButton::clicked, this, &PluginModelsDialog::queryClose);
    fill();
    // start on the size in use, or the one this machine is best served by
    const int row = rowOf(PluginManager::selectedVariant(manifest));
    if (row >= 0) {
        m_list->setCurrentRow(row);
    }
}

void PluginModelsDialog::fill()
{
    m_list->clear();
    const double vram = PluginManager::gpuVramGb();
    const QList<PluginVariant> variants = m_manifest.variants();
    for (const PluginVariant &variant : variants) {
        auto *item = new QListWidgetItem(variant.label, m_list);
        item->setData(VariantRole, variant.id);
        item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
        if (!variant.note.isEmpty()) {
            item->setToolTip(variant.note);
        }
        // A size this machine cannot run stays on the list — the machine may get
        // a card later — but cannot be picked, and the row says why.
        const QString blocker = PluginManifest::variantBlocker(variant, vram);
        if (!blocker.isEmpty()) {
            item->setFlags(Qt::NoItemFlags);
            item->setText(i18n("%1 — %2", variant.label, blocker));
            continue;
        }
        const bool ready = PluginManager::instance().variantReady(m_manifest, variant.id);
        item->setData(InstalledRole, ready ? 1 : 0);
        item->setIcon(QIcon::fromTheme(ready ? QStringLiteral("task-process-4") : QStringLiteral("task-process-0")));
    }
}

int PluginModelsDialog::rowOf(const QString &variantId) const
{
    for (int i = 0; i < m_list->count(); ++i) {
        if (m_list->item(i)->data(VariantRole).toString() == variantId) {
            return i;
        }
    }
    return -1;
}

QList<PluginModel> PluginModelsDialog::weightsOf(const QString &variantId) const
{
    QList<PluginModel> weights;
    const QList<PluginModel> models = m_manifest.modelsFor(PluginManager::gpuVramGb(), PluginManager::gpuBackend(), variantId);
    for (const PluginModel &model : models) {
        if (model.variant == variantId) {
            weights.append(model);
        }
    }
    return weights;
}

void PluginModelsDialog::updateButton(int row)
{
    QListWidgetItem *item = m_list->item(row);
    if (item == nullptr || !(item->flags() & Qt::ItemIsEnabled)) {
        m_button->setEnabled(false);
        return;
    }
    m_button->setEnabled(true);
    const int state = item->data(InstalledRole).toInt();
    if (state == -1) {
        m_button->setIcon(QIcon::fromTheme(QStringLiteral("dialog-cancel")));
        m_button->setText(i18n("Abort downloads"));
        return;
    }
    if (state == 1) {
        m_button->setIcon(QIcon::fromTheme(QStringLiteral("edit-delete-remove")));
        m_button->setText(i18n("Remove model"));
        m_message->hide();
        return;
    }
    m_button->setIcon(QIcon::fromTheme(QStringLiteral("list-add")));
    m_button->setText(i18n("Install model"));
    // what picking this one would fetch
    qint64 bytes = 0;
    const QList<PluginModel> weights = weightsOf(item->data(VariantRole).toString());
    for (const PluginModel &model : weights) {
        if (PluginManager::instance().modelState(m_manifest.id(), model) != PluginManager::ModelReady) {
            bytes += model.sizeMb * 1024 * 1024;
        }
    }
    if (bytes > 0 && m_job == nullptr) {
        m_message->setMessageType(KMessageWidget::Information);
        m_message->setText(i18n("Total download size: %1", KIO::convertSize(KIO::filesize_t(bytes))));
        m_message->show();
    } else {
        m_message->hide();
    }
}

void PluginModelsDialog::act()
{
    QListWidgetItem *item = m_list->currentItem();
    if (item == nullptr) {
        return;
    }
    const int state = item->data(InstalledRole).toInt();
    if (state == 1) {
        removeVariant(m_list->row(item));
        return;
    }
    if (state == -1) {
        // abort: the job's own end takes the fragments away
        if (m_job != nullptr) {
            m_aborting = true;
            m_job->kill(KJob::EmitResult);
        }
        return;
    }
    if (m_job != nullptr) {
        return; // one size at a time
    }
    m_queue.clear();
    m_queueDone = 0;
    qint64 bytes = 0;
    const QList<PluginModel> weights = weightsOf(item->data(VariantRole).toString());
    for (const PluginModel &model : weights) {
        if (PluginManager::instance().modelState(m_manifest.id(), model) != PluginManager::ModelReady && !model.url.isEmpty()) {
            m_queue.append(model);
            bytes += model.sizeMb * 1024 * 1024;
        }
    }
    // Refuse before the first byte when the disk cannot hold what is coming:
    // filling the partition takes the user's projects down with it.
    const QString blocker = PluginManager::downloadBlocker(bytes);
    if (!blocker.isEmpty()) {
        m_message->setMessageType(KMessageWidget::Warning);
        m_message->setText(blocker);
        m_message->show();
        return;
    }
    m_downloadingRow = m_list->row(item);
    m_aborting = false;
    item->setData(InstalledRole, -1);
    item->setIcon(QIcon::fromTheme(QStringLiteral("task-process-1")));
    m_message->hide();
    m_progress->setValue(0);
    m_downloadGroup->setVisible(true);
    updateButton(m_downloadingRow);
    startNext();
}

void PluginModelsDialog::startNext()
{
    if (m_queueDone >= m_queue.size()) {
        finishDownload(true, QString());
        return;
    }
    const PluginModel model = m_queue.at(m_queueDone);
    const QString dest = PluginManager::instance().downloadTarget(m_manifest.id(), model);
    // a weight's name may carry a folder ("small/model.safetensors"): make it
    if (!QDir().mkpath(QFileInfo(dest).absolutePath())) {
        finishDownload(false, i18n("Could not create the folder for %1.", model.name));
        return;
    }
    const int count = m_queue.size();
    const int done = m_queueDone;
    m_job = new FileDownloadJob(QUrl(model.url), dest, this);
    connect(m_job, &KJob::percentChanged, this, [this, count, done](KJob *, unsigned long percent) {
        m_progress->setValue(int((done * 100.0 + double(percent)) / count));
    });
    connect(m_job, &KJob::result, this, [this, model, dest](KJob *job) {
        m_job = nullptr;
        if (job->error() != 0) {
            if (m_aborting || job->error() == KJob::KilledJobError) {
                // the user's own Abort: what came down goes, the size is not installed
                removeVariant(m_downloadingRow);
                m_downloadGroup->setVisible(false);
                m_downloadingRow = -1;
                updateButton(m_list->currentRow());
                return;
            }
            finishDownload(false, i18n("Could not download %1: %2", QFileInfo(dest).fileName(), job->errorText()));
            return;
        }
        if (!model.unpack.isEmpty()) {
            QString unpackError;
            const bool unpacked = PluginManager::instance().unpackModel(m_manifest.id(), model, dest, &unpackError);
            QFile::remove(dest);
            if (!unpacked) {
                finishDownload(false, unpackError);
                return;
            }
        } else if (PluginManager::instance().modelState(m_manifest.id(), model, true) != PluginManager::ModelReady) {
            // the checksum is worth reading hundreds of megabytes for exactly
            // once: right after the download that produced them
            QFile::remove(dest);
            finishDownload(false, i18n("%1 arrived damaged and was removed — please download it again.", model.name));
            return;
        }
        ++m_queueDone;
        startNext();
    });
    m_job->start();
}

void PluginModelsDialog::finishDownload(bool ok, const QString &error)
{
    m_downloadGroup->setVisible(false);
    QListWidgetItem *item = m_list->item(m_downloadingRow);
    m_downloadingRow = -1;
    if (item != nullptr) {
        item->setData(InstalledRole, ok ? 1 : 0);
        item->setIcon(QIcon::fromTheme(ok ? QStringLiteral("task-process-4") : QStringLiteral("task-process-0")));
    }
    if (ok) {
        m_changed = true;
    } else {
        m_message->setMessageType(KMessageWidget::Error);
        m_message->setText(error);
        m_message->show();
    }
    updateButton(m_list->currentRow());
}

void PluginModelsDialog::removeVariant(int row)
{
    QListWidgetItem *item = m_list->item(row);
    if (item == nullptr) {
        return;
    }
    const QString variantId = item->data(VariantRole).toString();
    const QList<PluginModel> weights = weightsOf(variantId);
    for (const PluginModel &model : weights) {
        const QString path = PluginManager::instance().modelPath(m_manifest.id(), model);
        if (QFileInfo(path).isDir()) {
            QDir(path).removeRecursively();
        } else {
            QFile::remove(path);
        }
        QFile::remove(PluginManager::instance().downloadTarget(m_manifest.id(), model));
        // the folder the size kept its weights in, once it is empty
        QDir().rmdir(QFileInfo(path).absolutePath());
    }
    item->setData(InstalledRole, 0);
    item->setIcon(QIcon::fromTheme(QStringLiteral("task-process-0")));
    m_changed = true;
    if (row == m_list->currentRow()) {
        updateButton(row);
    }
}

void PluginModelsDialog::queryClose()
{
    if (m_job != nullptr) {
        if (KMessageBox::questionTwoActions(this, i18n("A download is in progress, do you want to abort it ?"), {}, KGuiItem(i18n("Abort download")),
                                            KStandardGuiItem::cancel()) == KMessageBox::SecondaryAction) {
            return;
        }
        m_aborting = true;
        m_job->kill(KJob::EmitResult);
    }
    close();
}
