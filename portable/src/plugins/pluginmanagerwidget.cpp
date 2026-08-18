/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "pluginmanagerwidget.h"

#include <KLocalizedString>
#include <KMessageWidget>

#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QTableWidget>
#include <QUrl>
#include <QVBoxLayout>

PluginManagerWidget::PluginManagerWidget(QWidget *parent)
    : QWidget(parent)
{
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    // ---- Import section ----
    auto *importBox = new QGroupBox(i18n("Load a plugin"), this);
    auto *importLayout = new QVBoxLayout(importBox);

    auto *hint = new QLabel(i18n("Import a plugin from a <b>.wmplugin</b> archive or a plugin folder. "
                                 "Its details are shown for review before it is installed."),
                            importBox);
    hint->setWordWrap(true);
    hint->setStyleSheet(QStringLiteral("color:#696969"));
    importLayout->addWidget(hint);

    auto *chooseRow = new QHBoxLayout;
    auto *fromArchive = new QPushButton(QIcon::fromTheme(QStringLiteral("document-open")), i18n("From archive…"), importBox);
    auto *fromFolder = new QPushButton(QIcon::fromTheme(QStringLiteral("folder-open")), i18n("From folder…"), importBox);
    chooseRow->addWidget(fromArchive);
    chooseRow->addWidget(fromFolder);
    chooseRow->addStretch();
    importLayout->addLayout(chooseRow);

    m_info = new KMessageWidget(importBox);
    m_info->setWordWrap(true);
    m_info->setCloseButtonVisible(false);
    m_info->hide();
    importLayout->addWidget(m_info);

    auto *actionRow = new QHBoxLayout;
    actionRow->addStretch();
    m_cancelButton = new QPushButton(i18n("Cancel"), importBox);
    m_importButton = new QPushButton(QIcon::fromTheme(QStringLiteral("run-install")), i18n("Import"), importBox);
    m_importButton->setEnabled(false);
    m_cancelButton->setEnabled(false);
    actionRow->addWidget(m_cancelButton);
    actionRow->addWidget(m_importButton);
    importLayout->addLayout(actionRow);

    layout->addWidget(importBox);

    // ---- Installed section ----
    auto *installedBox = new QGroupBox(i18n("Installed plugins"), this);
    auto *installedLayout = new QVBoxLayout(installedBox);
    m_table = new QTableWidget(0, 2, installedBox);
    m_table->setHorizontalHeaderLabels({i18n("Name"), i18n("Version")});
    m_table->horizontalHeader()->setVisible(false);
    m_table->verticalHeader()->setVisible(false);
    m_table->setSelectionBehavior(QAbstractItemView::SelectRows);
    m_table->setSelectionMode(QAbstractItemView::SingleSelection);
    m_table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    m_table->setShowGrid(false);
    m_table->setAlternatingRowColors(true);
    m_table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::Stretch);
    m_table->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    installedLayout->addWidget(m_table);
    auto *manageRow = new QHBoxLayout;
    m_uninstallButton = new QPushButton(QIcon::fromTheme(QStringLiteral("edit-delete")), i18n("Delete plugin"), installedBox);
    m_uninstallButton->setEnabled(false);
    auto *openFolder = new QPushButton(QIcon::fromTheme(QStringLiteral("folder")), i18n("Open plugins folder"), installedBox);
    manageRow->addWidget(m_uninstallButton);
    manageRow->addStretch();
    manageRow->addWidget(openFolder);
    installedLayout->addLayout(manageRow);
    layout->addWidget(installedBox);

    connect(fromArchive, &QPushButton::clicked, this, &PluginManagerWidget::chooseArchive);
    connect(fromFolder, &QPushButton::clicked, this, &PluginManagerWidget::chooseFolder);
    connect(m_importButton, &QPushButton::clicked, this, &PluginManagerWidget::doImport);
    connect(m_cancelButton, &QPushButton::clicked, this, &PluginManagerWidget::cancelPreview);
    connect(m_uninstallButton, &QPushButton::clicked, this, &PluginManagerWidget::uninstallSelected);
    connect(openFolder, &QPushButton::clicked, this, [this]() {
        const QString dir = PluginManager::instance().userPluginsDir();
        QDir().mkpath(dir);
        QDesktopServices::openUrl(QUrl::fromLocalFile(dir));
    });
    connect(m_table, &QTableWidget::itemSelectionChanged, this, [this]() {
        auto *item = m_table->currentItem();
        const bool bundled = item && item->data(Qt::UserRole + 1).toBool();
        m_uninstallButton->setEnabled(item != nullptr && !bundled);
    });
    connect(&PluginManager::instance(), &PluginManager::pluginsChanged, this, &PluginManagerWidget::refreshList);

    refreshList();
}

void PluginManagerWidget::chooseArchive()
{
    const QString path = QFileDialog::getOpenFileName(this, i18n("Select a plugin archive"), QDir::homePath(),
                                                      i18n("Wunjo plugin (*.wmplugin *.zip)"));
    if (!path.isEmpty()) {
        preview(path);
    }
}

void PluginManagerWidget::chooseFolder()
{
    const QString path = QFileDialog::getExistingDirectory(this, i18n("Select a plugin folder"), QDir::homePath());
    if (!path.isEmpty()) {
        preview(path);
    }
}

void PluginManagerWidget::preview(const QString &path)
{
    m_candidate = PluginManager::instance().inspect(path);
    m_info->hide();
    if (m_candidate.sourceDir.isEmpty()) {
        m_candidate.manifest = PluginManifest();
        m_info->setMessageType(KMessageWidget::Error);
        m_info->setText(i18n("This file is not a readable plugin archive or folder."));
        m_importButton->setEnabled(false);
        m_cancelButton->setEnabled(true);
        m_info->animatedShow();
        return;
    }
    if (m_candidate.valid() && !m_candidate.manifest.osSupported()) {
        m_info->setMessageType(KMessageWidget::Error);
        m_info->setText(i18n("This plugin does not support %1 and cannot be installed here.", PluginManifest::currentOs()));
        m_importButton->setEnabled(false);
    } else if (m_candidate.valid()) {
        m_info->setMessageType(KMessageWidget::Positive);
        m_info->setText(m_candidate.manifest.summaryHtml());
        m_importButton->setEnabled(true);
    } else {
        m_info->setMessageType(KMessageWidget::Error);
        m_info->setText(i18n("This plugin cannot be imported:<ul><li>%1</li></ul>",
                             m_candidate.manifest.errors().join(QStringLiteral("</li><li>"))));
        m_importButton->setEnabled(false);
    }
    m_cancelButton->setEnabled(true);
    m_info->animatedShow();
}

void PluginManagerWidget::doImport()
{
    if (!m_candidate.valid()) {
        return;
    }
    const QString id = m_candidate.manifest.id();
    const QString name = m_candidate.manifest.name();
    if (PluginManager::instance().installedPlugins().size() > 0) {
        for (const PluginManifest &existing : PluginManager::instance().installedPlugins()) {
            if (existing.id() == id) {
                const auto answer = QMessageBox::question(this, i18n("Replace plugin"),
                                                          i18n("A plugin '%1' is already installed. Replace it?", id));
                if (answer != QMessageBox::Yes) {
                    return;
                }
                break;
            }
        }
    }
    QString error;
    if (PluginManager::instance().install(m_candidate, &error)) {
        m_info->setMessageType(KMessageWidget::Positive);
        m_info->setText(i18n("Imported '%1'.", name));
        m_info->animatedShow();
        m_importButton->setEnabled(false);
        m_cancelButton->setEnabled(false);
        m_candidate = PluginManager::ImportCandidate();
    } else {
        m_info->setMessageType(KMessageWidget::Error);
        m_info->setText(i18n("Import failed: %1", error));
        m_info->animatedShow();
    }
}

void PluginManagerWidget::cancelPreview()
{
    m_candidate = PluginManager::ImportCandidate();
    m_info->animatedHide();
    m_importButton->setEnabled(false);
    m_cancelButton->setEnabled(false);
}

void PluginManagerWidget::uninstallSelected()
{
    auto *item = m_table->currentItem();
    if (!item) {
        return;
    }
    const int row = item->row();
    QTableWidgetItem *nameItem = m_table->item(row, 0);
    const QString id = nameItem->data(Qt::UserRole).toString();
    const auto answer = QMessageBox::question(this, i18n("Delete plugin"),
                                              i18n("Delete '%1' and all its files, downloaded models and environment? "
                                                   "This cannot be undone.",
                                                   nameItem->text()));
    if (answer != QMessageBox::Yes) {
        return;
    }
    QString error;
    if (!PluginManager::instance().uninstall(id, &error)) {
        QMessageBox::warning(this, i18n("Delete plugin"), error);
    }
}

void PluginManagerWidget::refreshList()
{
    const auto plugins = PluginManager::instance().installedPlugins();
    m_table->setRowCount(plugins.size());
    int row = 0;
    for (const PluginManifest &manifest : plugins) {
        const bool bundled = PluginManager::instance().isBundled(manifest.id());
        QString name = manifest.name();
        if (bundled) {
            name += i18n("  (built-in)");
        }
        auto *nameItem = new QTableWidgetItem(name);
        nameItem->setData(Qt::UserRole, manifest.id());
        nameItem->setData(Qt::UserRole + 1, bundled);
        nameItem->setToolTip(manifest.description());
        m_table->setItem(row, 0, nameItem);
        m_table->setItem(row, 1, new QTableWidgetItem(manifest.version()));
        ++row;
    }
    m_uninstallButton->setEnabled(false);
}
