/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "pluginjobparamwidget.hpp"

#include "assets/model/assetparametermodel.hpp"
#include "core.h"
#include "doc/wunjodoc.h"
#include "mainwindow.h"
#include "plugins/plugineffects.h"
#include "effects/effectstack/model/effectitemmodel.hpp"
#include "plugins/pluginmanager.h"

#include <KLocalizedString>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QLabel>
#include <QJsonArray>
#include <QJsonObject>
#include <QProgressBar>
#include <QPushButton>
#include <QUuid>
#include <QVBoxLayout>

PluginJobParamWidget::PluginJobParamWidget(std::shared_ptr<AssetParameterModel> model, QModelIndex index, QWidget *parent)
    : AbstractParamWidget(std::move(model), index, parent)
{
    m_runLabel = m_model->data(m_index, Qt::DisplayRole).toString();
    m_rerunLabel = m_model->data(m_index, AssetParameterModel::AlternateNameRole).toString();
    if (m_rerunLabel.isEmpty()) {
        m_rerunLabel = m_runLabel;
    }
    const QVariantList jobParams = m_model->data(m_index, AssetParameterModel::FilterJobParamsRole).toList();
    for (const QVariant &entry : jobParams) {
        const QStringList pair = entry.toStringList();
        if (pair.size() != 2) {
            continue;
        }
        if (pair.at(0) == QLatin1String("key")) {
            m_resultParam = pair.at(1);
        } else if (pair.at(0) == QLatin1String("conditionalinfo")) {
            m_pendingHint = pair.at(1);
        }
    }
    m_pluginId = PluginManager::instance().pluginForEffect(m_model->getAssetId());
    m_owner = m_model->getOwnerId();
    // The effect's own place in the stack: a clip can carry two of the same
    // effect, and a render belongs to one of them.
    if (auto effect = std::dynamic_pointer_cast<EffectItemModel>(m_model)) {
        m_effectItemId = effect->getId();
    }

    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);
    m_button = new QPushButton(m_runLabel, this);
    layout->addWidget(m_button);
    // same thin bar the Analyse button of Motion Tracker draws under itself
    m_progress = new QProgressBar(this);
    m_progress->setMaximumHeight(m_button->height() / 5);
    m_progress->setTextVisible(false);
    m_progress->setStyleSheet(QStringLiteral("QProgressBar::chunk {background-color: %1;}").arg(m_progress->palette().highlight().color().name()));
    m_progress->setVisible(false);
    layout->addWidget(m_progress);
    setMinimumHeight(m_button->sizeHint().height());

    connect(m_button, &QPushButton::clicked, this, &PluginJobParamWidget::runJob);
    // A render outlives this widget, so pick up whatever is already going on and
    // keep following it.
    connect(&PluginManager::instance(), &PluginManager::effectJobProgressChanged, this, [this](const ObjectId &owner, int effectItemId, int progress) {
        if (owner == m_owner && effectItemId == m_effectItemId) {
            Q_UNUSED(progress)
            updateState();
        }
    });
    updateState();
}

void PluginJobParamWidget::updateState()
{
    const int progress = PluginManager::instance().effectJobProgress(m_owner, m_effectItemId);
    const bool queued = progress == PluginManager::JobQueued;
    const bool running = progress >= 0;
    const bool hasResult = !m_resultParam.isEmpty() && !m_model->getParamFromName(m_resultParam).toString().isEmpty();
    if (queued) {
        // its plugin is busy with another clip; this one goes next
        m_button->setText(i18n("Queued…"));
    } else if (running) {
        m_button->setText(i18n("Rendering… %1%", progress));
    } else {
        m_button->setText(hasResult ? m_rerunLabel : m_runLabel);
    }
    m_button->setEnabled(!running && !queued && !m_pluginId.isEmpty());
    m_progress->setVisible(running);
    m_progress->setValue(qMax(0, progress));
    setToolTip(running || queued ? QString() : (hasResult ? QString() : m_pendingHint));
}

void PluginJobParamWidget::runJob()
{
    if (m_pluginId.isEmpty() || PluginManager::instance().effectJobProgress(m_owner, m_effectItemId) != PluginManager::JobNone) {
        return;
    }
    const QJsonObject input = PluginEffects::buildJob(m_model, m_pluginId);
    if (input.isEmpty()) {
        pCore->displayMessage(i18n("This effect is not on a clip that can be rendered."), ErrorMessage);
        return;
    }
    // Hand it to the manager and forget it: what comes back is watched through
    // the signal, so leaving this clip or moving the playhead changes nothing.
    PluginManager::instance().runEffectJob(m_pluginId, m_owner, m_effectItemId, m_resultParam, input);
    updateState();
}

QLabel *PluginJobParamWidget::createLabel()
{
    // the row is the button alone, as with the Analyse button of Motion Tracker
    return new QLabel();
}

void PluginJobParamWidget::slotShowComment(bool show)
{
    Q_UNUSED(show)
}

void PluginJobParamWidget::slotRefresh()
{
    updateState();
}
