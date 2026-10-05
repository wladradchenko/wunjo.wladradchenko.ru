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
#include <QJsonDocument>
#include <QJsonObject>
#include <QProgressBar>
#include <QPushButton>
#include <QResizeEvent>
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
        } else if (pair.at(0) == QLatin1String("action")) {
            m_action = pair.at(1);
        } else if (pair.at(0) == QLatin1String("gate")) {
            m_gate = pair.at(1);
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
    if (!m_action.isEmpty()) {
        m_answer = new QLabel(this);
        m_answer->setWordWrap(true);
        // "top up at https://…" is meant to be followed
        m_answer->setTextFormat(Qt::RichText);
        m_answer->setOpenExternalLinks(true);
        m_answer->setVisible(false);
        layout->addWidget(m_answer);
    }
    setMinimumHeight(m_button->sizeHint().height());

    connect(m_button, &QPushButton::clicked, this, [this]() { m_action.isEmpty() ? runJob() : ask(); });
    if (!m_action.isEmpty() || !m_gate.isEmpty()) {
        // an answer holds for the effect as it was asked about: any change to
        // it, here or in the other button, is a reason to look again
        connect(&PluginManager::instance(), &PluginManager::effectAnswerChanged, this, [this](const ObjectId &owner, int effectItemId) {
            if (owner == m_owner && effectItemId == m_effectItemId) {
                updateState();
            }
        });
        connect(m_model.get(), &QAbstractItemModel::dataChanged, this, [this]() { updateState(); });
        // an edit made in this very panel does not come back as dataChanged
        // (the panel already shows it), only as updateChildren
        connect(m_model.get(), &AssetParameterModel::updateChildren, this, [this]() { updateState(); });
    }
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

void PluginJobParamWidget::fitHeight()
{
    // The panel gives every parameter the height it asked for when it was
    // built; a sentence that appears later has to ask for more, or it is cut.
    int height = m_button->sizeHint().height();
    if (m_answer && m_answer->isVisible()) {
        height += layout()->spacing() + m_answer->heightForWidth(qMax(120, width()));
    }
    if (height != minimumHeight()) {
        setMinimumHeight(height);
        Q_EMIT updateHeight();
    }
}

void PluginJobParamWidget::resizeEvent(QResizeEvent *event)
{
    AbstractParamWidget::resizeEvent(event);
    fitHeight();
}

QByteArray PluginJobParamWidget::asked() const
{
    return PluginEffects::askedJob(m_model, m_pluginId);
}

void PluginJobParamWidget::ask()
{
    if (m_pluginId.isEmpty()) {
        return;
    }
    // The answer belongs to the effect, not to this widget, which the stack
    // rebuilds at will: the manager keeps it and tells whoever shows it now.
    if (!PluginEffects::askEffect(m_model, m_pluginId, m_effectItemId, m_action)) {
        pCore->displayMessage(i18n("This effect is not on a clip that can be rendered."), ErrorMessage);
    }
}

void PluginJobParamWidget::updateState()
{
    if (!m_action.isEmpty()) {
        const PluginManager::EffectAnswer answer = PluginManager::instance().effectAnswer(m_owner, m_effectItemId, m_action);
        const bool current = !answer.asked.isEmpty() && answer.asked == asked();
        m_button->setText(m_runLabel);
        m_button->setEnabled(!answer.pending && !m_pluginId.isEmpty());
        m_answer->setText(PluginEffects::linkify(answer.message));
        m_answer->setVisible(current && !answer.pending && !answer.message.isEmpty());
        m_progress->setVisible(false);
        fitHeight();
        return;
    }
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
    bool gateOpen = true;
    if (!m_gate.isEmpty() && !running && !queued) {
        const PluginManager::EffectAnswer answer = PluginManager::instance().effectAnswer(m_owner, m_effectItemId, m_gate);
        gateOpen = answer.ok && !answer.pending && answer.asked == asked();
    }
    m_button->setEnabled(!running && !queued && gateOpen && !m_pluginId.isEmpty());
    m_progress->setVisible(running);
    m_progress->setValue(qMax(0, progress));
    setToolTip(running || queued ? QString() : (!gateOpen ? i18n("Check the price first") : (hasResult ? QString() : m_pendingHint)));
}

void PluginJobParamWidget::runJob()
{
    if (m_pluginId.isEmpty() || PluginManager::instance().effectJobProgress(m_owner, m_effectItemId) != PluginManager::JobNone) {
        return;
    }
    int price = -1;
    if (!PluginEffects::gateOpen(m_model, m_pluginId, m_effectItemId, m_gate, &price)) {
        return;
    }
    QJsonObject input = PluginEffects::buildJob(m_model, m_pluginId);
    if (input.isEmpty()) {
        pCore->displayMessage(i18n("This effect is not on a clip that can be rendered."), ErrorMessage);
        return;
    }
    // The user pressed it, having seen the price: the plugin refuses to charge
    // more than that (see plugins/README.md)
    input.insert(QStringLiteral("started_by"), QStringLiteral("user"));
    if (price >= 0) {
        input.insert(QStringLiteral("confirmed"), QJsonObject{{QStringLiteral("price"), price}});
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
