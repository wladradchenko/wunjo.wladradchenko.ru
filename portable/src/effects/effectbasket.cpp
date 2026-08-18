/*
    SPDX-FileCopyrightText: 2015 Jean-Baptiste Mardelle <jb@kdenlive.org>

SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "effectbasket.h"
#include "effects/effectsrepository.hpp"
#include "wunjosettings.h"
#include <KLocalizedString>

#include <QMimeData>

EffectBasket::EffectBasket(QWidget *parent)
    : QListWidget(parent)
{
    setFrameStyle(QFrame::NoFrame);
    setSelectionMode(QAbstractItemView::SingleSelection);
    setDragEnabled(true);
    slotReloadBasket();
    connect(this, &QListWidget::itemActivated, this, &EffectBasket::slotAddEffect);
}

void EffectBasket::slotReloadBasket()
{
    clear();
    const QStringList effNames = WunjoSettings::favorite_effects();
    for (const QString &effectId : effNames) {
        if (EffectsRepository::get()->exists(effectId)) {
            QListWidgetItem *it = new QListWidgetItem(EffectsRepository::get()->getName(effectId));
            it->setData(Qt::UserRole, effectId);
            addItem(it);
        }
    }
    sortItems();
}

QMimeData *EffectBasket::mimeData(const QList<QListWidgetItem *> &list) const
{
    if (list.isEmpty()) {
        return new QMimeData;
    }
    QDomDocument doc;
    QListWidgetItem *item = list.at(0);
    QString effectId = item->data(Qt::UserRole).toString();
    auto *mime = new QMimeData;
    mime->setData(QStringLiteral("wunjo/effect"), effectId.toUtf8());
    return mime;
}

void EffectBasket::showEvent(QShowEvent *event)
{
    QListWidget::showEvent(event);
    if (!currentItem()) {
        setCurrentRow(0);
    }
}

void EffectBasket::slotAddEffect(QListWidgetItem *item)
{
    QString assetId = item->data(Qt::UserRole).toString();
    QVariantMap mimeData;
    mimeData.insert(QStringLiteral("wunjo/effect"), assetId);
    Q_EMIT activateAsset(mimeData);
}
