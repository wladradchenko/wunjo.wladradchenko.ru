/*
    SPDX-FileCopyrightText: 2015-2016 Meltytech LLC
    SPDX-FileCopyrightText: 2019 Jean-Baptiste Mardelle <jb@kdenlive.org>

    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "timelineplayhead.h"
#include <QPainter>
#include <QPainterPath>

TimelinePlayhead::TimelinePlayhead(QQuickItem *parent)
    : QQuickPaintedItem(parent)
{
    connect(this, &TimelinePlayhead::colorChanged, this, [&](const QColor &) { update(); });
}

void TimelinePlayhead::paint(QPainter *painter)
{
    // Antialias the marker so its diagonal edges read as clean, not stair-stepped.
    painter->setRenderHint(QPainter::Antialiasing, true);
    QPainterPath path;
    path.moveTo(width(), 0);
    path.lineTo(width() / 2.0, height());
    path.lineTo(0, 0);
    painter->fillPath(path, m_color);
}
