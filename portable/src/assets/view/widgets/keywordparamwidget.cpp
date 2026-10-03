/*
    SPDX-FileCopyrightText: 2016 Nicolas Carion
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "keywordparamwidget.hpp"
#include "assets/model/assetparametermodel.hpp"

KeywordParamWidget::KeywordParamWidget(std::shared_ptr<AssetParameterModel> model, QModelIndex index, QWidget *parent)
    : AbstractParamWidget(std::move(model), index, parent)
{
    setupUi(this);

    QStringList kwrdValues = m_model->data(m_index, AssetParameterModel::ListValuesRole).toStringList();
    QStringList kwrdNames = m_model->data(m_index, AssetParameterModel::ListNamesRole).toStringList();
    comboboxwidget->addItems(kwrdNames);
    int i = 0;
    for (const QString &keywordVal : std::as_const(kwrdValues)) {
        if (i >= comboboxwidget->count()) {
            break;
        }
        comboboxwidget->setItemData(i, keywordVal);
        i++;
    }
    comboboxwidget->insertItem(0, i18n("Insert a Keyword…"));
    comboboxwidget->setCurrentIndex(0);
    // A "text" parameter has no keywords to offer: just the box, a few lines
    // tall, with the comment as the hint of what goes in it.
    if (kwrdValues.isEmpty()) {
        comboboxwidget->hide();
        lineeditwidget->setPlaceholderText(m_model->data(m_index, AssetParameterModel::CommentRole).toString());
        // compact="1" for a phrase (a style, a description), the full height
        // for something that is read out
        const int lines = m_model->data(m_index, AssetParameterModel::CompactRole).toBool() ? 2 : 5;
        lineeditwidget->setFixedHeight(lineeditwidget->fontMetrics().lineSpacing() * lines + 12);
    }

    label->setText(m_model->data(m_index, Qt::DisplayRole).toString());
    // set check state
    slotRefresh();

    // Q_EMIT the signal of the base class when appropriate
    connect(lineeditwidget, &QPlainTextEdit::textChanged, this, [this]() { Q_EMIT valueChanged(m_index, lineeditwidget->toPlainText(), true); });
    connect(comboboxwidget, static_cast<void (QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [this](int ix) {
        if (ix > 0) {
            QString comboval = comboboxwidget->itemData(ix).toString();
            this->lineeditwidget->insertPlainText(comboval);
            Q_EMIT valueChanged(m_index, lineeditwidget->toPlainText(), true);
            comboboxwidget->setCurrentIndex(0);
        }
    });
}

void KeywordParamWidget::slotShowComment(bool show)
{
    Q_UNUSED(show);
}

void KeywordParamWidget::slotRefresh()
{
    // Every keystroke goes to the model and comes back here; setting the same
    // text again would throw the cursor to the start of the box mid-sentence.
    const QString value = m_model->data(m_index, AssetParameterModel::ValueRole).toString();
    if (value != lineeditwidget->toPlainText()) {
        lineeditwidget->setPlainText(value);
    }
}
