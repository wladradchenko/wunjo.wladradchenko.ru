/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QHash>
#include <QJsonObject>
#include <QObject>
#include <QPair>
#include <QPointer>
#include <QString>

#include <functional>

class QWidget;

/** @class SpendGuard
    @brief The user's say over what an assistant spends.

    A run that a plugin prices first (a `gate` on a card's action or on an
    effect) costs the user money. Pressed by the user, it runs: they saw the
    price on the button they pressed. Asked for by an assistant, it does not run
    until the user agrees in the editor — never on the assistant's word that
    they did. What somebody wrote in the chat is not the agreement; the window
    the editor shows is.

    So that an assistant can work through a task without a question per clip,
    the user can give it an allowance: up to N credits of a plugin, spent
    without asking, counted here, shown in the chat with a way to take it back,
    forgotten when the editor closes. A run that fails gives its share back.
 */
class SpendGuard : public QObject
{
    Q_OBJECT
public:
    explicit SpendGuard(QWidget *window);

    struct Spend {
        QString pluginId;
        /** @brief What is made: the card's title, the effect's name. */
        QString title;
        /** @brief The plugin's own sentence about the price. */
        QString sentence;
        /** @brief The price in the plugin's credits, -1 when it gave no number. */
        int price = -1;
        /** @brief One question at a time per card or effect. */
        QString target;
        /** @brief Whether the price still holds for what would run now. */
        std::function<bool()> stillValid;
        /** @brief Start it; gets "user" or "allowance", returns the job id. */
        std::function<QString(const QString &confirmedBy)> run;
    };

    /** @brief A run an assistant asks for: within the allowance it starts at
     *  once, otherwise the user is asked. @return The request's number. */
    int request(const Spend &spend);
    /** @brief The user is asked to let the assistant spend up to @p credits of
     *  @p pluginId without asking; 0 only reads what is left. */
    int askAllowance(const QString &pluginId, int credits);
    /** @brief {"state": "waiting" | "accepted" | "declined" | "stale" | "failed",
     *  "job", "by", "left", "message"}. */
    QJsonObject state(int request) const;

    int allowance(const QString &pluginId) const;
    /** @brief Plugins with an allowance left. */
    QStringList allowed() const;
    void revoke(const QString &pluginId);

Q_SIGNALS:
    void allowanceChanged();

private:
    void settle(int request, const QString &job, const QString &by, const Spend &spend);

    QPointer<QWidget> m_window;
    int m_next = 0;
    QHash<int, QJsonObject> m_states;
    /** @brief Card or effect → the question open about it. */
    QHash<QString, int> m_open;
    QHash<QString, int> m_allowance;
    /** @brief Job → (plugin, credits) taken from an allowance, given back when
     *  the job fails. */
    QHash<QString, QPair<QString, int>> m_taken;
};
