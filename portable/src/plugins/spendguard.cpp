/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "spendguard.h"

#include "plugineffects.h"
#include "pluginmanager.h"

#include <KLocalizedString>

#include <QMessageBox>
#include <QPushButton>

SpendGuard::SpendGuard(QWidget *window)
    : QObject(window)
    , m_window(window)
{
    connect(&PluginManager::instance(), &PluginManager::jobFinished, this, [this](const QString &jobId, bool failed) {
        if (!m_taken.contains(jobId)) {
            return;
        }
        const QPair<QString, int> taken = m_taken.take(jobId);
        // nothing was made, and the server gives the credits back: so does the allowance
        if (failed && m_allowance.contains(taken.first)) {
            m_allowance[taken.first] += taken.second;
            Q_EMIT allowanceChanged();
        }
    });
}

void SpendGuard::settle(int request, const QString &job, const QString &by, const Spend &spend)
{
    QJsonObject state{{QStringLiteral("request"), request}, {QStringLiteral("by"), by}};
    if (job.isEmpty() || job.startsWith(QLatin1String("gate:"))) {
        // the card or the effect changed since the price was named
        state.insert(QStringLiteral("state"), QStringLiteral("stale"));
        if (by == QLatin1String("allowance") && spend.price > 0) {
            m_allowance[spend.pluginId] += spend.price;
            Q_EMIT allowanceChanged();
        }
    } else {
        state.insert(QStringLiteral("state"), QStringLiteral("accepted"));
        state.insert(QStringLiteral("job"), job);
        if (by == QLatin1String("allowance") && spend.price > 0) {
            m_taken.insert(job, {spend.pluginId, spend.price});
        }
    }
    state.insert(QStringLiteral("left"), m_allowance.value(spend.pluginId, 0));
    m_states.insert(request, state);
}

int SpendGuard::request(const Spend &spend)
{
    if (m_open.contains(spend.target)) {
        return m_open.value(spend.target);
    }
    const int id = ++m_next;
    const int left = m_allowance.value(spend.pluginId, 0);
    if (spend.price >= 0 && left > 0 && spend.price <= left) {
        if (spend.stillValid && !spend.stillValid()) {
            settle(id, QString(), QStringLiteral("allowance"), Spend());
            return id;
        }
        // taken before the run, so two runs at once cannot both fit in what is left
        m_allowance[spend.pluginId] = left - spend.price;
        Q_EMIT allowanceChanged();
        settle(id, spend.run(QStringLiteral("allowance")), QStringLiteral("allowance"), spend);
        return id;
    }

    m_states.insert(id, QJsonObject{{QStringLiteral("request"), id}, {QStringLiteral("state"), QStringLiteral("waiting")}});
    m_open.insert(spend.target, id);
    const QString plugin = PluginManager::instance().plugin(spend.pluginId).name();
    QString text = QStringLiteral("<b>%1</b><br/>%2").arg(spend.title.toHtmlEscaped(), PluginEffects::linkify(spend.sentence));
    text.append(QStringLiteral("<br/><br/>") + i18n("The assistant asks to run it").toHtmlEscaped());
    auto *box = new QMessageBox(QMessageBox::Question, plugin.isEmpty() ? spend.pluginId : plugin, text, QMessageBox::NoButton, m_window);
    box->setTextFormat(Qt::RichText);
    box->setAttribute(Qt::WA_DeleteOnClose);
    QPushButton *yes = box->addButton(i18n("Generate"), QMessageBox::AcceptRole);
    QPushButton *no = box->addButton(QMessageBox::Cancel);
    // a stray Enter must not spend money
    box->setDefaultButton(no);
    connect(box, &QMessageBox::finished, this, [this, id, spend, box, yes]() {
        m_open.remove(spend.target);
        if (box->clickedButton() != yes) {
            m_states.insert(id, QJsonObject{{QStringLiteral("request"), id}, {QStringLiteral("state"), QStringLiteral("declined")}});
            return;
        }
        if (spend.stillValid && !spend.stillValid()) {
            settle(id, QString(), QStringLiteral("user"), Spend());
            return;
        }
        settle(id, spend.run(QStringLiteral("user")), QStringLiteral("user"), spend);
    });
    box->open();
    return id;
}

int SpendGuard::askAllowance(const QString &pluginId, int credits)
{
    const int id = ++m_next;
    const PluginManifest manifest = PluginManager::instance().plugin(pluginId);
    if (manifest.id().isEmpty()) {
        m_states.insert(id, QJsonObject{{QStringLiteral("request"), id}, {QStringLiteral("state"), QStringLiteral("failed")},
                                        {QStringLiteral("message"), QStringLiteral("no plugin '%1'").arg(pluginId)}});
        return id;
    }
    if (credits <= 0) {
        m_states.insert(id, QJsonObject{{QStringLiteral("request"), id}, {QStringLiteral("state"), QStringLiteral("accepted")},
                                        {QStringLiteral("left"), m_allowance.value(pluginId, 0)}});
        return id;
    }
    const QString target = QStringLiteral("allowance:") + pluginId;
    if (m_open.contains(target)) {
        return m_open.value(target);
    }
    m_states.insert(id, QJsonObject{{QStringLiteral("request"), id}, {QStringLiteral("state"), QStringLiteral("waiting")}});
    m_open.insert(target, id);
    auto *box = new QMessageBox(QMessageBox::Question, manifest.name(),
                                i18n("The assistant asks to spend up to %1 credits without asking each time", credits), QMessageBox::NoButton, m_window);
    box->setAttribute(Qt::WA_DeleteOnClose);
    QPushButton *yes = box->addButton(i18n("Allow"), QMessageBox::AcceptRole);
    QPushButton *no = box->addButton(QMessageBox::Cancel);
    box->setDefaultButton(no);
    connect(box, &QMessageBox::finished, this, [this, id, target, pluginId, credits, box, yes]() {
        m_open.remove(target);
        if (box->clickedButton() != yes) {
            m_states.insert(id, QJsonObject{{QStringLiteral("request"), id}, {QStringLiteral("state"), QStringLiteral("declined")}});
            return;
        }
        // a new allowance replaces the old one rather than adding to it
        m_allowance.insert(pluginId, credits);
        Q_EMIT allowanceChanged();
        m_states.insert(id, QJsonObject{{QStringLiteral("request"), id}, {QStringLiteral("state"), QStringLiteral("accepted")},
                                        {QStringLiteral("left"), credits}});
    });
    box->open();
    return id;
}

QJsonObject SpendGuard::state(int request) const
{
    return m_states.value(request, QJsonObject{{QStringLiteral("request"), request}, {QStringLiteral("state"), QStringLiteral("failed")},
                                               {QStringLiteral("message"), QStringLiteral("no such request")}});
}

int SpendGuard::allowance(const QString &pluginId) const
{
    return m_allowance.value(pluginId, 0);
}

QStringList SpendGuard::allowed() const
{
    QStringList found;
    for (auto it = m_allowance.constBegin(); it != m_allowance.constEnd(); ++it) {
        if (it.value() > 0) {
            found << it.key();
        }
    }
    found.sort();
    return found;
}

void SpendGuard::revoke(const QString &pluginId)
{
    if (m_allowance.remove(pluginId) > 0) {
        Q_EMIT allowanceChanged();
    }
}
