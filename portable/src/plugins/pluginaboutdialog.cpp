/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "pluginaboutdialog.h"

#include <KAboutApplicationDialog>
#include <KAboutData>
#include <KLocalizedString>

#include <QTabBar>
#include <QTabWidget>
#include <QTextBrowser>

namespace {

/** @brief The plugin, described the way KAboutData describes an application.
 *
 * Reusing KAboutData means reusing KDE's own About window rather than drawing
 * something that resembles it: the same header, the same person rows with their
 * mail button, the same component list. What differs is only what goes in.
 */
KAboutData aboutDataFor(const PluginManifest &manifest)
{
    // The licence by name — "MIT" in a manifest is the MIT licence, and showing
    // it as "Custom" only hides what it says.
    // byKeyword understands the names people actually write in a manifest —
    // "MIT", "GPL-3.0", "Apache-2.0" — and falls back to Custom on its own.
    const KAboutLicense::LicenseKey license =
        manifest.license().isEmpty() ? KAboutLicense::Unknown : KAboutLicense::byKeyword(manifest.license()).key();
    KAboutData about(manifest.id().toUtf8(), manifest.name(), manifest.version(), manifest.description(), license, QString(), QString(), manifest.homepage());
    if (!manifest.author().isEmpty()) {
        about.addAuthor(manifest.author(), i18n("Author and maintainer"), manifest.authorEmail());
    }
    // KDE fills the author page with where to report bugs and where to ask for
    // help — at bugs.kde.org and kde.org/support, neither of which knows
    // anything about somebody's plugin. Emptied, the page is just the author.
    about.setCustomAuthorText(QString(), QString());
    return about;
}

} // namespace

QList<QPair<QString, QString>> PluginAboutDialog::usageSteps(const PluginManifest &manifest)
{
    QList<QPair<QString, QString>> steps;
    const QList<PluginEffect> effects = manifest.effects();
    const QStringList targets = manifest.targets();

    if (targets.contains(QLatin1String("agent"))) {
        steps.append({i18n("Ask in the Chat panel"),
                      i18n("Say what you want done. The assistant works in the editor and mirrors what it does back into the chat.")});
    }
    if (targets.contains(QLatin1String("face"))) {
        steps.append({i18n("Pick a face in the Project Monitor"),
                      i18n("Turn on face detection for the clip, then click the face you want — the plugin works on that one.")});
    }
    if (targets.contains(QLatin1String("video")) || targets.contains(QLatin1String("audio"))) {
        const QString kind = targets.contains(QLatin1String("video")) && targets.contains(QLatin1String("audio"))
            ? i18n("a video or audio clip")
            : (targets.contains(QLatin1String("video")) ? i18n("a video clip") : i18n("an audio clip"));
        steps.append({i18n("Select a clip on the timeline"),
                      i18n("Right-click %1 and choose this plugin from the Artificial Intelligence submenu.", kind)});
    }
    if (targets.contains(QLatin1String("generator"))) {
        steps.append({i18n("Run it from the Artificial Intelligence menu"), i18n("What it makes lands in the project bin.")});
    }
    if (!effects.isEmpty()) {
        QStringList names;
        for (const PluginEffect &effect : effects) {
            names << effect.name;
        }
        steps.append({i18n("It works through effects"),
                      i18n("Add %1 to the clip and set it up in the Effect/Composition Stack; it renders with the project.",
                           names.join(i18nc("separator in a list of effect names", ", ")))});
    }
    const PluginSetsUi sets = manifest.setsUi();
    if (!sets.label.isEmpty()) {
        steps.append({i18n("Register what it needs first"), i18n("On this page: %1. Then choose it in the effect's parameters.", sets.label)});
    }
    if (manifest.kind() == QLatin1String("api")) {
        steps.append({i18n("It needs an API key"), i18n("This plugin works through an online service, so it needs a key on this page and a connection.")});
    } else if (manifest.hasDependencies()) {
        steps.append({i18n("The first run installs its environment"), i18n("That download takes minutes; afterwards the plugin runs offline.")});
    }
    return steps;
}

namespace {

/** @brief The steps, as a plain list — no headings, nothing but what to do. */
QString usageHtml(const PluginManifest &manifest)
{
    const QList<QPair<QString, QString>> steps = PluginAboutDialog::usageSteps(manifest);
    if (steps.isEmpty()) {
        return QStringLiteral("<p>%1</p>").arg(i18n("This plugin does not say how it is meant to be used."));
    }
    QString html = QStringLiteral("<ul style='margin-left:0'>");
    for (const auto &step : steps) {
        html += QStringLiteral("<li style='margin-bottom:10px'>%1</li>").arg(step.second.toHtmlEscaped());
    }
    html += QStringLiteral("</ul>");
    return html;
}

} // namespace

void PluginAboutDialog::show(const PluginManifest &manifest, QWidget *parent)
{
    // no translators tab for a plugin: it has none, and an empty tab reads as
    // something missing
    KAboutApplicationDialog dialog(aboutDataFor(manifest), KAboutApplicationDialog::HideTranslators, parent);
    dialog.setWindowTitle(i18n("About %1", manifest.name()));
    if (auto *tabs = dialog.findChild<QTabWidget *>()) {
        // The components tab lists Qt, KDE Frameworks and the runtime — true of
        // the application, meaningless for a plugin.
        for (int i = tabs->count() - 1; i >= 0; --i) {
            if (tabs->tabText(i).remove(QLatin1Char('&')) == i18n("Components")) {
                tabs->removeTab(i);
            }
        }
        auto *usage = new QTextBrowser(tabs);
        usage->setOpenExternalLinks(true);
        usage->setFrameShape(QFrame::NoFrame);
        usage->viewport()->setAutoFillBackground(false);
        usage->setStyleSheet(QStringLiteral("QTextBrowser { background: transparent; }"));
        usage->setHtml(usageHtml(manifest));
        tabs->addTab(usage, i18n("How to use"));
    }
    dialog.exec();
}
