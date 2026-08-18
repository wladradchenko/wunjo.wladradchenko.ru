/*
    SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "theme.h"
#include "wunjosettings.h"

#include <KConfigGroup>
#include <KSharedConfig>
#include <KColorSchemeManager>
#include <KIconTheme>

#include <QApplication>
#include <QCoreApplication>
#include <QEvent>
#include <QDir>
#include <QRegularExpression>
#include <QStandardPaths>
#include <QFile>
#include <QGuiApplication>
#include <QIcon>
#include <QList>
#include <QPalette>
#include <QPixmapCache>
#include <QWindow>

#include <cmath>

namespace {
QString hx(const QColor &c)
{
    return c.name(QColor::HexRgb).toUpper();
}

QColor blend(const QColor &a, const QColor &b, qreal t)
{
    return QColor(qRound(a.red() * t + b.red() * (1 - t)), qRound(a.green() * t + b.green() * (1 - t)), qRound(a.blue() * t + b.blue() * (1 - t)));
}

/** WCAG relative luminance, and the contrast a colour reaches against white —
 *  the light theme's base. 4.5:1 is the body-text threshold, 3:1 the one for
 *  borders and other non-text UI. */
qreal luminance(const QColor &c)
{
    const auto channel = [](qreal v) { return v <= 0.03928 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * channel(c.redF()) + 0.7152 * channel(c.greenF()) + 0.0722 * channel(c.blueF());
}

qreal contrastOnWhite(const QColor &c)
{
    return 1.05 / (luminance(c) + 0.05);
}

// Canonical mint accent shades from data/color-schemes/Wunjo.colors + style.qss.
const QColor kMint(200, 237, 210);    // #C8EDD2
const QColor kMintDim(162, 224, 178); // #A2E0B2
const QColor kMintTint(43, 58, 50);   // #2B3A32
} // namespace

WunjoTheme *WunjoTheme::instance()
{
    static WunjoTheme *s_instance = new WunjoTheme;
    return s_instance;
}

WunjoTheme::WunjoTheme()
    : QObject(nullptr)
{
}

QColor WunjoTheme::accentInk() const
{
    if (m_dark) {
        return m_accent;
    }
    float h, s, l, a;
    m_accent.getHslF(&h, &s, &l, &a);
    if (h < 0) {
        h = 0; // achromatic accent: no hue to preserve, only lightness matters
    }
    QColor ink = m_accent;
    for (int i = 0; i < 100 && contrastOnWhite(ink) < 4.5; ++i) {
        l = qMax(0.0f, l - 0.01f);
        ink = QColor::fromHslF(h, s, l, a);
    }
    return ink;
}

QColor WunjoTheme::onAccent() const
{
    // The accent fill keeps its pastel value in both themes, so what sits on it
    // follows the fill's own lightness, not the theme's.
    return m_accent.lightness() > 140 ? QColor(13, 13, 13) : QColor(255, 255, 255);
}

QColor WunjoTheme::accentHover() const
{
    if (!m_dark) {
        // Hover/focus feedback lands on the light background just like the ink
        // does, so it is a step deeper into the ink rather than off the pastel.
        return accentInk().darker(115);
    }
    if (m_accent == kMint) {
        return kMintDim;
    }
    // A slightly deeper accent for hover/pressed feedback.
    return m_accent.lightness() > 128 ? m_accent.darker(112) : m_accent.lighter(118);
}

QColor WunjoTheme::accentTint() const
{
    if (m_accent == kMint && m_dark) {
        return kMintTint; // keep the current dark theme pixel-identical
    }
    // Selection/checked backgrounds: a faint wash of the accent over the base.
    return m_dark ? blend(m_accent, QColor(20, 20, 20), 0.20) : blend(m_accent, QColor(255, 255, 255), 0.26);
}

QString WunjoTheme::applyTokens(QString css) const
{
    struct Pair
    {
        QString from;
        QString to;
    };
    QList<Pair> map;

    // Accent tokens — substituted in both themes (identity for the dark default,
    // where ink and fill are the same mint).
    map.append({QStringLiteral("@accent-fill"), hx(m_accent)});
    map.append({QStringLiteral("@on-accent"), hx(onAccent())});
    map.append({QStringLiteral("#C8EDD2"), hx(accentInk())});
    map.append({QStringLiteral("#A2E0B2"), hx(accentHover())});
    map.append({QStringLiteral("#2B3A32"), hx(accentTint())});

    if (!m_dark) {
        // Theme greys — invert the dark-mode lightness hierarchy. Only in light
        // mode, so dark mode keeps the original literals untouched.
        // Matched to Wunjo Design's light palette: the page is white, panels sit
        // a hair off it rather than a shade of grey, borders are barely there,
        // and the text is a soft near-black — a hard #0D0D0D on white reads as
        // harsher than anything else in the family.
        map.append({QStringLiteral("#0D0D0D"), QStringLiteral("#FFFFFF")}); // base / on-accent text
        map.append({QStringLiteral("#161616"), QStringLiteral("#FAFBFC")}); // menus, headers
        map.append({QStringLiteral("#1F1F1F"), QStringLiteral("#F5F6F8")}); // surfaces, inputs, cards
        map.append({QStringLiteral("#2D2D2D"), QStringLiteral("#E6E8EB")}); // borders, scrollbars
        map.append({QStringLiteral("#2E2E2E"), QStringLiteral("#FFFFFF")}); // tooltips
        map.append({QStringLiteral("#696969"), QStringLiteral("#6B7280")}); // dim text
        map.append({QStringLiteral("#FFFFFF"), QStringLiteral("#1F2328")}); // primary text
        // Chat-only tints (dark greens / error reds) kept legible on light.
        map.append({QStringLiteral("#223529"), hx(accentTint())});
        map.append({QStringLiteral("#3A5A46"), hx(accentHover())});
        map.append({QStringLiteral("#3A181D"), QStringLiteral("#FBE4E7")});
        map.append({QStringLiteral("#FF9C9C"), QStringLiteral("#C0392B")});
    }

    // Two-pass replace via unique private-use sentinels so overlapping swaps
    // (e.g. #0D0D0D <-> #FFFFFF in light mode) never clobber each other.
    const QChar sentOpen(0xE000);
    const QChar sentClose(0xE001);
    for (int i = 0; i < map.size(); ++i) {
        const QString sentinel = QString(sentOpen) + QString::number(i) + sentClose;
        css.replace(map.at(i).from, sentinel, Qt::CaseInsensitive);
    }
    for (int i = 0; i < map.size(); ++i) {
        const QString sentinel = QString(sentOpen) + QString::number(i) + sentClose;
        css.replace(sentinel, map.at(i).to);
    }
    return css;
}

QString WunjoTheme::buildStyleSheet() const
{
    QFile f(QStringLiteral(":/data/style.qss"));
    if (!f.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return QString();
    }
    return applyTokens(QString::fromUtf8(f.readAll()));
}

void WunjoTheme::applyScheme()
{
    KColorSchemeManager::instance()->activateSchemeId(m_dark ? QStringLiteral("Wunjo") : QStringLiteral("WunjoLight"));
}

void WunjoTheme::applyIconTheme()
{
    // The bundled icons bake a white (#fcfcfc) stroke and are loaded via plain
    // QIcon (no palette recoloring), so a light theme needs a dark-stroke copy.
    // "wunjo-light" is generated from "wunjo" at build time (data/icons-wunjo).
    const QString iconTheme = m_dark ? QStringLiteral("wunjo") : QStringLiteral("wunjo-light");
    if (QIcon::themeName() == iconTheme) {
        return;
    }
    QIcon::setFallbackThemeName(m_dark ? QStringLiteral("breeze-dark") : QStringLiteral("breeze"));
    QIcon::setThemeName(iconTheme);
    KIconTheme::forceThemeForTests(iconTheme);
    QPixmapCache::clear();
    // Prod already-created windows to re-resolve their icons (no-op at startup
    // when none exist yet — icons then load with the correct theme directly).
    QEvent themeEvent(QEvent::ThemeChange);
    const auto windows = QGuiApplication::topLevelWindows();
    for (QWindow *w : windows) {
        QCoreApplication::sendEvent(w, &themeEvent);
    }
}

void WunjoTheme::applyPalette()
{
    if (qApp == nullptr) {
        return;
    }
    // The colour scheme files carry the mint accent, so everything drawn from
    // the palette rather than from our stylesheet — the selection in the bin and
    // the effect lists, the highlighted row in every list — stayed mint while
    // the user had chosen another accent. The scheme is loaded first and its
    // highlight overwritten here, so one setting really does colour the whole
    // application.
    QPalette palette = qApp->palette();
    // On the dark theme the selection is the accent damped towards the base
    // rather than the accent itself: text stays white on it, which is what
    // every icon in the application already is. Flipping the label to black on
    // a bright mint left those icons — pale monochrome strokes — invisible, and
    // they cannot follow the text the way a colour can.
    const QColor selection = m_dark ? blend(m_accent, QColor(20, 20, 20), 0.45) : m_accent;
    const QColor onSelection = m_dark ? QColor(255, 255, 255) : onAccent();
    palette.setColor(QPalette::Highlight, selection);
    palette.setColor(QPalette::HighlightedText, onSelection);
    palette.setColor(QPalette::Disabled, QPalette::Highlight, accentTint());
    palette.setColor(QPalette::Disabled, QPalette::HighlightedText, onSelection);
    // Links follow the accent too; a mint link on a blue theme reads as a
    // leftover rather than as a link. A link is ink by definition — on the light
    // theme it takes the darkened shade, not the pastel one.
    palette.setColor(QPalette::Link, accentInk());
    palette.setColor(QPalette::LinkVisited, accentHover());
    qApp->setPalette(palette);

}

void WunjoTheme::apply()
{
    applyScheme();
    applyPalette();
    applyIconTheme();
    if (qApp) {
        qApp->setStyleSheet(buildStyleSheet());
    }
    Q_EMIT themeChanged(m_dark);
    Q_EMIT accentChanged(m_accent);
}

void WunjoTheme::init()
{
    m_dark = WunjoSettings::apptheme() == 0; // 0 = dark, 1 = light
    const QColor c = WunjoSettings::accentcolor();
    if (c.isValid()) {
        m_accent = c;
    }
    apply();
}

void WunjoTheme::setDark(bool dark)
{
    m_dark = dark;
    WunjoSettings::setApptheme(dark ? 0 : 1);
    WunjoSettings::self()->save();
    apply();
}

void WunjoTheme::setAccent(const QColor &accent)
{
    if (!accent.isValid()) {
        return;
    }
    m_accent = accent;
    WunjoSettings::setAccentcolor(accent);
    WunjoSettings::self()->save();
    apply();
}
