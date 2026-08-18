/*
    SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QColor>
#include <QObject>

/**
 * Central authority for the two Wunjo color themes (dark default / light) and
 * the user-selectable primary/accent color.
 *
 * - Theme is applied via KColorSchemeManager (schemes "Wunjo" / "WunjoLight"),
 *   which recolors every palette-driven (Breeze) widget.
 * - The accent + theme greys are baked into the global stylesheet
 *   (src/assets/style.qss) as literal hexes; applyTokens() rewrites them so both
 *   the QSS chrome and the chat stylesheet follow the current theme + accent.
 *   The mint literal #C8EDD2 stands for the *ink* role (see accentInk()); the
 *   fill role, where the accent is a background carrying dark glyphs, is written
 *   as the named tokens @accent-fill / @on-accent because no single hex can mean
 *   both once the light theme pulls the two apart.
 * - Exposed to QML as the context property "wunjoTheme" (see Core::sharedQmlEngine),
 *   so timeline/monitor accents update live via the `accent` NOTIFY property.
 *
 * By construction the dark theme with the default mint accent produces a
 * byte-identical stylesheet to the original, so the current look is unchanged.
 */
class WunjoTheme : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QColor accent READ accent NOTIFY accentChanged)
    Q_PROPERTY(bool dark READ isDark NOTIFY themeChanged)

public:
    static WunjoTheme *instance();

    QColor accent() const { return m_accent; }
    bool isDark() const { return m_dark; }

    /** Derived accent shades used by the stylesheet tokens. */
    QColor accentHover() const;
    QColor accentTint() const;

    /** @brief The accent as *ink* — accent text, borders, underlines, the reply
     *  dot: everything drawn directly on the window background.
     *
     * On the dark theme this is the accent itself. On the light theme a pastel
     * accent cannot play this role (the default mint scores 1.3:1 on white, i.e.
     * invisible), so the colour keeps its hue and saturation and loses lightness
     * until it clears 4.5:1. This is deliberately not a special case for mint:
     * the colour dialog behind "Primary Color → Custom…" hands out pastels too.
     */
    QColor accentInk() const;

    /** @brief Text and glyphs placed *on* the accent fill (selection blocks, the
     *  chat send button), i.e. the counterpart of accentInk(). */
    QColor onAccent() const;

    /** Rewrite the mint/grey literal hexes in @p css for the current theme + accent. */
    QString applyTokens(QString css) const;
    /** The global stylesheet (:/data/style.qss) rewritten for the current theme + accent. */
    QString buildStyleSheet() const;

    /** Load the persisted theme + accent from WunjoSettings and apply. Call once at startup. */
    void init();
    /** Re-activate the scheme and re-apply the stylesheet, then notify. */
    void apply();

public Q_SLOTS:
    void setDark(bool dark);
    void setAccent(const QColor &accent);

Q_SIGNALS:
    void accentChanged(const QColor &accent);
    void themeChanged(bool dark);

private:
    explicit WunjoTheme();
    void applyScheme();
    /** @brief Push the chosen accent into the application palette, so widgets
     *  that draw from it follow the theme instead of the scheme's mint. */
    void applyPalette();
    void applyIconTheme();

    bool m_dark = true;
    QColor m_accent = QColor(200, 237, 210); // mint #C8EDD2
};
