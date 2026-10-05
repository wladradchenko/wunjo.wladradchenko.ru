/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "pluginmanifest.h"

#include <QJsonObject>
#include <QList>
#include <QString>
#include <functional>
#include <memory>

class AssetParameterModel;
class EffectStackModel;
class QAction;
class QMenu;

/** @namespace PluginEffects
    @brief Putting the effects a plugin brought on a clip.

    A plugin can split its work over two effects: one that marks the region it
    works on (a head, over time) and one that holds the animation and reads that
    region — `wunjo_requires` in the effect XML ties them. Both live in the
    effect stack with their own keyframes, and both are applied together with a
    shared bond id, so the region can be corrected without touching the
    animation.
 */
namespace PluginEffects {

/** @brief The effects of @p plugin that are offered in a menu: the ones another
 *  effect requires come along with it and are not entries of their own. */
QList<PluginEffect> menuEffects(const PluginManifest &plugin);

/** @brief One thing a plugin offers in a menu: an effect, its set of effects,
 *  a card, or a plain run. */
struct MenuEntry {
    QString text;
    std::function<void()> trigger;
};

/** @brief How many things @p plugin offers across all menus: its effects (one
 *  when they apply together), its cards, or one run when it has neither. */
int declaredActions(const PluginManifest &plugin);

/** @brief Hang @p plugin in @p menu with the @p entries that fit where the menu
 *  was opened.
 *
 *  A plugin that offers one thing is one row named after the plugin. One that
 *  offers several is a submenu named after it, even where only some of them
 *  fit: effects of different plugins are often called alike, and a flat list
 *  mixed them up. A submenu also ends with Update when the site has a newer
 *  release of the plugin. Returns the action added to @p menu (a submenu's
 *  own), or nullptr when @p entries is empty. */
QAction *addToMenu(QMenu *menu, const PluginManifest &plugin, const QList<MenuEntry> &entries);

/** @brief Remove from @p menu what @ref addToMenu put there, recognised by
 *  @p objectName, or every entry when it is empty. A submenu goes with its
 *  menu, which a plain QMenu::clear() would leave behind. */
void clearMenuEntries(QMenu *menu, const QString &objectName = QString());

/** @brief What ties @p effect to the other half of its pair: the value of its
 *  `wunjo_fill="region"` parameter, prefixed with its plugin so two plugins
 *  never match. Empty for an effect that is not one half of a plugin pair. */
QString bondOf(const std::shared_ptr<AssetParameterModel> &effect);

/** @brief A short mark telling @p effect's pair from the other pairs of the
 *  same plugin on @p stack ("a3f9"), the same on both halves; empty when the
 *  plugin has only one pair there, where a mark would only be noise. */
QString pairMark(const std::shared_ptr<EffectStackModel> &stack, const std::shared_ptr<AssetParameterModel> &effect);

/** @brief True when @p effectId is a region: another effect of its plugin
 *  declares `wunjo_requires` on it. */
bool isRegion(const QString &effectId);

/** @brief Add @p effect to @p stack, preceded by the region effect it requires.
 *  @p faceTrack (keyframes of an animated rect, may be empty) fills the
 *  parameters marked `wunjo_fill="face"`. Returns false if nothing was added. */
bool apply(const std::shared_ptr<EffectStackModel> &stack, const PluginManifest &plugin, const PluginEffect &effect, const QString &faceTrack);

/** @brief True when @p effect or the region it requires wants a face track. */
bool needsFace(const PluginManifest &plugin, const PluginEffect &effect);
/** @brief The effect to apply when @p effectId is asked for.
 *
 *  A plugin's region effect exists only to mark where another one works. The
 *  menus never offer it on its own, but a caller reading the manifest can ask
 *  for it by id — and used to get exactly that: a clip that looks treated,
 *  carries no worker effect, and cannot be rendered, because the render belongs
 *  to the effect that requires the region rather than to the region itself.
 *  Asking for the marker means asking for the work, so that is what comes back;
 *  for anything else the id is returned unchanged. */
QString effectToApply(const QString &effectId);

/** @brief Add every effect of @p plugin to @p stack at once, on one region and
 *  one bond — for a plugin whose effects are three ways of working on the same
 *  face. Returns false if nothing was added. */
bool applyAll(const std::shared_ptr<EffectStackModel> &stack, const PluginManifest &plugin, const QString &faceTrack);

/** @brief The job that makes @p pluginId produce what its effect describes.
 *
 *  Everything the plugin needs is in there: the clip it sits on with its range,
 *  and every effect of that plugin on that clip with all of its parameters —
 *  keyframed ones as the animation string they hold, so a plugin can follow them
 *  frame by frame. A pair (a region and what works inside it) therefore arrives
 *  whole, and the plugin matches the two by their shared bond itself.
 *  Returns an empty object when the clip cannot be resolved. */
QJsonObject buildJob(const std::shared_ptr<AssetParameterModel> &model, const QString &pluginId);

/** @brief The value of `<jobparam name="@p name">` on any render parameter of
 *  the effect: "gate" and "action" name the price question, "key" the
 *  parameter the result is written into. Empty when none says it. */
QString jobParam(const std::shared_ptr<AssetParameterModel> &model, const QString &name);

/** @brief The effect's job as a price question is asked about it: an answer
 *  holds only while this stays the same. */
QByteArray askedJob(const std::shared_ptr<AssetParameterModel> &model, const QString &pluginId);

/** @brief Ask the plugin @p action (what the render would cost) for the effect
 *  as it is now. The answer is kept by the plugin manager for the effect, so
 *  its panel and an assistant read the same one. False when the effect is not
 *  on a clip a job can be made of. */
bool askEffect(const std::shared_ptr<AssetParameterModel> &model, const QString &pluginId, int effectItemId, const QString &action);

/** @brief Whether the price question @p gate has been answered yes for the
 *  effect as it is now; @p price gets the price the plugin named, -1 for none. */
bool gateOpen(const std::shared_ptr<AssetParameterModel> &model, const QString &pluginId, int effectItemId, const QString &gate, int *price = nullptr);

/** @brief Rich text of a plugin's sentence with its web addresses clickable. */
QString linkify(const QString &text);

} // namespace PluginEffects
