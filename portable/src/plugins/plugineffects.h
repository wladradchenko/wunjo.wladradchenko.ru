/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "pluginmanifest.h"

#include <QJsonObject>
#include <QList>
#include <QString>
#include <memory>

class AssetParameterModel;
class EffectStackModel;

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

} // namespace PluginEffects
