/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "plugineffects.h"

#include "assets/model/assetparametermodel.hpp"
#include "bin/projectclip.h"
#include "bin/projectitemmodel.h"
#include "core.h"
#include "definitions.h"
#include "effects/effectstack/model/effectitemmodel.hpp"
#include "effects/effectstack/model/effectstackmodel.hpp"
#include "pluginmanager.h"

#include <QJsonArray>
#include <QSet>
#include <QUuid>

namespace {

/** @brief The effect @p effect works inside, if it declares one. */
PluginEffect requiredEffect(const PluginManifest &plugin, const PluginEffect &effect)
{
    if (effect.requiresEffect.isEmpty()) {
        return {};
    }
    const QList<PluginEffect> effects = plugin.effects();
    for (const PluginEffect &candidate : effects) {
        if (candidate.id == effect.requiresEffect) {
            return candidate;
        }
    }
    return {};
}

bool appendOne(const std::shared_ptr<EffectStackModel> &stack, const PluginEffect &effect, const QString &faceTrack, const QString &bond)
{
    stringMap params;
    for (const QString &param : effect.faceParams) {
        if (!faceTrack.isEmpty()) {
            params.insert(param, faceTrack);
        }
    }
    for (const QString &param : effect.regionParams) {
        params.insert(param, bond);
    }
    return stack->appendEffect(effect.id, true, params);
}

} // namespace

namespace PluginEffects {

QList<PluginEffect> menuEffects(const PluginManifest &plugin)
{
    const QList<PluginEffect> effects = plugin.effects();
    QSet<QString> required;
    for (const PluginEffect &effect : effects) {
        if (!effect.requiresEffect.isEmpty()) {
            required.insert(effect.requiresEffect);
        }
    }
    QList<PluginEffect> result;
    for (const PluginEffect &effect : effects) {
        if (!required.contains(effect.id)) {
            result.append(effect);
        }
    }
    return result;
}

QString effectToApply(const QString &effectId)
{
    const QList<PluginManifest> plugins = PluginManager::instance().installedPlugins();
    for (const PluginManifest &plugin : plugins) {
        const QList<PluginEffect> effects = plugin.effects();
        for (const PluginEffect &effect : effects) {
            if (effect.requiresEffect == effectId) {
                return effect.id;
            }
        }
    }
    return effectId;
}

bool needsFace(const PluginManifest &plugin, const PluginEffect &effect)
{
    const PluginEffect region = requiredEffect(plugin, effect);
    return !effect.faceParams.isEmpty() || !region.faceParams.isEmpty();
}

bool applyAll(const std::shared_ptr<EffectStackModel> &stack, const PluginManifest &plugin, const QString &faceTrack)
{
    if (stack == nullptr) {
        return false;
    }
    // One detection, one bond: the region goes on first and every effect that
    // works inside it follows, so the user tunes whichever one they came for.
    const QString bond = QUuid::createUuid().toString(QUuid::WithoutBraces).left(8);
    const QList<PluginEffect> effects = plugin.effects();
    QSet<QString> required;
    for (const PluginEffect &effect : effects) {
        if (!effect.requiresEffect.isEmpty()) {
            required.insert(effect.requiresEffect);
        }
    }
    bool added = false;
    for (const PluginEffect &effect : effects) {
        if (required.contains(effect.id)) {
            added = appendOne(stack, effect, faceTrack, bond) || added;
        }
    }
    for (const PluginEffect &effect : effects) {
        if (!required.contains(effect.id)) {
            added = appendOne(stack, effect, faceTrack, bond) || added;
        }
    }
    return added;
}

QJsonObject buildJob(const std::shared_ptr<AssetParameterModel> &model, const QString &pluginId)
{
    if (model == nullptr || pluginId.isEmpty()) {
        return {};
    }
    const ObjectId owner = model->getOwnerId();
    QString binId;
    if (owner.type == WunjoObjectType::BinClip) {
        binId = QString::number(owner.itemId);
    } else if (owner.type == WunjoObjectType::TimelineClip) {
        binId = pCore->getTimelineClipBinId(owner);
    }
    std::shared_ptr<ProjectClip> clip = binId.isEmpty() ? nullptr : pCore->projectItemModel()->getClipByBinID(binId);
    if (clip == nullptr) {
        return {};
    }
    QJsonObject source;
    source.insert(QStringLiteral("bin_id"), binId);
    source.insert(QStringLiteral("path"), clip->url());
    const int in = pCore->getItemIn(owner);
    source.insert(QStringLiteral("in"), in);
    source.insert(QStringLiteral("out"), in + qMax(0, pCore->getItemDuration(owner) - 1));

    // Hand over the effect that asked and the region it works inside — nothing
    // else. Sister effects of the same plugin are other jobs; letting them into
    // this one only invites the plugin to read the wrong parameters.
    const QString askedId = model->getAssetId();
    const PluginManifest manifest = PluginManager::instance().plugin(pluginId);
    PluginEffect asked;
    const QList<PluginEffect> known = manifest.effects();
    for (const PluginEffect &candidate : known) {
        if (candidate.id == askedId) {
            asked = candidate;
            break;
        }
    }
    const PluginEffect region = requiredEffect(manifest, asked);

    QJsonArray effects;
    std::shared_ptr<EffectStackModel> stack = pCore->getItemEffectStack(owner.uuid, int(owner.type), owner.itemId);
    if (stack) {
        auto readParams = [](const std::shared_ptr<EffectItemModel> &item) {
            QJsonObject params;
            for (int row = 0; row < item->rowCount(); ++row) {
                const QModelIndex ix = item->index(row, 0);
                // the value as the effect holds it — an animation string for a
                // keyframed parameter, so the plugin can follow it frame by frame
                params.insert(item->data(ix, AssetParameterModel::NameRole).toString(), item->data(ix, AssetParameterModel::ValueRole).toString());
            }
            return params;
        };
        // The bond tells one pair from another when the clip carries two faces.
        QString bond;
        QJsonObject askedParams;
        for (int i = 0; i < stack->rowCount(); ++i) {
            auto item = std::static_pointer_cast<EffectItemModel>(stack->getEffectStackRow(i));
            if (item == nullptr || item.get() != model.get()) {
                continue;
            }
            askedParams = readParams(item);
            for (const QString &param : asked.regionParams) {
                if (askedParams.contains(param)) {
                    bond = askedParams.value(param).toString();
                }
            }
            break;
        }
        if (!region.id.isEmpty()) {
            for (int i = 0; i < stack->rowCount(); ++i) {
                auto item = std::static_pointer_cast<EffectItemModel>(stack->getEffectStackRow(i));
                if (item == nullptr || item->getAssetId() != region.id) {
                    continue;
                }
                const QJsonObject params = readParams(item);
                bool mine = bond.isEmpty();
                for (const QString &param : region.regionParams) {
                    mine = mine || params.value(param).toString() == bond;
                }
                if (!mine) {
                    continue;
                }
                QJsonObject effect;
                effect.insert(QStringLiteral("id"), item->getAssetId());
                effect.insert(QStringLiteral("params"), params);
                effects.append(effect);
                break;
            }
        }
        if (!askedParams.isEmpty()) {
            QJsonObject effect;
            effect.insert(QStringLiteral("id"), askedId);
            effect.insert(QStringLiteral("params"), askedParams);
            effects.append(effect);
        }
    }

    QJsonObject input;
    input.insert(QStringLiteral("action"), QStringLiteral("generate"));
    // Which effect asked. A plugin whose effects sit on the clip together would
    // otherwise have to guess, and guessing means the wrong engine runs.
    input.insert(QStringLiteral("effect_id"), model->getAssetId());
    input.insert(QStringLiteral("clips"), QJsonArray{source});
    input.insert(QStringLiteral("effects"), effects);
    return input;
}

bool apply(const std::shared_ptr<EffectStackModel> &stack, const PluginManifest &plugin, const PluginEffect &effect, const QString &faceTrack)
{
    if (stack == nullptr) {
        return false;
    }
    // The bond only has to be unique inside one clip; it lets a plugin tell its
    // pair apart when the same clip carries two of them (two faces).
    const QString bond = QUuid::createUuid().toString(QUuid::WithoutBraces).left(8);
    bool added = false;
    const PluginEffect region = requiredEffect(plugin, effect);
    if (!region.id.isEmpty()) {
        added = appendOne(stack, region, faceTrack, bond);
    }
    return appendOne(stack, effect, faceTrack, bond) || added;
}

} // namespace PluginEffects
