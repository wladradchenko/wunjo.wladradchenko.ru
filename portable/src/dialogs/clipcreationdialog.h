/*
SPDX-FileCopyrightText: 2015 Jean-Baptiste Mardelle <jb@kdenlive.org>
This file is part of Wunjo. See www.wunjo.online.

SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QDockWidget>

class WunjoDoc;
class Bin;
class ProjectClip;
class ProjectItemModel;

/**
 * @namespace ClipCreationDialog
 * @brief This namespace contains a list of static methods displaying widgets
 *  allowing creation of all clip types.
 */
namespace ClipCreationDialog {

void createColorClip(
    WunjoDoc *doc, const QString &parentFolder, std::shared_ptr<ProjectItemModel> model,
    const std::function<void(const QString &)> &readyCallBack = [](const QString &) {}, int suggestedDuration = -1);
void createQTextClip(
    const QString &parentId, Bin *bin, ProjectClip *clip = nullptr, const std::function<void(const QString &)> &readyCallBack = [](const QString &) {},
    int suggestedDuration = -1);
void createAnimationClip(
    WunjoDoc *doc, const QString &parentId, const std::function<void(const QString &)> &readyCallBack = [](const QString &) {}, int suggestedDuration = -1);
void createSlideshowClip(
    WunjoDoc *doc, const QString &parentId, std::shared_ptr<ProjectItemModel> model,
    const std::function<void(const QString &)> &readyCallBack = [](const QString &) {}, int suggestedDuration = -1);
void createTitleClip(
    WunjoDoc *doc, const QString &parentFolder, const QString &templatePath, std::shared_ptr<ProjectItemModel> model,
    const std::function<void(const QString &)> &readyCallBack = [](const QString &) {}, int suggestedDuration = -1);
void createTitleTemplateClip(
    WunjoDoc *doc, const QString &parentFolder, std::shared_ptr<ProjectItemModel> model,
    const std::function<void(const QString &)> &readyCallBack = [](const QString &) {}, int suggestedDuration = -1);
void createClipsCommand(
    WunjoDoc *doc, const QString &parentFolder, const std::shared_ptr<ProjectItemModel> &model,
    const std::function<void(const QString &)> &readyCallBack = [](const QString &) {}, int suggestedDuration = -1);
const QString createPlaylistClip(
    const QString &name, std::pair<int, int> tracks, const QString &parentFolder, std::shared_ptr<ProjectItemModel> model,
    const std::function<void(const QString &)> &readyCallBack = [](const QString &) {}, int suggestedDuration = -1);

} // namespace ClipCreationDialog
