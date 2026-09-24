/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

/** MainWindow scripting API — D-Bus methods driving the editor for the MCP
    server in mcp/ (assistant "hands"). Ported from the D-Ogi/kdenlive fork's
    scripting patch set and adapted to the Wunjo rebrand; effect-expression
    methods are stubbed until the expressions subsystem is ported. */

#include "mainwindow.h"
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QTemporaryFile>
#include <QTimer>
#include "assets/assetpanel.hpp"
#include "assets/keyframes/model/keyframemodel.hpp"
#include "assets/keyframes/model/keyframemodellist.hpp"
#include "audiomixer/mixermanager.hpp"
#include "bin/bincommands.h"
#include "bin/clipcreator.hpp"
#include "bin/generators/generators.h"
#include "bin/mediabrowser.h"
#include "bin/model/subtitlemodel.hpp"
#include "bin/projectclip.h"
#include "bin/projectfolder.h"
#include "bin/projectitemmodel.h"
#include "ai/facedatastore.h"
#include "ai/faceeffect.h"
#include "chat/chatguidancestore.h"
#include "chat/chatwidget.h"
#include "core.h"
#include "dialogs/speechdialog.h"
#include "jobs/facedetecttask.h"
#include "assets/keyframes/model/automask/automaskhelper.hpp"
#include "jobs/melttask.h"
#include "plugins/plugineffects.h"
#include "plugins/pluginmanager.h"
#include "plugins/pluginsetstore.h"
#include "wunjosettings.h"
#include "dialogs/clipcreationdialog.h"
#include "dialogs/clipjobmanager.h"
#include "dialogs/renderwidget.h"
#include "dialogs/settings/wunjosettingsdialog.h"
#include "dialogs/subtitleedit.h"
#include "dialogs/wizard.h"
#include "doc/docundostack.hpp"
#include "doc/wunjodoc.h"
#include "doc/kthumb.h"
#include "effects/effectbasket.h"
#include "effects/effectlist/view/effectlistwidget.hpp"
#include "effects/effectstack/model/effectitemmodel.hpp"
#include "assets/model/assetparametermodel.hpp"
#include "effects/effectstack/model/effectstackmodel.hpp"
#include "jobs/audiolevels/audiolevelstask.h"
#include "jobs/customjobtask.h"
#include "jobs/scenesplittask.h"
#include "jobs/speedtask.h"
#include "jobs/stabilizetask.h"
#include "jobs/transcodetask.h"
#include "kddocksetup.h"
#include "layouts/layoutmanagement.h"
#include "library/librarywidget.h"
#include "render/renderrequest.h"
#include "render/renderserver.h"
#include <KEditToolBar>
#include "bin/model/markerlistmodel.hpp"
#include "dialogs/markerdialog.h"
#include "dialogs/textbasededit.h"
#include "dialogs/timeremap.h"
#include "filefilter.h"
#include "lib/localeHandling.h"
#include "mltconnection.h"
#include "mltcontroller/clipcontroller.h"
#include "monitor/monitor.h"
#include "monitor/monitormanager.h"
#include "monitor/monitorproxy.h"
#include "monitor/scopes/audiographspectrum.h"
#include "onlineresources/resourcewidget.hpp"
#include "profiles/profilemodel.hpp"
#include "profiles/profilerepository.hpp"
#include "project/cliptranscode.h"
#include "project/dialogs/archivewidget.h"
#include "project/dialogs/guideslist.h"
#include "project/dialogs/projectsettings.h"
#include "project/dialogs/temporarydata.h"
#include "project/projectmanager.h"
#include "scopes/scopemanager.h"
#include "timeline2/model/timelinefunctions.hpp"
#include "timeline2/model/timelineitemmodel.hpp"
#include "timeline2/view/timelinecontroller.h"
#include "timeline2/view/timelinetabs.hpp"
#include "timeline2/view/timelinewidget.h"
#include "titler/titlewidget.h"
#include "transitions/transitionlist/view/transitionlistwidget.hpp"
#include "transitions/transitionsrepository.hpp"
#include "effects/effectsrepository.hpp"
#include "widgets/progressbutton.h"
#include <config-wunjo.h>
#include "jogshuttle/jogmanager.h"
#include <kddockwidgets/core/FloatingWindow.h>
#include <KAboutData>
#include <KActionCollection>
#include <KActionMenu>
#include <KColorScheme>
#include <KConfigDialog>
#include <KCoreAddons>
#include <KDualAction>
#include <KIconEffect>
#include <KIconTheme>
#include <KLocalizedString>
#include <KMessageBox>
#include <KNSWidgets/Dialog>
#include <KNotifyConfigWidget>
#include <KRecentDirs>
#include <KShortcutsDialog>
#include <KStandardAction>
#include <KStyleManager>
#include <KToggleFullScreenAction>
#include <KToolBar>
#include <KXMLGUIFactory>
#include <kddockwidgets/kddockwidgets_version.h>
#include <kwidgetsaddons_version.h>
#include <kxmlgui_version.h>
#include <KConfigGroup>
#include <QAction>
#include <QClipboard>
#include <QCollator>
#include <QDesktopServices>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QMenu>
#include <QMenuBar>
#include <QRegularExpression>
#include <QPushButton>
#include <QScreen>
#include <QStandardPaths>
#include <QStatusBar>
#include <QStyleFactory>
#include <QUndoGroup>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

// ── Local helpers (ported from the D-Ogi scripting patch set) ──

static std::shared_ptr<KeyframeModelList> getEffectKeyframeModelAt(const std::shared_ptr<EffectStackModel> &stack, int effectIndex)
{
    if (!stack || effectIndex < 0 || effectIndex >= stack->rowCount()) return nullptr;
    auto effect = std::static_pointer_cast<EffectItemModel>(stack->getEffectStackRow(effectIndex));
    if (!effect) return nullptr;
    return effect->getKeyframeModel();
}

static QString keyframeTypeName(int type)
{
    switch (type) {
    case mlt_keyframe_linear:
        return QStringLiteral("linear");
    case mlt_keyframe_discrete:
        return QStringLiteral("discrete");
    case mlt_keyframe_smooth_natural:
        return QStringLiteral("smooth");
    case mlt_keyframe_bounce_in:
        return QStringLiteral("bounce_in");
    case mlt_keyframe_bounce_out:
        return QStringLiteral("bounce_out");
    case mlt_keyframe_cubic_in:
        return QStringLiteral("cubic_in");
    case mlt_keyframe_cubic_out:
        return QStringLiteral("cubic_out");
    case mlt_keyframe_exponential_in:
        return QStringLiteral("exponential_in");
    case mlt_keyframe_exponential_out:
        return QStringLiteral("exponential_out");
    case mlt_keyframe_circular_in:
        return QStringLiteral("circular_in");
    case mlt_keyframe_circular_out:
        return QStringLiteral("circular_out");
    case mlt_keyframe_elastic_in:
        return QStringLiteral("elastic_in");
    case mlt_keyframe_elastic_out:
        return QStringLiteral("elastic_out");
    default:
        return QStringLiteral("linear");
    }
}

static std::pair<std::shared_ptr<KeyframeModelList>, KeyframeModel *> getEffectKeyframeByParam(
    const std::shared_ptr<EffectStackModel> &stack, const QString &effectId, const QString &paramName)
{
    if (!stack) return {nullptr, nullptr};
    int row = stack->effectRow(effectId);
    if (row < 0) return {nullptr, nullptr};
    auto effect = std::static_pointer_cast<EffectItemModel>(stack->getEffectStackRow(row));
    if (!effect) return {nullptr, nullptr};
    auto listModel = effect->getKeyframeModel();
    if (!listModel) return {nullptr, nullptr};
    if (paramName.isEmpty()) {
        return {listModel, listModel->getKeyModel()};
    }
    QModelIndex paramIndex = effect->getParamIndexFromName(paramName);
    if (!paramIndex.isValid()) return {nullptr, nullptr};
    KeyframeModel *kfModel = listModel->getKeyModel(QPersistentModelIndex(paramIndex));
    if (!kfModel) return {nullptr, nullptr};
    return {listModel, kfModel};
}

static QVariantMap subtitleStyleToMap(const QString &name, const SubtitleStyle &s)
{
    QVariantMap m;
    m[QStringLiteral("name")] = name;
    m[QStringLiteral("fontName")] = s.fontName();
    m[QStringLiteral("fontSize")] = s.fontSize();
    m[QStringLiteral("primaryColour")] = s.primaryColour().name(QColor::HexArgb);
    m[QStringLiteral("secondaryColour")] = s.secondaryColour().name(QColor::HexArgb);
    m[QStringLiteral("outlineColour")] = s.outlineColour().name(QColor::HexArgb);
    m[QStringLiteral("backColour")] = s.backColour().name(QColor::HexArgb);
    m[QStringLiteral("bold")] = s.bold();
    m[QStringLiteral("italic")] = s.italic();
    m[QStringLiteral("underline")] = s.underline();
    m[QStringLiteral("strikeOut")] = s.strikeOut();
    m[QStringLiteral("scaleX")] = s.scaleX();
    m[QStringLiteral("scaleY")] = s.scaleY();
    m[QStringLiteral("spacing")] = s.spacing();
    m[QStringLiteral("angle")] = s.angle();
    m[QStringLiteral("borderStyle")] = s.borderStyle();
    m[QStringLiteral("outline")] = s.outline();
    m[QStringLiteral("shadow")] = s.shadow();
    m[QStringLiteral("alignment")] = s.alignment();
    m[QStringLiteral("marginL")] = s.marginL();
    m[QStringLiteral("marginR")] = s.marginR();
    m[QStringLiteral("marginV")] = s.marginV();
    return m;
}

static void applyStyleOverrides(SubtitleStyle &style, const QStringList &keys, const QStringList &values)
{
    int count = qMin(keys.size(), values.size());
    for (int i = 0; i < count; ++i) {
        const QString &k = keys.at(i);
        const QString &v = values.at(i);
        if (k == QLatin1String("fontName"))
            style.setFontName(v);
        else if (k == QLatin1String("fontSize"))
            style.setFontSize(v.toDouble());
        else if (k == QLatin1String("primaryColour"))
            style.setPrimaryColour(QColor(v));
        else if (k == QLatin1String("secondaryColour"))
            style.setSecondaryColour(QColor(v));
        else if (k == QLatin1String("outlineColour"))
            style.setOutlineColour(QColor(v));
        else if (k == QLatin1String("backColour"))
            style.setBackColour(QColor(v));
        else if (k == QLatin1String("bold"))
            style.setBold(v == QLatin1String("true") || v == QLatin1String("1"));
        else if (k == QLatin1String("italic"))
            style.setItalic(v == QLatin1String("true") || v == QLatin1String("1"));
        else if (k == QLatin1String("underline"))
            style.setUnderline(v == QLatin1String("true") || v == QLatin1String("1"));
        else if (k == QLatin1String("strikeOut"))
            style.setStrikeOut(v == QLatin1String("true") || v == QLatin1String("1"));
        else if (k == QLatin1String("scaleX"))
            style.setScaleX(v.toDouble());
        else if (k == QLatin1String("scaleY"))
            style.setScaleY(v.toDouble());
        else if (k == QLatin1String("spacing"))
            style.setSpacing(v.toDouble());
        else if (k == QLatin1String("angle"))
            style.setAngle(v.toDouble());
        else if (k == QLatin1String("borderStyle"))
            style.setBorderStyle(v.toInt());
        else if (k == QLatin1String("outline"))
            style.setOutline(v.toDouble());
        else if (k == QLatin1String("shadow"))
            style.setShadow(v.toDouble());
        else if (k == QLatin1String("alignment"))
            style.setAlignment(v.toInt());
        else if (k == QLatin1String("marginL"))
            style.setMarginL(v.toInt());
        else if (k == QLatin1String("marginR"))
            style.setMarginR(v.toInt());
        else if (k == QLatin1String("marginV"))
            style.setMarginV(v.toInt());
    }
}


bool MainWindow::scriptRenderWithParams(const QString &outputFile, const QString &presetName,
                                         int inFrame, int outFrame,
                                         const QStringList &paramKeys, const QStringList &paramValues)
{
    Q_UNUSED(outputFile) Q_UNUSED(presetName) Q_UNUSED(inFrame) Q_UNUSED(outFrame)
    Q_UNUSED(paramKeys) Q_UNUSED(paramValues)
    // TODO: implement when RenderPresetRepository API is public
    return false;
}

QStringList MainWindow::scriptGetRenderPresets()
{
    // TODO: implement when RenderPresetRepository API is public
    return {};
}

QVariantList MainWindow::scriptGetRenderJobs()
{
    // TODO: implement when render job list API is public
    return {};
}

bool MainWindow::scriptAbortRenderJob(const QString &outputPath)
{
    if (outputPath.isEmpty()) return false;
    Q_EMIT abortRenderJob(outputPath);
    return true;
}

QString MainWindow::scriptNewProject(const QString &name)
{
    Q_UNUSED(name)
    // Deferred out of the D-Bus dispatch: newFile() tears the document down and
    // calls qApp->processEvents(), and closing a modified project pops a modal
    // "Save changes?" dialog — spinning a nested event loop inside a D-Bus
    // method call reenters the dispatcher and crashes. So: hop to the next event
    // loop tick, close the current document WITHOUT prompting, then create fresh.
    QTimer::singleShot(0, this, [this]() {
        // And no questions while it happens: an auto-save left by a previous
        // run asks whether to recover it, and that dialog — opened with nobody
        // there, in the middle of a document being torn down — took the editor
        // down with it.
        auto *manager = pCore->projectManager();
        manager->setPrompting(false);
        if (!pCore->currentDoc() || manager->closeCurrentDocument(false, false)) {
            manager->newFile(false);
        }
        manager->setPrompting(true);
    });
    // Creation completes asynchronously; the caller polls get_project_info.
    return QStringLiteral("_untitled.wmproj");
}

bool MainWindow::scriptOpenProject(const QString &filePath)
{
    if (filePath.isEmpty()) return false;
    // Same reason as scriptNewProject: opening a project can find an auto-save
    // and ask about it, and there is nobody at this end to answer.
    auto *manager = pCore->projectManager();
    manager->setPrompting(false);
    manager->openFile(QUrl::fromLocalFile(filePath));
    // Put it back only once the open has finished with the event loops it spins
    // on the way. Restoring it on the next line meant a recovery question asked
    // from a queued continuation found prompting switched on again — a modal
    // dialog with nobody at this end to answer it.
    QTimer::singleShot(0, manager, [manager]() { manager->setPrompting(true); });
    return pCore->currentDoc() != nullptr;
}

bool MainWindow::scriptSaveProject()
{
    if (!pCore->currentDoc()) return false;
    pCore->projectManager()->saveFile();
    return true;
}

bool MainWindow::scriptSaveProjectAs(const QString &filePath)
{
    if (!pCore->currentDoc() || filePath.isEmpty()) return false;
    pCore->projectManager()->saveFileAs(filePath);
    return true;
}

bool MainWindow::scriptUndo(int steps)
{
    if (!m_commandStack || !m_commandStack->activeStack()) return false;
    int done = 0;
    for (int i = 0; i < steps; ++i) {
        if (!m_commandStack->activeStack()->canUndo()) break;
        m_commandStack->activeStack()->undo();
        ++done;
    }
    return done > 0;
}

bool MainWindow::scriptRedo(int steps)
{
    if (!m_commandStack || !m_commandStack->activeStack()) return false;
    int done = 0;
    for (int i = 0; i < steps; ++i) {
        if (!m_commandStack->activeStack()->canRedo()) break;
        m_commandStack->activeStack()->redo();
        ++done;
    }
    return done > 0;
}

QString MainWindow::scriptUndoStatus()
{
    if (!m_commandStack || !m_commandStack->activeStack()) {
        return QStringLiteral("can_undo=false;can_redo=false;undo_text=;redo_text=;index=0;count=0");
    }
    auto *stack = m_commandStack->activeStack();
    return QStringLiteral("can_undo=%1;can_redo=%2;undo_text=%3;redo_text=%4;index=%5;count=%6")
        .arg(stack->canUndo() ? QStringLiteral("true") : QStringLiteral("false"), stack->canRedo() ? QStringLiteral("true") : QStringLiteral("false"),
             stack->undoText(), stack->redoText(), QString::number(stack->index()), QString::number(stack->count()));
}

QString MainWindow::scriptGetProjectName()
{
    if (!pCore->currentDoc()) return QString();
    return pCore->currentDoc()->url().fileName();
}

QString MainWindow::scriptGetProjectPath()
{
    if (!pCore->currentDoc()) return QString();
    return pCore->currentDoc()->url().toLocalFile();
}

double MainWindow::scriptGetProjectFps()
{
    return pCore->getCurrentFps();
}

int MainWindow::scriptGetProjectResolutionWidth()
{
    if (!pCore->currentDoc()) return 0;
    return pCore->currentDoc()->width();
}

int MainWindow::scriptGetProjectResolutionHeight()
{
    if (!pCore->currentDoc()) return 0;
    return pCore->currentDoc()->height();
}

QString MainWindow::scriptGetProjectProperty(const QString &key)
{
    if (!pCore->currentDoc()) return QString();
    return pCore->currentDoc()->getDocumentProperty(key);
}

bool MainWindow::scriptSetProjectProperty(const QString &key, const QString &value)
{
    if (!pCore->currentDoc()) return false;
    pCore->currentDoc()->setDocumentProperty(key, value);
    return true;
}

bool MainWindow::scriptSetProjectProfile(int width, int height, int fpsNum, int fpsDen)
{
    if (!pCore->currentDoc()) return false;
    if (width <= 0 || height <= 0 || fpsNum <= 0 || fpsDen <= 0) return false;

    // Create a ProfileParam with the requested parameters
    std::unique_ptr<ProfileParam> newProfile(new ProfileParam(
        width, height, fpsNum, fpsDen,
        width, height,   // display aspect ratio (square pixels assumed)
        1, 1,            // sample aspect ratio
        709,             // colorspace (ITU-R 709)
        false            // progressive (not interlaced)
    ));
    newProfile->m_description = QStringLiteral("%1x%2 %3/%4fps").arg(width).arg(height).arg(fpsNum).arg(fpsDen);

    // Try to find matching existing profile
    QString matchingPath = ProfileRepository::get()->findMatchingProfile(newProfile.get());

    if (matchingPath.isEmpty()) {
        // Save as custom profile
        matchingPath = ProfileRepository::get()->saveProfile(newProfile.get());
    }

    if (matchingPath.isEmpty()) return false;

    pCore->setCurrentProfile(matchingPath);
    pCore->currentDoc()->resetProfile(true);
    return true;
}

int MainWindow::scriptCopyClips()
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->controller()) return -1;
    return timeline->controller()->copyItem();
}

bool MainWindow::scriptCutClips()
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->controller()) return false;
    timeline->controller()->cutItem();
    return true;
}

bool MainWindow::scriptPasteClips(int position, int trackId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->controller()) return false;
    return timeline->controller()->pasteItem(position, trackId);
}

QStringList MainWindow::scriptImportMedia(const QStringList &filePaths, const QString &folderId)
{
    QStringList binIds;
    if (!pCore->currentDoc()) return binIds;

    auto model = pCore->projectItemModel();
    QString folder = folderId;
    if (folder == QStringLiteral("-1")) {
        folder = model->getRootFolder()->clipId();
    }

    for (const QString &path : filePaths) {
        // Check if clip already exists in bin
        QStringList existing = model->getClipByUrl(QFileInfo(path));
        if (!existing.isEmpty()) {
            binIds.append(existing.first());
            continue;
        }
        // Import new clip
        ClipCreator::createClipFromFile(path, folder, model);
        // Retrieve the newly created bin ID
        QStringList newIds = model->getClipByUrl(QFileInfo(path));
        if (!newIds.isEmpty()) {
            binIds.append(newIds.first());
        }
    }
    return binIds;
}

QString MainWindow::scriptCreateFolder(const QString &name, const QString &parentId)
{
    if (!pCore->currentDoc()) return QString();

    auto model = pCore->projectItemModel();
    QString parent = parentId;
    if (parent == QStringLiteral("-1")) {
        parent = model->getRootFolder()->clipId();
    }

    QString newId;
    Fun undo = []() { return true; };
    Fun redo = []() { return true; };
    if (model->requestAddFolder(newId, name, parent, undo, redo)) {
        pCore->pushUndo(undo, redo, i18n("Create folder"));
        return newId;
    }
    return QString();
}

QStringList MainWindow::scriptGetAllClipIds()
{
    QStringList result;
    if (!pCore->currentDoc()) return result;

    auto model = pCore->projectItemModel();
    std::vector<QString> ids = model->getAllClipIds();
    for (const QString &id : ids) {
        result.append(id);
    }
    return result;
}

QStringList MainWindow::scriptGetFolderClipIds(const QString &folderId)
{
    QStringList result;
    if (!pCore->currentDoc()) return result;

    auto model = pCore->projectItemModel();
    // Get folder and iterate its children
    auto folder = model->getFolderByBinId(folderId);
    if (!folder) return result;

    int count = folder->childCount();
    for (int i = 0; i < count; i++) {
        auto child = std::static_pointer_cast<AbstractProjectItem>(folder->child(i));
        if (child && child->itemType() == AbstractProjectItem::ClipItem) {
            result.append(child->clipId());
        }
    }
    return result;
}

int MainWindow::scriptGetProjectDuration()
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return 0;
    return timeline->model()->duration();
}

QString MainWindow::scriptGetProjectColorSpace()
{
    if (!pCore->currentDoc()) return QString();
    // Color space is stored as a project property
    return pCore->currentDoc()->getDocumentProperty(QStringLiteral("color_space"), QStringLiteral("709"));
}

bool MainWindow::scriptSetProjectColorSpace(const QString &colorSpace)
{
    if (!pCore->currentDoc()) return false;
    pCore->currentDoc()->setDocumentProperty(QStringLiteral("color_space"), colorSpace);
    return true;
}

int MainWindow::scriptGetProjectAudioSampleRate()
{
    if (!pCore->currentDoc()) return 0;
    // MLT profiles do not store audio sample rate (sample_aspect_num() is SAR, not audio).
    // Kdenlive uses 48000 Hz as the default audio sample rate for capture and processing.
    // Return the capture sample rate from settings, which defaults to 48000.
    return WunjoSettings::audiocapturesamplerate();
}

QVariantMap MainWindow::scriptGetClipProperties(const QString &binId)
{
    QVariantMap props;
    if (!pCore->currentDoc()) return props;

    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) return props;

    // Use cached AbstractProjectItem members instead of ClipController methods
    // that acquire m_producerLock. The producer lock can deadlock when called
    // from the main thread (D-Bus handler) while a background thread holds the
    // write lock and waits for the main event loop.
    props[QStringLiteral("name")] = clip->getData(AbstractProjectItem::DataName);
    props[QStringLiteral("type")] = clip->getData(AbstractProjectItem::ClipType);
    props[QStringLiteral("url")] = clip->clipUrl();
    props[QStringLiteral("id")] = binId;

    // Duration: convert cached timecode string to frames.
    // Format is SMPTE: "HH:MM:SS:FF" or "HH:MM:SS;FF" (drop-frame).
    QString durationStr = clip->getData(AbstractProjectItem::DataDuration).toString();
    int fps = qRound(pCore->getCurrentFps());
    if (!durationStr.isEmpty() && fps > 0) {
        // Replace semicolons (drop-frame separator) with colons for uniform parsing
        QString normalized = durationStr.replace(QLatin1Char(';'), QLatin1Char(':'));
        QStringList parts = normalized.split(QLatin1Char(':'));
        if (parts.size() == 4) {
            props[QStringLiteral("duration")] = parts[0].toInt() * 3600 * fps + parts[1].toInt() * 60 * fps + parts[2].toInt() * fps + parts[3].toInt();
        } else {
            props[QStringLiteral("duration")] = 0;
        }
    } else {
        props[QStringLiteral("duration")] = 0;
    }

    return props;
}

bool MainWindow::scriptDeleteBinClip(const QString &binId)
{
    if (!pCore->currentDoc()) return false;

    auto model = pCore->projectItemModel();
    auto clip = model->getClipByBinID(binId);
    if (!clip) return false;

    Fun undo = []() { return true; };
    Fun redo = []() { return true; };
    bool result = model->requestBinClipDeletion(clip, undo, redo);
    if (result) {
        pCore->pushUndo(undo, redo, i18n("Delete clip"));
    }
    return result;
}

bool MainWindow::scriptRelinkBinClip(const QString &binId, const QString &newFilePath)
{
    if (!QFile::exists(newFilePath)) {
        qWarning() << "scriptRelinkBinClip: file does not exist:" << newFilePath;
        return false;
    }
    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) {
        qWarning() << "scriptRelinkBinClip: clip not found:" << binId;
        return false;
    }
    clip->setProducerProperty(QStringLiteral("resource"), newFilePath);
    clip->setProducerProperty(QStringLiteral("kdenlive:originalurl"), newFilePath);
    clip->reloadProducer(false, false, false);
    return true;
}

QString MainWindow::scriptCreateTitleClip(const QString &titleXml, int durationFrames, const QString &clipName, const QString &parentFolderId)
{
    if (!pCore->currentDoc()) return QStringLiteral("-1");
    auto model = pCore->projectItemModel();
    QString folder = parentFolderId;
    if (folder == QStringLiteral("-1")) folder = model->getRootFolder()->clipId();

    std::unordered_map<QString, QString> properties;
    properties[QStringLiteral("xmldata")] = titleXml;

    return ClipCreator::createTitleClip(properties, durationFrames, clipName.isEmpty() ? i18n("Title clip") : clipName, folder, model);
}

QString MainWindow::scriptGetTitleXml(const QString &binId)
{
    if (!pCore->currentDoc()) return QString();
    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) return QString();
    return clip->getProducerProperty(QStringLiteral("xmldata"));
}

bool MainWindow::scriptSetTitleXml(const QString &binId, const QString &newXml)
{
    if (!pCore->currentDoc()) return false;
    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) return false;
    clip->setProducerProperty(QStringLiteral("xmldata"), newXml);
    clip->reloadProducer(false, false, false);
    return true;
}

bool MainWindow::scriptRenameBinClip(const QString &binId, const QString &newName)
{
    if (!pCore->currentDoc()) return false;
    if (newName.isEmpty()) return false;
    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) return false;
    return clip->rename(newName, 0);
}

bool MainWindow::scriptMoveBinClip(const QString &binId, const QString &targetFolderId)
{
    if (!pCore->currentDoc()) return false;
    auto item = pCore->projectItemModel()->getItemByBinId(binId);
    if (!item) return false;
    auto currentParent = item->parent();
    if (!currentParent) return false;
    QString oldParentId = std::static_pointer_cast<AbstractProjectItem>(currentParent)->clipId();
    if (oldParentId == targetFolderId) return true; // already in target folder
    // Verify target folder exists
    auto targetFolder = pCore->projectItemModel()->getFolderByBinId(targetFolderId);
    if (!targetFolder) return false;
    QMap<QString, std::pair<QString, QString>> idsMap;
    idsMap.insert(binId, {targetFolderId, oldParentId});
    auto *moveCommand = new QUndoCommand();
    new MoveBinClipCommand(pCore->bin(), idsMap, moveCommand);
    pCore->currentDoc()->commandStack()->push(moveCommand);
    return true;
}

QVariantMap MainWindow::scriptGetClipMetadata(const QString &binId)
{
    QVariantMap meta;
    if (!pCore->currentDoc()) return meta;

    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) return meta;

    meta[QStringLiteral("id")] = binId;
    meta[QStringLiteral("name")] = clip->getData(AbstractProjectItem::DataName);
    meta[QStringLiteral("type")] = clip->getData(AbstractProjectItem::ClipType);
    meta[QStringLiteral("url")] = clip->clipUrl();

    // File info
    QFileInfo fileInfo(clip->clipUrl());
    if (fileInfo.exists()) {
        meta[QStringLiteral("fileSize")] = fileInfo.size();
        meta[QStringLiteral("fileName")] = fileInfo.fileName();
    }

    // Try to get codec/resolution info from producer properties (safe, no lock)
    // TODO: verify getProducerProperty is safe in all contexts (reads from cached Mlt::Properties)
    meta[QStringLiteral("videoCodec")] = clip->getProducerProperty(QStringLiteral("meta.media.0.codec.name"));
    meta[QStringLiteral("audioCodec")] = clip->getProducerProperty(QStringLiteral("meta.media.1.codec.name"));
    meta[QStringLiteral("width")] = clip->getProducerProperty(QStringLiteral("meta.media.width"));
    meta[QStringLiteral("height")] = clip->getProducerProperty(QStringLiteral("meta.media.height"));
    meta[QStringLiteral("frameRate")] = clip->getProducerProperty(QStringLiteral("meta.media.frame_rate_num"));
    meta[QStringLiteral("sampleRate")] = clip->getProducerProperty(QStringLiteral("meta.media.1.codec.sample_rate"));
    meta[QStringLiteral("channels")] = clip->getProducerProperty(QStringLiteral("meta.media.1.codec.channels"));

    // Duration from cached data (safe approach from scriptGetClipProperties)
    QString durationStr = clip->getData(AbstractProjectItem::DataDuration).toString();
    int fps = qRound(pCore->getCurrentFps());
    if (!durationStr.isEmpty() && fps > 0) {
        QString normalized = durationStr.replace(QLatin1Char(';'), QLatin1Char(':'));
        QStringList parts = normalized.split(QLatin1Char(':'));
        if (parts.size() == 4) {
            meta[QStringLiteral("durationFrames")] = parts[0].toInt() * 3600 * fps + parts[1].toInt() * 60 * fps + parts[2].toInt() * fps + parts[3].toInt();
        }
    }
    meta[QStringLiteral("durationTimecode")] = durationStr;

    return meta;
}

int MainWindow::scriptGetTrackCount(const QString &trackType)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return 0;

    QPair<int, int> counts = timeline->model()->getAVtracksCount();
    if (trackType == QStringLiteral("audio")) {
        return counts.second;
    }
    return counts.first; // video
}

QVariantMap MainWindow::scriptGetTrackInfo(int trackIndex)
{
    QVariantMap info;
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return info;

    auto model = timeline->model();
    QList<int> trackIds = model->getTracksIds(false); // video
    QList<int> audioIds = model->getTracksIds(true);  // audio
    QList<int> allIds;
    allIds.append(trackIds);
    allIds.append(audioIds);

    if (trackIndex < 0 || trackIndex >= allIds.size()) return info;

    int trackId = allIds.at(trackIndex);
    info[QStringLiteral("id")] = trackId;
    info[QStringLiteral("name")] = model->getTrackTagById(trackId);
    info[QStringLiteral("audio")] = model->isAudioTrack(trackId);
    info[QStringLiteral("position")] = model->getTrackPosition(trackId);
    return info;
}

QVariantList MainWindow::scriptGetAllTracksInfo()
{
    QVariantList result;
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return result;

    auto model = timeline->model();
    QList<int> videoIds = model->getTracksIds(false);
    QList<int> audioIds = model->getTracksIds(true);

    for (int trackId : videoIds) {
        QVariantMap info;
        info[QStringLiteral("id")] = trackId;
        info[QStringLiteral("name")] = model->getTrackTagById(trackId);
        info[QStringLiteral("audio")] = false;
        info[QStringLiteral("position")] = model->getTrackPosition(trackId);
        int hide = model->getTrackProperty(trackId, QStringLiteral("hide")).toInt();
        info[QStringLiteral("mute")] = bool(hide & 2);
        info[QStringLiteral("hidden")] = bool(hide & 1);
        result.append(info);
    }
    for (int trackId : audioIds) {
        QVariantMap info;
        info[QStringLiteral("id")] = trackId;
        info[QStringLiteral("name")] = model->getTrackTagById(trackId);
        info[QStringLiteral("audio")] = true;
        info[QStringLiteral("position")] = model->getTrackPosition(trackId);
        int hide = model->getTrackProperty(trackId, QStringLiteral("hide")).toInt();
        info[QStringLiteral("mute")] = bool(hide & 2);
        info[QStringLiteral("hidden")] = bool(hide & 1);
        result.append(info);
    }
    return result;
}

int MainWindow::scriptAddTrack(const QString &name, bool audioTrack, int position)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return -1;
    auto model = timeline->model();

    int pos = position;
    if (pos < 0) {
        // Video tracks sit above audio ones and the two never interleave — that
        // is the one rule the timeline's layout has. So a new track's place
        // follows from its kind, and the caller does not have to know the
        // ordering: video goes above everything, audio above the other audio.
        //
        // Adding every track at the very top, which is what this did, put a new
        // audio track above the video and broke that rule.
        pos = audioTrack ? int(model->getTracksIds(true).count()) : model->getTracksCount();
    }
    int newId = -1;
    bool success = model->requestTrackInsertion(pos, newId, name, audioTrack);
    return success ? newId : -1;
}

bool MainWindow::scriptDeleteTrack(int trackId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isTrack(trackId)) return false;
    return timeline->model()->requestTrackDeletion(trackId);
}

bool MainWindow::scriptInsertSpace(int trackId, int position, int duration, bool allTracks)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (duration <= 0) return false;

    int targetTrack = allTracks ? -1 : trackId;
    int cid = timeline->controller()->requestSpacerStartOperation(targetTrack, position);
    if (cid == -1) return false;

    int start = timeline->model()->getItemPosition(cid);
    return timeline->controller()->requestSpacerEndOperation(cid, start, start + duration, targetTrack, {}, -1);
}

bool MainWindow::scriptRemoveSpace(int trackId, int position, bool allTracks)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    return TimelineFunctions::requestDeleteBlankAt(timeline->model(), trackId, position, allTracks);
}

int MainWindow::scriptInsertClip(const QString &binClipId, int trackId, int position)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return -1;

    // Refuse what the timeline would only choke on. A person cannot drag a clip
    // that is not in the bin onto a track that does not exist; an assistant can
    // ask for exactly that, and the model below assumes the ids are real — the
    // lookup throws rather than returns, and an uncaught throw takes the whole
    // editor down with the user's project in it.
    if (!pCore->projectItemModel()->getClipByBinID(binClipId)) {
        return -1;
    }
    if (!timeline->model()->isTrack(trackId) || position < 0) {
        return -1;
    }

    int newClipId = -1;
    bool success = timeline->model()->requestClipInsertion(binClipId, trackId, position, newClipId,
                                                           true, // logUndo
                                                           true, // refreshView
                                                           false // useTargets
    );
    return success ? newClipId : -1;
}

QVariantList MainWindow::scriptInsertClipsSequentially(const QStringList &binClipIds, int trackId, int startPosition)
{
    QVariantList result;
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return result;

    int position = startPosition;
    for (const QString &binId : binClipIds) {
        int newClipId = -1;
        bool success = timeline->model()->requestClipInsertion(binId, trackId, position, newClipId, true, true, false);
        if (success && newClipId >= 0) {
            result.append(newClipId);
            // Advance position by the clip's duration
            int duration = timeline->model()->getClipPlaytime(newClipId);
            position += duration;
        } else {
            result.append(-1);
        }
    }
    return result;
}

bool MainWindow::scriptMoveClip(int clipId, int trackId, int position)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;

    return timeline->model()->requestClipMove(clipId, trackId, position,
                                              true,  // moveMirrorTracks
                                              true,  // updateView
                                              true); // logUndo
}

int MainWindow::scriptResizeClip(int clipId, int newDuration, bool fromRight)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return -1;

    return timeline->model()->requestItemResize(clipId, newDuration, fromRight, true);
}

bool MainWindow::scriptDeleteTimelineClip(int clipId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;

    return timeline->model()->requestItemDeletion(clipId, true);
}

QVariantList MainWindow::scriptGetClipsOnTrack(int trackId)
{
    QVariantList result;
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return result;

    auto model = timeline->model();
    // A track id that names no track took the application down with it: the
    // range lookup below assumes the track exists. Anything driving this from
    // outside — a script, a plugin — can only guess ids, so it must be able to
    // guess wrong without killing the editor.
    if (!model->isTrack(trackId)) return result;
    // Get all clips in range [0, max) on this track
    std::unordered_set<int> clipIds = model->getItemsInRange(trackId, 0, -1, false);
    for (int cid : clipIds) {
        if (!model->isClip(cid)) continue;
        QVariantMap info;
        info[QStringLiteral("id")] = cid;
        info[QStringLiteral("position")] = model->getClipPosition(cid);
        info[QStringLiteral("duration")] = model->getClipPlaytime(cid);
        info[QStringLiteral("trackId")] = model->getClipTrackId(cid);
        info[QStringLiteral("in")] = model->getClipIn(cid);
        QString binId = model->getClipBinId(cid);
        info[QStringLiteral("binId")] = binId;
        // Resolve human-readable name and source URL from bin
        if (!binId.isEmpty() && pCore->bin()) {
            info[QStringLiteral("name")] = pCore->bin()->getBinClipName(binId);
        }
        auto clip = pCore->projectItemModel()->getClipByBinID(binId);
        if (clip) {
            info[QStringLiteral("url")] = clip->clipUrl();
        }
        result.append(info);
    }
    return result;
}

QVariantMap MainWindow::scriptGetTimelineClipInfo(int clipId)
{
    QVariantMap info;
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return info;

    auto model = timeline->model();
    if (!model->isClip(clipId)) return info;

    info[QStringLiteral("id")] = clipId;
    info[QStringLiteral("position")] = model->getClipPosition(clipId);
    info[QStringLiteral("duration")] = model->getClipPlaytime(clipId);
    info[QStringLiteral("trackId")] = model->getClipTrackId(clipId);
    auto inOut = model->getClipInOut(clipId);
    info[QStringLiteral("in")] = inOut.first;
    info[QStringLiteral("out")] = inOut.second;
    QString binId = model->getClipBinId(clipId);
    info[QStringLiteral("binId")] = binId;

    // Resolve bin clip name and source URL
    if (!binId.isEmpty() && pCore->bin()) {
        info[QStringLiteral("name")] = pCore->bin()->getBinClipName(binId);
    }
    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (clip) {
        info[QStringLiteral("url")] = clip->clipUrl();
        info[QStringLiteral("maxDuration")] = (int)clip->frameDuration();
    }
    return info;
}

bool MainWindow::scriptSlipClip(int clipId, int offset)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->controller()) return false;
    auto model = timeline->model();
    if (!model || !model->isClip(clipId)) return false;

    // Save and replace selection
    QList<int> prevSel = timeline->controller()->selection();
    timeline->controller()->selectItems({clipId});

    // Perform slip
    model->requestSlipSelection(offset, true);

    // Restore selection
    if (prevSel.isEmpty()) {
        timeline->controller()->selectItems(QList<int>());
    } else {
        timeline->controller()->selectItems(prevSel);
    }

    return true;
}

bool MainWindow::scriptCutClip(int clipId, int position)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;

    return TimelineFunctions::requestClipCut(timeline->model(), clipId, position);
}

bool MainWindow::scriptRippleDelete(int clipId)
{
    // Delete clip and close the gap (ripple delete)
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    int position = timeline->model()->getClipPosition(clipId);
    int duration = timeline->model()->getClipPlaytime(clipId);

    // Delete the clip first
    Fun undo = []() { return true; };
    Fun redo = []() { return true; };
    bool deleted = timeline->model()->requestItemDeletion(clipId, true);
    if (!deleted) return false;

    // Remove the gap left behind using removeSpace
    QPoint zone(position, position + duration);
    bool spaceRemoved = TimelineFunctions::removeSpace(timeline->model(), zone, undo, redo);
    if (spaceRemoved) {
        pCore->pushUndo(undo, redo, i18n("Ripple delete"));
    }
    return deleted;
}

bool MainWindow::scriptRippleTrim(int clipId, int delta, bool fromRight)
{
    // Trim clip and shift all following clips by delta
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    int position = timeline->model()->getClipPosition(clipId);
    int duration = timeline->model()->getClipPlaytime(clipId);
    int newDuration = duration + delta;
    if (newDuration < 1) return false;

    Fun undo = []() { return true; };
    Fun redo = []() { return true; };

    // Resize the clip
    bool ok = timeline->model()->requestItemResize(clipId, newDuration, fromRight, true);
    if (!ok) return false;

    // Shift subsequent clips by delta to maintain ripple behavior
    if (delta < 0) {
        // Clip got shorter — remove the gap
        int gapStart = position + newDuration;
        if (!fromRight) {
            // Trimmed from left: gap is before clip at new position
            int newPosition = timeline->model()->getClipPosition(clipId);
            gapStart = newPosition + newDuration;
        }
        QPoint zone(gapStart, gapStart + (-delta));
        TimelineFunctions::removeSpace(timeline->model(), zone, undo, redo);
    } else if (delta > 0) {
        // Clip got longer — insert space to push subsequent clips
        int spaceStart = position + duration;  // old end position
        if (!fromRight) {
            spaceStart = position;
        }
        QPoint zone(spaceStart, spaceStart + delta);
        TimelineFunctions::requestInsertSpace(timeline->model(), zone, undo, redo);
    }

    pCore->pushUndo(undo, redo, i18n("Ripple trim"));
    return true;
}

bool MainWindow::scriptRollEdit(int clipId, int delta)
{
    // Move edit point between two adjacent clips (trim both)
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto model = timeline->model();
    int trackId = model->getClipTrackId(clipId);
    int position = model->getClipPosition(clipId);
    int duration = model->getClipPlaytime(clipId);
    int clipEnd = position + duration;

    // Find the adjacent clip to the right
    int nextClipId = model->getClipByPosition(trackId, clipEnd);
    if (nextClipId < 0 || !model->isClip(nextClipId)) return false;

    int nextDuration = model->getClipPlaytime(nextClipId);

    // Validate: new durations must be positive
    int newDuration = duration + delta;
    int newNextDuration = nextDuration - delta;
    if (newDuration < 1 || newNextDuration < 1) return false;

    // Resize current clip from right edge
    int result1 = model->requestItemResize(clipId, newDuration, true, false);
    if (result1 < 0) return false;

    // Resize next clip from left edge
    int result2 = model->requestItemResize(nextClipId, newNextDuration, false, true);
    if (result2 < 0) {
        // Revert first resize
        model->requestItemResize(clipId, duration, true, false);
        return false;
    }
    return true;
}

bool MainWindow::scriptSlideEdit(int clipId, int delta)
{
    // Move clip without changing its duration (adjust neighbor trim points)
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto model = timeline->model();
    int trackId = model->getClipTrackId(clipId);
    int position = model->getClipPosition(clipId);
    int duration = model->getClipPlaytime(clipId);

    // Find adjacent clips
    int prevClipId = (position > 0) ? model->getClipByPosition(trackId, position - 1) : -1;
    int nextClipId = model->getClipByPosition(trackId, position + duration);

    // Validate we have at least one neighbor to adjust
    if (prevClipId < 0 && nextClipId < 0) return false;

    // Check that neighbors can absorb the delta
    if (prevClipId >= 0 && model->isClip(prevClipId)) {
        int prevDur = model->getClipPlaytime(prevClipId);
        if (prevDur + delta < 1) return false;
    }
    if (nextClipId >= 0 && model->isClip(nextClipId)) {
        int nextDur = model->getClipPlaytime(nextClipId);
        if (nextDur - delta < 1) return false;
    }

    // Adjust previous clip (extend/shrink right edge)
    if (prevClipId >= 0 && model->isClip(prevClipId)) {
        int prevDur = model->getClipPlaytime(prevClipId);
        model->requestItemResize(prevClipId, prevDur + delta, true, false);
    }

    // Move the clip itself
    model->requestClipMove(clipId, trackId, position + delta, true, true, false);

    // Adjust next clip (shrink/extend left edge)
    if (nextClipId >= 0 && model->isClip(nextClipId)) {
        int nextDur = model->getClipPlaytime(nextClipId);
        model->requestItemResize(nextClipId, nextDur - delta, false, true);
    }

    return true;
}

bool MainWindow::scriptAddMix(int clipIdA, int clipIdB, int durationFrames)
{
    Q_UNUSED(clipIdA)
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;

    // Use the mix request — Kdenlive creates same-track mixes via the second clip
    int trackId = timeline->model()->getClipTrackId(clipIdB);
    int posB = timeline->model()->getClipPosition(clipIdB);
    std::pair<int, int> clipIds = {clipIdA, clipIdB};
    std::pair<int, int> mixDurations = {durationFrames / 2, durationFrames - durationFrames / 2};

    Fun undo = []() { return true; };
    Fun redo = []() { return true; };
    bool success = timeline->model()->requestClipMix(QStringLiteral("luma"), clipIds, mixDurations, trackId, posB, true, true, true, undo, redo, false);
    if (success) {
        pCore->pushUndo(undo, redo, i18n("Add mix"));
    }
    return success;
}

int MainWindow::scriptAddComposition(const QString &transitionId, int trackId, int position, int duration)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return -1;

    int newId = -1;
    bool success = timeline->model()->requestCompositionInsertion(transitionId, trackId, position, duration, nullptr, newId, true);
    return success ? newId : -1;
}

bool MainWindow::scriptRemoveMix(int clipId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;

    return timeline->model()->requestItemDeletion(clipId, true);
}

QVariantList MainWindow::scriptGetAvailableTransitions()
{
    // Same as scriptGetCompositionTypes — transitions and compositions share the same repository
    QVariantList result;
    auto allTransitions = TransitionsRepository::get()->getNames();
    for (const auto &pair : allTransitions) {
        QVariantMap info;
        info[QStringLiteral("id")] = pair.first;
        info[QStringLiteral("name")] = pair.second;
        result.append(info);
    }
    return result;
}

QVariantMap MainWindow::scriptGetMixParams(int clipId)
{
    QVariantMap result;
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return result;
    if (!timeline->model()->isClip(clipId)) return result;

    auto model = timeline->model();
    int trackId = model->getClipTrackId(clipId);
    int mixDuration = model->getMixDuration(clipId);

    result[QStringLiteral("clipId")] = clipId;
    result[QStringLiteral("trackId")] = trackId;
    result[QStringLiteral("duration")] = mixDuration;

    if (mixDuration > 0) {
        std::pair<int, int> mixInOut = model->getMixInOut(clipId);
        int mixCut = model->getMixCutPos(clipId);
        result[QStringLiteral("mixIn")] = mixInOut.first;
        result[QStringLiteral("mixOut")] = mixInOut.second;
        result[QStringLiteral("mixCut")] = mixCut;
    }

    return result;
}

bool MainWindow::scriptSetMixDuration(int clipId, int newDuration)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;
    if (newDuration < 1) return false;

    auto model = timeline->model();
    int currentDuration = model->getMixDuration(clipId);
    if (currentDuration <= 0) return false;  // No mix on this clip

    // Use requestResizeMix with center alignment
    model->requestResizeMix(clipId, newDuration, MixAlignment::AlignCenter);
    // Verify it took effect
    int resultDuration = model->getMixDuration(clipId);
    return resultDuration > 0;
}

QVariantList MainWindow::scriptGetCompositions()
{
    QVariantList result;
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return result;

    auto model = timeline->model();
    const QList<int> ids = model->getCompositionIds();
    for (int compoId : ids) {
        QVariantMap info;
        info[QStringLiteral("id")] = compoId;
        info[QStringLiteral("trackId")] = model->getCompositionTrackId(compoId);
        info[QStringLiteral("position")] = model->getCompositionPosition(compoId);
        info[QStringLiteral("duration")] = model->getCompositionPlaytime(compoId);
        auto paramModel = model->getCompositionParameterModel(compoId);
        if (paramModel) {
            info[QStringLiteral("type")] = paramModel->getAssetId();
        }
        result.append(info);
    }
    return result;
}

QVariantMap MainWindow::scriptGetCompositionInfo(int compoId)
{
    QVariantMap result;
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return result;
    if (!timeline->model()->isComposition(compoId)) return result;

    auto model = timeline->model();
    result[QStringLiteral("id")] = compoId;
    result[QStringLiteral("trackId")] = model->getCompositionTrackId(compoId);
    result[QStringLiteral("position")] = model->getCompositionPosition(compoId);
    result[QStringLiteral("duration")] = model->getCompositionPlaytime(compoId);
    auto paramModel = model->getCompositionParameterModel(compoId);
    if (paramModel) {
        result[QStringLiteral("type")] = paramModel->getAssetId();
    }
    return result;
}

bool MainWindow::scriptMoveComposition(int compoId, int trackId, int position)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isComposition(compoId)) return false;
    return timeline->model()->requestCompositionMove(compoId, trackId, position, true, true);
}

int MainWindow::scriptResizeComposition(int compoId, int newDuration, bool fromRight)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return -1;
    if (!timeline->model()->isComposition(compoId)) return -1;
    return timeline->model()->requestItemResize(compoId, newDuration, fromRight, true);
}

bool MainWindow::scriptDeleteComposition(int compoId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isComposition(compoId)) return false;
    return timeline->model()->requestItemDeletion(compoId, true);
}

QVariantList MainWindow::scriptGetCompositionTypes()
{
    QVariantList result;
    auto allTransitions = TransitionsRepository::get()->getNames();
    for (const auto &pair : allTransitions) {
        QVariantMap info;
        info[QStringLiteral("id")] = pair.first;
        info[QStringLiteral("name")] = pair.second;
        result.append(info);
    }
    return result;
}

bool MainWindow::scriptSetCompositionParam(int compoId, const QString &paramName, const QString &paramValue)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isComposition(compoId)) return false;

    auto paramModel = timeline->model()->getCompositionParameterModel(compoId);
    if (!paramModel) return false;
    paramModel->setParameter(paramName, paramValue, true);
    return true;
}

QString MainWindow::scriptGetCompositionParam(int compoId, const QString &paramName)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return QString();
    if (!timeline->model()->isComposition(compoId)) return QString();

    auto paramModel = timeline->model()->getCompositionParameterModel(compoId);
    if (!paramModel) return QString();
    return paramModel->getParam(paramName);
}

QVariantList MainWindow::scriptGetAvailableEffects()
{
    QVariantList result;
    auto allEffects = EffectsRepository::get()->getNames();
    for (const auto &pair : allEffects) {
        QVariantMap info;
        info[QStringLiteral("id")] = pair.first;
        info[QStringLiteral("name")] = pair.second;
        info[QStringLiteral("type")] = EffectsRepository::get()->isAudioEffect(pair.first) ? QStringLiteral("audio") : QStringLiteral("video");
        result.append(info);
    }
    return result;
}

namespace {

/** @brief Put a plugin's effect on a clip the way the editor's own menus do.
 *
 *  An effect that belongs to a plugin is rarely alone: it may declare a region
 *  effect it works inside, and the two have to arrive together, sharing a bond,
 *  or the plugin renders nothing because it cannot tell where to look. Adding
 *  it the plain way — which is right for an MLT filter — leaves exactly that
 *  half-applied state, and the failure is silent. So both scripting entry
 *  points come through here, and every plugin behaves the same: the one that
 *  works on a face, and the one that works on the whole clip.
 *
 *  @p faceTrack is the animated rectangle to fill face parameters with, empty
 *  when there is no face in play. @p handled says whether the effect belonged
 *  to a plugin at all; when it did not, the caller adds it as before.
 */
bool applyPluginEffect(const std::shared_ptr<EffectStackModel> &stack, const QString &effectId, const QString &faceTrack, bool &handled)
{
    const QString pluginId = PluginManager::instance().pluginForEffect(effectId);
    handled = !pluginId.isEmpty();
    if (!handled) {
        return false;
    }
    const PluginManifest plugin = PluginManager::instance().plugin(pluginId);
    if (plugin.appliesEffectsTogether()) {
        // Its effects are three ways of working on one region: the plugin says
        // they belong together, so asking for one asks for the set.
        return PluginEffects::applyAll(stack, plugin, faceTrack);
    }
    const QList<PluginEffect> effects = plugin.effects();
    for (const PluginEffect &effect : effects) {
        if (effect.id == effectId) {
            return PluginEffects::apply(stack, plugin, effect, faceTrack);
        }
    }
    return false;
}

/** @brief Write @p params onto @p effectId once it is on the clip. */
void setEffectParams(const std::shared_ptr<EffectStackModel> &stack, const QString &effectId, const QMap<QString, QString> &params)
{
    if (params.isEmpty()) {
        return;
    }
    auto asset = stack->getAssetModelById(effectId);
    if (!asset) {
        return;
    }
    for (auto it = params.constBegin(); it != params.constEnd(); ++it) {
        asset->setParameter(it.key(), it.value(), true);
    }
}

} // namespace

bool MainWindow::scriptAddClipEffect(int clipId, const QString &wantedId, const QStringList &paramKeys, const QStringList &paramValues)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    // Asking for a plugin's region effect means asking for the effect that
    // works inside it: on its own the region renders nothing, and a clip
    // carrying only that looked treated while being unable to produce anything.
    const QString effectId = PluginEffects::effectToApply(wantedId);

    QMap<QString, QString> params;
    int count = qMin(paramKeys.size(), paramValues.size());
    for (int i = 0; i < count; ++i) {
        params.insert(paramKeys.at(i), paramValues.at(i));
    }

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack) return false;
    // A plugin's effect brings whatever it requires with it; an ordinary filter
    // is added as it always was.
    bool handled = false;
    const bool applied = applyPluginEffect(stack, effectId, QString(), handled);
    if (handled) {
        if (applied) {
            setEffectParams(stack, effectId, params);
        }
        return applied;
    }
    return stack->appendEffect(effectId, false, params);
}

bool MainWindow::scriptRemoveClipEffect(int clipId, const QString &effectId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack || !stack->hasFilter(effectId)) return false;
    Fun undo = []() { return true; };
    Fun redo = []() { return true; };
    QString effectName;
    stack->removeEffectWithUndo(effectId, effectName, -1, undo, redo);
    return !effectName.isEmpty();
}

QString MainWindow::scriptGetClipEffects(int clipId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return QString();
    if (!timeline->model()->isClip(clipId)) return QString();

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack) return QString();
    return stack->effectNames();
}

bool MainWindow::scriptSetEffectParam(int clipId, const QString &effectId, const QString &paramName, const QString &paramValue)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack) return false;
    auto asset = stack->getAssetModelById(effectId);
    if (!asset) return false;
    asset->setParameter(paramName, paramValue, true);
    return true;
}

QString MainWindow::scriptGetEffectParam(int clipId, const QString &effectId, const QString &paramName)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return QString();
    if (!timeline->model()->isClip(clipId)) return QString();

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack) return QString();
    auto asset = stack->getAssetModelById(effectId);
    if (!asset) return QString();
    return asset->getParam(paramName);
}

bool MainWindow::scriptSetEffectExpression(int clipId, const QString &effectId, const QString &paramName, const QString &expression, double baseValue)
{
    Q_UNUSED(clipId)
    Q_UNUSED(effectId)
    Q_UNUSED(paramName)
    Q_UNUSED(expression)
    Q_UNUSED(baseValue)
    qWarning() << "scriptSetEffectExpression: effect expressions are not supported in this build";
    return false;
}

bool MainWindow::scriptClearEffectExpression(int clipId, const QString &effectId, const QString &paramName)
{
    Q_UNUSED(clipId)
    Q_UNUSED(effectId)
    Q_UNUSED(paramName)
    qWarning() << "scriptClearEffectExpression: effect expressions are not supported in this build";
    return false;
}

QString MainWindow::scriptCopyClipEffects(int clipId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return QString();
    if (!timeline->model()->isClip(clipId)) return QString();

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack) return QString();

    QDomDocument doc;
    QDomElement xml = stack->toXml(doc);
    doc.appendChild(xml);
    return doc.toString(-1);
}

bool MainWindow::scriptPasteClipEffects(int targetClipId, const QString &effectsXml)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(targetClipId)) return false;

    auto stack = timeline->model()->getClipEffectStackModel(targetClipId);
    if (!stack) return false;

    QDomDocument doc;
    if (!doc.setContent(effectsXml)) return false;

    Fun undo = []() { return true; };
    Fun redo = []() { return true; };
    bool result = stack->fromXml(doc.documentElement(), undo, redo);
    if (result) {
        pCore->pushUndo(undo, redo, i18n("Paste effects"));
    }
    return result;
}

QVariantList MainWindow::scriptGetEffectKeyframes(int clipId, int effectIndex)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return {};
    if (!timeline->model()->isClip(clipId)) return {};

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    auto listModel = getEffectKeyframeModelAt(stack, effectIndex);
    if (!listModel) return {};

    double fps = pCore->getCurrentFps();
    // Get the first (primary) keyframe model from the list
    KeyframeModel *kfModel = listModel->getKeyModel();
    if (!kfModel) return {};

    QList<GenTime> positions = kfModel->getKeyframePos();
    QVariantList result;
    for (const GenTime &pos : positions) {
        bool ok = false;
        Keyframe kf = kfModel->getKeyframe(pos, &ok);
        if (!ok) continue;
        QVariantMap m;
        m[QStringLiteral("frame")] = pos.frames(fps);
        m[QStringLiteral("type")] = keyframeTypeName(static_cast<int>(kf.second));
        m[QStringLiteral("value")] = kfModel->getInterpolatedValue(pos).toString();
        result.append(m);
    }
    return result;
}

bool MainWindow::scriptAddEffectKeyframe(int clipId, int effectIndex, int frame, double normalizedValue, int keyframeType)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    auto listModel = getEffectKeyframeModelAt(stack, effectIndex);
    if (!listModel) return false;

    if (keyframeType >= 0) {
        double fps = pCore->getCurrentFps();
        return listModel->addKeyframe(GenTime(frame, fps), static_cast<KeyframeType::KeyframeEnum>(keyframeType));
    }
    return listModel->addKeyframe(frame, normalizedValue);
}

bool MainWindow::scriptRemoveEffectKeyframe(int clipId, int effectIndex, int frame)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    auto listModel = getEffectKeyframeModelAt(stack, effectIndex);
    if (!listModel) return false;

    double fps = pCore->getCurrentFps();
    return listModel->removeKeyframe(GenTime(frame, fps));
}

bool MainWindow::scriptUpdateEffectKeyframe(int clipId, int effectIndex, int oldFrame, int newFrame, double normalizedValue)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    auto listModel = getEffectKeyframeModelAt(stack, effectIndex);
    if (!listModel) return false;

    double fps = pCore->getCurrentFps();
    QVariant val = (normalizedValue >= 0) ? QVariant(normalizedValue) : QVariant();
    return listModel->updateKeyframe(GenTime(oldFrame, fps), GenTime(newFrame, fps), val);
}

QVariantList MainWindow::scriptGetEffectKeyframesByParam(int clipId, const QString &effectId, const QString &paramName)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return {};
    if (!timeline->model()->isClip(clipId)) return {};

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    auto [listModel, kfModel] = getEffectKeyframeByParam(stack, effectId, paramName);
    if (!kfModel) return {};

    double fps = pCore->getCurrentFps();
    QList<GenTime> positions = kfModel->getKeyframePos();
    QVariantList result;
    for (const GenTime &pos : positions) {
        bool ok = false;
        Keyframe kf = kfModel->getKeyframe(pos, &ok);
        if (!ok) continue;
        QVariantMap m;
        m[QStringLiteral("frame")] = pos.frames(fps);
        m[QStringLiteral("type")] = keyframeTypeName(static_cast<int>(kf.second));
        m[QStringLiteral("value")] = kfModel->getInterpolatedValue(pos).toString();
        result.append(m);
    }
    return result;
}

bool MainWindow::scriptAddEffectKeyframeByParam(int clipId, const QString &effectId, const QString &paramName,
                                                 int frame, const QString &value, int keyframeType)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    auto [listModel, kfModel] = getEffectKeyframeByParam(stack, effectId, paramName);
    if (!kfModel || !listModel) return false;

    double fps = pCore->getCurrentFps();
    GenTime pos(frame, fps);
    auto type = (keyframeType >= 0) ? static_cast<KeyframeType::KeyframeEnum>(keyframeType) : KeyframeType::Linear;
    if (!listModel->addKeyframe(pos, type)) return false;
    if (!value.isEmpty()) {
        kfModel->updateKeyframe(pos, QVariant(value));
    }
    return true;
}

bool MainWindow::scriptRemoveEffectKeyframeByParam(int clipId, const QString &effectId, const QString &paramName, int frame)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    auto [listModel, kfModel] = getEffectKeyframeByParam(stack, effectId, paramName);
    if (!kfModel || !listModel) return false;

    double fps = pCore->getCurrentFps();
    return listModel->removeKeyframe(GenTime(frame, fps));
}

bool MainWindow::scriptEnableTimeRemap(int clipId, bool enable)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;
    return timeline->model()->requestClipTimeRemap(clipId, enable);
}

QVariantMap MainWindow::scriptGetTimeRemap(int clipId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return {};
    if (!timeline->model()->isClip(clipId)) return {};

    auto clip = timeline->model()->getClipPtr(clipId);
    if (!clip) return {};

    QVariantMap result;
    result[QStringLiteral("enabled")] = clip->hasTimeRemap();

    if (clip->hasTimeRemap()) {
        QMap<QString, QString> vals = clip->getRemapValues();
        result[QStringLiteral("time_map")] = vals.value(QStringLiteral("time_map"));
        result[QStringLiteral("pitch")] = vals.value(QStringLiteral("pitch"), QStringLiteral("0")).toInt();
        result[QStringLiteral("image_mode")] = vals.value(QStringLiteral("image_mode"), QStringLiteral("nearest"));
    }
    return result;
}

bool MainWindow::scriptSetTimeRemap(int clipId, const QString &timeMap, int pitch, const QString &imageMode)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto clip = timeline->model()->getClipPtr(clipId);
    if (!clip || !clip->hasTimeRemap()) return false;

    if (!timeMap.isEmpty()) {
        clip->setRemapValue(QStringLiteral("time_map"), timeMap);
    }
    clip->setRemapValue(QStringLiteral("pitch"), QString::number(pitch));
    if (!imageMode.isEmpty()) {
        clip->setRemapValue(QStringLiteral("image_mode"), imageMode);
    }
    return true;
}

bool MainWindow::scriptSetClipSpeed(int clipId, double speed, bool pitchCompensate)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;
    // speed is percentage: 100=normal, 50=half, 200=double
    return timeline->model()->requestClipTimeWarp(clipId, speed, pitchCompensate, true);
}

QVariantList MainWindow::scriptGetClipTransformKeyframes(int clipId)
{
    QVariantList result;
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return result;
    if (!timeline->model()->isClip(clipId)) return result;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack || !stack->hasFilter(QStringLiteral("qtblend"))) return result;
    auto asset = stack->getAssetModelById(QStringLiteral("qtblend"));
    if (!asset) return result;

    // Get the KeyframeModelList, then the first KeyframeModel (rect param)
    auto kfModelList = asset->getKeyframeModel();
    if (!kfModelList) return result;
    auto *kfModel = kfModelList->getKeyModel();
    if (!kfModel) return result;

    for (int i = 0; i < kfModel->rowCount(); ++i) {
        QModelIndex idx = kfModel->index(i, 0);
        QVariantMap kf;
        kf[QStringLiteral("frame")] = kfModel->data(idx, Qt::UserRole);  // FrameRole
        kf[QStringLiteral("value")] = kfModel->data(idx, Qt::DisplayRole);  // ValueRole
        result.append(kf);
    }
    return result;
}

bool MainWindow::scriptSetClipTransform(int clipId, int frame, int x, int y, int width, int height, double opacity)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack) return false;
    // Ensure qtblend effect exists
    if (!stack->hasFilter(QStringLiteral("qtblend"))) {
        stack->appendEffect(QStringLiteral("qtblend"), false);
    }
    auto asset = stack->getAssetModelById(QStringLiteral("qtblend"));
    if (!asset) return false;

    // Build rect value in animation format: "frame=x y w h opacity"
    // where opacity is 0-100 (percentage)
    int intOpacity = qBound(0, static_cast<int>(opacity * 100.0), 100);
    QString newKf = QStringLiteral("%1=%2 %3 %4 %5 %6").arg(frame).arg(x).arg(y).arg(width).arg(height).arg(intOpacity);

    // Read existing rect value and merge keyframes
    QString existingValue;
    for (int i = 0; i < asset->rowCount(); ++i) {
        QModelIndex idx = asset->index(i, 0);
        QString paramName = asset->data(idx, AssetParameterModel::NameRole).toString();
        if (paramName == QLatin1String("rect")) {
            existingValue = asset->data(idx, AssetParameterModel::ValueRole).toString();
            break;
        }
    }

    // Parse existing keyframes into a map (frame -> "x y w h op")
    QMap<int, QString> keyframes;
    if (!existingValue.isEmpty() && existingValue.contains(QLatin1Char('='))) {
        const QStringList parts = existingValue.split(QLatin1Char(';'), Qt::SkipEmptyParts);
        for (const QString &part : parts) {
            int eqPos = part.indexOf(QLatin1Char('='));
            if (eqPos > 0) {
                bool ok;
                int kfFrame = part.left(eqPos).toInt(&ok);
                if (ok) {
                    keyframes[kfFrame] = part.mid(eqPos + 1);
                }
            }
        }
    }

    // Insert/update the new keyframe
    keyframes[frame] = QStringLiteral("%1 %2 %3 %4 %5").arg(x).arg(y).arg(width).arg(height).arg(intOpacity);

    // Build combined animation string sorted by frame
    QStringList result;
    for (auto it = keyframes.constBegin(); it != keyframes.constEnd(); ++it) {
        result.append(QStringLiteral("%1=%2").arg(it.key()).arg(it.value()));
    }
    QString rectValue = result.join(QLatin1Char(';'));

    // Set the combined animation value
    asset->setParameter(QStringLiteral("rect"), rectValue, true);
    return true;
}

bool MainWindow::scriptRemoveClipTransformKeyframe(int clipId, int frame)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack || !stack->hasFilter(QStringLiteral("qtblend"))) return false;
    auto asset = stack->getAssetModelById(QStringLiteral("qtblend"));
    if (!asset) return false;

    auto kfModelList = asset->getKeyframeModel();
    if (!kfModelList) return false;

    GenTime pos(frame, pCore->getCurrentFps());
    return kfModelList->removeKeyframe(pos);
}

double MainWindow::scriptGetClipOpacity(int clipId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return -1.0;
    if (!timeline->model()->isClip(clipId)) return -1.0;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack) return 1.0;

    // Look for qtblend effect and read its opacity param
    for (int i = 0; i < stack->rowCount(); ++i) {
        auto effect = std::static_pointer_cast<EffectItemModel>(stack->getEffectStackRow(i));
        if (effect && effect->getAssetId() == QLatin1String("qtblend")) {
            QString val = effect->filter().get("opacity");
            return val.isEmpty() ? 1.0 : val.toDouble() / 100.0;
        }
    }
    return 1.0;
}

bool MainWindow::scriptSetClipOpacity(int clipId, double opacity)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack) return false;

    // Look for existing qtblend effect
    for (int i = 0; i < stack->rowCount(); ++i) {
        auto effect = std::static_pointer_cast<EffectItemModel>(stack->getEffectStackRow(i));
        if (effect && effect->getAssetId() == QLatin1String("qtblend")) {
            int intOpacity = qBound(0, static_cast<int>(opacity * 100.0), 100);
            effect->filter().set("opacity", intOpacity);
            return true;
        }
    }
    // No qtblend found — add one with opacity param
    int intOpacity = qBound(0, static_cast<int>(opacity * 100.0), 100);
    QMap<QString, QString> params;
    params[QStringLiteral("opacity")] = QString::number(intOpacity);
    return stack->appendEffect(QStringLiteral("qtblend"), false, params);
}

bool MainWindow::scriptIsClipEnabled(int clipId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto clip = timeline->model()->getClipPtr(clipId);
    if (!clip) return false;
    return clip->clipState() == PlaylistState::Disabled ? false : true;
}

bool MainWindow::scriptSetClipEnabled(int clipId, bool enabled)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto clip = timeline->model()->getClipPtr(clipId);
    if (!clip) return false;

    PlaylistState::ClipState newState;
    if (enabled) {
        int tid = clip->getCurrentTrackId();
        bool isAudio = timeline->model()->isAudioTrack(tid);
        newState = isAudio ? PlaylistState::AudioOnly : PlaylistState::VideoOnly;
    } else {
        newState = PlaylistState::Disabled;
    }

    Fun undo = []() { return true; };
    Fun redo = []() { return true; };
    bool success = TimelineFunctions::changeClipState(timeline->model(), clipId, newState, undo, redo);
    if (success) {
        pCore->pushUndo(undo, redo, enabled ? i18n("Enable clip") : i18n("Disable clip"));
    }
    return success;
}

QString MainWindow::scriptGetClipColor(int clipId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return QString();
    if (!timeline->model()->isClip(clipId)) return QString();

    auto clip = timeline->model()->getClipPtr(clipId);
    if (!clip) return QString();
    return clip->clipTag();
}

bool MainWindow::scriptSetClipColor(int clipId, const QString &colorTag)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto clip = timeline->model()->getClipPtr(clipId);
    if (!clip) return false;
    QString binId = clip->binId();
    auto binClip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!binClip) return false;
    binClip->setTags(colorTag);
    return true;
}

bool MainWindow::scriptSplitAudio(int clipId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    QList<int> audioTracks = timeline->model()->getActiveAudioTrackIndexes();
    if (audioTracks.isEmpty()) return false;

    return TimelineFunctions::requestSplitAudio(timeline->model(), clipId, audioTracks);
}

bool MainWindow::scriptSetClipVolume(int clipId, double dB)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack) return false;

    // Find built-in volume effect (added by appendAudioBuildInEffects)
    for (int i = 0; i < stack->rowCount(); ++i) {
        auto effect = std::static_pointer_cast<EffectItemModel>(stack->getEffectStackRow(i));
        if (effect && effect->getAssetId() == QLatin1String("volume") && effect->isBuiltIn()) {
            effect->filter().set("level", dB);
            effect->filter().set("disable", 0); // enable it
            return true;
        }
    }
    // No built-in volume found — add a regular volume effect
    QMap<QString, QString> params;
    params[QStringLiteral("level")] = QString::number(dB);
    return stack->appendEffect(QStringLiteral("volume"), false, params);
}

double MainWindow::scriptGetClipVolume(int clipId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return 0.0;
    if (!timeline->model()->isClip(clipId)) return 0.0;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack) return 0.0;

    for (int i = 0; i < stack->rowCount(); ++i) {
        auto effect = std::static_pointer_cast<EffectItemModel>(stack->getEffectStackRow(i));
        if (effect && effect->getAssetId() == QLatin1String("volume")) {
            if (effect->filter().get_int("disable") == 1) return 0.0; // disabled = unity
            return effect->filter().get_double("level");
        }
    }
    return 0.0; // no volume effect = unity gain
}

bool MainWindow::scriptSetAudioFade(int clipId, int fadeInFrames, int fadeOutFrames)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack) return false;

    bool ok = true;
    if (fadeInFrames >= 0) {
        ok = stack->adjustFadeLength(fadeInFrames, true, true, false, true) && ok;
    }
    if (fadeOutFrames >= 0) {
        ok = stack->adjustFadeLength(fadeOutFrames, false, true, false, true) && ok;
    }
    return ok;
}

bool MainWindow::scriptSetClipPan(int clipId, double pan)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack) return false;

    // Ensure audiopan effect exists
    if (!stack->hasFilter(QStringLiteral("audiopan"))) {
        stack->appendEffect(QStringLiteral("audiopan"), false);
    }
    auto asset = stack->getAssetModelById(QStringLiteral("audiopan"));
    if (!asset) return false;

    // Pan value: 0.0 = full left, 0.5 = center, 1.0 = full right
    double normalized = qBound(0.0, (pan + 100.0) / 200.0, 1.0);
    asset->setParameter(QStringLiteral("start"), QString::number(normalized, 'f', 3), true);
    return true;
}

double MainWindow::scriptGetClipPan(int clipId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return 0.0;
    if (!timeline->model()->isClip(clipId)) return 0.0;

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack || !stack->hasFilter(QStringLiteral("audiopan"))) return 0.0;
    auto asset = stack->getAssetModelById(QStringLiteral("audiopan"));
    if (!asset) return 0.0;

    QString val = asset->getParam(QStringLiteral("start"));
    if (val.isEmpty()) return 0.0;
    // Convert 0.0-1.0 back to -100..+100
    return (val.toDouble() * 200.0) - 100.0;
}

bool MainWindow::scriptSetTrackMute(int trackId, bool mute)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isTrack(trackId)) return false;

    auto model = timeline->model();
    int currentHide = model->getTrackProperty(trackId, QStringLiteral("hide")).toInt();
    int newHide;
    if (mute) {
        newHide = currentHide | 2; // set audio-mute bit
    } else {
        newHide = currentHide & ~2; // clear audio-mute bit
    }
    model->setTrackProperty(trackId, QStringLiteral("hide"), QString::number(newHide));
    return true;
}

bool MainWindow::scriptGetTrackMute(int trackId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isTrack(trackId)) return false;
    return timeline->model()->getTrackProperty(trackId, QStringLiteral("hide")).toInt() & 2;
}

bool MainWindow::scriptSetTrackLocked(int trackId, bool locked)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    timeline->model()->setTrackLockedState(trackId, locked);
    return true;
}

bool MainWindow::scriptGetTrackLocked(int trackId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isTrack(trackId)) return false;
    return timeline->model()->trackIsLocked(trackId);
}

bool MainWindow::scriptSetTrackHidden(int trackId, bool hidden)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;

    auto model = timeline->model();
    int currentHide = model->getTrackProperty(trackId, QStringLiteral("hide")).toInt();
    int newHide;
    if (hidden) {
        newHide = currentHide | 1; // set video-hide bit
    } else {
        newHide = currentHide & ~1; // clear video-hide bit
    }
    model->setTrackProperty(trackId, QStringLiteral("hide"), QString::number(newHide));
    return true;
}

bool MainWindow::scriptGetTrackHidden(int trackId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isTrack(trackId)) return false;
    return timeline->model()->getTrackProperty(trackId, QStringLiteral("hide")).toInt() & 1;
}

QString MainWindow::scriptGetTrackName(int trackId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return QString();
    if (!timeline->model()->isTrack(trackId)) return QString();
    return timeline->model()->getTrackProperty(trackId, QStringLiteral("kdenlive:track_name")).toString();
}

bool MainWindow::scriptSetTrackName(int trackId, const QString &name)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    timeline->model()->setTrackProperty(trackId, QStringLiteral("kdenlive:track_name"), name);
    return true;
}

int MainWindow::scriptGetTrackColor(int trackId)
{
    // Kdenlive does not store per-track colors as properties.
    // Track colors are computed from the Qt palette in QML (Timeline.qml getTrackColor()).
    // There is no kdenlive:track_color or similar MLT property to read.
    Q_UNUSED(trackId);
    return -1;
}

bool MainWindow::scriptSetTrackColor(int trackId, int color)
{
    // Kdenlive does not support per-track custom colors.
    // Track colors are theme-derived (palette-based) in QML, not stored as properties.
    Q_UNUSED(trackId);
    Q_UNUSED(color);
    return false;
}

bool MainWindow::scriptGetTrackSolo(int trackId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isTrack(trackId)) return false;
    // Solo = all other audio tracks muted, this one not
    // Check via track property
    return timeline->model()->getTrackProperty(trackId, QStringLiteral("kdenlive:solo")).toInt() != 0;
}

bool MainWindow::scriptSetTrackSolo(int trackId, bool solo)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    // TODO: Verify kdenlive:solo property exists in TimelineModel
    timeline->model()->setTrackProperty(trackId, QStringLiteral("kdenlive:solo"), solo ? QStringLiteral("1") : QStringLiteral("0"));
    return true;
}

QVariantList MainWindow::scriptGetAudioLevels(const QString &binId, int stream, int downsample, int mode)
{
    QVariantList result;
    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) return result;

    QVector<int16_t> levels = clip->audioFrameCache(stream);
    if (levels.isEmpty()) return result;

    int16_t maxVal = clip->getAudioMax(stream);
    if (maxVal <= 0) maxVal = 1;

    int step = qMax(1, downsample);
    double normMax = double(maxVal);

    for (int i = 0; i < levels.size(); i += step) {
        int end = qMin(i + step, levels.size());
        if (mode == 1) {
            // RMS mode
            double sumSq = 0.0;
            int count = 0;
            for (int j = i; j < end; ++j) {
                double v = double(levels[j]) / normMax;
                sumSq += v * v;
                ++count;
            }
            result.append(count > 0 ? std::sqrt(sumSq / count) : 0.0);
        } else {
            // Peak mode (default)
            int16_t peak = 0;
            for (int j = i; j < end; ++j) {
                peak = qMax(peak, qAbs(levels[j]));
            }
            result.append(double(peak) / normMax);
        }
    }
    return result;
}

bool MainWindow::scriptAddGuide(int frame, const QString &comment, int category)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;

    auto guideModel = timeline->model()->getGuideModel();
    if (!guideModel) return false;

    double fps = pCore->getCurrentFps();
    GenTime pos(frame, fps);
    return guideModel->addMarker(pos, comment, category);
}

QVariantList MainWindow::scriptGetGuides()
{
    QVariantList result;
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return result;

    auto guideModel = timeline->model()->getGuideModel();
    if (!guideModel) return result;

    double fps = pCore->getCurrentFps();
    // MarkerListModel stores markers sorted by GenTime
    QList<CommentedTime> markers = guideModel->getAllMarkers();
    for (const CommentedTime &marker : markers) {
        QVariantMap m;
        m[QStringLiteral("frame")] = marker.time().frames(fps);
        m[QStringLiteral("comment")] = marker.comment();
        m[QStringLiteral("category")] = marker.markerType();
        result.append(m);
    }
    return result;
}

bool MainWindow::scriptDeleteGuide(int frame)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;

    auto guideModel = timeline->model()->getGuideModel();
    if (!guideModel) return false;

    double fps = pCore->getCurrentFps();
    GenTime pos(frame, fps);
    return guideModel->removeMarker(pos);
}

bool MainWindow::scriptDeleteGuidesByCategory(int category)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;

    auto guideModel = timeline->model()->getGuideModel();
    if (!guideModel) return false;

    double fps = pCore->getCurrentFps();
    QList<CommentedTime> markers = guideModel->getAllMarkers();
    bool anyDeleted = false;
    for (const CommentedTime &marker : markers) {
        if (marker.markerType() == category) {
            if (guideModel->removeMarker(marker.time())) {
                anyDeleted = true;
            }
        }
    }
    return anyDeleted;
}

bool MainWindow::scriptAddClipMarker(const QString &binId, int frame, const QString &comment, int category)
{
    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) return false;
    auto markerModel = clip->getMarkerModel();
    if (!markerModel) return false;
    double fps = pCore->getCurrentFps();
    GenTime pos(frame, fps);
    return markerModel->addMarker(pos, comment, category);
}

QVariantList MainWindow::scriptGetClipMarkers(const QString &binId)
{
    QVariantList result;
    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) return result;
    auto markerModel = clip->getMarkerModel();
    if (!markerModel) return result;
    double fps = pCore->getCurrentFps();
    QList<CommentedTime> markers = markerModel->getAllMarkers();
    for (const CommentedTime &marker : markers) {
        QVariantMap m;
        m[QStringLiteral("frame")] = marker.time().frames(fps);
        m[QStringLiteral("comment")] = marker.comment();
        m[QStringLiteral("category")] = marker.markerType();
        result.append(m);
    }
    return result;
}

bool MainWindow::scriptDeleteClipMarker(const QString &binId, int frame)
{
    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) return false;
    auto markerModel = clip->getMarkerModel();
    if (!markerModel) return false;
    double fps = pCore->getCurrentFps();
    GenTime pos(frame, fps);
    return markerModel->removeMarker(pos);
}

bool MainWindow::scriptDeleteClipMarkersByCategory(const QString &binId, int category)
{
    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) return false;
    auto markerModel = clip->getMarkerModel();
    if (!markerModel) return false;
    QList<CommentedTime> markers = markerModel->getAllMarkers();
    bool anyDeleted = false;
    for (const CommentedTime &marker : markers) {
        if (marker.markerType() == category) {
            if (markerModel->removeMarker(marker.time())) {
                anyDeleted = true;
            }
        }
    }
    return anyDeleted;
}

void MainWindow::scriptSeek(int frame)
{
    if (m_projectMonitor) {
        m_projectMonitor->slotSeek(frame);
    }
}

int MainWindow::scriptGetPosition()
{
    if (m_projectMonitor) {
        return m_projectMonitor->position();
    }
    return -1;
}

void MainWindow::scriptPlay()
{
    if (pCore->monitorManager()) {
        pCore->monitorManager()->slotPlay();
    }
}

void MainWindow::scriptPause()
{
    if (pCore->monitorManager()) {
        pCore->monitorManager()->slotPause();
    }
}

bool MainWindow::scriptSetPlaybackSpeed(double speed)
{
    if (!m_projectMonitor) return false;
    if (qFuzzyIsNull(speed)) {
        m_projectMonitor->slotPlay(); // toggle play/pause
        return true;
    }
    if (speed > 0) {
        m_projectMonitor->slotForward(speed);
    } else {
        m_projectMonitor->slotRewind(-speed);
    }
    return true;
}

double MainWindow::scriptGetPlaybackSpeed()
{
    if (!m_projectMonitor) return 0.0;
    auto *proxy = m_projectMonitor->getControllerProxy();
    if (!proxy) return 0.0;
    return proxy->property("speed").toDouble();
}

int MainWindow::scriptGoToNextMarker()
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->controller()) return -1;

    auto model = timeline->model()->getGuideModel();
    if (!model) return -1;

    int currentPos = pCore->getMonitorPosition();
    // Get all guides sorted by frame
    auto guides = model->getAllMarkers();
    for (const auto &marker : guides) {
        int frame = marker.time().frames(pCore->getCurrentFps());
        if (frame > currentPos) {
            pCore->seekMonitor(Wunjo::ProjectMonitor, frame);
            return frame;
        }
    }
    return -1; // No next marker
}

int MainWindow::scriptGoToPreviousMarker()
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->controller()) return -1;

    auto model = timeline->model()->getGuideModel();
    if (!model) return -1;

    int currentPos = pCore->getMonitorPosition();
    auto guides = model->getAllMarkers();
    int prevFrame = -1;
    for (const auto &marker : guides) {
        int frame = marker.time().frames(pCore->getCurrentFps());
        if (frame < currentPos) {
            prevFrame = frame;
        } else {
            break;
        }
    }
    if (prevFrame >= 0) {
        pCore->seekMonitor(Wunjo::ProjectMonitor, prevFrame);
    }
    return prevFrame;
}

int MainWindow::scriptGoToNextEdit()
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->controller()) return -1;

    int currentPos = pCore->getMonitorPosition();
    int nextEdit = -1;
    auto model = timeline->model();

    // Iterate all tracks (video + audio)
    QList<int> allTracks;
    allTracks.append(model->getTracksIds(false)); // video
    allTracks.append(model->getTracksIds(true));  // audio
    for (int tid : allTracks) {
        std::unordered_set<int> clipIds = model->getItemsInRange(tid, 0, -1, false);
        for (int cid : clipIds) {
            if (!model->isClip(cid)) continue;
            int pos = model->getClipPosition(cid);
            int endPos = pos + model->getClipPlaytime(cid);

            if (pos > currentPos && (nextEdit < 0 || pos < nextEdit)) {
                nextEdit = pos;
            }
            if (endPos > currentPos && (nextEdit < 0 || endPos < nextEdit)) {
                nextEdit = endPos;
            }
        }
    }

    if (nextEdit >= 0) {
        pCore->seekMonitor(Wunjo::ProjectMonitor, nextEdit);
    }
    return nextEdit;
}

int MainWindow::scriptGoToPreviousEdit()
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->controller()) return -1;

    int currentPos = pCore->getMonitorPosition();
    int prevEdit = -1;
    auto model = timeline->model();

    QList<int> allTracks;
    allTracks.append(model->getTracksIds(false));
    allTracks.append(model->getTracksIds(true));
    for (int tid : allTracks) {
        std::unordered_set<int> clipIds = model->getItemsInRange(tid, 0, -1, false);
        for (int cid : clipIds) {
            if (!model->isClip(cid)) continue;
            int pos = model->getClipPosition(cid);
            int endPos = pos + model->getClipPlaytime(cid);

            if (pos < currentPos && pos > prevEdit) {
                prevEdit = pos;
            }
            if (endPos < currentPos && endPos > prevEdit) {
                prevEdit = endPos;
            }
        }
    }

    if (prevEdit >= 0) {
        pCore->seekMonitor(Wunjo::ProjectMonitor, prevEdit);
    }
    return prevEdit;
}

QVariantList MainWindow::scriptDetectScenes(const QString &binClipId, double threshold, int minDuration)
{
    QVariantList result;

    // 1. Get clip from bin by ID
    auto clip = pCore->projectItemModel()->getClipByBinID(binClipId);
    if (!clip) {
        qWarning() << "scriptDetectScenes: clip not found:" << binClipId;
        return result;
    }

    // 2. Get source file path
    QString sourceUrl = clip->url();
    if (sourceUrl.isEmpty()) {
        qWarning() << "scriptDetectScenes: clip has no source URL";
        return result;
    }

    // 3. Build FFmpeg command (from SceneSplitTask logic)
    QString ffmpegPath = WunjoSettings::ffmpegpath();
    QStringList args;
    args << QStringLiteral("-y") << QStringLiteral("-loglevel") << QStringLiteral("info") << QStringLiteral("-i") << sourceUrl << QStringLiteral("-filter:v")
         << QStringLiteral("select='gt(scene,%1)',showinfo").arg(threshold) << QStringLiteral("-vsync") << QStringLiteral("vfr") << QStringLiteral("-f")
         << QStringLiteral("null") << QStringLiteral("-");

    // 4. Run FFmpeg synchronously
    QProcess process;
    process.start(ffmpegPath, args);
    process.waitForFinished(-1);

    // 5. Parse output for scene-change timestamps
    QString output = process.readAllStandardError();
    QStringList lines = output.split(QLatin1Char('\n'));

    double fps = pCore->getCurrentFps();
    QList<double> timestamps;

    for (const QString &line : lines) {
        if (line.contains(QStringLiteral("[Parsed_showinfo"))) {
            int pos = line.indexOf(QStringLiteral("pts_time:"));
            if (pos > -1) {
                QString timeStr = line.mid(pos + 9);
                int endPos = timeStr.indexOf(QLatin1Char(' '));
                if (endPos > -1) {
                    timeStr = timeStr.left(endPos);
                }
                bool ok;
                double time = timeStr.toDouble(&ok);
                if (ok) {
                    if (minDuration > 0 && !timestamps.isEmpty()) {
                        double minSecs = minDuration / fps;
                        if (time - timestamps.last() < minSecs) {
                            continue;
                        }
                    }
                    timestamps.append(time);
                }
            }
        }
    }

    // 6. Convert to QVariantList for D-Bus serialization
    for (double t : timestamps) {
        result.append(t);
    }

    return result;
}

bool MainWindow::scriptFillFrame(int clipId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isClip(clipId)) return false;

    // Get bin clip to read source resolution
    const QString binId = timeline->model()->getClipBinId(clipId);
    auto binClip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!binClip) return false;

    const QSize srcSize = binClip->frameSize();
    const int srcW = srcSize.width();
    const int srcH = srcSize.height();
    if (srcW <= 0 || srcH <= 0) return false;

    // Project resolution
    const int projW = pCore->getCurrentProfile()->width();
    const int projH = pCore->getCurrentProfile()->height();

    // Already matches — no fill needed
    if (srcW == projW && srcH == projH) return true;

    // Scale-to-fill: uniform scale so the smaller dimension fills the frame
    const double scale = qMax(double(projW) / srcW, double(projH) / srcH);
    const int w = qRound(srcW * scale);
    const int h = qRound(srcH * scale);
    const int x = (projW - w) / 2;
    const int y = (projH - h) / 2;

    const QString rect = QStringLiteral("%1 %2 %3 %4 1").arg(x).arg(y).arg(w).arg(h);

    // Apply qtblend effect with calculated rect
    QMap<QString, QString> params;
    params.insert(QStringLiteral("rect"), rect);
    params.insert(QStringLiteral("distort"), QStringLiteral("1"));

    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack) return false;
    return stack->appendEffect(QStringLiteral("qtblend"), false, params);
}

QString MainWindow::scriptRenderBinFrame(const QString &binId, int frame, int width, int height, const QString &outputPath)
{
    // 1. Get clip from bin
    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) {
        qWarning() << "scriptRenderBinFrame: clip not found:" << binId;
        return QString();
    }

    QString sourceUrl = clip->url();
    if (sourceUrl.isEmpty()) {
        qWarning() << "scriptRenderBinFrame: clip has no source URL";
        return QString();
    }

    // 2. Create producer and render frame
    Mlt::Producer producer(pCore->getProjectProfile(), sourceUrl.toUtf8().constData());
    if (!producer.is_valid()) {
        qWarning() << "scriptRenderBinFrame: invalid producer for" << sourceUrl;
        return QString();
    }

    QImage img = KThumb::getFrame(&producer, frame, width, height);
    if (img.isNull()) {
        qWarning() << "scriptRenderBinFrame: got null image";
        return QString();
    }

    // 3. Scale to requested size (aspect-preserving) and save
    if (img.width() != width || img.height() != height) {
        img = img.scaled(width, height, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }

    if (!img.save(outputPath, "JPEG", 90)) {
        qWarning() << "scriptRenderBinFrame: failed to save" << outputPath;
        return QString();
    }

    return outputPath;
}

QString MainWindow::scriptRenderTimelineFrame(int frame, int width, int height, const QString &outputPath)
{
    // 1. Get the timeline tractor (composited producer with all tracks, effects, transitions)
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) {
        qWarning() << "scriptRenderTimelineFrame: no active timeline";
        return QString();
    }

    Mlt::Tractor *tractor = timeline->model()->tractor();
    if (!tractor || !tractor->is_valid()) {
        qWarning() << "scriptRenderTimelineFrame: invalid tractor";
        return QString();
    }

    // 2. Render the composited frame
    QImage img = KThumb::getFrame(tractor, frame, width, height);
    if (img.isNull()) {
        qWarning() << "scriptRenderTimelineFrame: got null image";
        return QString();
    }

    // 3. Scale and save
    if (img.width() != width || img.height() != height) {
        img = img.scaled(width, height, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }

    if (!img.save(outputPath, "JPEG", 90)) {
        qWarning() << "scriptRenderTimelineFrame: failed to save" << outputPath;
        return QString();
    }

    return outputPath;
}

QString MainWindow::scriptCaptureWindow(int maxSize, const QString &outputPath)
{
    QPixmap pixmap = grab();
    if (pixmap.isNull()) {
        qWarning() << "scriptCaptureWindow: grab() returned null";
        return QString();
    }

    QImage img = pixmap.toImage();
    if (img.isNull()) {
        qWarning() << "scriptCaptureWindow: toImage() returned null";
        return QString();
    }

    if (maxSize > 0 && (img.width() > maxSize || img.height() > maxSize)) {
        img = img.scaled(maxSize, maxSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
    }

    if (!img.save(outputPath, "JPEG", 90)) {
        qWarning() << "scriptCaptureWindow: failed to save" << outputPath;
        return QString();
    }

    return outputPath;
}

QString MainWindow::scriptGetPanelGeometries()
{
    QJsonArray panels;

    // Main window geometry
    QRect mainGeo = geometry();
    QJsonObject mainObj;
    mainObj[QStringLiteral("name")] = QStringLiteral("main_window");
    mainObj[QStringLiteral("title")] = QStringLiteral("Main Window");
    mainObj[QStringLiteral("x")] = mainGeo.x();
    mainObj[QStringLiteral("y")] = mainGeo.y();
    mainObj[QStringLiteral("width")] = mainGeo.width();
    mainObj[QStringLiteral("height")] = mainGeo.height();
    mainObj[QStringLiteral("visible")] = true;
    panels.append(mainObj);

    // All dock widgets
    auto docks = mainDockWindow->findChildren<KDDockWidgets::QtWidgets::DockWidget *>();
    for (auto *dock : docks) {
        QJsonObject obj;
        obj[QStringLiteral("name")] = dock->objectName();
        obj[QStringLiteral("title")] = dock->title();
        bool vis = dock->isVisible();
        obj[QStringLiteral("visible")] = vis;
        if (vis) {
            // Map dock widget coordinates to main window coordinates
            QPoint topLeft = dock->mapToGlobal(QPoint(0, 0)) - this->mapToGlobal(QPoint(0, 0));
            obj[QStringLiteral("x")] = topLeft.x();
            obj[QStringLiteral("y")] = topLeft.y();
            obj[QStringLiteral("width")] = dock->width();
            obj[QStringLiteral("height")] = dock->height();
        } else {
            obj[QStringLiteral("x")] = 0;
            obj[QStringLiteral("y")] = 0;
            obj[QStringLiteral("width")] = 0;
            obj[QStringLiteral("height")] = 0;
        }
        panels.append(obj);
    }

    QJsonDocument doc(panels);
    return QString::fromUtf8(doc.toJson(QJsonDocument::Compact));
}

QVariantList MainWindow::scriptGetSubtitles()
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return {};
    if (!timeline->model()->hasSubtitleModel()) return {};
    auto subModel = timeline->model()->getSubtitleModel();
    double fps = pCore->getCurrentFps();
    auto allSubs = subModel->getAllSubtitles();
    QVariantList result;
    for (const auto &sub : allSubs) {
        int layer = sub.first.first;
        GenTime start = sub.first.second;
        const SubtitleEvent &event = sub.second;
        QVariantMap m;
        m[QStringLiteral("id")] = subModel->getIdForStartPos(layer, start);
        m[QStringLiteral("layer")] = layer;
        m[QStringLiteral("startFrame")] = start.frames(fps);
        m[QStringLiteral("endFrame")] = event.endTime().frames(fps);
        m[QStringLiteral("text")] = event.text();
        m[QStringLiteral("styleName")] = event.styleName();
        result.append(m);
    }
    return result;
}

int MainWindow::scriptAddSubtitle(int startFrame, int endFrame, const QString &text, int layer)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return -1;
    bool created = false;
    if (!timeline->model()->hasSubtitleModel()) {
        timeline->model()->createSubtitleModel();
        created = true;
    }
    auto subModel = timeline->model()->getSubtitleModel();
    if (created) {
        // Ensure QML view is connected to the new subtitle model
        pCore->subtitleWidget()->setModel(subModel);
        timeline->connectSubtitleModel(true);
        WunjoSettings::setShowSubtitles(true);
    }
    double fps = pCore->getCurrentFps();
    GenTime startPos(startFrame, fps);
    GenTime endPos(endFrame, fps);
    SubtitleEvent event(true, endPos, subModel->getLayerDefaultStyle(layer), QString(), 0, 0, 0, QString(), text);
    Fun undo = []() { return true; };
    Fun redo = []() { return true; };
    if (subModel->addSubtitle({layer, startPos}, event, undo, redo)) {
        pCore->pushUndo(undo, redo, i18n("Add subtitle"));
        // Force QML DelegateModel to pick up the new row
        Q_EMIT subModel->layoutChanged();
        return subModel->getIdForStartPos(layer, startPos);
    }
    return -1;
}

bool MainWindow::scriptEditSubtitle(int subtitleId, const QString &newText)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model() || !timeline->model()->hasSubtitleModel()) return false;
    return timeline->model()->getSubtitleModel()->editSubtitle(subtitleId, newText);
}

bool MainWindow::scriptMoveSubtitle(int subtitleId, int newStartFrame)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model() || !timeline->model()->hasSubtitleModel()) return false;
    auto subModel = timeline->model()->getSubtitleModel();
    double fps = pCore->getCurrentFps();
    GenTime newPos(newStartFrame, fps);
    int layer = subModel->getLayerForId(subtitleId);
    return subModel->moveSubtitle(subtitleId, layer, newPos, true, true);
}

bool MainWindow::scriptResizeSubtitle(int subtitleId, int newDuration, bool fromRight)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model() || !timeline->model()->hasSubtitleModel()) return false;
    return timeline->model()->getSubtitleModel()->requestResize(subtitleId, newDuration, fromRight);
}

bool MainWindow::scriptImportSubtitle(const QString &filePath, int offset, const QString &encoding)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!QFile::exists(filePath)) return false;
    bool created = false;
    if (!timeline->model()->hasSubtitleModel()) {
        timeline->model()->createSubtitleModel();
        created = true;
    }
    auto subModel = timeline->model()->getSubtitleModel();
    if (created) {
        pCore->subtitleWidget()->setModel(subModel);
        timeline->connectSubtitleModel(true);
        WunjoSettings::setShowSubtitles(true);
    }
    double fps = pCore->getCurrentFps();
    QByteArray enc = encoding.isEmpty() ? QByteArrayLiteral("UTF-8") : encoding.toUtf8();
    subModel->importSubtitle(filePath, offset, true, fps, fps, enc);
    return true;
}

bool MainWindow::scriptDeleteSubtitle(int subtitleId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model() || !timeline->model()->hasSubtitleModel()) return false;
    return timeline->model()->getSubtitleModel()->removeSubtitle(subtitleId);
}

bool MainWindow::scriptExportSubtitles(const QString &filePath)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model() || !timeline->model()->hasSubtitleModel()) return false;
    auto subModel = timeline->model()->getSubtitleModel();
    int ix = pCore->currentDoc()->getSequenceProperty(timeline->model()->uuid(), QStringLiteral("kdenlive:activeSubtitleIndex"), QStringLiteral("0")).toInt();
    subModel->copySubtitle(filePath, ix, false, false);
    return QFile::exists(filePath);
}

bool MainWindow::scriptSpeechRecognition(const QString &model, const QString &language, int maxLineWidth)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    const int duration = timeline->model()->duration();
    if (duration <= 0) return false;
    // The recognition result is imported into the subtitle model; in the GUI
    // that model is created when subtitles are first shown, so create it here
    // on demand (mirrors slotEditSubtitle) — otherwise the import dereferences
    // a null model and crashes.
    if (!timeline->hasSubtitles()) {
        std::shared_ptr<SubtitleModel> subtitleModel = timeline->model()->createSubtitleModel();
        pCore->subtitleWidget()->setModel(subtitleModel);
        WunjoSettings::setShowSubtitles(true);
        timeline->connectSubtitleModel(true);
    }
    // Headless: the dialog is configured programmatically, never shown, and
    // self-deletes when the transcription job finishes (SRT auto-imported).
    auto *dialog = new SpeechDialog(timeline->model(), QPoint(0, duration), -1, false, false, this);
    if (!dialog->startHeadless(model, language, maxLineWidth)) {
        dialog->deleteLater();
        return false;
    }
    return true;
}

QVariantList MainWindow::scriptGetSubtitleStyles(bool global)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return {};
    if (!timeline->model()->hasSubtitleModel()) return {};
    auto subModel = timeline->model()->getSubtitleModel();
    const auto &styles = subModel->getAllSubtitleStyles(global);
    QVariantList result;
    for (const auto &pair : styles) {
        result.append(subtitleStyleToMap(pair.first, pair.second));
    }
    return result;
}

bool MainWindow::scriptSetSubtitleStyle(const QString &name, const QStringList &keys, const QStringList &values, bool global)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->hasSubtitleModel()) return false;
    auto subModel = timeline->model()->getSubtitleModel();

    // Start from existing style or defaults
    const auto &styles = subModel->getAllSubtitleStyles(global);
    SubtitleStyle style;
    auto it = styles.find(name);
    if (it != styles.end()) {
        style = it->second;
    }

    applyStyleOverrides(style, keys, values);
    subModel->setSubtitleStyle(name, style, global);
    Q_EMIT subModel->layoutChanged();
    return true;
}

bool MainWindow::scriptDeleteSubtitleStyle(const QString &name, bool global)
{
    if (name == QLatin1String("Default")) return false;
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->hasSubtitleModel()) return false;
    auto subModel = timeline->model()->getSubtitleModel();
    subModel->deleteSubtitleStyle(name, global);
    Q_EMIT subModel->layoutChanged();
    return true;
}

bool MainWindow::scriptSetSubtitleStyleName(int subtitleId, const QString &styleName)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->hasSubtitleModel()) return false;
    auto subModel = timeline->model()->getSubtitleModel();
    subModel->setStyleName(subtitleId, styleName);
    Q_EMIT subModel->layoutChanged();
    return true;
}

QString MainWindow::scriptCreateSequence(const QString &name, int audioTracks, int videoTracks,
                                          const QString &parentFolder)
{
    if (!pCore->currentDoc()) return QStringLiteral("-1");

    int aTracks = (audioTracks <= 0) ? WunjoSettings::audiotracks() : audioTracks;
    int vTracks = (videoTracks <= 0) ? WunjoSettings::videotracks() : videoTracks;

    Fun undo = []() { return true; };
    Fun redo = []() { return true; };

    QString binId = ClipCreator::createPlaylistClipWithUndo(
        name, {aTracks, vTracks}, parentFolder, pCore->projectItemModel(), undo, redo);

    if (binId != QStringLiteral("-1")) {
        pCore->pushUndo(undo, redo, i18n("Create sequence"));
    }
    return binId;
}

QVariantList MainWindow::scriptGetSequences()
{
    WunjoDoc *project = pCore->currentDoc();
    if (!project) return {};
    QList<QUuid> uuids = project->getTimelinesUuids();
    QUuid activeUuid;
    if (getCurrentTimeline()) {
        activeUuid = getCurrentTimeline()->getUuid();
    }
    QVariantList result;
    for (const QUuid &uuid : uuids) {
        auto model = project->getTimeline(uuid, true);
        if (!model) continue;
        QVariantMap m;
        m[QStringLiteral("uuid")] = uuid.toString(QUuid::WithoutBraces);
        m[QStringLiteral("name")] = model->tractor() ? QString(model->tractor()->get("kdenlive:clipname")) : QString();
        m[QStringLiteral("duration")] = model->duration();
        m[QStringLiteral("tracks")] = model->getTracksCount();
        m[QStringLiteral("active")] = (uuid == activeUuid);
        result.append(m);
    }
    return result;
}

QVariantMap MainWindow::scriptGetActiveSequence()
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return {};
    WunjoDoc *project = pCore->currentDoc();
    if (!project) return {};
    QUuid uuid = timeline->getUuid();
    auto model = timeline->model();
    QVariantMap m;
    m[QStringLiteral("uuid")] = uuid.toString(QUuid::WithoutBraces);
    m[QStringLiteral("name")] = model->tractor() ? QString(model->tractor()->get("kdenlive:clipname")) : QString();
    m[QStringLiteral("duration")] = model->duration();
    m[QStringLiteral("tracks")] = model->getTracksCount();
    return m;
}

bool MainWindow::scriptSetActiveSequence(const QString &uuid)
{
    QUuid targetUuid = QUuid::fromString(uuid);
    if (targetUuid.isNull()) return false;
    // First try to raise an already-open tab
    if (raiseTimeline(targetUuid)) {
        return true;
    }
    // Tab not open — find the bin clip for this sequence and open it
    WunjoDoc *project = pCore->currentDoc();
    if (!project) return false;
    // Search all bin clips for one matching this UUID
    auto binModel = pCore->projectItemModel();
    std::vector<QString> allIds = binModel->getAllClipIds();
    for (const QString &binId : allIds) {
        auto clip = binModel->getClipByBinID(binId);
        if (clip && clip->getSequenceUuid() == targetUuid) {
            return pCore->projectManager()->openTimeline(binId, -1, targetUuid, -1);
        }
    }
    return false;
}

QVariantMap MainWindow::scriptGetZone()
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->controller()) return {};
    QVariantMap m;
    m[QStringLiteral("zoneIn")] = timeline->controller()->zoneIn();
    m[QStringLiteral("zoneOut")] = timeline->controller()->zoneOut();
    return m;
}

bool MainWindow::scriptSetZone(int inFrame, int outFrame)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->controller()) return false;
    timeline->controller()->setZone(QPoint(inFrame, outFrame), true);
    return true;
}

bool MainWindow::scriptSetZoneIn(int inFrame)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->controller()) return false;
    timeline->controller()->setZoneIn(inFrame);
    return true;
}

bool MainWindow::scriptSetZoneOut(int outFrame)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->controller()) return false;
    timeline->controller()->setZoneOut(outFrame);
    return true;
}

bool MainWindow::scriptExtractZone(int inFrame, int outFrame, bool liftOnly)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->controller()) return false;
    timeline->controller()->extractZone(QPoint(inFrame, outFrame), liftOnly);
    return true;
}

int MainWindow::scriptGroupClips(const QList<int> &itemIds)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return -1;

    std::unordered_set<int> ids(itemIds.begin(), itemIds.end());
    if (static_cast<int>(ids.size()) < 2) return -1;

    return timeline->model()->requestClipsGroup(ids, true, GroupType::Normal);
}

bool MainWindow::scriptUngroupClips(int itemId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isInGroup(itemId)) return false;

    return timeline->model()->requestClipUngroup(itemId, true);
}

QVariantMap MainWindow::scriptGetGroupInfo(int itemId)
{
    QVariantMap result;
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return result;

    auto model = timeline->model();

    bool inGroup = model->isInGroup(itemId);
    result[QStringLiteral("isInGroup")] = inGroup;
    result[QStringLiteral("isGroup")] = model->isGroup(itemId);

    if (!inGroup) {
        // Item is not grouped — return minimal info, avoid calling getType on a leaf
        result[QStringLiteral("rootId")] = itemId;
        result[QStringLiteral("groupType")] = QStringLiteral("Leaf");
        result[QStringLiteral("members")] = QVariantList();
        return result;
    }

    int rootId = model->getGroupRootId(itemId);
    result[QStringLiteral("rootId")] = rootId;
    result[QStringLiteral("groupType")] = groupTypeToStr(model->getGroupType(rootId));

    // Collect leaf members (clips/compositions) of the root group
    std::unordered_set<int> leaves = model->getGroupElements(itemId);
    QVariantList memberList;
    memberList.reserve(int(leaves.size()));
    for (int leaf : leaves) {
        QVariantMap member;
        member[QStringLiteral("id")] = leaf;
        if (model->isClip(leaf)) {
            member[QStringLiteral("type")] = QStringLiteral("clip");
            member[QStringLiteral("trackId")] = model->getClipTrackId(leaf);
            member[QStringLiteral("position")] = model->getClipPosition(leaf);
        } else if (model->isComposition(leaf)) {
            member[QStringLiteral("type")] = QStringLiteral("composition");
            member[QStringLiteral("trackId")] = model->getCompositionTrackId(leaf);
            member[QStringLiteral("position")] = model->getCompositionPosition(leaf);
        }
        memberList.append(member);
    }
    result[QStringLiteral("members")] = memberList;

    return result;
}

bool MainWindow::scriptRemoveFromGroup(int itemId)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    if (!timeline->model()->isInGroup(itemId)) return false;

    return timeline->model()->requestRemoveFromGroup(itemId, true);
}

QVariantMap MainWindow::scriptGetClipProxyStatus(const QString &binId)
{
    QVariantMap result;
    if (!pCore->currentDoc()) return result;

    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) return result;

    result[QStringLiteral("supportsProxy")] = clip->supportsProxy();
    result[QStringLiteral("hasProxy")] = clip->hasProxy();
    QString proxyPath = clip->getProducerProperty(QStringLiteral("kdenlive:proxy"));
    result[QStringLiteral("proxyPath")] = proxyPath;
    result[QStringLiteral("originalUrl")] = clip->getProducerProperty(QStringLiteral("kdenlive:originalurl"));
    ObjectId oid(WunjoObjectType::BinClip, binId.toInt(), QUuid());
    result[QStringLiteral("isGenerating")] = pCore->taskManager.hasPendingJob(oid, AbstractTask::PROXYJOB);
    return result;
}

bool MainWindow::scriptSetClipProxy(const QString &binId, bool enabled)
{
    if (!pCore->currentDoc()) return false;

    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) return false;
    if (!clip->supportsProxy()) return false;

    QList<std::shared_ptr<ProjectClip>> clipList{clip};
    pCore->currentDoc()->slotProxyCurrentItem(enabled, clipList);
    return true;
}

bool MainWindow::scriptDeleteClipProxy(const QString &binId)
{
    if (!pCore->currentDoc()) return false;

    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) return false;
    if (!clip->supportsProxy()) return false;

    clip->deleteProxy(true);
    return true;
}

bool MainWindow::scriptRebuildClipProxy(const QString &binId)
{
    if (!pCore->currentDoc()) return false;

    auto clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) return false;
    if (!clip->supportsProxy()) return false;

    QList<std::shared_ptr<ProjectClip>> clipList{clip};
    pCore->currentDoc()->slotProxyCurrentItem(true, clipList, true);
    return true;
}

QVariantList MainWindow::scriptGetSelection()
{
    QVariantList result;
    if (!getCurrentTimeline() || !getCurrentTimeline()->controller()) return result;
    const QList<int> sel = getCurrentTimeline()->controller()->selection();
    for (int id : sel) {
        result.append(id);
    }
    return result;
}

bool MainWindow::scriptSetSelection(const QList<int> &ids)
{
    if (!getCurrentTimeline() || !getCurrentTimeline()->controller()) return false;
    getCurrentTimeline()->controller()->selectItems(ids);
    return true;
}

bool MainWindow::scriptAddToSelection(int itemId, bool clear)
{
    if (!getCurrentTimeline() || !getCurrentTimeline()->controller()) return false;
    auto model = getCurrentTimeline()->controller()->getModel();
    if (!model) return false;
    model->requestAddToSelection(itemId, clear);
    return true;
}

bool MainWindow::scriptClearSelection()
{
    if (!getCurrentTimeline() || !getCurrentTimeline()->controller()) return false;
    auto model = getCurrentTimeline()->controller()->getModel();
    if (!model) return false;
    return model->requestClearSelection();
}

bool MainWindow::scriptSelectAll()
{
    if (!getCurrentTimeline() || !getCurrentTimeline()->controller()) return false;
    getCurrentTimeline()->controller()->selectAll();
    return true;
}

bool MainWindow::scriptSelectCurrentTrack()
{
    if (!getCurrentTimeline() || !getCurrentTimeline()->controller()) return false;
    getCurrentTimeline()->controller()->selectCurrentTrack();
    return true;
}

bool MainWindow::scriptSelectItems(const QList<int> &trackIds, int startFrame, int endFrame)
{
    if (!getCurrentTimeline() || !getCurrentTimeline()->controller()) return false;
    QVariantList tracks;
    for (int tid : trackIds) {
        tracks.append(tid);
    }
    // addToSelect=false, selectBottomCompositions=true, selectSubTitles=true
    getCurrentTimeline()->controller()->selectItems(tracks, startFrame, endFrame, false, true, true);
    return true;
}


// ── Chat narration: push conversation + progress into the chat dock ──

bool MainWindow::scriptChatMessage(int author, const QString &text)
{
    if (!m_chatWidget) return false;
    m_chatWidget->externalMessage(author, text);
    return true;
}

bool MainWindow::scriptChatStream(const QString &text)
{
    if (!m_chatWidget) return false;
    m_chatWidget->externalStreamAssistant(text);
    return true;
}

bool MainWindow::scriptChatThinking(bool on)
{
    if (!m_chatWidget) return false;
    m_chatWidget->externalThinking(on);
    return true;
}

bool MainWindow::scriptChatToolStart(const QString &id, const QString &name)
{
    if (!m_chatWidget) return false;
    m_chatWidget->externalToolStart(id, name);
    return true;
}

bool MainWindow::scriptChatToolProgress(const QString &id, int percent, const QString &message)
{
    if (!m_chatWidget) return false;
    m_chatWidget->externalToolProgress(id, percent, message);
    return true;
}

bool MainWindow::scriptChatToolEnd(const QString &id, bool isError, const QString &result)
{
    if (!m_chatWidget) return false;
    m_chatWidget->externalToolEnd(id, isError, result);
    return true;
}

// ── Assistant guidance: skills & loops (backed by ChatGuidanceStore) ──

QStringList MainWindow::scriptListSkills()
{
    return ChatGuidanceStore::list(ChatGuidanceStore::Kind::Skill);
}

QString MainWindow::scriptGetSkill(const QString &name)
{
    return ChatGuidanceStore::read(ChatGuidanceStore::Kind::Skill, name);
}

bool MainWindow::scriptSaveSkill(const QString &name, const QString &content)
{
    const bool ok = ChatGuidanceStore::write(ChatGuidanceStore::Kind::Skill, name, content);
    if (ok && m_chatWidget) {
        m_chatWidget->refreshGuidance();
    }
    return ok;
}

bool MainWindow::scriptDeleteSkill(const QString &name)
{
    const bool ok = ChatGuidanceStore::remove(ChatGuidanceStore::Kind::Skill, name);
    if (ok && m_chatWidget) {
        m_chatWidget->refreshGuidance();
    }
    return ok;
}

QStringList MainWindow::scriptGetSelectedSkills()
{
    return ChatGuidanceStore::selectedSkills();
}

bool MainWindow::scriptSetSelectedSkills(const QStringList &names)
{
    const QStringList library = ChatGuidanceStore::list(ChatGuidanceStore::Kind::Skill);
    for (const QString &name : names) {
        if (!library.contains(name)) {
            return false;
        }
    }
    ChatGuidanceStore::setSelectedSkills(names);
    if (m_chatWidget) {
        m_chatWidget->refreshGuidance();
    }
    return true;
}

QStringList MainWindow::scriptListLoops()
{
    return ChatGuidanceStore::list(ChatGuidanceStore::Kind::Loop);
}

QString MainWindow::scriptGetLoop(const QString &name)
{
    return ChatGuidanceStore::read(ChatGuidanceStore::Kind::Loop, name);
}

bool MainWindow::scriptSaveLoop(const QString &name, const QString &content)
{
    const bool ok = ChatGuidanceStore::write(ChatGuidanceStore::Kind::Loop, name, content);
    if (ok && m_chatWidget) {
        m_chatWidget->refreshGuidance();
    }
    return ok;
}

bool MainWindow::scriptDeleteLoop(const QString &name)
{
    const bool ok = ChatGuidanceStore::remove(ChatGuidanceStore::Kind::Loop, name);
    if (ok && m_chatWidget) {
        m_chatWidget->refreshGuidance();
    }
    return ok;
}

QString MainWindow::scriptGetSelectedLoop()
{
    return ChatGuidanceStore::selectedLoop();
}

bool MainWindow::scriptSelectLoop(const QString &name)
{
    if (!name.isEmpty() && !ChatGuidanceStore::list(ChatGuidanceStore::Kind::Loop).contains(name)) {
        return false;
    }
    ChatGuidanceStore::setSelectedLoop(name);
    if (m_chatWidget) {
        m_chatWidget->refreshGuidance();
    }
    return true;
}

// ── AI plugins: enumerate + run headless (job.json + QProcess machinery) ──

QVariantList MainWindow::scriptListPlugins()
{
    // The plugin's own manifest, handed over as it was written.
    //
    // Anything the assistant needs to know about a plugin — what it is for,
    // what it takes, what its effects are called — the author already wrote in
    // plugin.json. Picking those fields out here would mean a plugin could only
    // say what this function had been taught to repeat, and a plugin somebody
    // writes next month would be mute until the editor was changed to suit it.
    // So the file is passed through, less the parts that are the editor's own
    // business (how the environment is built, where the weights come from) and
    // the parts that must never reach a model at all.
    static const QStringList internal = {
        QStringLiteral("manifest_version"),  QStringLiteral("entry"),         QStringLiteral("python"),
        QStringLiteral("requirements"),      QStringLiteral("requirements_cuda"),
        QStringLiteral("venv"),              QStringLiteral("os"),            QStringLiteral("models"),
        QStringLiteral("icon"),              QStringLiteral("hardware"),      QStringLiteral("bundle_models"),
    };

    QVariantList result;
    const QList<PluginManifest> plugins = PluginManager::instance().installedPlugins();
    for (const PluginManifest &manifest : plugins) {
        // An assistant plugin is whoever is asking — offering it as a tool only
        // invites it to call itself.
        if (manifest.isAgent()) {
            continue;
        }
        QFile file(manifest.rootDir() + QStringLiteral("/plugin.json"));
        if (!file.open(QIODevice::ReadOnly)) {
            continue;
        }
        QJsonObject declared = QJsonDocument::fromJson(file.readAll()).object();
        for (const QString &key : internal) {
            declared.remove(key);
        }
        // The provider block names the setting an API key is stored under; the
        // key itself never leaves the editor, and neither does its name.
        if (declared.contains(QStringLiteral("provider"))) {
            QJsonObject provider = declared.value(QStringLiteral("provider")).toObject();
            provider.remove(QStringLiteral("key_setting"));
            declared.insert(QStringLiteral("provider"), provider);
        }
        // The id is what every other tool addresses the plugin by, so it is the
        // one field that cannot be left to the manifest to remember.
        declared.insert(QStringLiteral("id"), manifest.id());
        // The manifest names its effects by file, because that is what the
        // packer needs; every tool that puts one on a clip needs the effect's
        // id instead, and that lives inside the XML. Naming them the way they
        // will have to be addressed is the difference between a plugin that can
        // be used and a list of paths.
        if (declared.contains(QStringLiteral("effects"))) {
            QJsonArray ids;
            const QList<PluginEffect> effects = manifest.effects();
            for (const PluginEffect &effect : effects) {
                QJsonObject e;
                e.insert(QStringLiteral("id"), effect.id);
                e.insert(QStringLiteral("name"), effect.name);
                if (!effect.requiresEffect.isEmpty()) {
                    e.insert(QStringLiteral("requires"), effect.requiresEffect);
                }
                ids.append(e);
            }
            declared.insert(QStringLiteral("effects"), ids);
        }
        result.append(declared.toVariantMap());
    }
    return result;
}

QString MainWindow::scriptRunPlugin(const QString &id, const QString &inputJson)
{
    const PluginManifest manifest = PluginManager::instance().plugin(id);
    // Assistant plugins are started by the chat, never by a tool call: letting
    // one launch another (or itself) would nest agents nobody asked for.
    // The id is what says whether it is installed — an unknown one comes back
    // as an empty manifest, which has no errors and so calls itself valid.
    if (manifest.id().isEmpty() || manifest.isAgent()) {
        return {};
    }
    const QJsonDocument doc = QJsonDocument::fromJson(inputJson.toUtf8());
    if (!doc.isObject()) {
        return {};
    }
    // Non-blocking, and the answer is not "started": it is the id of the job, so the caller can ask what became of it — a plugin
    // that declines says so in one sentence, and an assistant that cannot read
    // it reports work that never happened. @ref scriptPluginJobStatus reads it.
    return PluginManager::instance().runPlugin(id, doc.object(), this);
}

QString MainWindow::scriptGenerateEffect(int clipId, const QString &effectId)
{
    // The render an effect describes, started the way its button starts it.
    //
    // A plugin that cannot run inside MLT — a face swap, a lip sync — puts its
    // settings on an effect and renders from them when asked. Asking was a
    // button on the effect and nothing else, so an assistant could set every
    // parameter correctly and still leave the clip untouched, with no way to
    // say what was missing. This is that button.
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model() || !timeline->model()->isClip(clipId)) {
        return {};
    }
    auto stack = timeline->model()->getClipEffectStackModel(clipId);
    if (!stack) {
        return {};
    }
    auto asset = stack->getAssetModelById(effectId);
    if (!asset) {
        return {};
    }
    const QString pluginId = PluginManager::instance().pluginForEffect(effectId);
    if (pluginId.isEmpty()) {
        return {};
    }
    auto effect = std::dynamic_pointer_cast<EffectItemModel>(asset);
    if (!effect) {
        return {};
    }
    // Which parameter the produced file is written into. It is named by the
    // render parameter itself, the same place the button reads it from, so an
    // effect that renders differently keeps working without changes here.
    QString resultParam;
    for (int row = 0; row < asset->rowCount(); ++row) {
        const QVariantList jobParams = asset->data(asset->index(row, 0), AssetParameterModel::FilterJobParamsRole).toList();
        for (const QVariant &entry : jobParams) {
            const QStringList pair = entry.toStringList();
            if (pair.size() == 2 && pair.at(0) == QLatin1String("key")) {
                resultParam = pair.at(1);
            }
        }
        if (!resultParam.isEmpty()) {
            break;
        }
    }
    const QJsonObject input = PluginEffects::buildJob(asset, pluginId);
    if (input.isEmpty()) {
        return {};
    }
    // The job's id: a render takes minutes, and whoever asked for it has to be
    // able to come back and ask how it went — including when it declined to
    // start at all, which used to be a message banner and nothing else.
    return PluginManager::instance().runEffectJob(pluginId, asset->getOwnerId(), effect->getId(), resultParam, input);
}

QString MainWindow::scriptPluginJobStatus(const QString &jobId)
{
    const QJsonObject outcome = PluginManager::instance().jobOutcome(jobId);
    if (outcome.isEmpty()) {
        // Not a job this editor started, or one it has long forgotten. Say so
        // rather than answering "running" and being waited on for ever.
        return QStringLiteral(R"({"state":"unknown"})");
    }
    return QString::fromUtf8(QJsonDocument(outcome).toJson(QJsonDocument::Compact));
}

QVariantList MainWindow::scriptListPluginSets(const QString &pluginId, const QString &kind)
{
    // What a plugin has recorded so far, and where each one lives.
    //
    // An effect that reads a set wants the file, not the name it is shown under
    // — the list widget resolves that silently, and anything driving the editor
    // from outside had no way to. Recording a face and then being unable to say
    // which face is a dead end, so the paths are reported.
    QVariantList result;
    const QVector<PluginSets::Set> sets = PluginSets::sets(pluginId, kind);
    for (const PluginSets::Set &set : sets) {
        QVariantMap entry;
        entry.insert(QStringLiteral("name"), set.name);
        entry.insert(QStringLiteral("kind"), set.kind);
        entry.insert(QStringLiteral("file"), set.file);
        entry.insert(QStringLiteral("source"), set.source);
        entry.insert(QStringLiteral("count"), set.count);
        result.append(entry);
    }
    return result;
}

AutomaskHelper *MainWindow::scriptMaskHelper()
{
    if (!m_scriptMaskHelper) {
        m_scriptMaskHelper = new AutomaskHelper(this);
        connect(m_scriptMaskHelper, &AutomaskHelper::samJobFinished, this, [this]() { m_scriptSamReady = true; });
        connect(m_scriptMaskHelper, &AutomaskHelper::showMessage, this,
                [this](const QString &message, KMessageWidget::MessageType type) {
                    // The panel shows these in a banner; a script has nowhere to
                    // look, so the last one is kept for scriptSamStatus.
                    if (type == KMessageWidget::Error || type == KMessageWidget::Warning) {
                        m_scriptMaskError = message;
                    }
                });
    }
    return m_scriptMaskHelper;
}

bool MainWindow::scriptSamPrepare(const QString &binId, int zoneIn, int zoneOut)
{
    // Step one of three: give Segment Anything the frames to look at.
    //
    // It works on stills, not on a video file, so the zone is exported first —
    // the same export the mask panel does before it lets anyone click. Long
    // enough to be worth reporting: this returns at once and the caller polls
    // scriptSamStatus.
    std::shared_ptr<ProjectClip> clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip || zoneOut <= zoneIn) {
        return false;
    }
    if (!scriptMaskHelper()->pythonReady()) {
        m_scriptMaskError = i18n("The object detection environment is not installed yet.");
        return false;
    }
    bool ok = false;
    QDir sourceFolder = pCore->currentDoc()->getCacheDir(CacheMaskSource, &ok);
    if (!ok) {
        m_scriptMaskError = i18n("Cannot access the project folder.");
        return false;
    }
    if (sourceFolder.dirName() == QLatin1String("source-frames")) {
        sourceFolder.removeRecursively();
        sourceFolder.mkpath(QStringLiteral("."));
    }

    QTemporaryFile playlist(QDir::temp().absoluteFilePath(QStringLiteral("XXXXXX.mlt")));
    if (!playlist.open()) {
        return false;
    }
    playlist.close();
    {
        Mlt::Consumer consumer(pCore->getProjectProfile(), "xml", playlist.fileName().toUtf8().constData());
        Mlt::Playlist list(pCore->getProjectProfile());
        Mlt::Producer producer(clip->originalProducer()->parent());
        list.append(producer, zoneIn, zoneOut);
        consumer.connect(list);
        consumer.run();
    }
    playlist.setAutoRemove(false);

    m_scriptMaskError.clear();
    m_scriptMaskFile.clear();
    m_scriptSamReady = false;
    const QStringList args = {QStringLiteral("xml:%1").arg(playlist.fileName()), QStringLiteral("-consumer"),
                              QStringLiteral("avformat:%1").arg(sourceFolder.absoluteFilePath(QStringLiteral("%05d.jpg"))),
                              QStringLiteral("start_number=0"), QStringLiteral("progress=1"), QStringLiteral("-preset"),
                              QStringLiteral("stills/JPEG")};
    const ObjectId owner(WunjoObjectType::BinClip, binId.toInt());
    MeltTask::start(owner, binId, playlist.fileName(), args, i18n("Exporting video frames"), clip.get(), [this, zoneIn, owner]() {
        bool ok = false;
        QDir frames = pCore->currentDoc()->getCacheDir(CacheMask, &ok);
        if (!ok) {
            return;
        }
        if (!frames.exists(QStringLiteral("source-frames"))) {
            frames.mkpath(QStringLiteral("source-frames"));
        }
        frames.cd(QStringLiteral("source-frames"));
        scriptMaskHelper()->launchSam(frames, zoneIn, owner, false, zoneIn);
        m_scriptSamReady = true;
    });
    return true;
}

QVariantMap MainWindow::scriptSamStatus()
{
    QVariantMap status;
    status.insert(QStringLiteral("installed"), m_scriptMaskHelper ? m_scriptMaskHelper->pythonReady() : false);
    status.insert(QStringLiteral("ready"), m_scriptSamReady);
    status.insert(QStringLiteral("running"), m_scriptMaskHelper ? m_scriptMaskHelper->jobRunning() : false);
    status.insert(QStringLiteral("mask"), m_scriptMaskFile);
    status.insert(QStringLiteral("error"), m_scriptMaskError);
    return status;
}

bool MainWindow::scriptSamGenerate(const QString &binId, const QString &maskName, const QString &includePoints,
                                   const QString &excludePoints, const QString &boxes, int zoneIn, int zoneOut)
{
    // Steps two and three: where to look, then follow it.
    //
    // Points and boxes arrive the way the mask panel stores them —
    // "frame=x,y;frame=x,y" — and are given to the helper directly rather than
    // through the monitor, which is what makes this usable without a window.
    if (!m_scriptMaskHelper || !m_scriptSamReady) {
        m_scriptMaskError = i18n("Prepare the frames first with sam_prepare.");
        return false;
    }
    if (includePoints.isEmpty() && boxes.isEmpty()) {
        m_scriptMaskError = i18n("Say what to follow: at least one point or box on the first frame.");
        return false;
    }
    bool ok = false;
    QDir frames = pCore->currentDoc()->getCacheDir(CacheMask, &ok);
    if (!ok) {
        return false;
    }
    if (!frames.cd(QStringLiteral("source-frames"))) {
        m_scriptMaskError = i18n("The exported frames are gone — prepare again.");
        return false;
    }
    m_scriptMaskError.clear();
    m_scriptMaskFile.clear();
    m_scriptMaskHelper->loadData(includePoints, excludePoints, boxes, zoneIn, frames);
    return m_scriptMaskHelper->generateMask(binId, maskName.isEmpty() ? i18n("mask") : maskName, QString(), QPoint(zoneIn, zoneOut));
}

QVariantList MainWindow::scriptListMasks(const QString &binId)
{
    QVariantList result;
    std::shared_ptr<ProjectClip> clip = pCore->projectItemModel()->getClipByBinID(binId);
    if (!clip) {
        return result;
    }
    const QVector<MaskInfo> masks = clip->masks();
    for (const MaskInfo &mask : masks) {
        QVariantMap entry;
        entry.insert(QStringLiteral("name"), mask.maskName);
        entry.insert(QStringLiteral("file"), mask.maskFile);
        entry.insert(QStringLiteral("in"), mask.in);
        entry.insert(QStringLiteral("out"), mask.out);
        result.append(entry);
    }
    return result;
}

bool MainWindow::scriptInstallPlugin(const QString &id)
{
    const PluginManifest manifest = PluginManager::instance().plugin(id);
    if (manifest.id().isEmpty() || manifest.isAgent()) {
        return false;
    }
    // Say no here rather than start something that cannot finish: the caller
    // reads the reason out of plugin_status["blocker"].
    if (!PluginManager::instance().installBlocker(manifest).isEmpty()) {
        return false;
    }
    PluginManager::instance().installPlugin(id);
    return true;
}

QVariantMap MainWindow::scriptPluginStatus(const QString &id)
{
    QVariantMap status;
    PluginManager &pm = PluginManager::instance();
    const PluginManifest m = pm.plugin(id);
    status.insert(QStringLiteral("installed"), m.isValid());
    QStringList missing;
    if (!m.isValid()) {
        missing << QStringLiteral("plugin");
        status.insert(QStringLiteral("missing"), missing);
        status.insert(QStringLiteral("ready"), false);
        return status;
    }
    status.insert(QStringLiteral("id"), m.id());
    status.insert(QStringLiteral("name"), m.name());
    status.insert(QStringLiteral("kind"), m.kind());
    status.insert(QStringLiteral("target"), m.target());

    // Why an install would refuse to start — the wrong application version, no
    // network, no room on the disk. Empty when it may go ahead. Reported as a
    // sentence because it is meant to be repeated to the user as one.
    status.insert(QStringLiteral("blocker"), pm.installBlocker(m));

    // API key (only api plugins with a provider need one)
    const QString provider = m.providerName();
    status.insert(QStringLiteral("provider"), provider);
    const bool keyNeeded = m.kind() == QLatin1String("api") && !provider.isEmpty();
    const bool keySet = keyNeeded ? !pm.apiKey(provider).isEmpty() : true;
    status.insert(QStringLiteral("api_key_set"), keySet);
    if (keyNeeded && !keySet) {
        missing << QStringLiteral("api_key");
    }

    // venv / deps — the dependency banner builds the venv and pip-installs in one
    // step, so an existing venv interpreter is a good proxy for "deps installed".
    const QString venvName = m.venvName();
    bool venvReady = true;
    if (!venvName.isEmpty() && m.hasDependencies()) {
        venvReady = !PluginManager::venvPython(venvName).isEmpty();
    }
    status.insert(QStringLiteral("venv"), venvName);
    status.insert(QStringLiteral("venv_ready"), venvReady);
    if (!venvReady) {
        missing << QStringLiteral("deps");
    }

    // models the plugin declares (downloaded on demand into its models/ dir)
    QVariantList models;
    const QString modelsDir = pm.modelsDir(m.id());
    const QList<PluginModel> declaredModels = m.models();
    for (const PluginModel &model : declaredModels) {
        QVariantMap mm;
        mm.insert(QStringLiteral("name"), model.name);
        const bool present = QFile::exists(modelsDir + QLatin1Char('/') + model.name);
        mm.insert(QStringLiteral("present"), present);
        models.append(mm);
        if (!present && !missing.contains(QStringLiteral("models"))) {
            missing << QStringLiteral("models");
        }
    }
    status.insert(QStringLiteral("models"), models);

    status.insert(QStringLiteral("missing"), missing);
    status.insert(QStringLiteral("ready"), missing.isEmpty());
    return status;
}

// ── Built-in face detection (FaceDataStore + FaceDetectTask, headless) ──

bool MainWindow::scriptSetFaceDetection(const QString &binId, bool enabled, int from, int to)
{
    if (binId.isEmpty() || !pCore->projectItemModel()->getClipByBinID(binId)) {
        return false;
    }
    FaceDataStore::instance().setEnabled(binId, enabled);
    const QString dataFolder = pCore->currentDoc() ? pCore->currentDoc()->projectDataFolder() : QString();
    if (enabled) {
        FaceDataStore::instance().ensureLoaded(binId, dataFolder);
        if (!FaceDataStore::instance().isComplete(binId)) {
            FaceDetectTask::start(binId, nullptr, from, to);
        }
    } else {
        FaceDataStore::instance().saveClip(binId, dataFolder);
    }
    if (m_projectMonitor) {
        m_projectMonitor->refreshFaceDetection();
    }
    return true;
}

QVariantMap MainWindow::scriptGetFaceDetectionStatus(const QString &binId)
{
    QVariantMap status;
    status[QStringLiteral("enabled")] = FaceDataStore::instance().isEnabled(binId);
    status[QStringLiteral("complete")] = FaceDataStore::instance().isComplete(binId);
    status[QStringLiteral("analysedFrames")] = FaceDataStore::instance().clipData(binId).size();
    return status;
}

QVariantList MainWindow::scriptGetFacesAtFrame(const QString &binId, int position)
{
    QVariantList faces;
    const QList<QRectF> rects = FaceDataStore::instance().interpolatedFrameData(binId, position);
    for (const QRectF &rect : rects) {
        QVariantMap face; // normalized 0..1 coordinates
        face[QStringLiteral("x")] = rect.x();
        face[QStringLiteral("y")] = rect.y();
        face[QStringLiteral("width")] = rect.width();
        face[QStringLiteral("height")] = rect.height();
        faces.append(face);
    }
    return faces;
}

bool MainWindow::scriptApplyFaceEffect(int clipId, int faceIndex, int atFrame, const QString &effectId, const QString &paramsJson)
{
    auto timeline = getCurrentTimeline();
    if (!timeline || !timeline->model()) return false;
    auto model = timeline->model();
    const QString binId = model->getClipBinId(clipId);
    if (binId.isEmpty()) return false;

    // Select the face: pick faceIndex among the faces detected at atFrame.
    const QList<QRectF> faces = FaceDataStore::instance().interpolatedFrameData(binId, atFrame);
    if (faceIndex < 0 || faceIndex >= faces.size()) return false;

    const int clipIn = model->getClipInOut(clipId).first;
    const QSize profile = pCore->getCurrentFrameSize();
    const QString results = FaceEffect::buildTrackerResults(binId, atFrame, clipIn, faces.at(faceIndex), profile);
    if (results.isEmpty()) return false;

    std::shared_ptr<EffectStackModel> stack = model->getClipEffectStackModel(clipId);
    if (!stack) return false;

    const QString effect = effectId.isEmpty() ? QStringLiteral("opencv.tracker") : effectId;
    stringMap params;
    if (effect == QLatin1String("opencv.tracker")) {
        // Hide-Face defaults (pixelate, strength auto-scaled to face size);
        // all overridable via paramsJson.
        params.insert(QStringLiteral("results"), results);
        params.insert(QStringLiteral("blur_type"), QStringLiteral("2"));
        params.insert(QStringLiteral("blur"), QString::number(FaceEffect::autoStrength(faces.at(faceIndex), profile)));
        params.insert(QStringLiteral("shape_width"), QStringLiteral("0"));
    }
    // User-chosen effect parameters override/extend the defaults.
    const QJsonObject obj = QJsonDocument::fromJson(paramsJson.toUtf8()).object();
    for (auto it = obj.constBegin(); it != obj.constEnd(); ++it) {
        params.insert(it.key(), it.value().isString() ? it.value().toString() : QString::number(it.value().toDouble()));
    }
    // A plugin's effect is applied with the region it works inside and the face
    // track filled in — the same thing the face menu does. Without that the
    // plugin has no idea where the face is and renders nothing.
    bool handled = false;
    const bool applied = applyPluginEffect(stack, effect, results, handled);
    if (handled) {
        if (applied) {
            setEffectParams(stack, effect, params);
        }
        return applied;
    }
    return stack->appendEffect(effect, true, params);
}
