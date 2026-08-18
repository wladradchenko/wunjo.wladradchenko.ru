/*
    SPDX-FileCopyrightText: 2017 Jean-Baptiste Mardelle
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include <KLocalizedQmlContext>

#include "../model/builders/meltBuilder.hpp"

// Required to pass the c++ classes to qml
#include "ai/facedatastore.h"
#include "bin/projectclip.h"
#include "bin/projectitemmodel.h"
#include "bin/model/markerlistmodel.hpp"
#include "bin/model/markersortmodel.h"
#include "bin/model/subtitlemodel.hpp"
#include "capture/mediacapture.h"

#include "core.h"
#include "doc/wunjodoc.h"
#include "effects/effectsrepository.hpp"
#include "wunjosettings.h"
#include "mainwindow.h"
#include "monitor/monitorproxy.h"
#include "monitormanager.h"
#include "plugins/plugineffects.h"
#include "plugins/pluginmanager.h"
#include "project/dialogs/guideslist.h"
#include "timelinewidget.h"

#include <KLocalizedString>

#include <QAction>
#include <QActionGroup>
#include <QFontDatabase>
#include <QJsonArray>
#include <QMenu>
#include <QQmlContext>
#include <QQmlEngine>
#include <QQuickItem>
#include <QSortFilterProxyModel>
#include <QTimer>
#include <QUuid>

const int TimelineWidget::comboScale[] = {1, 2, 4, 8, 15, 30, 50, 75, 100, 150, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 6000, 15000, 30000};

TimelineWidget::TimelineWidget(const QUuid uuid, QWidget *parent)
    : QQuickWidget(Core::sharedQmlEngine(), parent)
    , timelineController(this)
    , m_uuid(uuid)
{
    setClearColor(palette().window().color());
    m_sortModel = std::make_unique<QSortFilterProxyModel>(this);
    setResizeMode(QQuickWidget::SizeRootObjectToView);
    setVisible(false);
    setFont(QFontDatabase::systemFont(QFontDatabase::SmallestReadableFont));
    setFocusPolicy(Qt::StrongFocus);
    m_favEffects = new QMenu(i18n("Insert an effect..."), this);
    m_favCompositions = new QMenu(i18n("Insert a composition..."), this);
    installEventFilter(this);
    connect(&timelineController, &TimelineController::zoneMoved, this, &TimelineWidget::zoneMoved);
    connect(&timelineController, &TimelineController::ungrabHack, this, &TimelineWidget::slotUngrabHack);
    connect(&timelineController, &TimelineController::regainFocus, this, &TimelineWidget::regainFocus, Qt::DirectConnection);
    connect(&timelineController, &TimelineController::stopAudioRecord, this, &TimelineWidget::stopAudioRecord, Qt::DirectConnection);
    m_targetsMenu = new QMenu(this);
}

TimelineWidget::~TimelineWidget()
{
    rootObject()->blockSignals(true);
    timelineController.prepareClose();
    setSource(QUrl());
}

void TimelineWidget::updateEffectFavorites()
{
    const QMap<QString, QString> effects = sortedItems(WunjoSettings::favorite_effects(), false);
    QMapIterator<QString, QString> i(effects);
    m_favEffects->clear();
    while (i.hasNext()) {
        i.next();
        QAction *ac = m_favEffects->addAction(i.key());
        ac->setData(i.value());
    }
}

void TimelineWidget::updateTransitionFavorites()
{
    const QMap<QString, QString> effects = sortedItems(WunjoSettings::favorite_transitions(), true);
    QMapIterator<QString, QString> i(effects);
    m_favCompositions->clear();
    while (i.hasNext()) {
        i.next();
        QAction *ac = m_favCompositions->addAction(i.key());
        ac->setData(i.value());
    }
}

const QMap<QString, QString> TimelineWidget::sortedItems(const QStringList &items, bool isTransition)
{
    QMap<QString, QString> sortedItems;
    for (const QString &effect : items) {
        sortedItems.insert(timelineController.getAssetName(effect, isTransition), effect);
    }
    return sortedItems;
}

void TimelineWidget::setTimelineMenu(QMenu *clipMenu, QMenu *compositionMenu, QMenu *timelineMenu, QMenu *guideMenu, QMenu *timelineRulerMenu,
                                     QAction *editGuideAction, QMenu *headerMenu, QMenu *thumbsMenu, QMenu *subtitleClipMenu, QMenu *addClipMenu)
{
    m_timelineClipMenu = new QMenu(this);
    QList<QAction *> cActions = clipMenu->actions();
    for (auto &a : cActions) {
        m_timelineClipMenu->addAction(a);
    }
    m_timelineCompositionMenu = new QMenu(this);
    cActions = compositionMenu->actions();
    for (auto &a : cActions) {
        m_timelineCompositionMenu->addAction(a);
    }
    m_timelineMixMenu = new QMenu(this);
    QAction *deleteAction = pCore->window()->actionCollection()->action(QLatin1String("delete_timeline_clip"));
    m_timelineMixMenu->addAction(deleteAction);

    m_timelineMenu = new QMenu(this);
    cActions = timelineMenu->actions();
    for (auto &a : cActions) {
        m_timelineMenu->addAction(a);
    }
    m_timelineRulerMenu = new QMenu(this);
    cActions = timelineRulerMenu->actions();
    for (auto &a : cActions) {
        m_timelineRulerMenu->addAction(a);
    }
    m_guideMenu = guideMenu;
    m_headerMenu = headerMenu;
    m_thumbsMenu = thumbsMenu;
    m_headerMenu->addMenu(m_thumbsMenu);
    m_timelineSubtitleClipMenu = subtitleClipMenu;
    m_editGuideAcion = editGuideAction;
    m_addClipMenu = addClipMenu;
    updateEffectFavorites();
    updateTransitionFavorites();
    connect(m_favEffects, &QMenu::triggered, this, [&](QAction *ac) { timelineController.addEffectToClip(ac->data().toString()); });
    connect(m_favCompositions, &QMenu::triggered, this, [&](QAction *ac) { timelineController.addCompositionToClip(ac->data().toString()); });
    connect(m_guideMenu, &QMenu::triggered, this, [&](QAction *ac) { timelineController.setPosition(ac->data().toInt()); });
    connect(m_thumbsMenu, &QMenu::triggered, this,
            [&](QAction *ac) { timelineController.setActiveTrackProperty(QStringLiteral("wunjo:thumbs_format"), ac->data().toString()); });
    // Fix qml focus issue
    connect(m_headerMenu, &QMenu::aboutToHide, this, &TimelineWidget::slotUngrabHack, Qt::DirectConnection);
    connect(m_timelineClipMenu, &QMenu::aboutToHide, this, &TimelineWidget::slotUngrabHack, Qt::DirectConnection);
    connect(m_timelineClipMenu, &QMenu::triggered, this, &TimelineWidget::slotResetContextPos);
    connect(m_timelineCompositionMenu, &QMenu::aboutToHide, this, &TimelineWidget::slotUngrabHack, Qt::DirectConnection);
    connect(m_timelineRulerMenu, &QMenu::aboutToHide, this, &TimelineWidget::slotUngrabHack, Qt::DirectConnection);
    connect(m_timelineMenu, &QMenu::aboutToHide, this, &TimelineWidget::slotUngrabHack, Qt::DirectConnection);
    connect(m_timelineMenu, &QMenu::triggered, this, &TimelineWidget::slotResetContextPos);
    connect(m_timelineMenu, &QMenu::aboutToShow, this, &TimelineWidget::updateAddClipMenuStatus);
    connect(m_timelineMenu, &QMenu::triggered, this, &TimelineWidget::updateAddClipMenuStatus);
    connect(m_timelineSubtitleClipMenu, &QMenu::aboutToHide, this, &TimelineWidget::slotUngrabHack, Qt::DirectConnection);

    m_timelineClipMenu->addMenu(m_favEffects);
    m_timelineClipMenu->addMenu(m_favCompositions);
    m_timelineMenu->addMenu(m_favCompositions);
    m_timelineMenu->addMenu(m_addClipMenu);
}

const QUuid &TimelineWidget::getUuid() const
{
    return m_uuid;
}

void TimelineWidget::setModel(const std::shared_ptr<TimelineItemModel> &model, MonitorProxy *proxy)
{
    loading = true;
    Q_ASSERT(model != nullptr);
    connect(&timelineController, &TimelineController::timelineMouseOffsetChanged, this, &TimelineWidget::emitMousePos, Qt::QueuedConnection);
    m_sortModel->setSourceModel(model.get());
    m_sortModel->setSortRole(TimelineItemModel::SortRole);
    m_sortModel->sort(0, Qt::DescendingOrder);
    timelineController.setModel(model);
    setInitialProperties({{"controller", QVariant::fromValue(model.get())},
                          {"timeline", QVariant::fromValue(&timelineController)},
                          {"multitrack", QVariant::fromValue(m_sortModel.get())},
                          {"guidesModel", QVariant::fromValue(model->getFilteredGuideModel().get())},
                          {"proxy", QVariant::fromValue(pCore->monitorManager()->projectMonitor()->getControllerProxy())},
                          {"subtitleModel", QVariant::fromValue(model->getSubtitleModel().get())}});
    loadFromModule(QStringLiteral("online.wunjo.make"), QStringLiteral("Timeline"));

    connect(rootObject(), SIGNAL(zoomIn(bool)), pCore->window(), SLOT(slotZoomIn(bool)));
    connect(rootObject(), SIGNAL(zoomOut(bool)), pCore->window(), SLOT(slotZoomOut(bool)));
    connect(rootObject(), SIGNAL(processingDrag(bool)), pCore->window(), SIGNAL(enableUndo(bool)));
    connect(&timelineController, &TimelineController::seeked, proxy, &MonitorProxy::setPosition);
    connect(rootObject(), SIGNAL(showClipMenu(int)), this, SLOT(showClipMenu(int)));
    connect(rootObject(), SIGNAL(showMixMenu(int)), this, SLOT(showMixMenu(int)));
    connect(rootObject(), SIGNAL(showCompositionMenu()), this, SLOT(showCompositionMenu()));
    connect(rootObject(), SIGNAL(showTimelineMenu()), this, SLOT(showTimelineMenu()));
    connect(rootObject(), SIGNAL(showRulerMenu()), this, SLOT(showRulerMenu()));
    connect(rootObject(), SIGNAL(showHeaderMenu()), this, SLOT(showHeaderMenu()));
    connect(rootObject(), SIGNAL(showTargetMenu(int)), this, SLOT(showTargetMenu(int)));
    connect(rootObject(), SIGNAL(showSubtitleClipMenu()), this, SLOT(showSubtitleClipMenu()));
    connect(rootObject(), SIGNAL(markerActivated(int)), pCore->guidesList(), SLOT(markerActivated(int)));
    connect(rootObject(), SIGNAL(updateTimelineMousePos(int, int)), pCore->window(), SLOT(slotUpdateMousePosition(int, int)));
    timelineController.setRoot(rootObject());
    setVisible(true);
    loading = false;
    timelineController.checkDuration();
}

void TimelineWidget::emitMousePos(int offset)
{
    pCore->window()->slotUpdateMousePosition(int((offset + mapFromGlobal(QCursor::pos()).x()) / timelineController.scaleFactor()),
                                             timelineController.duration());
}

void TimelineWidget::mousePressEvent(QMouseEvent *event)
{
    Q_EMIT focusProjectMonitor();
    m_clickPos = event->globalPosition().toPoint();
    QQuickWidget::mousePressEvent(event);
}

void TimelineWidget::mouseMoveEvent(QMouseEvent *event)
{
    if (isEnabled()) {
        emitMousePos(timelineController.timelineMouseOffset());
    }
    QQuickWidget::mouseMoveEvent(event);
}

void TimelineWidget::showClipMenu(int cid)
{
    // Hide not applicable effects
    QList<QAction *> effects = m_favEffects->actions();
    int tid = model()->getClipTrackId(cid);
    bool isAudioTrack = false;
    if (tid > -1) {
        isAudioTrack = model()->isAudioTrack(tid);
    }
    m_favCompositions->setEnabled(!isAudioTrack);
    for (auto ac : std::as_const(effects)) {
        const QString &id = ac->data().toString();
        if (EffectsRepository::get()->isAudioEffect(id) != isAudioTrack) {
            ac->setVisible(false);
        } else {
            ac->setVisible(true);
        }
    }
    // AI submenu: video tools only on clips with video, audio tools only on
    // clips that carry sound
    const QString binId = model()->getClipBinId(cid);
    bool clipHasAudio = isAudioTrack;
    if (!clipHasAudio) {
        if (auto binClip = pCore->projectItemModel()->getClipByBinID(binId)) {
            clipHasAudio = binClip->hasAudioAndVideo() || binClip->clipType() == ClipType::Audio;
        }
    }
    const QList<QAction *> menuActions = m_timelineClipMenu->actions();
    for (QAction *action : menuActions) {
        if (action->objectName() == QLatin1String("ai_menu") && action->menu()) {
            QMenu *aiMenu = action->menu();
            const QList<QAction *> aiActions = aiMenu->actions();
            for (QAction *aiAction : aiActions) {
                if (aiAction->objectName() == QLatin1String("ai_speech_proxy")) {
                    aiAction->setVisible(clipHasAudio);
                    continue;
                }
                if (m_dynamicAiActions.contains(aiAction)) {
                    continue; // handled in the rebuild below
                }
                aiAction->setVisible(!isAudioTrack);
                if (aiAction->objectName() == QLatin1String("ai_detect_faces")) {
                    // Point the checkable property action at this clip, and at
                    // the part of it that is on the timeline: analysis is per
                    // bin clip, but a ten-minute source trimmed to five seconds
                    // must not spend minutes on footage that was cut out before
                    // it reaches the piece the user is looking at.
                    QSignalBlocker blocker(aiAction);
                    const int clipIn = model()->getClipIn(cid);
                    const int clipOut = clipIn + model()->getClipPlaytime(cid);
                    aiAction->setData(QVariantList{binId, clipIn, clipOut});
                    const bool faceDetectOn = FaceDataStore::instance().isEnabled(binId);
                    aiAction->setChecked(faceDetectOn);
                    aiAction->setText(faceDetectOn ? i18n("Disable Face Detection") : i18n("Enable Face Detection"));
                }
            }
            // Append user plugins that apply to this clip. Rebuilt every popup
            // so newly imported (or removed) plugins appear without a restart;
            // video plugins are hidden on audio-only clips and vice versa.
            qDeleteAll(m_dynamicAiActions);
            m_dynamicAiActions.clear();
            QList<PluginManifest> applicable;
            // A plugin that works on both the picture and the sound answers to
            // either list, so it would be offered twice on a clip that has both.
            QSet<QString> alreadyOffered;
            auto offer = [&applicable, &alreadyOffered](const PluginManifest &plugin) {
                if (!alreadyOffered.contains(plugin.id())) {
                    alreadyOffered.insert(plugin.id());
                    applicable << plugin;
                }
            };
            for (const PluginManifest &plugin : PluginManager::instance().pluginsForTarget(QStringLiteral("video"))) {
                if (!isAudioTrack) {
                    offer(plugin);
                }
            }
            for (const PluginManifest &plugin : PluginManager::instance().pluginsForTarget(QStringLiteral("audio"))) {
                if (clipHasAudio) {
                    offer(plugin);
                }
            }
            if (!applicable.isEmpty()) {
                // An audio effect belongs on the sound, and the sound of an A/V
                // clip is a clip of its own on an audio track: the video half
                // refuses audio effects outright, so picking one from the video
                // half did nothing at all. Follow the group across to the audio
                // side and hang it there instead.
                auto stackFor = [this](const QString &effectId, int clipId) {
                    if (!EffectsRepository::get()->isAudioEffect(effectId) || model()->isAudioTrack(model()->getClipTrackId(clipId))) {
                        return model()->getClipEffectStackModel(clipId);
                    }
                    const std::unordered_set<int> siblings = model()->getGroupElements(clipId);
                    for (int other : siblings) {
                        if (other != clipId && model()->isClip(other) && model()->isAudioTrack(model()->getClipTrackId(other))) {
                            return model()->getClipEffectStackModel(other);
                        }
                    }
                    return model()->getClipEffectStackModel(clipId);
                };
                m_dynamicAiActions << aiMenu->addSeparator();
                for (const PluginManifest &plugin : std::as_const(applicable)) {
                    // A plugin that brings an effect works through it: the entry
                    // drops that effect on the clip, where it is set up and
                    // keyframed. Only a plugin without one runs its script here.
                    const QList<PluginEffect> effects = PluginEffects::menuEffects(plugin);
                    if (!effects.isEmpty() && plugin.appliesEffectsTogether()) {
                        QAction *setAction = aiMenu->addAction(plugin.icon(), plugin.name());
                        connect(setAction, &QAction::triggered, this,
                                [this, plugin, cid]() { PluginEffects::applyAll(model()->getClipEffectStackModel(cid), plugin, QString()); });
                        m_dynamicAiActions << setAction;
                        continue;
                    }
                    if (!effects.isEmpty()) {
                        for (const PluginEffect &effect : effects) {
                            QAction *effectAction = aiMenu->addAction(plugin.icon(), effect.name);
                            connect(effectAction, &QAction::triggered, this, [this, plugin, effect, cid, stackFor]() {
                                // an effect that works inside a region brings that region along
                                PluginEffects::apply(stackFor(effect.id, cid), plugin, effect, QString());
                            });
                            m_dynamicAiActions << effectAction;
                        }
                        continue;
                    }
                    const QString pluginId = plugin.id();
                    const QString target = plugin.target();
                    QAction *pluginAction = aiMenu->addAction(plugin.icon(), plugin.name());
                    connect(pluginAction, &QAction::triggered, this, [this, pluginId, target, cid]() {
                        PluginManager::instance().runPlugin(pluginId, buildPluginClipInput(target, cid), this);
                    });
                    m_dynamicAiActions << pluginAction;
                }
            }
            break;
        }
    }
    m_timelineClipMenu->popup(m_clickPos);
}

QJsonObject TimelineWidget::buildPluginClipInput(const QString &target, int cid)
{
    // Operate on the whole selection when the clicked clip belongs to it,
    // otherwise just on the clicked clip — the convention users expect.
    std::unordered_set<int> ids = model()->getCurrentSelection();
    if (ids.find(cid) == ids.end()) {
        ids.clear();
        ids.insert(cid);
    }
    QJsonArray clips;
    for (int id : ids) {
        if (!model()->isClip(id)) {
            continue;
        }
        const int trackId = model()->getClipTrackId(id);
        const bool onAudioTrack = trackId > -1 && model()->isAudioTrack(trackId);
        const QString binId = model()->getClipBinId(id);
        std::shared_ptr<ProjectClip> binClip = pCore->projectItemModel()->getClipByBinID(binId);
        const bool hasAudio = onAudioTrack || (binClip && (binClip->hasAudioAndVideo() || binClip->clipType() == ClipType::Audio));
        if (target == QLatin1String("video") && onAudioTrack) {
            continue;
        }
        if (target == QLatin1String("audio") && !hasAudio) {
            continue;
        }
        QJsonObject clip;
        clip.insert(QStringLiteral("bin_id"), binId);
        if (binClip) {
            clip.insert(QStringLiteral("path"), binClip->url());
        }
        clip.insert(QStringLiteral("in"), 0);
        clip.insert(QStringLiteral("out"), model()->getClipPlaytime(id));
        clips.append(clip);
    }
    QJsonObject input;
    input.insert(QStringLiteral("clips"), clips);
    return input;
}

void TimelineWidget::showMixMenu(int /*cid*/)
{
    // Show mix menu
    m_timelineMixMenu->popup(m_clickPos);
}

void TimelineWidget::showCompositionMenu()
{
    m_timelineCompositionMenu->popup(m_clickPos);
}

void TimelineWidget::showHeaderMenu()
{
    bool isAudio = timelineController.isActiveTrackAudio();
    QList<QAction *> menuActions = m_headerMenu->actions();
    QList<QAction *> audioActions;
    QStringList allowedActions = {QLatin1String("show_track_record"), QLatin1String("separate_channels"), QLatin1String("normalize_channels")};
    for (QAction *ac : std::as_const(menuActions)) {
        if (allowedActions.contains(ac->data().toString())) {
            if (ac->data().toString() == QLatin1String("separate_channels")) {
                ac->setChecked(WunjoSettings::displayallchannels());
            }
            audioActions << ac;
        }
    }
    if (!isAudio) {
        // Video track
        int currentThumbs = timelineController.getActiveTrackProperty(QStringLiteral("wunjo:thumbs_format")).toInt();
        QList<QAction *> actions = m_thumbsMenu->actions();
        for (QAction *ac : std::as_const(actions)) {
            if (ac->data().toInt() == currentThumbs) {
                ac->setChecked(true);
                break;
            }
        }
        m_thumbsMenu->menuAction()->setVisible(true);
        for (auto ac : std::as_const(audioActions)) {
            ac->setVisible(false);
        }
    } else {
        // Audio track
        m_thumbsMenu->menuAction()->setVisible(false);
        for (auto ac : std::as_const(audioActions)) {
            ac->setVisible(true);
            if (ac->data().toString() == QLatin1String("show_track_record")) {
                ac->setChecked(timelineController.getActiveTrackProperty(QStringLiteral("wunjo:audio_rec")).toInt() == 1);
            }
        }
    }
    m_headerMenu->popup(m_clickPos);
}

void TimelineWidget::showTargetMenu(int tid)
{
    int currentTargetStream;
    if (tid == -1) {
        // Called through shortcut
        tid = timelineController.activeTrack();
        if (tid == -1) {
            return;
        }
        if (timelineController.clipTargets() < 2 || !model()->isAudioTrack(tid)) {
            pCore->displayMessage(i18n("No available stream"), MessageType::ErrorMessage);
            return;
        }
        QVariant returnedValue;
        QMetaObject::invokeMethod(rootObject(), "getActiveTrackStreamPos", Qt::DirectConnection, Q_RETURN_ARG(QVariant, returnedValue));
        m_clickPos = mapToGlobal(QPoint(5, y())) + QPoint(0, returnedValue.toInt());
    }
    QMap<int, QString> possibleTargets = timelineController.getCurrentTargets(tid, currentTargetStream);
    m_targetsMenu->clear();
    if (m_targetsGroup) {
        delete m_targetsGroup;
    }
    m_targetsGroup = new QActionGroup(this);
    QMapIterator<int, QString> i(possibleTargets);
    while (i.hasNext()) {
        i.next();
        QAction *ac = m_targetsMenu->addAction(i.value());
        ac->setData(i.key());
        m_targetsGroup->addAction(ac);
        ac->setCheckable(true);
        if (i.key() == currentTargetStream) {
            ac->setChecked(true);
        }
    }
    connect(m_targetsGroup, &QActionGroup::triggered, this, [this, tid](QAction *action) {
        int targetStream = action->data().toInt();
        timelineController.assignAudioTarget(tid, targetStream);
    });
    if (m_targetsMenu->isEmpty() || possibleTargets.isEmpty()) {
        m_headerMenu->popup(m_clickPos);
    } else {
        m_targetsMenu->popup(m_clickPos);
    }
}

void TimelineWidget::showRulerMenu()
{
    m_guideMenu->clear();
    const QList<CommentedTime> guides = pCore->currentDoc()->getGuideModel(m_uuid)->getAllMarkers();
    m_editGuideAcion->setEnabled(false);
    double fps = pCore->getCurrentFps();
    int currentPos = rootObject()->property("consumerPosition").toInt();
    for (const auto &guide : guides) {
        auto *ac = new QAction(guide.comment(), this);
        int frame = guide.time().frames(fps);
        ac->setData(frame);
        if (frame == currentPos) {
            m_editGuideAcion->setEnabled(true);
        }
        m_guideMenu->addAction(ac);
    }
    m_timelineRulerMenu->popup(m_clickPos);
}

void TimelineWidget::showTimelineMenu()
{
    m_guideMenu->clear();
    const QList<CommentedTime> guides = pCore->currentDoc()->getGuideModel(m_uuid)->getAllMarkers();
    m_editGuideAcion->setEnabled(false);
    double fps = pCore->getCurrentFps();
    int currentPos = rootObject()->property("consumerPosition").toInt();
    for (const auto &guide : guides) {
        auto ac = new QAction(guide.comment(), this);
        int frame = guide.time().frames(fps);
        ac->setData(frame);
        if (frame == currentPos) {
            m_editGuideAcion->setEnabled(true);
        }
        m_guideMenu->addAction(ac);
    }
    m_addMenuConnection = connect(m_addClipMenu, &QMenu::aboutToShow, this, [this]() {
        QPoint posInWidget = mapFromGlobal(m_clickPos);
        int addClipFrame = timelineController.getMousePos(posInWidget);
        int addClipTrack = timelineController.getMouseTrack(posInWidget);
        // Calculate maximum available space on this track
        int maxSpace = timelineController.getFreeSpace(addClipTrack, addClipFrame);
        pCore->bin()->setSuggestedDuration(maxSpace);
        pCore->bin()->setReadyCallBack([this, addClipTrack, addClipFrame](const QString &clipId) {
            qDebug() << "CALLBACK TRIGGERED FOR CLIP:" << clipId;
            // Process with insertion
            timelineController.insertClips(addClipTrack, addClipFrame, QStringList(clipId), true, true);
        });
        QObject::disconnect(m_addMenuConnection);
    });
    m_timelineMenu->popup(m_clickPos);
}

void TimelineWidget::showSubtitleClipMenu()
{
    m_timelineSubtitleClipMenu->popup(m_clickPos);
}

void TimelineWidget::updateAddClipMenuStatus()
{
    int tid = timelineController.getMouseTrack();
    if (tid == -2 || tid == -1 || !model()->isTrack(tid) || model()->isAudioTrack(tid)) {
        m_addClipMenu->setEnabled(false);
    } else {
        m_addClipMenu->setEnabled(true);
    }
}

void TimelineWidget::slotChangeZoom(int value, bool zoomOnMouse)
{
    double pixelScale = QFontMetrics(font()).maxWidth() * 2;
    timelineController.setScaleFactorOnMouse(pixelScale / comboScale[value], zoomOnMouse);
}

void TimelineWidget::slotCenterView()
{
    QMetaObject::invokeMethod(rootObject(), "centerViewOnCursor");
}

void TimelineWidget::slotFitZoom()
{
    QVariant returnedValue;
    double prevScale = timelineController.scaleFactor();
    QMetaObject::invokeMethod(rootObject(), "fitZoom", Qt::DirectConnection, Q_RETURN_ARG(QVariant, returnedValue));
    double scale = returnedValue.toDouble();
    QMetaObject::invokeMethod(rootObject(), "scrollPos", Qt::DirectConnection, Q_RETURN_ARG(QVariant, returnedValue));
    int scrollPos = returnedValue.toInt();
    if (qFuzzyCompare(prevScale, scale) && scrollPos == 0) {
        scale = m_prevScale;
        scrollPos = m_scrollPos;
    } else {
        m_prevScale = prevScale;
        m_scrollPos = scrollPos;
        scrollPos = 0;
    }
    timelineController.setScaleFactorOnMouse(scale, false);
    // Update zoom slider
    Q_EMIT timelineController.updateZoom(scale);
    QMetaObject::invokeMethod(rootObject(), "goToStart", Q_ARG(QVariant, scrollPos));
}

Mlt::Tractor *TimelineWidget::tractor()
{
    return timelineController.tractor();
}

TimelineController *TimelineWidget::controller()
{
    return &timelineController;
}

std::shared_ptr<TimelineItemModel> TimelineWidget::model()
{
    return timelineController.getModel();
}

void TimelineWidget::zoneUpdated(const QPoint &zone)
{
    timelineController.setZone(zone, false);
}

void TimelineWidget::zoneUpdatedWithUndo(const QPoint &oldZone, const QPoint &newZone)
{
    timelineController.updateZone(oldZone, newZone);
}

QPair<int, int> TimelineWidget::getAvTracksCount() const
{
    return timelineController.getAvTracksCount();
}

void TimelineWidget::slotUngrabHack()
{
    // Workaround bug: https://bugreports.qt.io/browse/QTBUG-59044
    // https://phabricator.kde.org/D5515
    QTimer::singleShot(250, this, [this]() {
        // Reset menu position, necessary if user closes the menu without selecting any action
        rootObject()->setProperty("clickFrame", -1);
        QObject::disconnect(m_addMenuConnection);
    });
    if (quickWindow()) {
        if (quickWindow()->mouseGrabberItem()) {
            quickWindow()->mouseGrabberItem()->ungrabMouse();
            QPoint mousePos = mapFromGlobal(QCursor::pos());
            QMetaObject::invokeMethod(rootObject(), "regainFocus", Qt::DirectConnection, Q_ARG(QVariant, mousePos));
        } else {
            QMetaObject::invokeMethod(rootObject(), "endDrag", Qt::DirectConnection);
        }
    }
}

void TimelineWidget::slotResetContextPos(QAction *)
{
    rootObject()->setProperty("clickFrame", -1);
    m_clickPos = QPoint();
}

int TimelineWidget::zoomForScale(double value) const
{
    int scale = int(100 / value);
    int ix = 13;
    while (comboScale[ix] > scale && ix > 0) {
        ix--;
    }
    return ix;
}

void TimelineWidget::focusTimeline()
{
    setFocus();
    if (rootObject()) {
        rootObject()->setFocus(true);
    }
}

void TimelineWidget::endDrag()
{
    if (rootObject()) {
        QMetaObject::invokeMethod(rootObject(), "endBinDrag");
    }
}

void TimelineWidget::startAudioRecord(int tid)
{
    if (rootObject()) {
        QMetaObject::invokeMethod(rootObject(), "startAudioRecord", Qt::DirectConnection, Q_ARG(QVariant, tid));
    }
}

void TimelineWidget::stopAudioRecord()
{
    if (rootObject()) {
        QMetaObject::invokeMethod(rootObject(), "stopAudioRecord", Qt::DirectConnection);
    }
}

void TimelineWidget::regainFocus()
{
    if (underMouse() && rootObject()) {
        QPoint mousePos = mapFromGlobal(QCursor::pos());
        QMetaObject::invokeMethod(rootObject(), "regainFocus", Qt::DirectConnection, Q_ARG(QVariant, mousePos));
    }
}

bool TimelineWidget::hasSubtitles() const
{
    return timelineController.getModel()->hasSubtitleModel();
}

void TimelineWidget::connectSubtitleModel(bool firstConnect)
{
    qDebug() << "root context get sub model new function";
    if (!model()->hasSubtitleModel()) {
        return;
    }

    if (firstConnect) {
        rootObject()->setProperty("subtitleModel", QVariant::fromValue(model()->getSubtitleModel().get()));
        QQmlEngine::setObjectOwnership(model()->getSubtitleModel().get(), QQmlEngine::CppOwnership);
    }
}
