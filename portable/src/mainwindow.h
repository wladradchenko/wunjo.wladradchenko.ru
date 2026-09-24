/*
    SPDX-FileCopyrightText: 2007 Jean-Baptiste Mardelle <jb@kdenlive.org>

SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QComboBox>
#include <QDockWidget>
#include <QEvent>
#include <QImage>
#include <QMap>
#include <QProcessEnvironment>
#include <QProgressDialog>
#include <QShortcut>
#include <QString>
#include <QUndoView>
#include <QUuid>

#include <KActionCategory>
#include <KAutoSaveFile>
#include <KColorSchemeManager>
#include <KIO/Global>
#include <KSelectAction>
#include <KXmlGuiWindow>
#include <kddockwidgets/DockWidget.h>
#include <kddockwidgets/MainWindow.h>
#include <kddockwidgets/core/Layout.h>

#include <mlt++/Mlt.h>
#include <utility>

#include "bin/bin.h"
#include "definitions.h"
#include "jobs/abstracttask.h"
#include "otio/otioexport.h"
#include "otio/otioimport.h"
#include "powermanagementinterface.h"
#include "statusbarmessagelabel.h"

class ScriptingServer;
class AssetPanel;
class AudioGraphSpectrum;
class AutomaskHelper;
class ChatWidget;
class EffectBasket;
class EffectListWidget;
class TransitionListWidget;
class KIconLoader;
class WunjoDoc;
class Monitor;
class Render;
class RenderWidget;
class ScopeManager;
class TimelineTabs;
class TimelineWidget;
class TimelineContainer;
class Transition;
class TimelineItemModel;
class MonitorProxy;
class KDualAction;

class MltErrorEvent : public QEvent
{
public:
    explicit MltErrorEvent(QString message)
        : QEvent(QEvent::User)
        , m_message(std::move(message))
    {
    }

    QString message() const { return m_message; }

private:
    QString m_message;
};

class MainWindow : public KXmlGuiWindow
{
    Q_OBJECT

public:
    friend class RenderWidget;
    friend class Monitor;
    friend class WunjoSettingsDialog;

    explicit MainWindow(QWidget *parent = nullptr);
    /** @brief Initialises the main window.
     * @param MltPath (optional) path to MLT environment
     * @param Url (optional) file to open
     * @param clipsToLoad (optional) a comma separated list of clips to import in project
     *
     * If Url is present, it will be opened, otherwise, if openlastproject is
     * set, latest project will be opened. If no file is open after trying this,
     * a default new file will be created. */
    void init();
    ~MainWindow() override;

    /** @brief The socket this window's scriptable methods are served on, or
     *  empty when the server did not start. Every process that has to talk back
     *  to *this* copy of the editor — an assistant plugin above all — is given
     *  it rather than left to look for it, which cannot be done at all on
     *  Windows and finds the wrong copy when two are open. */
    QString scriptingSocketName() const;

    /** @brief Cache for luma files thumbnails. */
    static QMap<QString, QImage> m_lumacache;
    static QMap<QString, QStringList> m_lumaFiles;

    /** @brief Adds an action to the action collection and stores the name. */
    void addAction(const QString &name, QAction *action, const QKeySequence &shortcut = QKeySequence(), KActionCategory *category = nullptr);
    /** @brief Same as above, but takes a string for category to populate it with wunjoCategoryMap */
    void addAction(const QString &name, QAction *action, const QKeySequence &shortcut, const QString &category);
    /** @brief Adds an action to the action collection and stores the name. */
    QAction *addAction(const QString &name, const QString &text, const QObject *receiver, const char *member, const QIcon &icon = QIcon(),
                       const QKeySequence &shortcut = QKeySequence(), KActionCategory *category = nullptr);
    /** @brief Same as above, but takes a string for category to populate it with wunjoCategoryMap */
    QAction *addAction(const QString &name, const QString &text, const QObject *receiver, const char *member, const QIcon &icon, const QKeySequence &shortcut,
                       const QString &category);

    void processRestoreState(const QByteArray &state);

    /**
     * @brief Adds a new dock widget to this window.
     * @param title title of the dock widget
     * @param objectName objectName of the dock widget (required for storing layouts)
     * @param widget widget to use in the dock
     * @param area area to which the dock should be added to
     * @param otherDockWidget if any, the widget that will be used to get the relative area
     * @returns the created dock widget
     */
    KDDockWidgets::QtWidgets::DockWidget *addDock(const QString &title, const QString &objectName, QWidget *widget,
                                                  KDDockWidgets::Location area = KDDockWidgets::Location_OnRight,
                                                  KDDockWidgets::QtWidgets::DockWidget *otherDockWidget = nullptr, const QSize preferredSize = QSize());

    QUndoGroup *m_commandStack{nullptr};
    QUndoView *m_undoView;
    /** @brief holds info about whether movit is available on this system */
    bool m_gpuAllowed;
    int m_exitCode{EXIT_SUCCESS};
    QMap<QString, KActionCategory *> wunjoCategoryMap;
    QList<QAction *> getExtraActions(const QString &name);

    /** @brief Returns true if mixer widget is tabbed */
    bool isMixedTabbed() const;

    /** @brief Returns a pointer to the current timeline */
    TimelineWidget *getCurrentTimeline() const;
    /** @brief Returns a pointer to the timeline with @uuid */
    TimelineWidget *getTimeline(const QUuid uuid) const;
    void getSequenceProperties(const QUuid &uuid, QMap<QString, QString> &props);
    void closeTimelineTab(const QUuid uuid, bool onDeletion, bool checkActiveClosed=false);
    /** @brief Returns a list of opened tabs uuids */
    const QStringList openedSequences() const;

    /** @brief Returns true if a timeline widget is available */
    bool hasTimeline() const;

    /** @brief Returns true if the timeline widget is visible */
    bool timelineVisible() const;

    /** @brief Raise (show) the clip or project monitor */
    void raiseMonitor(bool clipMonitor, bool raise = false);

    /** @brief Raise (show) the project bin
     * @param unconditionally if false, we won't raise the bin if docked with the project monitor */
    void raiseBin(bool unconditionally = true);
    /** @brief Give focus to the active timeline widget */
    void focusTimeline();
    /** @brief Add a bin widget*/
    void addBin(Bin *bin, const QString &binName = QString(), bool updateCount = true, const QString &objectName = QString());
    /** @brief Clean current document references from all bins*/
    void cleanBins();
    /** @brief Get the main (first) bin*/
    Bin *getBin();
    /** @brief Block/Unblock all bin selection signals*/
    void blockBins(bool block);
    /** @brief Get the active (focused) bin or first one if none is active*/
    Bin *activeBin();
    int binCount() const;
    void loadBins(QStringList binInfo);

    ToolType::ProjectTool activeTool();

    /** @brief Hide subtitle track and delete its temporary file*/
    void resetSubtitles(const QUuid &uuid);

    /** @brief Show current tool key combination in status bar */
    void showToolMessage();
    /** @brief Show the widget's default key binding message */
    void setWidgetKeyBinding(const QString &text = QString());
    /** @brief Show a key binding in status bar */
    void showKeyBinding(const QString &text = QString());
    /** @brief Disable multicam mode if it was active */
    void disableMulticam();
    /** @brief Get the folder id for project bins */
    const QStringList extraBinIds() const;
    /** @brief Load the project bins of a project */
    void loadExtraBins(const QStringList binInfo);
    void folderRenamed(const QString &binId, const QString &folderName);
    /** @brief Seek timeline if it is active */
    void seekIfCurrent(const QUuid uuid, int pos);

    /** @brief Check if the maximum cached data size is not exceeded. */
    void checkMaxCacheSize();
    /** @brief Ask the release feed whether a newer version exists and, if so,
     *  offer the download page.
     *
     *  Wunjo Make is installed from a file, so nothing updates it on the user's
     *  behalf: without this, a 3.0.0 stays 3.0.0 forever. Silent about
     *  everything except a genuinely newer release — no dialog on startup, no
     *  message when the check fails, and nothing at all once the user has
     *  switched it off in Settings. Runs at most once a week. */
    void checkForNewVersion();
    TimelineWidget *openTimeline(const QUuid &uuid, int ix, const QString &tabName, std::shared_ptr<TimelineItemModel> timelineModel,
                                 bool openInMonitor = true);
    /** @brief Bring a timeline tab in front. Returns false if no tab exists for this timeline. */
    bool raiseTimeline(const QUuid &uuid);
    void connectTimeline();
    void disconnectTimeline(TimelineWidget *timeline, bool onClose = false);
    static QProcessEnvironment getCleanEnvironement();
    ObjectId effectStackOwner();
    bool effectIsMasterOnly(const QString &assetId) const;
    void reloadAssetPanel();
    /** @brief If any task is running, ask user before closing */
    bool hasRunningTask() const;
    /** @brief If a render task is running */
    bool hasRunningRenderTask() const;

protected:
    /** @brief Closes the window.
     * @return false if the user presses "Cancel" on a confirmation dialog or
     *     the operation requested (starting waiting jobs or saving file) fails,
     *     true otherwise */
    bool queryClose() override;
    bool m_windowClosing{false};
    void closeEvent(QCloseEvent *) override;
    QSize sizeHint() const override;
    bool eventFilter(QObject *object, QEvent *event) override;

    /** @brief Reports a message in the status bar when an error occurs. */
    void customEvent(QEvent *e) override;

    /** @brief Stops the active monitor when the window gets hidden. */
    void hideEvent(QHideEvent *e) override;

    /** @brief Saves the file and the window properties when saving the session. */
    void saveProperties(KConfigGroup &config) override;

    void saveNewToolbarConfig() override;
    /** @brief Power management to inhibit sleep while rendering */
    PowerManagementInterface mPowerInterface;
    Wunjo::ConfigPage m_lastConfigPage = Wunjo::NoPage;

private:
    /** @brief Serves this window's Q_SCRIPTABLE slots over a local socket —
     *  what the assistant and the MCP server call. Replaces the D-Bus object
     *  this class used to register. */
    ScriptingServer *m_scriptingServer{nullptr};

    /** @brief Sets up all the actions and attaches them to the collection. */
    void setupActions();
    /** @brief Rebuild the dock menu according to existing dock widgets. */
    void updateDockMenu();
    /** @brief Update the audio thumbnails action icon based on current zoom and toggle state */
    void updateAudioWaveformActionIcon();

    /** @brief The mask session a script is driving, made on first use. */
    AutomaskHelper *scriptMaskHelper();
    /** @brief Kept apart from the one the mask panel uses so the two cannot
     *  tread on each other's points. */
    AutomaskHelper *m_scriptMaskHelper{nullptr};
    /** @brief Where the last scripted mask landed, and what went wrong if it
     *  did not: scriptSamStatus has nothing else to report from. */
    QString m_scriptMaskFile;
    QString m_scriptMaskError;
    bool m_scriptSamReady{false};

    OtioExport *m_otioExport{nullptr};
    OtioImport *m_otioImport{nullptr};
    KColorSchemeManager *m_colorschemes;
    ScopeManager *m_scopesManager{nullptr};
    KDDockWidgets::QtWidgets::MainWindow *mainDockWindow;
    KDDockWidgets::QtWidgets::DockWidget *m_timelineDock{nullptr};
    KDDockWidgets::QtWidgets::DockWidget *m_projectBinDock{nullptr};
    KDDockWidgets::QtWidgets::DockWidget *m_effectListDock{nullptr};
    KDDockWidgets::QtWidgets::DockWidget *m_compositionListDock{nullptr};
    TransitionListWidget *m_compositionList{nullptr};
    EffectListWidget *m_effectList2{nullptr};

    AssetPanel *m_assetPanel{nullptr};
    KDDockWidgets::QtWidgets::DockWidget *m_effectStackDock{nullptr};

    KDDockWidgets::QtWidgets::DockWidget *m_clipMonitorDock{nullptr};
    Monitor *m_clipMonitor{nullptr};

    KDDockWidgets::QtWidgets::DockWidget *m_projectMonitorDock{nullptr};
    Monitor *m_projectMonitor{nullptr};

    AudioGraphSpectrum *m_audioSpectrum{nullptr};

    KDDockWidgets::QtWidgets::DockWidget *m_undoViewDock{nullptr};
    KDDockWidgets::QtWidgets::DockWidget *m_chatDock{nullptr};
    ChatWidget *m_chatWidget{nullptr};
    KDDockWidgets::QtWidgets::DockWidget *m_mixerDock{nullptr};
    KDDockWidgets::QtWidgets::DockWidget *m_onlineResourcesDock{nullptr};

    KSelectAction *m_timeFormatButton;
    QAction *m_compositeAction;

    // Tool message styling state tracking
    TimelineMode::EditMode m_currentEditMode{TimelineMode::NormalEdit};

    TimelineTabs *m_timelineTabs{nullptr};
    QVector<Bin *> m_binWidgets;

    KActionCategory *m_effectActions;
    KActionCategory *m_transitionActions;
    QMenu *m_effectsMenu{nullptr};
    QMenu *m_transitionsMenu{nullptr};
    QMenu *m_timelineContextMenu{nullptr};
    QMenu *m_binsListMenu{nullptr};
    QMenu *m_scopesListMenu{nullptr};
    QList<QAction *> m_timelineClipActions;
    KDualAction *m_useTimelineZone{nullptr};

    /** Action names that can be used in the slotDoAction() slot, with their i18n() names */
    QStringList m_actionNames;

    /** @brief Shortcut to remove the focus from any element.
     *
     * It allows one to get out of e.g. text input fields and to press another
     * shortcut. */
    QShortcut *m_shortcutRemoveFocus{nullptr};

    RenderWidget *m_renderWidget{nullptr};
    StatusBarMessageLabel *m_messageLabel{nullptr};
    QList<QAction *> m_transitions;
    QAction *m_buttonAudioThumbs;
    QAction *m_buttonVideoThumbs;
    QPushButton *m_statusZoomLevelButton;
    QMenu *m_audioThumbsMenu;
    QAction *m_buttonShowMarkers;
    QAction *m_buttonFitZoom;
    QAction *m_buttonTimelineTags;
    QAction *m_buttonMouseZoomOnPlayhead;
    QAction *m_normalEditTool;
    QAction *m_overwriteEditTool;
    QAction *m_insertEditTool;
    QAction *m_buttonSelectTool;
    QAction *m_buttonRazorTool;
    QAction *m_buttonSpacerTool;
    QAction *m_buttonRippleTool;
    QAction *m_buttonRollTool;
    QAction *m_buttonSlipTool;
    QAction *m_buttonSlideTool;
    QAction *m_buttonMulticamTool;
    QAction *m_buttonSnap;
    QAction *m_buttonHideClipOverlays;
    QAction *m_saveAction;
    QSlider *m_zoomSlider;
    QAction *m_zoomIn;
    QAction *m_zoomOut;
    QAction *m_audioZoomIn;
    QAction *m_audioZoomOut;
    QAction *m_audioZoomReset;
    QAction *m_audioZoomCycle;
    QAction *m_loopZone;
    QAction *m_playZone;
    QAction *m_playZoneFromCursor;
    QAction *m_loopClip;
    QAction *m_proxyClip;
    QAction *m_buttonSubtitleEditTool;
    QString m_theme;
    KIconLoader *m_iconLoader;
    KToolBar *m_timelineToolBar;
    TimelineContainer *m_timelineToolBarContainer;
    QLabel *m_trimLabel;
    QActionGroup *m_scaleGroup;
    ToolType::ProjectTool m_activeTool;
    /** @brief Store latest mouse position in timeline. */
    int m_mousePosition;

    KHamburgerMenu *m_hamburgerMenu;

    /** @brief initialize startup values, return true if first run. */
    bool readOptions();
    void saveOptions();

    QStringList m_pluginFileNames;
    void buildDynamicActions();
    void loadClipActions();
    void loadContainerActions();

    QTime m_timer;
    KXMLGUIClient *m_extraFactory;
    bool m_themeInitialized{false};
    bool m_isDarkTheme{false};
    EffectBasket *m_effectBasket{nullptr};
    QProgressDialog *m_loadingDialog{nullptr};

    // ── Scripting API (D-Bus: online.wunjo.make.MainWindow) ──────────────
public Q_SLOTS:
    // Project Management
    Q_SCRIPTABLE QString scriptNewProject(const QString &name);
    Q_SCRIPTABLE bool scriptOpenProject(const QString &filePath);
    Q_SCRIPTABLE bool scriptSaveProject();
    Q_SCRIPTABLE bool scriptSaveProjectAs(const QString &filePath);
    Q_SCRIPTABLE QString scriptGetProjectName();
    Q_SCRIPTABLE QString scriptGetProjectPath();
    Q_SCRIPTABLE double scriptGetProjectFps();
    Q_SCRIPTABLE int scriptGetProjectResolutionWidth();
    Q_SCRIPTABLE int scriptGetProjectResolutionHeight();
    Q_SCRIPTABLE QString scriptGetProjectProperty(const QString &key);
    Q_SCRIPTABLE bool scriptSetProjectProperty(const QString &key, const QString &value);
    Q_SCRIPTABLE int scriptGetProjectDuration();
    Q_SCRIPTABLE QString scriptGetProjectColorSpace();
    Q_SCRIPTABLE bool scriptSetProjectColorSpace(const QString &colorSpace);
    Q_SCRIPTABLE int scriptGetProjectAudioSampleRate();

    // Media Pool (Bin)
    Q_SCRIPTABLE QStringList scriptImportMedia(const QStringList &filePaths, const QString &folderId = QStringLiteral("-1"));
    Q_SCRIPTABLE QString scriptCreateFolder(const QString &name, const QString &parentId = QStringLiteral("-1"));
    Q_SCRIPTABLE QStringList scriptGetAllClipIds();
    Q_SCRIPTABLE QStringList scriptGetFolderClipIds(const QString &folderId);
    Q_SCRIPTABLE QVariantMap scriptGetClipProperties(const QString &binId);
    Q_SCRIPTABLE bool scriptDeleteBinClip(const QString &binId);
    Q_SCRIPTABLE QString scriptCreateTitleClip(const QString &titleXml, int durationFrames, const QString &clipName = QStringLiteral("Title clip"),
                                               const QString &parentFolderId = QStringLiteral("-1"));
    Q_SCRIPTABLE QString scriptGetTitleXml(const QString &binId);
    Q_SCRIPTABLE bool scriptSetTitleXml(const QString &binId, const QString &newXml);
    Q_SCRIPTABLE bool scriptRenameBinClip(const QString &binId, const QString &newName);
    Q_SCRIPTABLE bool scriptMoveBinClip(const QString &binId, const QString &targetFolderId);
    Q_SCRIPTABLE QVariantMap scriptGetClipMetadata(const QString &binId);

    // Timeline
    Q_SCRIPTABLE int scriptGetTrackCount(const QString &trackType);
    Q_SCRIPTABLE QVariantMap scriptGetTrackInfo(int trackIndex);
    Q_SCRIPTABLE QVariantList scriptGetAllTracksInfo();
    Q_SCRIPTABLE int scriptAddTrack(const QString &name, bool audioTrack, int position = -1);
    Q_SCRIPTABLE bool scriptDeleteTrack(int trackId);
    Q_SCRIPTABLE bool scriptInsertSpace(int trackId, int position, int duration, bool allTracks);
    Q_SCRIPTABLE bool scriptRemoveSpace(int trackId, int position, bool allTracks);
    Q_SCRIPTABLE int scriptInsertClip(const QString &binClipId, int trackId, int position);
    Q_SCRIPTABLE QVariantList scriptInsertClipsSequentially(const QStringList &binClipIds, int trackId, int startPosition);
    Q_SCRIPTABLE bool scriptMoveClip(int clipId, int trackId, int position);
    Q_SCRIPTABLE int scriptResizeClip(int clipId, int newDuration, bool fromRight);
    Q_SCRIPTABLE bool scriptDeleteTimelineClip(int clipId);
    Q_SCRIPTABLE QVariantList scriptGetClipsOnTrack(int trackId);
    Q_SCRIPTABLE QVariantMap scriptGetTimelineClipInfo(int clipId);
    Q_SCRIPTABLE bool scriptSlipClip(int clipId, int offset);
    Q_SCRIPTABLE bool scriptCutClip(int clipId, int position);

    // Ripple/Roll/Slide Editing
    Q_SCRIPTABLE bool scriptRippleDelete(int clipId);
    Q_SCRIPTABLE bool scriptRippleTrim(int clipId, int delta, bool fromRight);
    Q_SCRIPTABLE bool scriptRollEdit(int clipId, int delta);
    Q_SCRIPTABLE bool scriptSlideEdit(int clipId, int delta);

    // Transitions & Mixes
    Q_SCRIPTABLE bool scriptAddMix(int clipIdA, int clipIdB, int durationFrames);
    Q_SCRIPTABLE int scriptAddComposition(const QString &transitionId, int trackId, int position, int duration);
    Q_SCRIPTABLE bool scriptRemoveMix(int clipId);
    Q_SCRIPTABLE QVariantList scriptGetAvailableTransitions();
    Q_SCRIPTABLE QVariantMap scriptGetMixParams(int clipId);
    Q_SCRIPTABLE bool scriptSetMixDuration(int clipId, int newDuration);

    // Compositions
    Q_SCRIPTABLE QVariantList scriptGetCompositions();
    Q_SCRIPTABLE QVariantMap scriptGetCompositionInfo(int compoId);
    Q_SCRIPTABLE bool scriptMoveComposition(int compoId, int trackId, int position);
    Q_SCRIPTABLE int scriptResizeComposition(int compoId, int newDuration, bool fromRight);
    Q_SCRIPTABLE bool scriptDeleteComposition(int compoId);
    Q_SCRIPTABLE QVariantList scriptGetCompositionTypes();
    Q_SCRIPTABLE bool scriptSetCompositionParam(int compoId, const QString &paramName, const QString &paramValue);
    Q_SCRIPTABLE QString scriptGetCompositionParam(int compoId, const QString &paramName);

    // Effects
    Q_SCRIPTABLE QVariantList scriptGetAvailableEffects();
    Q_SCRIPTABLE bool scriptAddClipEffect(int clipId, const QString &effectId, const QStringList &paramKeys, const QStringList &paramValues);
    Q_SCRIPTABLE bool scriptRemoveClipEffect(int clipId, const QString &effectId);
    Q_SCRIPTABLE QString scriptGetClipEffects(int clipId);
    Q_SCRIPTABLE bool scriptSetEffectParam(int clipId, const QString &effectId, const QString &paramName, const QString &paramValue);
    Q_SCRIPTABLE QString scriptGetEffectParam(int clipId, const QString &effectId, const QString &paramName);
    Q_SCRIPTABLE bool scriptSetEffectExpression(int clipId, const QString &effectId, const QString &paramName, const QString &expression, double baseValue);
    Q_SCRIPTABLE bool scriptClearEffectExpression(int clipId, const QString &effectId, const QString &paramName);
    Q_SCRIPTABLE QString scriptCopyClipEffects(int clipId);
    Q_SCRIPTABLE bool scriptPasteClipEffects(int targetClipId, const QString &effectsXml);

    // Speed
    Q_SCRIPTABLE bool scriptSetClipSpeed(int clipId, double speed, bool pitchCompensate);

    // Effect Keyframes
    Q_SCRIPTABLE QVariantList scriptGetEffectKeyframes(int clipId, int effectIndex);
    Q_SCRIPTABLE bool scriptAddEffectKeyframe(int clipId, int effectIndex, int frame, double normalizedValue, int keyframeType);
    Q_SCRIPTABLE bool scriptRemoveEffectKeyframe(int clipId, int effectIndex, int frame);
    Q_SCRIPTABLE bool scriptUpdateEffectKeyframe(int clipId, int effectIndex, int oldFrame, int newFrame, double normalizedValue);

    // Effect keyframes by parameter name
    Q_SCRIPTABLE QVariantList scriptGetEffectKeyframesByParam(int clipId, const QString &effectId, const QString &paramName);
    Q_SCRIPTABLE bool scriptAddEffectKeyframeByParam(int clipId, const QString &effectId, const QString &paramName,
                                                      int frame, const QString &value, int keyframeType);
    Q_SCRIPTABLE bool scriptRemoveEffectKeyframeByParam(int clipId, const QString &effectId, const QString &paramName, int frame);

    // Time Remap (speed ramping)
    Q_SCRIPTABLE bool scriptEnableTimeRemap(int clipId, bool enable);
    Q_SCRIPTABLE QVariantMap scriptGetTimeRemap(int clipId);
    Q_SCRIPTABLE bool scriptSetTimeRemap(int clipId, const QString &timeMap, int pitch, const QString &imageMode);

    // Speed

    // Clip Transform Keyframes
    Q_SCRIPTABLE QVariantList scriptGetClipTransformKeyframes(int clipId);
    Q_SCRIPTABLE bool scriptSetClipTransform(int clipId, int frame, int x, int y, int width, int height, double opacity);
    Q_SCRIPTABLE bool scriptRemoveClipTransformKeyframe(int clipId, int frame);

    // Clip Properties
    Q_SCRIPTABLE double scriptGetClipOpacity(int clipId);
    Q_SCRIPTABLE bool scriptSetClipOpacity(int clipId, double opacity);
    Q_SCRIPTABLE bool scriptIsClipEnabled(int clipId);
    Q_SCRIPTABLE bool scriptSetClipEnabled(int clipId, bool enabled);
    Q_SCRIPTABLE QString scriptGetClipColor(int clipId);
    Q_SCRIPTABLE bool scriptSetClipColor(int clipId, const QString &colorTag);

    // Track Properties
    Q_SCRIPTABLE QString scriptGetTrackName(int trackId);
    Q_SCRIPTABLE bool scriptSetTrackName(int trackId, const QString &name);
    Q_SCRIPTABLE int scriptGetTrackColor(int trackId);
    Q_SCRIPTABLE bool scriptSetTrackColor(int trackId, int color);
    Q_SCRIPTABLE bool scriptGetTrackSolo(int trackId);
    Q_SCRIPTABLE bool scriptSetTrackSolo(int trackId, bool solo);

    // Audio
    Q_SCRIPTABLE bool scriptSplitAudio(int clipId);
    Q_SCRIPTABLE bool scriptSetClipVolume(int clipId, double dB);
    Q_SCRIPTABLE double scriptGetClipVolume(int clipId);
    Q_SCRIPTABLE bool scriptSetAudioFade(int clipId, int fadeInFrames, int fadeOutFrames);
    Q_SCRIPTABLE bool scriptSetClipPan(int clipId, double pan);
    Q_SCRIPTABLE double scriptGetClipPan(int clipId);
    Q_SCRIPTABLE bool scriptSetTrackMute(int trackId, bool mute);
    Q_SCRIPTABLE bool scriptGetTrackMute(int trackId);
    Q_SCRIPTABLE bool scriptSetTrackLocked(int trackId, bool locked);
    Q_SCRIPTABLE bool scriptGetTrackLocked(int trackId);
    Q_SCRIPTABLE bool scriptSetTrackHidden(int trackId, bool hidden);
    Q_SCRIPTABLE bool scriptGetTrackHidden(int trackId);
    Q_SCRIPTABLE QVariantList scriptGetAudioLevels(const QString &binId, int stream, int downsample, int mode = 0);

    // Markers & Guides
    Q_SCRIPTABLE bool scriptAddGuide(int frame, const QString &comment, int category);
    Q_SCRIPTABLE QVariantList scriptGetGuides();
    Q_SCRIPTABLE bool scriptDeleteGuide(int frame);
    Q_SCRIPTABLE bool scriptDeleteGuidesByCategory(int category);

    // Clip-level markers
    Q_SCRIPTABLE bool scriptAddClipMarker(const QString &binId, int frame, const QString &comment, int category);
    Q_SCRIPTABLE QVariantList scriptGetClipMarkers(const QString &binId);
    Q_SCRIPTABLE bool scriptDeleteClipMarker(const QString &binId, int frame);
    Q_SCRIPTABLE bool scriptDeleteClipMarkersByCategory(const QString &binId, int category);

    // Render with parameters
    Q_SCRIPTABLE bool scriptRenderWithParams(const QString &outputFile, const QString &presetName,
                                              int inFrame, int outFrame,
                                              const QStringList &paramKeys, const QStringList &paramValues);
    Q_SCRIPTABLE QStringList scriptGetRenderPresets();
    Q_SCRIPTABLE QVariantList scriptGetRenderJobs();
    Q_SCRIPTABLE bool scriptAbortRenderJob(const QString &outputPath);

    // Project Profile
    Q_SCRIPTABLE bool scriptSetProjectProfile(int width, int height, int fpsNum, int fpsDen);

    // Copy/Cut/Paste
    Q_SCRIPTABLE int scriptCopyClips();
    Q_SCRIPTABLE bool scriptCutClips();
    Q_SCRIPTABLE bool scriptPasteClips(int position, int trackId);

    // Playback & Monitor
    Q_SCRIPTABLE void scriptSeek(int frame);
    Q_SCRIPTABLE int scriptGetPosition();
    Q_SCRIPTABLE void scriptPlay();
    Q_SCRIPTABLE void scriptPause();
    Q_SCRIPTABLE bool scriptSetPlaybackSpeed(double speed);
    Q_SCRIPTABLE double scriptGetPlaybackSpeed();

    // Timeline Navigation
    Q_SCRIPTABLE int scriptGoToNextMarker();
    Q_SCRIPTABLE int scriptGoToPreviousMarker();
    Q_SCRIPTABLE int scriptGoToNextEdit();
    Q_SCRIPTABLE int scriptGoToPreviousEdit();

    // Additional
    Q_SCRIPTABLE QVariantList scriptDetectScenes(const QString &binClipId, double threshold = 0.4, int minDuration = 0);

    // Sequences (multi-timeline)
    Q_SCRIPTABLE QString scriptCreateSequence(const QString &name, int audioTracks, int videoTracks,
                                               const QString &parentFolder = QStringLiteral("-1"));
    Q_SCRIPTABLE QVariantList scriptGetSequences();
    Q_SCRIPTABLE QVariantMap scriptGetActiveSequence();
    Q_SCRIPTABLE bool scriptSetActiveSequence(const QString &uuid);

    // Zones (timeline in/out points)
    Q_SCRIPTABLE QVariantMap scriptGetZone();
    Q_SCRIPTABLE bool scriptSetZone(int inFrame, int outFrame);
    Q_SCRIPTABLE bool scriptSetZoneIn(int inFrame);
    Q_SCRIPTABLE bool scriptSetZoneOut(int outFrame);
    Q_SCRIPTABLE bool scriptExtractZone(int inFrame, int outFrame, bool liftOnly);

    // Fill Frame (remove black bars)
    Q_SCRIPTABLE bool scriptFillFrame(int clipId);

    // Preview / Frame Rendering
    Q_SCRIPTABLE QString scriptRenderBinFrame(const QString &binId, int frame, int width, int height, const QString &outputPath);
    Q_SCRIPTABLE QString scriptRenderTimelineFrame(int frame, int width, int height, const QString &outputPath);
    Q_SCRIPTABLE QString scriptCaptureWindow(int maxSize, const QString &outputPath);
    Q_SCRIPTABLE QString scriptGetPanelGeometries();

    // Subtitles
    Q_SCRIPTABLE QVariantList scriptGetSubtitles();
    Q_SCRIPTABLE int scriptAddSubtitle(int startFrame, int endFrame, const QString &text, int layer = 0);
    Q_SCRIPTABLE bool scriptEditSubtitle(int subtitleId, const QString &newText);
    Q_SCRIPTABLE bool scriptMoveSubtitle(int subtitleId, int newStartFrame);
    Q_SCRIPTABLE bool scriptResizeSubtitle(int subtitleId, int newDuration, bool fromRight);
    Q_SCRIPTABLE bool scriptImportSubtitle(const QString &filePath, int offset, const QString &encoding);
    Q_SCRIPTABLE bool scriptDeleteSubtitle(int subtitleId);
    Q_SCRIPTABLE bool scriptExportSubtitles(const QString &filePath);

    // Speech Recognition
    Q_SCRIPTABLE bool scriptSpeechRecognition(const QString &model, const QString &language, int maxLineWidth);

    // Subtitle Styles
    Q_SCRIPTABLE QVariantList scriptGetSubtitleStyles(bool global);
    Q_SCRIPTABLE bool scriptSetSubtitleStyle(const QString &name, const QStringList &keys, const QStringList &values, bool global);
    Q_SCRIPTABLE bool scriptDeleteSubtitleStyle(const QString &name, bool global);
    Q_SCRIPTABLE bool scriptSetSubtitleStyleName(int subtitleId, const QString &styleName);

    // Groups
    Q_SCRIPTABLE int scriptGroupClips(const QList<int> &itemIds);
    Q_SCRIPTABLE bool scriptUngroupClips(int itemId);
    Q_SCRIPTABLE QVariantMap scriptGetGroupInfo(int itemId);
    Q_SCRIPTABLE bool scriptRemoveFromGroup(int itemId);

    // Proxy Clips
    Q_SCRIPTABLE QVariantMap scriptGetClipProxyStatus(const QString &binId);
    Q_SCRIPTABLE bool scriptSetClipProxy(const QString &binId, bool enabled);
    Q_SCRIPTABLE bool scriptDeleteClipProxy(const QString &binId);
    Q_SCRIPTABLE bool scriptRebuildClipProxy(const QString &binId);

    // Bin Relink
    Q_SCRIPTABLE bool scriptRelinkBinClip(const QString &binId, const QString &newFilePath);

    // Undo / Redo
    Q_SCRIPTABLE bool scriptUndo(int steps = 1);
    Q_SCRIPTABLE bool scriptRedo(int steps = 1);
    Q_SCRIPTABLE QString scriptUndoStatus();

    // Additional

    // Selection (clip/composition/subtitle selection) -----------
    Q_SCRIPTABLE QVariantList scriptGetSelection();
    Q_SCRIPTABLE bool scriptSetSelection(const QList<int> &ids);
    Q_SCRIPTABLE bool scriptAddToSelection(int itemId, bool clear);
    Q_SCRIPTABLE bool scriptClearSelection();
    Q_SCRIPTABLE bool scriptSelectAll();
    Q_SCRIPTABLE bool scriptSelectCurrentTrack();
    Q_SCRIPTABLE bool scriptSelectItems(const QList<int> &trackIds, int startFrame, int endFrame);

    // AI plugins (.wunjoplugin): enumerate with option schemas, run headless.
    Q_SCRIPTABLE QVariantList scriptListPlugins();
    /** @brief Start a plugin run and answer with the id of the job, so the
     *  caller can ask what became of it. Empty when it could not be started. */
    Q_SCRIPTABLE QString scriptRunPlugin(const QString &id, const QString &inputJson);
    /** @brief `{"state": "running"|"done"|"failed", "percent": n, "message": …}`
     *  for a job id from @ref scriptRunPlugin or @ref scriptGenerateEffect.
     *
     *  A plugin says what happened in one sentence — "choose an audio preset
     *  first" — and an assistant that cannot read it reports work it never did.
     *  This is where that sentence is read. */
    Q_SCRIPTABLE QString scriptPluginJobStatus(const QString &jobId);
    Q_SCRIPTABLE bool scriptInstallPlugin(const QString &id);

    // ── Object masks (SAM-2) ────────────────────────────────────────────
    // Segment Anything is promptable by geometry, not by language: point at a
    // thing on one frame and it follows that thing through the rest. Everything
    // needed for that already existed behind the mask panel; these expose the
    // same three steps so an agent can use it too — prepare the frames, say
    // where to look, generate.
    Q_SCRIPTABLE bool scriptSamPrepare(const QString &binId, int zoneIn, int zoneOut);
    Q_SCRIPTABLE QVariantMap scriptSamStatus();
    Q_SCRIPTABLE bool scriptSamGenerate(const QString &binId, const QString &maskName, const QString &includePoints,
                                        const QString &excludePoints, const QString &boxes, int zoneIn, int zoneOut);
    Q_SCRIPTABLE QVariantList scriptListMasks(const QString &binId);
    Q_SCRIPTABLE QVariantList scriptListPluginSets(const QString &pluginId, const QString &kind);
    /** @brief Start the render an effect describes; answers with the job id to
     *  follow with @ref scriptPluginJobStatus, or empty when there is no such
     *  effect on that clip. */
    Q_SCRIPTABLE QString scriptGenerateEffect(int clipId, const QString &effectId);
    // Preflight: is a plugin ready to run (installed, API key, venv/deps, models)?
    Q_SCRIPTABLE QVariantMap scriptPluginStatus(const QString &id);

    // Built-in face detection (RetinaFace ONNX, background analysis, no UI).
    /** @param from,to the part of the clip to analyse before the rest, in
     *  frames; -1 for "no part first". A caller working on one piece of a long
     *  source — the assistant, a plugin — gets its boxes in seconds instead of
     *  waiting for footage that was cut out. */
    Q_SCRIPTABLE bool scriptSetFaceDetection(const QString &binId, bool enabled, int from = -1, int to = -1);
    Q_SCRIPTABLE QVariantMap scriptGetFaceDetectionStatus(const QString &binId);
    Q_SCRIPTABLE QVariantList scriptGetFacesAtFrame(const QString &binId, int position);
    // Select a detected face on a timeline clip and apply the chosen effect,
    // tracked across the clip. effectId defaults to the "Hide Face" Motion
    // Tracker (opencv.tracker); paramsJson overrides/extends its parameters.
    Q_SCRIPTABLE bool scriptApplyFaceEffect(int clipId, int faceIndex, int atFrame, const QString &effectId, const QString &paramsJson);

    // Chat narration: an external agent (Claude Code over MCP) mirrors the
    // conversation and its work into the chat dock — user request, replies,
    // a "thinking" state and live tool/plugin/whisper progress.
    Q_SCRIPTABLE bool scriptChatMessage(int author, const QString &text);
    Q_SCRIPTABLE bool scriptChatStream(const QString &text);
    Q_SCRIPTABLE bool scriptChatThinking(bool on);
    Q_SCRIPTABLE bool scriptChatToolStart(const QString &id, const QString &name);
    Q_SCRIPTABLE bool scriptChatToolProgress(const QString &id, int percent, const QString &message);
    Q_SCRIPTABLE bool scriptChatToolEnd(const QString &id, bool isError, const QString &result);

    // Assistant guidance: skills (multi-select) and loops (single scenario).
    // Library is global; selection is stored per project. Mirrors the chat UI.
    Q_SCRIPTABLE QStringList scriptListSkills();
    Q_SCRIPTABLE QString scriptGetSkill(const QString &name);
    Q_SCRIPTABLE bool scriptSaveSkill(const QString &name, const QString &content);
    Q_SCRIPTABLE bool scriptDeleteSkill(const QString &name);
    Q_SCRIPTABLE QStringList scriptGetSelectedSkills();
    Q_SCRIPTABLE bool scriptSetSelectedSkills(const QStringList &names);
    Q_SCRIPTABLE QStringList scriptListLoops();
    Q_SCRIPTABLE QString scriptGetLoop(const QString &name);
    Q_SCRIPTABLE bool scriptSaveLoop(const QString &name, const QString &content);
    Q_SCRIPTABLE bool scriptDeleteLoop(const QString &name);
    Q_SCRIPTABLE QString scriptGetSelectedLoop();
    Q_SCRIPTABLE bool scriptSelectLoop(const QString &name);


public Q_SLOTS:
    void slotReloadEffects(const QStringList &paths);
    /** @brief A plugin was installed or removed: put the effects it brings on
     *  the effect list, or take away the ones it took with it. */
    void slotUpdatePluginEffects(const QStringList &addedFiles, const QStringList &removedIds);
    Q_SCRIPTABLE void setRenderingProgress(const QString &url, int progress, int frame);
    Q_SCRIPTABLE void setRenderingFinished(const QString &url, int status, const QString &error);
    Q_SCRIPTABLE void addProjectClip(const QString &url, const QString &folder = QStringLiteral("-1"));
    Q_SCRIPTABLE void addTimelineClip(const QString &url);
    Q_SCRIPTABLE void addEffect(const QString &effectId);
    Q_SCRIPTABLE void scriptRender(const QString &url);
    Q_SCRIPTABLE void exitApp();

    void slotSwitchVideoThumbs();
    void slotSwitchAudioThumbs();
    void appHelpActivated();
    /** @brief Restart the application and delete config files if clean is true */
    void cleanRestart(bool clean, bool forceQuit = false);

    void slotPreferences();
    void slotShowPreferencePage(Wunjo::ConfigPage page, int option = -1);
    /** @brief Open the settings on the tab of plugin @p pluginId — where its
     *  environment is built and its weights are fetched. */
    void showPluginSettings(const QString &pluginId);
    void connectDocument();
    /** @brief Reload project profile in config dialog if changed. */
    void slotRefreshProfiles();
    /** @brief Decreases the timeline zoom level by 1. */
    void slotZoomIn(bool zoomOnMouse = false);
    /** @brief Increases the timeline zoom level by 1. */
    void slotZoomOut(bool zoomOnMouse = false);
    /** @brief Enable or disable the use of timeline zone for edits. */
    void slotSwitchTimelineZone(bool toggled);
    /** @brief Open the online services search dialog. */
    void slotDownloadResources();
    /** @brief Initialize the subtitle model on project load. */
    void slotInitSubtitle(const QMap<QString, QString> &subProperties, const QUuid &uuid);
    /** @brief Display the subtitle track and initialize subtitleModel if necessary. */
    void slotEditSubtitle(const QMap<QString, QString> &subProperties = {});
    /** @brief Show/hide subtitle track. */
    void slotShowSubtitles(bool show);
    void slotTranscode(const QStringList &urls = QStringList());
    /** @brief Open the transcode to edit friendly format dialog. */
    void slotFriendlyTranscode(const QString &binId, bool checkProfile);
    /** @brief Add subtitle clip to timeline */
    void slotAddSubtitle(const QString &text = QString());
    /** @brief Ensure subtitle track is displayed */
    void showSubtitleTrack();
    /** @brief The path of the current document changed (save as), update render settings */
    void updateProjectPath(const QString &path);
    /** @brief Update compositing action to display current project setting. */
    void slotUpdateCompositeAction(bool enable);
    /** @brief Update duration of project in timeline toolbar. */
    void slotUpdateProjectDuration(int pos);
    /** @brief The current timeline selection zone changed... */
    void slotUpdateZoneDuration();
    /** @brief Remove all unused clips from the project. */
    void slotCleanProject();
    void slotEditProjectSettings(int ix = 0);
    /** @brief Sets the timeline zoom slider to @param value.
     *
     * Also disables zoomIn and zoomOut actions if they cannot be used at the moment. */
    void slotSetZoom(int value, bool zoomOnMouse = false);
    /** @brief if modified is true adds "modified" to the caption and enables the save button.
     * (triggered by WunjoDoc::setModified()) */
    void slotUpdateDocumentState(bool modified);
    /** @brief Open the clip job management dialog */
    void manageClipJobs(AbstractTask::JOBTYPE type = AbstractTask::NOJOBTYPE, QWidget *parentWidget = nullptr);
    /** @brief Deletes item in timeline, project tree or effect stack depending on focus. */
    void slotDeleteItem();
    /** @brief Export a subtitle file */
    void slotExportSubtitle();
    /** @brief Display current mouse pos */
    void slotUpdateMousePosition(int pos, int duration = -1);
    /** @brief Focus current timeline clip in bin and display its range */
    void slotClipInProjectTree(ObjectId ownerId = ObjectId(), bool seekToStart = false);
    /** @brief Normalize audio channels before displaying them */
    void slotNormalizeAudioChannel(bool normalize);
    /** @brief Recursively calculate folder size */
    KIO::filesize_t fetchFolderSize(const QString path);

private Q_SLOTS:
    /** @brief Shows the shortcut dialog. */
    void slotEditKeys();
    void slotEditToolbars();
    void loadDockActions();
    /** @brief Reflects setting changes to the GUI. */
    void updateConfiguration();
    void slotConnectMonitors();
    void slotSwitchMarkersComments();
    void slotSwitchSnap();
    void slotSwitchClipOverlays();
    void slotShowTimelineTags();
    void slotMouseZoomOnPlayhead();
    void slotRenderProject();
    void slotStopRenderProject();
    void slotFullScreen();
    /** @brief Process a few last things as soon as ui is built */
    void finishUiSetup();
    /** @brief Close Wunjo and try to restart it */
    void slotRestart(bool clean = false);

    /** @brief Makes the timeline zoom level fit the timeline content. */
    void slotFitZoom();
    /** @brief Updates the zoom slider tooltip to fit @param zoomlevel. */
    void slotUpdateZoomSliderToolTip(int zoomlevel);
    /** @brief Timeline was zoom, update slider to reflect that */
    void updateZoomSlider(int value);

    /** @brief Displays the zoom slider tooltip.
     * @param zoomlevel (optional) The zoom level to show in the tooltip.
     *
     * Adopted from Dolphin (src/statusbar/dolphinstatusbar.cpp) */
    void slotShowZoomSliderToolTip(int zoomlevel = -1);
    void slotAddClipMarker();
    void slotDeleteClipMarker(bool allowGuideDeletion = false);
    void slotDeleteAllClipMarkers();
    void slotDeleteAllSequenceMarkers();
    void slotEditClipMarker();

    /** @brief Adds marker or guide at the current position without showing the marker dialog.
     *
     * Adds a marker if clip monitor is active, otherwise a guide.
     * The comment is set to the current position (therefore not dialog).
     * This can be useful to mark something during playback. */
    void slotAddMarkerGuideQuickly();
    void slotAddMarkerWithCategory();
    void slotCutTimelineClip();
    void slotReplaceTimelineClip();
    void slotCutTimelineAllClips();
    void slotInsertClipOverwrite();
    void slotInsertClipInsert();
    void slotExtractZone();
    void slotLiftZone();
    void slotCreateRangeMarkerFromZone();
    void slotCreateRangeMarkerFromZoneQuick();
    void slotPreviewRender();
    void slotStopPreviewRender();
    void slotDefinePreviewRender();
    void slotRemovePreviewRender();
    void slotClearPreviewRender(bool resetZones = true);
    void slotSelectTimelineClip();
    void slotSelectTimelineZone();
    void slotSelectTimelineTransition();
    void slotDeselectTimelineClip();
    void slotDeselectTimelineTransition();
    void slotSelectAddTimelineClip();
    void slotSelectAddTimelineTransition();
    void slotAddEffect(QAction *result);
    void slotAddTransition(QAction *result);
    void slotAddProjectClip(const QUrl &url, const QString &folderInfo);
    void slotAddTextNote(const QString &text);
    void slotAddProjectClipList(const QList<QUrl> &urls);
    void slotChangeTool(QAction *action);
    void slotChangeEdit(QAction *action);
    void slotSetTool(ToolType::ProjectTool tool);
    void slotSnapForward();
    void slotSnapRewind();
    void slotGuideForward();
    void slotGuideRewind();
    void slotClipStart();
    void slotClipEnd();
    void slotSelectClipInTimeline();
    void slotClipInTimeline(const QString &clipId, const QList<int> &ids);

    void slotInsertSpace();
    void slotRemoveSpace();
    void slotRemoveSpaceInAllTracks();
    void slotRemoveAllSpacesInTrack();
    void slotRemoveAllClipsInTrack();
    void slotAddMarkersAtGaps();
    void slotAddMarkersAtGapsOnTrack();
    void slotAddGuide();
    void slotEditGuide();
    void slotExportGuides();
    void slotLockGuides(bool lock);
    void slotDeleteGuide();
    void slotDeleteAllGuides();

    void slotCopy();
    void slotCut();
    void slotPaste();
    void slotPasteEffects();
    void slotResizeItemStart();
    void slotResizeItemEnd();
    void configureNotifications();
    void slotSeparateAudioChannel();
    /** @brief Toggle automatic fit track height */
    void slotAutoTrackHeight(bool enable);
    void slotInsertTrack();
    void slotDeleteTrack();
    void slotMoveTrackUp();
    void slotMoveTrackDown();
    /** @brief Show context menu to switch current track target audio stream. */
    void slotSwitchTrackAudioStream();
    void slotShowTrackRec(bool checked);
    /** @brief Select all clips in active track. */
    void slotSelectTrack();
    /** @brief Select all clips in timeline. */
    void slotSelectAllTracks();
    void slotUnselectAllTracks();
    void slotRunWizard();
    void slotGroupClips();
    void slotUnGroupClips();
    void slotEditItemDuration();
    // void slotClipToProjectTree();
    void slotSplitAV();
    void slotSwitchClip();
    void slotSetAudioAlignReference();
    void slotAlignAudio();
    void slotSetTimecodeReference();
    void slotAlignTimecode();
    void slotUpdateTimelineView(QAction *action);
    void slotTranscodeClip();
    /** @brief Archive project: creates a copy of the project file with all clips in a new folder. */
    void slotArchiveProject();
    void slotSetDocumentRenderProfile(const QMap<QString, QString> &props);

    /** @brief Switches between displaying frames or timecode.
     * @param ix 0 = display timecode, 1 = display frames. */
    void slotUpdateTimecodeFormat(int ix);
    /** @brief Apply tool message styling based on current edit mode */
    void applyToolMessageStyling();
    /** @brief Apply zoom level button styling based on current zoom level */
    void applyZoomLevelButtonStyling();

    /** @brief Removes the focus of anything. */
    void slotRemoveFocus();
    void slotShutdown();

    void slotSwitchMonitors();
    void slotSwitchMonitorOverlay(QAction *);
    void slotSwitchDropFrames(bool drop);
    void slotCheckRenderStatus();
    void slotInsertZoneToTree();
    /** @brief Focus the timecode widget of current monitor. */
    void slotFocusTimecode();

    /** @brief Update project because the use of proxy clips was enabled / disabled. */
    void slotUpdateProxySettings();
    /** @brief Disable proxies for this project. */
    void slotDisableProxies();

    /** @brief Process keyframe data sent from a clip to effect / transition stack. */
    void slotProcessImportKeyframes(GraphicsRectItem type, const QString &tag, const QString &keyframes);
    /** @brief Move playhead to mouse cursor position if defined key is pressed */
    void slotAlignPlayheadToMousePos();

    void triggerKey(QKeyEvent *ev);
    /** @brief Update monitor overlay actions on monitor switch */
    void slotUpdateMonitorOverlays(int id, int code);
    /** @brief Create temporary top track to preview an effect */
    void createSplitOverlay(std::shared_ptr<Mlt::Filter> filter);
    void removeSplitOverlay();
    /** @brief Create a generator's setup dialog */
    void buildGenerator(QAction *action);
    void slotCheckTabPosition();
    /** @brief Toggle automatic timeline preview on/off */
    void slotToggleAutoPreview(bool enable);
    void showTimelineToolbarMenu(const QPoint &pos);
    /** @brief Open Cached Data management dialog. */
    void slotManageCache();
    void showMenuBar(bool show);
    /** @brief Toggle current project's compositing mode. */
    void slotUpdateCompositing(bool checked);
    /** @brief Set timeline toolbar icon size. */
    void setTimelineToolbarIconSize(QAction *a);
    void slotEditItemSpeed();
    void slotRemapItemTime();
    /** @brief Request adjust of timeline track height */
    void resetTimelineTracks();
    /** @brief Set keyboard grabbing on current timeline item */
    void slotGrabItem();
    /** @brief Collapse or expand current item (depending on focused widget: effet, track)*/
    void slotCollapse();
    void slotCollapseAll();
    void slotAudioZoomIn();
    void slotAudioZoomOut();
    void slotAudioZoomReset();
    void slotAudioZoomCycle();
    /** @brief Save currently selected timeline clip as bin subclip*/
    void slotExtractClip();
    /** @brief Save currently selected timeline clip as bin subclip*/
    void slotSaveZoneToBin();
    /** @brief Expand current timeline clip (recover clips and tracks from an MLT playlist) */
    void slotExpandClip();
    /** @brief Focus and activate an audio track from a shortcut sequence */
    void slotActivateAudioTrackSequence();
    /** @brief Focus and activate a video track from a shortcut sequence */
    void slotActivateVideoTrackSequence();
    /** @brief Select target for current track */
    void slotActivateTarget();
    /** @brief Enable/disable subtitle track */
    void slotDisableSubtitle();
    /** @brief Lock / unlock subtitle track */
    void slotLockSubtitle();
    /** @brief Import a subtitle file */
    void slotImportSubtitle();
    /** @brief Display the subtitle manager widget */
    void slotManageSubtitle();
    /** @brief Start a speech recognition on timeline zone */
    void slotSpeechRecognition();
    /** @brief Copy debug information like lib versions, gpu mode state,... to clipboard */
    void slotCopyDebugInfo();
    void slotRemoveBinDock(const QString &name);
    /** @brief Focus the guides list search line */
    void slotSearchGuide();
    /** @brief Focus the bin search line */
    void slotSearchBin();
    /** @brief Move current timeline selection to a new sequence clip / Timeline tab */
    void slotCreateSequenceFromSelection();
    /** @brief Copy current timeline selection to a new sequence clip / Timeline tab */
    void slotCopyAndCreateSequenceFromSelection();

Q_SIGNALS:
    Q_SCRIPTABLE void abortRenderJob(const QString &url);
    void abortAllRenderJobs();
    void configurationChanged();
    void setPreviewProgress(int);
    void setRenderProgress(int);
    void displayMessage(const QString &, MessageType, int);
    void displaySelectionMessage(const QString &);
    void displayProgressMessage(const QString &, MessageType, int, bool canBeStopped = false);
    /** @brief Project profile changed, update render widget accordingly. */
    void updateRenderWidgetProfile();
    /** @brief Clear asset view if itemId is displayed. */
    void clearAssetPanel(int itemId = -1);
    void assetPanelWarning(const QString service, const QString message, const QString log = QString());
    void adjustAssetPanelRange(int itemId, int in, int out);
    /** @brief Enable or disable the undo stack. For example undo/redo should not be enabled when dragging a clip in timeline or we risk corruption. */
    void enableUndo(bool enable);
    void removeBinDock(const QString &name);
    /** @brief Connect a newly created dock to signals updating/hiding its title bar. */
    void connectDockAfterInit(QDockWidget *);
};
