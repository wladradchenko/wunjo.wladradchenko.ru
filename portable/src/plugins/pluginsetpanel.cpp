/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "pluginsetpanel.h"

#include "pluginmanager.h"
#include "pluginsetstore.h"

#include <KLocalizedString>
#include <KMessageBox>
#include <KMessageWidget>

#include <QAction>
#include <QAudioOutput>
#include <QDesktopServices>
#include <QFileDialog>
#include <QFileInfo>
#include <QFontDatabase>
#include <QHBoxLayout>
#include <QInputDialog>
#include <QJsonArray>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QLocale>
#include <QMediaPlayer>
#include <QMenu>
#include <QMovie>
#include <QPixmap>
#include <QPointer>
#include <QProgressBar>
#include <QPushButton>
#include <QToolButton>
#include <QTreeWidget>
#include <QUrl>
#include <QVBoxLayout>
#include <QtConcurrent/QtConcurrentRun>

namespace {
constexpr int kFileRole = Qt::UserRole;
constexpr int kSourceRole = Qt::UserRole + 1;
constexpr int kThumbRole = Qt::UserRole + 2;
constexpr int kPreviewRole = Qt::UserRole + 3;
/** @brief Row pictures in the list, and the larger one below it. */
constexpr int kListIcon = 48;
constexpr int kPreviewHeight = 176;
} // namespace

PluginSetPanel::PluginSetPanel(QWidget *parent)
    : QWidget(parent)
{
    setFont(QFontDatabase::systemFont(QFontDatabase::SmallestReadableFont));
    auto *layout = new QVBoxLayout(this);

    m_title = new QLabel(this);
    m_title->setWordWrap(true);
    layout->addWidget(m_title);

    // ---- record a new set ----
    auto *sourceRow = new QHBoxLayout;
    m_choose = new QPushButton(QIcon::fromTheme(QStringLiteral("document-open")), i18n("Video or photo…"), this);
    m_choose->setToolTip(i18n("The media this preset is made from"));
    m_sourceLabel = new QLabel(i18n("nothing chosen"), this);
    m_sourceLabel->setStyleSheet(QStringLiteral("color:#696969"));
    m_playSource = new QToolButton(this);
    m_playSource->setIcon(QIcon::fromTheme(QStringLiteral("media-playback-start")));
    m_playSource->setToolTip(i18n("Listen to the chosen track"));
    m_playSource->setAutoRaise(true);
    m_playSource->hide();
    sourceRow->addWidget(m_choose);
    sourceRow->addWidget(m_playSource);
    sourceRow->addWidget(m_sourceLabel, 1);
    layout->addLayout(sourceRow);

    auto *analyseRow = new QHBoxLayout;
    m_analyse = new QPushButton(QIcon::fromTheme(QStringLiteral("media-playback-start")), i18n("Analyse"), this);
    m_analyse->setEnabled(false);
    m_progress = new QProgressBar(this);
    m_progress->setRange(0, 100);
    m_progress->hide();
    analyseRow->addWidget(m_analyse);
    analyseRow->addWidget(m_progress, 1);
    layout->addLayout(analyseRow);

    m_status = new KMessageWidget(this);
    m_status->setCloseButtonVisible(false);
    m_status->setWordWrap(true);
    m_status->hide();
    layout->addWidget(m_status);

    // ---- what has been recorded ----
    m_sets = new QTreeWidget(this);
    m_sets->setColumnCount(2);
    m_sets->setHeaderLabels({i18n("Set"), i18n("Frames")});
    m_sets->setRootIsDecorated(false);
    m_sets->setAlternatingRowColors(true);
    m_sets->setAllColumnsShowFocus(true);
    m_sets->setIconSize(QSize(kListIcon, kListIcon));
    m_sets->setContextMenuPolicy(Qt::CustomContextMenu);
    layout->addWidget(m_sets, 1);

    // ---- and what the selected one looks like ----
    m_preview = new QLabel(this);
    m_preview->setAlignment(Qt::AlignCenter);
    m_preview->setFixedHeight(kPreviewHeight);
    m_preview->hide();
    layout->addWidget(m_preview);

    auto *buttons = new QHBoxLayout;
    m_delete = new QToolButton(this);
    m_delete->setIcon(QIcon::fromTheme(QStringLiteral("edit-delete")));
    m_delete->setToolTip(i18n("Delete set"));
    m_delete->setAutoRaise(true);
    m_import = new QToolButton(this);
    m_import->setIcon(QIcon::fromTheme(QStringLiteral("document-import")));
    m_import->setToolTip(i18n("Import a set from another project"));
    m_import->setAutoRaise(true);
    m_playSet = new QToolButton(this);
    m_playSet->setIcon(QIcon::fromTheme(QStringLiteral("media-playback-start")));
    m_playSet->setToolTip(i18n("Listen to the selected track"));
    m_playSet->setAutoRaise(true);
    m_playSet->hide();
    auto *close = new QPushButton(i18n("Back to effects"), this);
    buttons->addWidget(m_delete);
    buttons->addWidget(m_import);
    buttons->addWidget(m_playSet);
    buttons->addStretch();
    buttons->addWidget(close);
    layout->addLayout(buttons);

    // Renaming, from the context menu and from F2 on the list: two photos both
    // called portrait.jpg arrive under one name, and the name is what the
    // effect's list shows.
    auto *renameAction = new QAction(QIcon::fromTheme(QStringLiteral("edit-rename")), i18n("Rename set…"), this);
    renameAction->setShortcut(Qt::Key_F2);
    renameAction->setShortcutContext(Qt::WidgetWithChildrenShortcut);
    m_sets->addAction(renameAction);
    connect(renameAction, &QAction::triggered, this, &PluginSetPanel::renameSet);

    connect(m_choose, &QPushButton::clicked, this, &PluginSetPanel::chooseSource);
    connect(m_analyse, &QPushButton::clicked, this, &PluginSetPanel::analyse);
    connect(m_playSource, &QToolButton::clicked, this, [this]() { togglePlay(m_source); });
    connect(m_playSet, &QToolButton::clicked, this, [this]() {
        QTreeWidgetItem *item = m_sets->currentItem();
        togglePlay(item ? item->data(0, kSourceRole).toString() : QString());
    });
    // A track is unrecognisable by name; double-clicking the row plays it. A
    // picture is recognisable, and double-clicking opens what it was made from.
    connect(m_sets, &QTreeWidget::itemDoubleClicked, this, [this](QTreeWidgetItem *item) {
        if (item == nullptr) {
            return;
        }
        const QString source = item->data(0, kSourceRole).toString();
        if (isAudioKind()) {
            togglePlay(source);
        } else if (!source.isEmpty() && QFileInfo::exists(source)) {
            QDesktopServices::openUrl(QUrl::fromLocalFile(source));
        } else {
            showStatus(i18n("The file this preset was made from is gone."), true);
        }
    });
    connect(m_delete, &QToolButton::clicked, this, &PluginSetPanel::deleteSet);
    connect(m_import, &QToolButton::clicked, this, &PluginSetPanel::importSet);
    connect(close, &QPushButton::clicked, this, [this]() {
        // leaving the panel must not leave a track playing behind it
        stopPlayback();
        Q_EMIT closeRequested();
    });
    connect(m_sets, &QTreeWidget::currentItemChanged, this, [this]() {
        updateButtons();
        updatePreview();
    });
    connect(m_sets, &QTreeWidget::customContextMenuRequested, this, [this, renameAction](const QPoint &pos) {
        if (m_sets->itemAt(pos) == nullptr) {
            return;
        }
        QMenu menu(this);
        menu.addAction(renameAction);
        QAction *exportAction = menu.addAction(QIcon::fromTheme(QStringLiteral("document-export")), i18n("Export set…"));
        connect(exportAction, &QAction::triggered, this, &PluginSetPanel::exportSet);
        menu.exec(m_sets->viewport()->mapToGlobal(pos));
    });
    updateButtons();
}

PluginSetPanel::~PluginSetPanel()
{
    // the picture job holds only copies and a guarded pointer back here, but
    // it reads the project's media: let it finish rather than pull the project
    // out from under it
    m_thumbJob.waitForFinished();
}

void PluginSetPanel::setPlugin(const QString &pluginId, const QString &kind)
{
    stopPlayback();
    m_pluginId = pluginId;
    m_kind = kind;
    const PluginManifest manifest = PluginManager::instance().plugin(pluginId);
    // What a preset is differs per plugin — one records a performance, another
    // registers a face — so the plugin supplies the wording and the filter.
    const PluginSetsUi ui = manifest.setsUi(kind);
    const QString name = manifest.isValid() ? manifest.name() : pluginId;
    m_sourceFilter = ui.filter.isEmpty() ? i18n("Video and image files (*.mp4 *.mov *.avi *.webm *.mkv *.png *.jpg *.jpeg *.webp);;All files (*)") : ui.filter;
    m_title->setText(ui.label.isEmpty()
                         ? i18n("<b>%1</b> — recorded sets. Analyse a video or a photo to record how it moves, then pick "
                                "the set in the effect.",
                                name)
                         : QStringLiteral("<b>%1</b> — %2").arg(name, ui.label.toHtmlEscaped()));
    m_analyse->setText(ui.action.isEmpty() ? i18n("Analyse") : ui.action);
    m_choose->setText(ui.filter.isEmpty() ? i18n("Video or photo…") : i18n("Choose file…"));
    m_sets->setHeaderLabels({i18n("Set"), ui.detail.isEmpty() ? i18n("Frames") : ui.detail});
    refresh();
}

void PluginSetPanel::chooseSource()
{
    const QString path = QFileDialog::getOpenFileName(this, m_choose->text(), QString(), m_sourceFilter);
    if (path.isEmpty()) {
        return;
    }
    m_source = path;
    m_sourceLabel->setText(QFileInfo(path).fileName());
    updateButtons();
}

void PluginSetPanel::analyse()
{
    if (m_source.isEmpty() || m_pluginId.isEmpty() || m_running) {
        return;
    }
    if (PluginSets::folder(m_pluginId).isEmpty()) {
        showStatus(i18n("Save the project first — sets are stored next to it."), true);
        return;
    }
    m_running = true;
    m_progress->setValue(0);
    m_progress->show();
    showStatus(QStringLiteral("%1: %2…").arg(m_analyse->text(), QFileInfo(m_source).fileName()), false);
    updateButtons();

    QJsonObject input;
    input.insert(QStringLiteral("action"), QStringLiteral("analyse"));
    input.insert(QStringLiteral("source"), m_source);
    if (!m_kind.isEmpty()) {
        // which of its presets the plugin is being asked for
        input.insert(QStringLiteral("kind"), m_kind);
    }
    const QString name = QFileInfo(m_source).completeBaseName();
    PluginManager::instance().runPluginJob(
        m_pluginId, input, this, [this](int progress) { m_progress->setValue(progress); },
        [this, name](const QJsonObject &result, const QString &error) {
            m_running = false;
            m_progress->hide();
            updateButtons();
            if (!error.isEmpty()) {
                showStatus(error, true);
                return;
            }
            // Everything the plugin handed back goes to the store: the set, and
            // the pictures of it when the plugin drew some.
            const QJsonArray outputs = result.value(QStringLiteral("outputs")).toArray();
            const QString file = outputs.isEmpty() ? QString() : outputs.first().toObject().value(QStringLiteral("path")).toString();
            if (file.isEmpty()) {
                showStatus(i18n("Nothing came back for %1.", QFileInfo(m_source).fileName()), true);
                return;
            }
            QString storeError;
            const PluginSets::Set stored = PluginSets::store(m_pluginId, name, m_kind, outputs, &storeError);
            if (!stored.isValid()) {
                showStatus(storeError.isEmpty() ? i18n("The set could not be saved.") : storeError, true);
                return;
            }
            showStatus(i18n("Recorded '%1' — %2 frames.", stored.name, stored.count), false);
            refresh();
            Q_EMIT setsChanged();
        });
}

void PluginSetPanel::refresh()
{
    const QString current = m_sets->currentItem() ? m_sets->currentItem()->data(0, kFileRole).toString() : QString();
    m_sets->clear();
    const QVector<PluginSets::Set> sets = PluginSets::sets(m_pluginId, m_kind);
    QVector<PluginSets::Set> withoutPicture;
    for (const PluginSets::Set &set : sets) {
        auto *item = new QTreeWidgetItem(m_sets, {set.name, QString::number(set.count)});
        item->setData(0, kFileRole, set.file);
        item->setData(0, kSourceRole, set.source);
        item->setData(0, kThumbRole, set.thumb);
        item->setData(0, kPreviewRole, set.preview);
        if (!set.thumb.isEmpty()) {
            item->setIcon(0, QIcon(set.thumb));
        } else if (!set.source.isEmpty() && !m_thumbTried.contains(set.file)) {
            withoutPicture.append(set);
        }
        // where it came from, and when: two recordings of one file are told
        // apart by their dates when their names say the same thing
        const QString recorded = QLocale().toString(QFileInfo(set.file).lastModified(), QLocale::ShortFormat);
        item->setToolTip(0, set.source.isEmpty() ? i18n("Recorded %1", recorded) : i18n("%1\nRecorded %2", set.source, recorded));
        if (set.file == current) {
            m_sets->setCurrentItem(item);
        }
    }
    m_sets->resizeColumnToContents(0);
    updateButtons();
    updatePreview();
    makeMissingThumbnails(withoutPicture);
}

void PluginSetPanel::makeMissingThumbnails(const QVector<PluginSets::Set> &sets)
{
    if (sets.isEmpty() || m_thumbJob.isRunning()) {
        return;
    }
    for (const PluginSets::Set &set : sets) {
        m_thumbTried.insert(set.file);
    }
    QPointer<PluginSetPanel> guard(this);
    m_thumbJob = QtConcurrent::run([guard, sets]() {
        for (const PluginSets::Set &set : sets) {
            PluginSets::makeThumbnail(set);
        }
        if (guard) {
            QMetaObject::invokeMethod(
                guard.data(),
                [guard]() {
                    if (guard) {
                        guard->refresh();
                        Q_EMIT guard->setsChanged();
                    }
                },
                Qt::QueuedConnection);
        }
    });
}

void PluginSetPanel::updatePreview()
{
    if (m_movie) {
        m_movie->stop();
        m_preview->setMovie(nullptr);
        delete m_movie;
        m_movie = nullptr;
    }
    m_preview->clear();
    QTreeWidgetItem *item = m_sets->currentItem();
    if (item == nullptr || isAudioKind()) {
        m_preview->hide();
        return;
    }
    const QString preview = item->data(0, kPreviewRole).toString();
    if (!preview.isEmpty() && QFileInfo::exists(preview)) {
        m_movie = new QMovie(preview, QByteArray(), this);
        m_preview->setMovie(m_movie);
        m_movie->start();
        m_preview->show();
        return;
    }
    const QString thumb = item->data(0, kThumbRole).toString();
    const QPixmap picture(thumb);
    if (thumb.isEmpty() || picture.isNull()) {
        m_preview->hide();
        return;
    }
    m_preview->setPixmap(picture.scaled(QSize(2 * kPreviewHeight, kPreviewHeight), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    m_preview->show();
}

void PluginSetPanel::updateButtons()
{
    const bool hasSelection = m_sets->currentItem() != nullptr;
    m_delete->setEnabled(hasSelection && !m_running);
    m_import->setEnabled(!m_running && !m_pluginId.isEmpty());
    m_analyse->setEnabled(!m_running && !m_source.isEmpty());
    const bool audio = isAudioKind();
    m_playSource->setVisible(audio);
    m_playSet->setVisible(audio);
    m_playSource->setEnabled(!m_source.isEmpty());
    m_playSet->setEnabled(hasSelection && !m_sets->currentItem()->data(0, kSourceRole).toString().isEmpty());
}

bool PluginSetPanel::isAudioKind() const
{
    // The plugin says what it collects: either by naming the kind, or — for a
    // plugin with a single kind — by the file filter it asked the dialog for.
    return m_kind == QLatin1String("audio") || m_sourceFilter.contains(QLatin1String("*.mp3"), Qt::CaseInsensitive)
        || m_sourceFilter.contains(QLatin1String("*.wav"), Qt::CaseInsensitive);
}

void PluginSetPanel::togglePlay(const QString &path)
{
    if (path.isEmpty() || !QFileInfo::exists(path)) {
        showStatus(i18n("The track this preset was made from is gone."), true);
        return;
    }
    if (m_player && m_playing == path && m_player->playbackState() == QMediaPlayer::PlayingState) {
        stopPlayback();
        return;
    }
    if (!m_player) {
        m_player = new QMediaPlayer(this);
        m_audioOut = new QAudioOutput(this);
        m_player->setAudioOutput(m_audioOut);
        connect(m_player, &QMediaPlayer::playbackStateChanged, this, [this](QMediaPlayer::PlaybackState state) {
            if (state != QMediaPlayer::PlayingState) {
                m_playing.clear();
            }
            const QString icon = state == QMediaPlayer::PlayingState ? QStringLiteral("media-playback-stop") : QStringLiteral("media-playback-start");
            m_playSource->setIcon(QIcon::fromTheme(icon));
            m_playSet->setIcon(QIcon::fromTheme(icon));
        });
    }
    m_playing = path;
    m_player->setSource(QUrl::fromLocalFile(path));
    m_player->play();
}

void PluginSetPanel::stopPlayback()
{
    if (m_player) {
        m_player->stop();
    }
    m_playing.clear();
}

void PluginSetPanel::showStatus(const QString &message, bool error)
{
    m_status->setText(message);
    m_status->setMessageType(error ? KMessageWidget::Warning : KMessageWidget::Information);
    m_status->show();
}

void PluginSetPanel::deleteSet()
{
    QTreeWidgetItem *item = m_sets->currentItem();
    if (item == nullptr) {
        return;
    }
    const QString file = item->data(0, kFileRole).toString();
    if (KMessageBox::warningContinueCancel(this, i18n("Delete the set <b>%1</b>? This cannot be undone.", item->text(0))) != KMessageBox::Continue) {
        return;
    }
    if (PluginSets::remove(file)) {
        refresh();
        Q_EMIT setsChanged();
    }
}

void PluginSetPanel::renameSet()
{
    QTreeWidgetItem *item = m_sets->currentItem();
    if (item == nullptr) {
        return;
    }
    bool accepted = false;
    const QString name = QInputDialog::getText(this, i18n("Rename set"), i18n("Name:"), QLineEdit::Normal, item->text(0), &accepted).trimmed();
    if (!accepted || name.isEmpty() || name == item->text(0)) {
        return;
    }
    QString error;
    if (!PluginSets::rename(item->data(0, kFileRole).toString(), name, &error)) {
        showStatus(error.isEmpty() ? i18n("The set could not be renamed.") : error, true);
        return;
    }
    refresh();
    Q_EMIT setsChanged();
}

void PluginSetPanel::importSet()
{
    const QString path = QFileDialog::getOpenFileName(this, i18n("Import a set"), QString(), i18n("Sets (*.json);;All files (*)"));
    if (path.isEmpty()) {
        return;
    }
    QString error;
    if (PluginSets::importSet(m_pluginId, path, &error).isEmpty()) {
        showStatus(error, true);
        return;
    }
    refresh();
    Q_EMIT setsChanged();
}

void PluginSetPanel::exportSet()
{
    QTreeWidgetItem *item = m_sets->currentItem();
    if (item == nullptr) {
        return;
    }
    const QString destination =
        QFileDialog::getSaveFileName(this, i18n("Export set"), item->text(0) + QStringLiteral(".json"), i18n("Sets (*.json)"));
    if (destination.isEmpty()) {
        return;
    }
    QString error;
    if (!PluginSets::exportSet(item->data(0, kFileRole).toString(), destination, &error)) {
        showStatus(error, true);
    }
}
