/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "generatorcard.h"

#include "plugins/pluginmanager.h"
#include "plugins/pluginsetstore.h"

#include <KLocalizedString>

#include <QAudioOutput>
#include <QButtonGroup>
#include <QCheckBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMediaPlayer>
#include <QPainter>
#include <QPainterPath>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QResizeEvent>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>

namespace {
/** @brief Strings from a manifest go through the application's catalog, like
 *  the names of a plugin's effects do: the first-party plugins are translated
 *  there, and a third-party one simply reads as written. */
QString tr8(const QString &text)
{
    return text.isEmpty() ? text : i18n(text.toUtf8().constData());
}

/** @brief A round picture for a preset: its own thumbnail when it has one,
 *  otherwise its initials on a colour picked from its name, so the same voice
 *  always wears the same colour. */
QPixmap avatar(const PluginSets::Set &set, int size)
{
    static const QList<QColor> palette = {QColor(0xC8, 0xED, 0xD2), QColor(0xF7, 0xD9, 0xA8), QColor(0xBF, 0xD9, 0xF2), QColor(0xE7, 0xC6, 0xEC),
                                          QColor(0xF2, 0xC3, 0xB5), QColor(0xCD, 0xE7, 0xB0), QColor(0xB9, 0xE4, 0xE0), QColor(0xE9, 0xE1, 0xA6)};
    const qreal dpr = 2.0;
    QPixmap pix(int(size * dpr), int(size * dpr));
    pix.setDevicePixelRatio(dpr);
    pix.fill(Qt::transparent);
    QPainter p(&pix);
    p.setRenderHint(QPainter::Antialiasing);
    QPainterPath circle;
    circle.addEllipse(0, 0, size, size);
    p.setClipPath(circle);
    const QImage thumb(set.thumb);
    if (!thumb.isNull()) {
        p.drawImage(QRect(0, 0, size, size), thumb.scaled(int(size * dpr), int(size * dpr), Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation));
        return pix;
    }
    p.fillRect(QRect(0, 0, size, size), palette.at(int(qHash(set.name) % uint(palette.size()))));
    QString letters;
    const QStringList words = set.name.split(QLatin1Char(' '), Qt::SkipEmptyParts);
    for (const QString &word : words) {
        if (letters.size() < 2 && !word.isEmpty() && word.at(0).isLetter()) {
            letters += word.at(0).toUpper();
        }
    }
    QFont font = p.font();
    font.setPixelSize(int(size * 0.4));
    font.setBold(true);
    p.setFont(font);
    p.setPen(QColor(0x10, 0x26, 0x1A));
    p.drawText(QRect(0, 0, size, size), Qt::AlignCenter, letters);
    return pix;
}

QString duration(int seconds)
{
    return seconds > 0 ? QStringLiteral("%1:%2").arg(seconds / 60).arg(seconds % 60, 2, 10, QLatin1Char('0')) : QString();
}

/** @brief The recording a preset was made from, as it is kept beside it. */
QString soundOf(const PluginSets::Set &set)
{
    const QString kept = QFileInfo(set.file).absolutePath() + QLatin1Char('/') + QFileInfo(set.file).completeBaseName() + QStringLiteral(".wav");
    return QFileInfo::exists(kept) ? kept : set.source;
}
} // namespace

GeneratorCard::GeneratorCard(const QString &cardId, const PluginManifest &manifest, const PluginGenerator &generator, const QJsonObject &payload,
                             QWidget *parent)
    : QFrame(parent)
    , m_cardId(cardId)
    , m_manifest(manifest)
    , m_generator(generator)
    , m_payload(payload)
{
    setObjectName(QStringLiteral("chatToolCard"));
    auto *layout = new QVBoxLayout(this);
    layout->setContentsMargins(12, 10, 12, 12);
    layout->setSpacing(8);

    auto *header = new QHBoxLayout;
    header->setSpacing(8);
    auto *icon = new QLabel(this);
    icon->setPixmap(manifest.icon().pixmap(16, 16));
    auto *title = new QLabel(tr8(generator.title), this);
    QFont titleFont = title->font();
    titleFont.setWeight(QFont::DemiBold);
    title->setFont(titleFont);
    m_byLabel = new QLabel(i18n("filled in by the assistant"), this);
    m_byLabel->setObjectName(QStringLiteral("chatToolStatus"));
    m_fold = new QToolButton(this);
    m_fold->setAutoRaise(true);
    m_fold->setToolTip(i18n("Collapse"));
    header->addWidget(icon);
    header->addWidget(title);
    header->addWidget(m_byLabel);
    header->addStretch();
    header->addWidget(m_fold);
    layout->addLayout(header);

    m_summary = new QLabel(this);
    m_summary->setObjectName(QStringLiteral("chatToolStatus"));
    m_summary->setWordWrap(true);
    layout->addWidget(m_summary);

    m_body = new QWidget(this);
    m_form = new QVBoxLayout(m_body);
    m_form->setContentsMargins(0, 0, 0, 0);
    m_form->setSpacing(8);
    layout->addWidget(m_body);
    buildForm();
    m_building = false;

    connect(m_fold, &QToolButton::clicked, this, [this]() {
        m_payload.insert(QStringLiteral("collapsed"), !m_payload.value(QStringLiteral("collapsed")).toBool());
        refreshHeader();
        Q_EMIT edited(m_cardId, m_payload);
    });
    connect(&PluginManager::instance(), &PluginManager::setsChanged, this, [this](const QString &pluginId) {
        if (pluginId == m_manifest.id()) {
            refreshLibrary();
        }
    });
    refreshHeader();
    refreshVisibility();
    refreshFoot();
}

GeneratorCard::~GeneratorCard()
{
    if (m_popup) {
        m_popup->deleteLater();
    }
}

void GeneratorCard::buildForm()
{
    const QJsonObject values = m_payload.value(QStringLiteral("values")).toObject();
    const QList<PluginField> fields = m_generator.fields;
    for (const PluginField &field : fields) {
        FieldWidgets w;
        auto *row = new QWidget(m_body);
        auto *rowLayout = new QVBoxLayout(row);
        rowLayout->setContentsMargins(0, 0, 0, 0);
        rowLayout->setSpacing(4);
        w.row = row;
        if (!field.label.isEmpty() && field.type != QLatin1String("set")) {
            auto *label = new QLabel(tr8(field.label), row);
            label->setObjectName(QStringLiteral("chatToolStatus"));
            rowLayout->addWidget(label);
        }
        const QJsonValue current = values.contains(field.key) ? values.value(field.key) : QJsonValue::fromVariant(field.defaultValue);
        if (!values.contains(field.key) && !current.isNull() && !current.isUndefined()) {
            setValue(field.key, current);
        }
        if (field.type == QLatin1String("text")) {
            auto *edit = new QPlainTextEdit(row);
            edit->setPlaceholderText(tr8(field.placeholder));
            edit->setPlainText(current.toString());
            edit->setTabChangesFocus(true);
            // the text wraps, so there is never anything to scroll sideways to
            edit->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
            const int lines = field.compact ? 2 : 5;
            edit->setFixedHeight(edit->fontMetrics().lineSpacing() * lines + 14);
            connect(edit, &QPlainTextEdit::textChanged, this, [this, key = field.key, edit]() {
                if (!m_applying) {
                    setValue(key, edit->toPlainText());
                }
            });
            rowLayout->addWidget(edit);
            w.input = edit;
        } else if (field.type == QLatin1String("enum")) {
            // A handful of choices reads as tabs: every option in view, one tap away.
            auto *tabs = new QWidget(row);
            tabs->setObjectName(QStringLiteral("chatGeneratorTabs"));
            auto *tabsLayout = new QHBoxLayout(tabs);
            tabsLayout->setContentsMargins(2, 2, 2, 2);
            tabsLayout->setSpacing(2);
            auto *group = new QButtonGroup(tabs);
            group->setExclusive(true);
            for (int i = 0; i < field.options.size(); ++i) {
                auto *tab = new QToolButton(tabs);
                tab->setCheckable(true);
                tab->setText(tr8(field.labels.value(i, field.options.at(i))));
                tab->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
                tab->setChecked(current.toString() == field.options.at(i));
                group->addButton(tab, i);
                tabsLayout->addWidget(tab);
                w.choices << tab;
            }
            connect(group, &QButtonGroup::idClicked, this, [this, key = field.key, options = field.options](int id) {
                if (!m_applying && id >= 0 && id < options.size()) {
                    setValue(key, options.at(id));
                    refreshVisibility();
                }
            });
            rowLayout->addWidget(tabs);
            w.input = tabs;
        } else if (field.type == QLatin1String("set")) {
            m_setKey = field.key;
            m_setKind = field.kind;
            rowLayout->addWidget(buildLibrary(field));
            w.input = m_libraryLine;
        } else if (field.type == QLatin1String("bool")) {
            auto *box = new QCheckBox(tr8(field.label), row);
            box->setChecked(current.toBool());
            connect(box, &QCheckBox::toggled, this, [this, key = field.key](bool on) {
                if (!m_applying) {
                    setValue(key, on);
                }
            });
            rowLayout->addWidget(box);
            w.input = box;
        } else if (field.type == QLatin1String("number")) {
            auto *spin = new QDoubleSpinBox(row);
            spin->setRange(-1e9, 1e9);
            spin->setValue(current.toDouble());
            connect(spin, &QDoubleSpinBox::valueChanged, this, [this, key = field.key](double v) {
                if (!m_applying) {
                    setValue(key, v);
                }
            });
            rowLayout->addWidget(spin);
            w.input = spin;
        } else {
            auto *edit = new QLineEdit(current.toString(), row);
            edit->setPlaceholderText(tr8(field.placeholder));
            connect(edit, &QLineEdit::textChanged, this, [this, key = field.key](const QString &text) {
                if (!m_applying) {
                    setValue(key, text);
                }
            });
            rowLayout->addWidget(edit);
            w.input = edit;
        }
        m_form->addWidget(row);
        m_fields.insert(field.key, w);
    }

    auto *foot = new QHBoxLayout;
    foot->addStretch();
    m_generate = new QPushButton(i18nc("@action:button make what the card describes", "Generate"), m_body);
    m_generate->setObjectName(QStringLiteral("chatGenerateButton"));
    m_generate->setCursor(Qt::PointingHandCursor);
    connect(m_generate, &QPushButton::clicked, this, [this]() { Q_EMIT generateRequested(m_cardId); });
    foot->addWidget(m_generate);
    m_form->addLayout(foot);
}

QWidget *GeneratorCard::buildLibrary(const PluginField &field)
{
    Q_UNUSED(field)
    auto *holder = new QWidget(m_body);
    auto *layout = new QVBoxLayout(holder);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    m_libraryButton = new QPushButton(holder);
    m_libraryButton->setObjectName(QStringLiteral("chatLibraryButton"));
    m_libraryButton->setIconSize(QSize(28, 28));
    m_libraryButton->setCursor(Qt::PointingHandCursor);
    connect(m_libraryButton, &QPushButton::clicked, this, &GeneratorCard::openLibrary);

    // An empty library has one thing to offer, and it is the same button the
    // list ends with.
    m_libraryEmpty = new QPushButton(QIcon::fromTheme(QStringLiteral("list-add")), i18n("Upload voice"), holder);
    m_libraryEmpty->setObjectName(QStringLiteral("chatLibraryButton"));
    connect(m_libraryEmpty, &QPushButton::clicked, this, &GeneratorCard::uploadVoice);

    layout->addWidget(m_libraryButton);
    layout->addWidget(m_libraryEmpty);
    m_libraryLine = holder;
    refreshLibrary();
    return holder;
}

void GeneratorCard::refreshLibrary()
{
    if (m_libraryButton == nullptr) {
        return;
    }
    const QVector<PluginSets::Set> sets = PluginSets::sets(m_manifest.id(), m_setKind);
    QString chosen = value(m_setKey).toString();
    if (!m_pendingSource.isEmpty()) {
        for (const PluginSets::Set &set : sets) {
            if (set.source == m_pendingSource) {
                chosen = set.file;
                m_pendingSource.clear();
                setValue(m_setKey, chosen, false);
                break;
            }
        }
    }
    const PluginSets::Set *current = nullptr;
    for (const PluginSets::Set &set : sets) {
        if (set.file == chosen) {
            current = &set;
        }
    }
    if (current == nullptr && !sets.isEmpty()) {
        current = &sets.first();
        setValue(m_setKey, current->file, false);
    } else if (current == nullptr && !chosen.isEmpty()) {
        setValue(m_setKey, QString(), false);
    }
    m_libraryEmpty->setVisible(sets.isEmpty());
    m_libraryButton->setVisible(!sets.isEmpty());
    if (current) {
        m_libraryButton->setIcon(QIcon(avatar(*current, 28)));
        m_currentName = current->name;
        m_currentLength = duration(current->count);
        updateLibraryButtonText();
        m_libraryButton->setToolTip(current->name);
    }
    if (m_popup && m_popup->isVisible()) {
        openLibrary(); // redraw the open list in place
    }
    refreshFoot();
}

void GeneratorCard::updateLibraryButtonText()
{
    if (m_libraryButton == nullptr || m_currentName.isEmpty()) {
        return;
    }
    const QString name = fontMetrics().elidedText(m_currentName, Qt::ElideRight, qMax(80, m_libraryLine->width() - 110));
    m_libraryButton->setText(m_currentLength.isEmpty() ? name : QStringLiteral("%1   %2").arg(name, m_currentLength));
}

void GeneratorCard::resizeEvent(QResizeEvent *event)
{
    QFrame::resizeEvent(event);
    updateLibraryButtonText();
}

void GeneratorCard::openLibrary()
{
    const QVector<PluginSets::Set> sets = PluginSets::sets(m_manifest.id(), m_setKind);
    const bool reopen = m_popup && m_popup->isVisible();
    const QString query = m_search ? m_search->text() : QString();
    if (!m_popup) {
        // Over the card rather than inside it: opening the list moves nothing.
        m_popup = new QFrame(this, Qt::Popup);
        m_popup->setObjectName(QStringLiteral("chatLibraryPopup"));
        auto *layout = new QVBoxLayout(m_popup);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(0);
        m_search = new QLineEdit(m_popup);
        m_search->setObjectName(QStringLiteral("chatSearchField"));
        m_search->setPlaceholderText(i18n("Find a voice"));
        m_search->setClearButtonEnabled(true);
        m_list = new QListWidget(m_popup);
        m_list->setObjectName(QStringLiteral("chatLibraryList"));
        m_list->setSelectionMode(QAbstractItemView::NoSelection);
        m_list->setFrameShape(QFrame::NoFrame);
        m_list->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        auto *upload = new QPushButton(QIcon::fromTheme(QStringLiteral("list-add")), i18n("Upload voice"), m_popup);
        upload->setObjectName(QStringLiteral("chatLibraryUpload"));
        upload->setFlat(true);
        connect(upload, &QPushButton::clicked, this, [this]() {
            m_popup->hide();
            uploadVoice();
        });
        layout->addWidget(m_search);
        layout->addWidget(m_list);
        layout->addWidget(upload);
        connect(m_search, &QLineEdit::textChanged, this, [this]() { openLibrary(); });
    }
    m_search->setVisible(sets.size() > 6);
    m_list->clear();
    const QString chosen = value(m_setKey).toString();
    const QString needle = query.trimmed().toLower();
    for (const PluginSets::Set &set : sets) {
        if (!needle.isEmpty() && !set.name.toLower().contains(needle)) {
            continue;
        }
        auto *item = new QListWidgetItem(m_list);
        auto *row = new QWidget(m_list);
        auto *h = new QHBoxLayout(row);
        h->setContentsMargins(8, 4, 6, 4);
        h->setSpacing(8);
        // a long name is cut to the row and shown whole on hover, never
        // scrolled sideways
        const int nameWidth = qMax(80, m_libraryLine->width() - 140);
        auto *pick = new QPushButton(QIcon(avatar(set, 26)), fontMetrics().elidedText(set.name, Qt::ElideRight, nameWidth), row);
        pick->setToolTip(set.name);
        pick->setObjectName(QStringLiteral("chatLibraryRow"));
        pick->setIconSize(QSize(26, 26));
        pick->setFlat(true);
        pick->setCheckable(true);
        pick->setChecked(set.file == chosen);
        pick->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        auto *length = new QLabel(duration(set.count), row);
        length->setObjectName(QStringLiteral("chatToolStatus"));
        auto *play = new QToolButton(row);
        play->setAutoRaise(true);
        play->setIcon(QIcon::fromTheme(m_playing == soundOf(set) ? QStringLiteral("media-playback-stop") : QStringLiteral("media-playback-start")));
        play->setToolTip(i18n("Listen"));
        auto *remove = new QToolButton(row);
        remove->setAutoRaise(true);
        remove->setIcon(QIcon::fromTheme(QStringLiteral("edit-delete")));
        remove->setToolTip(m_armedDelete == set.file ? i18n("Click again to delete") : i18n("Delete"));
        remove->setProperty("armed", m_armedDelete == set.file);
        h->addWidget(pick, 1);
        h->addWidget(length);
        h->addWidget(play);
        h->addWidget(remove);
        item->setSizeHint(row->sizeHint());
        m_list->setItemWidget(item, row);
        const QString file = set.file;
        const QString sound = soundOf(set);
        // A row's buttons rebuild the list they sit in, so the rebuild waits
        // for the click to finish: the button must not be deleted under itself.
        connect(pick, &QPushButton::clicked, this, [this, file]() {
            setValue(m_setKey, file);
            m_popup->hide();
            QTimer::singleShot(0, this, &GeneratorCard::refreshLibrary);
        });
        connect(play, &QToolButton::clicked, this, [this, sound]() {
            togglePlay(sound);
            QTimer::singleShot(0, this, &GeneratorCard::openLibrary);
        });
        connect(remove, &QToolButton::clicked, this, [this, file]() {
            if (m_armedDelete != file) {
                // the first click arms the button; a moment later it is safe again
                m_armedDelete = file;
                QTimer::singleShot(3000, this, [this, file]() {
                    if (m_armedDelete == file) {
                        m_armedDelete.clear();
                        if (m_popup && m_popup->isVisible()) {
                            openLibrary();
                        }
                    }
                });
                QTimer::singleShot(0, this, &GeneratorCard::openLibrary);
                return;
            }
            m_armedDelete.clear();
            QTimer::singleShot(0, this, [this, file]() {
                PluginSets::remove(file);
                PluginManager::instance().noteSetsChanged(m_manifest.id());
            });
        });
    }
    const int rows = qMin(6, qMax(1, m_list->count()));
    const int rowHeight = m_list->count() > 0 ? m_list->sizeHintForRow(0) : 36;
    m_list->setFixedHeight(rows * rowHeight + 4);
    m_popup->setFixedWidth(m_libraryLine->width());
    if (!reopen) {
        m_popup->move(m_libraryLine->mapToGlobal(QPoint(0, m_libraryLine->height() + 4)));
        m_popup->show();
        if (m_search->isVisible()) {
            m_search->setFocus();
        }
    }
    m_popup->adjustSize();
}

void GeneratorCard::uploadVoice()
{
    const QString filter = m_manifest.setsUi(m_setKind).filter;
    const QString file = QFileDialog::getOpenFileName(this, i18n("Upload voice"), QString(),
                                                      filter.isEmpty() ? i18n("Audio (*.wav *.mp3 *.m4a *.aac *.ogg *.flac)") : filter);
    if (file.isEmpty()) {
        return;
    }
    m_pendingSource = file;
    PluginManager::instance().registerSet(m_manifest.id(), file, m_setKind);
}

void GeneratorCard::togglePlay(const QString &path)
{
    if (!m_player) {
        m_player = new QMediaPlayer(this);
        m_audio = new QAudioOutput(this);
        m_player->setAudioOutput(m_audio);
        connect(m_player, &QMediaPlayer::playbackStateChanged, this, [this](QMediaPlayer::PlaybackState state) {
            if (state != QMediaPlayer::PlayingState) {
                m_playing.clear();
                if (m_popup && m_popup->isVisible()) {
                    openLibrary();
                }
            }
        });
    }
    if (m_playing == path) {
        m_player->stop();
        m_playing.clear();
        return;
    }
    m_playing = path;
    m_player->setSource(QUrl::fromLocalFile(path));
    m_player->play();
}

void GeneratorCard::setValue(const QString &key, const QJsonValue &v, bool byUser)
{
    QJsonObject values = m_payload.value(QStringLiteral("values")).toObject();
    if (values.value(key) == v) {
        return;
    }
    values.insert(key, v);
    m_payload.insert(QStringLiteral("values"), values);
    if (m_building) {
        return; // a default, not an edit
    }
    // what the user changes is theirs, whoever filled the card in first
    if (byUser && !m_applying) {
        m_payload.insert(QStringLiteral("author"), QStringLiteral("user"));
    }
    refreshHeader();
    refreshFoot();
    Q_EMIT edited(m_cardId, m_payload);
}

QJsonValue GeneratorCard::value(const QString &key) const
{
    return m_payload.value(QStringLiteral("values")).toObject().value(key);
}

void GeneratorCard::refreshVisibility()
{
    const QList<PluginField> fields = m_generator.fields;
    for (const PluginField &field : fields) {
        const FieldWidgets w = m_fields.value(field.key);
        if (w.row && !field.showIfKey.isEmpty()) {
            w.row->setVisible(value(field.showIfKey).toVariant().toString() == field.showIfValue);
        }
    }
    refreshFoot();
}

QString GeneratorCard::missing() const
{
    const QList<PluginField> fields = m_generator.fields;
    for (const PluginField &field : fields) {
        if (!field.showIfKey.isEmpty() && value(field.showIfKey).toVariant().toString() != field.showIfValue) {
            continue;
        }
        if ((field.type == QLatin1String("text") || field.type == QLatin1String("set")) && value(field.key).toString().trimmed().isEmpty()) {
            return field.type == QLatin1String("set") ? i18n("Upload a voice first") : i18n("Fill in %1", tr8(field.label).toLower());
        }
    }
    return {};
}

void GeneratorCard::refreshFoot()
{
    if (m_generate) {
        const QString why = missing();
        m_generate->setEnabled(why.isEmpty());
        m_generate->setToolTip(why);
    }
}

void GeneratorCard::refreshHeader()
{
    const bool collapsed = m_payload.value(QStringLiteral("collapsed")).toBool();
    m_body->setVisible(!collapsed);
    m_fold->setArrowType(collapsed ? Qt::RightArrow : Qt::DownArrow);
    m_fold->setToolTip(collapsed ? i18n("Expand") : i18n("Collapse"));
    m_byLabel->setVisible(m_payload.value(QStringLiteral("author")).toString() == QLatin1String("assistant"));
    // Folded, the card is one line of what it says: the start of its first
    // text field.
    QString first;
    const QList<PluginField> fields = m_generator.fields;
    for (const PluginField &field : fields) {
        if (field.type == QLatin1String("text")) {
            first = value(field.key).toString().section(QLatin1Char('\n'), 0, 0).trimmed();
            break;
        }
    }
    if (first.size() > 70) {
        first = first.left(70).trimmed() + QStringLiteral("…");
    }
    m_summary->setText(first);
    m_summary->setVisible(collapsed && !first.isEmpty());
}

void GeneratorCard::setPayload(const QJsonObject &payload)
{
    m_payload = payload;
    m_applying = true;
    const QJsonObject values = payload.value(QStringLiteral("values")).toObject();
    const QList<PluginField> fields = m_generator.fields;
    for (const PluginField &field : fields) {
        const FieldWidgets w = m_fields.value(field.key);
        const QJsonValue v = values.value(field.key);
        if (auto *edit = qobject_cast<QPlainTextEdit *>(w.input)) {
            if (edit->toPlainText() != v.toString()) {
                edit->setPlainText(v.toString());
            }
        } else if (auto *line = qobject_cast<QLineEdit *>(w.input)) {
            line->setText(v.toString());
        } else if (auto *box = qobject_cast<QCheckBox *>(w.input)) {
            box->setChecked(v.toBool());
        } else if (auto *spin = qobject_cast<QDoubleSpinBox *>(w.input)) {
            spin->setValue(v.toDouble());
        }
        for (int i = 0; i < w.choices.size(); ++i) {
            w.choices.at(i)->setChecked(field.options.value(i) == v.toString());
        }
    }
    m_applying = false;
    refreshLibrary();
    refreshHeader();
    refreshVisibility();
}

void GeneratorCard::focusFirstField()
{
    for (const PluginField &field : m_generator.fields) {
        if (QWidget *input = m_fields.value(field.key).input) {
            if (input->isVisible() && (qobject_cast<QPlainTextEdit *>(input) || qobject_cast<QLineEdit *>(input))) {
                input->setFocus();
                return;
            }
        }
    }
}
