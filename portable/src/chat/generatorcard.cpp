/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "generatorcard.h"

#include "bin/projectclip.h"
#include "bin/projectitemmodel.h"
#include "core.h"
#include "doc/wunjodoc.h"
#include "mainwindow.h"
#include "monitor/monitor.h"
#include "plugins/plugineffects.h"
#include "plugins/pluginmanager.h"
#include "plugins/pluginsetstore.h"
#include "timeline2/model/timelineitemmodel.hpp"
#include "timeline2/view/timelinewidget.h"

#include <KLocalizedString>

#include <QAudioOutput>
#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QDateTime>
#include <QDir>
#include <QDoubleSpinBox>
#include <QMenu>
#include <QSlider>
#include <QStandardPaths>
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
#include <QRegularExpression>
#include <QResizeEvent>
#include <QScrollArea>
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
        } else if (field.type == QLatin1String("enum") && field.options.size() >= 5) {
            // Twenty voices do not fit in a row of tabs: past a handful, a list.
            auto *combo = new QComboBox(row);
            for (int i = 0; i < field.options.size(); ++i) {
                combo->addItem(tr8(field.labels.value(i, field.options.at(i))), field.options.at(i));
            }
            combo->setCurrentIndex(qMax(0, combo->findData(current.toString())));
            connect(combo, &QComboBox::currentIndexChanged, this, [this, key = field.key, combo](int) {
                if (!m_applying) {
                    setValue(key, combo->currentData().toString());
                    refreshVisibility();
                }
            });
            rowLayout->addWidget(combo);
            w.input = combo;
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
        } else if (field.type == QLatin1String("media")) {
            rowLayout->addWidget(buildMedia(field, w));
        } else if (field.type == QLatin1String("media_list")) {
            rowLayout->addWidget(buildMediaList(field));
            w.input = m_lists.value(field.key).items;
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
        } else if (field.type == QLatin1String("number") && field.max > field.min) {
            // A range reads as a slider: both ends in view, and nothing outside
            // them can be typed. The slider counts steps, the value is worked out.
            const double step = field.step > 0 ? field.step : 1;
            const int steps = qMax(1, qRound((field.max - field.min) / step));
            const int decimals = step >= 1 ? 0 : (step >= 0.1 ? 1 : 2);
            auto *holder = new QWidget(row);
            auto *line = new QHBoxLayout(holder);
            line->setContentsMargins(0, 0, 0, 0);
            auto *slider = new QSlider(Qt::Horizontal, holder);
            slider->setRange(0, steps);
            slider->setValue(qBound(0, qRound((current.toDouble() - field.min) / step), steps));
            auto *shown = new QLabel(holder);
            shown->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
            shown->setMinimumWidth(shown->fontMetrics().horizontalAdvance(QStringLiteral("000.00")));
            shown->setText(QString::number(field.min + slider->value() * step, 'f', decimals));
            connect(slider, &QSlider::valueChanged, this, [this, key = field.key, min = field.min, step, decimals, shown](int index) {
                const double v = qRound((min + index * step) * 1000.0) / 1000.0;
                shown->setText(QString::number(v, 'f', decimals));
                if (!m_applying) {
                    setValue(key, v);
                }
            });
            line->addWidget(slider, 1);
            line->addWidget(shown);
            rowLayout->addWidget(holder);
            w.input = slider;
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
        if (field.type == QLatin1String("media")) {
            refreshMedia(field.key);
        } else if (field.type == QLatin1String("media_list")) {
            refreshMediaList(field.key);
        }
    }

    m_answerLabel = new QLabel(m_body);
    m_answerLabel->setObjectName(QStringLiteral("chatToolStatus"));
    m_answerLabel->setWordWrap(true);
    // "top up at https://…" is meant to be followed
    m_answerLabel->setTextFormat(Qt::RichText);
    m_answerLabel->setOpenExternalLinks(true);
    m_answerLabel->setVisible(false);
    m_form->addWidget(m_answerLabel);

    // A card with a gate goes in two steps: its gate action first (Calculate),
    // then Cancel or Generate for what it answered. Other actions stay aside.
    auto *foot = new QHBoxLayout;
    for (const PluginAction &action : std::as_const(m_generator.actions)) {
        auto *button = new QPushButton(tr8(action.label), m_body);
        button->setCursor(Qt::PointingHandCursor);
        connect(button, &QPushButton::clicked, this, [this, id = action.id]() { Q_EMIT actionRequested(m_cardId, id); });
        if (action.gate && m_gateId.isEmpty()) {
            m_gateId = action.id;
            m_gateButton = button;
            button->setObjectName(QStringLiteral("chatGenerateButton"));
            continue;
        }
        foot->addWidget(button);
        m_actionButtons << button;
    }
    foot->addStretch();
    if (m_gateButton) {
        m_cancel = new QPushButton(i18nc("@action:button drop the calculated price", "Cancel"), m_body);
        m_cancel->setCursor(Qt::PointingHandCursor);
        connect(m_cancel, &QPushButton::clicked, this, [this]() {
            clearAnswer();
            Q_EMIT priceDropped(m_cardId);
        });
        foot->addWidget(m_cancel);
        foot->addWidget(m_gateButton);
    }
    m_generate = new QPushButton(i18nc("@action:button make what the card describes", "Generate"), m_body);
    m_generate->setObjectName(QStringLiteral("chatGenerateButton"));
    m_generate->setCursor(Qt::PointingHandCursor);
    connect(m_generate, &QPushButton::clicked, this, [this]() { Q_EMIT generateRequested(m_cardId); });
    foot->addWidget(m_generate);
    m_form->addLayout(foot);
}

QWidget *GeneratorCard::buildMedia(const PluginField &field, FieldWidgets &w)
{
    // A slot on the card: what is in it, and the ways to put something there.
    // Only the path is kept, the file stays where it is.
    auto *slot = new QWidget(m_body);
    auto *layout = new QHBoxLayout(slot);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);
    w.thumb = new QLabel(slot);
    w.thumb->setFixedSize(64, 40);
    w.thumb->setAlignment(Qt::AlignCenter);
    w.thumb->setObjectName(QStringLiteral("chatAttachThumb"));
    w.name = new QLabel(slot);
    w.name->setObjectName(QStringLiteral("chatToolStatus"));
    w.name->setTextInteractionFlags(Qt::NoTextInteraction);
    layout->addWidget(w.thumb);
    layout->addWidget(w.name, 1);
    auto tool = [slot, layout](const QString &icon, const QString &tip) {
        auto *button = new QToolButton(slot);
        button->setIcon(QIcon::fromTheme(icon));
        button->setToolTip(tip);
        button->setAutoRaise(true);
        layout->addWidget(button);
        return button;
    };
    auto *add = tool(field.accept == QLatin1String("video") ? QStringLiteral("video-x-generic") : QStringLiteral("insert-image"), tr8(field.label));
    offerSources(add, field.accept, tr8(field.label), [this, key = field.key](const QJsonValue &got) {
        setValue(key, got);
        refreshMedia(key);
    });
    connect(tool(QStringLiteral("edit-clear"), i18n("Remove")), &QToolButton::clicked, this, [this, key = field.key]() {
        setValue(key, QJsonValue());
        refreshMedia(key);
    });
    w.input = slot;
    return slot;
}

void GeneratorCard::offerSources(QToolButton *button, const QString &accept, const QString &title, const std::function<void(const QJsonValue &)> &take)
{
    // one button, its ways in a menu: a file, or what the editor shows now
    auto *menu = new QMenu(button);
    connect(menu->addAction(QIcon::fromTheme(QStringLiteral("document-open")), i18n("Choose a file")), &QAction::triggered, this,
            [this, accept, title, take]() {
                const QJsonValue got = chooseFile(accept, title);
                if (!got.isNull()) {
                    take(got);
                }
            });
    if (accept == QLatin1String("video")) {
        connect(menu->addAction(QIcon::fromTheme(QStringLiteral("video-x-generic")), i18n("Selected clip on the timeline")), &QAction::triggered,
                this, [this, take]() {
                    const QJsonValue got = selectedClip();
                    if (!got.isNull()) {
                        take(got);
                    }
                });
    } else {
        connect(menu->addAction(QIcon::fromTheme(QStringLiteral("camera-photo")), i18n("Current frame of the monitor")), &QAction::triggered,
                this, [this, take]() {
                    const QJsonValue got = monitorFrame();
                    if (!got.isNull()) {
                        take(got);
                    }
                });
    }
    button->setMenu(menu);
    button->setPopupMode(QToolButton::InstantPopup);
}

QJsonValue GeneratorCard::chooseFile(const QString &accept, const QString &title)
{
    const bool video = accept == QLatin1String("video");
    const QString filter = video ? i18n("Videos (*.mp4 *.mov *.mkv *.webm *.avi)") : i18n("Images (*.png *.jpg *.jpeg *.webp *.bmp)");
    const QString file = QFileDialog::getOpenFileName(this, title, QString(), filter);
    return file.isEmpty() ? QJsonValue() : QJsonValue(file);
}

QJsonValue GeneratorCard::monitorFrame()
{
    Monitor *monitor = pCore->getMonitor(Wunjo::ProjectMonitor);
    if (!monitor || !pCore->currentDoc()) {
        return {};
    }
    // into the project, beside what the plugins make: a frame is only worth
    // something to the card it was taken for, but it must outlive the run
    QString folder = pCore->currentDoc()->projectDataFolder();
    if (folder.isEmpty()) {
        folder = QStandardPaths::writableLocation(QStandardPaths::TempLocation);
    }
    folder += QStringLiteral("/plugin-frames/") + m_manifest.id();
    QDir().mkpath(folder);
    const QString path = folder + QStringLiteral("/frame-%1.png").arg(QDateTime::currentDateTime().toString(QStringLiteral("yyyyMMdd-HHmmss-zzz")));
    monitor->extractFrame(path);
    if (!QFileInfo::exists(path)) {
        pCore->displayMessage(i18n("The frame could not be saved."), ErrorMessage);
        return {};
    }
    return path;
}

QJsonValue GeneratorCard::selectedClip()
{
    TimelineWidget *timeline = pCore->window()->getCurrentTimeline();
    if (!timeline || !timeline->model()) {
        return {};
    }
    const auto model = timeline->model();
    for (int id : model->getCurrentSelection()) {
        if (!model->isClip(id) || model->isAudioTrack(model->getClipTrackId(id))) {
            continue;
        }
        std::shared_ptr<ProjectClip> clip = pCore->projectItemModel()->getClipByBinID(model->getClipBinId(id));
        if (!clip) {
            continue;
        }
        // the part of the source that is on the timeline, in project frames
        const int in = model->getClipIn(id);
        QJsonObject value;
        value.insert(QStringLiteral("path"), clip->url());
        value.insert(QStringLiteral("in"), in);
        value.insert(QStringLiteral("out"), in + qMax(0, model->getClipPlaytime(id) - 1));
        return value;
    }
    pCore->displayMessage(i18n("Select a video clip on the timeline first."), ErrorMessage);
    return {};
}

namespace {
QString mediaPath(const QJsonValue &v)
{
    return v.isObject() ? v.toObject().value(QStringLiteral("path")).toString() : v.toString();
}

/** @brief A picture of what is on a slot, or the kind of thing it holds. */
void showThumb(QLabel *thumb, const QJsonValue &v)
{
    const QString path = mediaPath(v);
    QPixmap picture;
    if (!path.isEmpty() && QFileInfo::exists(path)) {
        picture = QPixmap(path);
    }
    if (!picture.isNull()) {
        thumb->setPixmap(picture.scaled(thumb->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    } else {
        thumb->setPixmap(QIcon::fromTheme(path.isEmpty() ? QStringLiteral("insert-image") : QStringLiteral("video-x-generic")).pixmap(24, 24));
    }
}

/** @brief The file's name, and how long the stretch is for a timeline clip. */
QString describeMedia(const QJsonValue &v)
{
    const QString path = mediaPath(v);
    if (path.isEmpty() || !QFileInfo::exists(path)) {
        return {};
    }
    QString name = QFileInfo(path).fileName();
    if (v.isObject()) {
        const double fps = pCore->getCurrentFps() > 0 ? pCore->getCurrentFps() : 25.0;
        const int frames = v.toObject().value(QStringLiteral("out")).toInt() - v.toObject().value(QStringLiteral("in")).toInt() + 1;
        name = i18n("%1, %2 s", name, QString::number(frames / fps, 'f', 1));
    }
    return name;
}
} // namespace

QWidget *GeneratorCard::buildMediaList(const PluginField &field)
{
    // References the user adds one by one, up to what each kind allows. Each
    // keeps its name (@image2) for good: the text may already mention it, so
    // taking one away leaves a gap rather than renaming the rest.
    auto *holder = new QWidget(m_body);
    auto *layout = new QVBoxLayout(holder);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(4);
    MediaList list;
    list.field = field;
    list.items = new QWidget(holder);
    auto *itemsLayout = new QVBoxLayout(list.items);
    itemsLayout->setContentsMargins(0, 0, 0, 0);
    itemsLayout->setSpacing(4);
    // Thirty references must not stretch the card down the whole chat: past a
    // few rows the list scrolls inside it
    auto *scroll = new QScrollArea(holder);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setWidgetResizable(true);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setWidget(list.items);
    list.scroll = scroll;
    layout->addWidget(scroll);
    auto *adds = new QHBoxLayout;
    adds->setContentsMargins(0, 0, 0, 0);
    for (const PluginMediaKind &kind : field.kinds) {
        auto *add = new QToolButton(holder);
        add->setIcon(QIcon::fromTheme(QStringLiteral("list-add")));
        add->setText(tr8(kind.label));
        add->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
        add->setAutoRaise(true);
        offerSources(add, kind.accept, tr8(kind.label), [this, key = field.key, kind](const QJsonValue &got) {
            QJsonObject entries = value(key).toObject();
            for (int n = 1; n <= kind.max; ++n) {
                const QString name = kind.key + QString::number(n);
                if (!entries.contains(name)) {
                    entries.insert(name, got);
                    break;
                }
            }
            setValue(key, entries);
            refreshMediaList(key);
        });
        adds->addWidget(add);
        list.adds.insert(kind.key, add);
    }
    adds->addStretch();
    layout->addLayout(adds);
    m_lists.insert(field.key, list);
    return holder;
}

QStringList GeneratorCard::kindNames(const QJsonObject &entries, const PluginMediaKind &kind)
{
    // by number, not as text: image2 before image10
    static const QRegularExpression numbered(QStringLiteral("^(\\D+)(\\d+)$"));
    QList<QPair<int, QString>> found;
    for (auto it = entries.constBegin(); it != entries.constEnd(); ++it) {
        const QRegularExpressionMatch match = numbered.match(it.key());
        if (match.hasMatch() && match.captured(1) == kind.key) {
            found.append({match.captured(2).toInt(), it.key()});
        }
    }
    std::sort(found.begin(), found.end());
    QStringList names;
    for (const auto &one : std::as_const(found)) {
        names << one.second;
    }
    return names;
}

int GeneratorCard::allowed(const PluginMediaKind &kind) const
{
    if (kind.maxByKey.isEmpty()) {
        return kind.max;
    }
    return kind.maxBy.value(value(kind.maxByKey).toVariant().toString(), kind.max);
}

void GeneratorCard::refreshMediaList(const QString &key)
{
    const MediaList list = m_lists.value(key);
    if (!list.items) {
        return;
    }
    const QList<QWidget *> old = list.items->findChildren<QWidget *>(QString(), Qt::FindDirectChildrenOnly);
    qDeleteAll(old);
    const QJsonObject entries = value(key).toObject();
    int rows = 0;
    for (const PluginMediaKind &kind : list.field.kinds) {
        const QStringList names = kindNames(entries, kind);
        const int most = allowed(kind);
        for (int i = 0; i < names.size(); ++i) {
            const QString &name = names.at(i);
            // Over what the chosen model takes: kept, since the text may name
            // it, but marked, and the card does not run until it goes
            const bool over = i >= most;
            auto *row = new QWidget(list.items);
            auto *line = new QHBoxLayout(row);
            line->setContentsMargins(0, 0, 0, 0);
            line->setSpacing(6);
            auto *thumb = new QLabel(row);
            thumb->setFixedSize(64, 40);
            thumb->setAlignment(Qt::AlignCenter);
            thumb->setObjectName(QStringLiteral("chatAttachThumb"));
            showThumb(thumb, entries.value(name));
            auto *label = new QLabel(QStringLiteral("@") + name + QStringLiteral("  ") + describeMedia(entries.value(name)), row);
            label->setToolTip(over ? (most == 0 ? i18n("%1 is not taken here", tr8(kind.label)) : i18n("%1, no more than %2", tr8(kind.label), most))
                                   : mediaPath(entries.value(name)));
            if (over) {
                label->setEnabled(false);
                thumb->setEnabled(false);
            }
            auto *remove = new QToolButton(row);
            remove->setIcon(QIcon::fromTheme(QStringLiteral("edit-clear")));
            remove->setToolTip(i18n("Remove"));
            remove->setAutoRaise(true);
            connect(remove, &QToolButton::clicked, this, [this, key, name]() {
                QJsonObject now = value(key).toObject();
                now.remove(name);
                setValue(key, now);
                refreshMediaList(key);
            });
            line->addWidget(thumb);
            line->addWidget(label, 1);
            line->addWidget(remove);
            list.items->layout()->addWidget(row);
            ++rows;
        }
        // an add button goes once its kind is full for the chosen model
        if (QToolButton *add = list.adds.value(kind.key)) {
            add->setVisible(names.size() < most);
        }
    }
    if (list.scroll) {
        // as tall as the rows, up to about six of them
        list.items->adjustSize();
        const int rowHeight = 40 + 4;
        list.scroll->setFixedHeight(qMin(rows, 6) * rowHeight);
        list.scroll->setVisible(rows > 0);
    }
}

void GeneratorCard::refreshMedia(const QString &key)
{
    const FieldWidgets w = m_fields.value(key);
    if (!w.thumb) {
        return;
    }
    const QJsonValue v = value(key);
    showThumb(w.thumb, v);
    w.name->setText(describeMedia(v));
    w.name->setToolTip(mediaPath(v));
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
    // a field that sets how many references a list takes (the model)
    for (auto it = m_lists.constBegin(); it != m_lists.constEnd(); ++it) {
        for (const PluginMediaKind &kind : it.value().field.kinds) {
            if (kind.maxByKey == key) {
                refreshMediaList(it.key());
                break;
            }
        }
    }
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
        if (field.type == QLatin1String("media_list")) {
            const QJsonObject entries = value(field.key).toObject();
            if (field.required && entries.isEmpty()) {
                return i18n("Add %1", tr8(field.label).toLower());
            }
            for (const PluginMediaKind &kind : field.kinds) {
                const int most = allowed(kind);
                if (kindNames(entries, kind).size() > most) {
                    return most == 0 ? i18n("%1 is not taken here", tr8(kind.label)) : i18n("%1, no more than %2", tr8(kind.label), most);
                }
            }
            continue;
        }
        if (field.type == QLatin1String("media")) {
            const QJsonValue v = value(field.key);
            const QString path = v.isObject() ? v.toObject().value(QStringLiteral("path")).toString() : v.toString();
            if (field.required && (path.isEmpty() || !QFileInfo::exists(path))) {
                return i18n("Add %1", tr8(field.label));
            }
            continue;
        }
        if (field.required && (field.type == QLatin1String("text") || field.type == QLatin1String("set")) &&
            value(field.key).toString().trimmed().isEmpty()) {
            return field.type == QLatin1String("set") ? i18n("Upload a voice first") : i18n("Fill in %1", tr8(field.label).toLower());
        }
    }
    return {};
}

void GeneratorCard::refreshFoot()
{
    if (!m_generate) {
        return;
    }
    const QString why = missing();
    // the gate answered yes for exactly what is on the card
    const bool open = m_gateId.isEmpty() || gateAction().isEmpty();
    m_generate->setVisible(open);
    m_generate->setEnabled(why.isEmpty() && open);
    m_generate->setToolTip(why);
    if (m_gateButton) {
        m_gateButton->setVisible(!open);
        m_gateButton->setEnabled(why.isEmpty() && !m_answer.pending);
        m_gateButton->setToolTip(why);
        m_cancel->setVisible(open);
    }
    for (QPushButton *button : std::as_const(m_actionButtons)) {
        button->setEnabled(!m_answer.pending);
    }
    if (m_answerLabel) {
        // an answer about values that have changed since says nothing any more
        const bool current = !m_answer.message.isEmpty() && !m_answer.pending && m_answer.asked == values();
        m_answerLabel->setText(PluginEffects::linkify(m_answer.message));
        m_answerLabel->setVisible(current);
    }
}

void GeneratorCard::clearAnswer()
{
    m_answer = Answer();
    refreshFoot();
}

QJsonObject GeneratorCard::values() const
{
    return m_payload.value(QStringLiteral("values")).toObject();
}

QString GeneratorCard::gateAction() const
{
    for (const PluginAction &action : m_generator.actions) {
        if (!action.gate) {
            continue;
        }
        const bool open = m_answer.action == action.id && m_answer.ok && !m_answer.pending && m_answer.asked == values();
        if (!open) {
            return action.id;
        }
    }
    return {};
}

QString GeneratorCard::gateBlocker() const
{
    const QString id = gateAction();
    for (const PluginAction &action : m_generator.actions) {
        if (action.id == id) {
            return tr8(action.label);
        }
    }
    return {};
}

void GeneratorCard::showAsking(const QString &action, const QJsonObject &asked)
{
    m_answer = Answer{action, asked, false, true, QString()};
    refreshFoot();
}

void GeneratorCard::showAnswer(const QString &action, const QJsonObject &asked, bool ok, const QString &message)
{
    m_answer = Answer{action, asked, ok, false, message};
    refreshFoot();
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
        } else if (auto *combo = qobject_cast<QComboBox *>(w.input)) {
            combo->setCurrentIndex(qMax(0, combo->findData(v.toString())));
        } else if (auto *slider = qobject_cast<QSlider *>(w.input)) {
            const double step = field.step > 0 ? field.step : 1;
            slider->setValue(qRound((v.toDouble() - field.min) / step));
        }
        if (field.type == QLatin1String("media")) {
            refreshMedia(field.key);
        } else if (field.type == QLatin1String("media_list")) {
            refreshMediaList(field.key);
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
