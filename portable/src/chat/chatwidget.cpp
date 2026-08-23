/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "chatwidget.h"
#include "chatbackend.h"
#include "chatbrain.h"
#include "chatguidancestore.h"
#include "chathistorystore.h"
#include "chatvoiceinput.h"
#include "core.h"
#include "mainwindow.h"
#include "pluginchatbackend.h"
#include "plugins/pluginmanager.h"
#include "theme.h"

#include <KActionCollection>
#include <KIconEffect>
#include <KLocalizedString>
#include <KMessageWidget>

#include <cmath>
#include <functional>

#include <QAction>
#include <QApplication>
#include <QClipboard>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QHBoxLayout>
#include <QIcon>
#include <QKeyEvent>
#include <QKeySequence>
#include <QMouseEvent>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QMenu>
#include <QActionGroup>
#include <QPainter>
#include <QPixmap>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QSet>
#include <QStackedWidget>
#include <QStyle>
#include <QTimer>
#include <QTextCursor>
#include <QToolButton>
#include <QUrl>
#include <QVBoxLayout>

// Bubble/card styling is scoped by objectName; global rules live in
// src/assets/style.qss (which deliberately never styles these).
namespace {

/** @brief The side of an attachment square. Big enough to recognise a photo in,
 *  small enough that a row of them does not become the panel. */
constexpr int kAttachTile = 44;

/** @brief The bullet in the reply gutter, in the theme's accent. It is drawn on
 *  the panel background, so it takes the ink shade. */
QPixmap dotPixmap()
{
    const QColor accent = WunjoTheme::instance()->accentInk();
    QPixmap pix(8, 8);
    pix.fill(Qt::transparent);
    QPainter painter(&pix);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(Qt::NoPen);
    painter.setBrush(accent);
    painter.drawEllipse(0, 0, 8, 8);
    return pix;
}

/** @brief The send arrow. The button behind it is filled with the accent ink —
 *  mint on the dark theme, a deep shade of it on the light one — so the glyph
 *  cannot follow the icon theme and is recolored to whichever of black/white
 *  reads on that fill. */
/** @brief The square that replaces the arrow while the assistant is working.
 *  Drawn rather than themed for the same reason as the arrow: it sits on the
 *  accent fill and has to read against it. */
QIcon stopIcon()
{
    const QColor fill = WunjoTheme::instance()->accentInk();
    const QColor glyph = fill.lightness() > 140 ? QColor(0x0D, 0x0D, 0x0D) : QColor(0xFF, 0xFF, 0xFF);
    QPixmap pix(18, 18);
    pix.fill(Qt::transparent);
    QPainter painter(&pix);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.setPen(Qt::NoPen);
    painter.setBrush(glyph);
    painter.drawRoundedRect(QRectF(4.5, 4.5, 9, 9), 1.5, 1.5);
    return QIcon(pix);
}

QIcon sendIcon()
{
    const QColor fill = WunjoTheme::instance()->accentInk();
    const QColor glyph = fill.lightness() > 140 ? QColor(0x0D, 0x0D, 0x0D) : QColor(0xFF, 0xFF, 0xFF);
    QImage image = QIcon::fromTheme(QStringLiteral("arrow-up")).pixmap(18, 18).toImage();
    KIconEffect::toMonochrome(image, glyph, glyph, 1.0f);
    return QIcon(QPixmap::fromImage(image));
}

} // namespace

static const char CHAT_STYLE[] = R"(
#chatBubbleUser {
    background: #2B3A32; border: 1px solid #C8EDD2; border-radius: 10px;
}
#chatBubbleError {
    background: #3A181D; border: 1px solid rgba(255, 48, 73, 0.6); border-radius: 12px;
}
#chatBubbleUser QLabel { color: #FFFFFF; background: transparent; }
#chatBubbleError QLabel { color: #FF9C9C; background: transparent; }
/* A reply is a step in a run, not a speech: a dot marks it, a hairline joins
   it to the next one, and the text is a size smaller than what the user typed
   so their own words stay the loudest thing on the page. */
#chatAssistantText { color: #FFFFFF; background: transparent; }
#chatStepDot { background: transparent; }
#chatStepRule { background: #2D2D2D; }
#chatSlashButton {
    background: transparent; border: 1px solid #2D2D2D; border-radius: 8px;
    color: #696969; font-weight: bold; padding: 0; min-width: 32px; max-width: 32px;
}
#chatSlashButton:hover { border-color: #C8EDD2; color: #C8EDD2; }
/* the popup arrow beside the slash makes it read as a dropdown rather than as
   the single square the reference has */
#chatSlashButton::menu-indicator { image: none; width: 0; }
#chatSearchField {
    background: #1F1F1F; border: 1px solid #2D2D2D; border-radius: 8px; padding: 4px 8px;
}
#chatToolCard {
    background: #1F1F1F; border: 1px solid #2D2D2D; border-radius: 12px; min-width: 240px;
}
#chatToolCard QLabel { background: transparent; }
#chatToolStatus { color: #696969; }
#chatToolCard QProgressBar {
    background: #2D2D2D; border: none; border-radius: 2px; max-height: 4px;
}
#chatToolCard QProgressBar::chunk { background: #A2E0B2; border-radius: 2px; }
#chatInputShell {
    background: #1F1F1F; border: 1px solid #2D2D2D; border-radius: 12px;
}
#chatInput {
    background: transparent; border: none; padding: 0;
    selection-background-color: @accent-fill; selection-color: @on-accent;
}
#chatSendButton {
    background: #C8EDD2; border: none; border-radius: 16px;
}
#chatSendButton:disabled { background: #2B3A32; }
#chatMicButton {
    background: transparent; border: 1px solid #2D2D2D; border-radius: 8px;
}
#chatMicButton:hover { background: #2B3A32; border-color: #C8EDD2; }
#chatAttachButton {
    background: transparent; border: 1px solid #2D2D2D; border-radius: 8px;
}
#chatAttachButton:hover { background: #2B3A32; border-color: #C8EDD2; }
/* one chip per attached file: its name, and a click to take it off again */
#chatAttachStrip { background: transparent; border: none; }
#chatAttachTile {
    background: #0D0D0D; border: 1px solid #2D2D2D; border-radius: 8px;
}
#chatAttachTile:hover { border-color: #C8EDD2; }
#chatAttachRemove {
    background: #0D0D0D; border: 1px solid #2D2D2D; border-radius: 8px;
    color: #FFFFFF; padding: 0; font-size: 11px;
}
#chatAttachRemove:hover { background: #FF3049; border-color: #FF3049; }
#chatMicButton[recording="true"] { background: #FF3049; border-color: #FF3049; }
#chatHint { color: #696969; }
#chatTabButton {
    background: transparent; border: none; border-radius: 0; color: #696969;
    padding: 3px 1px; border-bottom: 2px solid transparent;
}
#chatTabButton:hover { color: #FFFFFF; }
#chatTabButton:checked { color: #FFFFFF; border-bottom: 2px solid #C8EDD2; }
#chatGuidanceHint { color: #696969; }
#chatThinking { color: #A2E0B2; font-style: italic; padding: 2px 4px; }
#chatBrainCard {
    background: #1F1F1F; border: 1px solid #2D2D2D; border-radius: 12px;
}
#chatBrainCard:hover { border-color: #C8EDD2; background: #2B3A32; }
#chatBrainCard QLabel { background: transparent; }
#chatBrainTitle { color: #FFFFFF; font-weight: bold; }
#chatBrainHint { color: #696969; }
#chatExternalCard {
    background: #1F1F1F; border: 1px solid #2D2D2D; border-radius: 12px;
}
#chatExternalCard QLabel { background: transparent; color: #696969; }
#chatExternalCommand {
    background: #0D0D0D; border: 1px solid #2D2D2D; border-radius: 8px;
    color: #C8EDD2; padding: 6px;
}
#chatSuggestion {
    background: #1F1F1F; border: 1px solid #2D2D2D; border-radius: 8px;
}
#chatSuggestion:hover { border-color: #C8EDD2; background: #2B3A32; }
#chatSuggestion QLabel { background: transparent; }
#chatSuggestionName { color: #FFFFFF; }
#chatSuggestionNameOff { color: #696969; }
#chatSuggestionKeys { color: #C8EDD2; }
)";

namespace {

/** @brief A choice presented as a card: a heading, an explanation that wraps,
 *  and a click.
 *
 * QCommandLinkButton is the same idea and was the first attempt, but a push
 * button keeps one line's worth of height however long its description is, and
 * in a dock this narrow the explanation was cut off mid-sentence. Text that has
 * to wrap wants labels, so the card is built out of them and made clickable —
 * no signal needed, the callback is the whole contract.
 */
class BrainCard : public QFrame
{
public:
    BrainCard(const QString &title, const QString &description, QWidget *parent)
        : QFrame(parent)
    {
        setObjectName(QStringLiteral("chatBrainCard"));
        setAttribute(Qt::WA_Hover); // so the :hover rule in the stylesheet fires
        setCursor(Qt::PointingHandCursor);
        auto *layout = new QVBoxLayout(this);
        layout->setContentsMargins(12, 10, 12, 10);
        layout->setSpacing(4);
        auto *heading = new QLabel(title, this);
        heading->setObjectName(QStringLiteral("chatBrainTitle"));
        heading->setWordWrap(true);
        auto *body = new QLabel(description, this);
        body->setObjectName(QStringLiteral("chatBrainHint"));
        body->setWordWrap(true);
        layout->addWidget(heading);
        layout->addWidget(body);
    }

    std::function<void()> onClick;

protected:
    void mouseReleaseEvent(QMouseEvent *event) override
    {
        // On release, not press: a click begun here and dragged away should not
        // count, the same way a button behaves.
        if (event->button() == Qt::LeftButton && rect().contains(event->position().toPoint()) && onClick) {
            onClick();
        }
        QFrame::mouseReleaseEvent(event);
    }
};

/** @brief How many suggestions the strip may show at once.
 *
 * Five, and the rest are simply not shown. A scrolling list of everything that
 * matched would be a second panel above the panel, and the point of the strip
 * is to be glanced at, not read.
 */
constexpr int kMaxSuggestions = 5;

/** @brief One line of that strip: what the action is called, and the key that
 *  does the same thing without opening this panel at all.
 *
 * The key is the reason the strip earns its space. Finding the command is worth
 * one use; seeing its shortcut is worth every use after that, and after a week
 * the strip is not needed for that command any more. Disabled actions are shown
 * too, greyed — "you cannot do this right now" is an answer, and hiding them
 * would fail exactly the person who is searching because nothing is selected.
 */
class SuggestionRow : public QFrame
{
public:
    SuggestionRow(const QString &title, const QString &shortcut, bool enabled, QWidget *parent)
        : QFrame(parent)
    {
        setObjectName(QStringLiteral("chatSuggestion"));
        setAttribute(Qt::WA_Hover); // so the :hover rule in the stylesheet fires
        setCursor(Qt::PointingHandCursor);
        auto *layout = new QHBoxLayout(this);
        layout->setContentsMargins(10, 6, 10, 6);
        layout->setSpacing(8);
        auto *name = new QLabel(title, this);
        name->setObjectName(enabled ? QStringLiteral("chatSuggestionName") : QStringLiteral("chatSuggestionNameOff"));
        layout->addWidget(name, 1);
        if (!shortcut.isEmpty()) {
            auto *keys = new QLabel(shortcut, this);
            keys->setObjectName(QStringLiteral("chatSuggestionKeys"));
            layout->addWidget(keys);
        }
    }

    std::function<void()> onClick;

protected:
    void mouseReleaseEvent(QMouseEvent *event) override
    {
        if (event->button() == Qt::LeftButton && rect().contains(event->position().toPoint()) && onClick) {
            onClick();
        }
        QFrame::mouseReleaseEvent(event);
    }
};

} // namespace

ChatInputEdit::ChatInputEdit(QWidget *parent)
    : QPlainTextEdit(parent)
{
    setObjectName(QStringLiteral("chatInput"));
    setPlaceholderText(i18n("Ask Wunjo…"));
    setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    setTabChangesFocus(true);
    connect(this, &QPlainTextEdit::textChanged, this, &ChatInputEdit::adjustHeight);
    adjustHeight();
}

void ChatInputEdit::keyPressEvent(QKeyEvent *event)
{
    if ((event->key() == Qt::Key_Return || event->key() == Qt::Key_Enter) && !(event->modifiers() & Qt::ShiftModifier)) {
        Q_EMIT sendRequested();
        event->accept();
        return;
    }
    QPlainTextEdit::keyPressEvent(event);
}

void ChatInputEdit::resizeEvent(QResizeEvent *event)
{
    QPlainTextEdit::resizeEvent(event);
    // wrapped line count depends on the width
    adjustHeight();
}

void ChatInputEdit::adjustHeight()
{
    const int lineHeight = fontMetrics().lineSpacing();
    // QPlainTextEdit reports the document height in visual (wrapped) lines,
    // so long text without newlines grows the field too
    const int lines = qBound(1, int(std::ceil(document()->size().height())), 5);
    setFixedHeight(lines * lineHeight + 14);
}

ChatWidget::ChatWidget(QWidget *parent)
    : QWidget(parent)
    , m_store(new ChatHistoryStore(this))
{
    // The bubble/card palette + accent follow the active Wunjo theme; re-apply
    // whenever the theme or primary color changes.
    const auto applyChatStyle = [this]() {
        setStyleSheet(WunjoTheme::instance()->applyTokens(QString::fromLatin1(CHAT_STYLE)));
        // The two accent bits that are painted rather than styled: the send
        // glyph and the gutter dots of every reply already on screen.
        if (m_sendButton != nullptr) {
            m_sendButton->setIcon(sendIcon());
        }
        const QPixmap dot = dotPixmap();
        const QList<QLabel *> dots = findChildren<QLabel *>(QStringLiteral("chatStepDot"));
        for (QLabel *label : dots) {
            label->setPixmap(dot);
        }
    };
    applyChatStyle();
    connect(WunjoTheme::instance(), &WunjoTheme::accentChanged, this, applyChatStyle);
    connect(WunjoTheme::instance(), &WunjoTheme::themeChanged, this, applyChatStyle);

    auto *rootLayout = new QVBoxLayout(this);
    rootLayout->setContentsMargins(8, 4, 8, 8);
    rootLayout->setSpacing(6);

    // Top bar: session title, new chat, history
    auto *topBar = new QHBoxLayout;
    topBar->setSpacing(2);
    m_titleLabel = new QLabel(i18n("New chat"), this);
    auto *newButton = new QToolButton(this);
    newButton->setIcon(QIcon::fromTheme(QStringLiteral("list-add")));
    newButton->setToolTip(i18n("New chat"));
    newButton->setAutoRaise(true);
    auto *historyButton = new QToolButton(this);
    historyButton->setIcon(QIcon::fromTheme(QStringLiteral("document-open-recent")));
    historyButton->setToolTip(i18n("Chat history"));
    historyButton->setAutoRaise(true);
    topBar->addWidget(m_titleLabel);
    topBar->addStretch();
    topBar->addWidget(newButton);
    topBar->addWidget(historyButton);
    rootLayout->addLayout(topBar);

    // Tab row (mint underline, same language as the hub tabs):
    // Chat = conversation, Skills = "how to work" notes (multi-select),
    // Loops = start-to-finish pipeline scenario (single-select).
    auto *tabRow = new QHBoxLayout;
    tabRow->setSpacing(12);
    auto makeTab = [this](const QString &text) {
        auto *tab = new QToolButton(this);
        tab->setObjectName(QStringLiteral("chatTabButton"));
        tab->setText(text);
        tab->setCheckable(true);
        tab->setAutoRaise(true);
        tab->setToolButtonStyle(Qt::ToolButtonTextOnly);
        tab->setCursor(Qt::PointingHandCursor);
        return tab;
    };
    m_tabChat = makeTab(i18n("Chat"));
    m_tabSkills = makeTab(i18n("Skills"));
    m_tabLoops = makeTab(i18n("Loops"));
    // Plugins sits with them because being seen is the whole point: an entry in
    // the slash menu is found by people who already know it exists, which is
    // not who needs it. It opens the plugins page rather than showing one of
    // its own, so it is a tab in looks and a button in behaviour — it never
    // stays pressed and never takes the panel away from the conversation.
    m_tabPlugins = makeTab(i18n("Plugins"));
    m_tabChat->setChecked(true);
    tabRow->addWidget(m_tabChat);
    tabRow->addWidget(m_tabSkills);
    tabRow->addWidget(m_tabLoops);
    tabRow->addWidget(m_tabPlugins);
    tabRow->addStretch();
    rootLayout->addLayout(tabRow);

    m_stack = new QStackedWidget(this);
    rootLayout->addWidget(m_stack, 1);

    // Page 0: conversation
    m_scrollArea = new QScrollArea(m_stack);
    m_scrollArea->setWidgetResizable(true);
    m_scrollArea->setFrameShape(QFrame::NoFrame);
    auto *messagesHost = new QWidget(m_scrollArea);
    m_messagesLayout = new QVBoxLayout(messagesHost);
    m_messagesLayout->setContentsMargins(0, 0, 0, 0);
    m_messagesLayout->setSpacing(8);
    m_messagesLayout->addStretch(1);
    m_scrollArea->setWidget(messagesHost);
    m_stack->addWidget(m_scrollArea);

    // Page 1: history — a filter over the saved chats above the list, because
    // by the tenth session the titles are all "how do I…" and scrolling is not
    // how anybody finds one.
    auto *historyPage = new QWidget(m_stack);
    auto *historyLayout = new QVBoxLayout(historyPage);
    historyLayout->setContentsMargins(0, 0, 0, 0);
    historyLayout->setSpacing(6);
    m_historySearch = new QLineEdit(historyPage);
    m_historySearch->setObjectName(QStringLiteral("chatSearchField"));
    m_historySearch->setPlaceholderText(i18n("Search chats…"));
    m_historySearch->setClearButtonEnabled(true);
    m_historySearch->addAction(QIcon::fromTheme(QStringLiteral("search")), QLineEdit::LeadingPosition);
    historyLayout->addWidget(m_historySearch);
    m_historyList = new QListWidget(historyPage);
    m_historyList->setFrameShape(QFrame::NoFrame);
    m_historyList->setContextMenuPolicy(Qt::ActionsContextMenu);
    auto *deleteAction = new QAction(QIcon::fromTheme(QStringLiteral("edit-delete")), i18n("Delete chat"), m_historyList);
    m_historyList->addAction(deleteAction);
    historyLayout->addWidget(m_historyList, 1);
    m_stack->addWidget(historyPage);
    connect(m_historySearch, &QLineEdit::textChanged, this, [this](const QString &needle) {
        for (int i = 0; i < m_historyList->count(); ++i) {
            QListWidgetItem *item = m_historyList->item(i);
            item->setHidden(!needle.isEmpty() && !item->text().contains(needle, Qt::CaseInsensitive));
        }
    });

    // Page 2: skills, page 3: loops, page 4: who answers here
    m_stack->addWidget(buildGuidancePage(false));
    m_stack->addWidget(buildGuidancePage(true));
    m_stack->addWidget(buildBrainPage());

    // Input shell (ChatGPT-like): text on top, button row pinned below —
    // growing text pushes the shell up, buttons never mix with the text
    m_inputShell = new QFrame(this);
    m_inputShell->setObjectName(QStringLiteral("chatInputShell"));
    auto *shellLayout = new QVBoxLayout(m_inputShell);
    shellLayout->setContentsMargins(10, 8, 10, 8);
    shellLayout->setSpacing(6);
    m_input = new ChatInputEdit(m_inputShell);
    m_input->setFrameStyle(QFrame::NoFrame);

    auto *buttonRow = new QHBoxLayout;
    buttonRow->setSpacing(6);
    m_micButton = new QToolButton(m_inputShell);
    m_micButton->setObjectName(QStringLiteral("chatMicButton"));
    m_micButton->setFixedSize(32, 32);
    m_micButton->setIcon(QIcon::fromTheme(QStringLiteral("audio-input-microphone")));
    m_micButton->setToolTip(i18n("Voice input (Whisper)"));
    m_sendButton = new QToolButton(m_inputShell);
    m_sendButton->setObjectName(QStringLiteral("chatSendButton"));
    m_sendButton->setFixedSize(32, 32);
    m_sendButton->setIcon(sendIcon());
    m_sendButton->setToolTip(i18n("Send (Enter)"));
    // Everything that is not typing lives behind one square: who answers, which
    // model, and the commands that can be run by name. A row of controls above
    // the field said the same thing while taking the space the conversation
    // wants.
    m_slashButton = new QToolButton(m_inputShell);
    m_slashButton->setObjectName(QStringLiteral("chatSlashButton"));
    m_slashButton->setFixedSize(32, 32);
    m_slashButton->setText(QStringLiteral("/"));
    m_slashButton->setToolTip(i18n("Commands and settings"));
    m_slashButton->setPopupMode(QToolButton::InstantPopup);
    m_slashButton->setMenu(buildSlashMenu());
    // Naming a file is worse than pointing at one: a description has to be
    // guessed from, a path is the thing itself. What is attached here is handed
    // to the assistant as absolute paths, so it works on those files instead of
    // hunting for something with a similar name in a folder somebody mentioned.
    m_attachButton = new QToolButton(m_inputShell);
    m_attachButton->setObjectName(QStringLiteral("chatAttachButton"));
    m_attachButton->setFixedSize(32, 32);
    m_attachButton->setIcon(QIcon::fromTheme(QStringLiteral("mail-attachment")));
    m_attachButton->setToolTip(i18n("Attach files for the assistant to work on"));
    buttonRow->addWidget(m_slashButton);
    buttonRow->addWidget(m_attachButton);
    buttonRow->addWidget(m_micButton);
    buttonRow->addStretch();
    buttonRow->addWidget(m_sendButton);

    // The strip of what is attached, between the conversation and the field: one
    // small square per file, scrolling sideways when there are more than fit.
    // Written as a row of name buttons first, it pushed the whole input shell
    // wider than the dock with three screenshots on it — a panel this narrow has
    // no width to spend on file names.
    m_attachmentBar = new QScrollArea(m_inputShell);
    m_attachmentBar->setObjectName(QStringLiteral("chatAttachStrip"));
    m_attachmentBar->setWidgetResizable(true);
    m_attachmentBar->setFrameShape(QFrame::NoFrame);
    m_attachmentBar->setFixedHeight(kAttachTile + 10);
    m_attachmentBar->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_attachmentBar->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto *strip = new QWidget(m_attachmentBar);
    m_attachmentLayout = new QHBoxLayout(strip);
    m_attachmentLayout->setContentsMargins(0, 0, 0, 0);
    m_attachmentLayout->setSpacing(6);
    m_attachmentBar->setWidget(strip);
    m_attachmentBar->hide();

    shellLayout->addWidget(m_attachmentBar);
    shellLayout->addWidget(m_input);
    shellLayout->addLayout(buttonRow);

    // One line about whatever tab is open, in the same inline-message shape the
    // rest of the editor uses ("Switch to clip…" in the bin): it sits above the
    // input, and its close button takes it away for good. An explanation that
    // cannot be dismissed is a permanent tax on the screen.
    m_hintBar = new KMessageWidget(this);
    m_hintBar->setMessageType(KMessageWidget::Information);
    m_hintBar->setCloseButtonVisible(true);
    m_hintBar->setWordWrap(true);
    connect(m_hintBar, &KMessageWidget::hideAnimationFinished, this, [this]() { m_hintDismissed = true; });
    rootLayout->addWidget(m_hintBar);
    updateHintBar();

    // "Assistant is thinking…" indicator (driven by the external agent), just
    // above the input; hidden unless a request is in flight.
    m_thinkingLabel = new QLabel(i18n("Assistant is thinking…"), this);
    m_thinkingLabel->setObjectName(QStringLiteral("chatThinking"));
    m_thinkingLabel->setVisible(false);
    rootLayout->addWidget(m_thinkingLabel);

    // What the editor itself can do, matched against whatever is being typed and
    // offered above the field. It never takes Enter: what somebody wrote was
    // written for the assistant, and a panel that quietly did something else to
    // their timeline instead would be unusable after the first surprise. A
    // suggestion is taken by clicking it, and nothing about it reaches the
    // conversation — pressing a button is not a thing worth transcribing.
    //
    // With no model chosen this is the whole of what the panel does, and it
    // needs no environment, no agent and no network to do it.
    m_suggestions = new QWidget(this);
    m_suggestionLayout = new QVBoxLayout(m_suggestions);
    m_suggestionLayout->setContentsMargins(0, 0, 0, 4);
    m_suggestionLayout->setSpacing(4);
    m_suggestions->setVisible(false);
    rootLayout->addWidget(m_suggestions);
    connect(m_input, &QPlainTextEdit::textChanged, this, &ChatWidget::refreshSuggestions);

    rootLayout->addWidget(m_inputShell);

    // Shown in place of the input while an outside agent is driving: typing here
    // would go nowhere, and the useful thing is the folder and the command.
    m_externalCard = new QFrame(this);
    m_externalCard->setObjectName(QStringLiteral("chatExternalCard"));
    auto *externalLayout = new QVBoxLayout(m_externalCard);
    externalLayout->setContentsMargins(10, 10, 10, 10);
    externalLayout->setSpacing(6);
    auto *externalText = new QLabel(i18n("Claude Code is driving. Ask it there — what it does shows up here."), m_externalCard);
    externalText->setWordWrap(true);
    auto *commandLabel = new QLabel(ChatBrain::launchCommand(), m_externalCard);
    commandLabel->setObjectName(QStringLiteral("chatExternalCommand"));
    commandLabel->setWordWrap(true);
    commandLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    auto *externalButtons = new QHBoxLayout;
    auto *openFolder = new QPushButton(QIcon::fromTheme(QStringLiteral("folder")), i18n("Open folder"), m_externalCard);
    auto *copyCommand = new QPushButton(QIcon::fromTheme(QStringLiteral("edit-copy")), i18n("Copy command"), m_externalCard);
    externalButtons->addWidget(openFolder);
    externalButtons->addWidget(copyCommand);
    externalButtons->addStretch();
    externalLayout->addWidget(externalText);
    externalLayout->addWidget(commandLabel);
    externalLayout->addLayout(externalButtons);
    m_externalCard->setVisible(false);
    rootLayout->addWidget(m_externalCard);
    connect(openFolder, &QPushButton::clicked, this, [this]() {
        QString error;
        // Written again on every visit: the project may have been saved
        // somewhere else since, and a stale folder points at nothing.
        const QString folder = ChatBrain::prepareAgentFolder(&error);
        if (folder.isEmpty()) {
            m_model.appendText(ChatMessage::Author::Error, error);
            return;
        }
        QDesktopServices::openUrl(QUrl::fromLocalFile(folder));
    });
    connect(copyCommand, &QPushButton::clicked, this,
            [commandLabel]() { QApplication::clipboard()->setText(commandLabel->text()); });

    connect(newButton, &QToolButton::clicked, this, &ChatWidget::startNewSession);
    connect(historyButton, &QToolButton::clicked, this, &ChatWidget::showHistory);
    connect(m_tabChat, &QToolButton::clicked, this, [this]() { switchTab(0); });
    connect(m_tabSkills, &QToolButton::clicked, this, [this]() { switchTab(2); });
    connect(m_tabLoops, &QToolButton::clicked, this, [this]() { switchTab(3); });
    connect(m_tabPlugins, &QToolButton::clicked, this, [this]() {
        if (auto *window = pCore ? pCore->window() : nullptr) {
            window->showPluginSettings(QString());
        }
        // Put the pressed state back where it was: the panel did not move, a
        // window opened in front of it, and a tab left looking selected would
        // say otherwise.
        switchTab(m_stack->currentIndex());
    });
    connect(m_historyList, &QListWidget::itemActivated, this, &ChatWidget::openSession);
    connect(m_historyList, &QListWidget::itemClicked, this, &ChatWidget::openSession);
    connect(deleteAction, &QAction::triggered, this, [this]() {
        if (QListWidgetItem *item = m_historyList->currentItem()) {
            m_store->deleteSession(item->data(Qt::UserRole).toString());
            refreshHistoryList();
        }
    });
    connect(m_input, &ChatInputEdit::sendRequested, this, &ChatWidget::submitInput);
    connect(m_sendButton, &QToolButton::clicked, this, &ChatWidget::submitInput);
    connect(m_attachButton, &QToolButton::clicked, this, &ChatWidget::attachFiles);

    // Voice input: mic toggles recording; Whisper text lands in the field
    m_voice = new ChatVoiceInput(this);
    connect(m_micButton, &QToolButton::clicked, this, [this]() {
        Q_EMIT voiceInputRequested();
        m_voice->toggle();
    });
    connect(m_voice, &ChatVoiceInput::recordingChanged, this, [this](bool recording) {
        m_micButton->setProperty("recording", recording);
        m_micButton->setToolTip(recording ? i18n("Stop recording") : i18n("Voice input (Whisper)"));
        m_micButton->style()->unpolish(m_micButton);
        m_micButton->style()->polish(m_micButton);
    });
    connect(m_voice, &ChatVoiceInput::busyChanged, this, [this](bool busy) {
        m_micButton->setEnabled(!busy);
        m_micButton->setToolTip(busy ? i18n("Transcribing…") : i18n("Voice input (Whisper)"));
    });
    connect(m_voice, &ChatVoiceInput::transcribed, this, [this](const QString &text) {
        if (!m_input->toPlainText().isEmpty() && !m_input->toPlainText().endsWith(QLatin1Char(' '))) {
            m_input->insertPlainText(QStringLiteral(" "));
        }
        m_input->insertPlainText(text);
        m_input->setFocus();
    });
    connect(m_voice, &ChatVoiceInput::errorOccurred, this,
            [this](const QString &message) { m_model.appendText(ChatMessage::Author::Error, message); });

    connect(&m_model, &QAbstractItemModel::rowsInserted, this, &ChatWidget::onRowsInserted);
    connect(&m_model, &QAbstractItemModel::dataChanged, this, &ChatWidget::onDataChanged);
    connect(&m_model, &QAbstractItemModel::modelReset, this, &ChatWidget::rebuildMessageArea);

    // A plugin finishing its setup is what turns a greyed-out choice into a
    // usable one, so the panel follows that rather than asking on every click.
    connect(&PluginManager::instance(), &PluginManager::pluginsChanged, this, &ChatWidget::refreshBrains);

    refreshGuidance();
    startNewSession(); // ends with applyBrain(), which decides the first page
}

// ── External narration API (driven over the scripting socket) ───────────

void ChatWidget::externalMessage(int author, const QString &text)
{
    if (m_stack->currentIndex() != 0) {
        switchTab(0); // make sure the mirrored conversation is visible
    }
    m_model.appendText(static_cast<ChatMessage::Author>(qBound(0, author, 2)), text);
    persistSession();
}

void ChatWidget::externalStreamAssistant(const QString &text)
{
    m_model.updateLastText(ChatMessage::Author::Assistant, text);
}

void ChatWidget::externalThinking(bool on)
{
    if (m_thinkingLabel) {
        m_thinkingLabel->setVisible(on);
    }
}

void ChatWidget::externalToolStart(const QString &id, const QString &name)
{
    if (m_stack->currentIndex() != 0) {
        switchTab(0);
    }
    m_model.appendToolCard(id, name);
}

void ChatWidget::externalToolProgress(const QString &id, int percent, const QString &message)
{
    m_model.updateTool(id, ChatMessage::ToolStatus::Running, percent, message);
}

void ChatWidget::externalToolEnd(const QString &id, bool isError, const QString &result)
{
    m_model.updateTool(id, isError ? ChatMessage::ToolStatus::Failed : ChatMessage::ToolStatus::Done, 100, result);
    persistSession();
}

void ChatWidget::switchTab(int page)
{
    m_tabChat->setChecked(page == 0);
    m_tabSkills->setChecked(page == 2);
    m_tabLoops->setChecked(page == 3);
    m_tabPlugins->setChecked(false); // never a page of its own — see its connect
    m_stack->setCurrentIndex(page);
    updateHintBar();
    // Which of the two sits under the conversation depends on who answers, so
    // the way of talking decides it and this only says whether either shows.
    const bool onConversation = page == 0;
    m_inputShell->setVisible(onConversation);
    m_externalCard->setVisible(false);
    if (page == 2 || page == 3) {
        refreshGuidance();
    } else if (page == 4) {
        refreshBrainPage();
    }
}

// ── Who answers here ────────────────────────────────────────────────────

QWidget *ChatWidget::buildBrainPage()
{
    auto *page = new QWidget(m_stack);
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(8);

    auto *title = new QLabel(i18n("How would you like to work with Wunjo?"), page);
    title->setObjectName(QStringLiteral("chatBrainTitle"));
    title->setWordWrap(true);
    layout->addWidget(title);

    auto *hint = new QLabel(i18n("Both drive the editor the same way. You can change this at any time."), page);
    hint->setObjectName(QStringLiteral("chatBrainHint"));
    hint->setWordWrap(true);
    layout->addWidget(hint);

    m_brainLayout = new QVBoxLayout;
    m_brainLayout->setSpacing(8);
    layout->addLayout(m_brainLayout);
    layout->addStretch(1);
    return page;
}

void ChatWidget::refreshBrainPage()
{
    if (m_brainLayout == nullptr) {
        return;
    }
    while (QLayoutItem *item = m_brainLayout->takeAt(0)) {
        delete item->widget();
        delete item;
    }
    const QList<ChatBrain::Option> options = ChatBrain::options();
    for (const ChatBrain::Option &option : options) {
        // The name, and nothing appended to it. Which one is in use is what the
        // tick in the model menu is for, and whether one still needs setting up
        // is the model's own business to say on its settings page — a card that
        // spells both out is repeating what two other places already show.
        auto *card = new BrainCard(option.name, option.description, m_brainLayout->parentWidget());
        card->onClick = [this, id = option.id]() { chooseBrain(id); };
        m_brainLayout->addWidget(card);
    }
}

void ChatWidget::refreshSuggestions()
{
    if (m_suggestionLayout == nullptr) {
        return;
    }
    while (QLayoutItem *item = m_suggestionLayout->takeAt(0)) {
        delete item->widget();
        delete item;
    }
    const QString needle = m_input->toPlainText().trimmed();
    auto *window = pCore ? pCore->window() : nullptr;
    // Two characters before anything is offered: one letter matches most of the
    // application and turns the strip into noise on the way to a sentence.
    if (needle.size() < 2 || window == nullptr) {
        m_suggestions->setVisible(false);
        return;
    }

    // The editor's own actions are the catalogue, and it maintains itself:
    // whatever is added to the application anywhere shows up here with no
    // change to this function. Their text is already in the user's language,
    // which is why a plain substring match works across languages.
    int shown = 0;
    QSet<QString> seen;
    const QList<QAction *> actions = window->actionCollection()->actions();
    for (QAction *action : actions) {
        if (shown >= kMaxSuggestions) {
            break;
        }
        if (action == nullptr || action->isSeparator()) {
            continue;
        }
        // "&Split" is how a menu spells its keyboard accelerator; nobody types
        // the ampersand, and it must not be matched against or displayed.
        const QString text = KLocalizedString::removeAcceleratorMarker(action->text());
        if (text.isEmpty() || !text.contains(needle, Qt::CaseInsensitive) || seen.contains(text)) {
            continue;
        }
        seen.insert(text);
        auto *row = new SuggestionRow(text, action->shortcut().toString(QKeySequence::NativeText), action->isEnabled(), m_suggestions);
        // trigger() is a no-op on a disabled action, so a greyed row costs a
        // click and does nothing rather than needing a guard of its own.
        row->onClick = [this, action]() {
            action->trigger();
            m_input->clear();
        };
        m_suggestionLayout->addWidget(row);
        ++shown;
    }
    m_suggestions->setVisible(shown > 0);
}

void ChatWidget::refreshBrains()
{
    refreshBrainPage();
    // The slash menu lists the models too and was built once, in the
    // constructor. That went unnoticed while a model shipped with the
    // application: it existed before the menu was made, so the menu was never
    // wrong. Now that a model arrives as an install, one done with the window
    // open would only appear under "Switch model" after a restart.
    if (m_slashButton != nullptr) {
        QMenu *stale = m_slashButton->menu();
        m_slashButton->setMenu(buildSlashMenu());
        if (stale != nullptr) {
            stale->deleteLater();
        }
    }
    applyBrain();
}

void ChatWidget::chooseBrain(const QString &id)
{
    if (id == ChatBrain::External()) {
        QString error;
        const QString folder = ChatBrain::prepareAgentFolder(&error);
        if (folder.isEmpty()) {
            // Nothing is chosen on a failure: leaving the user in a mode whose
            // folder was never written would only puzzle them later.
            m_model.appendText(ChatMessage::Author::Error, error);
            switchTab(0);
            return;
        }
        ChatBrain::setCurrent(id);
        applyBrain();
        QDesktopServices::openUrl(QUrl::fromLocalFile(folder));
        // No backticks around the command: the bubble renders rich text, and a
        // grave accent next to a letter is drawn as an accent on it — the hint
        // came out as "run ìclaudè".
        m_model.appendText(ChatMessage::Author::Assistant,
                           i18n("Everything Claude Code needs is in %1. Open a terminal there and run the claude command.", folder));
        // The folder is written either way, but the tool server it names cannot
        // start until its environment exists. Said here rather than built here:
        // it installs on its own settings tab, the way every other environment
        // in the application does, so there is one place that knows how to
        // build one and one banner that reports what pip said.
        if (!ChatBrain::serverReady()) {
            if (auto *window = pCore ? pCore->window() : nullptr) {
                window->showPluginSettings(QString());
            }
            m_model.appendText(ChatMessage::Author::Assistant,
                               i18n("MCP is not installed yet — install it in Settings, Plugins, on the MCP tab. Until then the "
                                    "agent's first question will fail in its terminal."));
        }
        switchTab(0);
        return;
    }

    ChatBrain::setCurrent(id);
    applyBrain();
    // A model on this machine reaches the editor by the same tools an agent in
    // a terminal does, so it needs the tool server just as much. It has its own
    // environment rather than joining the shared venv: that one carries whisper
    // and torch, and the server would then wait on a multi-gigabyte install to
    // answer "what is on the timeline". Where the server does need the heavy
    // stack it borrows it by path — see WUNJO_SPEECH_PYTHON in data/mcp_start.
    if (!ChatBrain::serverReady()) {
        m_model.appendText(ChatMessage::Author::Assistant,
                           i18n("This model drives the editor through MCP, which is not installed yet. Install it in Settings, "
                                "Plugins, on the MCP tab."));
    }
    const QString blocker = PluginManager::instance().runBlocker(id);
    const QList<ChatBrain::Option> options = ChatBrain::options();
    for (const ChatBrain::Option &option : options) {
        if (option.id == id && !option.ready) {
            // Straight to the page that finishes the job, rather than a note
            // about where to find it.
            if (auto *window = pCore ? pCore->window() : nullptr) {
                window->showPluginSettings(id);
            }
            m_model.appendText(ChatMessage::Author::Assistant,
                               blocker.isEmpty() ? i18n("Install the assistant in the settings that just opened, then come back here.") : blocker);
            break;
        }
    }
    switchTab(0);
}

void ChatWidget::applyBrain()
{
    QString mode = ChatBrain::current();
    // A mode naming a model that is not here any more — uninstalled, or renamed
    // between versions, as "agent" became "qwen35" when it stopped shipping by
    // default — points at nothing. Empty is the honest state for that, and the
    // panel knows how to explain it; a backend built on a missing plugin would
    // instead fail on every message with the reason buried in a blocker string.
    if (!mode.isEmpty() && mode != ChatBrain::External()) {
        bool installed = false;
        const QList<ChatBrain::Option> ways = ChatBrain::options();
        for (const ChatBrain::Option &way : ways) {
            if (way.id == mode) {
                installed = true;
                break;
            }
        }
        if (!installed) {
            mode.clear();
            ChatBrain::setCurrent(mode);
        }
    }
    // Three states, and empty is a real one rather than a gap to be filled in.
    // A fresh install ships no model at all — the editor is driven from Claude
    // Code or Cursor, and a model on this machine is a plugin somebody adds if
    // they want one. Choosing on the user's behalf here is what used to send
    // people to install several gigabytes to answer a question they had not
    // asked. What is typed with nothing chosen gets an answer saying how to
    // choose; see submitInput().
    const bool driveHere = !mode.isEmpty() && mode != ChatBrain::External();

    if (m_ownedBackend != nullptr && (!driveHere || m_backend != m_ownedBackend)) {
        if (auto *plugin = qobject_cast<PluginChatBackend *>(m_ownedBackend)) {
            plugin->release(); // hand back the graphics card before letting go
        }
        m_ownedBackend->deleteLater();
        m_ownedBackend = nullptr;
        m_backend = nullptr;
    }
    if (driveHere && m_ownedBackend == nullptr) {
        auto *backend = new PluginChatBackend(mode, this);
        backend->setSession(m_sessionId);
        m_ownedBackend = backend;
        setBackend(backend);
    }
    if (m_stack->currentIndex() == 4) {
        switchTab(0);
    } else {
        switchTab(m_stack->currentIndex());
    }
}

QWidget *ChatWidget::buildGuidancePage(bool loops)
{
    auto *page = new QWidget(m_stack);
    auto *layout = new QVBoxLayout(page);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(6);

    auto *list = new QListWidget(page);
    list->setFrameShape(QFrame::NoFrame);
    layout->addWidget(list, 1);
    (loops ? m_loopsList : m_skillsList) = list;

    auto *buttonRow = new QHBoxLayout;
    buttonRow->setSpacing(4);
    auto makeButton = [page](const QString &icon, const QString &tip) {
        auto *button = new QToolButton(page);
        button->setIcon(QIcon::fromTheme(icon));
        button->setToolTip(tip);
        button->setAutoRaise(true);
        return button;
    };
    auto *addButton = makeButton(QStringLiteral("list-add"), loops ? i18n("New loop") : i18n("New skill"));
    auto *importButton = makeButton(QStringLiteral("document-import"), i18n("Import from file…"));
    auto *editButton = makeButton(QStringLiteral("document-edit"), i18n("Edit"));
    auto *deleteButton = makeButton(QStringLiteral("edit-delete"), i18n("Delete"));
    buttonRow->addWidget(addButton);
    buttonRow->addWidget(importButton);
    buttonRow->addStretch();
    buttonRow->addWidget(editButton);
    buttonRow->addWidget(deleteButton);
    layout->addLayout(buttonRow);

    const auto kind = loops ? ChatGuidanceStore::Kind::Loop : ChatGuidanceStore::Kind::Skill;
    connect(addButton, &QToolButton::clicked, this, [this, loops]() { editGuidanceDocument(loops, QString()); });
    connect(editButton, &QToolButton::clicked, this, [this, loops, list]() {
        if (QListWidgetItem *item = list->currentItem()) {
            editGuidanceDocument(loops, item->text());
        }
    });
    connect(importButton, &QToolButton::clicked, this, [this, loops, kind]() {
        const QStringList files = QFileDialog::getOpenFileNames(this, loops ? i18n("Import Loops") : i18n("Import Skills"),
                                                                QDir::homePath(), i18n("Text documents (*.md *.txt)"));
        for (const QString &file : files) {
            QFile source(file);
            if (source.open(QIODevice::ReadOnly | QIODevice::Text)) {
                ChatGuidanceStore::write(kind, QFileInfo(file).completeBaseName(), QString::fromUtf8(source.readAll()));
            }
        }
        refreshGuidance();
    });
    connect(deleteButton, &QToolButton::clicked, this, [this, kind, list]() {
        if (QListWidgetItem *item = list->currentItem()) {
            ChatGuidanceStore::remove(kind, item->text());
            refreshGuidance();
        }
    });
    connect(list, &QListWidget::itemDoubleClicked, this,
            [this, loops](QListWidgetItem *item) { editGuidanceDocument(loops, item->text()); });
    connect(list, &QListWidget::itemChanged, this, [this, loops]() { applyGuidanceSelection(loops); });

    return page;
}

void ChatWidget::editGuidanceDocument(bool loops, const QString &existingName)
{
    const auto kind = loops ? ChatGuidanceStore::Kind::Loop : ChatGuidanceStore::Kind::Skill;

    QDialog dialog(this);
    dialog.setWindowTitle(existingName.isEmpty() ? (loops ? i18n("New Loop") : i18n("New Skill"))
                                                 : (loops ? i18n("Edit Loop") : i18n("Edit Skill")));
    dialog.resize(560, 480);
    auto *layout = new QVBoxLayout(&dialog);
    auto *nameEdit = new QLineEdit(existingName, &dialog);
    nameEdit->setPlaceholderText(i18n("Name"));
    auto *contentEdit = new QPlainTextEdit(&dialog);
    contentEdit->setPlainText(existingName.isEmpty() ? QString() : ChatGuidanceStore::read(kind, existingName));
    contentEdit->setPlaceholderText(loops ? i18n("Describe the pipeline step by step: what to produce and which "
                                                 "tools or plugins to use at every stage…")
                                          : i18n("Describe how the assistant should work…"));
    auto *buttons = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, &dialog);
    connect(buttons, &QDialogButtonBox::accepted, &dialog, &QDialog::accept);
    connect(buttons, &QDialogButtonBox::rejected, &dialog, &QDialog::reject);
    layout->addWidget(nameEdit);
    layout->addWidget(contentEdit, 1);
    layout->addWidget(buttons);

    if (dialog.exec() != QDialog::Accepted) {
        return;
    }
    const QString name = nameEdit->text().trimmed();
    if (name.isEmpty()) {
        return;
    }
    if (!existingName.isEmpty() && name != existingName) {
        ChatGuidanceStore::remove(kind, existingName); // rename = delete + create
    }
    ChatGuidanceStore::write(kind, name, contentEdit->toPlainText());
    refreshGuidance();
}

void ChatWidget::applyGuidanceSelection(bool loops)
{
    if (m_updatingGuidance) {
        return;
    }
    QListWidget *list = loops ? m_loopsList : m_skillsList;
    if (loops) {
        // Single selection: checking one loop unchecks the others
        QString selected;
        for (int i = 0; i < list->count(); ++i) {
            QListWidgetItem *item = list->item(i);
            if (item->checkState() == Qt::Checked && item->text() != ChatGuidanceStore::selectedLoop()) {
                selected = item->text();
            }
        }
        if (selected.isEmpty()) { // toggle-off or no change
            for (int i = 0; i < list->count(); ++i) {
                if (list->item(i)->checkState() == Qt::Checked) {
                    selected = list->item(i)->text();
                    break;
                }
            }
        }
        ChatGuidanceStore::setSelectedLoop(selected);
    } else {
        QStringList names;
        for (int i = 0; i < list->count(); ++i) {
            if (list->item(i)->checkState() == Qt::Checked) {
                names << list->item(i)->text();
            }
        }
        ChatGuidanceStore::setSelectedSkills(names);
    }
    refreshGuidance();
}

void ChatWidget::refreshGuidance()
{
    if (!m_skillsList || !m_loopsList || m_updatingGuidance) {
        return;
    }
    m_updatingGuidance = true;
    const QStringList selectedSkills = ChatGuidanceStore::selectedSkills();
    m_skillsList->clear();
    const QStringList skills = ChatGuidanceStore::list(ChatGuidanceStore::Kind::Skill);
    for (const QString &name : skills) {
        auto *item = new QListWidgetItem(name, m_skillsList);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(selectedSkills.contains(name) ? Qt::Checked : Qt::Unchecked);
    }
    const QString selectedLoop = ChatGuidanceStore::selectedLoop();
    m_loopsList->clear();
    const QStringList loops = ChatGuidanceStore::list(ChatGuidanceStore::Kind::Loop);
    for (const QString &name : loops) {
        auto *item = new QListWidgetItem(name, m_loopsList);
        item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
        item->setCheckState(name == selectedLoop ? Qt::Checked : Qt::Unchecked);
    }
    m_updatingGuidance = false;
}

void ChatWidget::setBackend(AbstractChatBackend *backend)
{
    if (m_backend) {
        m_backend->disconnect(this);
    }
    m_backend = backend;
    if (!m_backend) {
        return;
    }
    connect(m_backend, &AbstractChatBackend::assistantText, this, [this](const QString &text) {
        m_model.appendText(ChatMessage::Author::Assistant, text);
        persistSession();
    });
    connect(m_backend, &AbstractChatBackend::errorOccurred, this,
            [this](const QString &message) { m_model.appendText(ChatMessage::Author::Error, message); });
    connect(m_backend, &AbstractChatBackend::toolStarted, this,
            [this](const QString &id, const QString &name) { m_model.appendToolCard(id, name); });
    connect(m_backend, &AbstractChatBackend::toolProgress, this, [this](const QString &id, int percent, const QString &message) {
        m_model.updateTool(id, ChatMessage::ToolStatus::Running, percent, message);
    });
    connect(m_backend, &AbstractChatBackend::toolFinished, this, [this](const QString &id, bool isError, const QString &result) {
        m_model.updateTool(id, isError ? ChatMessage::ToolStatus::Failed : ChatMessage::ToolStatus::Done, 100, result);
        persistSession();
    });
    connect(m_backend, &AbstractChatBackend::busyChanged, this, [this](bool busy) {
        // The button does not go dead while the assistant works — it becomes the
        // way to stop it. A turn can run for minutes down a path the user can
        // already see is wrong, and the only thing worse than waiting for it is
        // having no way to say so.
        m_busy = busy;
        m_sendButton->setEnabled(true);
        m_sendButton->setIcon(busy ? stopIcon() : sendIcon());
        m_sendButton->setToolTip(busy ? i18n("Stop the assistant") : i18n("Send (Enter)"));
        // The assistant plugin turns this on and off while it works;
        // for the stretch before its first word, the backend answers for it.
        if (busy && m_thinkingLabel) {
            m_thinkingLabel->setVisible(true);
        } else if (!busy && m_thinkingLabel) {
            m_thinkingLabel->setVisible(false);
        }
        if (!busy) {
            // The turn is over, so nothing the assistant opened is still being
            // watched by it. Its own cards are ended here rather than left
            // turning for the rest of the session; the editor's cards, which
            // outlive the turn, keep going and end when their work does.
            endOrphanedCards();
        }
    });
}

/** @brief Tell an assistant plugin which conversation the next message joins. */
void ChatWidget::noteSessionChanged()
{
    if (auto *plugin = qobject_cast<PluginChatBackend *>(m_backend)) {
        plugin->setSession(m_sessionId);
    }
}

void ChatWidget::setProjectFolder(const QString &projectDataFolder)
{
    refreshGuidance(); // per-project selection may differ
    m_store->setProjectFolder(projectDataFolder);
    startNewSession();
}

void ChatWidget::startNewSession()
{
    m_sessionId = m_store->createSessionId();
    m_sessionTitle.clear();
    m_titleLabel->setText(i18n("New chat"));
    m_model.clear();
    noteSessionChanged();
    // Which page to land on is the way-of-talking's decision, not this one:
    // opening a project starts a session too, and jumping straight to an empty
    // conversation would step over a picker the user has not answered yet.
    applyBrain();
}

void ChatWidget::attachFiles()
{
    const QStringList chosen = QFileDialog::getOpenFileNames(this, i18n("Attach files"), QDir::homePath());
    for (const QString &path : chosen) {
        if (!m_attachments.contains(path)) {
            m_attachments.append(path);
        }
    }
    refreshAttachments();
}

void ChatWidget::refreshAttachments()
{
    // Rebuilt rather than patched: the strip is a handful of squares, and a list
    // always drawn from m_attachments cannot drift away from it.
    QWidget *strip = m_attachmentBar->widget();
    while (QLayoutItem *item = m_attachmentLayout->takeAt(0)) {
        delete item->widget();
        delete item;
    }
    for (const QString &path : std::as_const(m_attachments)) {
        auto *tile = new QFrame(strip);
        tile->setObjectName(QStringLiteral("chatAttachTile"));
        tile->setFixedSize(kAttachTile, kAttachTile);
        tile->setToolTip(path);
        auto *picture = new QLabel(tile);
        picture->setGeometry(0, 0, kAttachTile, kAttachTile);
        picture->setAlignment(Qt::AlignCenter);
        // A photograph shows itself; everything else shows what kind of thing it
        // is. Either way the square is the same size, so the strip stays a strip.
        const QPixmap preview(path);
        if (!preview.isNull()) {
            picture->setPixmap(preview.scaled(kAttachTile, kAttachTile, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation));
            picture->setScaledContents(false);
        } else {
            const QString suffix = QFileInfo(path).suffix().toLower();
            static const QStringList sounds{QStringLiteral("wav"), QStringLiteral("mp3"),  QStringLiteral("m4a"),
                                            QStringLiteral("aac"), QStringLiteral("flac"), QStringLiteral("ogg")};
            const QString icon = sounds.contains(suffix) ? QStringLiteral("audio-x-generic") : QStringLiteral("video-x-generic");
            picture->setPixmap(QIcon::fromTheme(icon).pixmap(kAttachTile / 2, kAttachTile / 2));
        }
        // The cross sits on the corner of the square rather than beside it: at
        // this size there is no room beside anything.
        auto *remove = new QToolButton(tile);
        remove->setObjectName(QStringLiteral("chatAttachRemove"));
        remove->setFixedSize(16, 16);
        remove->setText(QStringLiteral("×"));
        remove->setCursor(Qt::PointingHandCursor);
        remove->setToolTip(i18n("Remove %1", QFileInfo(path).fileName()));
        remove->move(kAttachTile - 17, 1);
        remove->raise();
        connect(remove, &QToolButton::clicked, this, [this, path]() {
            m_attachments.removeAll(path);
            refreshAttachments();
        });
        m_attachmentLayout->addWidget(tile);
    }
    m_attachmentLayout->addStretch();
    m_attachmentBar->setVisible(!m_attachments.isEmpty());
}

void ChatWidget::submitInput()
{
    if (m_busy) {
        // The same square that sends is what stops. What the assistant was
        // asked goes back into the field so the request can be corrected and
        // sent again rather than typed out a second time — unless something is
        // already typed there, which is nobody's to throw away.
        if (m_backend) {
            m_backend->cancel();
        }
        m_busy = false;
        m_sendButton->setIcon(sendIcon());
        m_sendButton->setToolTip(i18n("Send (Enter)"));
        if (m_thinkingLabel) {
            m_thinkingLabel->setVisible(false);
        }
        endOrphanedCards();
        m_model.appendText(ChatMessage::Author::Error, i18n("Stopped."));
        if (m_input->toPlainText().trimmed().isEmpty() && !m_lastSentText.isEmpty()) {
            m_input->setPlainText(m_lastSentText);
            m_input->moveCursor(QTextCursor::End);
        }
        persistSession();
        return;
    }
    const QString text = m_input->toPlainText().trimmed();
    if (text.isEmpty() && m_attachments.isEmpty()) {
        return;
    }
    m_input->clear();
    m_stack->setCurrentIndex(0);
    if (m_sessionTitle.isEmpty()) {
        m_sessionTitle = text.length() > 40 ? text.left(40) + QStringLiteral("…") : text;
        m_titleLabel->setText(m_sessionTitle);
    }
    // Whatever was left turning from the previous turn belongs to an agent that
    // is not coming back — an outside one that was stopped mid-answer leaves no
    // other trace of it. A new question is the moment to say so.
    endOrphanedCards();
    // What the panel shows is what the person wrote, with the files named under
    // it. What the assistant receives carries their absolute paths, because a
    // name is something to search for and a path is the file itself.
    QString shown = text;
    QString sent = text;
    if (!m_attachments.isEmpty()) {
        QStringList names;
        for (const QString &path : std::as_const(m_attachments)) {
            names << QFileInfo(path).fileName();
        }
        shown = text.isEmpty() ? i18n("Attached: %1", names.join(QStringLiteral(", ")))
                               : text + QStringLiteral("\n") + i18n("Attached: %1", names.join(QStringLiteral(", ")));
        sent = text + QStringLiteral("\n\nAttached files, by absolute path — work on these, do not go looking for "
                                     "anything similar elsewhere:\n- ")
               + m_attachments.join(QStringLiteral("\n- "));
        m_attachments.clear();
        refreshAttachments();
    }
    m_lastSentText = text;
    m_model.appendText(ChatMessage::Author::User, shown);
    Q_EMIT messageSent(sent);
    if (m_backend) {
        m_backend->sendMessage(sent);
    } else if (ChatBrain::current() == ChatBrain::External()) {
        // Not a failure — the mode working as intended. The request was typed
        // in the wrong window, so the answer names the right one instead of
        // saying that nothing happened.
        m_model.appendText(ChatMessage::Author::Assistant,
                           i18n("Your own agent is driving. Ask it in the terminal you opened in %1, and what it does appears here.",
                                ChatBrain::agentFolder()));
    } else {
        // Nothing chosen, which is how a fresh install starts. Both ways out
        // are named, because neither is obvious from an empty panel.
        m_model.appendText(ChatMessage::Author::Assistant,
                           i18n("No model is selected. Press / to choose one, or pick \"External MCP\" to drive the editor from an "
                                "agent you already use, such as Claude Code or Cursor."));
    }
    persistSession();
}

void ChatWidget::showHistory()
{
    refreshHistoryList();
    m_stack->setCurrentIndex(m_stack->currentIndex() == 1 ? 0 : 1);
}

void ChatWidget::openSession(QListWidgetItem *item)
{
    if (!item) {
        return;
    }
    QString title;
    const QJsonArray messages = m_store->loadSession(item->data(Qt::UserRole).toString(), &title);
    m_sessionId = item->data(Qt::UserRole).toString();
    m_sessionTitle = title;
    m_titleLabel->setText(title.isEmpty() ? i18n("New chat") : title);
    m_model.loadJson(messages);
    noteSessionChanged();
    applyBrain();
}

void ChatWidget::refreshHistoryList()
{
    m_historyList->clear();
    const QList<ChatSessionInfo> sessions = m_store->sessions();
    for (const ChatSessionInfo &info : sessions) {
        auto *item = new QListWidgetItem(m_historyList);
        const QString title = info.title.isEmpty() ? i18n("Untitled chat") : info.title;
        item->setText(title + QLatin1Char('\n') + info.updatedAt.toString(QStringLiteral("d MMM yyyy, hh:mm")));
        item->setData(Qt::UserRole, info.id);
    }
    if (sessions.isEmpty()) {
        auto *item = new QListWidgetItem(i18n("No saved chats for this project yet"), m_historyList);
        item->setFlags(Qt::NoItemFlags);
    }
}

void ChatWidget::endOrphanedCards()
{
    // "plugin:" and "render:" are the editor's own, from PluginManager. Anything
    // else was opened by an agent, and an agent that has stopped answering will
    // never close it.
    m_model.endRunningTools(i18n("interrupted"), {QStringLiteral("plugin:"), QStringLiteral("render:")});
    persistSession();
}

void ChatWidget::persistSession()
{
    m_store->saveSession(m_sessionId, m_sessionTitle, m_model.toJson());
}

void ChatWidget::updateHintBar()
{
    if (m_hintBar == nullptr || m_hintDismissed) {
        return;
    }
    QString text;
    switch (m_stack->currentIndex()) {
    case 2:
        text = i18n("Skills are simple instructions that tell the assistant how to work in this project.");
        break;
    case 3:
        text = i18n("A loop is a set of steps the assistant follows to complete a task using the tools or plugins you choose.");
        break;
    case 0:
        text = i18n("Use local model or external MSP for third-party agents to automatically generate and edit content and mix effects from plugins.");
        break;
    default:
        m_hintBar->hide();
        return;
    }
    m_hintBar->setText(text);
    m_hintBar->show();
}

QMenu *ChatWidget::buildSlashMenu()
{
    auto *menu = new QMenu(this);

    // ── Settings ───────────────────────────────────────────────────────────
    auto *settings = menu->addSection(i18n("Settings"));
    Q_UNUSED(settings)
    // The one switch that matters: keep answering here, or hand the keys to an
    // agent outside and let this panel mirror it.
    auto *external = menu->addAction(QIcon::fromTheme(QStringLiteral("folder")), i18n("External MCP…"));
    external->setToolTip(i18n("Write the folder an outside agent works in — Claude Code, Cursor, Codex — and open it"));
    connect(external, &QAction::triggered, this, [this]() {
        // Not a mode — an action. The folder is written and opened; the files in
        // it explain themselves, this panel goes on working, and an outside
        // agent connects through the .mcp.json in there whenever it likes.
        QString error;
        const QString folder = ChatBrain::prepareAgentFolder(&error);
        if (folder.isEmpty()) {
            m_model.appendText(ChatMessage::Author::Error, error);
            return;
        }
        QDesktopServices::openUrl(QUrl::fromLocalFile(folder));
    });
    // ── Model ──────────────────────────────────────────────────────────────
    // One entry that opens the list, rather than the models themselves: today
    // there is a single local one, and a flat list of one reads as a mistake.
    // What is offered is what this machine can run; picking one that is not
    // built yet opens its page, which is where the download happens.
    auto *models = menu->addMenu(i18n("Switch model…"));
    const QList<ChatBrain::Option> options = ChatBrain::options();
    auto *group = new QActionGroup(models);
    group->setExclusive(true);
    for (const ChatBrain::Option &option : options) {
        if (option.id == ChatBrain::External()) {
            continue; // that one is a folder to open, not a model
        }
        // The name alone. Whether it still needs setting up is not something to
        // label a menu entry with — picking one that is not ready says so in
        // the conversation, with what to do about it, which is the moment the
        // answer is of any use.
        auto *action = models->addAction(option.name);
        action->setCheckable(true);
        action->setChecked(ChatBrain::current() == option.id);
        action->setToolTip(option.description);
        group->addAction(action);
        const QString id = option.id;
        const bool ready = option.ready;
        connect(action, &QAction::triggered, this, [this, id, ready]() {
            ChatBrain::setCurrent(id);
            applyBrain();
            if (!ready) {
                // Nothing to talk to yet: send them where it is installed.
                if (auto *window = pCore->window()) {
                    window->showPluginSettings(id);
                }
            }
        });
    }
    if (models->isEmpty()) {
        models->addAction(i18n("No models installed"))->setEnabled(false);
    }

    // ── Commands ───────────────────────────────────────────────────────────
    // The same things the tabs do, reachable by name — and the place where the
    // editor's own tools will be listed for calling on their own.
    menu->addSection(i18n("Commands"));
    menu->addAction(i18n("/new — start a new chat"), this, &ChatWidget::startNewSession);
    menu->addAction(i18n("/history — find an earlier chat"), this, &ChatWidget::showHistory);
    menu->addAction(i18n("/skills — how it should work"), this, [this]() { switchTab(2); });
    menu->addAction(i18n("/loops — a pipeline to follow"), this, [this]() { switchTab(3); });
    return menu;
}

QWidget *ChatWidget::buildBubble(const ChatMessage &message)
{
    // Assistant replies flow full-width as plain text (ChatGPT/Claude style);
    // only user messages and errors get a bubble.
    if (message.author == ChatMessage::Author::Assistant) {
        auto *step = new QWidget(this);
        auto *row = new QHBoxLayout(step);
        row->setContentsMargins(2, 2, 2, 2);
        row->setSpacing(8);
        // The gutter: a dot for this step and a hairline continuing to the next,
        // so a long answer reads as one thread of work rather than loose text.
        auto *gutter = new QWidget(step);
        gutter->setFixedWidth(10);
        auto *gutterLayout = new QVBoxLayout(gutter);
        gutterLayout->setContentsMargins(0, 4, 0, 0);
        gutterLayout->setSpacing(0);
        auto *dot = new QLabel(gutter);
        dot->setObjectName(QStringLiteral("chatStepDot"));
        dot->setFixedSize(8, 8);
        dot->setPixmap(dotPixmap());
        gutterLayout->addWidget(dot, 0, Qt::AlignHCenter);
        auto *rule = new QFrame(gutter);
        rule->setObjectName(QStringLiteral("chatStepRule"));
        rule->setFixedWidth(1);
        gutterLayout->addWidget(rule, 1, Qt::AlignHCenter);
        row->addWidget(gutter);

        auto *label = new QLabel(message.text, step);
        label->setObjectName(QStringLiteral("chatAssistantText"));
        label->setWordWrap(true);
        label->setTextInteractionFlags(Qt::TextSelectableByMouse);
        row->addWidget(label, 1);
        return step;
    }

    auto *bubble = new QFrame(this);
    bubble->setObjectName(message.author == ChatMessage::Author::User ? QStringLiteral("chatBubbleUser") : QStringLiteral("chatBubbleError"));
    auto *layout = new QVBoxLayout(bubble);
    layout->setContentsMargins(10, 8, 10, 8);
    auto *label = new QLabel(message.text, bubble);
    label->setWordWrap(true);
    label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    layout->addWidget(label);

    auto *row = new QWidget(this);
    auto *rowLayout = new QHBoxLayout(row);
    rowLayout->setContentsMargins(0, 0, 0, 0);
    if (message.author == ChatMessage::Author::User) {
        // full width: it is the heading of everything that follows it
        rowLayout->addWidget(bubble, 1);
    } else {
        rowLayout->addWidget(bubble, 4);
        rowLayout->addStretch(1);
    }
    return row;
}

QWidget *ChatWidget::buildToolCard(const ChatMessage &message)
{
    auto *card = new QFrame(this);
    card->setObjectName(QStringLiteral("chatToolCard"));
    auto *layout = new QVBoxLayout(card);
    layout->setContentsMargins(10, 8, 10, 8);
    layout->setSpacing(4);

    auto *header = new QHBoxLayout;
    auto *statusIcon = new QLabel(card);
    statusIcon->setFixedSize(16, 16);
    statusIcon->setPixmap(QIcon::fromTheme(QStringLiteral("task-process-1")).pixmap(16, 16));
    auto *name = new QLabel(message.toolName, card);
    QFont nameFont = name->font();
    nameFont.setWeight(QFont::DemiBold);
    name->setFont(nameFont);
    header->addWidget(statusIcon);
    header->addWidget(name);
    header->addStretch();
    layout->addLayout(header);

    auto *status = new QLabel(message.text.isEmpty() ? i18n("Running…") : message.text, card);
    status->setObjectName(QStringLiteral("chatToolStatus"));
    status->setWordWrap(true);
    layout->addWidget(status);

    auto *progress = new QProgressBar(card);
    progress->setRange(0, 100);
    progress->setValue(qMax(0, message.toolProgress));
    progress->setTextVisible(false);
    layout->addWidget(progress);

    m_toolIcons.insert(message.toolId, statusIcon);
    m_toolStatusLabels.insert(message.toolId, status);
    m_toolBars.insert(message.toolId, progress);

    auto *row = new QWidget(this);
    auto *rowLayout = new QHBoxLayout(row);
    rowLayout->setContentsMargins(0, 0, 0, 0);
    rowLayout->addWidget(card, 4);
    rowLayout->addStretch(1);
    return row;
}

void ChatWidget::refreshToolCard(const QModelIndex &index)
{
    const QString id = index.data(ChatMessageModel::ToolIdRole).toString();
    const auto status = ChatMessage::ToolStatus(index.data(ChatMessageModel::ToolStatusRole).toInt());
    if (QProgressBar *bar = m_toolBars.value(id)) {
        bar->setValue(qMax(0, index.data(ChatMessageModel::ToolProgressRole).toInt()));
        bar->setVisible(status == ChatMessage::ToolStatus::Running);
    }
    if (QLabel *label = m_toolStatusLabels.value(id)) {
        const QString text = index.data(ChatMessageModel::TextRole).toString();
        if (!text.isEmpty()) {
            label->setText(text);
        }
    }
    if (QLabel *icon = m_toolIcons.value(id)) {
        QString iconName = QStringLiteral("task-process-1");
        if (status == ChatMessage::ToolStatus::Done) {
            iconName = QStringLiteral("task-process-4");
        } else if (status == ChatMessage::ToolStatus::Failed) {
            iconName = QStringLiteral("data-warning");
        }
        icon->setPixmap(QIcon::fromTheme(iconName).pixmap(16, 16));
    }
}

void ChatWidget::onRowsInserted(const QModelIndex &parent, int first, int last)
{
    Q_UNUSED(parent)
    for (int row = first; row <= last; ++row) {
        const QModelIndex index = m_model.index(row);
        ChatMessage message;
        message.author = ChatMessage::Author(index.data(ChatMessageModel::AuthorRole).toInt());
        message.kind = ChatMessage::Kind(index.data(ChatMessageModel::KindRole).toInt());
        message.text = index.data(ChatMessageModel::TextRole).toString();
        message.toolId = index.data(ChatMessageModel::ToolIdRole).toString();
        message.toolName = index.data(ChatMessageModel::ToolNameRole).toString();
        message.toolProgress = index.data(ChatMessageModel::ToolProgressRole).toInt();
        // insert before the trailing stretch
        const int position = m_messagesLayout->count() - 1;
        m_messagesLayout->insertWidget(position, message.kind == ChatMessage::Kind::ToolCard ? buildToolCard(message) : buildBubble(message));
    }
    scrollToBottom();
}

void ChatWidget::onDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight)
{
    for (int row = topLeft.row(); row <= bottomRight.row(); ++row) {
        const QModelIndex index = m_model.index(row);
        if (ChatMessage::Kind(index.data(ChatMessageModel::KindRole).toInt()) == ChatMessage::Kind::ToolCard) {
            refreshToolCard(index);
        }
    }
}

void ChatWidget::rebuildMessageArea()
{
    m_toolBars.clear();
    m_toolStatusLabels.clear();
    m_toolIcons.clear();
    while (m_messagesLayout->count() > 1) {
        QLayoutItem *item = m_messagesLayout->takeAt(0);
        if (QWidget *w = item->widget()) {
            w->deleteLater();
        }
        delete item;
    }
    if (m_model.rowCount() > 0) {
        onRowsInserted(QModelIndex(), 0, m_model.rowCount() - 1);
    } else {
        // Nothing to say here: what this panel is for is written on the strip
        // above the input, where it can be dismissed once read.
    }
}

void ChatWidget::scrollToBottom()
{
    QTimer::singleShot(0, m_scrollArea, [this]() { m_scrollArea->verticalScrollBar()->setValue(m_scrollArea->verticalScrollBar()->maximum()); });
}
