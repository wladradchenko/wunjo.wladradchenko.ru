/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "chatmessagemodel.h"

#include <QMap>
#include <QPlainTextEdit>
#include <QWidget>

class AbstractChatBackend;
class ChatHistoryStore;
class ChatVoiceInput;
class QLabel;
class QListWidget;
class QListWidgetItem;
class QProgressBar;
class QScrollArea;
class QStackedWidget;
class KMessageWidget;
class QMenu;
class QLineEdit;
class QToolButton;
class QHBoxLayout;
class QVBoxLayout;

/** @brief Input field: Enter sends, Shift+Enter inserts a newline,
    height grows with (wrapped) content up to five lines, then scrolls. */
class ChatInputEdit : public QPlainTextEdit
{
    Q_OBJECT
public:
    explicit ChatInputEdit(QWidget *parent = nullptr);

Q_SIGNALS:
    void sendRequested();

protected:
    void keyPressEvent(QKeyEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private:
    void adjustHeight();
};

/** @class ChatWidget
    @brief The Chat dock: per-project assistant chat (UI skeleton).
    A backend (Claude via MCP) plugs in through setBackend(); with no backend
    the widget renders conversations and stores them per project, replying
    with a "not connected yet" note.
 */
class ChatWidget : public QWidget
{
    Q_OBJECT
public:
    explicit ChatWidget(QWidget *parent = nullptr);

    /** @brief Attach an assistant backend (may be nullptr). Not owned. */
    void setBackend(AbstractChatBackend *backend);

    // External narration API (driven over the scripting socket by an agent such as Claude
    // Code): mirror the conversation and long-running work into the chat so the
    // user sees the request, the assistant's replies, a "thinking" state and
    // live tool/plugin/whisper progress. Pure main-thread model appends.
    /** @brief Append a bubble. @p author: 0=User, 1=Assistant, 2=Error. */
    void externalMessage(int author, const QString &text);
    /** @brief Update the last assistant bubble in place (streaming text). */
    void externalStreamAssistant(const QString &text);
    /** @brief Show/hide the "assistant is thinking" indicator. */
    void externalThinking(bool on);
    void externalToolStart(const QString &id, const QString &name);
    void externalToolProgress(const QString &id, int percent, const QString &message);
    void externalToolEnd(const QString &id, bool isError, const QString &result);

public Q_SLOTS:
    /** @brief Called when a project (re)connects; points history at it. */
    void setProjectFolder(const QString &projectDataFolder);
    void startNewSession();
    /** @brief Re-read which ways of talking exist and whether the chosen one is
     *  set up (called after a plugin is installed or its environment built). */
    void refreshBrains();
    /** @brief Re-reads the skills/loops library and per-project selection
        (called after MCP-side edits so the UI mirrors them live). */
    void refreshGuidance();

Q_SIGNALS:
    /** @brief Emitted for every user message (also passed to the backend). */
    void messageSent(const QString &text);
    /** @brief Mic button pressed; a Whisper-based voice input hooks in here. */
    void voiceInputRequested();

private Q_SLOTS:
    void submitInput();
    void showHistory();
    void openSession(QListWidgetItem *item);
    void onRowsInserted(const QModelIndex &parent, int first, int last);
    void onDataChanged(const QModelIndex &topLeft, const QModelIndex &bottomRight);

private:
    QWidget *buildBubble(const ChatMessage &message);
    /** @brief The slash menu: settings, model, and the commands by name. */
    QMenu *buildSlashMenu();
    /** @brief Put the line for the open tab on the strip, unless it was closed. */
    void updateHintBar();
    QWidget *buildToolCard(const ChatMessage &message);
    void refreshToolCard(const QModelIndex &index);
    void rebuildMessageArea();
    void refreshHistoryList();
    /** @brief Ask for files to hand to the assistant with the next message. */
    void attachFiles();
    /** @brief Redraw the strip of attached files from @ref m_attachments. */
    void refreshAttachments();
    /** @brief End the cards an agent opened and never closed. The editor's own
     *  cards — the ones it opens for a plugin run — are left alone: they end
     *  when the work does, however long after the answer that is. */
    void endOrphanedCards();
    void persistSession();
    /** @brief Absolute paths waiting to go with the next message. Cleared as
     *  soon as it is sent: they belong to that request, not to the session. */
    QStringList m_attachments;
    /** @brief True while the assistant is answering: the send button is a stop
     *  button then, and Enter stops instead of sending. */
    bool m_busy{false};
    /** @brief The last thing sent, so stopping can put it back in the field to
     *  be corrected rather than retyped. */
    QString m_lastSentText;
    void scrollToBottom();

    /** @brief Builds the Skills or Loops page (list + toolbar). */
    QWidget *buildGuidancePage(bool loops);
    /** @brief Builds the page that asks who should answer here. Shown by itself
     *  the first time, and reachable from the header afterwards. */
    QWidget *buildBrainPage();
    void refreshBrainPage();
    /** @brief Take up a way of talking: remember it, wire the backend, and lead
     *  the user to whatever still has to be set up. */
    void chooseBrain(const QString &id);
    /** @brief Bring the panel in line with the chosen way of talking. */
    void applyBrain();
    /** @brief Tell an assistant plugin which conversation it is now in. */
    void noteSessionChanged();
    /** @brief Opens the create/edit dialog for a guidance document. */
    void editGuidanceDocument(bool loops, const QString &existingName);
    /** @brief Stores the checked items as the project selection. */
    void applyGuidanceSelection(bool loops);
    void switchTab(int page);

    ChatMessageModel m_model;
    ChatHistoryStore *m_store;
    AbstractChatBackend *m_backend{nullptr};
    /** @brief The backend this widget made for the chosen assistant plugin.
     *  setBackend() also accepts one from outside, which this must not delete. */
    AbstractChatBackend *m_ownedBackend{nullptr};
    QString m_sessionId;
    QString m_sessionTitle;

    QLabel *m_titleLabel;
    QStackedWidget *m_stack;
    QScrollArea *m_scrollArea;
    QVBoxLayout *m_messagesLayout;
    QListWidget *m_historyList;
    QListWidget *m_skillsList{nullptr};
    QListWidget *m_loopsList{nullptr};
    QToolButton *m_tabChat{nullptr};
    QToolButton *m_tabSkills{nullptr};
    QToolButton *m_tabLoops{nullptr};
    /** @brief The one square beside the field: commands, model, who answers. */
    QToolButton *m_slashButton{nullptr};
    /** @brief Filter over the saved chats, shown with the history list. */
    QLineEdit *m_historySearch{nullptr};
    /** @brief The dismissible line above the input explaining the open tab. */
    KMessageWidget *m_hintBar{nullptr};
    /** @brief Closed once, gone for the rest of the session. */
    bool m_hintDismissed{false};
    QVBoxLayout *m_brainLayout{nullptr};  ///< the cards, rebuilt on refresh
    QWidget *m_externalCard{nullptr};     ///< shown instead of the input field
                                          ///< while an outside agent is driving
    QWidget *m_inputShell{nullptr};
    bool m_updatingGuidance{false};
    ChatInputEdit *m_input;
    QToolButton *m_sendButton{nullptr};
    QToolButton *m_micButton{nullptr};
    /** @brief The paperclip, and the strip of chips it fills. */
    QToolButton *m_attachButton{nullptr};
    QScrollArea *m_attachmentBar{nullptr};
    QHBoxLayout *m_attachmentLayout{nullptr};
    ChatVoiceInput *m_voice{nullptr};
    QLabel *m_thinkingLabel{nullptr}; // "assistant is thinking…" (external agent)
    QMap<QString, QProgressBar *> m_toolBars;
    QMap<QString, QLabel *> m_toolStatusLabels;
    QMap<QString, QLabel *> m_toolIcons;
};
