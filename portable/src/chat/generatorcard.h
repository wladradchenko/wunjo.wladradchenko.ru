/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "plugins/pluginmanifest.h"

#include <QFrame>
#include <QHash>
#include <QJsonObject>
#include <QPointer>

class QAudioOutput;
class QLabel;
class QLineEdit;
class QListWidget;
class QMediaPlayer;
class QPushButton;
class QToolButton;
class QVBoxLayout;

/** @class GeneratorCard
    @brief The form of a generator plugin as it stands in the chat: what to
    make, and the button that makes it.

    The card only collects the parameters. A run is a card of its own below it,
    the same one every plugin job gets, so the form stays editable while one
    renders and every run keeps its own line in the conversation.

    The fields come from one card of the manifest's `generate` list. Its state lives in the
    chat model (the payload), never here: the conversation is redrawn from the
    model, and an assistant fills the same card over MCP. Every edit goes out as
    @ref edited; a change from outside comes back through @ref setPayload.
 */
class GeneratorCard : public QFrame
{
    Q_OBJECT
public:
    GeneratorCard(const QString &cardId, const PluginManifest &manifest, const PluginGenerator &generator, const QJsonObject &payload,
                  QWidget *parent = nullptr);
    ~GeneratorCard() override;

    QString cardId() const { return m_cardId; }
    /** @brief The state as the card holds it, defaults filled in. */
    QJsonObject payload() const { return m_payload; }
    /** @brief Show a state that changed elsewhere (an assistant's edit). */
    void setPayload(const QJsonObject &payload);
    /** @brief Put the cursor in the first field, for a card just asked for. */
    void focusFirstField();

protected:
    void resizeEvent(QResizeEvent *event) override;

Q_SIGNALS:
    void edited(const QString &cardId, const QJsonObject &payload);
    void generateRequested(const QString &cardId);

private:
    struct FieldWidgets {
        QWidget *row = nullptr;  ///< what show_if hides
        QWidget *input = nullptr;
        QList<QToolButton *> choices; ///< enum as tabs
    };

    void buildForm();
    QWidget *buildLibrary(const PluginField &field);
    void refreshLibrary();
    void openLibrary();
    void uploadVoice();
    void togglePlay(const QString &path);
    void refreshVisibility();
    void refreshFoot();
    void refreshHeader();
    /** @brief @p byUser false for what the card corrects on its own (the voice
     *  that was chosen got deleted): that is nobody's edit. */
    void setValue(const QString &key, const QJsonValue &value, bool byUser = true);
    QJsonValue value(const QString &key) const;
    /** @brief What still has to be filled in before Generate can run, or empty. */
    QString missing() const;

    QString m_cardId;
    PluginManifest m_manifest;
    PluginGenerator m_generator;
    QJsonObject m_payload;
    bool m_applying = false;
    /** @brief True while the form is first built: defaults filled in then are
     *  not an edit by anyone. */
    bool m_building = true;

    QLabel *m_byLabel = nullptr;
    QLabel *m_summary = nullptr;
    QToolButton *m_fold = nullptr;
    QWidget *m_body = nullptr;
    QVBoxLayout *m_form = nullptr;
    QPushButton *m_generate = nullptr;
    QHash<QString, FieldWidgets> m_fields;

    // the library of presets (one `set` field per card is what a plugin needs)
    QString m_setKey;
    QString m_setKind;
    QPushButton *m_libraryButton = nullptr;
    QPushButton *m_libraryEmpty = nullptr;
    QWidget *m_libraryLine = nullptr;
    QPointer<QFrame> m_popup;
    QLineEdit *m_search = nullptr;
    QListWidget *m_list = nullptr;
    /** @brief A recording being registered: once it shows up in the library it
     *  is chosen, since that is why it was uploaded. */
    QString m_pendingSource;
    /** @brief Delete is two clicks on the same button, the first arms it. */
    QString m_armedDelete;

    QMediaPlayer *m_player = nullptr;
    QAudioOutput *m_audio = nullptr;
    QString m_playing;
    /** @brief The chosen voice as the button shows it, re-cut to the width
     *  whenever the card is resized. */
    QString m_currentName;
    QString m_currentLength;
    void updateLibraryButtonText();
};
