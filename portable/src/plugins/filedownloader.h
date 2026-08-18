/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QFile>
#include <QList>
#include <QObject>
#include <QPair>
#include <QTimer>
#include <QUrl>

class QNetworkAccessManager;
class QNetworkReply;

/** @class FileDownloader
    @brief One resumable HTTP download, from a plugin's weights to the plugin
    package itself.

    It exists because `KIO::file_copy` cannot do the two things a three-gigabyte
    weight over a domestic connection needs.

    It cannot **notice a dead link**: the KF6 http worker is a
    QNetworkAccessManager in a worker process that never sets a transfer timeout
    and no longer reads `ReadTimeout` from `kioslaverc`, so a connection that
    stops delivering bytes without closing — a phone that lost its cell, a NAT
    that dropped the session — leaves the job waiting until the operating
    system's TCP stack gives up, which is a quarter of an hour at best and never
    at worst. The download appeared to freeze at some percentage and did not
    continue when the network came back.

    And it cannot **carry on where it stopped**: every attempt started at byte
    zero, so a link that breaks every few hundred megabytes never finishes the
    file no matter how many times it is retried.

    So this class keeps the fragment in `<dest>.part` and asks for the rest with
    a `Range` header, watches the byte stream and aborts a stalled transfer
    itself, and retries on a backoff ladder that resets whenever real progress
    is made — a flapping connection downloads the file in pieces without the
    user pressing anything. The fragment survives the application being closed,
    so tomorrow's attempt continues today's.

    It is also the only thing that works on all three platforms: the http worker
    KIO needs lives in kio-extras, which the Windows and macOS builds do not
    ship, and there `KIO::file_copy` on an https URL fails outright.

    The caller owns the verification: this delivers the bytes the server sent
    and renames them into place, @ref PluginManager::modelState says whether the
    checksum matches.
 */
class FileDownloader : public QObject
{
    Q_OBJECT
public:
    explicit FileDownloader(QObject *parent = nullptr);
    ~FileDownloader() override;

    /** @brief How much of @p destPath is already on disk from an earlier
     *  attempt, so the settings page can offer "continue" instead of "download"
     *  and the disk-space check only asks for what is still to come. */
    static qint64 resumableBytes(const QString &destPath);

    /** @brief Fetch @p url into @p destPath, continuing a previous attempt when
     *  its fragment is still there. */
    void start(const QUrl &url, const QString &destPath);
    /** @brief A header to send with every request, kept across retries — a
     *  stock library asks for its key that way ("Authorization: Bearer …"), and
     *  a resumed transfer that dropped it would come back as 401 instead of the
     *  rest of the file. */
    void setHeader(const QByteArray &name, const QByteArray &value);
    /** @brief Stop, keeping the fragment: what is downloaded stays downloaded. */
    void cancel();
    /** @brief Let go of the connection but remember the file. Resuming asks for
     *  the rest of it, which is the same thing a broken line does — the
     *  mechanism was there, this only gives the user the button. */
    void pause();
    void resume();
    bool isRunning() const;
    bool isPaused() const { return m_paused; }

Q_SIGNALS:
    /** @brief @p total is 0 until the server states a size — the caller knows
     *  the one the manifest declares and can show that in the meantime. */
    void progress(qint64 received, qint64 total);
    /** @brief The transfer broke and the next attempt is @p seconds away —
     *  said out loud, because a silent wait looks like the freeze this class
     *  was written to end. */
    void retrying(const QString &reason, int seconds);
    /** @brief @p error is empty when @ref cancel stopped it. */
    void finished(bool ok, const QString &error);

private:
    /** @brief Open the fragment and ask the server for everything after it. */
    void requestRest();
    /** @brief Read what arrived, append it, and keep the watchdog satisfied. */
    void readData();
    /** @brief Decide what the response means before the first byte is written:
     *  a resumed transfer, a server that ignored the range and starts over, or
     *  an answer that is not the file at all — which lands in @ref
     *  m_pendingError and is acted on once the reply is over. */
    void takeHeaders();
    void replyFinished();
    /** @brief Rename the fragment into place once it is whole. */
    void complete();
    /** @brief Wait, then try again from where the fragment ends. */
    void retryLater(const QString &reason);
    void fail(const QString &error);
    void cleanup();

    QNetworkAccessManager *m_net = nullptr;
    QNetworkReply *m_reply = nullptr;
    QFile m_file;
    QUrl m_url;
    QString m_dest;
    QByteArray m_etag;
    QList<QPair<QByteArray, QByteArray>> m_headers;
    /** @brief What the response already told us, acted on once the reply is
     *  over: reading the headers happens while the reply is still delivering,
     *  and ending it from in there would re-enter its own slots. */
    QString m_pendingError;
    bool m_pendingFatal = false;
    qint64 m_received = 0;
    qint64 m_total = 0;
    /** @brief Bytes this attempt brought in — an attempt that moved the file
     *  forward is not a failed connection, it is a slow one, and it must not
     *  count towards giving up. */
    qint64 m_attemptBytes = 0;
    int m_failures = 0;
    bool m_headersTaken = false;
    bool m_cancelled = false;
    bool m_paused = false;
    /** @brief No byte for this long means the link is dead even though the
     *  socket still pretends otherwise. */
    QTimer m_watchdog;
    QTimer m_retry;
};
