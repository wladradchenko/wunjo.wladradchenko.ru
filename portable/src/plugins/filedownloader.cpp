/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "filedownloader.h"

#include <KLocalizedString>

#include <QFileInfo>
#include <QNetworkAccessManager>
#include <QNetworkProxyFactory>
#include <QNetworkReply>
#include <QNetworkRequest>

namespace
{
/** @brief A connection that has not delivered a byte for this long is dead. The
 *  socket can stay open for a quarter of an hour after the network underneath
 *  it is gone, and waiting for the operating system to admit that is what made
 *  a download look frozen. */
constexpr int kStallMs = 30000;
/** @brief The wait before each further attempt, in seconds. Only attempts that
 *  brought nothing in climb this ladder — see @ref kProgressBytes. */
constexpr int kBackoffSeconds[] = {2, 5, 15, 30, 60};
constexpr int kBackoffSteps = int(sizeof(kBackoffSeconds) / sizeof(kBackoffSeconds[0]));
/** @brief How many attempts in a row may fail without moving the file forward
 *  before the download gives up and says so. */
constexpr int kMaxFailures = 8;
/** @brief What an attempt has to deliver to count as a working connection
 *  rather than a lucky handshake. A link that breaks every minute but downloads
 *  something each time will finish the file; one that only ever opens and dies
 *  must not retry forever. */
constexpr qint64 kProgressBytes = 64 * 1024;

/** @brief Where an unfinished download lives. The final name is taken only by a
 *  file that is whole, so nothing — not the settings page, not the plugin —
 *  ever sees a fragment and takes it for a model. */
QString partPath(const QString &destPath)
{
    return destPath + QStringLiteral(".part");
}
} // namespace

FileDownloader::FileDownloader(QObject *parent)
    : QObject(parent)
    , m_net(new QNetworkAccessManager(this))
{
    // On Unix Qt only looks at http_proxy and friends when asked to; the user
    // behind a company proxy has no other way to reach the weights.
    QNetworkProxyFactory::setUseSystemConfiguration(true);

    m_watchdog.setSingleShot(true);
    connect(&m_watchdog, &QTimer::timeout, this, [this]() {
        if (m_reply == nullptr) {
            return;
        }
        // Say what happened rather than letting the abort surface as Qt's
        // "Operation canceled", which reads like the user did it.
        m_pendingError = i18n("no data for %1 seconds", kStallMs / 1000);
        m_reply->abort();
    });
    m_retry.setSingleShot(true);
    connect(&m_retry, &QTimer::timeout, this, &FileDownloader::requestRest);
}

FileDownloader::~FileDownloader()
{
    cleanup();
}

qint64 FileDownloader::resumableBytes(const QString &destPath)
{
    const QFileInfo info(partPath(destPath));
    return info.exists() ? info.size() : 0;
}

bool FileDownloader::isRunning() const
{
    return m_reply != nullptr || m_retry.isActive() || m_paused;
}

void FileDownloader::pause()
{
    if (m_paused || m_cancelled) {
        return;
    }
    m_paused = true;
    m_watchdog.stop();
    m_retry.stop();
    if (m_reply != nullptr) {
        disconnect(m_reply, nullptr, this, nullptr);
        m_reply->abort();
        m_reply->deleteLater();
        m_reply = nullptr;
    }
}

void FileDownloader::resume()
{
    if (!m_paused) {
        return;
    }
    m_paused = false;
    // Not a new download: the fragment on disk decides where this one asks the
    // server to start.
    requestRest();
}

void FileDownloader::start(const QUrl &url, const QString &destPath)
{
    if (isRunning()) {
        return;
    }
    m_url = url;
    m_dest = destPath;
    m_etag.clear();
    m_total = 0;
    m_received = 0;
    m_failures = 0;
    m_cancelled = false;
    m_paused = false;
    requestRest();
}

void FileDownloader::setHeader(const QByteArray &name, const QByteArray &value)
{
    m_headers.append({name, value});
}

void FileDownloader::cancel()
{
    if (!isRunning()) {
        return;
    }
    m_cancelled = true;
    cleanup();
    // The fragment stays on disk: what has been downloaded has been downloaded,
    // and the next attempt continues from it. An empty error says this was the
    // user's decision, not a failure to report.
    Q_EMIT finished(false, QString());
}

void FileDownloader::requestRest()
{
    if (m_cancelled) {
        return;
    }
    // A retry after a break continues into the file the last attempt left open;
    // renaming it is what closes it for good.
    if (!m_file.isOpen()) {
        m_file.setFileName(partPath(m_dest));
        if (!m_file.open(QIODevice::WriteOnly | QIODevice::Append)) {
            fail(i18n("Cannot write to %1.", m_file.fileName()));
            return;
        }
    }
    // What is already there is where this attempt starts — including a fragment
    // left by a previous run of the application.
    m_received = m_file.size();
    m_attemptBytes = 0;
    m_headersTaken = false;
    m_pendingFatal = false;
    m_pendingError.clear();

    QNetworkRequest request(m_url);
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);
    // Take the bytes as they are: a compressed transfer counts different bytes
    // than the file has, and a range into a compressed stream is not a range
    // into the file.
    request.setRawHeader("Accept-Encoding", "identity");
    for (const auto &header : std::as_const(m_headers)) {
        request.setRawHeader(header.first, header.second);
    }
    if (m_received > 0) {
        request.setRawHeader("Range", "bytes=" + QByteArray::number(m_received) + '-');
        // Continue this file, not whatever stands at that address now: if the
        // weight was replaced on the server, its tail must not be glued to the
        // head of the old one. Without the tag the server would answer 206 and
        // hand back a file that is corrupt in a way only the checksum finds.
        if (!m_etag.isEmpty()) {
            request.setRawHeader("If-Range", m_etag);
        }
    }
    m_reply = m_net->get(request);
    connect(m_reply, &QIODevice::readyRead, this, &FileDownloader::readData);
    connect(m_reply, &QNetworkReply::finished, this, &FileDownloader::replyFinished);
    m_watchdog.start(kStallMs);
    Q_EMIT progress(m_received, m_total);
}

void FileDownloader::takeHeaders()
{
    const int status = m_reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
    if (status == 416) {
        // The fragment is at or past the end of what the server has: either the
        // file is complete and only the rename is missing, or it is not the file
        // this fragment belongs to. Starting over is the answer that is right in
        // both cases, and only costs the rare one a second download.
        m_file.resize(0);
        m_received = 0;
        m_pendingError = i18n("the download had to start over");
        return;
    }
    if (status >= 400) {
        // A server that is busy or overloaded will serve this file later; one
        // that says the file is not there never will.
        m_pendingFatal = !(status == 408 || status == 429 || status >= 500);
        m_pendingError = i18n("the server answered %1", status);
        return;
    }
    if (m_received > 0 && status != 206) {
        // The range was refused, or If-Range found a file that had changed:
        // what follows is the whole thing from the start, so the fragment goes.
        m_file.resize(0);
        m_received = 0;
    }
    const QByteArray etag = m_reply->rawHeader("ETag");
    if (!etag.isEmpty()) {
        m_etag = etag;
    }
    // A partial answer states the whole size behind the slash of
    // "bytes 30932046-3400080613/3400080614"; a full one states it as the length
    // of what it is about to send.
    const QByteArray contentRange = m_reply->rawHeader("Content-Range");
    const int slash = contentRange.lastIndexOf('/');
    if (status == 206 && slash > 0) {
        m_total = contentRange.mid(slash + 1).trimmed().toLongLong();
    } else {
        const qint64 length = m_reply->header(QNetworkRequest::ContentLengthHeader).toLongLong();
        m_total = length > 0 ? m_received + length : 0;
    }
}

void FileDownloader::readData()
{
    if (m_reply == nullptr) {
        return;
    }
    if (!m_headersTaken) {
        m_headersTaken = true;
        takeHeaders();
    }
    // An error page is not the file — and it arrives in as many chunks as it
    // likes, so this has to be asked on every read, not only on the first.
    if (!m_pendingError.isEmpty()) {
        m_reply->readAll();
        return;
    }
    const QByteArray data = m_reply->readAll();
    if (data.isEmpty()) {
        return;
    }
    if (m_file.write(data) != data.size()) {
        // Nothing here will get better by trying again: the partition is full,
        // or the folder went away with the drive it was on.
        fail(i18n("Cannot write to %1 — the disk may be full.", m_file.fileName()));
        return;
    }
    m_received += data.size();
    m_attemptBytes += data.size();
    m_watchdog.start(kStallMs);
    Q_EMIT progress(m_received, m_total);
}

void FileDownloader::replyFinished()
{
    if (m_reply == nullptr) {
        return;
    }
    m_watchdog.stop();
    const QNetworkReply::NetworkError error = m_reply->error();
    const QString errorText = m_reply->errorString();
    m_reply->deleteLater();
    m_reply = nullptr;
    if (m_cancelled) {
        return;
    }
    if (!m_pendingError.isEmpty()) {
        if (m_pendingFatal) {
            fail(m_pendingError);
        } else {
            retryLater(m_pendingError);
        }
        return;
    }
    if (error != QNetworkReply::NoError) {
        retryLater(errorText);
        return;
    }
    complete();
}

void FileDownloader::complete()
{
    m_file.close();
    if (m_total > 0 && m_received < m_total) {
        // A connection can close cleanly in the middle of a file. What is on
        // disk is a fragment like any other, and the next attempt continues it.
        retryLater(i18n("the connection closed before the file was complete"));
        return;
    }
    // rename() does not replace an existing file on Windows, and the final name
    // can be held by a model that failed its checksum.
    QFile::remove(m_dest);
    if (!QFile::rename(partPath(m_dest), m_dest)) {
        fail(i18n("Cannot move the downloaded file to %1.", m_dest));
        return;
    }
    Q_EMIT finished(true, QString());
}

void FileDownloader::retryLater(const QString &reason)
{
    if (m_attemptBytes >= kProgressBytes) {
        // This link works, it is only being interrupted — a download that grows
        // by a hundred megabytes between breaks has to be allowed to finish.
        m_failures = 0;
    } else {
        ++m_failures;
    }
    if (m_failures > kMaxFailures) {
        fail(reason);
        return;
    }
    const int seconds = kBackoffSeconds[qMin(m_failures, kBackoffSteps - 1)];
    Q_EMIT retrying(reason, seconds);
    m_retry.start(seconds * 1000);
}

void FileDownloader::fail(const QString &error)
{
    cleanup();
    // The fragment is kept: the reason may be gone tomorrow, and three
    // gigabytes are worth more than the disk space they sit on.
    Q_EMIT finished(false, error);
}

void FileDownloader::cleanup()
{
    m_watchdog.stop();
    m_retry.stop();
    if (m_reply != nullptr) {
        // Off with the slots first: aborting delivers finished(), and this is
        // often called from inside one of them.
        disconnect(m_reply, nullptr, this, nullptr);
        m_reply->abort();
        m_reply->deleteLater();
        m_reply = nullptr;
    }
    m_file.close();
}
