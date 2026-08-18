/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "filedownloadjob.h"

#include "filedownloader.h"

#include <KLocalizedString>

#include <QTimer>

FileDownloadJob::FileDownloadJob(const QUrl &url, const QString &destPath, QObject *parent)
    : KJob(parent)
    , m_downloader(new FileDownloader(this))
    , m_url(url)
    , m_dest(destPath)
{
    // Without this the window has no Pause and no Cancel: the tracker only
    // offers what the job says it can do.
    setCapabilities(KJob::Killable | KJob::Suspendable);

    connect(m_downloader, &FileDownloader::progress, this, &FileDownloadJob::report);
    // The line broke and the download is already waiting to continue. It goes
    // where the window puts a sentence, not where it puts an error.
    connect(m_downloader, &FileDownloader::retrying, this, [this](const QString &reason, int seconds) {
        Q_EMIT infoMessage(this, i18n("%1 — retrying in %2 s", reason, seconds));
    });
    connect(m_downloader, &FileDownloader::finished, this, [this](bool ok, const QString &error) {
        if (m_killed) {
            return; // kill() has already reported this job's end
        }
        if (!ok) {
            setError(error.isEmpty() ? int(KJob::KilledJobError) : int(KJob::UserDefinedError));
            setErrorText(error);
        }
        emitResult();
    });
}

void FileDownloadJob::setHeader(const QByteArray &name, const QByteArray &value)
{
    m_downloader->setHeader(name, value);
}

void FileDownloadJob::start()
{
    // A KJob may not do anything before the caller has had the chance to
    // connect to it, which is what this delay is for.
    QTimer::singleShot(0, this, [this]() {
        // The same two lines KIO's file copy puts in the window, under the same
        // title — this is the window the user knows, and it stays that way.
        Q_EMIT description(this, i18nc("@title job", "Copying"), qMakePair(i18n("Source"), m_url.toString()), qMakePair(i18n("Destination"), m_dest));
        m_since.start();
        m_downloader->start(m_url, m_dest);
    });
}

bool FileDownloadJob::doKill()
{
    m_killed = true;
    m_downloader->cancel();
    return true;
}

bool FileDownloadJob::doSuspend()
{
    m_downloader->pause();
    return true;
}

bool FileDownloadJob::doResume()
{
    m_downloader->resume();
    return true;
}

void FileDownloadJob::report(qint64 received, qint64 total)
{
    setTotalAmount(KJob::Bytes, total);
    setProcessedAmount(KJob::Bytes, received);
    if (m_startBytes < 0) {
        // What a previous attempt left on disk was not downloaded now and must
        // not count towards the speed of this one.
        m_startBytes = received;
    }
    // The window's "1.2 MiB/s" and the time it is left to guess from it. Read
    // over the whole transfer rather than between two chunks: the tracker
    // averages nothing, and a per-chunk figure is unreadable.
    const qint64 elapsed = m_since.elapsed();
    if (elapsed > 1000 && received > m_lastReported) {
        emitSpeed(static_cast<unsigned long>((received - m_startBytes) * 1000 / elapsed));
        m_lastReported = received;
    }
}
