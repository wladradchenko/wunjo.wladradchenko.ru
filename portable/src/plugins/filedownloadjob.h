/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <KJob>

#include <QElapsedTimer>
#include <QString>
#include <QUrl>

class FileDownloader;

/** @class FileDownloadJob
    @brief A @ref FileDownloader dressed as a KJob, so the download shows the
    progress window the application has always shown.

    The window is not ours: it belongs to KJob's tracker, the same one
    `KIO::file_copy` used to be registered with. It draws whatever the job
    reports — Source and Destination, "29.5 MiB of 3.2 GiB complete", the speed,
    Pause and Cancel — so this reports the same fields under the same title and
    the window is the one the user knows, down to the wording.

    What changed is underneath: the bytes come from @ref FileDownloader instead
    of a KIO worker, which is what makes the transfer survive a broken line and
    work on Windows and macOS, where the http worker does not exist. Pause and
    Cancel are honest here — pausing lets go of the connection and keeps the
    fragment, and continuing asks the server for the rest of it.
 */
class FileDownloadJob : public KJob
{
    Q_OBJECT
public:
    /** @param label what the download is called in the window, e.g. the model's
     *  name; the URL is shown as the source. */
    FileDownloadJob(const QUrl &url, const QString &destPath, QObject *parent = nullptr);

    void start() override;
    /** @brief Where the file lands — what a caller reads when the job is done. */
    QString destination() const { return m_dest; }
    /** @brief A header for every request of this download, retries included. */
    void setHeader(const QByteArray &name, const QByteArray &value);

protected:
    bool doKill() override;
    bool doSuspend() override;
    bool doResume() override;

private:
    /** @brief Turn a byte count into the tracker's percentage, size line and
     *  speed reading. */
    void report(qint64 received, qint64 total);

    FileDownloader *m_downloader;
    QUrl m_url;
    QString m_dest;
    /** @brief For the speed reading, which the tracker does not compute itself. */
    QElapsedTimer m_since;
    qint64 m_lastReported = 0;
    /** @brief Where this run started, so a resumed download's speed is what it
     *  is doing now and not what it did last night. */
    qint64 m_startBytes = -1;
    /** @brief kill() reports the result itself; the downloader's own answer to
     *  being cancelled must not report it a second time. */
    bool m_killed = false;
};
