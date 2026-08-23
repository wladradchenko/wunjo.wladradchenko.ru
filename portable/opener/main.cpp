/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

/** Opens a file, folder or URL in whatever the desktop uses for it.
 *
 * Installed as `kde-open`, and that name is the whole point. KIO's download
 * window does not open a finished file through a library call — it *runs a
 * program*, and the program it runs is hard-coded. Without one on PATH its
 * "Open File" and "Open Destination" buttons do nothing at all and leave only
 *
 *     kf.jobwidgets: Could not find kde-open executable in PATH
 *
 * The real kde-open ships in kde-cli-tools, which belongs to the Plasma
 * desktop rather than to the runtime an application is built against, and
 * exists on Linux alone. Carrying it would fix one platform and leave the
 * other two, which is the same bargain D-Bus was removed to avoid.
 *
 * So this is not a stand-in for kde-open: it is the same job done portably.
 * QDesktopServices::openUrl already knows what each platform wants — the
 * portal on Linux, ShellExecute on Windows, `open` on macOS — and is what the
 * rest of the application uses to open a folder. This is that one call, in a
 * process of its own, because a process is what KIO insists on.
 */

#include <QDesktopServices>
#include <QDir>
#include <QGuiApplication>
#include <QUrl>

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);
    const QStringList arguments = QCoreApplication::arguments();
    if (arguments.size() < 2) {
        return 1;
    }

    bool opened = true;
    for (int i = 1; i < arguments.size(); ++i) {
        // Both forms have to work: KIO passes what it was given, and that is a
        // "file:///…" URL from one caller and a bare path from the next.
        // AssumeLocalFile keeps a Windows path like C:\… from being read as a
        // URL whose scheme is the drive letter.
        const QUrl url = QUrl::fromUserInput(arguments.at(i), QDir::currentPath(), QUrl::AssumeLocalFile);
        if (!QDesktopServices::openUrl(url)) {
            opened = false;
        }
    }
    return opened ? 0 : 1;
}
