/*
   SPDX-FileCopyrightText: 2019 (c) Matthieu Gallien <matthieu_gallien@yahoo.fr>
   SPDX-FileCopyrightText: 2025 (c) Jean-Baptiste Mardelle <jb@kdenlive.org>
   SPDX-FileCopyrightText: 2026 (c) Vladislav Radchenko <i@wladradchenko.ru>
   SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "powermanagementinterface.h"
#include "wunjosettings.h"
#include <KLocalizedString>

#if defined Q_OS_WIN
// clang-format off
#include <windows.h>
#include <winbase.h>
// clang-format on
#elif defined Q_OS_MAC
#include <IOKit/pwr_mgt/IOPMLib.h>
#else
#include <QProcess>
#include <QStandardPaths>
#endif

#include <QGuiApplication>
#include <QString>

class PowerManagementInterfacePrivate
{
public:
    bool mPreventSleep = false;
    bool mPreventDim = false;

    bool mInhibitedSleep = false;
    bool mInhibitedDim = false;

#if defined Q_OS_MAC
    IOPMAssertionID mSleepAssertion = 0;
    IOPMAssertionID mDimAssertion = 0;
#elif !defined Q_OS_WIN
    // logind hands out the lock to a process and takes it back when that
    // process goes away, so the handle *is* the process.
    QProcess *mSleepInhibitor = nullptr;
    QProcess *mDimInhibitor = nullptr;
#endif
};

#if !defined Q_OS_WIN && !defined Q_OS_MAC
namespace {
/** @brief Start a `systemd-inhibit` holding @p what ("sleep" or "idle").
 *
 *  `--mode=block` is the real lock rather than the advisory one, and the child
 *  is `sleep infinity` because the lock lives exactly as long as the process
 *  does. Returns nullptr when there is no systemd-inhibit on this machine,
 *  which is not an error: the application simply does not get to keep a
 *  non-logind system awake, exactly as before.
 */
QProcess *startInhibitor(const QString &what, const QString &why, QObject *parent)
{
    static const QString tool = QStandardPaths::findExecutable(QStringLiteral("systemd-inhibit"));
    if (tool.isEmpty()) {
        return nullptr;
    }
    auto *process = new QProcess(parent);
    process->setProgram(tool);
    process->setArguments({QStringLiteral("--what=") + what, QStringLiteral("--who=") + QGuiApplication::applicationDisplayName(),
                           QStringLiteral("--why=") + why, QStringLiteral("--mode=block"), QStringLiteral("sleep"), QStringLiteral("infinity")});
    process->start();
    if (!process->waitForStarted(2000)) {
        delete process;
        return nullptr;
    }
    return process;
}

/** @brief Drop the lock by ending the process that holds it. */
void stopInhibitor(QProcess *&process)
{
    if (process == nullptr) {
        return;
    }
    process->kill();
    // Reaped rather than left behind: a zombie per render would accumulate over
    // a long session.
    process->waitForFinished(2000);
    delete process;
    process = nullptr;
}
} // namespace
#endif

PowerManagementInterface::PowerManagementInterface(QObject *parent)
    : QObject(parent)
    , d(std::make_unique<PowerManagementInterfacePrivate>())
{
}

PowerManagementInterface::~PowerManagementInterface()
{
    // Whatever is held has to be released here, or the machine keeps refusing
    // to sleep after the application is gone.
    uninhibitSleep();
    uninhibitDim();
}

bool PowerManagementInterface::preventSleep() const
{
    return d->mPreventSleep;
}

bool PowerManagementInterface::preventDim() const
{
    return d->mPreventDim;
}

bool PowerManagementInterface::sleepInhibited() const
{
    return d->mInhibitedSleep;
}

bool PowerManagementInterface::dimInhibited() const
{
    return d->mInhibitedDim;
}

void PowerManagementInterface::setPreventSleep(bool value)
{
    if (d->mPreventSleep == value || !WunjoSettings::usePowerManagement()) {
        return;
    }

    if (value) {
        inhibitSleep();
    } else {
        uninhibitSleep();
    }
    d->mPreventSleep = value;

    Q_EMIT preventSleepChanged();
}

void PowerManagementInterface::setPreventDim(bool value)
{
    if (d->mPreventDim == value || !WunjoSettings::usePowerManagement()) {
        return;
    }

    if (value) {
        inhibitDim();
    } else {
        uninhibitDim();
    }
    d->mPreventDim = value;

    Q_EMIT preventDimChanged();
}

void PowerManagementInterface::retryInhibitingSleep()
{
    if (d->mPreventSleep && !d->mInhibitedSleep) {
        inhibitSleep();
    }
}

void PowerManagementInterface::retryInhibitingDim()
{
    if (d->mPreventDim && !d->mInhibitedDim) {
        inhibitDim();
    }
}

void PowerManagementInterface::inhibitSleep()
{
    if (d->mInhibitedSleep) {
        return;
    }
#if defined Q_OS_WIN
    // ES_CONTINUOUS makes the state stick until it is cleared rather than
    // applying to one call.
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
    d->mInhibitedSleep = true;
#elif defined Q_OS_MAC
    const QString why = i18n("Wunjo Make is playing back or rendering");
    const CFStringRef reason = why.toCFString();
    d->mInhibitedSleep = IOPMAssertionCreateWithName(kIOPMAssertionTypeNoIdleSleep, kIOPMAssertionLevelOn, reason, &d->mSleepAssertion) == kIOReturnSuccess;
    CFRelease(reason);
#else
    d->mSleepInhibitor = startInhibitor(QStringLiteral("sleep"), i18n("Wunjo Make is playing back or rendering"), this);
    d->mInhibitedSleep = d->mSleepInhibitor != nullptr;
#endif
    Q_EMIT sleepInhibitedChanged();
}

void PowerManagementInterface::uninhibitSleep()
{
    if (!d->mInhibitedSleep) {
        return;
    }
#if defined Q_OS_WIN
    SetThreadExecutionState(ES_CONTINUOUS);
    // Windows has one execution state for the thread, not one per reason, so
    // clearing sleep clears dim with it — put dim back if it is still wanted.
    if (d->mPreventDim) {
        SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED);
    }
#elif defined Q_OS_MAC
    IOPMAssertionRelease(d->mSleepAssertion);
    d->mSleepAssertion = 0;
#else
    stopInhibitor(d->mSleepInhibitor);
#endif
    d->mInhibitedSleep = false;
    Q_EMIT sleepInhibitedChanged();
}

void PowerManagementInterface::inhibitDim()
{
    if (d->mInhibitedDim) {
        return;
    }
#if defined Q_OS_WIN
    SetThreadExecutionState(ES_CONTINUOUS | ES_DISPLAY_REQUIRED);
    d->mInhibitedDim = true;
#elif defined Q_OS_MAC
    const QString why = i18n("Wunjo Make is playing back");
    const CFStringRef reason = why.toCFString();
    d->mInhibitedDim = IOPMAssertionCreateWithName(kIOPMAssertionTypeNoDisplaySleep, kIOPMAssertionLevelOn, reason, &d->mDimAssertion) == kIOReturnSuccess;
    CFRelease(reason);
#else
    // "idle" is what stops the session going idle, which is what dims the
    // screen and starts the screensaver.
    d->mDimInhibitor = startInhibitor(QStringLiteral("idle"), i18n("Wunjo Make is playing back"), this);
    d->mInhibitedDim = d->mDimInhibitor != nullptr;
#endif
    Q_EMIT dimInhibitedChanged();
}

void PowerManagementInterface::uninhibitDim()
{
    if (!d->mInhibitedDim) {
        return;
    }
#if defined Q_OS_WIN
    SetThreadExecutionState(ES_CONTINUOUS);
    if (d->mPreventSleep) {
        SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);
    }
#elif defined Q_OS_MAC
    IOPMAssertionRelease(d->mDimAssertion);
    d->mDimAssertion = 0;
#else
    stopInhibitor(d->mDimInhibitor);
#endif
    d->mInhibitedDim = false;
    Q_EMIT dimInhibitedChanged();
}
