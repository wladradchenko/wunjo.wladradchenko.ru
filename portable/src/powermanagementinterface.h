/*
   SPDX-FileCopyrightText: 2019 (c) Matthieu Gallien <matthieu_gallien@yahoo.fr>
   SPDX-FileCopyrightText: 2025 (c) Jean-Baptiste Mardelle <jb@kdenlive.org>
   SPDX-FileCopyrightText: 2026 (c) Vladislav Radchenko <i@wladradchenko.ru>
   SPDX-License-Identifier: LGPL-3.0-or-later
 */

#ifndef POWERMANAGEMENTINTERFACE_H
#define POWERMANAGEMENTINTERFACE_H

#include <QObject>

#include <memory>

class PowerManagementInterfacePrivate;

/** @class PowerManagementInterface
    @brief Keeps the machine awake while it is playing back or rendering.

    Each platform is asked in its own way, and none of them through D-Bus:

    - **Linux** holds a `systemd-inhibit` process for as long as the inhibition
      lasts, and kills it to release. That is the same lock the desktop's own
      D-Bus calls take, reached through the tool logind ships for the purpose,
      so the application needs no bus of its own.
    - **Windows** calls `SetThreadExecutionState`.
    - **macOS** takes an `IOPMAssertion`.

    Two things are inhibited separately, because they are separate: *sleep* is
    the machine suspending, *dim* is the screen going dark while the machine
    stays up. A render needs the first; watching playback needs both.
 */
class PowerManagementInterface : public QObject
{

    Q_OBJECT

    Q_PROPERTY(bool preventSleep READ preventSleep WRITE setPreventSleep NOTIFY preventSleepChanged)
    Q_PROPERTY(bool preventDim READ preventDim WRITE setPreventDim NOTIFY preventDimChanged)

    Q_PROPERTY(bool sleepInhibited READ sleepInhibited NOTIFY sleepInhibitedChanged)
    Q_PROPERTY(bool dimInhibited READ dimInhibited NOTIFY dimInhibitedChanged)

public:
    explicit PowerManagementInterface(QObject *parent = nullptr);

    ~PowerManagementInterface() override;

    [[nodiscard]] bool preventSleep() const;
    [[nodiscard]] bool preventDim() const;

    [[nodiscard]] bool sleepInhibited() const;
    [[nodiscard]] bool dimInhibited() const;

Q_SIGNALS:

    void preventSleepChanged();
    void sleepInhibitedChanged();
    void preventDimChanged();
    void dimInhibitedChanged();

public Q_SLOTS:

    void setPreventSleep(bool value);
    void setPreventDim(bool value);

    /** @brief Try again after a failure. An inhibition that could not be taken
     *  the first time — the tool was busy, the assertion was refused — is worth
     *  one more attempt rather than silently leaving the screen to go dark. */
    void retryInhibitingSleep();
    void retryInhibitingDim();

private:
    void inhibitSleep();
    void uninhibitSleep();
    void inhibitDim();
    void uninhibitDim();

    std::unique_ptr<PowerManagementInterfacePrivate> d;
};

#endif // POWERMANAGEMENTINTERFACE_H
