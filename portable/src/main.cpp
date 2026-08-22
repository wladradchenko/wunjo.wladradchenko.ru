 /*
    SPDX-FileCopyrightText: 2007 Marco Gittler <g.marco@freenet.de>
    SPDX-FileCopyrightText: 2008 Jean-Baptiste Mardelle <jb@kdenlive.org>

SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "core.h"
#ifdef CRASH_AUTO_TEST
#include "logger.hpp"
#endif
#include "definitions.h"
#include "dialogs/wizard.h"
#include "wunjo_debug.h"
#include "wunjosettings.h"
#include "theme.h"
// Required for MacOS definition of MLT_LC_NAME
#include "lib/localeHandling.h"
#include "render/renderrequest.h"
#include <config-wunjo.h>
#include <project/projectmanager.h>

#include <mlt++/Mlt.h>

#include <KAboutData>
#include <KConfigGroup>
#ifdef USE_DRMINGW
#include <exchndl.h>
#elif defined(KF5_USE_CRASH)
#include <KCrash>
#endif

#include <KIconLoader>
#include <KIconTheme>
#include <KNotification>
#include <KSandbox>
#include <KSharedConfig>


#include <KStyleManager>
#include <QProxyStyle>
#include <QStyle>
#include <kddockwidgets/DockWidget.h>

#include <KLocalizedString>
#include <QApplication>
#include <QCommandLineOption>
#include <QCommandLineParser>
#include <QDate>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFont>
#include <QFontDatabase>
#include <QIcon>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QQuickStyle>
#include <QStyleFactory>
#include <QQuickWindow>
#include <QResource>

#include <QUndoGroup>
#include <QUrl> //new
#include <QWindow>

#ifdef Q_OS_WIN
extern "C" {
// Inform the driver we could make use of the discrete gpu
// __declspec(dllexport) DWORD NvOptimusEnablement = 0x00000001;
// __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

static LinuxPackageType getPackageType()
{
    QString packageType;
    if (qEnvironmentVariableIsSet("PACKAGE_TYPE")) {
        packageType = qgetenv("PACKAGE_TYPE").toLower();
    }

    if (packageType == QStringLiteral("appimage")) {
        return LinuxPackageType::AppImage;
    }

    if (packageType == QStringLiteral("flatpak")) {
        return LinuxPackageType::Flatpak;
    }

    if (packageType == QStringLiteral("snap")) {
        return LinuxPackageType::Snap;
    }

    if (KSandbox::isFlatpak()) {
        return LinuxPackageType::Flatpak;
    }

    if (KSandbox::isSnap()) {
        return LinuxPackageType::Snap;
    }

    QString appPath = qApp->applicationDirPath();
    if (appPath.contains(QStringLiteral("/tmp/.mount_"))) {
        return LinuxPackageType::AppImage;
    }

    return LinuxPackageType::Unknown;
}

static void resetConfig()
{
    // Delete config file
    KSharedConfigPtr config = KSharedConfig::openConfig();
    if (config->name().contains(QLatin1String("wunjo"))) {
        // Make sure we delete our config file
        QFile f(QStandardPaths::locate(QStandardPaths::GenericConfigLocation, config->name(), QStandardPaths::LocateFile));
        if (f.exists()) {
            qDebug() << " = = = =\nGOT Deleted file: " << f.fileName();
            f.remove();
        }
    }

    // Delete xml ui rc file
    const QString configFile = QStandardPaths::locate(QStandardPaths::GenericDataLocation, QStringLiteral("kxmlgui5/wunjo/ui.rc"));

    if (configFile.isEmpty()) {
        return;
    }
    QFile f(configFile);
    if (!f.open(QIODevice::ReadOnly)) {
        return;
    }

    bool shortcutFound = false;
    QDomDocument doc;
    doc.setContent(&f);
    f.close();
    if (!doc.documentElement().isNull()) {
        QDomElement shortcuts = doc.documentElement().firstChildElement(QStringLiteral("ActionProperties"));
        if (!shortcuts.isNull()) {
            qDebug() << "==== FOUND CUSTOM SHORTCUTS!!!";
            // Copy the original settings and append custom shortcuts
            QFile f2(QStringLiteral(":/kxmlgui5/wunjo/ui.rc"));
            if (f2.exists() && f2.open(QIODevice::ReadOnly)) {
                QDomDocument doc2;
                doc2.setContent(&f2);
                f2.close();
                if (!doc2.documentElement().isNull()) {
                    doc2.documentElement().appendChild(doc2.importNode(shortcuts, true));
                    shortcutFound = true;
                    if (f.open(QIODevice::WriteOnly | QIODevice::Text)) {
                        // overwrite local xml config
                        QTextStream out(&f);
                        out << doc2.toString();
                        f.close();
                    }
                }
            }
        }
    }
    if (!shortcutFound) {
        // No custom shortcuts found, simply delete the xmlui file
        f.remove();
    }
}

// Wunjo: suppress the mnemonic-accelerator underlines (e.g. the "S" in "&Save")
// that some styles draw permanently — the brand UI wants clean labels.
class WunjoProxyStyle : public QProxyStyle
{
public:
    using QProxyStyle::QProxyStyle;
    int styleHint(StyleHint hint, const QStyleOption *option, const QWidget *widget, QStyleHintReturn *returnData) const override
    {
        if (hint == QStyle::SH_UnderlineShortcut) {
            return 0;
        }
        return QProxyStyle::styleHint(hint, option, widget, returnData);
    }
};

class Application : public QApplication
{
public:
    QUrl url;
    Application(int &argc, char **argv)
        : QApplication(argc, argv)
    {
    }

protected:
    bool event(QEvent *event) override
    {
        if (event->type() == QEvent::FileOpen) {
            QFileOpenEvent *openEvent = static_cast<QFileOpenEvent *>(event);
            url = QUrl::fromLocalFile(openEvent->file());
            return true;
        } else
            return QApplication::event(event);
    }
};

int main(int argc, char *argv[])
{
    int result = EXIT_SUCCESS;
#ifdef USE_DRMINGW
    ExcHndlInit();
#endif

#ifdef Q_OS_MAC
    // Launcher and Spotlight on macOS are not setting this environment
    // variable needed by setlocale() as used by MLT.
    if (QProcessEnvironment::systemEnvironment().value(MLT_LC_NAME).isEmpty()) {
        qputenv(MLT_LC_NAME, QLocale().name().toUtf8());

        QLocale localeByName(QLocale(QLocale().language(), QLocale().script(), QLocale().territory()));
        if (QLocale().decimalPoint() != localeByName.decimalPoint()) {
            // If region's numeric format does not match the language's, then we run
            // into problems because we told MLT and libc to use a different numeric
            // locale than actually in use by Qt because it is unable to give numeric
            // locale as a set of ISO-639 codes.
            QLocale::setDefault(localeByName);
            qputenv("LANG", QLocale().name().toUtf8());
        }
    }
#endif

    // Force QDomDocument to use a deterministic XML attribute order
    QHashSeed::setDeterministicGlobalSeed();

#ifdef CRASH_AUTO_TEST
    Logger::init();
#endif

#if defined(Q_OS_WIN)
    QQuickWindow::setGraphicsApi(QSGRendererInterface::Direct3D11);
#elif defined(Q_OS_MACOS)
    QQuickWindow::setGraphicsApi(QSGRendererInterface::Metal);
#else
    QQuickWindow::setGraphicsApi(QSGRendererInterface::OpenGL);
    QCoreApplication::setAttribute(Qt::AA_UseDesktopOpenGL, true);
#endif

    // Block MLT Qt5 module to prevent crashes
    qputenv("MLT_REPOSITORY_DENY", "libmltqt:libmltglaxnimate");

#if defined(Q_OS_WIN)
    QGuiApplication::setHighDpiScaleFactorRoundingPolicy(Qt::HighDpiScaleFactorRoundingPolicy::RoundPreferFloor);
#endif
    // TODO: is it a good option ?
    QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts, true);

    // trigger initialisation of proper icon theme
    KIconTheme::initTheme();

    Application app(argc, argv);

    // Wunjo: force the bundled monochrome outline icon theme (installed to
    // share/icons/wunjo, see data/icons-wunjo/). Unmapped names fall back to
    // the runtime's breeze-dark via the theme's Inherits and the Qt fallback.
    // Must run AFTER the QApplication constructor: KIconTheme::initTheme()
    // registers a pre-routine that re-applies breeze during app construction
    // and would overwrite an earlier setThemeName. KIconLoader (file and
    // mimetype icons in Bin/Media Browser) follows KIconTheme::current()
    // rather than QIcon::themeName, hence the extra force call.
#ifdef Q_OS_MACOS
    // Qt asks the platform theme where icon themes live, and the Cocoa one has
    // nothing to say — there is no XDG icon directory on macOS. So the only
    // place searched is ":/icons", the Qt resource, and the theme installed
    // into the bundle is never looked at. With qt.gui.icon.loader.debug on, the
    // whole application reads:
    //
    //     Probing theme file at ":/icons/wunjo/index.theme" false
    //     Theme "wunjo" not found
    //
    // Every one of those misses then goes to the platform icon engine, which on
    // macOS resolves names as SF Symbols, and AppKit aborts inside
    // NSImageSymbolRepProvider on macOS 13 the first time one is drawn. The
    // application opened and died on the first new project.
    //
    // Linux needs none of this: its platform theme hands Qt the XDG directories
    // and the theme is found where it was installed.
    QIcon::setThemeSearchPaths(QStringList(QDir::cleanPath(QCoreApplication::applicationDirPath()
                                                          + QStringLiteral("/../Resources/icons")))
                               + QIcon::themeSearchPaths());
    // Craft builds breeze-icons as a resource library, and that resource holds
    // exactly one theme — "breeze". There is no "breeze-dark" to fall back to
    // here, so naming it leaves the fallback stage finding nothing at all;
    // every lookup in the log probed ":/icons/breeze-dark/index.theme" and gave
    // up. Linux installs both as files and the darker one is the right choice
    // there.
    QIcon::setFallbackThemeName(QStringLiteral("breeze"));

    QIcon::setThemeName(QStringLiteral("wunjo"));
    KIconTheme::forceThemeForTests(QStringLiteral("wunjo"));
#else
    QIcon::setFallbackThemeName(QStringLiteral("breeze-dark"));
    QIcon::setThemeName(QStringLiteral("wunjo"));
    KIconTheme::forceThemeForTests(QStringLiteral("wunjo"));
#endif

    // Default to org.kde.desktop style unless the user forces another style
    if (qEnvironmentVariableIsEmpty("QT_QUICK_CONTROLS_STYLE")) {
        QQuickStyle::setStyle(QStringLiteral("org.kde.desktop"));
    }

    // trigger initialisation of proper application style
    KStyleManager::initStyle();

    // Wrap the active style so mnemonic accelerators are never underlined.
    // The style is rebuilt from its name rather than wrapped in place, because a
    // proxy takes ownership of its base and setStyle deletes the old one. Breeze
    // leaves objectName empty, and the guard on that silently skipped the wrap
    // altogether — which is why the dock tabs still underlined their first
    // letter.
    QString widgetStyleName;
    if (QStyle *baseStyle = qApp->style()) {
        QString baseName = baseStyle->objectName();
        if (baseName.isEmpty()) {
            baseName = QStringLiteral("breeze");
        }
#ifdef Q_OS_MACOS
        // The bundle ships no Breeze widget style — Contents/PlugIns/styles holds
        // libqmacstyle and nothing else — so KStyleManager leaves the native
        // macOS style in place, and the brand stylesheet, written for Breeze,
        // lands on top of a style it was never meant for. A native style takes
        // its metrics from AppKit rather than from the sheet, which is why a
        // combo box put its text in a corner with no padding around it and the
        // welcome screen drew one icon at several times its size.
        //
        // It also draws through AppKit's NSCell path, and that is where the
        // application died: an assertion inside NSCrackRect, AppKit's own
        // geometry, with nothing of ours on the stack.
        //
        // Breeze is a runtime dependency of the macOS package for exactly this
        // reason, so it should be here. Fusion stands in if a build ever lacks
        // it: compiled into QtWidgets rather than shipped as a plugin, so it is
        // always available, and the same non-native footing Breeze is built on.
        // Either is right; the native style is the one thing that is not.
        // Linux is not touched — there Breeze is present and already chosen.
        baseName = QStyleFactory::keys().contains(QStringLiteral("Breeze"), Qt::CaseInsensitive)
                       ? QStringLiteral("Breeze")
                       : QStringLiteral("Fusion");
#endif
        widgetStyleName = baseName;
        qApp->setStyle(new WunjoProxyStyle(baseName));
    }

    // Load bundled Wunjo brand fonts and set the default UI font
    {
        const QString wunjoFontsDir =
            QStandardPaths::locate(QStandardPaths::GenericDataLocation, QStringLiteral("wunjo/fonts"), QStandardPaths::LocateDirectory);
        if (!wunjoFontsDir.isEmpty()) {
            QDirIterator fontIt(wunjoFontsDir, {QStringLiteral("*.ttf"), QStringLiteral("*.otf")}, QDir::Files, QDirIterator::Subdirectories);
            while (fontIt.hasNext()) {
                QFontDatabase::addApplicationFont(fontIt.next());
            }
        }
        const QString wunjoUiFont = QStringLiteral("Zen Maru Gothic");
        if (QFontDatabase::families().contains(wunjoUiFont)) {
            QFont wunjoFont(wunjoUiFont);
            wunjoFont.setWeight(QFont::Medium);
            qApp->setFont(wunjoFont);
        }
    }

    // Wunjo: global brand stylesheet, layered on top of the Breeze QStyle.
    // Rules and constraints are documented in src/assets/style.qss. The accent
    // and theme greys are baked in as tokens; WunjoTheme::init() (below, once the
    // config file name is known) rewrites them for the persisted theme + accent
    // and calls qApp->setStyleSheet().

    // Try to detect package type
    LinuxPackageType packageType = getPackageType();

    // use a dedicated config file for sandbox packages,
    // however the next lines have no effect if the --config cmd option is used
    QString packageName;
    switch (packageType) {
    case LinuxPackageType::AppImage:
        packageName = QStringLiteral("appimage");
        break;
    case LinuxPackageType::Flatpak:
        packageName = QStringLiteral("flatpak");
        break;
    case LinuxPackageType::Snap:
        packageName = QStringLiteral("snap");
        break;
    default:
        break;
    }
    if (!packageName.isEmpty()) {
        KConfig::setMainConfigName(QStringLiteral("wunjo-%1rc").arg(packageName));
    }

    // Apply the persisted color theme (dark default / light) and primary accent
    // now that the config file name is set — before any window is shown.
    WunjoTheme::instance()->init();

    KLocalizedString::setApplicationDomain("wunjo");

    // Create KAboutData
    QString otherText = i18n("Please report bugs to <a href=\"%1\">%2</a>", QStringLiteral("https://github.com/wladradchenko/wunjo.wladradchenko.ru/issues"),
                             QStringLiteral("github.com/wladradchenko/wunjo.wladradchenko.ru"));

    // The copyright range ends wherever the build happens to be: hardcoding the
    // year means every January the About box quietly starts lying.
    const QString copyright = i18n("Copyright © 2024–%1 Wunjo", QString::number(QDate::currentDate().year()));
    KAboutData aboutData(QByteArray("wunjo"), i18n("Wunjo Make"), WUNJO_VERSION, i18n("An open source software."), KAboutLicense::GPL_V3, copyright,
                         otherText, QStringLiteral("https://wunjo.online"));
    aboutData.addAuthor(i18n("Wlad Radchenko"), i18n("Author and maintainer"), QStringLiteral("i@wladradchenko.ru"));

    // Kept as it was when KDBusService derived the bus name from it: it now only
    // reaches QCoreApplication::organizationDomain, and changing it would move
    // nothing but would invalidate the paths of anyone who already has settings.
    // The address shown to people is the homepage above.
    aboutData.setOrganizationDomain(QByteArray("wunjo.online"));
    // Not bugs.kde.org: reports belong in this project's own tracker, and a URL
    // (rather than a mail address) makes the report wizard open it directly.
    aboutData.setBugAddress(QByteArray("https://github.com/wladradchenko/wunjo.wladradchenko.ru/issues"));
    // What the wizard shows as the application it is reporting against — the
    // display name, not the "wunjo" component id.
    aboutData.setProductName(QByteArray("Wunjo Make"));

    aboutData.addComponent(aboutData.displayName(), QString(), WUNJO_FULL_VERSION_STRING, aboutData.homepage());

    aboutData.addComponent(i18n("MLT"), i18n("Open source multimedia framework."), mlt_version_get_string(),
                           QStringLiteral("https://mltframework.org") /*, KAboutLicense::LGPL_V2_1*/);
    aboutData.addComponent(i18n("FFmpeg"), i18n("A complete, cross-platform solution to record, convert and stream audio and video."), QString(),
                           QStringLiteral("https://ffmpeg.org"));

    aboutData.setDesktopFileName(QStringLiteral("online.wunjo.make"));

    // Set application data
    KAboutData::setApplicationData(aboutData);

    // Without a logo the About dialog falls back to the window icon, and that one
    // carries the dark tile every launcher expects — at the size the dialog draws
    // it, the tile reads as a black frame. So: the same mark without the tile, and
    // in one ink rather than mint, because it sits on the dialog background and
    // should read like the text next to it. Re-set on every theme change — the
    // help menu builds the dialog fresh each time and reads the logo then.
    const auto applyAboutLogo = []() {
        KAboutData about = KAboutData::applicationData();
        about.setProgramLogo(QIcon(WunjoTheme::instance()->isDark() ? QStringLiteral(":/pics/logo-mark-white.svg")
                                                                    : QStringLiteral(":/pics/logo-mark-black.svg")));
        KAboutData::setApplicationData(about);
    };
    applyAboutLogo();
    QObject::connect(WunjoTheme::instance(), &WunjoTheme::themeChanged, &app, applyAboutLogo);
    // Still needed with D-Bus gone: this is what associates the window with its
    // .desktop file, which is how Wayland compositors and task switchers find
    // the application's name and icon.
    QGuiApplication::setDesktopFileName(QStringLiteral("online.wunjo.make"));
#ifndef Q_OS_MACOS // skip this on macOS to have proper mime-type icon visible
    // The scalable icon rather than the 48px raster beside it: the About dialog
    // draws this at 64px and the rasters lag behind data/icons/sc-apps-wunjo.svg,
    // which is where the mark is actually maintained.
    app.setWindowIcon(QIcon(QStringLiteral(":/pics/wunjo.svg")));
#endif

    app.setAttribute(Qt::AA_DontCreateNativeWidgetSiblings, true);

    // Create command line parser with options
    QCommandLineParser parser;
    aboutData.setupCommandLine(&parser);

    // config option is processed in KConfig (src/core/kconfig.cpp)
    parser.addOption(QCommandLineOption(QStringLiteral("config"), i18n("Set a custom config file name."), QStringLiteral("config file")));
    QCommandLineOption mltPathOption(QStringLiteral("mlt-path"), i18n("Set the path for MLT environment."), QStringLiteral("mlt-path"));
    parser.addOption(mltPathOption);
    QCommandLineOption mltLogLevelOption(QStringLiteral("mlt-log"), i18n("Set the MLT log level. Leave this unset for level \"warning\"."),
                                         QStringLiteral("verbose/debug"));
    parser.addOption(mltLogLevelOption);
    QCommandLineOption clipsOption(QStringLiteral("i"), i18n("Comma separated list of files to add as clips to the bin."), QStringLiteral("clips"));
    parser.addOption(clipsOption);

    // render options
    QCommandLineOption renderOption(QStringLiteral("render"), i18n("Directly render the project and exit."));
    parser.addOption(renderOption);

    QCommandLineOption presetOption(QStringLiteral("render-preset"), i18n("Wunjo render preset name (MP4-H264/AAC will be used if none given)."),
                                    QStringLiteral("renderPreset"), QString());
    parser.addOption(presetOption);

    QCommandLineOption exitOption(QStringLiteral("render-async"),
                                  i18n("Exit after (detached) render process started, without this flag it exists only after it finished."));
    parser.addOption(exitOption);

    QCommandLineOption disableWelcome(QStringLiteral("no-welcome"), i18n("Do not show any welcome screen."));
    parser.addOption(disableWelcome);

    QCommandLineOption debugOption(QStringLiteral("debug"), i18n("Show some development specific features in the UI, disable all exclude lists for assets."));
    parser.addOption(debugOption);

    QCommandLineOption saveDebugOption(QStringLiteral("setup-report"), i18n("Save a json report about components in the given path."), QStringLiteral("reportFile"));
    parser.addOption(saveDebugOption);

    parser.addPositionalArgument(QStringLiteral("file"), i18n("Wunjo document to open."));
    parser.addPositionalArgument(QStringLiteral("rendering"), i18n("Output file for rendered video."));

    // Parse command line
    parser.process(app);
    aboutData.processCommandLine(&parser);
    if (parser.isSet(saveDebugOption)) {
        QJsonObject report, property;
        QJsonArray properties;
        const auto components = KAboutData::applicationData().components();
        for (auto &component : components) {
            property["name"] = component.name();
            property["version"] = component.version();
            properties.append(property);
            qDebug() << component.name() << " = " << component.version();
        }
        qDebug() << "Packaging = " << packageName;
        report["components"] = properties;
        report["packageType"] = packageName;

        // Where icons come from, and whether they can be found at all.
        //
        // This belongs in the report because it cannot be seen from outside the
        // application. Qt searches only the directories its platform theme
        // names, and the Cocoa one names none — so the icon theme was installed
        // into the bundle, every file present and correct, and Qt never looked
        // at the directory holding them. Nothing about the package is wrong in
        // that state; the application simply has no icons, each miss goes to
        // the platform engine, and AppKit aborts on the first SF Symbol it
        // cannot draw. Asking the application itself is the only way to tell.
        //
        // QFileInfo answers for ":/icons/..." as readily as for a filesystem
        // path, so a theme carried in a Qt resource counts as found.
        const QString themeName = QIcon::themeName();
        const QStringList iconSearchPaths = QIcon::themeSearchPaths();
        bool themeFound = false;
        for (const QString &path : iconSearchPaths) {
            if (QFileInfo::exists(path + QLatin1Char('/') + themeName + QStringLiteral("/index.theme"))) {
                themeFound = true;
                break;
            }
        }
        QJsonObject icons;
        icons[QStringLiteral("theme")] = themeName;
        // On macOS the Qt theme name is KDE's engine rather than a theme, so the
        // theme that is actually in force is this one. Reported separately or a
        // reader cannot tell a working bundle from a broken one.
        icons[QStringLiteral("kdeTheme")] = KIconTheme::current();
        icons[QStringLiteral("fallbackTheme")] = QIcon::fallbackThemeName();
        icons[QStringLiteral("themeFound")] = themeFound;
        icons[QStringLiteral("searchPaths")] = QJsonArray::fromStringList(iconSearchPaths);
        report[QStringLiteral("icons")] = icons;
        // Which widget style the brand stylesheet is sitting on. A native style
        // here means the sheet is decorating something it was not written for,
        // and the interface comes out wrong in ways no packaging check can see.
        report[QStringLiteral("widgetStyle")] = widgetStyleName;
        const QString outputFilename = parser.value(saveDebugOption);
        if (!outputFilename.isEmpty()) {
            QFile file(outputFilename);
            if (file.exists()) {
                qWarning() << "Cannot overwrite existing file " << outputFilename;
                return EXIT_FAILURE;
            }
            if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
                qWarning() << "Cannot write into file " << outputFilename;
                return EXIT_FAILURE;
            }
            file.write(QJsonDocument(report).toJson());
        } else {
            qCritical() << "You need to provide a valid file path to the --setup-report command line option.";
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }

    QUrl renderUrl;
    QString presetName;
    QStringList clipsToLoad;
    if (parser.positionalArguments().count() != 0) {
        const QString inputFilename = parser.positionalArguments().at(0);
        const QFileInfo fileInfo(inputFilename);
        app.url = QUrl(inputFilename);
        if (fileInfo.exists() || app.url.scheme().isEmpty()) { // easiest way to detect "invalid"/unintended URLs is no scheme
            app.url = QUrl::fromLocalFile(fileInfo.absoluteFilePath());
        }
        if (parser.positionalArguments().count() > 1) {
            if (parser.isSet(renderOption)) {
                // Output render
                const QString outputFilename = parser.positionalArguments().at(1);
                const QFileInfo outFileInfo(outputFilename);
                if (!outFileInfo.exists()) {
                    // easiest way to detect "invalid"/unintended URLs is no scheme
                    renderUrl = QUrl::fromLocalFile(outFileInfo.absoluteFilePath());
                }
            } else {
                // We may want to open several clips in a project
                if (!app.url.isEmpty()) {
                    // Check if first url is not a .wmproj project file
                    if (!app.url.toLocalFile().endsWith(QLatin1String(".wmproj"))) {
                        if (!ProjectManager::isWunjoProjectFile(app.url)) {
                            // We are trying to open clips
                            clipsToLoad << app.url.toLocalFile();
                            app.url.clear();
                        }
                    }
                    for (int i = 1; i < parser.positionalArguments().count(); i++) {
                        const QString outputFilename = parser.positionalArguments().at(i);
                        const QFileInfo outFileInfo(outputFilename);
                        if (outFileInfo.exists()) {
                            clipsToLoad << outFileInfo.absoluteFilePath();
                        }
                    }
                }
            }
        }
    }

    if (parser.isSet(renderOption)) {
        if (app.url.isEmpty()) {
            qCritical() << "You need to give a valid file if you want to render from the command line.";
            return EXIT_FAILURE;
        }
        if (renderUrl.isEmpty()) {
            qCritical() << "You need to give a non existing output file to render from the command line.";
            return EXIT_FAILURE;
        }
        if (parser.isSet(presetOption)) {
            presetName = parser.value(presetOption);
        } else {
            presetName = QStringLiteral("MP4-H264/AAC");
            qDebug() << "No render preset given, using default:" << presetName;
        }
        if (!Core::build(packageType, true)) {
            return EXIT_FAILURE;
        }
        pCore->initHeadless(app.url);
        app.processEvents();

        // ensure we have a proper wunjo_render path, particular important for AppImage
        Wizard::fixWunjoRenderPath();

        RenderRequest *renderrequest = new RenderRequest();
        renderrequest->setOutputFile(renderUrl.toLocalFile());
        renderrequest->loadPresetParams(presetName);
        // request->setPresetParams(m_params);
        renderrequest->setDelayedRendering(false);
        renderrequest->setProxyRendering(false);
        renderrequest->setEmbedSubtitles(false);
        renderrequest->setTwoPass(false);
        renderrequest->setAudioFilePerTrack(false);

        /*bool guideMultiExport = false;
        int guideCategory = m_view.guideCategoryChooser->currentCategory();
        renderrequest->setGuideParams(m_guidesModel, guideMultiExport, guideCategory);*/

        renderrequest->setOverlayData(QString());
        std::vector<RenderRequest::RenderJob> renderjobs = renderrequest->process();
        app.processEvents();

        if (!renderrequest->errorMessages().isEmpty()) {
            qInfo() << "The following errors occurred while trying to render:\n" << renderrequest->errorMessages().join(QLatin1Char('\n'));
        }

        int exitCode = EXIT_SUCCESS;

        for (const auto &job : renderjobs) {
            QStringList argsJob = RenderRequest::argsByJob(job, false);
            if (parser.value(mltLogLevelOption) == QStringLiteral("debug")) {
                argsJob << "--debug";
            }
            qDebug() << "* CREATED JOB WITH ARGS: " << argsJob;
            qDebug() << "starting wunjo_render process using: " << WunjoSettings::wunjorendererpath();
            if (!parser.isSet(exitOption)) {
                if (QProcess::execute(WunjoSettings::wunjorendererpath(), argsJob) != EXIT_SUCCESS) {
                    exitCode = EXIT_FAILURE;
                    break;
                }
            } else {
                if (!QProcess::startDetached(WunjoSettings::wunjorendererpath(), argsJob)) {
                    qCritical() << "Error starting render job" << argsJob;
                    exitCode = EXIT_FAILURE;
                    break;
                } else {
                    KNotification::event(QStringLiteral("RenderStarted"), i18n("Rendering %1 started", job.outputPath), QPixmap());
                }
            }
        }
        /*QMapIterator<QString, QString> i(rendermanager->m_renderFiles);
        while (i.hasNext()) {
            i.next();
            // qDebug() << i.key() << i.value() << rendermanager->startRendering(i.key(), i.value(), {});
        }*/
        pCore->projectManager()->closeCurrentDocument(false, false);
        app.processEvents();
        Core::clean();
        app.processEvents();
        return exitCode;
    }

#ifdef Q_OS_WIN
    QString path = qApp->applicationDirPath() + QLatin1Char(';') + qgetenv("PATH");
    qputenv("PATH", path.toUtf8().constData());
#endif

    if (QQuickWindow::graphicsApi() == QSGRendererInterface::Vulkan) {
        qWarning() << "::: Detected QML VULKAN backend, switching to OpenGL...";
        QQuickWindow::setGraphicsApi(QSGRendererInterface::OpenGL);
    }


    // qApp->processEvents(QEventLoop::AllEvents);

#if defined(KF5_USE_CRASH)
    KCrash::initialize();
#endif

    if (parser.value(mltLogLevelOption) == QStringLiteral("verbose")) {
        mlt_log_set_level(MLT_LOG_VERBOSE);
    } else if (parser.value(mltLogLevelOption) == QStringLiteral("debug")) {
        mlt_log_set_level(MLT_LOG_DEBUG);
    }
    if (parser.isSet(clipsOption)) {
        clipsToLoad = parser.value(clipsOption).split(QLatin1Char(','));
    }

    KDDockWidgets::initFrontend(KDDockWidgets::FrontendType::QtWidgets);

    if (!Core::build(packageType, false, parser.isSet(debugOption), app.url.isEmpty() && clipsToLoad.isEmpty() && !parser.isSet(disableWelcome))) {
        // App is crashing, delete config files and restart
        result = EXIT_CLEAN_RESTART;
    } else {
        pCore->initGUI(parser.value(mltPathOption), app.url, clipsToLoad);
        result = app.exec();
    }
    Core::clean();
    if (result == EXIT_RESTART || result == EXIT_CLEAN_RESTART) {
        qCDebug(WUNJO_LOG) << "restarting app";
        if (result == EXIT_CLEAN_RESTART) {
            resetConfig();
        }
        QStringList progArgs;
        if (argc > 1) {
            // Start at 1 to remove app name
            for (int i = 1; i < argc; i++) {
                progArgs << QString(argv[i]);
            }
        }
        auto *restart = new QProcess;
        restart->start(app.applicationFilePath(), progArgs);
        restart->waitForReadyRead();
        restart->waitForFinished(1000);
        result = EXIT_SUCCESS;
    }
    return result;
}
