# SPDX-FileCopyrightText: 2007 Kdenlive contributors
# SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL


### Find packages files
if(WIN32)
    set(MLT_MELTBIN "melt.exe")
    set(MLT_DATADIR "../share/mlt")
elseif(APPLE)
    set(MLT_MELTBIN "melt")
    set(MLT_DATADIR "../Resources/mlt")
else()
    find_package(PkgConfig QUIET)
    execute_process(
        COMMAND ${PKG_CONFIG_EXECUTABLE} --variable=mltdatadir mlt-framework-7
        OUTPUT_VARIABLE MLT_DATADIR
        RESULT_VARIABLE MLT_DATADIR_failed)
    if(NOT MLT_DATADIR_failed)
        string(REGEX REPLACE "[\r\n]" "" MLT_DATADIR "${MLT_DATADIR}")
    endif()

    execute_process(
        COMMAND ${PKG_CONFIG_EXECUTABLE} --variable=meltbin mlt-framework-7
        OUTPUT_VARIABLE MLT_MELTBIN
        RESULT_VARIABLE MLT_MELTBIN_failed)
    if(NOT MLT_MELTBIN_failed)
        string(REGEX REPLACE "[\r\n]" "" MLT_MELTBIN "${MLT_MELTBIN}")
    endif()
endif()
configure_file(mlt_config.h.in ${CMAKE_BINARY_DIR}/mlt_config.h)

check_include_files(linux/input.h HAVE_LINUX_INPUT_H)
if(HAVE_LINUX_INPUT_H)
    list(APPEND wunjo_SRCS
        jogshuttle/jogmanager.cpp
        jogshuttle/jogaction.cpp
        jogshuttle/jogshuttle.cpp
        jogshuttle/jogshuttleconfig.cpp)
endif()

### Find source files
add_subdirectory(abstractmodel)
add_subdirectory(ai)
add_subdirectory(assets)
add_subdirectory(audiomixer)
add_subdirectory(bin)
add_subdirectory(capture)
add_subdirectory(chat)
add_subdirectory(dialogs)
add_subdirectory(doc)
add_subdirectory(effects)
add_subdirectory(jobs)
add_subdirectory(lib)
add_subdirectory(library)
add_subdirectory(mltcontroller)
add_subdirectory(monitor)
add_subdirectory(onlineresources)
add_subdirectory(otio)
add_subdirectory(plugins)
add_subdirectory(profiles)
add_subdirectory(project)
add_subdirectory(pythoninterfaces)
add_subdirectory(render)
add_subdirectory(renderpresets)
add_subdirectory(scopes)
add_subdirectory(scripting)
add_subdirectory(timeline2)
add_subdirectory(titler)
add_subdirectory(transitions)
add_subdirectory(utils)
add_subdirectory(widgets)
add_subdirectory(xml)
add_subdirectory(layouts)

## toplevel source code files
list(APPEND wunjo_SRCS
    core.cpp
    definitions.cpp
    theme.cpp
    mainwindow.cpp
    mainwindowscripting.cpp
    kddocksetup.cpp
    mltconnection.cpp
    statusbarmessagelabel.cpp
    undohelper.cpp
    glaxnimatelauncher.cpp
    filefilter.cpp
    powermanagementinterface.cpp
)

if(CRASH_AUTO_TEST)
    list(APPEND wunjo_SRCS logger.cpp)
endif()

ecm_qt_declare_logging_category(wunjo_SRCS
    HEADER wunjo_debug.h
    IDENTIFIER WUNJO_LOG
    CATEGORY_NAME online.wunjo.make
    OLD_CATEGORY_NAMES org.kde.multimedia.wmproj
    DESCRIPTION "Wunjo"
    EXPORT WUNJO
)

## UI's
file(GLOB wunjo_UIS "ui/*.ui")
ki18n_wrap_ui(wunjo_UIS ${wunjo_UIS})

add_library(wunjoLib STATIC ${wunjo_SRCS} ${wunjo_UIS} definitions.h lib/localeHandling.cpp lib/localeHandling.h)

kconfig_target_kcfg_file(wunjoLib
    FILE wunjosettings.kcfg
    CLASS_NAME WunjoSettings
    MUTATORS
    SINGLETON
    GENERATE_PROPERTIES
    GENERATE_MOC
    QML_REGISTRATION
    DEFAULT_VALUE_GETTERS
)
install(FILES wunjosettings.kcfg DESTINATION ${KDE_INSTALL_KCFGDIR})

ecm_add_qml_module(wunjoLib
    URI online.wunjo.make
    GENERATE_PLUGIN_SOURCE
    DEPENDENCIES
      QtQuick
)

ecm_target_qml_sources(wunjoLib
    SOURCES
    qml/ZoomBar.qml
    monitor/view/AudioView.qml
    monitor/view/AudioZoomBar.qml
    monitor/view/ProjectMonitor.qml
    monitor/view/Countdown.qml
    monitor/view/ClipMonitor.qml
    monitor/view/MonitorAutomask.qml
    monitor/view/MonitorCornerScene.qml
    monitor/view/MonitorGeometryScene.qml
    monitor/view/MonitorOverlay.qml
    monitor/view/MonitorRotoScene.qml
    monitor/view/MonitorSafeZone.qml
    monitor/view/MonitorSplit.qml
    monitor/view/MonitorSplitTracks.qml
    monitor/view/MonitorToolButton.qml
    monitor/view/MonitorZoomButton.qml
    monitor/view/MonitorTrimming.qml
    monitor/view/SceneToolBar.qml
    monitor/view/MultiScreenToolBar.qml
    monitor/view/EffectToolBar.qml
    monitor/view/MaskToolBar.qml
    monitor/view/MonitorRuler.qml
    monitor/view/OverlayStandard.qml
    monitor/view/OverlayMinimal.qml
    monitor/view/OverlayCenter.qml
    monitor/view/OverlayCenterDiagonal.qml
    monitor/view/OverlayThirds.qml
    monitor/view/OverlayLabel.qml
    monitor/view/Safe169.qml
    monitor/view/Safe916.qml
    monitor/view/ResizeHandle.qml
    monitor/view/ResizeLogic.js
    monitor/view/SnappingLogic.js
    monitor/view/RotationHandle.qml
    monitor/view/Utils.js
    timeline2/view/qml/Timeline.qml
    timeline2/view/qml/TimelineLogic.js
    timeline2/view/qml/TrackHead.qml
    timeline2/view/qml/SubtitleTrackHead.qml
    timeline2/view/qml/AudioRecordingControls.qml
    timeline2/view/qml/AudioRecordButton.qml
    timeline2/view/qml/AudioLevels.qml
    timeline2/view/qml/Track.qml
    timeline2/view/qml/Ruler.qml
    timeline2/view/qml/RulerZone.qml
    timeline2/view/qml/Clip.qml
    timeline2/view/qml/SubTitle.qml
    timeline2/view/qml/MixShape.qml
    timeline2/view/qml/ClipThumbs.qml
    timeline2/view/qml/ClipAudioThumbs.qml
    timeline2/view/qml/KeyframeView.qml
    timeline2/view/qml/KeyframeDelegate.qml
    timeline2/view/qml/Composition.qml
    timeline2/view/qml/PulsingAnimation.qml
    timeline2/view/qml/AssetMenu.qml
    timeline2/view/qml/GuidesMenu.qml
    dialogs/Splash.qml
    dialogs/Simplesplash.qml
)

qt_add_resources(wunjo_extra_SRCS icons.qrc uiresources.qrc)

## Icon for Windows and MacOS.
## The svg (or svgz) icon must be provided first to be used on MacOS
set(ICONS_FOLDER ${CMAKE_CURRENT_SOURCE_DIR}/../data/icons)
set(ICON_FILES
  ${ICONS_FOLDER}/sc-apps-wunjo.svg
  ${ICONS_FOLDER}/16-apps-wunjo.png
  ${ICONS_FOLDER}/22-apps-wunjo.png
  ${ICONS_FOLDER}/32-apps-wunjo.png
  ${ICONS_FOLDER}/48-apps-wunjo.png
  ${ICONS_FOLDER}/64-apps-wunjo.png
  ${ICONS_FOLDER}/128-apps-wunjo.png
  ${ICONS_FOLDER}/256-apps-wunjo.png
  ${ICONS_FOLDER}/512-apps-wunjo.png
  ${ICONS_FOLDER}/1024-apps-wunjo.png
)
ecm_add_app_icon(wunjo ICONS ${ICON_FILES})

## Wunjo Executable
qt_add_executable(wunjo MACOSX_BUNDLE main.cpp ${wunjo_extra_SRCS} ${wunjo})
target_include_directories(wunjo
    PRIVATE ${CMAKE_BINARY_DIR}
    )

if (APPLE)
# Apple app package
set_target_properties(wunjo PROPERTIES MACOSX_BUNDLE_INFO_PLIST ${CMAKE_CURRENT_SOURCE_DIR}/../data/Info.plist)
# The bundle's own filename, which is what Finder and /Applications show — the
# plist keys below never touch it. Left as the target name it read "wunjo" on a
# user's Mac while every other platform said Wunjo Make. It renames the
# executable inside Contents/MacOS too, which is right: CFBundleExecutable is
# filled from this and the two have to agree.
#
# The name reaches two other places that must move with it: Craft finds the
# bundle by globbing "<appname>.app" (PackagerBase.getMacAppPath), so the
# blueprint sets appname to match, and the workflow looks for it by name.
set_target_properties(wunjo PROPERTIES OUTPUT_NAME "Wunjo Make")
set_target_properties(wunjo PROPERTIES
    # CFBundleName is what the menu bar shows. Apple asks for 15 characters or
    # fewer there and "Wunjo Make" is ten. The Finder name is CFBundleDisplayName,
    # which is spelled out in data/Info.plist: MACOSX_BUNDLE_DISPLAY_NAME below is
    # not one of the properties CMake substitutes into a plist, so setting it here
    # alone put the name nowhere at all.
    MACOSX_BUNDLE_DISPLAY_NAME "Wunjo Make"
    MACOSX_BUNDLE_BUNDLE_NAME "Wunjo Make"
    MACOSX_BUNDLE_LONG_VERSION_STRING "Wunjo Make ${WUNJO_VERSION}"
    MACOSX_BUNDLE_SHORT_VERSION_STRING "${RELEASE_SERVICE_VERSION_MAJOR}.${RELEASE_SERVICE_VERSION_MINOR}"
    MACOSX_BUNDLE_BUNDLE_VERSION "${WUNJO_VERSION}"
    MACOSX_BUNDLE_GUI_IDENTIFIER "online.wunjo.make"
    MACOSX_BUNDLE_COPYRIGHT "2000-2022 The Wunjo Authors")
#set_target_properties(wunjo PROPERTIES MACOSX_BUNDLE ON)
endif()

target_link_libraries(wunjo PUBLIC wunjoLib wunjoLibplugin)
install(TARGETS wunjo DESTINATION ${KDE_INSTALL_BINDIR})

## Wunjo Lib
target_include_directories(wunjoLib PRIVATE
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/external
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
    ${CMAKE_CURRENT_SOURCE_DIR}/monitor
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/capture
    ${CMAKE_CURRENT_SOURCE_DIR}/bin/model
    ${CMAKE_CURRENT_SOURCE_DIR}/timeline2/view
    ${CMAKE_CURRENT_SOURCE_DIR}/timeline2/view/qml
    )

# Adds Qt definitions and include directories, and sets QT_LIBRARIES according
# to the components requested in find_package().
# KDE definitions and include directories *must* always come first, Qt follows
# (to avoid breaking builds when KDE and/or Qt are installed to different
# prefixes).
# Face detection (src/ai) runs the bundled ONNX model through OpenCV DNN;
# OpenCV is already part of the flatpak dependencies (motion tracker)
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn)
target_link_libraries(wunjoLib PUBLIC opencv_core opencv_imgproc opencv_dnn)
# OpenCV reports errors via exceptions; KDE compiles with -fno-exceptions,
# so this single translation unit needs them back
set_source_files_properties(ai/facedetector.cpp PROPERTIES COMPILE_OPTIONS "-fexceptions")

target_link_libraries(wunjoLib PUBLIC
    KF${KF_MAJOR}::I18n
    KF${KF_MAJOR}::WidgetsAddons
    KF${KF_MAJOR}::Archive
    KF${KF_MAJOR}::CoreAddons
    KF${KF_MAJOR}::KIOCore
    KF${KF_MAJOR}::Codecs
    KF${KF_MAJOR}::KIOFileWidgets
    KF${KF_MAJOR}::KIOWidgets
    KF${KF_MAJOR}::NewStuffWidgets
    KF${KF_MAJOR}::NotifyConfig
    KF${KF_MAJOR}::XmlGui
    KF${KF_MAJOR}::GuiAddons
    KF${KF_MAJOR}::Notifications
    KF${KF_MAJOR}::TextWidgets
    KF${KF_MAJOR}::IconThemes
    KF${KF_MAJOR}::Solid
    KF${KF_MAJOR}::FileMetaData
    KF${KF_MAJOR}::Purpose
    KF${KF_MAJOR}::PurposeWidgets
    ${OPENGL_LIBRARIES}
    ${OPENGLES_LIBRARIES}
    Mlt7::mlt++
    OTIO::opentimelineio
    ${CMAKE_DL_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    kiss_fft
    )

if(KF6_VERSION VERSION_GREATER_EQUAL "6.8.0")
  target_link_libraries(wunjoLib PUBLIC KF${KF_MAJOR}::I18nQml)
endif()

target_link_libraries(wunjoLib PUBLIC
    Qt${QT_MAJOR_VERSION}::Core
    Qt${QT_MAJOR_VERSION}::Svg
    Qt${QT_MAJOR_VERSION}::Quick
    Qt${QT_MAJOR_VERSION}::QuickControls2
    Qt${QT_MAJOR_VERSION}::QuickWidgets
    Qt${QT_MAJOR_VERSION}::Concurrent
    Qt${QT_MAJOR_VERSION}::Multimedia
    Qt${QT_MAJOR_VERSION}::MultimediaWidgets
    Qt${QT_MAJOR_VERSION}::Network
    Qt${QT_MAJOR_VERSION}::NetworkAuth
    Qt${QT_MAJOR_VERSION}::SvgWidgets
    # Qt${QT_MAJOR_VERSION}::Test
)

target_link_libraries(wunjoLib PUBLIC KDAB::kddockwidgets)

target_link_libraries(wunjoLib PUBLIC FFmpeg::FFmpeg)

if(WIN32)
    target_link_libraries(wunjoLib PUBLIC d3d11 d3dcompiler)
endif()

if(BUILD_COVERAGE)
  target_link_libraries(wunjoLib PUBLIC gcov)
endif()
set_property(TARGET wunjoLib PROPERTY CXX_STANDARD 14)
add_definitions(${qt${QT_MAJOR_VERSION}gui_definitions} -DQT_STRICT_ITERATORS -DQT_NO_CAST_FROM_BYTEARRAY)

# Optional deps
if(CRASH_AUTO_TEST)
    target_compile_definitions(wunjoLib PRIVATE CRASH_AUTO_TEST)
    if(TARGET RTTR::Core)
        target_link_libraries(wunjoLib RTTR::Core)
    else()
        target_link_libraries(wunjoLib RTTR::Core_Lib)
    endif()
endif()

if(DRMINGW_FOUND)
    target_compile_definitions(wunjo PRIVATE -DUSE_DRMINGW)
    target_include_directories(wunjo SYSTEM PRIVATE ${DRMINGW_INCLUDE_DIR})
    target_link_libraries(wunjo PUBLIC ${DRMINGW_LIBRARY})
elseif(KF${KF_MAJOR}Crash_FOUND)
    target_compile_definitions(wunjo PRIVATE -DKF5_USE_CRASH)
    target_link_libraries(wunjo PUBLIC KF${KF_MAJOR}::Crash)
endif()

if(Q_WS_X11)
	target_include_directories(wunjoLib SYSTEM PRIVATE ${X11_Xlib_INCLUDE_PATH})
    target_link_libraries(wunjoLib PUBLIC ${X11_LIBRARIES})
endif()

if(SDL2_FOUND)
    target_link_libraries(wunjoLib PUBLIC ${SDL2_LIBRARY})
elseif(SDL_FOUND)
    target_link_libraries(wunjoLib PUBLIC ${SDL_LIBRARY})
endif()

if(HAVE_LINUX_INPUT_H)
    target_compile_definitions(wunjoLib PRIVATE -DUSE_JOGSHUTTLE)
    target_link_libraries(wunjoLib PUBLIC media_ctrl)
endif()

if (BUILD_QCH)
    ecm_add_qch(
        Wunjo_QCH
        NAME wunjo
        VERSION ${WUNJO_VERSION}
        ORG_DOMAIN org.kde
        SOURCE_DIRS
            .
        MD_MAINPAGE "${CMAKE_SOURCE_DIR}/README.md"
        TAGFILE_INSTALL_DESTINATION ${KDE_INSTALL_QTQCHDIR}
        QCH_INSTALL_DESTINATION ${KDE_INSTALL_QTQCHDIR}
        COMPONENT Devel
    )
endif()

add_dependencies(wunjo all_qmllint)

ecm_finalize_qml_module(wunjoLib
    DESTINATION ${KDE_INSTALL_QMLDIR})

