# SPDX-License-Identifier: BSD-2-Clause
# SPDX-FileCopyrightText: Jean-Baptiste Mardelle <jb@kdenlive.org>, Julius Künzel <julius.kuenzel@kde.org>

set(wunjo_SRCS
  ${wunjo_SRCS}
  widgets/choosecolorwidget.cpp
  widgets/colorpickerwidget.cpp
  widgets/doublewidget.cpp
  widgets/dragvalue.cpp
  widgets/elidedfilelinklabel.cpp
  widgets/geometrywidget.cpp
  widgets/markercategorychooser.cpp
  widgets/markercategorybutton.cpp
  widgets/pointwidget.cpp
  widgets/positionwidget.cpp
  widgets/precisionspinbox.cpp
  widgets/progressbutton.cpp
  widgets/timecodedisplay.cpp
  widgets/subtitletextedit.cpp
  PARENT_SCOPE
)

include(ECMAddQtDesignerPlugin)

if (BUILD_DESIGNERPLUGIN)
    ecm_qtdesignerplugin_widget(TimecodeDisplay
        INCLUDE_FILE widgets/timecodedisplay.h
        TOOLTIP "A custom QSpinBox widget to enter timecodes (Wunjo)"
        GROUP "Wunjo Widgets"
    )

    ecm_qtdesignerplugin_widget(ColorPickerWidget
        INCLUDE_FILE widgets/colorpickerwidget.h
        TOOLTIP "A custom QSpinBox widget to enter timecodes (Wunjo)"
        GROUP "Wunjo Widgets"
    )

    ecm_qtdesignerplugin_widget(ChooseColorWidget
        INCLUDE_FILE widgets/choosecolorwidget.h
        TOOLTIP "A custom QSpinBox widget to enter timecodes (Wunjo)"
        GROUP "Wunjo Widgets"
    )

    ecm_qtdesignerplugin_widget(MarkerCategoryChooser
        INCLUDE_FILE widgets/markercategorychooser.h
        TOOLTIP "A QComboBox that can be used to select a marker category (Wunjo)"
        GROUP "Wunjo Widgets"
    )

    ecm_qtdesignerplugin_widget(MarkerCategoryButton
        INCLUDE_FILE widgets/markercategorybutton.h
        TOOLTIP "A QToolButton that can be used to select one or several marker categories (Wunjo)"
        GROUP "Wunjo Widgets"
    )

    ecm_qtdesignerplugin_widget(PrecisionSpinBox
        INCLUDE_FILE widgets/precisionspinbox.h
        TOOLTIP "A double spinbox allowing an arbitrary number of decimals (Wunjo)"
        GROUP "Wunjo Widgets"
    )

    ecm_qtdesignerplugin_widget(ElidedFileLinkLabel
        INCLUDE_FILE widgets/elidedfilelinklabel.h
        TOOLTIP "An elided label with clickable text to open a and highlight a file in the file explorer (Wunjo)"
        GROUP "Wunjo Widgets"
    )

    ecm_add_qtdesignerplugin(wunjowidgets
        NAME WunjoWidgets
        WIDGETS
            TimecodeDisplay
            ColorPickerWidget
            ChooseColorWidget
            MarkerCategoryChooser
            MarkerCategoryButton
            PrecisionSpinBox
            ElidedFileLinkLabel
        LINK_LIBRARIES
            wunjoLib
        INSTALL_DESTINATION "${KDE_INSTALL_QTPLUGINDIR}/designer"
        COMPONENT Devel
    )
endif()
