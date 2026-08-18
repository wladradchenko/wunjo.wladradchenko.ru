# Wunjo monochrome outline icon theme (generated tree, see README.md)
install(DIRECTORY theme/ DESTINATION ${KDE_INSTALL_ICONDIR}/wunjo)

# Light variant of the same theme for the light color theme: identical geometry,
# dark stroke instead of the baked white (#fcfcfc). Generated at configure time
# from theme/ so there is a single committed source tree. WunjoTheme switches
# QIcon::setThemeName between "wunjo" (dark) and "wunjo-light" with the theme.
set(_wunjo_light_dir ${CMAKE_CURRENT_BINARY_DIR}/theme-light)
file(REMOVE_RECURSE ${_wunjo_light_dir})
file(COPY theme/ DESTINATION ${_wunjo_light_dir})
file(GLOB_RECURSE _wunjo_light_svgs ${_wunjo_light_dir}/*.svg)
foreach(_svg ${_wunjo_light_svgs})
    file(READ ${_svg} _svg_content)
    string(REPLACE "#fcfcfc" "#232323" _svg_content "${_svg_content}")
    file(WRITE ${_svg} "${_svg_content}")
endforeach()
# Replace the whole Inherits line rather than a fragment of it: the dark theme
# names several fallbacks and the light one needs them in the opposite order.
# Breeze artwork is authored dark (#232629) and recoloured at runtime only when
# a Plasma platform theme is present, which an AppImage has no reason to carry
# — so on a light background "breeze" is the one that already looks right, and
# on a dark background "breeze-dark" is. Craft ships neither as files: it builds
# breeze-icons with SKIP_INSTALL_ICONS=ON and ICONS_LIBRARY=ON, and the Qt
# resource inside libKF6BreezeIcons.so carries a single theme called "breeze".
# Hence both chains name both, differing only in which comes first.
file(READ theme/index.theme _wunjo_light_index)
string(REGEX REPLACE "Inherits=[^\n]*" "Inherits=breeze,breeze-dark,hicolor" _wunjo_light_index "${_wunjo_light_index}")
string(REPLACE "Name=Wunjo" "Name=Wunjo Light" _wunjo_light_index "${_wunjo_light_index}")
file(WRITE ${_wunjo_light_dir}/index.theme "${_wunjo_light_index}")
install(DIRECTORY ${_wunjo_light_dir}/ DESTINATION ${KDE_INSTALL_ICONDIR}/wunjo-light)
