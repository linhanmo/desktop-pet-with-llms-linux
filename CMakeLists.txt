cmake_minimum_required(VERSION 3.21)
project(XiaoMo LANGUAGES CXX)

if(APPLE)
    enable_language(OBJCXX)
endif()

if(WIN32 AND NOT MSVC)
    message(FATAL_ERROR
        "Windows builds currently require MSVC (Visual Studio 2022 / v143 or newer). "
        "The Live2D Cubism Core libraries used by this project are provided as MSVC import libraries, "
        "so MinGW/GCC toolchains are not supported."
    )
endif()

option(AMAIGIRL_ENABLE_APPIMAGE "Enable Linux AppImage packaging target" ON)
option(AMAIGIRL_APPIMAGE_CLEAN "Clean AppDir before packaging AppImage" ON)
option(AMAIGIRL_ENABLE_WINDEPLOYQT "Enable Windows deployment/package targets" ON)
option(AMAIGIRL_ENABLE_LIVE2D "Enable Live2D/Cubism rendering" ON)
set(AMAIGIRL_CUBISM_SDK_DIR "" CACHE PATH "Optional override path to Live2D Cubism SDK root (contains include/ and lib/)")
set(_AMAIGIRL_FETCH_DEPS_DEFAULT OFF)
if(UNIX AND NOT APPLE)
    set(_AMAIGIRL_FETCH_DEPS_DEFAULT ON)
endif()
option(AMAIGIRL_FETCH_LLAMA_CPP "Fetch and build llama.cpp runner (llama-cli) from official GitHub" ${_AMAIGIRL_FETCH_DEPS_DEFAULT})
option(AMAIGIRL_FETCH_SHERPA_ONNX "Fetch sherpa-onnx runtime from official GitHub releases" ${_AMAIGIRL_FETCH_DEPS_DEFAULT})
option(AMAIGIRL_FETCH_APPIMAGE_TOOLS "Fetch linuxdeploy/appimagetool from official GitHub releases when missing" ${_AMAIGIRL_FETCH_DEPS_DEFAULT})
set(AMAIGIRL_LLAMA_CPP_GIT_TAG "master" CACHE STRING "llama.cpp git tag/branch/commit to fetch")
set(AMAIGIRL_LLAMA_CPP_URL_PREFIX "" CACHE STRING "Optional prefix for llama.cpp source tarball URL (e.g. https://ghproxy.com/ )")
set(AMAIGIRL_SHERPA_ONNX_VERSION "1.12.10" CACHE STRING "sherpa-onnx release version to download (without leading 'v')")
option(AMAIGIRL_ENABLE_PORTABLE_LINUX_BUNDLE "Create a portable Linux folder (no conda needed at runtime) via install_portable target" ${_AMAIGIRL_FETCH_DEPS_DEFAULT})
set(AMAIGIRL_LINUXDEPLOY_EXECUTABLE "" CACHE FILEPATH "Optional path override for linuxdeploy executable")
set(AMAIGIRL_APPIMAGETOOL_EXECUTABLE "" CACHE FILEPATH "Optional path override for appimagetool executable")
set(AMAIGIRL_QMAKE_EXECUTABLE "" CACHE FILEPATH "Optional path override for qmake/qmake6 executable")
set(AMAIGIRL_WINDEPLOYQT_EXECUTABLE "" CACHE FILEPATH "Optional path override for windeployqt executable")

# Ensure minimum macOS version for deployment (downward compatibility)
if(APPLE)
    # Set globally so compilers encode the correct LC_BUILD_VERSION minOS
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version to target" FORCE)
    # Build a universal binary on macOS (Intel + Apple Silicon)
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "Build architectures for macOS" FORCE)
endif()

# Linux AppImage packaging (optional)
if(UNIX AND NOT APPLE AND AMAIGIRL_ENABLE_APPIMAGE)
    set(_APPIMAGE_ARCH "x86_64")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(_APPIMAGE_ARCH "aarch64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
        set(_APPIMAGE_ARCH "x86_64")
    else()
        set(_APPIMAGE_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
    endif()

    if(AMAIGIRL_LINUXDEPLOY_EXECUTABLE)
        set(LINUXDEPLOY_EXECUTABLE "${AMAIGIRL_LINUXDEPLOY_EXECUTABLE}")
    else()
        find_program(LINUXDEPLOY_EXECUTABLE
            NAMES
                linuxdeploy
                linuxdeploy.AppImage
                linuxdeploy-x86_64.AppImage
                linuxdeploy-aarch64.AppImage
        )
    endif()

    if(AMAIGIRL_APPIMAGETOOL_EXECUTABLE)
        set(APPIMAGETOOL_EXECUTABLE "${AMAIGIRL_APPIMAGETOOL_EXECUTABLE}")
    else()
        find_program(APPIMAGETOOL_EXECUTABLE
            NAMES
                appimagetool
                appimagetool.AppImage
                appimagetool-x86_64.AppImage
                appimagetool-aarch64.AppImage
        )
    endif()

    if(AMAIGIRL_FETCH_APPIMAGE_TOOLS AND (NOT LINUXDEPLOY_EXECUTABLE OR NOT APPIMAGETOOL_EXECUTABLE))
        set(_AMAIGIRL_TOOLS_DIR "${CMAKE_BINARY_DIR}/_amaigirl_tools")
        file(MAKE_DIRECTORY "${_AMAIGIRL_TOOLS_DIR}")

        function(_amaigirl_download_tool _name _url _out_var)
            set(_dst "${_AMAIGIRL_TOOLS_DIR}/${_name}.AppImage")
            set(_need_download OFF)
            if(NOT EXISTS "${_dst}")
                set(_need_download ON)
            else()
                file(SIZE "${_dst}" _dst_size)
                if(_dst_size EQUAL 0)
                    set(_need_download ON)
                endif()
            endif()
            if(_need_download)
                message(STATUS "Downloading ${_name} from ${_url}")
                file(REMOVE "${_dst}")
                set(_dl_tls_args "")
                if(EXISTS "/etc/ssl/certs/ca-certificates.crt")
                    list(APPEND _dl_tls_args TLS_CAINFO "/etc/ssl/certs/ca-certificates.crt")
                endif()
                file(DOWNLOAD "${_url}" "${_dst}"
                    SHOW_PROGRESS
                    STATUS _dl_status
                    LOG _dl_log
                    ${_dl_tls_args}
                )
                list(GET _dl_status 0 _code)
                list(GET _dl_status 1 _msg)
                if(NOT _code EQUAL 0)
                    message(WARNING "Failed to download ${_name}: ${_msg}\n${_dl_log}")
                else()
                    file(CHMOD "${_dst}"
                        PERMISSIONS
                            OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_EXECUTE
                            WORLD_READ WORLD_EXECUTE
                    )
                endif()
            endif()
            if(EXISTS "${_dst}")
                file(CHMOD "${_dst}"
                    PERMISSIONS
                        OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE
                )
                set(${_out_var} "${_dst}" PARENT_SCOPE)
            endif()
        endfunction()

        if(NOT LINUXDEPLOY_EXECUTABLE)
            _amaigirl_download_tool(
                "linuxdeploy-${_APPIMAGE_ARCH}"
                "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-${_APPIMAGE_ARCH}.AppImage"
                LINUXDEPLOY_EXECUTABLE
            )
        endif()
        if(NOT APPIMAGETOOL_EXECUTABLE)
            _amaigirl_download_tool(
                "appimagetool-${_APPIMAGE_ARCH}"
                "https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-${_APPIMAGE_ARCH}.AppImage"
                APPIMAGETOOL_EXECUTABLE
            )
        endif()
    endif()

    if(LINUXDEPLOY_EXECUTABLE AND APPIMAGETOOL_EXECUTABLE)
        set(AMAIGIRL_APPDIR "${CMAKE_BINARY_DIR}/AppDir")
        set(AMAIGIRL_DESKTOP_FILE "${CMAKE_SOURCE_DIR}/cmake/xiaomo.desktop")
        set(AMAIGIRL_ICON_FILE "${CMAKE_SOURCE_DIR}/res/icons/app-icon.png")

        set(_AMAIGIRL_APPIMAGE_CLEAN_CMD "")
        if(AMAIGIRL_APPIMAGE_CLEAN)
            set(_AMAIGIRL_APPIMAGE_CLEAN_CMD
                COMMAND ${CMAKE_COMMAND} -E rm -rf "${AMAIGIRL_APPDIR}"
            )
        endif()

        if(NOT AMAIGIRL_QMAKE_EXECUTABLE)
            if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/qmake6")
                set(AMAIGIRL_QMAKE_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/qmake6")
            endif()
            set(_AMAIGIRL_QT_BIN_HINTS "")
            foreach(_qt_bin_candidate
                "$ENV{QTDIR}/bin"
                "$ENV{QT_DIR}/bin"
                "$ENV{QT_ROOT_DIR}/bin"
                "$ENV{QT_ROOT}/bin")
                if(_qt_bin_candidate AND EXISTS "${_qt_bin_candidate}")
                    list(APPEND _AMAIGIRL_QT_BIN_HINTS "${_qt_bin_candidate}")
                endif()
            endforeach()

            if(Qt6Core_DIR)
                get_filename_component(_qt6core_dir_real "${Qt6Core_DIR}" REALPATH)
                get_filename_component(_qt6core_cmake_dir "${_qt6core_dir_real}" DIRECTORY)
                get_filename_component(_qt6_lib_dir "${_qt6core_cmake_dir}" DIRECTORY)
                get_filename_component(_qt6_root_dir "${_qt6_lib_dir}" DIRECTORY)
                if(EXISTS "${_qt6_root_dir}/bin")
                    list(APPEND _AMAIGIRL_QT_BIN_HINTS "${_qt6_root_dir}/bin")
                endif()
            endif()

            find_program(AMAIGIRL_QMAKE_EXECUTABLE
                NAMES qmake6 qmake
                HINTS ${_AMAIGIRL_QT_BIN_HINTS}
            )
        endif()

        if(AMAIGIRL_QMAKE_EXECUTABLE)
            set(_AMAIGIRL_LINUXDEPLOY_ENV_QT
                COMMAND ${CMAKE_COMMAND} -E env ARCH=${_APPIMAGE_ARCH} QMAKE=${AMAIGIRL_QMAKE_EXECUTABLE}
            )

            execute_process(
                COMMAND "${AMAIGIRL_QMAKE_EXECUTABLE}" -query QT_INSTALL_PLUGINS
                OUTPUT_VARIABLE AMAIGIRL_QT_PLUGINS_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            execute_process(
                COMMAND "${AMAIGIRL_QMAKE_EXECUTABLE}" -query QT_INSTALL_LIBS
                OUTPUT_VARIABLE AMAIGIRL_QT_LIBS_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        else()
            set(_AMAIGIRL_LINUXDEPLOY_ENV_QT
                COMMAND ${CMAKE_COMMAND} -E env ARCH=${_APPIMAGE_ARCH}
            )
            message(WARNING "qmake not found; linuxdeploy --plugin qt may fail")
        endif()

        set(_AMAIGIRL_COPY_WAYLAND_CMDS "")
        if(AMAIGIRL_QT_PLUGINS_DIR AND EXISTS "${AMAIGIRL_QT_PLUGINS_DIR}")
            set(_AMAIGIRL_QT_PLATFORMS_DIR "${AMAIGIRL_QT_PLUGINS_DIR}/platforms")
            if(EXISTS "${_AMAIGIRL_QT_PLATFORMS_DIR}/libqwayland-generic.so")
                list(APPEND _AMAIGIRL_COPY_WAYLAND_CMDS
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${AMAIGIRL_APPDIR}/usr/plugins/platforms"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_AMAIGIRL_QT_PLATFORMS_DIR}/libqwayland-generic.so" "${AMAIGIRL_APPDIR}/usr/plugins/platforms/libqwayland-generic.so"
                )
            endif()
            if(EXISTS "${_AMAIGIRL_QT_PLATFORMS_DIR}/libqwayland-egl.so")
                list(APPEND _AMAIGIRL_COPY_WAYLAND_CMDS
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${AMAIGIRL_APPDIR}/usr/plugins/platforms"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_AMAIGIRL_QT_PLATFORMS_DIR}/libqwayland-egl.so" "${AMAIGIRL_APPDIR}/usr/plugins/platforms/libqwayland-egl.so"
                )
            endif()

            foreach(_wayland_subdir
                wayland-shell-integration
                wayland-decoration-client
                wayland-graphics-integration-client
                wayland-graphics-integration-server)
                if(EXISTS "${AMAIGIRL_QT_PLUGINS_DIR}/${_wayland_subdir}")
                    list(APPEND _AMAIGIRL_COPY_WAYLAND_CMDS
                        COMMAND ${CMAKE_COMMAND} -E copy_directory "${AMAIGIRL_QT_PLUGINS_DIR}/${_wayland_subdir}" "${AMAIGIRL_APPDIR}/usr/plugins/${_wayland_subdir}"
                    )
                endif()
            endforeach()
        else()
            message(WARNING "Qt plugins directory not found; cannot prebundle Wayland plugins into AppImage")
        endif()

        if(AMAIGIRL_QT_LIBS_DIR AND EXISTS "${AMAIGIRL_QT_LIBS_DIR}")
            list(APPEND _AMAIGIRL_COPY_WAYLAND_CMDS
                COMMAND ${CMAKE_COMMAND} -E make_directory "${AMAIGIRL_APPDIR}/usr/lib"
            )
            foreach(_wayland_lib
                libQt6WaylandClient.so
                libQt6WaylandClient.so.6
                libQt6WaylandEglClientHwIntegration.so
                libQt6WaylandEglClientHwIntegration.so.6)
                if(EXISTS "${AMAIGIRL_QT_LIBS_DIR}/${_wayland_lib}")
                    list(APPEND _AMAIGIRL_COPY_WAYLAND_CMDS
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${AMAIGIRL_QT_LIBS_DIR}/${_wayland_lib}" "${AMAIGIRL_APPDIR}/usr/lib/${_wayland_lib}"
                    )
                endif()
            endforeach()
        else()
            message(WARNING "Qt libs directory not found; cannot prebundle Qt Wayland runtime libs into AppImage")
        endif()

        add_custom_target(package_appimage
            ${_AMAIGIRL_APPIMAGE_CLEAN_CMD}
            COMMAND ${CMAKE_COMMAND} -E make_directory "${AMAIGIRL_APPDIR}/usr/bin"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${AMAIGIRL_APPDIR}/usr/share/icons/hicolor/256x256/apps"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:XiaoMo>" "${AMAIGIRL_APPDIR}/usr/bin/XiaoMo"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/res" "${AMAIGIRL_APPDIR}/usr/bin/res"
            COMMAND ${CMAKE_COMMAND} -E copy "${AMAIGIRL_DESKTOP_FILE}" "${AMAIGIRL_APPDIR}/xiaomo.desktop"
            COMMAND ${CMAKE_COMMAND} -E copy "${AMAIGIRL_ICON_FILE}" "${AMAIGIRL_APPDIR}/xiaomo.png"
            COMMAND ${CMAKE_COMMAND} -E copy "${AMAIGIRL_ICON_FILE}" "${AMAIGIRL_APPDIR}/.DirIcon"
            COMMAND ${CMAKE_COMMAND} -E copy "${AMAIGIRL_ICON_FILE}" "${AMAIGIRL_APPDIR}/usr/share/icons/hicolor/256x256/apps/xiaomo.png"
            ${_AMAIGIRL_LINUXDEPLOY_ENV_QT}
                    "${LINUXDEPLOY_EXECUTABLE}"
                    --appdir "${AMAIGIRL_APPDIR}"
                    --desktop-file "${AMAIGIRL_APPDIR}/xiaomo.desktop"
                    --icon-file "${AMAIGIRL_APPDIR}/xiaomo.png"
                    --executable "${AMAIGIRL_APPDIR}/usr/bin/XiaoMo"
                    --plugin qt
                ${_AMAIGIRL_COPY_WAYLAND_CMDS}
            COMMAND ${CMAKE_COMMAND} -E copy "${AMAIGIRL_DESKTOP_FILE}" "${AMAIGIRL_APPDIR}/xiaomo.desktop"
            COMMAND ${CMAKE_COMMAND} -E copy "${AMAIGIRL_ICON_FILE}" "${AMAIGIRL_APPDIR}/xiaomo.png"
            COMMAND ${CMAKE_COMMAND} -E copy "${AMAIGIRL_ICON_FILE}" "${AMAIGIRL_APPDIR}/.DirIcon"
            COMMAND ${CMAKE_COMMAND} -E copy "${AMAIGIRL_ICON_FILE}" "${AMAIGIRL_APPDIR}/usr/share/icons/hicolor/256x256/apps/xiaomo.png"
            COMMAND ${CMAKE_COMMAND} -E env ARCH=${_APPIMAGE_ARCH}
                    "${LINUXDEPLOY_EXECUTABLE}"
                    --appdir "${AMAIGIRL_APPDIR}"
                    --output appimage
            DEPENDS XiaoMo
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            COMMENT "Packaging AppImage via linuxdeploy"
            VERBATIM
        )

        message(STATUS "Linux AppImage target enabled: package_appimage")
    else()
        message(STATUS "AppImage tools not found (need linuxdeploy + appimagetool), skip package_appimage target")
    endif()
endif()

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt meta build features
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Qt6
# Helper: allow specifying Qt6 path via environment variables for convenience.
if(NOT Qt6_DIR AND DEFINED ENV{Qt6_DIR})
    set(Qt6_DIR "$ENV{Qt6_DIR}")
endif()
if(NOT CMAKE_PREFIX_PATH)
    foreach(_qt_hint_var QTDIR QT_DIR QT_ROOT QT_ROOT_DIR)
        if(DEFINED ENV{${_qt_hint_var}})
            list(APPEND CMAKE_PREFIX_PATH "$ENV{${_qt_hint_var}}")
        endif()
    endforeach()
endif()

# Try to auto-discover a typical Qt6 installation on Windows if not provided
if(WIN32 AND NOT Qt6_DIR)
    set(_QT_CANDIDATE_ROOTS "C:/Qt")
    if(EXISTS "E:/qt")
        list(APPEND _QT_CANDIDATE_ROOTS "E:/qt")
    endif()
    if(DEFINED ENV{ProgramFiles})
        list(APPEND _QT_CANDIDATE_ROOTS "$ENV{ProgramFiles}/Qt")
    endif()
    if(DEFINED ENV{ProgramFiles\(x86\)})
        list(APPEND _QT_CANDIDATE_ROOTS "$ENV{ProgramFiles\(x86\)}/Qt")
    endif()
    foreach(_qt_root IN LISTS _QT_CANDIDATE_ROOTS)
        if(EXISTS "${_qt_root}")
            file(GLOB _qt_versions "${_qt_root}/6.*")
            foreach(_qt_ver_dir IN LISTS _qt_versions)
                set(_qt_toolchains)
                if(MSVC)
                    file(GLOB _qt_toolchains "${_qt_ver_dir}/msvc*")
                elseif(MINGW)
                    file(GLOB _qt_toolchains "${_qt_ver_dir}/mingw*")
                else()
                    file(GLOB _qt_toolchains
                        "${_qt_ver_dir}/msvc*"
                        "${_qt_ver_dir}/mingw*"
                    )
                endif()
                foreach(_qt_tc IN LISTS _qt_toolchains)
                    if(EXISTS "${_qt_tc}/lib/cmake/Qt6")
                        set(Qt6_DIR "${_qt_tc}/lib/cmake/Qt6")
                        list(APPEND CMAKE_PREFIX_PATH "${_qt_tc}")
                        break()
                    endif()
                endforeach()
                if(Qt6_DIR)
                    break()
                endif()
            endforeach()
        endif()
        if(Qt6_DIR)
            break()
        endif()
    endforeach()
    if(Qt6_DIR)
        message(STATUS "Detected Qt6_DIR: ${Qt6_DIR}")
    else()
        message(STATUS "Qt6 auto-discovery did not find a local installation; please set Qt6_DIR or CMAKE_PREFIX_PATH.")
    endif()
endif()

set(_AMAIGIRL_QT_COMPONENTS Core Gui Widgets Svg Multimedia)
if(AMAIGIRL_ENABLE_LIVE2D)
    list(APPEND _AMAIGIRL_QT_COMPONENTS OpenGL OpenGLWidgets)
endif()
find_package(Qt6 REQUIRED COMPONENTS ${_AMAIGIRL_QT_COMPONENTS})

if(UNIX AND NOT APPLE AND AMAIGIRL_ENABLE_LIVE2D)
    set(OpenGL_GL_PREFERENCE LEGACY)
    find_package(OpenGL REQUIRED)
    if(NOT TARGET OpenGL::GL AND OPENGL_gl_LIBRARY)
        add_library(OpenGL::GL UNKNOWN IMPORTED)
        set_target_properties(OpenGL::GL PROPERTIES
            IMPORTED_LOCATION "${OPENGL_gl_LIBRARY}"
        )
    endif()
endif()

if(MSVC AND (Qt6_DIR MATCHES "mingw" OR Qt6Core_DIR MATCHES "mingw"))
    message(FATAL_ERROR
        "Qt toolchain mismatch: MSVC build detected but Qt appears to be a MinGW build.\n"
        "Please install the MSVC Qt kit (e.g. 6.x.x\\msvc2022_64) and set Qt6_DIR to:\n"
        "  <Qt>\\6.x.x\\msvc2022_64\\lib\\cmake\\Qt6\n"
        "Current Qt6_DIR='${Qt6_DIR}'\n"
        "Current Qt6Core_DIR='${Qt6Core_DIR}'"
    )
endif()

# Fetch/build runtime dependencies for Linux from official GitHub repos
set(AMAIGIRL_LLAMA_CLI_PATH "")
set(AMAIGIRL_SHERPA_RUNTIME_ROOT "")
set(AMAIGIRL_SHERPA_RUNTIME_NAME "")
if(UNIX AND NOT APPLE AND (AMAIGIRL_FETCH_LLAMA_CPP OR AMAIGIRL_FETCH_SHERPA_ONNX))
    include(ExternalProject)
    set(AMAIGIRL_DEPS_DIR "${CMAKE_BINARY_DIR}/_amaigirl_deps")
    set(_AMAIGIRL_EP_DOWNLOAD_EXTRACT_TS_ARGS "")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(_AMAIGIRL_EP_DOWNLOAD_EXTRACT_TS_ARGS DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    endif()

    if(AMAIGIRL_FETCH_LLAMA_CPP)
        set(_AMAIGIRL_LLAMA_URL "")
        if(AMAIGIRL_LLAMA_CPP_GIT_TAG STREQUAL "master" OR AMAIGIRL_LLAMA_CPP_GIT_TAG STREQUAL "main")
            set(_AMAIGIRL_LLAMA_URL "https://codeload.github.com/ggerganov/llama.cpp/tar.gz/refs/heads/${AMAIGIRL_LLAMA_CPP_GIT_TAG}")
        else()
            set(_AMAIGIRL_LLAMA_URL "https://codeload.github.com/ggerganov/llama.cpp/tar.gz/${AMAIGIRL_LLAMA_CPP_GIT_TAG}")
        endif()
        if(AMAIGIRL_LLAMA_CPP_URL_PREFIX)
            set(_AMAIGIRL_LLAMA_URL "${AMAIGIRL_LLAMA_CPP_URL_PREFIX}${_AMAIGIRL_LLAMA_URL}")
        endif()

        ExternalProject_Add(amaigirl_llama_cpp
            URL "${_AMAIGIRL_LLAMA_URL}"
            ${_AMAIGIRL_EP_DOWNLOAD_EXTRACT_TS_ARGS}
            UPDATE_COMMAND ""
            SOURCE_DIR "${AMAIGIRL_DEPS_DIR}/llama.cpp-src"
            BINARY_DIR "${AMAIGIRL_DEPS_DIR}/llama.cpp-build"
            CMAKE_ARGS
                -DCMAKE_BUILD_TYPE=Release
                -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
                -DCMAKE_INSTALL_RPATH=$ORIGIN
                -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON
                -DLLAMA_BUILD_TESTS=OFF
                -DLLAMA_BUILD_EXAMPLES=ON
            INSTALL_COMMAND ""
        )
        set(AMAIGIRL_LLAMA_CLI_PATH "${AMAIGIRL_DEPS_DIR}/llama.cpp-build/bin/llama-cli")
    endif()

    if(AMAIGIRL_FETCH_SHERPA_ONNX)
        set(_AMAIGIRL_SHERPA_PLATFORM "linux-x64")
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
            set(_AMAIGIRL_SHERPA_PLATFORM "linux-aarch64")
        endif()
        set(AMAIGIRL_SHERPA_RUNTIME_NAME "sherpa-onnx-v${AMAIGIRL_SHERPA_ONNX_VERSION}-${_AMAIGIRL_SHERPA_PLATFORM}-shared")
        set(_AMAIGIRL_SHERPA_URL "https://github.com/k2-fsa/sherpa-onnx/releases/download/v${AMAIGIRL_SHERPA_ONNX_VERSION}/${AMAIGIRL_SHERPA_RUNTIME_NAME}.tar.bz2")

        ExternalProject_Add(amaigirl_sherpa_onnx_runtime
            URL "${_AMAIGIRL_SHERPA_URL}"
            ${_AMAIGIRL_EP_DOWNLOAD_EXTRACT_TS_ARGS}
            UPDATE_COMMAND ""
            SOURCE_DIR "${AMAIGIRL_DEPS_DIR}/sherpa-onnx"
            CONFIGURE_COMMAND ""
            BUILD_COMMAND ""
            INSTALL_COMMAND ""
        )
        set(AMAIGIRL_SHERPA_RUNTIME_ROOT "${AMAIGIRL_DEPS_DIR}/sherpa-onnx")
    endif()
endif()

# Live2D Cubism Core SDK
if(AMAIGIRL_CUBISM_SDK_DIR)
    set(CUBISM_SDK_DIR "${AMAIGIRL_CUBISM_SDK_DIR}")
else()
    set(CUBISM_SDK_DIR "${CMAKE_SOURCE_DIR}/sdk/cubism")
endif()
set(CUBISM_INCLUDE_DIR "${CUBISM_SDK_DIR}/include")

set(CUBISM_LIB_DIR "")
set(CUBISM_CORE_LIB "")
set(CUBISM_CORE_LIB_DEBUG "")
set(CUBISM_CORE_LIB_RELEASE "")
set(CUBISM_CORE_DLL_WINDOWS_X64 "")

set(AMAIGIRL_ENABLE_LIVE2D_EFFECTIVE "${AMAIGIRL_ENABLE_LIVE2D}")
if(AMAIGIRL_ENABLE_LIVE2D_EFFECTIVE)
    if(NOT EXISTS "${CUBISM_INCLUDE_DIR}/Live2DCubismCore.h")
        set(AMAIGIRL_ENABLE_LIVE2D_EFFECTIVE OFF)
        message(WARNING "Live2D Cubism SDK not found: missing header '${CUBISM_INCLUDE_DIR}/Live2DCubismCore.h'. Live2D will be disabled for this build.")
    endif()
endif()

if(AMAIGIRL_ENABLE_LIVE2D_EFFECTIVE AND APPLE)
    set(_CUBISM_MACOS_ARM64_LIB "${CUBISM_SDK_DIR}/lib/macos/arm64/libLive2DCubismCore.a")
    set(_CUBISM_MACOS_X64_LIB "${CUBISM_SDK_DIR}/lib/macos/x86_64/libLive2DCubismCore.a")

    if(CMAKE_OSX_ARCHITECTURES MATCHES "arm64" AND CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
        find_program(LIPO_EXECUTABLE lipo)
        set(CUBISM_LIB_DIR "${CMAKE_BINARY_DIR}/cubism")
        set(CUBISM_CORE_LIB "${CUBISM_LIB_DIR}/libLive2DCubismCore_universal.a")

        if(NOT LIPO_EXECUTABLE)
            message(FATAL_ERROR "lipo not found, cannot build universal Cubism library")
        endif()
        if(NOT EXISTS "${_CUBISM_MACOS_ARM64_LIB}")
            message(FATAL_ERROR "Cubism arm64 library not found: ${_CUBISM_MACOS_ARM64_LIB}")
        endif()
        if(NOT EXISTS "${_CUBISM_MACOS_X64_LIB}")
            message(FATAL_ERROR "Cubism x86_64 library not found: ${_CUBISM_MACOS_X64_LIB}")
        endif()

        add_custom_command(
            OUTPUT "${CUBISM_CORE_LIB}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CUBISM_LIB_DIR}"
            COMMAND "${LIPO_EXECUTABLE}" -create
                    "${_CUBISM_MACOS_ARM64_LIB}"
                    "${_CUBISM_MACOS_X64_LIB}"
                    -output "${CUBISM_CORE_LIB}"
            DEPENDS "${_CUBISM_MACOS_ARM64_LIB}" "${_CUBISM_MACOS_X64_LIB}"
            COMMENT "Creating universal Cubism Core static library"
            VERBATIM
        )
        add_custom_target(cubism_core_universal DEPENDS "${CUBISM_CORE_LIB}")
    elseif(CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
        set(CUBISM_LIB_DIR "${CUBISM_SDK_DIR}/lib/macos/arm64")
        set(CUBISM_CORE_LIB "${_CUBISM_MACOS_ARM64_LIB}")
    elseif(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
        set(CUBISM_LIB_DIR "${CUBISM_SDK_DIR}/lib/macos/x86_64")
        set(CUBISM_CORE_LIB "${_CUBISM_MACOS_X64_LIB}")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
        set(CUBISM_LIB_DIR "${CUBISM_SDK_DIR}/lib/macos/arm64")
        set(CUBISM_CORE_LIB "${_CUBISM_MACOS_ARM64_LIB}")
    else()
        set(CUBISM_LIB_DIR "${CUBISM_SDK_DIR}/lib/macos/x86_64")
        set(CUBISM_CORE_LIB "${_CUBISM_MACOS_X64_LIB}")
    endif()
elseif(AMAIGIRL_ENABLE_LIVE2D_EFFECTIVE AND WIN32)
    # Cubism Windows libs are grouped by MSVC toolset (e.g. 143)
    set(_CUBISM_WIN_TOOLSET "143")
    if(DEFINED MSVC_TOOLSET_VERSION AND EXISTS "${CUBISM_SDK_DIR}/lib/windows/x86_64/${MSVC_TOOLSET_VERSION}")
        set(_CUBISM_WIN_TOOLSET "${MSVC_TOOLSET_VERSION}")
    endif()
    set(CUBISM_LIB_DIR "${CUBISM_SDK_DIR}/lib/windows/x86_64/${_CUBISM_WIN_TOOLSET}")
    set(CUBISM_CORE_LIB_RELEASE "${CUBISM_LIB_DIR}/Live2DCubismCore_MD.lib")
    set(CUBISM_CORE_LIB_DEBUG "${CUBISM_LIB_DIR}/Live2DCubismCore_MDd.lib")
    file(GLOB_RECURSE _CUBISM_CORE_DLL_CANDIDATES
        "${CUBISM_SDK_DIR}/CubismSdkForNative-*/Core/dll/windows/x86_64/Live2DCubismCore.dll"
        "${CUBISM_SDK_DIR}/Core/dll/windows/x86_64/Live2DCubismCore.dll"
    )
    list(LENGTH _CUBISM_CORE_DLL_CANDIDATES _CUBISM_CORE_DLL_COUNT)
    if(_CUBISM_CORE_DLL_COUNT GREATER 0)
        list(GET _CUBISM_CORE_DLL_CANDIDATES 0 CUBISM_CORE_DLL_WINDOWS_X64)
    endif()
elseif(AMAIGIRL_ENABLE_LIVE2D_EFFECTIVE AND UNIX)
    set(CUBISM_LIB_DIR "${CUBISM_SDK_DIR}/lib/linux/x86_64")
    set(CUBISM_CORE_LIB "${CUBISM_LIB_DIR}/libLive2DCubismCore.a")
endif()

if(AMAIGIRL_ENABLE_LIVE2D_EFFECTIVE)
    # Import static Cubism Core library
    add_library(Live2DCubismCore STATIC IMPORTED GLOBAL)
    if(WIN32)
        if(EXISTS "${CUBISM_CORE_LIB_RELEASE}")
            set_target_properties(Live2DCubismCore PROPERTIES
                IMPORTED_LOCATION_RELEASE "${CUBISM_CORE_LIB_RELEASE}"
                INTERFACE_INCLUDE_DIRECTORIES "${CUBISM_INCLUDE_DIR}"
            )
            if(EXISTS "${CUBISM_CORE_LIB_DEBUG}")
                set_target_properties(Live2DCubismCore PROPERTIES
                    IMPORTED_LOCATION_DEBUG "${CUBISM_CORE_LIB_DEBUG}"
                )
            else()
                set_target_properties(Live2DCubismCore PROPERTIES
                    IMPORTED_LOCATION_DEBUG "${CUBISM_CORE_LIB_RELEASE}"
                )
            endif()
        else()
            message(FATAL_ERROR "Cubism Core library not found for Windows: ${CUBISM_CORE_LIB_RELEASE}")
        endif()
    elseif(EXISTS "${CUBISM_CORE_LIB}" OR (APPLE AND TARGET cubism_core_universal))
        set_target_properties(Live2DCubismCore PROPERTIES
            IMPORTED_LOCATION "${CUBISM_CORE_LIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUBISM_INCLUDE_DIR}"
        )
        if(APPLE AND TARGET cubism_core_universal)
            add_dependencies(Live2DCubismCore cubism_core_universal)
        endif()
    else()
        set(AMAIGIRL_ENABLE_LIVE2D_EFFECTIVE OFF)
        message(WARNING "Cubism Core library not found for this platform. Expected in: ${CUBISM_LIB_DIR}. Live2D will be disabled for this build.")
    endif()

    if(TARGET Live2DCubismCore)
        # Map unspecified multi-configs to Release to satisfy imported target requirements
        set_target_properties(Live2DCubismCore PROPERTIES
            MAP_IMPORTED_CONFIG_MINSIZEREL Release
            MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
        )
    endif()
endif()

# Sources (reflect new folder layout under src/)
set(APP_SOURCES
    src/app/main.cpp
    src/app/Version.hpp
)
set(ENGINE_SOURCES)
if(AMAIGIRL_ENABLE_LIVE2D_EFFECTIVE)
    set(ENGINE_SOURCES
        src/engine/Renderer.hpp
        src/engine/Renderer.cpp
        src/engine/Live2DCore.hpp
        src/engine/Live2DCore.cpp
        src/engine/Model.hpp
        src/engine/Model.cpp
        src/engine/Motion.hpp
        src/engine/Motion.cpp
        src/engine/Expression.hpp
        src/engine/Expression.cpp
        src/engine/PhysicsEngine.hpp
        src/engine/PhysicsEngine.cpp
    )
endif()
set(UI_SOURCES
    src/ui/theme/ThemeApi.hpp
    src/ui/theme/ThemeApi.cpp
    src/ui/theme/ThemeWidgets.hpp
    src/ui/era-style/EraStyleColor.hpp
    src/ui/era-style/EraStyleHelper.hpp
    src/ui/era-style/EraStyleHelper.cpp
    src/ui/era-style/EraChatWidgets.hpp
    src/ui/era-style/EraChatWidgets.cpp
    src/ui/era-style/EraButton.hpp
    src/ui/era-style/EraButton.cpp
    src/ui/era-style/EraTextEdit.hpp
    src/ui/era-style/EraTextEdit.cpp
    src/ui/era-style/EraLineEdit.hpp
    src/ui/era-style/EraLineEdit.cpp
    src/ui/era-style/EraPlainTextEdit.hpp
    src/ui/era-style/EraPlainTextEdit.cpp
    src/ui/era-style/EraSwitch.hpp
    src/ui/era-style/EraSwitch.cpp
    src/ui/era-style/EraComboBox.hpp
    src/ui/era-style/EraComboBox.cpp
    src/ui/era-style/EraTabBar.hpp
    src/ui/era-style/EraTabBar.cpp
    src/ui/SettingsWindow.hpp
    src/ui/SettingsWindow.cpp
    src/ui/ChatWindow.hpp
    src/ui/ChatWindow.cpp
)

if(APPLE)
    list(APPEND UI_SOURCES
        src/ui/era-style/EraMacWindowTheme.hpp
        src/ui/era-style/EraMacWindowTheme.mm
    )
endif()

if(WIN32)
    list(APPEND UI_SOURCES
        src/ui/era-style/EraWinWindowTheme.cpp
    )
endif()

if(UNIX AND NOT APPLE)
    list(APPEND UI_SOURCES
        src/ui/era-style/EraLinuxWindowTheme.cpp
    )
endif()
set(COMMON_SOURCES
    src/common/Utils.hpp
    src/common/Utils.cpp
    src/common/SettingsManager.hpp
    src/common/SettingsManager.cpp
)

# AI
set(AI_SOURCES
    src/ai/ChatController.hpp
    src/ai/ChatController.cpp
    src/ai/LocalLlmClient.hpp
    src/ai/LocalLlmClient.cpp
)

set(AUDIO_SOURCES
    src/audio/OfflineVoiceService.hpp
    src/audio/OfflineVoiceService.cpp
)

if(APPLE)
    add_executable(XiaoMo MACOSX_BUNDLE
        ${APP_SOURCES}
        ${ENGINE_SOURCES}
        ${UI_SOURCES}
        ${COMMON_SOURCES}
        ${AI_SOURCES}
        ${AUDIO_SOURCES}
    )
elseif(WIN32)
    add_executable(XiaoMo WIN32
        ${APP_SOURCES}
        ${ENGINE_SOURCES}
        ${UI_SOURCES}
        ${COMMON_SOURCES}
        ${AI_SOURCES}
        ${AUDIO_SOURCES}
    )
else()
    add_executable(XiaoMo
        ${APP_SOURCES}
        ${ENGINE_SOURCES}
        ${UI_SOURCES}
        ${COMMON_SOURCES}
        ${AI_SOURCES}
        ${AUDIO_SOURCES}
    )
endif()

if(WIN32)
    target_sources(XiaoMo PRIVATE
        cmake/AmaiGirl.rc
    )
endif()

# Also set per-target deployment target as a safeguard
if(APPLE)
    set_target_properties(XiaoMo PROPERTIES MACOSX_DEPLOYMENT_TARGET "13.0")
    if(TARGET cubism_core_universal)
        add_dependencies(XiaoMo cubism_core_universal)
    endif()
endif()

# Include dirs: single root only (Scheme A)
set(SRC_ROOT "${PROJECT_SOURCE_DIR}/src")
target_include_directories(XiaoMo PRIVATE ${SRC_ROOT})
if(AMAIGIRL_ENABLE_LIVE2D_EFFECTIVE)
    target_include_directories(XiaoMo PRIVATE ${CUBISM_INCLUDE_DIR})
endif()

if(MSVC)
    target_compile_options(XiaoMo PRIVATE /Zc:__cplusplus)
endif()

if(WIN32)
    target_compile_definitions(XiaoMo PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

target_link_libraries(XiaoMo PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Svg
)

target_link_libraries(XiaoMo PRIVATE Qt6::Multimedia)
target_compile_definitions(XiaoMo PRIVATE AMAIGIRL_USE_QT_MULTIMEDIA=1)

if(AMAIGIRL_ENABLE_LIVE2D_EFFECTIVE)
    target_link_libraries(XiaoMo PRIVATE
        Qt6::OpenGL
        Qt6::OpenGLWidgets
        Live2DCubismCore
    )
endif()

target_compile_definitions(XiaoMo PRIVATE AMAIGIRL_ENABLE_LIVE2D=$<IF:$<BOOL:${AMAIGIRL_ENABLE_LIVE2D_EFFECTIVE}>,1,0>)

if(UNIX AND NOT APPLE AND AMAIGIRL_ENABLE_LIVE2D_EFFECTIVE AND OPENGL_INCLUDE_DIR)
    target_include_directories(XiaoMo PRIVATE "${OPENGL_INCLUDE_DIR}")
endif()

if(WIN32)
    target_link_libraries(XiaoMo PRIVATE dwmapi windowsapp)

    set(_AMAIGIRL_SHERPA_ROOT "${CMAKE_SOURCE_DIR}/res/voice_deps/sherpa-onnx-v1.12.10-win-x64-shared")
endif()

# macOS specifics
if(APPLE)
    # Link common macOS frameworks if available
    find_library(APPKIT_FRAMEWORK AppKit)
    if(APPKIT_FRAMEWORK)
        target_link_libraries(XiaoMo PRIVATE ${APPKIT_FRAMEWORK})
    endif()
    find_library(OPENGL_FRAMEWORK OpenGL)
    if(OPENGL_FRAMEWORK)
        target_link_libraries(XiaoMo PRIVATE ${OPENGL_FRAMEWORK})
    endif()
    # For CGEvent (ApplicationServices)
    find_library(APPSERVICES_FRAMEWORK ApplicationServices)
    if(APPSERVICES_FRAMEWORK)
        target_link_libraries(XiaoMo PRIVATE ${APPSERVICES_FRAMEWORK})
    endif()

    # For bundled apps, this RPATH helps in Release after deployment
    set_target_properties(XiaoMo PROPERTIES
        INSTALL_RPATH "@executable_path/../Frameworks"
    )
endif()

# macOS app icon (.icns) generation from PNG and bundle embedding
if(APPLE)
    set(APP_ICON_PNG "${CMAKE_SOURCE_DIR}/res/icons/app-icon.png")
    if(EXISTS "${APP_ICON_PNG}")
        find_program(SIPS_EXECUTABLE sips)
        find_program(ICONUTIL_EXECUTABLE iconutil)
        if(SIPS_EXECUTABLE AND ICONUTIL_EXECUTABLE)
            set(APP_ICONSET_DIR "${CMAKE_CURRENT_BINARY_DIR}/appicon.iconset")
            set(APP_ICNS "${CMAKE_CURRENT_BINARY_DIR}/appicon.icns")
            add_custom_command(OUTPUT "${APP_ICNS}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${APP_ICONSET_DIR}"
                COMMAND "${SIPS_EXECUTABLE}" -s format png -z 16 16     "${APP_ICON_PNG}" --out "${APP_ICONSET_DIR}/icon_16x16.png"
                COMMAND "${SIPS_EXECUTABLE}" -s format png -z 32 32     "${APP_ICON_PNG}" --out "${APP_ICONSET_DIR}/icon_16x16@2x.png"
                COMMAND "${SIPS_EXECUTABLE}" -s format png -z 32 32     "${APP_ICON_PNG}" --out "${APP_ICONSET_DIR}/icon_32x32.png"
                COMMAND "${SIPS_EXECUTABLE}" -s format png -z 64 64     "${APP_ICON_PNG}" --out "${APP_ICONSET_DIR}/icon_32x32@2x.png"
                COMMAND "${SIPS_EXECUTABLE}" -s format png -z 128 128   "${APP_ICON_PNG}" --out "${APP_ICONSET_DIR}/icon_128x128.png"
                COMMAND "${SIPS_EXECUTABLE}" -s format png -z 256 256   "${APP_ICON_PNG}" --out "${APP_ICONSET_DIR}/icon_128x128@2x.png"
                COMMAND "${SIPS_EXECUTABLE}" -s format png -z 256 256   "${APP_ICON_PNG}" --out "${APP_ICONSET_DIR}/icon_256x256.png"
                COMMAND "${SIPS_EXECUTABLE}" -s format png -z 512 512   "${APP_ICON_PNG}" --out "${APP_ICONSET_DIR}/icon_256x256@2x.png"
                COMMAND "${SIPS_EXECUTABLE}" -s format png -z 512 512   "${APP_ICON_PNG}" --out "${APP_ICONSET_DIR}/icon_512x512.png"
                COMMAND "${SIPS_EXECUTABLE}" -s format png -z 1024 1024 "${APP_ICON_PNG}" --out "${APP_ICONSET_DIR}/icon_512x512@2x.png"
                COMMAND "${ICONUTIL_EXECUTABLE}" -c icns "${APP_ICONSET_DIR}" -o "${APP_ICNS}"
                COMMENT "Generating macOS .icns app icon from ${APP_ICON_PNG}"
                VERBATIM
            )
            add_custom_target(app_icon ALL DEPENDS "${APP_ICNS}")
            set_source_files_properties("${APP_ICNS}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
            target_sources(XiaoMo PRIVATE "${APP_ICNS}")
            set_target_properties(XiaoMo PROPERTIES MACOSX_BUNDLE_ICON_FILE "appicon.icns")
        else()
            message(WARNING "未找到 sips 或 iconutil，跳过 .icns 生成，应用将无自定义图标。")
        endif()
    else()
        message(WARNING "未找到应用图标 PNG: ${APP_ICON_PNG}，跳过打包图标。")
    endif()
endif()

# Finalize Qt bundle metadata
qt_finalize_executable(XiaoMo)

# Windows: deploy Qt runtime and package a portable zip via windeployqt
if(WIN32 AND AMAIGIRL_ENABLE_WINDEPLOYQT)
    set(_AMAIGIRL_MSVC_REDIST_CRT_DIR "")
    if(MSVC)
        get_filename_component(_amaigirl_msvc_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(_amaigirl_msvc_host_dir "${_amaigirl_msvc_bin_dir}" DIRECTORY)
        get_filename_component(_amaigirl_msvc_arch_dir "${_amaigirl_msvc_host_dir}" DIRECTORY)
        get_filename_component(_amaigirl_msvc_bin_root "${_amaigirl_msvc_arch_dir}" DIRECTORY)
        get_filename_component(_amaigirl_msvc_tools_dir "${_amaigirl_msvc_bin_root}" DIRECTORY)
        get_filename_component(_amaigirl_msvc_version_dir "${_amaigirl_msvc_tools_dir}" DIRECTORY)
        get_filename_component(_amaigirl_vc_root_dir "${_amaigirl_msvc_version_dir}" DIRECTORY)
        set(_amaigirl_redist_root "${_amaigirl_vc_root_dir}/Redist/MSVC")

        if(EXISTS "${_amaigirl_redist_root}")
            file(GLOB _amaigirl_redist_versions LIST_DIRECTORIES true "${_amaigirl_redist_root}/*")
            list(SORT _amaigirl_redist_versions ORDER DESCENDING)
            foreach(_amaigirl_redist_candidate ${_amaigirl_redist_versions})
                set(_amaigirl_redist_crt_dir "${_amaigirl_redist_candidate}/x64/Microsoft.VC143.CRT")
                if(EXISTS "${_amaigirl_redist_crt_dir}")
                    set(_AMAIGIRL_MSVC_REDIST_CRT_DIR "${_amaigirl_redist_crt_dir}")
                    break()
                endif()
            endforeach()
        endif()
    endif()

    if(AMAIGIRL_WINDEPLOYQT_EXECUTABLE)
        set(WINDEPLOYQT_EXECUTABLE "${AMAIGIRL_WINDEPLOYQT_EXECUTABLE}")
    else()
        find_program(WINDEPLOYQT_EXECUTABLE windeployqt)
        if(NOT WINDEPLOYQT_EXECUTABLE AND DEFINED Qt6Core_DIR)
            get_filename_component(_qt6core_dir_real "${Qt6Core_DIR}" REALPATH)
            get_filename_component(_qt6core_cmake_dir "${_qt6core_dir_real}" DIRECTORY)
            get_filename_component(_qt6_lib_dir "${_qt6core_cmake_dir}" DIRECTORY)
            get_filename_component(_qt6_root_dir "${_qt6_lib_dir}" DIRECTORY)
            set(_windeployqt_candidate "${_qt6_root_dir}/bin/windeployqt.exe")
            if(EXISTS "${_windeployqt_candidate}")
                set(WINDEPLOYQT_EXECUTABLE "${_windeployqt_candidate}")
            endif()
        endif()
    endif()

    if(WINDEPLOYQT_EXECUTABLE AND EXISTS "${WINDEPLOYQT_EXECUTABLE}")
        set(AMAIGIRL_WINDOWS_STAGE_DIR "${CMAKE_BINARY_DIR}/XiaoMo")
        set(AMAIGIRL_WINDOWS_PACKAGE_ZIP "${CMAKE_BINARY_DIR}/XiaoMo-windows.zip")
        option(AMAIGIRL_ENABLE_MSIX "Enable MSIX packaging target (Windows only)" ON)
        set(AMAIGIRL_MSIX_IDENTITY_NAME "linhanmo.XiaoMo" CACHE STRING "MSIX Identity Name (must be stable for updates)")
        set(AMAIGIRL_MSIX_PUBLISHER "CN=linhanmo" CACHE STRING "MSIX Publisher (must match certificate subject)")
        set(AMAIGIRL_MSIX_PUBLISHER_DISPLAY_NAME "linhanmo" CACHE STRING "MSIX publisher display name")
        set(AMAIGIRL_MSIX_DISPLAY_NAME "XiaoMo" CACHE STRING "MSIX application display name")
        set(AMAIGIRL_MSIX_DESCRIPTION "XiaoMo Desktop Pet" CACHE STRING "MSIX application description")
        set(AMAIGIRL_MSIX_VERSION "1.0.0.0" CACHE STRING "MSIX package version (Major.Minor.Build.Revision)")

        configure_file(
            "${CMAKE_SOURCE_DIR}/cmake/AppxManifest.xml.in"
            "${CMAKE_BINARY_DIR}/AppxManifest.xml"
            @ONLY
        )
        set(_AMAIGIRL_DEPLOY_WINDOWS_REDIST_COMMANDS "")
        set(_AMAIGIRL_PACKAGE_WINDOWS_REDIST_COMMANDS "")
        if(_AMAIGIRL_MSVC_REDIST_CRT_DIR)
            set(_AMAIGIRL_DEPLOY_WINDOWS_REDIST_COMMANDS
                COMMAND ${CMAKE_COMMAND} -E copy_directory "${_AMAIGIRL_MSVC_REDIST_CRT_DIR}" "$<TARGET_FILE_DIR:XiaoMo>"
            )
            set(_AMAIGIRL_PACKAGE_WINDOWS_REDIST_COMMANDS
                COMMAND ${CMAKE_COMMAND} -E copy_directory "${_AMAIGIRL_MSVC_REDIST_CRT_DIR}" "${AMAIGIRL_WINDOWS_STAGE_DIR}"
            )
        endif()

        add_custom_target(deploy_windows
            COMMAND "${WINDEPLOYQT_EXECUTABLE}"
                    --dir "$<TARGET_FILE_DIR:XiaoMo>"
                    --compiler-runtime
                    --no-translations
                    --no-opengl-sw
                    "$<TARGET_FILE:XiaoMo>"
            ${_AMAIGIRL_DEPLOY_WINDOWS_REDIST_COMMANDS}
            DEPENDS XiaoMo
            COMMENT "Deploying Qt runtime with windeployqt"
            VERBATIM
        )

        # Prepare optional Qt runtime copy commands
        set(_AMAIGIRL_QT_COPY_COMMANDS)
        if(EXISTS "${CMAKE_PREFIX_PATH}/bin/Qt6Core.dll")
            list(APPEND _AMAIGIRL_QT_COPY_COMMANDS
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_PREFIX_PATH}/bin/Qt6Core.dll" "${AMAIGIRL_WINDOWS_STAGE_DIR}/Qt6Core.dll"
            )
        endif()
        foreach(_qt_dll IN ITEMS Qt6Gui Qt6Widgets Qt6OpenGL Qt6OpenGLWidgets Qt6Svg Qt6Multimedia)
            if(EXISTS "${CMAKE_PREFIX_PATH}/bin/${_qt_dll}.dll")
                list(APPEND _AMAIGIRL_QT_COPY_COMMANDS
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_PREFIX_PATH}/bin/${_qt_dll}.dll" "${AMAIGIRL_WINDOWS_STAGE_DIR}/${_qt_dll}.dll"
                )
            endif()
        endforeach()
        foreach(_qt_plugin_dir IN ITEMS platforms imageformats styles multimedia multimedia_audio sqldrivers printsupport)
            if(EXISTS "${CMAKE_PREFIX_PATH}/plugins/${_qt_plugin_dir}")
                list(APPEND _AMAIGIRL_QT_COPY_COMMANDS
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${AMAIGIRL_WINDOWS_STAGE_DIR}/plugins/${_qt_plugin_dir}"
                )
            endif()
        endforeach()
        if(EXISTS "${CMAKE_PREFIX_PATH}/plugins/platforms/qwindows.dll")
            list(APPEND _AMAIGIRL_QT_COPY_COMMANDS
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_PREFIX_PATH}/plugins/platforms/qwindows.dll" "${AMAIGIRL_WINDOWS_STAGE_DIR}/plugins/platforms/qwindows.dll"
            )
        endif()
        foreach(_imgfmt IN ITEMS qsvg qico)
            if(EXISTS "${CMAKE_PREFIX_PATH}/plugins/imageformats/${_imgfmt}.dll")
                list(APPEND _AMAIGIRL_QT_COPY_COMMANDS
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_PREFIX_PATH}/plugins/imageformats/${_imgfmt}.dll" "${AMAIGIRL_WINDOWS_STAGE_DIR}/plugins/imageformats/${_imgfmt}.dll"
                )
            endif()
        endforeach()
        if(EXISTS "${CMAKE_PREFIX_PATH}/plugins/styles/qwindowsvistastyle.dll")
            list(APPEND _AMAIGIRL_QT_COPY_COMMANDS
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_PREFIX_PATH}/plugins/styles/qwindowsvistastyle.dll" "${AMAIGIRL_WINDOWS_STAGE_DIR}/plugins/styles/qwindowsvistastyle.dll"
            )
        endif()
        if(EXISTS "${CMAKE_PREFIX_PATH}/plugins/multimedia/ffmpegmediaplugin.dll")
            list(APPEND _AMAIGIRL_QT_COPY_COMMANDS
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_PREFIX_PATH}/plugins/multimedia/ffmpegmediaplugin.dll" "${AMAIGIRL_WINDOWS_STAGE_DIR}/plugins/multimedia/ffmpegmediaplugin.dll"
            )
        endif()
        foreach(_ssl IN ITEMS libcrypto-3-x64 libssl-3-x64)
            if(EXISTS "${CMAKE_PREFIX_PATH}/bin/${_ssl}.dll")
                list(APPEND _AMAIGIRL_QT_COPY_COMMANDS
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_PREFIX_PATH}/bin/${_ssl}.dll" "${AMAIGIRL_WINDOWS_STAGE_DIR}/${_ssl}.dll"
                )
            endif()
        endforeach()

        set(_AMAIGIRL_SHERPA_COPY_COMMANDS)
        if(_AMAIGIRL_SHERPA_ROOT AND EXISTS "${_AMAIGIRL_SHERPA_ROOT}/lib/sherpa-onnx-cxx-api.dll")
            list(APPEND _AMAIGIRL_SHERPA_COPY_COMMANDS
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_AMAIGIRL_SHERPA_ROOT}/lib/onnxruntime.dll" "${AMAIGIRL_WINDOWS_STAGE_DIR}/onnxruntime.dll"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_AMAIGIRL_SHERPA_ROOT}/lib/onnxruntime_providers_shared.dll" "${AMAIGIRL_WINDOWS_STAGE_DIR}/onnxruntime_providers_shared.dll"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_AMAIGIRL_SHERPA_ROOT}/lib/sherpa-onnx-c-api.dll" "${AMAIGIRL_WINDOWS_STAGE_DIR}/sherpa-onnx-c-api.dll"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_AMAIGIRL_SHERPA_ROOT}/lib/sherpa-onnx-cxx-api.dll" "${AMAIGIRL_WINDOWS_STAGE_DIR}/sherpa-onnx-cxx-api.dll"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_AMAIGIRL_SHERPA_ROOT}/lib/cargs.dll" "${AMAIGIRL_WINDOWS_STAGE_DIR}/cargs.dll"
            )
        endif()

        add_custom_target(package_windows
            COMMAND ${CMAKE_COMMAND} -E rm -rf "${AMAIGIRL_WINDOWS_STAGE_DIR}"
            COMMAND ${CMAKE_COMMAND} -E rm -f "${AMAIGIRL_WINDOWS_PACKAGE_ZIP}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${AMAIGIRL_WINDOWS_STAGE_DIR}"
            COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:XiaoMo>" "${AMAIGIRL_WINDOWS_STAGE_DIR}/XiaoMo.exe"
            COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/res" "${AMAIGIRL_WINDOWS_STAGE_DIR}/res"
            ${_AMAIGIRL_SHERPA_COPY_COMMANDS}
            ${_AMAIGIRL_QT_COPY_COMMANDS}
                ${_AMAIGIRL_PACKAGE_WINDOWS_REDIST_COMMANDS}
            COMMAND ${CMAKE_COMMAND} -E tar cf "${AMAIGIRL_WINDOWS_PACKAGE_ZIP}" --format=zip "XiaoMo"
            DEPENDS XiaoMo
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            COMMENT "Packaging Windows portable bundle"
            VERBATIM
        )

        if(AMAIGIRL_ENABLE_MSIX)
            set(MAKEAPPX_EXECUTABLE "")
            find_program(MAKEAPPX_EXECUTABLE NAMES makeappx makeappx.exe)
            if(NOT MAKEAPPX_EXECUTABLE)
                set(_AMAIGIRL_MSIX_KITS_ROOTS
                    "C:/Program Files (x86)/Windows Kits/10/bin"
                    "$ENV{ProgramFiles}/Windows Kits/10/bin"
                    "C:/Program Files (x86)/Windows Kits/11/bin"
                    "$ENV{ProgramFiles}/Windows Kits/11/bin"
                )
                foreach(_kits_root IN LISTS _AMAIGIRL_MSIX_KITS_ROOTS)
                    if(EXISTS "${_kits_root}")
                        file(GLOB _AMAIGIRL_MSIX_MAKEAPPX_CANDIDATES
                            "${_kits_root}/x64/makeappx.exe"
                            "${_kits_root}/x86/makeappx.exe"
                            "${_kits_root}/*/x64/makeappx.exe"
                            "${_kits_root}/*/x86/makeappx.exe"
                        )
                        if(_AMAIGIRL_MSIX_MAKEAPPX_CANDIDATES)
                            list(SORT _AMAIGIRL_MSIX_MAKEAPPX_CANDIDATES COMPARE NATURAL ORDER DESCENDING)
                            list(GET _AMAIGIRL_MSIX_MAKEAPPX_CANDIDATES 0 MAKEAPPX_EXECUTABLE)
                            break()
                        endif()
                    endif()
                endforeach()
            endif()
            if(NOT MAKEAPPX_EXECUTABLE)
                set(_AMAIGIRL_MSIX_WINGET_ROOT "$ENV{LOCALAPPDATA}/Microsoft/WinGet/Packages")
                if(EXISTS "${_AMAIGIRL_MSIX_WINGET_ROOT}")
                    file(GLOB _AMAIGIRL_MSIX_WINGET_MAKEAPPX
                        "${_AMAIGIRL_MSIX_WINGET_ROOT}/Microsoft.MSIX-Toolkit_*/*/MakeAppx.exe"
                        "${_AMAIGIRL_MSIX_WINGET_ROOT}/Microsoft.MSIX-Toolkit_*/*/makeappx.exe"
                        "${_AMAIGIRL_MSIX_WINGET_ROOT}/Microsoft.MSIX-Toolkit_*/*/*/MakeAppx.exe"
                        "${_AMAIGIRL_MSIX_WINGET_ROOT}/Microsoft.MSIX-Toolkit_*/*/*/makeappx.exe"
                    )
                    if(_AMAIGIRL_MSIX_WINGET_MAKEAPPX)
                        list(SORT _AMAIGIRL_MSIX_WINGET_MAKEAPPX COMPARE NATURAL ORDER DESCENDING)
                        list(GET _AMAIGIRL_MSIX_WINGET_MAKEAPPX 0 MAKEAPPX_EXECUTABLE)
                    endif()
                endif()
            endif()
            if(MAKEAPPX_EXECUTABLE AND EXISTS "${MAKEAPPX_EXECUTABLE}")
                set(AMAIGIRL_MSIX_STAGE_DIR "${CMAKE_BINARY_DIR}/XiaoMoMsix")
                set(AMAIGIRL_MSIX_PACKAGE "${CMAKE_BINARY_DIR}/XiaoMo.msix")
                set(AMAIGIRL_MSIX_ASSETS_DIR "${AMAIGIRL_MSIX_STAGE_DIR}/Assets")

                add_custom_target(package_msix
                    COMMAND ${CMAKE_COMMAND} -E rm -rf "${AMAIGIRL_MSIX_STAGE_DIR}"
                    COMMAND ${CMAKE_COMMAND} -E rm -f "${AMAIGIRL_MSIX_PACKAGE}"
                    COMMAND ${CMAKE_COMMAND} -E copy_directory "${AMAIGIRL_WINDOWS_STAGE_DIR}" "${AMAIGIRL_MSIX_STAGE_DIR}"
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${AMAIGIRL_MSIX_ASSETS_DIR}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/res/icons/app-icon.png" "${AMAIGIRL_MSIX_ASSETS_DIR}/StoreLogo.png"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/res/icons/app-icon.png" "${AMAIGIRL_MSIX_ASSETS_DIR}/Square44x44Logo.png"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/res/icons/app-icon.png" "${AMAIGIRL_MSIX_ASSETS_DIR}/Square150x150Logo.png"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_SOURCE_DIR}/res/icons/app-icon.png" "${AMAIGIRL_MSIX_ASSETS_DIR}/SplashScreen.png"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_BINARY_DIR}/AppxManifest.xml" "${AMAIGIRL_MSIX_STAGE_DIR}/AppxManifest.xml"
                    COMMAND "${MAKEAPPX_EXECUTABLE}" pack /d "${AMAIGIRL_MSIX_STAGE_DIR}" /p "${AMAIGIRL_MSIX_PACKAGE}" /o
                    DEPENDS package_windows
                    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
                    COMMENT "Packaging MSIX (unsigned) via makeappx"
                    VERBATIM
                )

                message(STATUS "Windows MSIX target enabled: package_msix")
            else()
                message(STATUS "makeappx not found, skip package_msix target")
            endif()
        endif()

        message(STATUS "Windows deployment targets enabled: deploy_windows, package_windows")
    else()
        message(STATUS "windeployqt not found, skip deploy_windows/package_windows targets")
    endif()
endif()

# macOS: supply Info.plist from template
if(APPLE)
    set_target_properties(XiaoMo PROPERTIES
        MACOSX_BUNDLE_INFO_PLIST "${CMAKE_SOURCE_DIR}/cmake/Info.plist.in"
    )
endif()

# Copy resources into app output
set(_AMAIGIRL_POST_BUILD_DEPS_CMDS "")
if(UNIX AND NOT APPLE)
    if(TARGET amaigirl_llama_cpp)
        add_dependencies(XiaoMo amaigirl_llama_cpp)
        list(APPEND _AMAIGIRL_POST_BUILD_DEPS_CMDS
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/res/bin"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${AMAIGIRL_LLAMA_CLI_PATH}" "${CMAKE_SOURCE_DIR}/res/bin/llama-cli"
        )
    endif()
    if(TARGET amaigirl_sherpa_onnx_runtime)
        add_dependencies(XiaoMo amaigirl_sherpa_onnx_runtime)
        list(APPEND _AMAIGIRL_POST_BUILD_DEPS_CMDS
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/res/voice_deps"
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${AMAIGIRL_SHERPA_RUNTIME_ROOT}" "${CMAKE_SOURCE_DIR}/res/voice_deps/${AMAIGIRL_SHERPA_RUNTIME_NAME}"
        )
    endif()
endif()

if(APPLE)
    # Bundle layout: XiaoMo.app/Contents/Resources
    add_custom_command(TARGET XiaoMo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_SOURCE_DIR}/res" "$<TARGET_FILE_DIR:XiaoMo>/../Resources"
        COMMENT "Copying resources to bundle Resources directory"
    )
else()
    add_custom_command(TARGET XiaoMo POST_BUILD
        ${_AMAIGIRL_POST_BUILD_DEPS_CMDS}
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_SOURCE_DIR}/res" "$<TARGET_FILE_DIR:XiaoMo>/res"
        COMMENT "Copying resources to output directory"
    )
endif()

# Portable Linux bundle (no conda needed at runtime): build -> install_portable -> run from portable directory
if(UNIX AND NOT APPLE AND AMAIGIRL_ENABLE_PORTABLE_LINUX_BUNDLE)
    set(AMAIGIRL_PORTABLE_PREFIX "${CMAKE_BINARY_DIR}/portable-dist" CACHE PATH "Portable bundle output directory")

    set_target_properties(XiaoMo PROPERTIES
        INSTALL_RPATH "$ORIGIN/lib"
    )

    if(NOT AMAIGIRL_QMAKE_EXECUTABLE)
        find_program(AMAIGIRL_QMAKE_EXECUTABLE NAMES qmake6 qmake)
    endif()

    set(AMAIGIRL_QT_PLUGINS_DIR "")
    set(AMAIGIRL_QT_LIBS_DIR "")
    if(AMAIGIRL_QMAKE_EXECUTABLE)
        execute_process(
            COMMAND "${AMAIGIRL_QMAKE_EXECUTABLE}" -query QT_INSTALL_PLUGINS
            OUTPUT_VARIABLE AMAIGIRL_QT_PLUGINS_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        execute_process(
            COMMAND "${AMAIGIRL_QMAKE_EXECUTABLE}" -query QT_INSTALL_LIBS
            OUTPUT_VARIABLE AMAIGIRL_QT_LIBS_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    endif()

    set(_AMAIGIRL_QT_PLUGINS_DIR_DERIVED "")
    set(_AMAIGIRL_QT_LIBS_DIR_DERIVED "")
    if(DEFINED Qt6_DIR AND EXISTS "${Qt6_DIR}")
        get_filename_component(_qt6_cmake_dir "${Qt6_DIR}" DIRECTORY)
        get_filename_component(_qt6_lib_dir "${_qt6_cmake_dir}" DIRECTORY)
        set(_AMAIGIRL_QT_LIBS_DIR_DERIVED "${_qt6_lib_dir}")
        set(_AMAIGIRL_QT_PLUGINS_DIR_DERIVED "${_qt6_lib_dir}/qt6/plugins")
    endif()

    set(_AMAIGIRL_QT_PLUGINS_DIR_EFFECTIVE "${AMAIGIRL_QT_PLUGINS_DIR}")
    if(NOT (_AMAIGIRL_QT_PLUGINS_DIR_EFFECTIVE AND EXISTS "${_AMAIGIRL_QT_PLUGINS_DIR_EFFECTIVE}"))
        if(_AMAIGIRL_QT_PLUGINS_DIR_DERIVED AND EXISTS "${_AMAIGIRL_QT_PLUGINS_DIR_DERIVED}")
            set(_AMAIGIRL_QT_PLUGINS_DIR_EFFECTIVE "${_AMAIGIRL_QT_PLUGINS_DIR_DERIVED}")
        endif()
    endif()

    set(_AMAIGIRL_QT_LIBS_DIR_EFFECTIVE "${AMAIGIRL_QT_LIBS_DIR}")
    if(NOT (_AMAIGIRL_QT_LIBS_DIR_EFFECTIVE AND EXISTS "${_AMAIGIRL_QT_LIBS_DIR_EFFECTIVE}"))
        if(_AMAIGIRL_QT_LIBS_DIR_DERIVED AND EXISTS "${_AMAIGIRL_QT_LIBS_DIR_DERIVED}")
            set(_AMAIGIRL_QT_LIBS_DIR_EFFECTIVE "${_AMAIGIRL_QT_LIBS_DIR_DERIVED}")
        endif()
    endif()

    if(_AMAIGIRL_QT_LIBS_DIR_EFFECTIVE AND _AMAIGIRL_QT_PLUGINS_DIR_EFFECTIVE)
        set(_AMAIGIRL_RUN_SH_TEMPLATE [=[
#!/usr/bin/env bash
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="@AMAIGIRL_QT_LIB_DIR@:$LD_LIBRARY_PATH"
export QT_PLUGIN_PATH="@AMAIGIRL_QT_PLUGIN_DIR@"
exec "$HERE/XiaoMo" "$@"
]=])
        set(AMAIGIRL_QT_LIB_DIR "${_AMAIGIRL_QT_LIBS_DIR_EFFECTIVE}")
        set(AMAIGIRL_QT_PLUGIN_DIR "${_AMAIGIRL_QT_PLUGINS_DIR_EFFECTIVE}")
        string(CONFIGURE "${_AMAIGIRL_RUN_SH_TEMPLATE}" _AMAIGIRL_RUN_SH_CONTENT @ONLY)
        file(WRITE "${CMAKE_BINARY_DIR}/run.sh" "${_AMAIGIRL_RUN_SH_CONTENT}")
        file(CHMOD "${CMAKE_BINARY_DIR}/run.sh"
            PERMISSIONS
                OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
        )
    endif()

    set(_AMAIGIRL_QTCONF_PATH "${CMAKE_BINARY_DIR}/qt.conf")
    file(WRITE "${_AMAIGIRL_QTCONF_PATH}" "[Paths]\nPlugins=plugins\n")

    install(TARGETS XiaoMo RUNTIME DESTINATION .)
    install(FILES "${_AMAIGIRL_QTCONF_PATH}" DESTINATION .)
    install(DIRECTORY "${CMAKE_SOURCE_DIR}/res" DESTINATION .)
    install(CODE [[
        set(_bin_dir "${CMAKE_INSTALL_PREFIX}/res/bin")
        if(EXISTS "${_bin_dir}")
            set(_llama_src "${CMAKE_BINARY_DIR}/_amaigirl_deps/llama.cpp-build/bin")
            if(EXISTS "${_llama_src}/llama-cli")
                file(COPY_FILE "${_llama_src}/llama-cli" "${_bin_dir}/llama-cli" ONLY_IF_DIFFERENT)
                file(CHMOD "${_bin_dir}/llama-cli"
                    PERMISSIONS
                        OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE
                        WORLD_READ WORLD_EXECUTE
                )
                foreach(_pat IN ITEMS "libllama.so*" "libmtmd.so*" "libggml.so*" "libggml-base.so*" "libggml-cpu.so*")
                    file(GLOB _hits "${_llama_src}/${_pat}")
                    foreach(_h IN LISTS _hits)
                        if(EXISTS "${_h}")
                            file(INSTALL DESTINATION "${_bin_dir}" TYPE SHARED_LIBRARY FOLLOW_SYMLINK_CHAIN FILES "${_h}")
                        endif()
                    endforeach()
                endforeach()
            endif()
            foreach(_f IN ITEMS llama-cli llama)
                if(EXISTS "${_bin_dir}/${_f}")
                    file(CHMOD "${_bin_dir}/${_f}"
                        PERMISSIONS
                            OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_EXECUTE
                            WORLD_READ WORLD_EXECUTE
                    )
                endif()
            endforeach()
        endif()

        if(EXISTS "${CMAKE_INSTALL_PREFIX}/res/voice_deps")
            file(GLOB_RECURSE _voice_bins
                RELATIVE "${CMAKE_INSTALL_PREFIX}"
                "${CMAKE_INSTALL_PREFIX}/res/voice_deps/*/bin/*"
            )
            foreach(_rel IN LISTS _voice_bins)
                set(_p "${CMAKE_INSTALL_PREFIX}/${_rel}")
                if(EXISTS "${_p}" AND NOT IS_DIRECTORY "${_p}")
                    file(CHMOD "${_p}"
                        PERMISSIONS
                            OWNER_READ OWNER_WRITE OWNER_EXECUTE
                            GROUP_READ GROUP_EXECUTE
                            WORLD_READ WORLD_EXECUTE
                    )
                endif()
            endforeach()
        endif()
    ]])

    if(_AMAIGIRL_QT_PLUGINS_DIR_EFFECTIVE AND EXISTS "${_AMAIGIRL_QT_PLUGINS_DIR_EFFECTIVE}")
        foreach(_sub platforms imageformats iconengines styles multimedia platformthemes)
            if(EXISTS "${_AMAIGIRL_QT_PLUGINS_DIR_EFFECTIVE}/${_sub}")
                install(DIRECTORY "${_AMAIGIRL_QT_PLUGINS_DIR_EFFECTIVE}/${_sub}" DESTINATION "plugins")
            endif()
        endforeach()
        if(EXISTS "${_AMAIGIRL_QT_PLUGINS_DIR_EFFECTIVE}/xcbglintegrations")
            install(DIRECTORY "${_AMAIGIRL_QT_PLUGINS_DIR_EFFECTIVE}/xcbglintegrations" DESTINATION "plugins")
        endif()
        foreach(_sub wayland-decoration-client wayland-graphics-integration-client wayland-shell-integration)
            if(EXISTS "${_AMAIGIRL_QT_PLUGINS_DIR_EFFECTIVE}/${_sub}")
                install(DIRECTORY "${_AMAIGIRL_QT_PLUGINS_DIR_EFFECTIVE}/${_sub}" DESTINATION "plugins")
            endif()
        endforeach()
    endif()

    set(_AMAIGIRL_PORTABLE_INSTALL_CODE_TEMPLATE [=[
if(POLICY CMP0207)
  cmake_policy(SET CMP0207 NEW)
endif()
set(_exe "${CMAKE_INSTALL_PREFIX}/XiaoMo")
set(_dest_lib "${CMAKE_INSTALL_PREFIX}/lib")
file(MAKE_DIRECTORY "${_dest_lib}")
set(_search_dirs "")
if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/lib")
  list(APPEND _search_dirs "$ENV{CONDA_PREFIX}/lib")
endif()
set(_amaigirl_qt_lib_dir "@AMAIGIRL_QT_LIB_DIR@")
if(_amaigirl_qt_lib_dir AND EXISTS "${_amaigirl_qt_lib_dir}")
  list(APPEND _search_dirs "${_amaigirl_qt_lib_dir}")
endif()
file(GET_RUNTIME_DEPENDENCIES
  EXECUTABLES "${_exe}"
  RESOLVED_DEPENDENCIES_VAR _resolved
  UNRESOLVED_DEPENDENCIES_VAR _unresolved
  DIRECTORIES ${_search_dirs}
  PRE_EXCLUDE_REGEXES
    ".*libc\\.so\\..*"
    ".*libm\\.so\\..*"
    ".*libpthread\\.so\\..*"
    ".*librt\\.so\\..*"
    ".*libdl\\.so\\..*"
    ".*ld-linux.*\\.so\\..*"
  POST_EXCLUDE_REGEXES
    "^/lib"
    "^/usr/lib"
)
foreach(_dep IN LISTS _resolved)
  file(INSTALL DESTINATION "${_dest_lib}" TYPE SHARED_LIBRARY FOLLOW_SYMLINK_CHAIN FILES "${_dep}")
endforeach()
if(_unresolved)
  message(WARNING "Unresolved runtime deps for portable bundle: ${_unresolved}")
endif()
]=])
    set(AMAIGIRL_QT_LIB_DIR "${_AMAIGIRL_QT_LIBS_DIR_EFFECTIVE}")
    string(CONFIGURE "${_AMAIGIRL_PORTABLE_INSTALL_CODE_TEMPLATE}" _AMAIGIRL_PORTABLE_INSTALL_CODE @ONLY)
    install(CODE "${_AMAIGIRL_PORTABLE_INSTALL_CODE}")

    add_custom_target(install_portable
        COMMAND "${CMAKE_COMMAND}" --install "${CMAKE_BINARY_DIR}" --prefix "${AMAIGIRL_PORTABLE_PREFIX}"
        DEPENDS XiaoMo
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        COMMENT "Installing portable Linux bundle into ${AMAIGIRL_PORTABLE_PREFIX}"
        VERBATIM
    )
endif()

if(WIN32 AND _AMAIGIRL_SHERPA_ROOT AND EXISTS "${_AMAIGIRL_SHERPA_ROOT}/lib/sherpa-onnx-cxx-api.dll")
    add_custom_command(TARGET XiaoMo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_AMAIGIRL_SHERPA_ROOT}/lib/onnxruntime.dll" "$<TARGET_FILE_DIR:XiaoMo>/onnxruntime.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_AMAIGIRL_SHERPA_ROOT}/lib/onnxruntime_providers_shared.dll" "$<TARGET_FILE_DIR:XiaoMo>/onnxruntime_providers_shared.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_AMAIGIRL_SHERPA_ROOT}/lib/sherpa-onnx-c-api.dll" "$<TARGET_FILE_DIR:XiaoMo>/sherpa-onnx-c-api.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_AMAIGIRL_SHERPA_ROOT}/lib/sherpa-onnx-cxx-api.dll" "$<TARGET_FILE_DIR:XiaoMo>/sherpa-onnx-cxx-api.dll"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_AMAIGIRL_SHERPA_ROOT}/lib/cargs.dll" "$<TARGET_FILE_DIR:XiaoMo>/cargs.dll"
        COMMENT "Copying sherpa-onnx runtime DLLs"
    )
endif()

if(WIN32 AND CUBISM_CORE_DLL_WINDOWS_X64 AND EXISTS "${CUBISM_CORE_DLL_WINDOWS_X64}")
    add_custom_command(TARGET XiaoMo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CUBISM_CORE_DLL_WINDOWS_X64}" "$<TARGET_FILE_DIR:XiaoMo>/Live2DCubismCore.dll"
        COMMENT "Copying Live2D Cubism Core runtime DLL"
    )
endif()

# Deploy Qt frameworks/plugins into the .app using macdeployqt ONLY in Release
if(APPLE)
    # Single-config generators (like Ninja used by CLion): rely on CMAKE_BUILD_TYPE
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        # Try to find macdeployqt robustly: PATH first, then derive from Qt6Core_DIR
        set(_MACDEPLOYQT "")
        find_program(_MACDEPLOYQT_PROG macdeployqt)
        if(_MACDEPLOYQT_PROG)
            set(_MACDEPLOYQT "${_MACDEPLOYQT_PROG}")
        endif()
        if(NOT _MACDEPLOYQT AND DEFINED Qt6Core_DIR)
            get_filename_component(_qt6core_cmake_dir "${Qt6Core_DIR}" REALPATH)
            get_filename_component(_qt6core_cmake_parent "${_qt6core_cmake_dir}" DIRECTORY) # …/lib/cmake
            get_filename_component(_qt6_lib_dir "${_qt6core_cmake_parent}" DIRECTORY) # …/lib
            get_filename_component(_qt6_macos_dir "${_qt6_lib_dir}" DIRECTORY) # …/macos
            set(_candidate "${_qt6_macos_dir}/bin/macdeployqt")
            if(EXISTS "${_candidate}")
                set(_MACDEPLOYQT "${_candidate}")
            endif()
        endif()
        if(_MACDEPLOYQT AND EXISTS "${_MACDEPLOYQT}")
            message(STATUS "Using macdeployqt: ${_MACDEPLOYQT}")
            add_custom_command(TARGET XiaoMo POST_BUILD
                COMMAND "${_MACDEPLOYQT}" "$<TARGET_BUNDLE_DIR:XiaoMo>" -always-overwrite -verbose=1
                COMMENT "Bundling Qt runtime with macdeployqt (Release)"
            )
        else()
            message(WARNING "未能自动找到 macdeployqt，可执行文件不存在。Release 构建将不打包依赖。若使用 Qt 官方安装器，通常位于 ~/Qt/<ver>/macos/bin/macdeployqt。请将其加入 PATH 或手动执行。")
        endif()
    endif()

    # Multi-config generators (e.g. Xcode/Ninja Multi-Config): conditionally run for Release
    if(CMAKE_CONFIGURATION_TYPES)
        # Resolve macdeployqt same way
        set(_MACDEPLOYQT_MULTI "")
        find_program(_MACDEPLOYQT_PROG_MULTI macdeployqt)
        if(_MACDEPLOYQT_PROG_MULTI)
            set(_MACDEPLOYQT_MULTI "${_MACDEPLOYQT_PROG_MULTI}")
        endif()
        if(NOT _MACDEPLOYQT_MULTI AND DEFINED Qt6Core_DIR)
            get_filename_component(_qt6core_cmake_dir "${Qt6Core_DIR}" REALPATH)
            get_filename_component(_qt6core_cmake_parent "${_qt6core_cmake_dir}" DIRECTORY)
            get_filename_component(_qt6_lib_dir "${_qt6core_cmake_parent}" DIRECTORY)
            get_filename_component(_qt6_macos_dir "${_qt6_lib_dir}" DIRECTORY)
            set(_candidate_multi "${_qt6_macos_dir}/bin/macdeployqt")
            if(EXISTS "${_candidate_multi}")
                set(_MACDEPLOYQT_MULTI "${_candidate_multi}")
            endif()
        endif()
        if(_MACDEPLOYQT_MULTI AND EXISTS "${_MACDEPLOYQT_MULTI}")
            add_custom_command(TARGET XiaoMo POST_BUILD
                COMMAND /bin/sh -c "if [ \"$<CONFIG>\" = \"Release\" ]; then \"${_MACDEPLOYQT_MULTI}\" \"$<TARGET_BUNDLE_DIR:XiaoMo>\" -always-overwrite -verbose=1; fi"
                COMMENT "Bundling Qt runtime with macdeployqt (Release)"
            )
        endif()
    endif()
endif()
