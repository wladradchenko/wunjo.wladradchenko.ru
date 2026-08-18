/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QHash>
#include <QJsonObject>
#include <QList>
#include <QIcon>
#include <QString>
#include <QStringList>
#include <QVariant>

/** @brief A model weight a plugin needs, downloaded on demand into models/. */
struct PluginModel {
    QString name;
    /** @brief Which part of the plugin needs it, when a plugin does several
     *  things: the settings page then lists the weights of each under its own
     *  heading instead of one heap. */
    QString group;
    QString url;
    QString sha256;
    qint64 sizeMb = 0;
    bool autoDownload = false;
    /** @brief "zip" when the download is an archive to unpack into
     *  `models/<name>/` (and mark executable). A weight is not always a single
     *  file: an inference runtime ships as a release archive, and asking every
     *  plugin to unpack it by hand would put that code in every plugin. */
    QString unpack;
    /** @brief OS/architecture this file is built for ("linux-x64",
     *  "windows-x64", "macos-arm64"). Empty = works anywhere. */
    QString platform;
    /** @brief Compute backend a runtime binary was built against ("vulkan",
     *  "cuda", "cpu"). Empty = not a runtime, or backend-agnostic. */
    QString backend;
    /** @brief VRAM window this variant is meant for, in GB. A plugin ships one
     *  model in two quantisations — the small one has to work on a weak card,
     *  the large one is worth its size only when there is room — and the editor
     *  offers the one this machine can actually run instead of asking the user
     *  to guess. 0 = no bound on that side. */
    double minVramGb = 0;
    double maxVramGb = 0;
};

/** @brief An effect a plugin brings along.

    The plugin ships a plain Wunjo effect XML; while the plugin is installed the
    file lives in the shared effects folder and the effect behaves like any
    other one (effect list, keyframes, project files). Uninstalling the plugin
    takes it away again.
 */
struct PluginEffect {
    QString file; ///< XML path, relative to the plugin root
    QString id;   ///< effect id, as the effect list and project files see it
    QString name; ///< display name read from the XML
    QString tag;  ///< MLT service the effect is built on
    /** @brief Parameters marked `wunjo_fill="face"`: applying the effect from a
     *  face box fills them with that face's track, so the effect follows the
     *  face the user picked instead of a static rectangle. */
    QStringList faceParams;
    /** @brief Parameters marked `wunjo_fill="region"`: they get the id that ties
     *  an effect to the region effect it works inside. */
    QStringList regionParams;
    /** @brief Effect id from `wunjo_requires`: the region effect this one works
     *  inside. Applying this effect applies that one first and bonds the two;
     *  the required effect is not offered on its own in the menus. */
    QString requiresEffect;
};

/** @brief One way to install a plugin's dependencies, chosen by what the
    machine's GPU driver can actually run.

    A plugin pins the versions it works with — a newer torch or a newer CUDA
    line may have dropped what it calls — while which line to take is decided
    here, from the driver. Without that, pip installs the newest wheel and it
    refuses to start on an older driver, silently falling back to the CPU.
 */
struct PluginRequirements {
    QString file;      ///< requirements file, relative to the plugin root
    double minDriverCuda{0}; ///< lowest CUDA version of the driver it needs (0 = no GPU needed)
};

/** @brief How a plugin's preset panel presents itself.

    The panel is shared by every plugin, but what a preset *is* differs: one
    records how a performance moves, another simply registers a face to use.
    Wording and the file filter therefore belong to the plugin, not to the
    editor — otherwise a face preset asks the user to analyse a video.
 */
struct PluginSetsUi {
    QString label;  ///< one line above the list, saying what these presets are
    QString action; ///< the button that makes one ("Analyse", "Register face"…)
    QString filter; ///< file dialog filter for the source media
    QString detail; ///< header of the second column ("Frames", "Faces"…)
};

/** @brief A configurable plugin parameter, rendered as one form row. */
struct PluginParam {
    QString key;
    QString label;
    /** @brief Heading this setting belongs under, for a plugin with several
     *  parts. Empty means the plugin's general settings. */
    QString group;
    QString type; // string | number | bool | enum | file
    QVariant defaultValue;
    QStringList options; // for enum
    QString filter;      // for file
    double min = 0;
    double max = 100;
    double step = 1;
};

/** @class PluginManifest
    @brief Parsed and validated `plugin.json` of a Wunjo Make plugin.

    The validation rules mirror `plugins/pack.py` (the format's source of truth):
    a manifest that fails any rule collects a human-readable reason in
    @ref errors and reports @ref isValid false, which is what gates the
    importer's "Import" button.
 */
class PluginManifest
{
public:
    PluginManifest() = default;

    /** @brief Read `<dir>/plugin.json`. @p checkFolderName enforces that the
     *  manifest id equals the folder name (true for folder imports, false when
     *  reading a freshly extracted archive in a temp dir). */
    static PluginManifest fromDir(const QString &dir, bool checkFolderName);

    bool isValid() const { return m_errors.isEmpty(); }
    const QStringList &errors() const { return m_errors; }

    QString id() const { return m_id; }
    QString name() const { return m_name; }
    QString version() const { return m_version; }
    QString author() const { return m_author; }
    /** @brief Where to write to the author, if the manifest says. */
    QString authorEmail() const { return m_authorEmail; }
    QString description() const { return m_description; }
    QString license() const { return m_license; }
    QString homepage() const { return m_homepage; }
    QString kind() const { return m_kind; }
    /** @brief "private" (own `venv-<id>`) or "shared" (global `venv`). */
    QString venv() const { return m_venv; }
    /** @brief What kind of clip this plugin works on. The first of its targets,
     *  kept for callers that only ever expect one. */
    QString target() const { return m_targets.isEmpty() ? QString() : m_targets.first(); }
    /** @brief Every kind it works on. A plugin can answer questions about both
     *  the picture and the sound — finding shot changes and speaker turns is one
     *  job with two halves — and then it belongs on either sort of clip. */
    QStringList targets() const { return m_targets; }
    /** @brief The plugin's own icon, ready to hang on a menu entry.
     *
     * Every entry used to carry the same wand, so a menu of plugins told the
     * user nothing about which was which. A plugin may ship an SVG and name it
     * in the manifest; drawn with `stroke="currentColor"` it follows the theme
     * like the built-in icons do. Falls back to the wand when there is none.
     */
    QIcon icon() const;
    bool hasTarget(const QString &target) const { return m_targets.contains(target); }
    QString entry() const { return m_entry; }
    QString requirements() const { return m_requirements; }
    /** @brief GPU variants of the requirements, richest first. */
    QList<PluginRequirements> requirementsVariants() const { return m_requirementsVariants; }
    /** @brief The requirements file to install for a driver that supports CUDA
     *  @p driverCuda (0 when there is no usable GPU): the first variant the
     *  driver satisfies, or the plain @ref requirements otherwise. */
    QString requirementsFor(double driverCuda) const;
    QStringList os() const { return m_os; }
    QString providerName() const { return m_providerName; }
    /** @brief Env var name the entry receives the API key in (api plugins). */
    QString providerKeySetting() const { return m_providerKeySetting; }
    QString providerSignupUrl() const { return m_providerSignupUrl; }
    QList<PluginModel> models() const { return m_models; }
    /** @brief The weights that apply to a machine with @p vramGb of video
     *  memory (0 = none usable) running @p backend. Variants meant for other
     *  hardware are left out entirely rather than shown greyed out: a download
     *  the user must not start is noise on the settings page. */
    QList<PluginModel> modelsFor(double vramGb, const QString &backend) const;
    /** @brief True for a plugin that drives the editor from the chat dock
     *  instead of processing a clip ("MCP control"). It appears in the chat's
     *  way-of-talking picker and nowhere else. */
    bool isAgent() const { return hasTarget(QStringLiteral("agent")); }
    QList<PluginParam> params() const { return m_params; }
    /** @brief Effects this plugin adds to the editor while it is installed. */
    QList<PluginEffect> effects() const { return m_effects; }
    /** @brief True when its effects belong together (`"effects_apply": "together"`).
     *  The menu then offers the plugin once and applies the whole set on one
     *  face: three ways of working on it that share a single detection, rather
     *  than three unrelated entries. */
    bool appliesEffectsTogether() const { return m_effectsTogether; }
    /** @brief How its preset panel should read for presets of @p kind. A plugin
     *  that does one thing declares the wording flat; one that does three
     *  declares a block per kind. Empty fields fall back to wording that fits a
     *  recorded performance. */
    PluginSetsUi setsUi(const QString &kind = QString()) const;

    QString inputClip() const { return m_inputClip; }
    bool inputMultiple() const { return m_inputMultiple; }

    QString rootDir() const { return m_rootDir; }
    void setRootDir(const QString &dir) { m_rootDir = dir; }

    /** @brief True when requirements.txt lists at least one real package
     *  (ignoring blank lines and #comments). Empty ⇒ run on system Python. */
    bool hasDependencies() const;

    /** @brief The venv directory name this plugin should use, or empty when it
     *  needs no environment. */
    QString venvName() const;

    /** @brief The OS the app runs on: "linux" | "windows" | "macos". */
    static QString currentOs();
    /** @brief The machine a downloaded binary has to match: "linux-x64",
     *  "macos-arm64", "windows-x64". An Intel Mac and an Apple Silicon one run
     *  the same OS and cannot run each other's builds, so the OS alone is not
     *  enough to pick one. */
    static QString currentPlatform();
    /** @brief True when the manifest does not restrict the OS, or lists it. */
    bool osSupported() const;

    /** @brief Oldest / newest Wunjo Make this plugin declares itself built for
     *  ("3.1", "3.4.2"), or empty when it does not say.
     *
     *  `manifest_version` pins the shape of this file; these pin the editor
     *  behind it. A plugin calls tools and effect parameters that only exist
     *  from some release on, and one written against a later SDK than the
     *  installed app fails deep inside its own code with something the user
     *  cannot act on. Declaring the window turns that into one sentence before
     *  anything is downloaded. */
    QString minAppVersion() const { return m_minAppVersion; }
    QString maxAppVersion() const { return m_maxAppVersion; }
    /** @brief Why this plugin does not fit the running application, or empty
     *  when it does. Absent bounds always fit: most plugins never need them. */
    QString appVersionBlocker() const;

    /** @brief Compare two dotted version strings ("3.1" < "3.1.1" < "3.2").
     *  Missing components count as zero, so "3.1" and "3.1.0" are equal.
     *  @return <0, 0 or >0 like strcmp. */
    static int compareVersions(const QString &left, const QString &right);

    /** @brief Rich-text summary for a metadata panel. @p withTitle prepends the
     *  name and version (used in the import preview; omitted on a plugin's own
     *  tab, where the tab title already shows them). Never lists the internal
     *  Systems / Environment details. */
    QString summaryHtml(bool withTitle = true) const;

private:
    QString m_id;
    QString m_name;
    QString m_version;
    QString m_author;
    QString m_description;
    QString m_license;
    QString m_authorEmail;
    QString m_homepage;
    QString m_kind;
    QString m_venv = QStringLiteral("private");
    QStringList m_targets;
    QString m_icon;
    QString m_entry;
    QString m_requirements;
    QList<PluginRequirements> m_requirementsVariants;
    QStringList m_os;
    QString m_minAppVersion;
    QString m_maxAppVersion;
    QString m_providerName;
    QString m_providerKeySetting;
    QString m_providerSignupUrl;
    QList<PluginModel> m_models;
    QList<PluginParam> m_params;
    QList<PluginEffect> m_effects;
    QHash<QString, PluginSetsUi> m_setsUi;
    bool m_effectsTogether{false};
    QString m_inputClip;
    bool m_inputMultiple = false;
    QString m_rootDir;
    QStringList m_errors;
};
