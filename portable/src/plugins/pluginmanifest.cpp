/*
SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "pluginmanifest.h"

#include "xml/xml.hpp"

#include <config-wunjo.h>

#include <KLocalizedString>

#include <QDir>
#include <QDomDocument>
#include <QFile>
#include <QFileInfo>
#include <QSvgRenderer>
#include <QPixmap>
#include <QPalette>
#include <QPainter>
#include <QApplication>
#include <QJsonArray>
#include <QJsonDocument>
#include <QRegularExpression>
#include <QSysInfo>

#include <algorithm>
#include <utility>

namespace {
const QStringList kKinds = {QStringLiteral("api"), QStringLiteral("local")};
// "shared" is gone: a plugin used to be able to install into the application's
// own venv, which put unrelated stacks in one place and made whichever plugin
// pinned hardest the one everybody else had to live with. uv builds a private
// environment fast enough, and from a cache, that sharing bought nothing but
// the coupling. Every plugin owns its own now.
const QStringList kVenvs = {QStringLiteral("private")};
const QStringList kTargets = {QStringLiteral("video"), QStringLiteral("audio"), QStringLiteral("face"), QStringLiteral("generator"),
                              QStringLiteral("agent")};
const QStringList kKnownOs = {QStringLiteral("linux"), QStringLiteral("windows"), QStringLiteral("macos")};
} // namespace

PluginManifest PluginManifest::fromDir(const QString &dir, bool checkFolderName)
{
    PluginManifest m;
    m.m_rootDir = dir;
    const QString manifestPath = dir + QStringLiteral("/plugin.json");
    QFile file(manifestPath);
    if (!file.open(QIODevice::ReadOnly)) {
        m.m_errors << i18n("plugin.json is missing");
        return m;
    }
    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll(), &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isObject()) {
        m.m_errors << i18n("plugin.json is not valid JSON: %1", parseError.errorString());
        return m;
    }
    const QJsonObject root = doc.object();

    m.m_id = root.value(QStringLiteral("id")).toString();
    m.m_name = root.value(QStringLiteral("name")).toString();
    m.m_version = root.value(QStringLiteral("version")).toString();
    m.m_author = root.value(QStringLiteral("author")).toString();
    m.m_authorEmail = root.value(QStringLiteral("author_email")).toString();
    m.m_description = root.value(QStringLiteral("description")).toString();
    m.m_license = root.value(QStringLiteral("license")).toString();
    m.m_homepage = root.value(QStringLiteral("homepage")).toString();
    m.m_kind = root.value(QStringLiteral("kind")).toString();
    m.m_venv = root.value(QStringLiteral("venv")).toString(QStringLiteral("private"));
    m.m_icon = root.value(QStringLiteral("icon")).toString();
    // "target": "video", or "target": ["video", "audio"] when it is both
    const QJsonValue targetValue = root.value(QStringLiteral("target"));
    if (targetValue.isArray()) {
        const QJsonArray entries = targetValue.toArray();
        for (const QJsonValue &entry : entries) {
            const QString name = entry.toString();
            if (!name.isEmpty() && !m.m_targets.contains(name)) {
                m.m_targets << name;
            }
        }
    } else if (!targetValue.toString().isEmpty()) {
        m.m_targets << targetValue.toString();
    }
    m.m_entry = root.value(QStringLiteral("entry")).toString();
    m.m_requirements = root.value(QStringLiteral("requirements")).toString();
    const QJsonArray variants = root.value(QStringLiteral("requirements_cuda")).toArray();
    for (const QJsonValue &value : variants) {
        const QJsonObject obj = value.toObject();
        PluginRequirements variant;
        variant.file = obj.value(QStringLiteral("file")).toString();
        variant.minDriverCuda = obj.value(QStringLiteral("min_driver_cuda")).toDouble();
        if (variant.file.isEmpty()) {
            m.m_errors << i18n("every 'requirements_cuda' entry needs a 'file'");
            continue;
        }
        if (!QFileInfo::exists(dir + QLatin1Char('/') + variant.file)) {
            m.m_errors << i18n("requirements file '%1' was not found", variant.file);
            continue;
        }
        m.m_requirementsVariants.append(variant);
    }
    std::sort(m.m_requirementsVariants.begin(), m.m_requirementsVariants.end(),
              [](const PluginRequirements &a, const PluginRequirements &b) { return a.minDriverCuda > b.minDriverCuda; });
    const QJsonValue provider = root.value(QStringLiteral("provider"));
    if (provider.isObject()) {
        const QJsonObject providerObj = provider.toObject();
        m.m_providerName = providerObj.value(QStringLiteral("name")).toString();
        m.m_providerKeySetting = providerObj.value(QStringLiteral("key_setting")).toString();
        m.m_providerSignupUrl = providerObj.value(QStringLiteral("signup_url")).toString();
    }
    const QJsonArray modelsArray = root.value(QStringLiteral("models")).toArray();
    for (const QJsonValue &value : modelsArray) {
        const QJsonObject obj = value.toObject();
        PluginModel model;
        model.name = obj.value(QStringLiteral("name")).toString();
        model.url = obj.value(QStringLiteral("url")).toString();
        model.sha256 = obj.value(QStringLiteral("sha256")).toString();
        model.sizeMb = qint64(obj.value(QStringLiteral("size_mb")).toDouble());
        model.autoDownload = obj.value(QStringLiteral("auto_download")).toBool();
        model.group = obj.value(QStringLiteral("group")).toString();
        model.unpack = obj.value(QStringLiteral("unpack")).toString();
        model.platform = obj.value(QStringLiteral("platform")).toString();
        model.backend = obj.value(QStringLiteral("backend")).toString();
        model.minVramGb = obj.value(QStringLiteral("min_vram_gb")).toDouble();
        model.maxVramGb = obj.value(QStringLiteral("max_vram_gb")).toDouble();
        if (!model.unpack.isEmpty() && model.unpack != QLatin1String("zip") && model.unpack != QLatin1String("tar.gz")) {
            m.m_errors << i18n("model '%1': 'unpack' only understands \"zip\" or \"tar.gz\"", model.name);
        }
        if (!model.name.isEmpty()) {
            m.m_models.append(model);
        }
    }
    const QJsonArray osArray = root.value(QStringLiteral("os")).toArray();
    for (const QJsonValue &value : osArray) {
        m.m_os << value.toString();
    }
    m.m_minAppVersion = root.value(QStringLiteral("min_app_version")).toString();
    m.m_maxAppVersion = root.value(QStringLiteral("max_app_version")).toString();
    const QJsonArray paramsArray = root.value(QStringLiteral("params")).toArray();
    for (const QJsonValue &value : paramsArray) {
        const QJsonObject obj = value.toObject();
        PluginParam param;
        param.key = obj.value(QStringLiteral("key")).toString();
        param.label = obj.value(QStringLiteral("label")).toString();
        param.type = obj.value(QStringLiteral("type")).toString();
        param.defaultValue = obj.value(QStringLiteral("default")).toVariant();
        const QJsonArray options = obj.value(QStringLiteral("options")).toArray();
        for (const QJsonValue &option : options) {
            param.options << option.toString();
        }
        param.filter = obj.value(QStringLiteral("filter")).toString();
        param.group = obj.value(QStringLiteral("group")).toString();
        param.min = obj.value(QStringLiteral("min")).toDouble(0);
        param.max = obj.value(QStringLiteral("max")).toDouble(100);
        param.step = obj.value(QStringLiteral("step")).toDouble(1);
        if (!param.key.isEmpty()) {
            m.m_params.append(param);
        }
    }
    // Effects the plugin brings along. They are read here rather than at install
    // time so a broken XML shows up as a manifest error in the importer, next to
    // the other format violations.
    const QString cleanRoot = QDir::cleanPath(dir);
    const QJsonArray effectsArray = root.value(QStringLiteral("effects")).toArray();
    for (const QJsonValue &value : effectsArray) {
        PluginEffect effect;
        effect.file = value.toString();
        if (effect.file.isEmpty()) {
            m.m_errors << i18n("every 'effects' entry must be the path of an effect XML");
            continue;
        }
        const QString path = QDir::cleanPath(cleanRoot + QLatin1Char('/') + effect.file);
        if (!path.startsWith(cleanRoot + QLatin1Char('/'))) {
            m.m_errors << i18n("effect '%1' must live inside the plugin folder", effect.file);
            continue;
        }
        QDomDocument doc;
        if (!Xml::docContentFromFile(doc, path, false)) {
            m.m_errors << i18n("effect '%1' is missing or is not valid XML", effect.file);
            continue;
        }
        const QDomElement base = doc.documentElement();
        if (base.tagName() != QLatin1String("effect")) {
            m.m_errors << i18n("effect '%1' must have a single <effect> root", effect.file);
            continue;
        }
        effect.id = base.attribute(QStringLiteral("id"));
        effect.tag = base.attribute(QStringLiteral("tag"));
        effect.name = Xml::getSubTagContent(base, QStringLiteral("name"));
        effect.requiresEffect = base.attribute(QStringLiteral("wunjo_requires"));
        if (effect.tag.isEmpty()) {
            m.m_errors << i18n("effect '%1' does not name the MLT service it is built on", effect.file);
        }
        if (effect.id.isEmpty()) {
            m.m_errors << i18n("effect '%1' has no id", effect.file);
            continue;
        }
        // Namespacing the id keeps a plugin from shadowing a built-in effect and
        // makes the owner obvious in a project file that outlived the plugin.
        if (!m.m_id.isEmpty() && effect.id != m.m_id && !effect.id.startsWith(m.m_id + QLatin1Char('.'))) {
            m.m_errors << i18n("effect id '%1' must be '%2' or start with '%2.'", effect.id, m.m_id);
        }
        if (effect.name.isEmpty()) {
            effect.name = effect.id;
        }
        const QDomNodeList parameters = base.elementsByTagName(QStringLiteral("parameter"));
        for (int i = 0; i < parameters.count(); ++i) {
            const QDomElement parameter = parameters.item(i).toElement();
            const QString fill = parameter.attribute(QStringLiteral("wunjo_fill"));
            if (fill == QLatin1String("face")) {
                effect.faceParams << parameter.attribute(QStringLiteral("name"));
            } else if (fill == QLatin1String("region")) {
                effect.regionParams << parameter.attribute(QStringLiteral("name"));
            }
        }
        m.m_effects.append(effect);
    }

    for (const PluginEffect &effect : std::as_const(m.m_effects)) {
        if (effect.requiresEffect.isEmpty()) {
            continue;
        }
        const auto shipped = std::find_if(m.m_effects.cbegin(), m.m_effects.cend(),
                                          [&effect](const PluginEffect &other) { return other.id == effect.requiresEffect; });
        if (shipped == m.m_effects.cend()) {
            m.m_errors << i18n("effect '%1' requires '%2', which this plugin does not ship", effect.id, effect.requiresEffect);
        }
    }

    m.m_effectsTogether = root.value(QStringLiteral("effects_apply")).toString() == QLatin1String("together");

    const QJsonObject setsUi = root.value(QStringLiteral("sets")).toObject();
    const auto readSetsUi = [](const QJsonObject &source) {
        PluginSetsUi ui;
        ui.label = source.value(QStringLiteral("label")).toString();
        ui.action = source.value(QStringLiteral("action")).toString();
        ui.filter = source.value(QStringLiteral("filter")).toString();
        ui.detail = source.value(QStringLiteral("detail")).toString();
        return ui;
    };
    for (auto it = setsUi.constBegin(); it != setsUi.constEnd(); ++it) {
        if (it.value().isObject()) {
            // "sets": { "face": {…}, "audio": {…} } — one block per kind
            m.m_setsUi.insert(it.key(), readSetsUi(it.value().toObject()));
        }
    }
    if (m.m_setsUi.isEmpty() && !setsUi.isEmpty()) {
        // "sets": { "label": …, "action": … } — a plugin with a single kind
        m.m_setsUi.insert(QString(), readSetsUi(setsUi));
    }

    const QJsonObject input = root.value(QStringLiteral("input")).toObject();
    m.m_inputClip = input.value(QStringLiteral("clip")).toString();
    m.m_inputMultiple = input.value(QStringLiteral("multiple")).toBool(false);

    // ---- validation (mirror of plugins/pack.py) ----
    if (root.value(QStringLiteral("manifest_version")).toInt() != 1) {
        m.m_errors << i18n("manifest_version must be 1");
    }
    // Bad bounds are worse than none: a typo that parses as 0 would quietly
    // block every version, so say so at import time instead.
    static const QRegularExpression versionRe(QStringLiteral("^\\d+(\\.\\d+)*$"));
    for (const auto &[field, value] : {std::pair{QStringLiteral("min_app_version"), m.m_minAppVersion},
                                       std::pair{QStringLiteral("max_app_version"), m.m_maxAppVersion}}) {
        if (!value.isEmpty() && !versionRe.match(value).hasMatch()) {
            m.m_errors << i18n("'%1' must look like \"3\", \"3.1\" or \"3.1.2\", not '%2'", field, value);
        }
    }
    if (!m.m_minAppVersion.isEmpty() && !m.m_maxAppVersion.isEmpty()
        && compareVersions(m.m_minAppVersion, m.m_maxAppVersion) > 0) {
        m.m_errors << i18n("'min_app_version' (%1) is newer than 'max_app_version' (%2)", m.m_minAppVersion, m.m_maxAppVersion);
    }
    static const QRegularExpression idRe(QStringLiteral("^[a-z0-9][a-z0-9-]{1,63}$"));
    if (m.m_id.isEmpty()) {
        m.m_errors << i18n("missing required field 'id'");
    } else if (!idRe.match(m.m_id).hasMatch()) {
        m.m_errors << i18n("id '%1' may only contain lowercase letters, digits and hyphens", m.m_id);
    } else if (checkFolderName) {
        const QString folder = QDir(dir).dirName();
        if (m.m_id != folder) {
            m.m_errors << i18n("id '%1' must match the folder name '%2'", m.m_id, folder);
        }
    }
    if (m.m_name.isEmpty()) {
        m.m_errors << i18n("missing required field 'name'");
    }
    if (m.m_version.isEmpty()) {
        m.m_errors << i18n("missing required field 'version'");
    }
    if (m.m_kind.isEmpty()) {
        m.m_errors << i18n("missing required field 'kind'");
    } else if (!kKinds.contains(m.m_kind)) {
        m.m_errors << i18n("kind must be 'api' or 'local'");
    }
    if (!kVenvs.contains(m.m_venv)) {
        m.m_errors << i18n("venv must be 'private'");
    }
    if (m.m_targets.isEmpty()) {
        m.m_errors << i18n("missing required field 'target'");
    } else {
        for (const QString &target : std::as_const(m.m_targets)) {
            if (!kTargets.contains(target)) {
                m.m_errors << i18n("target must be one of video, audio, face, generator, agent");
                break;
            }
        }
        // An agent drives the whole editor from the chat; it has no clip to work
        // on, so pairing it with a clip target would put it in menus where it
        // cannot do anything.
        if (m.m_targets.contains(QLatin1String("agent")) && m.m_targets.count() > 1) {
            m.m_errors << i18n("target 'agent' cannot be combined with another target");
        }
    }
    if (m.m_entry.isEmpty()) {
        m.m_errors << i18n("missing required field 'entry'");
    } else if (!QFileInfo::exists(dir + QLatin1Char('/') + m.m_entry)) {
        m.m_errors << i18n("entry script '%1' was not found", m.m_entry);
    }
    if (m.m_kind == QLatin1String("api") && m.m_providerName.isEmpty()) {
        m.m_errors << i18n("an 'api' plugin must declare provider.name");
    }
    for (const QString &os : std::as_const(m.m_os)) {
        if (!kKnownOs.contains(os)) {
            m.m_errors << i18n("unknown os '%1'", os);
        }
    }
    return m;
}

QList<PluginModel> PluginManifest::modelsFor(double vramGb, const QString &backend) const
{
    const QString os = currentOs();
    // How well a variant suits this machine. Lower is better; anything built
    // for hardware that is not here scores worst and is dropped.
    //
    // The backend is a preference, not a requirement, and that matters: the
    // machine may want CUDA while the project publishes no CUDA build for this
    // platform, and a plain filter would then offer no runtime at all. Vulkan
    // runs on any card, the CPU build runs anywhere, so there is always an
    // answer — slower, but present.
    const auto rank = [&backend](const PluginModel &model) {
        if (model.backend.isEmpty() || model.backend == backend) {
            return 0; // not a runtime at all, or exactly the one wanted
        }
        if (model.backend == QLatin1String("vulkan")) {
            return 1;
        }
        if (model.backend == QLatin1String("cpu")) {
            return 2;
        }
        return 3; // built for somebody else's hardware
    };

    const QString platform = currentPlatform();
    QList<PluginModel> selected;
    for (const PluginModel &model : m_models) {
        // Either the exact machine ("macos-arm64") or just the system
        // ("linux") when one build covers every processor it runs on.
        if (!model.platform.isEmpty() && model.platform != platform && model.platform != os) {
            continue;
        }
        if (model.minVramGb > 0 && vramGb < model.minVramGb) {
            continue;
        }
        if (model.maxVramGb > 0 && vramGb >= model.maxVramGb) {
            continue;
        }
        if (rank(model) > 2) {
            continue;
        }
        // One weight per name: the variants of a runtime are the same weight
        // built differently, and offering the user two "llama-server" rows to
        // choose between is offering them a decision they cannot make.
        auto existing = std::find_if(selected.begin(), selected.end(), [&model](const PluginModel &kept) { return kept.name == model.name; });
        if (existing == selected.end()) {
            selected.append(model);
        } else if (rank(model) < rank(*existing)) {
            *existing = model;
        }
    }
    return selected;
}

QString PluginManifest::requirementsFor(double driverCuda) const
{
    for (const PluginRequirements &variant : m_requirementsVariants) {
        if (driverCuda >= variant.minDriverCuda) {
            return variant.file;
        }
    }
    return m_requirements;
}

PluginSetsUi PluginManifest::setsUi(const QString &kind) const
{
    if (m_setsUi.contains(kind)) {
        return m_setsUi.value(kind);
    }
    return m_setsUi.value(QString());
}

bool PluginManifest::hasDependencies() const
{
    if (m_requirements.isEmpty()) {
        return false;
    }
    QFile file(m_rootDir + QLatin1Char('/') + m_requirements);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return false;
    }
    while (!file.atEnd()) {
        const QString line = QString::fromUtf8(file.readLine()).trimmed();
        if (!line.isEmpty() && !line.startsWith(QLatin1Char('#'))) {
            return true;
        }
    }
    return false;
}

QString PluginManifest::venvName() const
{
    if (!hasDependencies()) {
        return {};
    }
    return QStringLiteral("venv-") + m_id;
}

QString PluginManifest::currentOs()
{
#if defined(Q_OS_WIN)
    return QStringLiteral("windows");
#elif defined(Q_OS_MACOS)
    return QStringLiteral("macos");
#else
    return QStringLiteral("linux");
#endif
}

QString PluginManifest::currentPlatform()
{
    QString arch = QSysInfo::currentCpuArchitecture();
    if (arch == QLatin1String("x86_64") || arch == QLatin1String("i386")) {
        arch = QStringLiteral("x64");
    } else if (arch.startsWith(QLatin1String("arm"))) {
        arch = QStringLiteral("arm64");
    }
    return currentOs() + QLatin1Char('-') + arch;
}

bool PluginManifest::osSupported() const
{
    return m_os.isEmpty() || m_os.contains(currentOs());
}

int PluginManifest::compareVersions(const QString &left, const QString &right)
{
    const QStringList a = left.split(QLatin1Char('.'));
    const QStringList b = right.split(QLatin1Char('.'));
    for (int i = 0; i < qMax(a.size(), b.size()); ++i) {
        // A missing component is a zero, so "3.1" and "3.1.0" are the same
        // release and a manifest may write either.
        const int lhs = i < a.size() ? a.at(i).toInt() : 0;
        const int rhs = i < b.size() ? b.at(i).toInt() : 0;
        if (lhs != rhs) {
            return lhs < rhs ? -1 : 1;
        }
    }
    return 0;
}

QString PluginManifest::appVersionBlocker() const
{
    const QString app = QStringLiteral(WUNJO_VERSION);
    if (!m_minAppVersion.isEmpty() && compareVersions(app, m_minAppVersion) < 0) {
        return i18n("%1 needs Wunjo Make %2 or newer; this is %3.", m_name, m_minAppVersion, app);
    }
    if (!m_maxAppVersion.isEmpty() && compareVersions(app, m_maxAppVersion) > 0) {
        return i18n("%1 was built for Wunjo Make %2 and older; this is %3.", m_name, m_maxAppVersion, app);
    }
    return {};
}

namespace {
/** @brief An SVG drawn in the colour the rest of the interface writes in.
 *
 * Loading the file straight into a QIcon paints whatever colour the file names,
 * which is a picture that stays black on a dark theme. The built-in icons avoid
 * this by being part of an icon theme that recolours them; a file a plugin
 * brought along is not. So the colour is substituted here, before rendering:
 * `currentColor` becomes the palette's text colour, and the icon is white on
 * dark and dark on light without the plugin author having to think about it.
 */
QIcon recoloured(const QString &path)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        return {};
    }
    QString svg = QString::fromUtf8(file.readAll());
    const QString ink = qApp->palette().color(QPalette::WindowText).name();
    svg.replace(QLatin1String("currentColor"), ink);
    QSvgRenderer renderer(svg.toUtf8());
    if (!renderer.isValid()) {
        return {};
    }
    QIcon icon;
    for (int size : {16, 22, 32, 48}) {
        QPixmap pixmap(size, size);
        pixmap.fill(Qt::transparent);
        QPainter painter(&pixmap);
        renderer.render(&painter);
        painter.end();
        icon.addPixmap(pixmap);
    }
    return icon;
}
} // namespace

QIcon PluginManifest::icon() const
{
    if (!m_icon.isEmpty() && !m_rootDir.isEmpty()) {
        const QIcon drawn = recoloured(m_rootDir + QLatin1Char('/') + m_icon);
        if (!drawn.isNull()) {
            return drawn;
        }
    }
    return QIcon::fromTheme(QStringLiteral("tools-wizard"));
}


QString PluginManifest::summaryHtml(bool withTitle) const
{
    auto row = [](const QString &key, const QString &value) {
        return QStringLiteral("<tr><td style='color:#696969;padding-right:12px'>%1</td><td>%2</td></tr>").arg(key, value.toHtmlEscaped());
    };
    QStringList targetLabels;
    for (const QString &target : std::as_const(m_targets)) {
        if (target == QLatin1String("video")) {
            targetLabels << i18n("Video clips");
        } else if (target == QLatin1String("audio")) {
            targetLabels << i18n("Audio clips");
        } else if (target == QLatin1String("face")) {
            targetLabels << i18n("Detected faces");
        } else if (target == QLatin1String("generator")) {
            targetLabels << i18n("Media generation");
        } else if (target == QLatin1String("agent")) {
            targetLabels << i18n("The chat assistant");
        }
    }
    QString targetLabel = targetLabels.join(i18nc("separator between the kinds of clip a plugin works on", " and "));
    if (targetLabel.isEmpty()) {
        targetLabel = m_targets.join(QLatin1String(", "));
    }

    QString html;
    if (withTitle) {
        html += QStringLiteral("<b>%1</b> <span style='color:#696969'>%2</span><br/>").arg(m_name.toHtmlEscaped(), m_version.toHtmlEscaped());
    }
    if (!m_description.isEmpty()) {
        html += QStringLiteral("<p>%1</p>").arg(m_description.toHtmlEscaped());
    }
    // Only user-facing facts: what it does, who made it, its licence. The OS
    // list and the environment kind are internal install-time logic, not shown.
    html += QStringLiteral("<table>");
    html += row(i18n("Acts on"), targetLabel);
    if (!m_effects.isEmpty()) {
        QStringList names;
        for (const PluginEffect &effect : m_effects) {
            names << effect.name;
        }
        html += row(m_effects.count() > 1 ? i18n("Adds effects") : i18n("Adds effect"), names.join(QStringLiteral(", ")));
    }
    html += row(i18n("Type"), m_kind == QLatin1String("api") ? i18n("Online API (%1)", m_providerName) : i18n("Local model"));
    if (!m_author.isEmpty()) {
        html += row(i18n("Author"), m_author);
    }
    html += row(i18n("License"), m_license.isEmpty() ? i18n("not specified") : m_license);
    html += QStringLiteral("</table>");
    return html;
}
