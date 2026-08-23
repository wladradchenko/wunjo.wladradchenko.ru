#!/usr/bin/env python3
"""Validate and pack Wunjo Make plugins into ``.wmplugin`` archives.

Usage:
    python pack.py <plugin-dir> [<plugin-dir> ...]   # validate + pack each
    python pack.py --all                             # pack every plugin here
    python pack.py --check <plugin-dir>              # validate only, do not pack

An archive is a plain ZIP whose root is the plugin folder's contents. Model
weights (``models/``) and any environment (``venv*``, ``__pycache__``) are never
packed — they are fetched or built on the user's machine at install time.

The validation rules here are the source of truth for the format; the desktop
importer applies the same checks before it will enable an "Import" button.
"""
import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

MANIFEST = "plugin.json"
ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,63}$")
KINDS = {"api", "local"}
# Private only. Installing into the application's own venv used to be allowed
# and bought nothing but coupling: whichever plugin pinned hardest decided for
# the rest. uv builds a private environment from cache fast enough that there
# is no reason to share one.
VENVS = {"private"}
TARGETS = {"video", "audio", "face", "generator", "agent"}
# What a downloaded weight may be wrapped in. Runtimes ship as release archives
# — tarballs on Linux, zips on Windows — so a weight is not always one file.
UNPACK_KINDS = {"zip", "tar.gz"}
RESULT_TYPES = {"video", "audio", "image", "subtitle", "none"}
RESULT_PLACES = {"bin", "timeline", "replace-zone", "none"}
KNOWN_OS = {"linux", "windows", "macos"}
VERSION_RE = re.compile(r"^\d+(\.\d+)*$")


def version_tuple(value):
    """Dotted version as a comparable tuple; "3.1" and "3.1.0" compare equal."""
    parts = [int(part) for part in value.split(".")]
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)
# Anywhere in the tree: build leftovers, never part of a plugin.
EXCLUDE_DIRS = {"__pycache__", ".git"}
# Only at the plugin's own root. "models" there is the weights folder, fetched on
# the user's machine; deeper down it is ordinary source — LatentSync keeps its
# UNet in latentsync/models/, and dropping that shipped an archive that imported
# nothing and failed only at render time.
EXCLUDE_ROOT_DIRS = {"models"}
EXCLUDE_PREFIX = ("venv",)


def _local_name(tag):
    """Strip the XML namespace an effect file declares (xmlns=…/wunjo.online)."""
    return tag.rsplit("}", 1)[-1]


def validate_effects(plugin_dir, manifest, errors):
    """Check the effect XMLs a plugin brings along (manifest field ``effects``).

    Installing the plugin drops these into the editor's effects folder and
    uninstalling takes them away, so they must be readable and must not be able
    to shadow a built-in effect — hence the id rule.
    """
    plugin_id = manifest.get("id") or ""
    entries = manifest.get("effects", []) or []
    if not isinstance(entries, list):
        errors.append("'effects' must be a list of paths")
        return
    root = os.path.abspath(plugin_dir)
    shipped = {}
    for entry in entries:
        if not isinstance(entry, str) or not entry:
            errors.append("every 'effects' entry must be the path of an effect XML")
            continue
        path = os.path.abspath(os.path.join(root, entry))
        if os.path.commonpath([root, path]) != root:
            errors.append("effect '%s' must live inside the plugin folder" % entry)
            continue
        if not os.path.isfile(path):
            errors.append("effect '%s' was not found" % entry)
            continue
        try:
            base = ET.parse(path).getroot()
        except ET.ParseError as error:
            errors.append("effect '%s' is not valid XML: %s" % (entry, error))
            continue
        if _local_name(base.tag) != "effect":
            errors.append("effect '%s' must have a single <effect> root" % entry)
            continue
        if not base.get("tag"):
            errors.append("effect '%s' does not name the MLT service it is built on" % entry)
        effect_id = base.get("id")
        if not effect_id:
            errors.append("effect '%s' has no id" % entry)
            continue
        if plugin_id and effect_id != plugin_id and not effect_id.startswith(plugin_id + "."):
            errors.append("effect id '%s' must be '%s' or start with '%s.'"
                          % (effect_id, plugin_id, plugin_id))
        shipped[effect_id] = base.get("wunjo_requires")

    for effect_id, requires in shipped.items():
        if requires and requires not in shipped:
            errors.append("effect '%s' requires '%s', which this plugin does not ship"
                          % (effect_id, requires))


def validate(plugin_dir):
    """Return (manifest_dict, errors_list). manifest is None if unreadable."""
    errors = []
    manifest_path = os.path.join(plugin_dir, MANIFEST)
    if not os.path.isfile(manifest_path):
        return None, ["%s is missing" % MANIFEST]
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except ValueError as error:
        return None, ["%s is not valid JSON: %s" % (MANIFEST, error)]

    def require(field):
        if field not in manifest or manifest[field] in (None, ""):
            errors.append("missing required field '%s'" % field)
            return False
        return True

    if manifest.get("manifest_version") != 1:
        errors.append("manifest_version must be 1")

    folder = os.path.basename(os.path.normpath(plugin_dir))
    if require("id"):
        plugin_id = manifest["id"]
        if not ID_RE.match(plugin_id):
            errors.append("id '%s' must match %s" % (plugin_id, ID_RE.pattern))
        elif plugin_id != folder:
            errors.append("id '%s' must equal the folder name '%s'" % (plugin_id, folder))

    require("name")
    require("version")

    if require("kind") and manifest["kind"] not in KINDS:
        errors.append("kind must be one of %s" % sorted(KINDS))

    venv = manifest.get("venv", "private")
    if venv not in VENVS:
        errors.append("venv must be one of %s" % sorted(VENVS))

    # "target" is one kind of clip, or a list of them for a plugin that works on
    # more than one — finding shot changes and speaker turns is one job with two
    # halves, and it belongs on video and on audio alike.
    if require("target"):
        wanted = manifest["target"]
        wanted = wanted if isinstance(wanted, list) else [wanted]
        if not wanted:
            errors.append("'target' must name at least one kind of clip")
        for entry in wanted:
            if entry not in TARGETS:
                errors.append("target must be one of %s" % sorted(TARGETS))
                break
        # An agent drives the whole editor from the chat and has no clip to work
        # on; pairing it with a clip target would put it in menus where nothing
        # it does makes sense.
        if "agent" in wanted and len(wanted) > 1:
            errors.append("target 'agent' cannot be combined with another target")

    # Weights may come in variants — one per platform, per GPU backend, per
    # amount of video memory — and the editor picks the one this machine can run.
    for model in manifest.get("models", []) or []:
        if not isinstance(model, dict):
            errors.append("every 'models' entry must be an object")
            continue
        unpack = model.get("unpack")
        if unpack is not None and unpack not in UNPACK_KINDS:
            errors.append("model '%s': unpack must be one of %s" % (model.get("name", "?"), sorted(UNPACK_KINDS)))
        low, high = model.get("min_vram_gb"), model.get("max_vram_gb")
        if low is not None and high is not None and low >= high:
            errors.append("model '%s': min_vram_gb must be below max_vram_gb" % model.get("name", "?"))

    # An icon is optional, but a named one that is not there would silently
    # fall back to the generic wand and leave the author wondering why.
    icon = manifest.get("icon")
    if icon:
        if not isinstance(icon, str) or not os.path.isfile(os.path.join(plugin_dir, icon)):
            errors.append("icon '%s' was not found" % icon)
        elif not icon.lower().endswith(".svg"):
            errors.append("icon '%s' should be an SVG so it follows the theme" % icon)

    if require("entry"):
        if not os.path.isfile(os.path.join(plugin_dir, manifest["entry"])):
            errors.append("entry script '%s' not found" % manifest["entry"])

    if manifest.get("kind") == "api":
        provider = manifest.get("provider")
        if not isinstance(provider, dict) or not provider.get("name"):
            errors.append("kind 'api' requires provider.name")

    for key in ("input", "result"):
        if key in manifest and not isinstance(manifest[key], dict):
            errors.append("'%s' must be an object" % key)
    result = manifest.get("result", {})
    if isinstance(result, dict):
        if result.get("type", "none") not in RESULT_TYPES:
            errors.append("result.type must be one of %s" % sorted(RESULT_TYPES))
        if result.get("place", "none") not in RESULT_PLACES:
            errors.append("result.place must be one of %s" % sorted(RESULT_PLACES))

    for entry in manifest.get("os", []) or []:
        if entry not in KNOWN_OS:
            errors.append("unknown os '%s'" % entry)

    # The editor version this plugin was built against. Optional, but a typo
    # that parses as 0 would silently block every release, so it is checked.
    bounds = {}
    for field in ("min_app_version", "max_app_version"):
        value = manifest.get(field) or ""
        if not value:
            continue
        if not VERSION_RE.match(value):
            errors.append("'%s' must look like \"3\", \"3.1\" or \"3.1.2\", not '%s'" % (field, value))
        else:
            bounds[field] = version_tuple(value)
    if "min_app_version" in bounds and "max_app_version" in bounds:
        if bounds["min_app_version"] > bounds["max_app_version"]:
            errors.append("'min_app_version' (%s) is newer than 'max_app_version' (%s)"
                          % (manifest["min_app_version"], manifest["max_app_version"]))

    # GPU variants of the requirements: the plugin pins what it works with per
    # CUDA line, the editor picks by the driver it finds.
    variants = manifest.get("requirements_cuda", []) or []
    if not isinstance(variants, list):
        errors.append("'requirements_cuda' must be a list")
        variants = []
    for variant in variants:
        if not isinstance(variant, dict) or not variant.get("file"):
            errors.append("every 'requirements_cuda' entry needs a 'file'")
            continue
        if not os.path.isfile(os.path.join(plugin_dir, variant["file"])):
            errors.append("requirements file '%s' not found" % variant["file"])
        if not isinstance(variant.get("min_driver_cuda", 0), (int, float)):
            errors.append("'min_driver_cuda' of '%s' must be a number" % variant["file"])

    # How the preset panel reads for this plugin (all fields optional).
    sets_ui = manifest.get("sets", {}) or {}
    if not isinstance(sets_ui, dict):
        errors.append("'sets' must be an object")
    else:
        for field in ("label", "action", "filter", "detail"):
            if field in sets_ui and not isinstance(sets_ui[field], str):
                errors.append("'sets.%s' must be a string" % field)

    validate_effects(plugin_dir, manifest, errors)

    return manifest, errors


def _packable(root, names, plugin_dir, bundle_models=False):
    """Which subfolders go into the archive.

    ``models/`` at the plugin root is normally left out — weights are fetched on
    the user's machine, and nobody wants a five-gigabyte archive. A plugin whose
    models are small enough to carry says so with ``"bundle_models": true`` and
    ships them, so installing it is the whole of the setup.
    """
    at_root = os.path.abspath(root) == os.path.abspath(plugin_dir)
    excluded_here = set() if bundle_models else EXCLUDE_ROOT_DIRS
    names[:] = [
        n
        for n in names
        if n not in EXCLUDE_DIRS
        and not n.startswith(EXCLUDE_PREFIX)
        and not (at_root and n in excluded_here)
    ]
    return names


def pack(plugin_dir, out_dir):
    manifest, errors = validate(plugin_dir)
    if errors:
        for error in errors:
            print("  ERROR: %s" % error, file=sys.stderr)
        return None
    os.makedirs(out_dir, exist_ok=True)
    name = "%s-%s.wmplugin" % (manifest["id"], manifest["version"])
    archive = os.path.join(out_dir, name)
    bundle_models = bool(manifest.get("bundle_models"))
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(plugin_dir):
            _packable(root, dirs, plugin_dir, bundle_models)
            for filename in files:
                if filename.endswith((".pyc", ".pyo")):
                    continue
                full = os.path.join(root, filename)
                rel = os.path.relpath(full, plugin_dir)
                zf.write(full, rel)
    return archive


def main():
    parser = argparse.ArgumentParser(description="Pack Wunjo Make plugins.")
    parser.add_argument("plugins", nargs="*", help="plugin folders")
    parser.add_argument("--all", action="store_true", help="pack every plugin folder here")
    parser.add_argument("--check", action="store_true", help="validate only, do not pack")
    parser.add_argument("-o", "--out", default="dist", help="output directory (default: dist)")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.all:
        targets = [os.path.join(here, n) for n in sorted(os.listdir(here))
                   if os.path.isfile(os.path.join(here, n, MANIFEST))]
    else:
        targets = [os.path.abspath(p) for p in args.plugins]
    if not targets:
        parser.error("no plugin folders given (use folder names or --all)")

    failures = 0
    for plugin_dir in targets:
        label = os.path.basename(os.path.normpath(plugin_dir))
        manifest, errors = validate(plugin_dir)
        if errors:
            failures += 1
            print("%s: INVALID" % label)
            for error in errors:
                print("  - %s" % error)
            continue
        if args.check:
            print("%s: OK (%s %s, %s/%s)" % (label, manifest["name"], manifest["version"],
                                             manifest["kind"], manifest["target"]))
            continue
        archive = pack(plugin_dir, os.path.abspath(args.out))
        if archive:
            print("%s -> %s (%d bytes)" % (label, archive, os.path.getsize(archive)))
        else:
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
