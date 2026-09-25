#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Vladislav Radchenko <i@wladradchenko.ru>
# SPDX-License-Identifier: GPL-3.0-or-later
"""Bring po/<lang>/wunjo.po up to date with the strings in the source.

Messages.sh was written for KDE's translation servers, which run it with their
own extractrc and then do the merge themselves. Wunjo is not on those servers,
so nothing ever merged the strings added since the fork: they were simply
missing from every .po and stayed English. This script does the whole round
locally, with nothing but xgettext and msgmerge (gettext) and Python:

  1. collects the strings of .ui/.rc files, effect/transition XML and layout
     names into a temporary rc.cpp, as extractrc would;
  2. also collects the effect XML of the first-party plugins in ../plugins:
     the host translates an effect's name, description and parameter labels
     through the "wunjo" domain when it loads them, so a plugin's effects are
     translated by the application's own catalog;
  3. runs xgettext over the C++/QML sources and that rc.cpp into wunjo.pot;
  4. msgmerges the template into every po/<lang>/wunjo.po.

New strings then appear untranslated, and changed ones as fuzzy, in each .po.
Translate them there; ki18n_install() compiles the .po files at build time.

Usage:  python3 update-translations.py [lang ...]    (default: every language)
"""
import glob
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
SUBDIRS = ["plugins", "renderer", "data", "src"]
XML_TAGS = {"name", "description", "label", "comment", "paramlistdisplay", "text", "title", "tooltip", "whatsthis"}
RC_SKIP = {"encodingprofiles.rc", "camcorderfilters.rc", "externalproxies.rc"}

KEYWORDS = [
    "-ki18n:1", "-ki18nc:1c,2", "-ki18np:1,2", "-ki18ncp:1c,2,3",
    "-kki18n:1", "-kki18nc:1c,2", "-kki18np:1,2", "-kki18ncp:1c,2,3",
    "-kxi18n:1", "-kxi18nc:1c,2", "-kxi18np:1,2", "-kxi18ncp:1c,2,3",
    "-kkxi18n:1", "-kkxi18nc:1c,2", "-kkxi18np:1,2", "-kkxi18ncp:1c,2,3",
    "-kI18N_NOOP:1", "-kI18NC_NOOP:1c,2", "-kI18N_NOOP2:1c,2", "-kI18N_NOOP2_NOSTRIP:1c,2",
    "-ktr2i18n:1", "-ktr2xi18n:1",
]


def c_string(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def xml_strings(path):
    """(context, text) pairs of one .ui, .rc or asset XML file."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as err:
        print(f"skipped {os.path.relpath(path, HERE)}: {err}", file=sys.stderr)
        return
    if path.endswith(".ui"):
        for el in root.iter("string"):
            if el.get("notr") != "true" and el.text and el.text.strip():
                yield el.get("comment"), el.text.strip()
        return
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        # In a .rc file <name> is not shown to anyone; menus show <text>.
        if tag not in XML_TAGS or (tag == "name" and path.endswith(".rc")):
            continue
        if el.text and el.text.strip() and len(el) == 0:
            yield el.get("context"), el.text.strip()


def collect_files():
    files = []
    for sub in ("transitions", "transitions/frei0r", "effects", "effects/frei0r", "effects/avfilter", "effects/ladspa", "effects/sox", "generators"):
        files += sorted(glob.glob(os.path.join(HERE, "data", sub, "*.xml")))
    files += [os.path.join(HERE, "data", "profiles.xml"), os.path.join(HERE, "data", "effectscategory.rc")]
    for sub in SUBDIRS:
        for dirpath, _, names in os.walk(os.path.join(HERE, sub)):
            for n in sorted(names):
                if (n.endswith(".rc") and n not in RC_SKIP) or n.endswith(".ui"):
                    files.append(os.path.join(dirpath, n))
    files += sorted(glob.glob(os.path.join(HERE, "..", "plugins", "*", "effects", "*.xml")))
    return [f for f in dict.fromkeys(files) if os.path.isfile(f)]


def write_rc(path):
    lines = []
    for f in sorted(glob.glob(os.path.join(HERE, "data", "layouts", "*.json"))):
        try:
            for info in json.load(open(f, encoding="utf-8")).get("wunjoInfo", []):
                if info.get("displayName"):
                    lines += [f"// i18n: file: {os.path.relpath(f, HERE)}", f"i18n({c_string(info['displayName'])});"]
        except (OSError, ValueError, AttributeError):
            pass
    for f in collect_files():
        for ctx, text in xml_strings(f):
            lines.append(f"// i18n: file: {os.path.relpath(f, HERE)}")
            lines.append(f"i18nc({c_string(ctx)},{c_string(text)});" if ctx else f"i18n({c_string(text)});")
    with open(path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")


def main():
    langs = sys.argv[1:] or sorted(d for d in os.listdir(os.path.join(HERE, "po")) if os.path.isfile(os.path.join(HERE, "po", d, "wunjo.po")))
    # xgettext writes each path as it was given into the "#:" references, so
    # everything is passed relative to this directory — including rc.cpp,
    # which lives here only while xgettext reads it, as in Messages.sh.
    with tempfile.TemporaryDirectory() as tmp:
        rc = os.path.join(HERE, "rc.cpp")
        pot = os.path.join(tmp, "wunjo.pot")
        write_rc(rc)
        sources = []
        for sub in SUBDIRS:
            for dirpath, _, names in os.walk(os.path.join(HERE, sub)):
                sources += [os.path.relpath(os.path.join(dirpath, n), HERE) for n in sorted(names) if n.endswith((".cpp", ".h", ".qml"))]
        listing = os.path.join(tmp, "files.txt")
        with open(listing, "w", encoding="utf-8") as out:
            out.write("\n".join(sources + ["rc.cpp"]) + "\n")
        try:
            subprocess.run(["xgettext", "--from-code=UTF-8", "-C", "--kde", "-ci18n", *KEYWORDS, "--no-wrap",
                            "--package-name=wunjo", "--msgid-bugs-address=https://github.com/wladradchenko/wunjo.wladradchenko.ru/issues",
                            "-f", listing, "-o", pot], check=True, cwd=HERE)
        finally:
            os.remove(rc)
        for lang in langs:
            po = os.path.join(HERE, "po", lang, "wunjo.po")
            subprocess.run(["msgmerge", "--quiet", "--update", "--backup=off", "--previous", po, pot], check=True)
            stats = subprocess.run(["msgfmt", "--statistics", "-o", os.devnull, po], capture_output=True, text=True).stderr.strip()
            print(f"{lang}: {stats}")


if __name__ == "__main__":
    main()
