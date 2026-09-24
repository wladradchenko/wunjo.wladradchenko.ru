# Downloadable content ("Get New …") — what the server has to provide

The six `.knsrc` files in this folder drive the **Download New …** entries in the
application. They are plain INI files read by KNewStuff (KDE Frameworks). The
application code is complete and unchanged — the only thing missing is the server
they point at.

Until `https://wunjo.online/ocs/` answers, every dialog opens and shows an empty
list. Nothing else breaks: the lumas, LUTs, title templates and render presets
that ship inside the application keep working, and the "choose a file" buttons
next to those dialogs still browse the local disk.

## The six entry points

| `.knsrc` | Menu entry | `Categories` | Installs into `<AppDataLocation>/` | Payload |
|---|---|---|---|---|
| `wunjo_wipes.knsrc` | Download New Wipes | `Wunjo FX` | `wunjo/lumas/HD` | Greyscale PNG/PGM masks. The grey level decides in which order pixels change during a transition. Used by [wipe](../transitions/wipe.xml), `luma`, `dissolve`, `composite`, `region` |
| `wunjo_effects.knsrc` | Download New Effects | `Wunjo Effect Templates` | `wunjo/effect-templates` | Effect preset XMLs |
| `wunjo_titles.knsrc` | Download New Title Templates | `Wunjo Title Templates` | `wunjo/titles` | Title templates for the titler |
| `wunjo_luts.knsrc` | (button on the [lut3d](../effects/avfilter/avfilter_lut3d.xml) effect) | `Wunjo Color Look-Up Tables` | `wunjo/luts` | `.cube` / `.3dl` colour lookup tables |
| `wunjo_renderprofiles.knsrc` | Download New Render Profiles | `Wunjo Export Profiles` | `wunjo/export` | Export preset XMLs |
| `wunjo_keyboardschemes.knsrc` | Download New Keyboard Schemes | `Wunjo Keyboard Schemes` | `wunjo/shortcuts` | Shortcut scheme files |

`Uncompress=archive` means the download is expected to be an archive and is
unpacked into `TargetDir`. `TargetDir` is relative to
`QStandardPaths::AppDataLocation`.

## What to implement

Everything hangs off one entry point, set in all six files:

```ini
ProvidersUrl=https://wunjo.online/ocs/providers.xml
```

### 1. `GET /ocs/providers.xml`

An XML document naming the service and where its API lives. Minimal shape:

```xml
<?xml version="1.0"?>
<providers>
  <provider>
    <id>wunjo</id>
    <location>https://wunjo.online/ocs/v1/</location>
    <name>Wunjo Content</name>
    <icon>https://wunjo.online/favicon.ico</icon>
    <termsofuse>https://wunjo.online/terms</termsofuse>
    <register>https://wunjo.online/register</register>
    <services>
      <person ocsversion="1.7"/>
      <content ocsversion="1.7"/>
    </services>
  </provider>
</providers>
```

### 2. The OCS v1 API at `<location>`

KNewStuff speaks [Open Collaboration Services](https://www.freedesktop.org/wiki/Specifications/open-collaboration-services/).
Only a small part of it is used:

- `GET content/categories` — the list of categories. **Must contain the six
  `Categories` names in the table above verbatim**; that string is how each
  dialog asks for its own kind of content.
- `GET content/data?categories=<id>&sortmode=<new|down|rating>&page=&pagesize=&search=`
  — one page of items. Each item carries an id, name, description, author,
  version, licence, changed date, download count, rating, preview picture URLs
  and one or more download links.
- `GET content/data/<id>` — one item, same shape.
- `GET content/download/<id>/<downloadId>` — the actual link for a download
  item. Answer with the URL of the archive.

Responses are OCS XML: an `<ocs><meta><statuscode>100</statuscode></meta><data>…</data></ocs>`
envelope. `statuscode` 100 means success.

Voting, commenting and `person/*` are optional — leave them out and the dialog
simply does not offer those actions.

### 3. If content is ever sold

The OCS provider entry supports HTTP Basic authentication, and KNewStuff will
ask for the credentials named in `<register>`. A paid tier therefore means:
authenticate the request in `content/download/…` and return the archive URL only
for entitled accounts — the catalogue itself (`content/data`) can stay public so
people see what exists.

## Testing without a server

Point `ProvidersUrl` at a `file:///` path with a local `providers.xml` during
development; KNewStuff accepts any URL scheme Qt can fetch.

## Notes

- Translated `Name[…]` entries that still transliterated the upstream project's
  name were removed rather than re-translated; those languages fall back to the
  English name until real translations exist.
- The category strings are a contract between these files and the server. Renaming
  one means changing it in both places.
