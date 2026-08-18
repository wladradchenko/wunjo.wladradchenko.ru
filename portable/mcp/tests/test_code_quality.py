"""Automated code quality checks for MCP tool files.

Detects patterns known to cause bugs:
- Using t.get("type") instead of t.get("audio") for track type detection
- Calling known-buggy D-Bus methods (scriptImportMedia, scriptGetClipProperties, etc.)
- Using GetItemListInTrack with track_idx instead of track_id
- Treating get_clips_on_track() results as list[int] instead of list[dict]
- Using high-level wrapper methods that silently fail
"""

import ast
import os
import re
import sys
import unittest

# All MCP tool files + composite + helpers
MCP_TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "control", "tools")
MCP_HELPERS = os.path.join(os.path.dirname(__file__), "..", "control", "helpers.py")
APP_CLIENT = os.path.join(os.path.dirname(__file__), "..", "api", "app_client.py")


def _collect_py_files(*dirs_and_files):
    """Collect all .py files from dirs and individual file paths."""
    result = []
    for path in dirs_and_files:
        if os.path.isdir(path):
            for fname in os.listdir(path):
                if fname.endswith(".py") and fname != "__init__.py":
                    result.append(os.path.join(path, fname))
        elif os.path.isfile(path):
            result.append(path)
    return result


class TestNoBuggyDBusCalls(unittest.TestCase):
    """Ensure no tool code calls known-buggy D-Bus scripting methods."""

    BUGGY_METHODS = {
        "scriptImportMedia": "Use addProjectClip per file instead (scriptImportMedia returns folder/sequence IDs)",
        "scriptGetClipProperties": "Causes permanent deadlock in Wunjo Make",
        "scriptInsertClipsSequentially": "Returns -1 for all clips; use scriptInsertClip in a loop",
    }

    def test_no_buggy_calls_in_tools(self):
        files = _collect_py_files(MCP_TOOLS_DIR, MCP_HELPERS)
        errors = []
        for fpath in files:
            with open(fpath, encoding="utf-8") as f:
                content = f.read()
            for method, reason in self.BUGGY_METHODS.items():
                # Match _call("methodName" or direct .methodName( patterns
                if re.search(rf'["\']({method})["\']', content):
                    errors.append(f"{os.path.basename(fpath)}: calls '{method}' — {reason}")
        self.assertEqual(errors, [], "Buggy D-Bus methods found:\n" + "\n".join(errors))

    def test_no_buggy_calls_in_app_client(self):
        """app_client.py must not call methods that are buggy in OUR fork.

        Note: scriptGetClipProperties is safe in Wunjo Make — the ported
        implementation reads cached AbstractProjectItem data and never takes
        the producer lock (the old deadlocking variant was dropped during the
        D-Ogi port deduplication)."""
        with open(APP_CLIENT, encoding="utf-8") as f:
            content = f.read()
        for method in ("scriptImportMedia", "scriptInsertClipsSequentially"):
            matches = list(re.finditer(rf'self\._call\(["\']({method})["\']', content))
            self.assertEqual(len(matches), 0, f"app_client.py actively calls {method}")


class TestTrackTypeDetection(unittest.TestCase):
    """Track type must be detected via t.get("audio"), not t.get("type").

    The D-Bus parser returns dicts with 'audio' key ('true'/'false'),
    not a 'type' key. Using t.get("type") silently defaults to "video"
    for ALL tracks, including audio tracks.
    """

    def test_no_type_key_for_track_detection(self):
        files = _collect_py_files(MCP_TOOLS_DIR, MCP_HELPERS)
        errors = []
        for fpath in files:
            with open(fpath, encoding="utf-8") as f:
                content = f.read()
                lines = content.splitlines()

            # Skip media_table() in helpers — it formats media pool items
            # where "type" is a valid clip property, not a track property.
            in_media_table = False

            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if "def media_table" in stripped:
                    in_media_table = True
                elif in_media_table and stripped.startswith("def "):
                    in_media_table = False
                if in_media_table:
                    continue

                # Detect t.get("type") or .get("type", "video") patterns
                if re.search(r'\.get\(\s*["\']type["\']\s*,\s*["\'](?:video|audio)["\']', line):
                    errors.append(f"{os.path.basename(fpath)}:{i}: {line.strip()}")
                if re.search(r'\.get\(\s*["\']type["\']\s*\)\s*==\s*["\'](?:video|audio)["\']', line):
                    errors.append(f"{os.path.basename(fpath)}:{i}: {line.strip()}")
        self.assertEqual(
            errors, [],
            "Track type detected via .get('type') instead of .get('audio'):\n" + "\n".join(errors)
        )


class TestClipsOnTrackReturnType(unittest.TestCase):
    """get_clips_on_track() returns list[dict], not list[int].

    Any code that indexes into the result and treats it as an int
    (e.g., clip_ids[i] used as clip_id directly) is a bug.
    """

    def test_no_direct_indexing_as_int(self):
        files = _collect_py_files(MCP_TOOLS_DIR)
        errors = []
        for fpath in files:
            with open(fpath, encoding="utf-8") as f:
                content = f.read()
            # Pattern: get_clips_on_track(...) assigned to var, then var[i] used
            # without .get("id") — heuristic check
            if "get_clips_on_track" in content:
                # After assignment, check if result is indexed and used directly
                # as an int (no .get() call on the indexed result)
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    # Look for patterns like: old_clip_id = clip_ids[idx]
                    # where clip_ids came from get_clips_on_track
                    if re.search(r'=\s*\w+\[\w+\]\s*$', stripped):
                        # Check if the variable is from get_clips_on_track
                        context = "\n".join(lines[max(0, i-10):i+1])
                        if "get_clips_on_track" in context and ".get(" not in stripped:
                            # Check if next line(s) use .get() on the result
                            # (meaning it's correctly treated as dict)
                            next_lines = "\n".join(lines[i+1:i+4]) if i+1 < len(lines) else ""
                            var_name = stripped.split("=")[0].strip()
                            if f"{var_name}.get(" in next_lines:
                                continue  # correctly treated as dict
                            errors.append(
                                f"{os.path.basename(fpath)}:{i+1}: "
                                f"Possible direct indexing of get_clips_on_track result as int: "
                                f"{stripped}"
                            )
        self.assertEqual(
            errors, [],
            "get_clips_on_track() result indexed as int (should use .get('id')):\n"
            + "\n".join(errors)
        )


class TestNoTrackIdxVsTrackId(unittest.TestCase):
    """Ensure tools pass track_id (unique int), not track_idx (positional index).

    GetItemListInTrack has ambiguous index handling. Tools should use
    the track's 'id' field, not 'index' or 'position'.
    """

    def test_no_track_idx_usage(self):
        files = _collect_py_files(MCP_TOOLS_DIR)
        errors = []
        for fpath in files:
            # checkpoints.py reads the undo-stack "index" from scriptUndoStatus —
            # that is a legitimate key, not a track identifier.
            if os.path.basename(fpath) == "checkpoints.py":
                continue
            with open(fpath, encoding="utf-8") as f:
                content = f.read()
            # Detect: t.get("index", ...) used as track identifier
            if re.search(r'\.get\(\s*["\']index["\']', content):
                errors.append(
                    f"{os.path.basename(fpath)}: uses .get('index') — "
                    f"should use .get('id') for track identification"
                )
        self.assertEqual(
            errors, [],
            "Track 'index' used instead of 'id':\n" + "\n".join(errors)
        )


class TestWrapperConsistency(unittest.TestCase):
    """Check that tools don't mix wrapper and direct D-Bus for the same operation."""

    def test_no_raw_call_for_import(self):
        """Import should use app.import_media() or the addProjectClip
        pattern with ID diffing, but not bare _call without diffing."""
        files = _collect_py_files(MCP_TOOLS_DIR)
        errors = []
        for fpath in files:
            with open(fpath, encoding="utf-8") as f:
                content = f.read()
            # _call("addProjectClip") without get_all_clip_ids nearby = no validation
            if '_call("addProjectClip"' in content or "_call('addProjectClip'" in content:
                if "get_all_clip_ids" not in content:
                    errors.append(
                        f"{os.path.basename(fpath)}: calls addProjectClip without "
                        f"ID diffing (get_all_clip_ids). Import result is unverified."
                    )
        self.assertEqual(errors, [], "Unverified addProjectClip calls:\n" + "\n".join(errors))


def _find_enclosing_function(tree, target_lineno):
    """Find the FunctionDef node enclosing a given line number."""
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= target_lineno <= node.end_lineno:
                if best is None or node.lineno > best.lineno:
                    best = node
    return best


def _get_int_safe_vars(func_node, source_lines):
    """Collect variable names known to be int within a function.

    Returns a set of variable names that are either:
    - Function parameters annotated as int
    - Assigned via int(...)
    - Assigned via arithmetic (var = x + y, x - y, etc.)
    - Assigned via numeric literals
    - Assigned from Python API wrapper methods (not raw D-Bus)
    - Loop variables
    """
    safe = set()

    # Python API wrapper methods that return ints (not D-Bus strings)
    _WRAPPER_METHODS = {
        "GetStart", "GetEnd", "GetDuration", "GetTotalDuration",
        "GetTrackCount", "GetFps",
    }

    def _is_safe_call(call_node):
        """Check if a Call node returns a known-int value."""
        func = call_node.func
        if isinstance(func, ast.Name) and func.id in ("int", "round", "len", "abs"):
            return True
        if isinstance(func, ast.Attribute) and func.attr in _WRAPPER_METHODS:
            return True
        return False

    def _is_safe_rhs(rhs):
        """Check if an expression is known to produce an int."""
        if isinstance(rhs, ast.Call) and _is_safe_call(rhs):
            return True
        if isinstance(rhs, ast.Constant) and isinstance(rhs.value, (int, float)):
            return True
        if isinstance(rhs, ast.BinOp):
            return True
        if isinstance(rhs, ast.UnaryOp):
            return True
        # Ternary: x if cond else 0  — safe if both branches are safe
        if isinstance(rhs, ast.IfExp):
            body_safe = _is_safe_rhs(rhs.body)
            else_safe = _is_safe_rhs(rhs.orelse)
            if body_safe and else_safe:
                return True
            # Common pattern: item.GetX() if hasattr(...) else 0
            if isinstance(rhs.orelse, ast.Constant) and isinstance(rhs.orelse.value, (int, float)):
                if isinstance(rhs.body, ast.Call) and _is_safe_call(rhs.body):
                    return True
        return False

    # 1. Function parameters with int annotation (handles multi-line defs)
    for arg in func_node.args.args:
        if arg.annotation:
            if isinstance(arg.annotation, ast.Name) and arg.annotation.id == "int":
                safe.add(arg.arg)

    # 2. Walk function body for assignments
    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign):
            rhs = node.value

            if _is_safe_rhs(rhs):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        safe.add(target.id)
                    elif isinstance(target, ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                safe.add(elt.id)

            # Tuple assignment with safe values: s, e, d = int(...), int(...), int(...)
            if isinstance(rhs, ast.Tuple):
                if all(_is_safe_rhs(v) for v in rhs.elts):
                    for target in node.targets:
                        if isinstance(target, ast.Tuple):
                            for elt in target.elts:
                                if isinstance(elt, ast.Name):
                                    safe.add(elt.id)

        # for var in ... — loop variables
        elif isinstance(node, ast.For):
            if isinstance(node.target, ast.Name):
                safe.add(node.target.id)
            elif isinstance(node.target, ast.Tuple):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        safe.add(elt.id)

    # 3. Also check source lines for int() cast patterns the AST might miss
    start = func_node.lineno - 1
    end = func_node.end_lineno
    for line in source_lines[start:end]:
        stripped = line.strip()
        m = re.match(r'^(\w+)\s*=\s*int\(', stripped)
        if m:
            safe.add(m.group(1))
        m = re.match(r'^(\w+)\s*=\s*.*int\(', stripped)
        if m:
            safe.add(m.group(1))

    return safe


class TestFormatTcIntCast(unittest.TestCase):
    """Ensure format_tc() is never called with raw D-Bus string results.

    D-Bus returns all values as strings. Passing a string directly to
    format_tc() causes 'unsupported operand type(s) for /: str and float'.
    The first argument must be wrapped in int() or be a variable that was
    previously cast with int().
    """

    def test_format_tc_args_are_int_cast(self):
        files = _collect_py_files(MCP_TOOLS_DIR, MCP_HELPERS)
        errors = []
        for fpath in files:
            with open(fpath, encoding="utf-8") as f:
                source = f.read()
            source_lines = source.splitlines()
            try:
                tree = ast.parse(source, filename=fpath)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                # Match helpers.format_tc(...) or format_tc(...)
                func = node.func
                is_format_tc = False
                if isinstance(func, ast.Attribute) and func.attr == "format_tc":
                    is_format_tc = True
                elif isinstance(func, ast.Name) and func.id == "format_tc":
                    is_format_tc = True
                if not is_format_tc or not node.args:
                    continue

                first_arg = node.args[0]

                # 1. int(x) or round(x) call — explicitly cast
                if isinstance(first_arg, ast.Call):
                    if isinstance(first_arg.func, ast.Name) and first_arg.func.id in ("int", "round"):
                        continue
                    if isinstance(first_arg.func, ast.Attribute) and first_arg.func.attr in ("int", "round"):
                        continue

                # 2. Numeric literal
                if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, (int, float)):
                    continue

                # 3. Arithmetic expression (e.g. position + duration)
                if isinstance(first_arg, ast.BinOp):
                    continue

                # 4. UnaryOp (e.g. -position)
                if isinstance(first_arg, ast.UnaryOp):
                    continue

                # 5. Attribute access (e.g. clip.position) — Python object, not D-Bus string
                if isinstance(first_arg, ast.Attribute):
                    continue

                # 6. Name — variable: check if it's known-int within enclosing function
                if isinstance(first_arg, ast.Name):
                    var_name = first_arg.id
                    enclosing = _find_enclosing_function(tree, node.lineno)
                    if enclosing:
                        safe_vars = _get_int_safe_vars(enclosing, source_lines)
                        if var_name in safe_vars:
                            continue

                    errors.append(
                        f"{os.path.basename(fpath)}:{first_arg.lineno}: "
                        f"format_tc({var_name}, ...) — variable not verifiably int-cast"
                    )
                    continue

                # 7. Subscript (e.g. data[0]) — could be D-Bus string
                if isinstance(first_arg, ast.Subscript):
                    errors.append(
                        f"{os.path.basename(fpath)}:{first_arg.lineno}: "
                        f"format_tc() called with subscript — ensure int() cast"
                    )
                    continue

        self.assertEqual(
            errors, [],
            "format_tc() called without verified int() cast on first argument:\n"
            + "\n".join(errors)
        )


class TestNoDirectGetFps(unittest.TestCase):
    """All FPS access should go through helpers.get_fps(ctx), not .GetFps().

    Calling .GetFps() directly bypasses the centralized accessor and
    risks inconsistent FPS values (e.g., missing float cast).
    Exception: project.py may use proj.GetFps() since it directly manages
    the project object.
    """

    def test_no_direct_getfps_in_tools(self):
        files = _collect_py_files(MCP_TOOLS_DIR)
        errors = []
        for fpath in files:
            basename = os.path.basename(fpath)
            # project.py is allowed to use proj.GetFps() directly
            if basename == "project.py":
                continue
            with open(fpath, encoding="utf-8") as f:
                content = f.read()
            matches = list(re.finditer(r'\.GetFps\(\)', content))
            if matches:
                lines = content.splitlines()
                for m in matches:
                    # Find the line number
                    line_no = content[:m.start()].count("\n") + 1
                    errors.append(
                        f"{basename}:{line_no}: {lines[line_no - 1].strip()}"
                    )
        self.assertEqual(
            errors, [],
            "Direct .GetFps() calls found (use helpers.get_fps(ctx) instead):\n"
            + "\n".join(errors)
        )


class TestResolutionIntCast(unittest.TestCase):
    """get_project_resolution_width/height() return strings from D-Bus.

    They must be wrapped in int() before use in arithmetic or string
    formatting that expects numbers. This test scans preview.py and
    titles.py (the primary consumers) for uncast calls.
    """

    RESOLUTION_METHODS = [
        "get_project_resolution_width",
        "get_project_resolution_height",
    ]

    def test_resolution_calls_are_int_cast(self):
        files = _collect_py_files(MCP_TOOLS_DIR)
        errors = []
        for fpath in files:
            with open(fpath, encoding="utf-8") as f:
                content = f.read()
            lines = content.splitlines()
            for method in self.RESOLUTION_METHODS:
                for i, line in enumerate(lines, 1):
                    if method not in line:
                        continue
                    # Skip comments and docstrings
                    stripped = line.strip()
                    if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                        continue
                    # Check that it's wrapped in int()
                    # Valid: int(app.get_project_resolution_width())
                    # Invalid: app.get_project_resolution_width() used directly
                    if f"int({method}" in line or f"int(app.{method}" in line:
                        continue  # properly cast
                    # Also allow: x = int(some_var) where some_var = app.method()
                    # but direct usage without int() is a bug
                    if re.search(rf'=\s*\w+\.{method}\(\)', stripped):
                        # Assigned without int() — check if int() is on the same line
                        if "int(" not in line:
                            errors.append(
                                f"{os.path.basename(fpath)}:{i}: "
                                f"{method}() not wrapped in int(): {stripped}"
                            )
        self.assertEqual(
            errors, [],
            "Resolution D-Bus calls without int() cast:\n" + "\n".join(errors)
        )


if __name__ == "__main__":
    unittest.main()
