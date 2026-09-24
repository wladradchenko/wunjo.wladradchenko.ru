"""ProjectManager — create, load, save projects."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from api.app_client import WunjoMakeClient

from api.project import Project


class ProjectManager:
    """Manages Wunjo Make projects (open, save, create).

    Wunjo Make does not have a project database like Resolve. The folder
    navigation methods (GetProjectListInCurrentFolder, OpenFolder, etc.)
    are stubs that report the current project only.
    """

    def __init__(self, app: WunjoMakeClient):
        self._app = app

    def CreateProject(self, name: str) -> Project | None:
        """Create a new project with default settings."""
        result = self._app.new_project(name)
        if result:
            return Project(self._app)
        return None

    def LoadProject(self, file_path: str) -> Project | None:
        """Open an existing .wmproj project file."""
        if self._app.open_project(file_path):
            return Project(self._app)
        return None

    def SaveProject(self) -> bool:
        """Save the current project."""
        return self._app.save_project()

    def GetCurrentProject(self) -> Project:
        """Return a Project handle for the currently open project."""
        return Project(self._app)

    # ── Resolve project-database compatibility stubs ───────────────────

    def GetProjectListInCurrentFolder(self) -> list[str]:
        """Return project names in the current folder.

        Wunjo Make has no project database — returns the currently open
        project name in a list.
        """
        name = self._app.get_project_name()
        return [name] if name else []

    def GetFolderListInCurrentFolder(self) -> list[str]:
        """Return subfolder names in the current project folder.

        Always returns an empty list (no project database).
        """
        return []

    def OpenFolder(self, folder_name: str) -> bool:
        """Open a folder in the project database. No-op."""
        return False

    def GotoParentFolder(self) -> bool:
        """Navigate to the parent folder. No-op stub."""
        return True

    def GotoRootFolder(self) -> bool:
        """Navigate to the root folder. No-op stub."""
        return True

    def CloseProject(self, project: Project) -> bool:
        """Close a project. Stub — Wunjo Make always has one project."""
        return True
