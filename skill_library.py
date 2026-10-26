"""Discover project-owned skills and load their instructions only when requested."""

import json
from pathlib import Path, PurePosixPath
import re

import yaml


BUNDLED_SKILLS = Path(__file__).resolve().parent / "skills"
MAX_FILE_BYTES = 128 * 1024
SKILL_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


def _read_text(path: Path) -> str:
    with path.open("rb") as handle:
        data = handle.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"Skill files must be at most {MAX_FILE_BYTES} bytes.")
    if b"\0" in data:
        raise ValueError("Skill files must be UTF-8 text.")
    return data.decode("utf-8-sig")


def _metadata(content: str, folder_name: str) -> dict[str, str]:
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md needs YAML frontmatter with name and description.")
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise ValueError("SKILL.md frontmatter is not closed.")
    data = yaml.safe_load("\n".join(lines[1:end]))
    if not isinstance(data, dict):
        raise ValueError("Skill frontmatter must be a mapping.")
    name, description = data.get("name"), data.get("description")
    if (not isinstance(name, str) or len(name) > 64 or not SKILL_NAME.fullmatch(name)
            or name != folder_name):
        raise ValueError("Skill name must match its folder and use lowercase letters, digits and hyphens (max 64).")
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        raise ValueError("Skill description must be nonempty text (max 1024 characters).")
    if not "\n".join(lines[end + 1:]).strip():
        raise ValueError("SKILL.md needs instructions below its frontmatter.")
    return {"name": name, "description": description.strip()}


def _resource_path(folder: Path, value: str) -> Path:
    relative = PurePosixPath(value)
    if (not value or relative.is_absolute() or ".." in relative.parts or "\\" in value
            or ":" in value or relative == PurePosixPath(".")):
        raise ValueError("Use a relative path inside the skill, without '..'.")
    path = folder.joinpath(*relative.parts)
    # Do not follow symlinks, even when their current target happens to be local.
    current = path
    while current != folder.parent:
        if current.is_symlink():
            raise ValueError("Skill resources cannot use symlinks.")
        current = current.parent
    if not path.resolve().is_relative_to(folder.resolve()) or not path.is_file():
        raise ValueError("Skill resource was not found inside this skill.")
    return path


class SkillLibrary:
    """Bundled skills plus optional selected-project overrides; no global state."""

    def __init__(self, project: Path | None = None, *, bundled: Path = BUNDLED_SKILLS):
        self.roots = [("bundled", Path(bundled).absolute())]
        if project is not None:
            local = Path(project).absolute() / "skills"
            if local.resolve() != Path(bundled).resolve():
                self.roots.append(("project", local))

    def _discover(self):
        found, errors = {}, []
        for source, root in self.roots:
            if not root.exists():
                continue
            if root.is_symlink() or not root.is_dir():
                errors.append({"source": source, "error": "Skill root must be a directory, not a symlink."})
                continue
            try:
                folders = sorted(root.iterdir())
            except OSError:
                errors.append({"source": source, "error": "Cannot list skill root."})
                continue
            for folder in folders:
                if not folder.is_dir() and not folder.is_symlink():
                    continue
                if not (folder / "SKILL.md").exists() and not folder.is_symlink():
                    continue
                try:
                    path = _resource_path(folder, "SKILL.md")
                    metadata = _metadata(_read_text(path), folder.name)
                    found[metadata["name"]] = (metadata | {"source": source}, folder)
                except (OSError, ValueError, yaml.YAMLError):
                    errors.append({"source": source, "name": folder.name,
                                   "error": "Invalid or unreadable SKILL.md; check UTF-8, frontmatter, size and symlinks."})
        return found, errors

    def list_skills(self) -> dict:
        found, errors = self._discover()
        return {"skills": [found[name][0] for name in sorted(found)], "errors": errors}

    def catalog_prompt(self) -> str:
        # Descriptions are routing data. Full instructions stay out of the prompt.
        return "\n\nAvailable skills (name, description, source):\n" + json.dumps(
            self.list_skills(), ensure_ascii=False
        )

    def _find(self, name: str):
        found, _ = self._discover()
        if name not in found:
            raise ValueError(f"Unknown skill '{name}'. Use list_skills to see available names.")
        return found[name]

    def read_skill(self, name: str) -> dict:
        metadata, folder = self._find(name)
        content = _read_text(_resource_path(folder, "SKILL.md"))
        resources = []
        for path in sorted(folder.rglob("*")):
            relative = path.relative_to(folder).as_posix()
            if relative == "SKILL.md" or not path.is_file():
                continue
            try:
                _resource_path(folder, relative)
            except ValueError:
                continue
            resources.append(relative)
        return metadata | {"content": content, "resources": resources}

    def read_skill_resource(self, name: str, path: str) -> dict:
        _, folder = self._find(name)
        return {"name": name, "path": path,
                "content": _read_text(_resource_path(folder, path))}
