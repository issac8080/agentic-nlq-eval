"""Load versioned prompt files. Tokens use <<NAME>> so JSON examples in the files stay literal."""

from __future__ import annotations

from pathlib import Path


class PromptLibrary:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        version_path = self.directory / "VERSION"
        self.version = version_path.read_text(encoding="utf-8").strip()

    def render(self, name: str, **values: str) -> str:
        text = (self.directory / f"{name}.md").read_text(encoding="utf-8")
        for key, value in values.items():
            text = text.replace(f"<<{key}>>", value)
        return text
