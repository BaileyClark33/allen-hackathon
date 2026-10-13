from __future__ import annotations

from string import Template

from nemoroute.config import Config


def load_prompt(cfg: Config, name: str) -> Template:
    return Template((cfg.path(cfg.paths.prompts_dir) / f"{name}.txt").read_text())
