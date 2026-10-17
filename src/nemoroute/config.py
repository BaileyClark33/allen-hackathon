from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"
TOKEN_FACTORY = "token_factory"


class ApiConfig(BaseModel):
    base_url: str
    api_key_env: str
    timeout_s: float
    max_retries: int
    backoff_initial_s: float
    backoff_max_s: float
    thinking_control: Literal["chat_template_kwargs", "none"]
    json_mode: Literal["none", "json_object", "json_schema"]


class ModelConfig(BaseModel):
    id: str
    base_url: str | None = None
    api_key_env: str | None = None
    provider: Literal["openai_compatible", "anthropic"] = "openai_compatible"
    effort: str | None = None
    billing: str = "token_factory"
    input_usd_per_mtok: float
    output_usd_per_mtok: float


class RoleConfig(BaseModel):
    model: str
    thinking: bool
    max_tokens: int
    temperature: float
    top_p: float | None = None


class TokenEstimate(BaseModel):
    input: int
    output: int


class BudgetConfig(BaseModel):
    total_credits_usd: float
    dry_run_limit: int
    min_observed_calls: int


class DatasetConfig(BaseModel):
    dir: str
    id_prefix: str
    target: int


class GenerationConfig(BaseModel):
    batch_size: int
    dedup_cosine_threshold: float
    seed: int
    concurrency: int
    max_rounds: int
    subdomains: list[str]
    personas: list[str]


class LockedThresholds(BaseModel):
    pass_threshold: float
    min_accuracy: int
    locked_at: str
    source: str


class LabelingConfig(BaseModel):
    criteria: list[str]
    pass_threshold: float
    min_accuracy: int
    collapse_warn_frac: float
    concurrency: int
    locked: LockedThresholds | None = None


class TuningConfig(BaseModel):
    pass_threshold_grid: list[float]
    min_accuracy_grid: list[int]
    human_per_tier: int
    seed: int


class SplitConfig(BaseModel):
    seed: int
    test_frac: float


class RouterConfig(BaseModel):
    type: str


class EvaluationConfig(BaseModel):
    bootstrap_samples: int
    ci: float
    seed: int
    concurrency: int


class HumanAgreementConfig(BaseModel):
    sample_size: int
    seed: int


class AppConfig(BaseModel):
    session_query_cap: int
    daily_spend_cap_usd: float
    total_spend_cap_usd: float
    max_query_chars: int
    answer_max_tokens: int


class PathsConfig(BaseModel):
    calls_log: str
    app_spend_log: str
    splits_dir: str
    results_dir: str
    prompts_dir: str


class Config(BaseModel):
    api: ApiConfig
    models: dict[str, ModelConfig]
    roles: dict[str, RoleConfig]
    tier_roles: dict[int, str]
    estimates: dict[str, TokenEstimate]
    budget: BudgetConfig
    datasets: dict[str, DatasetConfig]
    generation: GenerationConfig
    labeling: LabelingConfig
    tuning: TuningConfig
    split: SplitConfig
    router: RouterConfig
    evaluation: EvaluationConfig
    human_agreement: HumanAgreementConfig
    app: AppConfig
    paths: PathsConfig

    def path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else ROOT / p

    def model_for_role(self, role: str) -> ModelConfig:
        return self.models[self.roles[role].model]

    def dataset_dir(self, name: str) -> Path:
        return self.path(self.datasets[name].dir)

    def tiers(self) -> list[int]:
        return sorted(self.tier_roles)


def load_config(path: Path | str | None = None) -> Config:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)


def api_key(cfg: Config, env_name: str | None = None) -> str:
    load_dotenv(ROOT / ".env")
    env_name = env_name or cfg.api.api_key_env
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise RuntimeError(f"{env_name} is not set. Add it to .env (see .env.example).")
    return key


def require_locked(cfg: Config) -> LockedThresholds:
    if cfg.labeling.locked is None:
        raise SystemExit(
            "Thresholds are not locked. Run scripts/tune_thresholds.py on the calibration set and lock them first."
        )
    return cfg.labeling.locked


def active_thresholds(cfg: Config) -> tuple[float, int]:
    if cfg.labeling.locked is not None:
        return cfg.labeling.locked.pass_threshold, cfg.labeling.locked.min_accuracy
    return cfg.labeling.pass_threshold, cfg.labeling.min_accuracy


LOCKED_LINE = re.compile(r"^(  locked:).*$", re.MULTILINE)


def write_locked_thresholds(
    pass_threshold: float,
    min_accuracy: int,
    source: str,
    config_path: Path | str | None = None,
) -> LockedThresholds:
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    text = config_path.read_text()
    if not LOCKED_LINE.search(text):
        raise RuntimeError("Could not find 'labeling.locked' in config.yaml")
    locked = LockedThresholds(
        pass_threshold=pass_threshold,
        min_accuracy=min_accuracy,
        locked_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source=source,
    )
    value = (
        f"{{pass_threshold: {locked.pass_threshold}, min_accuracy: {locked.min_accuracy}, "
        f"locked_at: '{locked.locked_at}', source: {locked.source}}}"
    )
    config_path.write_text(LOCKED_LINE.sub(lambda m: f"{m.group(1)} {value}", text, count=1))
    return locked
