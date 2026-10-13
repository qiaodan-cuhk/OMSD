"""Deterministic experiment grouping with unique runs and explicit dependencies."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import secrets


@dataclass(frozen=True)
class RunArtifacts:
    directory: Path
    checkpoint_path: Path
    config_path: Path
    metrics_path: Path
    component: str
    variant: str

    def step_checkpoint(self, step):
        return self.directory / (self.component + "_step" + str(step) + ".pt")


def _segment(value):
    """A readable, single path segment, with no traversal or path separators."""
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip(".-")
    return value or "unspecified"


def _dataset_id(config):
    label = _segment(config.data_type) + "-n" + str(config.dataset_num)
    if config.data_path:
        path = Path(config.data_path).expanduser().resolve()
        digest = hashlib.sha256(str(path).encode()).hexdigest()[:8]
        label += "-" + _segment(path.stem) + "-" + digest
    return label


def _run_id(config):
    explicit = getattr(config, "run_id", None)
    if explicit is not None:
        if not isinstance(explicit, str) or _segment(explicit) != explicit:
            raise ValueError("run_id must be a nonempty single path segment using letters, numbers, '.', '_' or '-'")
        return explicit
    digest = hashlib.sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()[:8]
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + digest + "-" + secrets.token_hex(3)


def prepare_run(config, dataset, component, variant="main"):
    """Resolve paths and reject prior phase output before creating any files."""
    if component not in {"behavior", "critic", "policy"}:
        raise ValueError("component must be behavior, critic, or policy")
    if variant not in {"main", "gmm", "order"}:
        raise ValueError("variant must be main, gmm, or order")
    if config.output_dir:
        directory = Path(config.output_dir).expanduser().resolve()
    else:
        base = Path(getattr(config, "output_root", "outputs")).expanduser().resolve()
        env_partition = _segment(config.env_id)
        if config.agent_conf:
            env_partition += "-" + _segment(config.agent_conf)
        base = base / _segment(config.profile) / env_partition / _dataset_id(config)
        if variant != "main":
            base = base / "ablation" / variant
        if component == "policy":
            stage, layout = "train", getattr(config, "algorithm", "OMSD")
        elif component == "critic":
            stage, layout = "critic", "ind-iql" if config.mode == "IND" else "joint-iql"
        else:
            stage = "behavior"
            layout = ("seq" if config.mode == "SEQ" else "ind") + "-" + ("gmm" if variant == "gmm" else "diffusion")
        directory = base / stage / _segment(layout)
        if component != "policy" and getattr(config, "pretrain_agent_id", None) is not None:
            directory = directory / ("agent" + str(config.pretrain_agent_id))
        directory = directory / ("seed" + str(config.seed)) / _run_id(config)
    artifacts = RunArtifacts(directory, directory / (component + ".pt"),
                             directory / (component + "_config.json"),
                             directory / (component + "_metrics.jsonl"), component, variant)
    occupied = [path for path in (artifacts.checkpoint_path, artifacts.config_path, artifacts.metrics_path) if path.exists()]
    if directory.exists():
        occupied += list(directory.glob(component + "_step*.pt"))
    if occupied:
        raise FileExistsError("This phase already has output at " + str(occupied[0]) + "; choose a new run ID or output directory")
    return artifacts


def _dependency(path):
    if isinstance(path, (list, tuple)):
        if not path:
            raise ValueError("Checkpoint file lists must not be empty")
        return [_dependency(item) for item in path]
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Input checkpoint must be an explicit file: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest())


def write_run_config(artifacts, config, dataset_summary, dependencies=None):
    """Record resolved configuration and exact checkpoint inputs before training."""
    inputs = {name: _dependency(path) for name, path in (dependencies or {}).items() if path}
    record = dict(config=config.to_dict(), dataset=dataset_summary,
                  run=dict(directory=str(artifacts.directory), component=artifacts.component,
                           variant=artifacts.variant), dependencies=inputs)
    artifacts.directory.mkdir(parents=True, exist_ok=True)
    with artifacts.config_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record
