"""Shared model/video/dataset path resolution for research scripts.

This module keeps the thesis experiment scripts portable across:
  - the local Windows workspace,
  - the repo-local Linux workstation checkout, and
  - alternate dataset mirror folders used during thesis work.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent


def _first_existing(candidates: List[Path], label: str) -> str:
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    tried = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"Could not resolve {label}. Tried:\n{tried}")


def _first_existing_or_fallback(candidates: List[Path]) -> str:
    """Return the first existing path, else the first candidate unchanged.

    Research scripts often filter to one requested video. In that workflow we
    should not fail at import time just because the rest of the catalog is not
    present on the current machine.
    """
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


MODEL_FILENAMES = {
    "glass": {
        "y8n": "glass_y8n_fast.pt",
        "y8s": "glass_y8s_accurate.pt",
    },
    "porcelain": {
        "y8n": "porcelain_y8n_fast.pt",
        "y8s": "porcelain_y8s_accurate.pt",
    },
    "composite": {
        "y8n": "comp_resplit_y8n_fast.pt",
        "y8s": "comp_resplit_y8s_accurate.pt",
    },
}

VIDEO_FILENAMES = {
    "glass": [
        "20190916-633-01.mp4",
        "20190916-722.mp4",
        "20190918-515.mp4",
        "glass_ins.mp4",
        "20190911-174-01.mp4",
        "20190912-233.mp4",
        "021_YUN_0001 (111).mp4",
        "161_YUN_0001 (96).mp4",
        "YUN_0001 (58).mp4",
    ],
    "porcelain": [
        "UAV_porcelain.mp4",
        "porce.mp4",
        "porcelain_maybe.mp4",
    ],
}

_MODEL_ROOTS = [
    PROJECT_ROOT / "models",
    WORKSPACE_ROOT / "MT_Chirravuri" / "models",
]

_VIDEO_ROOTS = {
    "glass": [
        PROJECT_ROOT / "input_videos" / "glass",
        WORKSPACE_ROOT / "MT_Chirravuri" / "input_videos" / "glass_insulator_videos",
        WORKSPACE_ROOT / "Teja_Master_Thesis_workspace" / "datasets" / "source_videos" / "glass",
    ],
    "porcelain": [
        PROJECT_ROOT / "input_videos" / "porcelain",
        WORKSPACE_ROOT / "MT_Chirravuri" / "input_videos" / "porcelain_insulator_videos",
        WORKSPACE_ROOT / "Teja_Master_Thesis_workspace" / "datasets" / "source_videos" / "porcelain",
        WORKSPACE_ROOT / "ROI_based" / "porcelain_videos",
    ],
}


def resolve_model(material: str, variant: str) -> str:
    filename = MODEL_FILENAMES[material][variant]
    candidates = [root / filename for root in _MODEL_ROOTS]
    return _first_existing(candidates, f"{material}/{variant} model")


def resolve_video(material: str, filename: str) -> str:
    candidates = [root / filename for root in _VIDEO_ROOTS[material]]
    return _first_existing_or_fallback(candidates)


def resolve_dataset_yaml(name: str) -> str:
    if name == "glass":
        filenames = ["glass_insulator.yaml", "glass.yaml"]
    elif name == "porcelain":
        filenames = ["porcelain_insulator.yaml", "porcelain.yaml"]
    else:
        filenames = ["composite_val.yaml"]
    candidates = []
    for filename in filenames:
        candidates.extend([
            PROJECT_ROOT / "research" / filename,
            WORKSPACE_ROOT / "MT_Chirravuri" / "research" / filename,
        ])
    return _first_existing(candidates, f"{name} dataset yaml")


MODELS: Dict[str, Dict[str, str]] = {
    material: {variant: resolve_model(material, variant) for variant in variants}
    for material, variants in MODEL_FILENAMES.items()
}

VIDEOS: Dict[str, List[str]] = {
    material: [resolve_video(material, filename) for filename in filenames]
    for material, filenames in VIDEO_FILENAMES.items()
}

GLASS_VIDEOS = VIDEOS["glass"]
PORCELAIN_VIDEOS = VIDEOS["porcelain"]


def get_datasets() -> Dict[str, Dict[str, str]]:
    return {
        "glass": {
            "data_yaml": resolve_dataset_yaml("glass"),
            "models": MODELS["glass"],
        },
        "porcelain": {
            "data_yaml": resolve_dataset_yaml("porcelain"),
            "models": MODELS["porcelain"],
        },
        "composite": {
            "data_yaml": resolve_dataset_yaml("composite"),
            "models": MODELS["composite"],
        },
    }
