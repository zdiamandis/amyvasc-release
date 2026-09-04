"""Compare peri-amygdalar task profiles with candidate upstream territories.

The functions in this module implement the Figure 7/S4/S5 estimands.  They
start from participant fixed-effect beta maps, intersect support across all
seven HCP tasks, extract left and right ROI voxel medians, and compare the
participant-mean task profiles with equal total weight per task.  Profile
correspondence does not establish causal source direction or quantify each
territory's venous contribution.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.processing import resample_from_to
from scipy.ndimage import distance_transform_edt, gaussian_filter


@dataclass(frozen=True)
class Condition:
    """One fixed-effect map in the 23-condition profile."""

    key: str
    task: str
    name: str
    effect: str


CONDITIONS = (
    Condition("emotion__face", "emotion", "face", "fear_canonical"),
    Condition("emotion__shape", "emotion", "shape", "shape_canonical"),
    Condition("wm__0bk_faces", "wm", "0bk_faces", "0bk_faces_canonical"),
    Condition("wm__0bk_places", "wm", "0bk_places", "0bk_places_canonical"),
    Condition("wm__0bk_tools", "wm", "0bk_tools", "0bk_tools_canonical"),
    Condition("wm__0bk_body", "wm", "0bk_body", "0bk_body_canonical"),
    Condition("wm__2bk_faces", "wm", "2bk_faces", "2bk_faces_canonical"),
    Condition("wm__2bk_places", "wm", "2bk_places", "2bk_places_canonical"),
    Condition("wm__2bk_tools", "wm", "2bk_tools", "2bk_tools_canonical"),
    Condition("wm__2bk_body", "wm", "2bk_body", "2bk_body_canonical"),
    Condition("social__mental", "social", "mental", "mental_canonical"),
    Condition("social__random", "social", "random", "random_canonical"),
    Condition("language__story", "language", "story", "story_canonical"),
    Condition("language__math", "language", "math", "math_canonical"),
    Condition("gambling__win", "gambling", "win", "win_canonical"),
    Condition("gambling__loss", "gambling", "loss", "loss_canonical"),
    Condition(
        "relational__relational", "relational", "relational", "relational_canonical"
    ),
    Condition("relational__matching", "relational", "matching", "matching_canonical"),
    Condition("motor__left_hand", "motor", "left_hand", "left_hand_canonical"),
    Condition("motor__right_hand", "motor", "right_hand", "right_hand_canonical"),
    Condition("motor__left_foot", "motor", "left_foot", "left_foot_canonical"),
    Condition("motor__right_foot", "motor", "right_foot", "right_foot_canonical"),
    Condition("motor__tongue", "motor", "tongue", "tongue_canonical"),
)


@dataclass(frozen=True)
class TargetSpec:
    """A peri-amygdalar target or a supplementary positive-control target."""

    key: str
    label: str
    path: Path
    segment: str = ""
    exclusion: str = ""
    bootstrap_seed_offset: int = 0
    bootstrap_draws: int | None = None
    primary: bool = False


@dataclass(frozen=True)
class CandidateSpec:
    """An ROI mask or a composite of previously declared ROI masks."""

    key: str
    label: str
    path: Path | None = None
    values: tuple[int, ...] = ()
    family: str = ""
    members: tuple[str, ...] = ()
    ranked: bool = True
    short_label: str = ""
    long_label: str = ""
    cortical_division: str = ""

    @property
    def is_composite(self) -> bool:
        return bool(self.members)


@dataclass(frozen=True)
class S4PositiveControlSpec:
    """One upstream-system positive control displayed in Figure S4."""

    display_order: int
    target_key: str
    target_label: str
    target_members: tuple[str, ...]
    target_support_mask: Path
    target_profile_rule: str
    atlas_rank_exclusions: tuple[str, ...]
    atlas_screen_n_candidates: int
    source_label: str
    source_members: tuple[str, ...]
    source_support_mask: Path


def read_subjects(path: Path) -> list[str]:
    """Read IDs from a text file or a CSV/TSV subject column."""
    if path.suffix.lower() in {".csv", ".tsv"}:
        separator = "\t" if path.suffix.lower() == ".tsv" else ","
        table = pd.read_csv(path, sep=separator, dtype=str)
        for column in ("subject_id", "subject", "Subject"):
            if column in table:
                values = table[column].dropna().astype(str).tolist()
                break
        else:
            if table.shape[1] != 1:
                raise ValueError(f"No subject column in {path}")
            values = table.iloc[:, 0].dropna().astype(str).tolist()
    else:
        values = [
            line.strip() for line in path.read_text().splitlines() if line.strip()
        ]
    values = list(dict.fromkeys(values))
    if not values:
        raise ValueError(f"No participants found in {path}")
    return values


def _split_values(value: object) -> tuple[str, ...]:
    if pd.isna(value) or not str(value).strip():
        return ()
    return tuple(
        part.strip() for part in str(value).replace(",", ";").split(";") if part.strip()
    )


def _optional_text(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


def _table_path(value: object, table_path: Path) -> Path:
    resolved = Path(str(value)).expanduser()
    return resolved if resolved.is_absolute() else table_path.parent / resolved


def _optional_bool(value: object, *, default: bool) -> bool:
    if pd.isna(value) or not str(value).strip():
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"Expected a boolean value, found {value!r}")


def load_targets(path: Path) -> list[TargetSpec]:
    """Read ``key, label, path[, segment, exclusion]`` from CSV or TSV."""
    table = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    missing = {"key", "label", "path"}.difference(table.columns)
    if missing:
        raise ValueError(f"Target table is missing {sorted(missing)}")
    return [
        TargetSpec(
            key=str(row.key),
            label=str(row.label),
            path=_table_path(row.path, path),
            segment=_optional_text(getattr(row, "segment", "")),
            exclusion=_optional_text(getattr(row, "exclusion", "")),
            bootstrap_seed_offset=int(
                0
                if pd.isna(getattr(row, "bootstrap_seed_offset", 0))
                else getattr(row, "bootstrap_seed_offset", 0)
            ),
            bootstrap_draws=(
                None
                if pd.isna(getattr(row, "bootstrap_draws", np.nan))
                else int(row.bootstrap_draws)
            ),
            primary=_optional_bool(getattr(row, "primary", ""), default=False),
        )
        for row in table.itertuples(index=False)
    ]


def load_candidates(path: Path) -> list[CandidateSpec]:
    """Read candidate masks and composites from CSV or TSV.

    Required columns are ``key`` and ``label``. A mask row has ``path`` and
    optionally ``values`` (semicolon-separated integer labels). A composite
    row instead has semicolon-separated ``members``.
    """
    table = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    missing = {"key", "label"}.difference(table.columns)
    if missing:
        raise ValueError(f"Candidate table is missing {sorted(missing)}")
    candidates: list[CandidateSpec] = []
    for row in table.to_dict("records"):
        members = _split_values(row.get("members", ""))
        raw_values = _split_values(row.get("values", ""))
        values = tuple(int(value) for value in raw_values)
        raw_path = row.get("path", "")
        candidate_path = (
            None
            if pd.isna(raw_path) or not str(raw_path).strip()
            else _table_path(raw_path, path)
        )
        if bool(members) == bool(candidate_path):
            raise ValueError(
                f"Candidate {row['key']} must define either path or members, not both"
            )
        candidates.append(
            CandidateSpec(
                key=str(row["key"]),
                label=str(row["label"]),
                path=candidate_path,
                values=values,
                family=_optional_text(row.get("family", "")),
                members=members,
                ranked=_optional_bool(row.get("ranked", ""), default=not members),
                short_label=_optional_text(row.get("short_label", "")),
                long_label=_optional_text(row.get("long_label", "")),
                cortical_division=_optional_text(row.get("cortical_division", "")),
            )
        )
    keys = [candidate.key for candidate in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("Candidate keys must be unique")
    key_set = set(keys)
    for candidate in candidates:
        unknown = set(candidate.members).difference(key_set)
        if unknown:
            raise ValueError(f"{candidate.key} has unknown members: {sorted(unknown)}")
    return candidates


def load_s4_positive_controls(path: Path) -> list[S4PositiveControlSpec]:
    """Read the positive-control specification written by ``01_prepare_masks.py``."""
    table = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",")
    required = {
        "display_order",
        "target_key",
        "target_label",
        "target_members",
        "target_support_mask",
        "target_profile_rule",
        "atlas_rank_exclusions",
        "atlas_screen_n_candidates",
        "source_label",
        "source_members",
        "source_support_mask",
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Positive-control table is missing {sorted(missing)}")
    controls: list[S4PositiveControlSpec] = []
    for row in table.to_dict("records"):
        target_members = _split_values(row["target_members"])
        source_members = _split_values(row["source_members"])
        exclusions = _split_values(row["atlas_rank_exclusions"])
        if not target_members or not source_members:
            raise ValueError(
                "Positive-control target and source members cannot be empty"
            )
        if set(exclusions) != set(target_members):
            raise ValueError(
                f"{row['target_key']}: atlas exclusions must equal target members"
            )
        expected_rule = (
            "weighted z-score"
            if len(target_members) == 1
            else (
                "weighted z-score of the mean of separately weighted-z-scored "
                "member profiles"
            )
        )
        if str(row["target_profile_rule"]) != expected_rule:
            raise ValueError(
                f"{row['target_key']}: unsupported target profile rule "
                f"{row['target_profile_rule']!r}"
            )
        controls.append(
            S4PositiveControlSpec(
                display_order=int(row["display_order"]),
                target_key=str(row["target_key"]),
                target_label=str(row["target_label"]),
                target_members=target_members,
                target_support_mask=_table_path(row["target_support_mask"], path),
                target_profile_rule=expected_rule,
                atlas_rank_exclusions=exclusions,
                atlas_screen_n_candidates=int(row["atlas_screen_n_candidates"]),
                source_label=str(row["source_label"]),
                source_members=source_members,
                source_support_mask=_table_path(row["source_support_mask"], path),
            )
        )
    orders = [control.display_order for control in controls]
    keys = [control.target_key for control in controls]
    if len(orders) != len(set(orders)) or len(keys) != len(set(keys)):
        raise ValueError(
            "Positive-control display orders and target keys must be unique"
        )
    return sorted(controls, key=lambda control: control.display_order)


def same_grid(
    a: nib.spatialimages.SpatialImage, b: nib.spatialimages.SpatialImage
) -> bool:
    """Return whether two images share shape and affine."""
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=1e-4)


def _load_on_reference(
    path: Path, reference: nib.Nifti1Image, *, labels: bool = False
) -> np.ndarray:
    image = nib.load(str(path))
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D image: {path}")
    if not same_grid(image, reference):
        image = resample_from_to(image, reference, order=0 if labels else 1)
    return np.asarray(image.dataobj)


def candidate_masks(
    candidates: Sequence[CandidateSpec],
    reference: nib.Nifti1Image,
    *,
    gray_matter_support: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Load every non-composite candidate, optionally restricting its support."""
    if gray_matter_support is not None:
        gray_matter_support = np.asarray(gray_matter_support, dtype=bool)
        if gray_matter_support.shape != reference.shape[:3]:
            raise ValueError("Gray-matter support shape differs from the reference")
    masks: dict[str, np.ndarray] = {}
    for candidate in candidates:
        if candidate.is_composite:
            continue
        if candidate.path is None:
            raise ValueError(f"Candidate {candidate.key} has no path")
        data = _load_on_reference(
            candidate.path, reference, labels=bool(candidate.values)
        )
        mask = (
            np.isin(np.rint(data).astype(int), candidate.values)
            if candidate.values
            else data > 0
        )
        if not np.any(mask):
            raise ValueError(f"Candidate mask is empty: {candidate.key}")
        if gray_matter_support is not None:
            mask &= gray_matter_support
            if not np.any(mask):
                raise ValueError(
                    f"Candidate is empty after gray-matter restriction: {candidate.key}"
                )
        masks[candidate.key] = mask
    return masks


def unrestricted_profile_column(candidate_key: str) -> str:
    """Name a raw atlas profile used only as an S4 control target."""
    return f"unrestricted_profile__{candidate_key}"


def load_gray_matter_support(
    path: Path,
    reference: nib.Nifti1Image,
) -> np.ndarray:
    """Load a binary support mask without resampling its fixed analysis grid."""
    image = nib.load(str(path))
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D gray-matter support mask: {path}")
    if not same_grid(image, reference):
        raise ValueError(
            f"Gray-matter support does not share the fixed-effect grid: {path}"
        )
    values = np.asarray(image.dataobj)
    if not np.isfinite(values).all():
        raise ValueError(f"Gray-matter support contains non-finite values: {path}")
    support = values > 0
    if not np.any(support):
        raise ValueError(f"Gray-matter support is empty: {path}")
    return support


def validate_s4_positive_control_support(
    controls: Sequence[S4PositiveControlSpec],
    candidates: Sequence[CandidateSpec],
    reference: nib.Nifti1Image,
    *,
    inventory: pd.DataFrame | None = None,
    extraction_target_key: str | None = None,
) -> None:
    """Validate control support masks and atlas-screen exclusions."""
    raw_candidates = [
        candidate for candidate in candidates if not candidate.is_composite
    ]
    by_key = {candidate.key: candidate for candidate in raw_candidates}
    masks = candidate_masks(raw_candidates, reference)
    ranked = {candidate.key for candidate in raw_candidates if candidate.ranked}

    for control in controls:
        members = (*control.target_members, *control.source_members)
        unknown = set(members).difference(by_key)
        if unknown:
            raise ValueError(
                f"{control.target_key}: unknown control members {sorted(unknown)}"
            )
        unranked = set(control.atlas_rank_exclusions).difference(ranked)
        if unranked:
            raise ValueError(
                f"{control.target_key}: exclusions are not ranked atlas parcels: "
                f"{sorted(unranked)}"
            )
        screen_size = len(ranked.difference(control.atlas_rank_exclusions))
        if screen_size != control.atlas_screen_n_candidates:
            raise ValueError(
                f"{control.target_key}: expected {control.atlas_screen_n_candidates} "
                f"ranked candidates after exclusion, found {screen_size}"
            )

        target_union = np.logical_or.reduce(
            [masks[key] for key in control.target_members]
        )
        source_union = np.logical_or.reduce(
            [masks[key] for key in control.source_members]
        )
        declared_target = (
            _load_on_reference(control.target_support_mask, reference, labels=True) > 0
        )
        declared_source = (
            _load_on_reference(control.source_support_mask, reference, labels=True) > 0
        )
        if not np.array_equal(declared_target, target_union):
            raise ValueError(
                f"{control.target_key}: target support mask differs from its members"
            )
        if not np.array_equal(declared_source, source_union):
            raise ValueError(
                f"{control.target_key}: source support mask differs from its members"
            )
        if np.any(target_union & source_union):
            raise ValueError(
                f"{control.target_key}: target and source supports overlap"
            )

    if inventory is None:
        return
    required = {"target_key", "candidate_key", "overlap_removed"}
    missing = required.difference(inventory.columns)
    if missing:
        raise ValueError(f"Candidate inventory is missing {sorted(missing)}")
    if extraction_target_key is None:
        raise ValueError("extraction_target_key is required when inventory is supplied")
    local = inventory.loc[inventory["target_key"].eq(extraction_target_key)]
    if local.empty:
        raise ValueError(f"No inventory rows for target {extraction_target_key}")
    control_members = {
        key
        for control in controls
        for key in (*control.target_members, *control.source_members)
    }
    overlap = (
        local.loc[local["candidate_key"].isin(control_members)]
        .groupby("candidate_key")["overlap_removed"]
        .sum()
    )
    missing_members = control_members.difference(overlap.index)
    if missing_members:
        raise ValueError(f"Inventory has no support rows for {sorted(missing_members)}")
    affected = overlap.loc[overlap.gt(0)]
    if not affected.empty:
        raise ValueError(
            "Positive-control profiles were altered by extraction-target overlap: "
            + ", ".join(
                f"{key} ({int(value)} voxels)" for key, value in affected.items()
            )
        )


def _world_hemispheres(reference: nib.Nifti1Image) -> dict[str, np.ndarray]:
    ijk = np.indices(reference.shape[:3]).reshape(3, -1).T
    x = nib.affines.apply_affine(reference.affine, ijk)[:, 0].reshape(
        reference.shape[:3]
    )
    return {"left": x < 0, "right": x > 0}


def _effect_path(root: Path, subject: str, condition: Condition) -> Path:
    return (
        root
        / condition.task
        / "fixed_effects"
        / f"sub-{subject}"
        / "contrasts"
        / f"{condition.effect}_effect.nii.gz"
    )


def _support_path(root: Path, subject: str, task: str) -> Path:
    return root / task / "fixed_effects" / f"sub-{subject}" / "support_mask.nii.gz"


def find_reference(
    fixed_effects_root: Path,
    subjects: Sequence[str],
    conditions: Sequence[Condition] = CONDITIONS,
) -> nib.Nifti1Image:
    """Load the first fixed-effect map and use it as the analysis grid."""
    for subject in subjects:
        for condition in conditions:
            path = _effect_path(fixed_effects_root, subject, condition)
            if path.exists():
                image = nib.load(str(path))
                if image.ndim != 3:
                    raise ValueError(f"Expected a 3D fixed-effect image: {path}")
                return image
    raise FileNotFoundError(f"No fixed-effect maps found under {fixed_effects_root}")


def common_support(
    fixed_effects_root: Path,
    subject: str,
    reference: nib.Nifti1Image,
    conditions: Sequence[Condition] = CONDITIONS,
) -> np.ndarray:
    """Intersect the participant's support masks across the seven tasks."""
    tasks = tuple(dict.fromkeys(condition.task for condition in conditions))
    if len(tasks) != 7:
        raise ValueError(
            f"The source profile must cover seven tasks; found {len(tasks)}"
        )
    support = np.ones(reference.shape[:3], dtype=bool)
    for task in tasks:
        path = _support_path(fixed_effects_root, subject, task)
        image = nib.load(str(path))
        if not same_grid(image, reference):
            raise ValueError(f"Support mask grid differs from fixed effects: {path}")
        support &= np.asarray(image.dataobj) > 0
    return support


def _finite_median(data: np.ndarray, mask: np.ndarray) -> float:
    values = np.asarray(data[mask], dtype=np.float32)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def extract_profiles(
    *,
    subjects: Sequence[str],
    fixed_effects_root: Path,
    targets: Sequence[TargetSpec],
    candidates: Sequence[CandidateSpec],
    gray_matter_support_path: Path,
    unrestricted_profile_keys: Sequence[str] = (),
    conditions: Sequence[Condition] = CONDITIONS,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, nib.Nifti1Image]:
    """Extract participant/condition/hemisphere ROI medians for every target.

    Participant IDs are used only to locate inputs. Returned tables contain a
    sequential ``participant`` index and remain local analysis outputs.
    Every source candidate is intersected with the gray-matter support before
    target overlap is removed. Targets themselves are never restricted.
    ``unrestricted_profile_keys`` adds raw atlas profiles for S4 control
    targets; those columns cannot enter the source-candidate screen.
    """
    reference = find_reference(fixed_effects_root, subjects, conditions)
    pre_support_candidates = candidate_masks(candidates, reference)
    gray_matter_support = load_gray_matter_support(
        gray_matter_support_path,
        reference,
    )
    source_candidates = candidate_masks(
        candidates,
        reference,
        gray_matter_support=gray_matter_support,
    )
    unknown_unrestricted = set(unrestricted_profile_keys).difference(
        pre_support_candidates
    )
    if unknown_unrestricted:
        raise ValueError(
            f"Unknown unrestricted control targets: {sorted(unknown_unrestricted)}"
        )
    hemispheres = _world_hemispheres(reference)
    target_masks = {
        target.key: _load_on_reference(target.path, reference, labels=True) > 0
        for target in targets
    }
    if any(not np.any(mask) for mask in target_masks.values()):
        empty = [key for key, mask in target_masks.items() if not np.any(mask)]
        raise ValueError(f"Empty target masks: {empty}")

    prepared: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    unrestricted: dict[str, dict[str, np.ndarray]] = {
        key: {} for key in unrestricted_profile_keys
    }
    inventory_rows: list[dict[str, object]] = []
    for target in targets:
        prepared[target.key] = {}
        target_mask = target_masks[target.key]
        for candidate in candidates:
            if candidate.is_composite:
                continue
            prepared[target.key][candidate.key] = {}
            for hemisphere, hemi_mask in hemispheres.items():
                pre_support = pre_support_candidates[candidate.key] & hemi_mask
                supported = source_candidates[candidate.key] & hemi_mask
                overlap = supported & target_mask
                clean = supported & ~target_mask
                prepared[target.key][candidate.key][hemisphere] = clean
                if candidate.key in unrestricted:
                    unrestricted[candidate.key][hemisphere] = pre_support
                inventory_rows.append(
                    {
                        "target_key": target.key,
                        "target_label": target.label,
                        "candidate_key": candidate.key,
                        "candidate_label": candidate.label,
                        "source_short_label": candidate.short_label,
                        "source_long_label": candidate.long_label,
                        "source_cortical_division": candidate.cortical_division,
                        "family": candidate.family,
                        "hemisphere": hemisphere,
                        "target_voxels": int((target_mask & hemi_mask).sum()),
                        "gray_matter_support_source": gray_matter_support_path.name,
                        "raw_voxels": int(pre_support.sum()),
                        "pre_support_voxels": int(pre_support.sum()),
                        "gray_matter_supported_voxels": int(supported.sum()),
                        "gray_matter_excluded_voxels": int(
                            (pre_support & ~supported).sum()
                        ),
                        "overlap_removed": int(overlap.sum()),
                        "retained_voxels": int(clean.sum()),
                        "status": "ok",
                        "source": (
                            candidate.path.name
                            if candidate.path is not None
                            else ";".join(candidate.members)
                        ),
                    }
                )

    rows = {target.key: [] for target in targets}
    for participant, subject in enumerate(subjects, start=1):
        support = common_support(fixed_effects_root, subject, reference, conditions)
        indices: dict[str, dict[str, np.ndarray]] = {}
        for target in targets:
            target_mask = target_masks[target.key]
            indices[target.key] = {
                hemisphere: support & target_mask & hemi_mask
                for hemisphere, hemi_mask in hemispheres.items()
            }
        candidate_indices = {
            target.key: {
                candidate.key: {
                    hemisphere: support
                    & prepared[target.key][candidate.key][hemisphere]
                    for hemisphere in hemispheres
                }
                for candidate in candidates
                if not candidate.is_composite
            }
            for target in targets
        }
        unrestricted_indices = {
            key: {hemisphere: support & mask for hemisphere, mask in masks.items()}
            for key, masks in unrestricted.items()
        }

        for condition in conditions:
            path = _effect_path(fixed_effects_root, subject, condition)
            image = nib.load(str(path))
            if not same_grid(image, reference):
                raise ValueError(f"Fixed-effect grid differs from reference: {path}")
            data = np.asarray(image.dataobj, dtype=np.float32)
            for target in targets:
                for hemisphere in hemispheres:
                    row: dict[str, object] = {
                        "participant": participant,
                        "task": condition.task,
                        "condition": condition.name,
                        "condition_key": condition.key,
                        "hemisphere": hemisphere,
                        target.key: _finite_median(
                            data, indices[target.key][hemisphere]
                        ),
                    }
                    for candidate in candidates:
                        if not candidate.is_composite:
                            row[candidate.key] = _finite_median(
                                data,
                                candidate_indices[target.key][candidate.key][
                                    hemisphere
                                ],
                            )
                    for candidate_key in unrestricted_profile_keys:
                        row[unrestricted_profile_column(candidate_key)] = (
                            _finite_median(
                                data,
                                unrestricted_indices[candidate_key][hemisphere],
                            )
                        )
                    rows[target.key].append(row)
    return (
        {key: pd.DataFrame(value) for key, value in rows.items()},
        pd.DataFrame(inventory_rows),
        reference,
    )


def task_equal_weights(cells: pd.DataFrame) -> np.ndarray:
    """Give each task equal total weight across its condition/hemisphere cells."""
    counts = cells.groupby("task")["task"].transform("size").to_numpy(float)
    weights = 1.0 / counts
    return weights / weights.sum()


def weighted_zscore(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    local = weights * finite
    if local.sum() <= 0:
        return np.full_like(values, np.nan)
    local /= local.sum()
    mean = np.nansum(local * values)
    variance = np.nansum(local * (values - mean) ** 2)
    return (
        (values - mean) / np.sqrt(variance)
        if variance > 0
        else np.full_like(values, np.nan)
    )


def weighted_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """Weighted Pearson correlation over finite paired profile cells."""
    finite = np.isfinite(x) & np.isfinite(y)
    local = weights * finite
    if local.sum() <= 0:
        return np.nan
    local /= local.sum()
    xc = x - np.nansum(local * x)
    yc = y - np.nansum(local * y)
    denominator = np.sqrt(np.nansum(local * xc**2) * np.nansum(local * yc**2))
    return (
        float(np.nansum(local * xc * yc) / denominator) if denominator > 0 else np.nan
    )


def weighted_corr_rows(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Vectorized weighted Pearson correlation for bootstrap rows."""
    finite = np.isfinite(x) & np.isfinite(y)
    local = finite * weights[None, :]
    sums = local.sum(axis=1, keepdims=True)
    local = np.divide(local, sums, out=np.zeros_like(local), where=sums > 0)
    safe_x = np.where(finite, x, 0.0)
    safe_y = np.where(finite, y, 0.0)
    mx = (local * safe_x).sum(axis=1, keepdims=True)
    my = (local * safe_y).sum(axis=1, keepdims=True)
    xc = np.where(finite, x - mx, 0.0)
    yc = np.where(finite, y - my, 0.0)
    numerator = (local * xc * yc).sum(axis=1)
    denominator = np.sqrt((local * xc**2).sum(axis=1) * (local * yc**2).sum(axis=1))
    return np.divide(
        numerator, denominator, out=np.full(x.shape[0], np.nan), where=denominator > 0
    )


def _standardized_composite_profile(
    group: pd.DataFrame,
    members: Sequence[str],
    weights: np.ndarray,
) -> np.ndarray:
    """Standardize members separately, average them, and standardize again."""
    component_profiles = np.column_stack(
        [weighted_zscore(group[member].to_numpy(float), weights) for member in members]
    )
    return weighted_zscore(np.nanmean(component_profiles, axis=1), weights)


def _weighted_r2(
    outcome: np.ndarray,
    predictors: Sequence[np.ndarray],
    weights: np.ndarray,
) -> float:
    """Weighted least-squares R2 with an intercept."""
    finite = np.isfinite(outcome)
    for predictor in predictors:
        finite &= np.isfinite(predictor)
    if finite.sum() <= len(predictors) + 1:
        return np.nan
    local_weights = np.asarray(weights[finite], float)
    local_weights /= local_weights.sum()
    local_outcome = np.asarray(outcome[finite], float)
    design = np.column_stack(
        [
            np.ones(finite.sum()),
            *(np.asarray(value[finite], float) for value in predictors),
        ]
    )
    sqrt_weights = np.sqrt(local_weights)
    beta, *_ = np.linalg.lstsq(
        design * sqrt_weights[:, None],
        local_outcome * sqrt_weights,
        rcond=None,
    )
    residual = local_outcome - design @ beta
    mean = float(np.sum(local_weights * local_outcome))
    total = float(np.sum(local_weights * (local_outcome - mean) ** 2))
    return (
        1.0 - float(np.sum(local_weights * residual**2)) / total
        if total > 0
        else np.nan
    )


def analyze_s4_positive_controls(
    wide: pd.DataFrame,
    controls: Sequence[S4PositiveControlSpec],
    candidates: Sequence[CandidateSpec],
    *,
    reference_key: str = "amy_proper",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the reported V2/V1 and auditory-belt/A1 profile screens.

    Participant profiles are first averaged within each condition/hemisphere
    cell. Each HCP task receives equal total weight. Composite target members
    are separately task-weighted-z-scored, averaged, and restandardized before
    their correlations and pairwise commonality terms are calculated.
    """
    if not controls:
        raise ValueError("No Figure S4 positive controls were supplied")
    participant_column = next(
        (column for column in ("participant", "subject") if column in wide), None
    )
    if participant_column is None:
        raise ValueError("Positive-control profiles require a participant column")
    metadata = ["task", "condition_key", "hemisphere"]
    missing_metadata = set(metadata).difference(wide.columns)
    if missing_metadata:
        raise ValueError(
            f"Positive-control profiles are missing {sorted(missing_metadata)}"
        )

    ranked_candidates = [
        candidate
        for candidate in candidates
        if candidate.ranked and not candidate.is_composite
    ]
    ranked_keys = [candidate.key for candidate in ranked_candidates]
    if reference_key not in ranked_keys:
        raise ValueError(f"Reference candidate not found: {reference_key}")
    unrestricted_target_keys = {
        member for control in controls for member in control.target_members
    }
    target_profile_columns = {
        key: unrestricted_profile_column(key) for key in unrestricted_target_keys
    }
    required_profiles = {*ranked_keys, *target_profile_columns.values()}
    missing_profiles = required_profiles.difference(wide.columns)
    if missing_profiles:
        raise ValueError(
            f"Positive-control input is missing profiles: {sorted(missing_profiles)}"
        )

    condition_order = {
        condition.key: index for index, condition in enumerate(CONDITIONS)
    }
    frame = wide[
        [participant_column, *metadata, *ranked_keys, *target_profile_columns.values()]
    ].copy()
    frame["_condition_order"] = frame["condition_key"].map(condition_order)
    frame["_hemisphere_order"] = frame["hemisphere"].map({"left": 0, "right": 1})
    if frame[["_condition_order", "_hemisphere_order"]].isna().any().any():
        raise ValueError(
            "Unexpected condition or hemisphere in positive-control profiles"
        )
    frame = frame.sort_values(
        [participant_column, "_condition_order", "_hemisphere_order"]
    )
    per_participant = frame.groupby(participant_column, sort=False).size()
    expected_cells = 2 * len(CONDITIONS)
    if not per_participant.eq(expected_cells).all():
        raise ValueError(
            f"Every participant must have all {expected_cells} profile cells"
        )
    if frame.duplicated([participant_column, *metadata]).any():
        raise ValueError("Positive-control input contains duplicate profile cells")

    group = frame.groupby(
        ["_condition_order", "_hemisphere_order", *metadata],
        as_index=False,
        sort=True,
    )[[*ranked_keys, *target_profile_columns.values()]].mean()
    if len(group) != expected_cells:
        raise ValueError(
            f"Expected {expected_cells} group profile cells, found {len(group)}"
        )
    if group["condition_key"].nunique() != len(CONDITIONS):
        raise ValueError("Positive controls require all 23 HCP task conditions")
    if group["task"].nunique() != 7:
        raise ValueError("Positive controls require all seven HCP tasks")
    if not np.isfinite(
        group[[*ranked_keys, *target_profile_columns.values()]].to_numpy(float)
    ).all():
        raise ValueError(
            "Positive-control atlas profiles contain non-finite group means"
        )

    weights = task_equal_weights(group)
    amygdala = weighted_zscore(group[reference_key].to_numpy(float), weights)
    metric_rows: list[dict[str, object]] = []
    ranking_rows: list[dict[str, object]] = []
    for control in controls:
        if set(control.target_members).difference(ranked_keys):
            raise ValueError(f"{control.target_key}: target members are unavailable")
        if set(control.source_members).difference(ranked_keys):
            raise ValueError(f"{control.target_key}: source members are unavailable")
        if len(control.source_members) != 1:
            raise ValueError(
                f"{control.target_key}: atlas ranking requires one expected source member"
            )
        target_profile_members = tuple(
            target_profile_columns[member] for member in control.target_members
        )
        target = _standardized_composite_profile(
            group,
            target_profile_members,
            weights,
        )
        source = _standardized_composite_profile(group, control.source_members, weights)
        excluded = set(control.atlas_rank_exclusions)
        screen = [
            candidate
            for candidate in ranked_candidates
            if candidate.key not in excluded
        ]
        if len(screen) != control.atlas_screen_n_candidates:
            raise ValueError(
                f"{control.target_key}: expected {control.atlas_screen_n_candidates} "
                f"screen candidates, found {len(screen)}"
            )
        correlations = {
            candidate.key: weighted_corr(
                target, group[candidate.key].to_numpy(float), weights
            )
            for candidate in screen
        }
        ranking = sorted(
            screen,
            key=lambda candidate: (-correlations[candidate.key], candidate.key),
        )
        expected_source = control.source_members[0]
        source_rank = next(
            index
            for index, candidate in enumerate(ranking, start=1)
            if candidate.key == expected_source
        )
        for rank, candidate in enumerate(ranking, start=1):
            ranking_rows.append(
                {
                    "display_order": control.display_order,
                    "target_key": control.target_key,
                    "candidate_key": candidate.key,
                    "candidate_label": candidate.label,
                    "family": candidate.family,
                    "group_profile_r": correlations[candidate.key],
                    "rank": rank,
                    "expected_source": candidate.key == expected_source,
                }
            )

        amy_r2 = _weighted_r2(target, [amygdala], weights)
        source_r2 = _weighted_r2(target, [source], weights)
        both_r2 = _weighted_r2(target, [amygdala, source], weights)
        metric_rows.append(
            {
                "display_order": control.display_order,
                "target_key": control.target_key,
                "target_label": control.target_label,
                "target_members": ";".join(control.target_members),
                "target_profile_members": ";".join(target_profile_members),
                "source_label": control.source_label,
                "source_members": ";".join(control.source_members),
                "n_profile_cells": len(group),
                "target_source_r": weighted_corr(target, source, weights),
                "expected_source_atlas_rank": source_rank,
                "atlas_screen_n_candidates": len(screen),
                "atlas_screen_excluded_target_members": ";".join(
                    control.atlas_rank_exclusions
                ),
                "atlas_screen_expected_source_r": correlations[expected_source],
                "r2_amy_alone": amy_r2,
                "r2_rival_alone": source_r2,
                "r2_both": both_r2,
                "unique_amy_beyond_rival": both_r2 - source_r2,
                "unique_rival_beyond_amy": both_r2 - amy_r2,
                "common_amy_rival": amy_r2 + source_r2 - both_r2,
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(ranking_rows)


def bootstrap_weights(n_participants: int, n_bootstrap: int, seed: int) -> np.ndarray:
    """Create participant-resampling weights shared across all comparisons."""
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, n_participants, size=(n_bootstrap, n_participants))
    weights = np.zeros((n_bootstrap, n_participants), dtype=np.float32)
    for draw, indices in enumerate(sampled):
        weights[draw] = np.bincount(indices, minlength=n_participants) / n_participants
    return weights


def bootstrap_mean(matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Participant-bootstrap means with cell-wise finite support."""
    finite = np.isfinite(values)
    safe = np.where(finite, values, 0.0)
    numerator = np.einsum("bs,sck->bck", matrix, safe, optimize=True)
    denominator = np.einsum("bs,sck->bck", matrix, finite.astype(float), optimize=True)
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=denominator > 0,
    )


def _zscore_rows(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    local = finite * weights[None, :]
    sums = local.sum(axis=1, keepdims=True)
    local = np.divide(local, sums, out=np.zeros_like(local), where=sums > 0)
    safe = np.where(finite, values, 0.0)
    mean = (local * safe).sum(axis=1, keepdims=True)
    variance = (local * np.where(finite, (values - mean) ** 2, 0.0)).sum(
        axis=1, keepdims=True
    )
    return np.divide(
        values - mean,
        np.sqrt(variance),
        out=np.full_like(values, np.nan),
        where=variance > 0,
    )


def _composite(
    group: np.ndarray, boot: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    standardized_group = np.column_stack(
        [weighted_zscore(group[:, index], weights) for index in range(group.shape[1])]
    )
    standardized_boot = np.stack(
        [_zscore_rows(boot[:, :, index], weights) for index in range(boot.shape[2])],
        axis=2,
    )
    return np.nanmean(standardized_group, axis=1), np.nanmean(standardized_boot, axis=2)


def _two_predictor_r2(
    r_y1: np.ndarray, r_y2: np.ndarray, r_12: np.ndarray
) -> np.ndarray:
    denominator = 1.0 - r_12**2
    numerator = r_y1**2 + r_y2**2 - 2.0 * r_y1 * r_y2 * r_12
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan),
        where=np.abs(denominator) > 1e-10,
    )


def analyze_profile(
    wide: pd.DataFrame,
    target: TargetSpec,
    candidates: Sequence[CandidateSpec],
    *,
    reference_key: str = "amy_proper",
    n_bootstrap: int = 1000,
    seed: int = 20_260_710,
    exclude_tasks: Iterable[str] = (),
) -> dict[str, pd.DataFrame]:
    """Calculate ranks, bootstrap intervals, paired differences, and commonality."""
    keep = ~wide["task"].isin(set(exclude_tasks))
    frame = wide.loc[keep].copy()
    condition_order = {
        condition.key: index for index, condition in enumerate(CONDITIONS)
    }
    frame["_condition_order"] = frame["condition_key"].map(condition_order)
    frame["_hemisphere_order"] = frame["hemisphere"].map({"left": 0, "right": 1})
    if frame[["_condition_order", "_hemisphere_order"]].isna().any().any():
        raise ValueError("Unexpected condition or hemisphere in source profiles")
    frame = frame.sort_values(
        ["participant", "_condition_order", "_hemisphere_order"]
    ).drop(columns=["_condition_order", "_hemisphere_order"])
    participants = frame["participant"].drop_duplicates().tolist()
    first = frame.loc[frame["participant"].eq(participants[0])]
    cells = first[["task", "condition_key", "hemisphere"]].reset_index(drop=True)
    expected_cells = len(cells)
    if not frame.groupby("participant").size().eq(expected_cells).all():
        raise ValueError(
            "Every participant must have the same condition/hemisphere cells"
        )
    for participant in participants:
        observed = frame.loc[
            frame["participant"].eq(participant),
            ["task", "condition_key", "hemisphere"],
        ].reset_index(drop=True)
        if not observed.equals(cells):
            raise ValueError(f"Profile cells differ for participant {participant}")
    raw = [candidate for candidate in candidates if not candidate.is_composite]
    raw_keys = [candidate.key for candidate in raw]
    values = (
        frame[[target.key, *raw_keys]]
        .to_numpy(float)
        .reshape(len(participants), expected_cells, len(raw_keys) + 1)
    )
    weights = task_equal_weights(cells)
    matrix = bootstrap_weights(len(participants), n_bootstrap, seed)
    group = np.nanmean(values, axis=0)
    boot = bootstrap_mean(matrix, values)
    target_group, target_boot = group[:, 0], boot[:, :, 0]

    groups = {key: group[:, index + 1] for index, key in enumerate(raw_keys)}
    boots = {key: boot[:, :, index + 1] for index, key in enumerate(raw_keys)}
    participant_profiles = {
        key: values[:, :, index + 1] for index, key in enumerate(raw_keys)
    }
    for candidate in candidates:
        if not candidate.is_composite:
            continue
        indices = [raw_keys.index(member) + 1 for member in candidate.members]
        groups[candidate.key], boots[candidate.key] = _composite(
            group[:, indices], boot[:, :, indices], weights
        )
        participant_profiles[candidate.key] = np.nanmean(
            np.stack(
                [
                    _zscore_rows(values[:, :, member_index], weights)
                    for member_index in indices
                ],
                axis=2,
            ),
            axis=2,
        )

    correlation_rows: list[dict[str, object]] = []
    bootstrap_correlations: dict[str, np.ndarray] = {}
    for candidate in candidates:
        point = weighted_corr(target_group, groups[candidate.key], weights)
        draws = weighted_corr_rows(target_boot, boots[candidate.key], weights)
        participant_r = weighted_corr_rows(
            values[:, :, 0], participant_profiles[candidate.key], weights
        )
        bootstrap_correlations[candidate.key] = draws
        low, median, high = np.nanquantile(draws, [0.025, 0.5, 0.975])
        correlation_rows.append(
            {
                "target_key": target.key,
                "target_label": target.label,
                "segment": target.segment,
                "exclusion": target.exclusion,
                "candidate_key": candidate.key,
                "candidate_label": candidate.label,
                "source_short_label": candidate.short_label,
                "source_long_label": candidate.long_label,
                "source_cortical_division": candidate.cortical_division,
                "family": candidate.family,
                "ranked": candidate.ranked,
                "group_r": point,
                "bootstrap_median": median,
                "bootstrap_q025": low,
                "bootstrap_q975": high,
                "participant_median_r": np.nanmedian(participant_r),
                "participant_q25_r": np.nanquantile(participant_r, 0.25),
                "participant_q75_r": np.nanquantile(participant_r, 0.75),
                "n_participants": len(participants),
                "n_conditions": int(cells["condition_key"].nunique()),
                "n_tasks": int(cells["task"].nunique()),
            }
        )
    correlations = pd.DataFrame(correlation_rows).sort_values(
        ["group_r", "candidate_key"], ascending=[False, True]
    )
    correlations["rank"] = pd.Series(pd.NA, index=correlations.index, dtype="Int64")
    ranked_rows = correlations["ranked"].astype(bool)
    correlations.loc[ranked_rows, "rank"] = np.arange(1, ranked_rows.sum() + 1)
    ranked_candidates = [candidate for candidate in candidates if candidate.ranked]
    draw_matrix = np.column_stack(
        [bootstrap_correlations[candidate.key] for candidate in ranked_candidates]
    )
    valid_draws = np.isfinite(draw_matrix).any(axis=1)
    winners = np.argmax(
        np.where(np.isfinite(draw_matrix), draw_matrix, -np.inf), axis=1
    )
    win_probability = {
        candidate.key: float(np.mean(winners[valid_draws] == index))
        for index, candidate in enumerate(ranked_candidates)
    }
    correlations["bootstrap_first_probability"] = correlations["candidate_key"].map(
        win_probability
    )

    if reference_key not in groups:
        raise ValueError(f"Reference candidate not found: {reference_key}")
    candidate_labels = {candidate.key: candidate.label for candidate in candidates}
    reference = groups[reference_key]
    reference_boot = boots[reference_key]
    r_reference = weighted_corr(target_group, reference, weights)
    r_reference_boot = bootstrap_correlations[reference_key]
    paired_rows: list[dict[str, object]] = []
    commonality_rows: list[dict[str, object]] = []
    for candidate in candidates:
        if candidate.key == reference_key:
            continue
        rival = groups[candidate.key]
        rival_boot = boots[candidate.key]
        r_rival = weighted_corr(target_group, rival, weights)
        r_rival_boot = bootstrap_correlations[candidate.key]
        difference = r_reference_boot - r_rival_boot
        low, median, high = np.nanquantile(difference, [0.025, 0.5, 0.975])
        paired_rows.append(
            {
                "target_key": target.key,
                "reference_key": reference_key,
                "reference_label": candidate_labels[reference_key],
                "rival_key": candidate.key,
                "rival_label": candidate.label,
                "reference_r": r_reference,
                "rival_r": r_rival,
                "r_difference": r_reference - r_rival,
                "bootstrap_median_difference": median,
                "bootstrap_q025": low,
                "bootstrap_q975": high,
                "probability_reference_greater": float(np.nanmean(difference > 0)),
                "n_participants": len(participants),
                "n_bootstrap": n_bootstrap,
            }
        )

        r_between = weighted_corr(reference, rival, weights)
        r2_both = float(
            _two_predictor_r2(
                np.asarray([r_reference]),
                np.asarray([r_rival]),
                np.asarray([r_between]),
            )[0]
        )
        r_between_boot = weighted_corr_rows(reference_boot, rival_boot, weights)
        r2_boot = _two_predictor_r2(r_reference_boot, r_rival_boot, r_between_boot)
        unique_reference = r2_boot - r_rival_boot**2
        unique_rival = r2_boot - r_reference_boot**2
        commonality_rows.append(
            {
                "target_key": target.key,
                "reference_key": reference_key,
                "rival_key": candidate.key,
                "r2_reference_alone": r_reference**2,
                "r2_rival_alone": r_rival**2,
                "r2_both": r2_both,
                "unique_reference": r2_both - r_rival**2,
                "unique_rival": r2_both - r_reference**2,
                "shared": r_reference**2 + r_rival**2 - r2_both,
                "unique_reference_bootstrap_median": np.nanmedian(unique_reference),
                "unique_reference_q025": np.nanquantile(unique_reference, 0.025),
                "unique_reference_q975": np.nanquantile(unique_reference, 0.975),
                "unique_rival_bootstrap_median": np.nanmedian(unique_rival),
                "unique_rival_q025": np.nanquantile(unique_rival, 0.025),
                "unique_rival_q975": np.nanquantile(unique_rival, 0.975),
            }
        )

    profile_rows: list[dict[str, object]] = []
    for key, profile in {target.key: target_group, **groups}.items():
        standardized = weighted_zscore(profile, weights)
        cell_table = cells.copy()
        cell_table["standardized_beta"] = standardized
        condition_table = cell_table.groupby(
            ["task", "condition_key"], as_index=False, sort=False
        )["standardized_beta"].mean()
        for row in condition_table.itertuples(index=False):
            profile_rows.append(
                {
                    "target_key": target.key,
                    "profile_key": key,
                    "task": row.task,
                    "condition_key": row.condition_key,
                    "standardized_beta": row.standardized_beta,
                }
            )
    return {
        "correlations": correlations,
        "paired": pd.DataFrame(paired_rows),
        "commonality": pd.DataFrame(commonality_rows),
        "profiles": pd.DataFrame(profile_rows),
    }


def segment_grid(
    results: Sequence[dict[str, pd.DataFrame]],
    targets: Sequence[TargetSpec],
    *,
    inventory: pd.DataFrame | None = None,
    reference_key: str = "amy_proper",
    rival_key: str = "glasser__pir",
) -> pd.DataFrame:
    """Collect S5 segment/exclusion ranks from independently extracted targets."""
    rows: list[dict[str, object]] = []
    for target, result in zip(targets, results, strict=True):
        correlations = result["correlations"].set_index("candidate_key")
        if reference_key not in correlations.index:
            continue
        row: dict[str, object] = {
            "target_key": target.key,
            "segment": target.segment,
            "exclusion": target.exclusion,
            "r_amy": correlations.loc[reference_key, "group_r"],
            "r_amy_lo": correlations.loc[reference_key, "bootstrap_q025"],
            "r_amy_hi": correlations.loc[reference_key, "bootstrap_q975"],
            "rank_amy": int(correlations.loc[reference_key, "rank"]),
            "p_amy_first": correlations.loc[
                reference_key, "bootstrap_first_probability"
            ],
        }
        if inventory is not None:
            target_inventory = inventory.loc[inventory["target_key"].eq(target.key)]
            if not target_inventory.empty:
                row["n_vox"] = int(
                    target_inventory.groupby("hemisphere")["target_voxels"]
                    .first()
                    .sum()
                )
        if rival_key in correlations.index:
            row["r_pir"] = correlations.loc[rival_key, "group_r"]
            row["rank_pir"] = int(correlations.loc[rival_key, "rank"])
            row["p_pir_first"] = correlations.loc[
                rival_key, "bootstrap_first_probability"
            ]
        rows.append(row)
    return pd.DataFrame(rows)


def extract_target_voxels(
    *,
    subjects: Sequence[str],
    fixed_effects_root: Path,
    target: TargetSpec,
    reference: nib.Nifti1Image,
    conditions: Sequence[Condition] = CONDITIONS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract participant x condition x target-voxel arrays for Figure 7D."""
    target_mask = _load_on_reference(target.path, reference, labels=True) > 0
    ijk = np.argwhere(target_mask)
    xyz = nib.affines.apply_affine(reference.affine, ijk)
    hemispheres = np.where(xyz[:, 0] < 0, "left", "right")
    values = np.full((len(subjects), len(conditions), len(ijk)), np.nan, np.float32)
    voxel_tuple = tuple(ijk[:, axis] for axis in range(3))
    for participant, subject in enumerate(subjects):
        support = common_support(fixed_effects_root, subject, reference, conditions)
        supported = support[voxel_tuple]
        for condition_index, condition in enumerate(conditions):
            image = nib.load(str(_effect_path(fixed_effects_root, subject, condition)))
            if not same_grid(image, reference):
                raise ValueError("Fixed-effect grid changed during voxel extraction")
            row = np.asarray(image.dataobj, dtype=np.float32)[voxel_tuple]
            values[participant, condition_index] = np.where(supported, row, np.nan)
    return values, ijk, hemispheres


def extract_reference_support(
    *,
    subjects: Sequence[str],
    fixed_effects_root: Path,
    amygdala_mask: Path,
    gray_matter_support: Path,
    target: TargetSpec,
    reference: nib.Nifti1Image,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the anatomical-amygdala ROI and support used for its task profile."""
    mask = _load_on_reference(amygdala_mask, reference, labels=True) > 0
    mask &= load_gray_matter_support(gray_matter_support, reference)
    mask &= ~(_load_on_reference(target.path, reference, labels=True) > 0)
    hemispheres = _world_hemispheres(reference)
    mask &= hemispheres["left"] | hemispheres["right"]
    ijk = np.argwhere(mask)
    hemisphere = np.where(hemispheres["left"][mask], "left", "right")
    support = np.stack(
        [
            common_support(fixed_effects_root, subject, reference)[mask]
            for subject in subjects
        ]
    )
    return ijk, hemisphere, support


def normalize_blur_prediction(
    target_prediction: np.ndarray,
    target_hemispheres: np.ndarray,
    reference_prediction: np.ndarray,
    reference_hemispheres: np.ndarray,
    reference_support: np.ndarray,
) -> np.ndarray:
    """Normalize by participant voxel medians, then their mean, per hemisphere."""
    if reference_support.ndim != 2 or reference_support.shape[1] != len(
        reference_prediction
    ):
        raise ValueError("Reference support must be participant by reference voxel")
    if not np.isin(target_hemispheres, ("left", "right")).all():
        raise ValueError("Target hemispheres must be left or right")
    normalized = np.empty_like(target_prediction, dtype=float)
    for hemisphere in ("left", "right"):
        selected = reference_hemispheres == hemisphere
        support = reference_support[:, selected]
        if not support.any(axis=1).all():
            raise ValueError(f"Empty anatomical-amygdala reference in {hemisphere}")
        medians = np.nanmedian(
            np.where(support, reference_prediction[selected], np.nan), axis=1
        )
        normalizer = float(medians.mean())
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError(f"Invalid modeled reference amplitude in {hemisphere}")
        targets = target_hemispheres == hemisphere
        normalized[targets] = target_prediction[targets] / normalizer
    return normalized


def modeled_blur(
    *,
    voxel_profiles: np.ndarray,
    voxel_ijk: np.ndarray,
    voxel_hemispheres: np.ndarray,
    amygdala_profiles: dict[str, np.ndarray],
    amygdala_probability: np.ndarray,
    reference_ijk: np.ndarray,
    reference_hemispheres: np.ndarray,
    reference_support: np.ndarray,
    reference: nib.Nifti1Image,
    fwhm_mm: Sequence[float] = (2, 3, 4, 5, 6, 8),
) -> pd.DataFrame:
    """Compare observed amplitude with reference-normalized partial-volume blur.

    Each Gaussian prediction is normalized to the same anatomical-amygdala
    reference as the measured amplitude. Take the largest normalized prediction
    at each target voxel before averaging within distance shells (Figure 7D).
    """
    if amygdala_probability.shape != reference.shape[:3]:
        raise ValueError("Amygdala probability map must be on the fixed-effect grid")
    if reference_support.shape[0] != voxel_profiles.shape[0]:
        raise ValueError(
            "Target profiles and reference support must use the same cohort"
        )
    tasks = pd.DataFrame({"task": [condition.task for condition in CONDITIONS]})
    weights = task_equal_weights(tasks)
    group = np.nanmean(voxel_profiles, axis=0).T
    amplitude = np.full(group.shape[0], np.nan)
    for hemisphere in ("left", "right"):
        selected = voxel_hemispheres == hemisphere
        profiles = group[selected]
        reference_profile = np.asarray(amygdala_profiles[hemisphere], float)
        finite = np.isfinite(profiles)
        local = finite * weights[None, :]
        sums = local.sum(axis=1, keepdims=True)
        local = np.divide(local, sums, out=np.zeros_like(local), where=sums > 0)
        safe = np.where(finite, profiles, 0.0)
        mean_p = (local * safe).sum(axis=1, keepdims=True)
        mean_r = (local * reference_profile).sum(axis=1, keepdims=True)
        covariance = (local * (safe - mean_p) * (reference_profile - mean_r)).sum(
            axis=1
        )
        variance = (local * (reference_profile - mean_r) ** 2).sum(axis=1)
        amplitude[selected] = np.divide(
            covariance,
            variance,
            out=np.full(selected.sum(), np.nan),
            where=variance > 0,
        )

    zooms = np.asarray(reference.header.get_zooms()[:3], float)
    predicted = []
    for fwhm in fwhm_mm:
        sigma = float(fwhm) / np.sqrt(8.0 * np.log(2.0)) / zooms
        smoothed = gaussian_filter(amygdala_probability.astype(float), sigma=sigma)
        predicted.append(
            normalize_blur_prediction(
                smoothed[tuple(voxel_ijk.T)],
                voxel_hemispheres,
                smoothed[tuple(reference_ijk.T)],
                reference_hemispheres,
                reference_support,
            )
        )
    maximum = np.max(np.stack(predicted), axis=0)
    distance = distance_transform_edt(amygdala_probability < 0.50, sampling=zooms)
    distance = distance[tuple(voxel_ijk[:, axis] for axis in range(3))]
    shells = (
        ("<=2 mm", 0.0, 2.01),
        ("2-4 mm", 2.01, 4.01),
        ("4-6 mm", 4.01, 6.01),
        (">6 mm", 6.01, np.inf),
    )
    rows = []
    for label, low, high in shells:
        selected = (distance > low if low > 0 else distance >= 0) & (distance <= high)
        row: dict[str, object] = {
            "shell": label,
            "n_voxels": int(selected.sum()),
            "observed_amygdala_profile_amplitude": float(
                np.nanmean(amplitude[selected])
            ),
            "maximum_modeled_blur": float(np.nanmean(maximum[selected])),
        }
        for fwhm, values in zip(fwhm_mm, predicted, strict=True):
            row[f"predicted_{float(fwhm):g}mm"] = float(np.nanmean(values[selected]))
        rows.append(row)
    return pd.DataFrame(rows)
