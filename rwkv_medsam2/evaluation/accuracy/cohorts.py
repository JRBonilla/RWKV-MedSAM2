"""Audited reporting membership, independent of prompts and model inference.

Absence of a detected overlap is not proof of unseen patients or of absence from
external models' pretraining. Unannotated historical runs never default to main.
"""

from collections import Counter
import json
from pathlib import Path

from ..shared.data import digest, file_hash, jsonl

GROUPS = ("main", "supplementary", "unclassified")
POLICY = "all_cases_categorized_actual_training_overlap_v2"
LIMITATION = (
    "Main means no overlap identified by the local audit, not certified unseen patients. "
    "External baseline pretraining and complete teacher exposure are not established."
)
REPORT_FIELDS = ("reporting", "global_source_id", "task_label_provenance")


def reporting_group(case):
    """
    Return the validated audit group, or unclassified when no audit is present.

    Args:
        case (dict): Evaluation case metadata, including input paths and sequence settings.

    Returns:
        str: Reporting group for the case.
    """
    if "reporting" not in case:
        return "unclassified"

    validate_reporting(case)
    return case["reporting"]["group"]


def validate_reporting(case):
    """
    Validate reporting membership against overlap decisions and audit provenance.

    Args:
        case (dict): Evaluation case metadata, including input paths and sequence settings.

    Returns:
        None.
    """
    r = case.get("reporting", {})
    category = r.get("training_overlap")

    if category not in (
        "none_identified",
        "exact_image_or_scan",
        "same_scan_or_video",
        "same_patient",
    ):
        raise ValueError("Unknown training overlap category")

    expected = (
        "supplementary"
        if category != "none_identified" or r.get("validation_overlap")
        else "main"
    )
    if r.get("group") != expected:
        raise ValueError("Inconsistent audited reporting group")
    if (
        r.get("policy") != POLICY
        or not r.get("audit_sha256")
        or not r.get("limitations")
    ):
        raise ValueError("Missing reporting audit provenance")
    if not isinstance(r.get("reasons"), list) or (
        r["group"] == "supplementary" and not r["reasons"]
    ):
        raise ValueError("Supplementary cases require overlap reasons")


def coverage(cases):
    """
    Count targets, conditions, and exposure categories by reporting group.

    Args:
        cases (list[dict]): Frozen evaluation cases.

    Returns:
        dict: Coverage statistics keyed by reporting group.
    """
    result = {}
    for group in GROUPS:
        selected = [c for c in cases if reporting_group(c) == group]
        result[group] = dict(
            targets=len(selected),
            eligible_targets=sum(c.get("eligible", False) for c in selected),
            expected_conditions=sum(
                len(c.get("modes", [])) for c in selected if c.get("eligible", False)
            ),
            datasets=sorted({c["dataset"] for c in selected}),
            exposure_counts=dict(
                Counter(
                    c.get("reporting", {}).get("exposure", "unclassified")
                    for c in selected
                )
            ),
            source_identity_unresolved_targets=sum(
                c["source_identity_status"] != "verified" for c in selected
            ),
        )

    return result


def evaluation_tasks(metadata):
    """
    Resolve evaluation task labels from metadata or reviewed dataset repairs.

    Args:
        metadata (dict): Dataset metadata containing task and source annotations.

    Returns:
        tuple[list[str], str]: Task identifiers and the provenance of their assignment.
    """
    if metadata.get("tasks"):
        return metadata["tasks"], "original_test_metadata"

    fallback = {
        "Neonatal": "brain_regions",
        "NCI": "prostate",
        "ROSE-1": "vessel",
        "WBC": "cell_structures",
        "WMH": "white_matter_hyperintensity",
    }
    task = fallback.get(metadata["dataset"])
    if metadata["dataset"] == "MSD" and metadata["subdataset"] == "Heart":
        task = "cardiac_substructures"
    if not task:
        raise ValueError(
            f"Unresolved test task label: {metadata['dataset']}/{metadata.get('subdataset')}"
        )

    return [
        task
    ], "evaluation_only_task_label_repair_20261008; historical_training_unchanged"


class AuditedCohort:
    """
    Bind overlap decisions to the exact complete inventory used in the audit.
    """

    def __init__(self, inventory, audit_dir, categories_dir=None):
        """
        Load and verify overlap audits for the exact complete inventory.

        Args:
            inventory (str | pathlib.Path): Path to the complete legacy inventory JSONL file.
            audit_dir (str | pathlib.Path): Directory containing the completed overlap audit and
                its checks.
            categories_dir (str | pathlib.Path | None): Optional directory containing reviewed
                exposure categories.

        Returns:
            None.
        """
        audit_dir = Path(audit_dir)
        files = [
            "report-manifest.json",
            "report-checks.json",
            "assessment.json",
            "all_reserved_split_overlap.proposed-exclusions.jsonl",
            "verified_local_exposure_minimum.proposed-exclusions.jsonl",
            "remaining-exact-test-reuse.json",
        ]
        self.hashes = {name: file_hash(audit_dir / name) for name in files}
        manifest = json.loads((audit_dir / "report-manifest.json").read_text())
        inventory_hashes = [
            r["sha256"]
            for r in manifest
            if Path(r["path"]).name == Path(inventory).name
        ]

        if inventory_hashes != [file_hash(inventory)]:
            raise ValueError("Inventory does not match the completed overlap audit")

        checks = json.loads((audit_dir / "report-checks.json").read_text())
        if checks.get("status") != "passed" or not checks.get(
            "training_plan_hashes_match"
        ):
            raise ValueError("Overlap audit checks have not passed")

        self.inventory = {}
        for row in jsonl(inventory):
            if row["legacy_index"] in self.inventory:
                raise ValueError("Duplicate legacy index")

            self.inventory[row["legacy_index"]] = row["metadata"]

        self.reserved = self._load(audit_dir / files[3])
        self.actual = self._load(audit_dir / files[4])
        assessment = json.loads((audit_dir / "assessment.json").read_text())

        if (
            len(self.inventory) != assessment["original"]["targets"]
            or not self.actual.keys() <= self.reserved.keys()
        ):
            raise ValueError("Inconsistent audited population")
        for name, mapping in [
            ("all_reserved_split_overlap", self.reserved),
            ("verified_local_exposure_minimum", self.actual),
        ]:
            if len(mapping) != assessment["scenarios"][name]["removed"]["targets"]:
                raise ValueError("Overlap count differs from completed assessment")

        # Connect all targets/views of a source, including cross-dataset exact scan aliases.
        self.parent = {}

        def root(x):
            """
            Find a source alias's canonical parent with path compression.

            Args:
                x (str): Source key whose canonical alias parent is requested.

            Returns:
                str: Canonical source identifier in the alias graph.
            """
            self.parent.setdefault(x, x)
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]

            return x

        self.source_key = {}
        for index, m in self.inventory.items():
            sd = (
                m.get("subdataset", "")
                if m["dataset"]
                in {"QUBIQ2021", "CHAOS", "WBC", "WMH", "ROSE-1", "NCI", "MSD"}
                else ""
            )
            key = digest(
                [
                    m["dataset"],
                    sd,
                    str(
                        m.get("source_identifier")
                        or m.get("identifier")
                        or m["sequence"][0][0]
                    ),
                ]
            )
            self.source_key[index] = key
            root(key)

        aliases = json.loads((audit_dir / files[5]).read_text())
        for group in aliases:
            keys = {
                self.source_key[i] for member in group for i in member["legacy_indices"]
            }
            roots = sorted({root(k) for k in keys})
            for k in roots[1:]:
                self.parent[k] = roots[0]

        # Only override cross-dataset alias clusters; preserve reviewed within-dataset source IDs elsewhere.
        self.aliases = {
            i: "exact_scan_alias:" + root(self.source_key[i])
            for group in aliases
            for member in group
            for i in member["legacy_indices"]
        }
        aliased_roots = set(self.aliases.values())
        self.aliases = {
            i: "exact_scan_alias:" + root(key)
            for i, key in self.source_key.items()
            if "exact_scan_alias:" + root(key) in aliased_roots
        }
        categories_dir = Path(
            categories_dir or audit_dir.parent / "overlap_categories_20261008"
        )
        self.categories = {
            r["legacy_index"]: r
            for r in jsonl(categories_dir / "target-categories.jsonl")
        }

        if set(self.categories) != set(self.actual):
            raise ValueError(
                "Training categories differ from verified local exposure inventory"
            )
        # Confirm that the old union contains no validation-only targets in this audit.
        if any(
            r["confirmed_actual_training_targets"]
            != r["confirmed_local_exposure_union"]
            for r in assessment["dataset_rows"]
        ):
            raise ValueError(
                "Training-only and validation union changed; re-audit category membership"
            )
        for name in ("target-categories.jsonl", "summary.json"):
            self.hashes["categories/" + name] = file_hash(categories_dir / name)

        self.validation = set()
        for row in json.loads((audit_dir / "identity-evidence.json").read_text()):
            if any("val" in r["actual_splits"] for r in row["matches"]):
                self.validation.update(row["test"]["legacy_indices"])
        for name, key in (
            ("pixel-matches.json", "legacy_indices"),
            ("combined-volume-matches.json", "eligible_indices"),
        ):
            for row in json.loads((audit_dir / name).read_text()):
                if any("val" in r["actual_splits"] for _, r in row["train"]):
                    for _, r in row["test"]:
                        self.validation.update(r[key])

        exposure = json.loads((audit_dir / "raw-actual-exposure.json").read_text())
        for row in jsonl(audit_dir / "normalized-verified.jsonl"):
            if row["full_voxel_equal_after_lower_clipping"] and "val" in exposure.get(
                row["train"]["path"], []
            ):
                for r in row["test_records"]:
                    self.validation.update(r["eligible_indices"])
        for name in (
            "identity-evidence.json",
            "pixel-matches.json",
            "combined-volume-matches.json",
            "raw-actual-exposure.json",
            "normalized-verified.jsonl",
        ):
            self.hashes[name] = file_hash(audit_dir / name)

        expected_val = sum(
            r["confirmed_actual_validation_targets"] for r in assessment["dataset_rows"]
        )
        if len(self.validation) != expected_val:
            raise ValueError("Validation-only flag accounting differs from audit")

        self.audit_id = digest(self.hashes)

    def _load(self, path):
        """
        Load proposed exclusions indexed by legacy inventory position.

        Args:
            path (str | pathlib.Path): JSONL file containing audited exclusion decisions.

        Returns:
            dict: Legacy indices mapped to audited exclusion records.
        """
        result = {}
        for r in jsonl(path):
            i = r["legacy_index"]
            if (
                i in result
                or i not in self.inventory
                or self.inventory[i]["dataset"] != r["dataset"]
                or not r["reasons"]
            ):
                raise ValueError("Invalid or duplicated audit membership")

            result[i] = r

        return result

    def annotate(self, case):
        """
        Attach overlap, source-alias, and category annotations to a case.

        Args:
            case (dict): Evaluation case metadata, including input paths and sequence settings.

        Returns:
            dict: Annotated copy of the case.
        """
        if "unit_id" in case or "file_sha256" in case:
            raise ValueError(
                "Annotate reviewed inputs before freezing; never relabel frozen runs"
            )

        i = case.get("legacy_index")
        if i not in self.inventory:
            raise ValueError("Case missing from audited inventory")

        m = self.inventory[i]
        tasks, provenance = evaluation_tasks(m)
        if (
            case["dataset"] != m["dataset"]
            or case["task"] not in tasks
            or [case["image"], case["mask"]] != list(m["sequence"][0])
        ):
            raise ValueError("Case paths/dataset/task differ from audited inventory")

        category = self.categories.get(i, {}).get("category", "none_identified")
        evidence = self.categories.get(i, {}).get("evidence", [])
        supplementary = category != "none_identified" or i in self.validation
        out = dict(
            case,
            reporting=dict(
                policy=POLICY,
                group="supplementary" if supplementary else "main",
                exposure=(
                    "verified_local_exposure"
                    if supplementary
                    else "no_identified_actual_local_overlap"
                ),
                training_overlap=category,
                validation_overlap=i in self.validation,
                reserved_pool_overlap=i in self.reserved,
                reserved_pool_only=i in self.reserved
                and i not in self.actual
                and i not in self.validation,
                source_history_uncertain=case["source_identity_status"] != "verified",
                altered_scan_copy=any(
                    e["subtype"].startswith("same_scan_")
                    and ("clipped" in e["subtype"] or "flipped" in e["subtype"])
                    for e in evidence
                ),
                evidence_subtypes=sorted({e["subtype"] for e in evidence}),
                reasons=self.reserved.get(i, {}).get("reasons", []),
                audit_sha256=self.audit_id,
                limitations=LIMITATION,
            ),
            task_label_provenance=provenance,
        )

        if i in self.aliases:
            out["global_source_id"] = self.aliases[i]

        validate_reporting(out)
        return out


def report_views(case):
    """
    Overlapping reporting views, not additional model predictions.

    Args:
        case (dict): Evaluation case metadata, including input paths and sequence settings.

    Returns:
        list[str]: Reporting views to which the case belongs.
    """
    views = ["all_data"]
    if "reporting" not in case:
        return views + ["unclassified"]

    validate_reporting(case)
    r = case["reporting"]
    views += [r["group"], "training_" + r["training_overlap"]]

    if r["validation_overlap"]:
        views.append("validation_overlap")
    if r["reserved_pool_only"]:
        views.append("reserved_pool_only")
    if r["source_history_uncertain"]:
        views.append("source_history_uncertain")
    if r["altered_scan_copy"]:
        views.append("altered_scan_copy")

    return views
