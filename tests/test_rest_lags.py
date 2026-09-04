"""Scientific and file-interface checks without HCP imaging or Rapidtide installed."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from amyvasc_release.rest_lags import (
    RUNS,
    Consensus,
    LagTarget,
    build_consensus,
    hcp_rest_paths,
    load_consensus,
    prepare_runs,
    save_consensus,
    select_rest_cohort,
    summarize_rapidtide,
)


class RestLagTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.shape = (4, 8, 2)
        self.affine = np.diag([2.0, 2.0, 2.0, 1.0])
        self.affine[0, 3] = -3.0
        self.reference = nib.Nifti1Image(np.zeros(self.shape, np.float32), self.affine)

    def write_image(self, path, data, affine=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(
                np.asarray(data, np.float32), self.affine if affine is None else affine
            ),
            path,
        )

    def test_select_cohort_reports_missing_motion_and_preserves_strict_planning(self):
        for subject in ("complete", "missing_motion", "missing_bold"):
            for run in RUNS:
                for path in hcp_rest_paths(self.root, subject, run):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
        hcp_rest_paths(self.root, "missing_motion", RUNS[0])[1].unlink()
        hcp_rest_paths(self.root, "missing_bold", RUNS[-1])[0].unlink()
        audit = select_rest_cohort(
            ["complete", "missing_motion", "missing_bold"], self.root
        )
        self.assertEqual(audit["included"].tolist(), [True, False, False])
        self.assertEqual(audit["n_complete_runs"].tolist(), [4, 3, 3])
        with self.assertRaises(FileNotFoundError):
            prepare_runs(audit.subject_id.tolist(), self.root, self.root / "outputs")
        self.assertEqual(
            len(prepare_runs(["complete"], self.root, self.root / "outputs")), 4
        )

    def test_consensus_uses_corrfit_and_retains_valid_zero_delay(self):
        subject = "synthetic"
        for run, value in zip(RUNS, [-2.0, 0.0, 4.0, 8.0]):
            prefix = self.root / subject / run / f"amyvasc_{subject}_{run}"
            valid = np.ones(self.shape)
            if value == 8.0:
                valid[0, 0, 0] = 0
            self.write_image(
                Path(f"{prefix}_desc-maxtimerefined_map.nii.gz"),
                np.full(self.shape, value),
            )
            self.write_image(
                Path(f"{prefix}_desc-maxcorr_map.nii.gz"), np.ones(self.shape)
            )
            self.write_image(Path(f"{prefix}_desc-corrfit_mask.nii.gz"), valid)
        consensus = build_consensus(self.root, subject)
        self.assertEqual(consensus.delay[1, 0, 0], 2.0)
        self.assertEqual(consensus.coverage[1, 0, 0], 4)
        self.assertEqual(consensus.delay[0, 0, 0], 0.0)
        self.assertEqual(consensus.coverage[0, 0, 0], 3)
        save_consensus(self.root, consensus)
        loaded = load_consensus(self.root, subject)
        self.assertEqual(loaded.runs, RUNS)
        np.testing.assert_array_equal(loaded.delay, consensus.delay)

    def test_cluster_metadata_and_consensus_only_summary_need_no_run_maps(self):
        amy = np.zeros(self.shape)
        amy[1:3, 0, :] = 1
        target = np.zeros(self.shape)
        target[[0, 3], 1:, :] = 1
        amy_path, target_path = self.root / "amy.nii.gz", self.root / "target.nii.gz"
        self.write_image(amy_path, amy)
        self.write_image(target_path, target)
        for subject, offset in (("one", 0.0), ("two", 0.2)):
            delay = np.zeros(self.shape, np.float32)
            spatial_delay = (
                0.5 + offset + 0.005 * (1 + offset) * (7 - np.indices(self.shape)[1])
            )
            delay[target > 0] = spatial_delay[target > 0]
            consensus = Consensus(
                subject,
                delay,
                delay,
                np.full(self.shape, 4),
                np.ones(self.shape),
                RUNS,
                self.reference,
            )
            save_consensus(self.root, consensus)
            info = (
                self.root
                / subject
                / "consensus"
                / f"amyvasc_{subject}_consensus_info.json"
            )
            # This is the schema of the saved manuscript cluster outputs.
            info.write_text(
                json.dumps(
                    {"subject_id": subject, "n_runs_found": 4, "runs_used": list(RUNS)}
                )
            )
        tables = summarize_rapidtide(
            rapidtide_root=self.root,
            subjects=["one", "two"],
            amygdala_mask_path=amy_path,
            targets=[LagTarget("test", "Test target", target_path)],
            output_dir=self.root / "tables",
            build_missing_consensus=False,
            include_run_reproducibility=False,
        )
        np.testing.assert_allclose(
            tables["participant_lags"]["target_minus_amy_ms"], [515, 718], atol=0.001
        )
        self.assertNotIn("run_reproducibility", tables)
        self.assertEqual(tables["consensus_qc"]["n_runs"].tolist(), [4, 4])

    def test_misaligned_consensus_companion_map_is_rejected(self):
        consensus = Consensus(
            "one",
            np.ones(self.shape),
            np.ones(self.shape),
            np.full(self.shape, 4),
            np.ones(self.shape),
            RUNS,
            self.reference,
        )
        save_consensus(self.root, consensus)
        coverage_path = (
            self.root / "one/consensus/amyvasc_one_desc-consensusCoverage_map.nii.gz"
        )
        shifted = self.affine.copy()
        shifted[0, 3] += 2
        self.write_image(coverage_path, np.full(self.shape, 4), shifted)
        with self.assertRaisesRegex(ValueError, "grid differs"):
            load_consensus(self.root, "one")

    def test_cli_select_cohort_writes_retained_subjects_and_audit(self):
        for run in RUNS:
            for path in hcp_rest_paths(self.root, "complete", run):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
        subjects = self.root / "subjects.tsv"
        subjects.write_text("subject_id\ncomplete\nincomplete\n")
        script = Path(__file__).resolve().parents[1] / "analysis/07_run_rapidtide.py"
        spec = importlib.util.spec_from_file_location("run_rapidtide_cli", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        retained, audit = self.root / "retained.tsv", self.root / "audit.tsv"
        self.assertEqual(
            module.main(
                [
                    "select-cohort",
                    "--subjects",
                    str(subjects),
                    "--hcp-root",
                    str(self.root),
                    "--output",
                    str(retained),
                    "--audit-output",
                    str(audit),
                ]
            ),
            0,
        )
        self.assertEqual(retained.read_text(), "subject_id\ncomplete\n")
        self.assertIn("incomplete", audit.read_text())


if __name__ == "__main__":
    unittest.main()
