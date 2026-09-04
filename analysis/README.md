# Analysis workflow

The analysis starts from the preprocessed imaging products specified in the
manuscript Methods. Each command lists its required arguments with `--help`.

| Stage | Script | Output |
|---|---|---|
| Cohort | `01_select_hcp_cohort.py` | HCP-YA participants meeting the behavioral and imaging criteria |
| HCP tasks | `02_fit_hcp_tasks.py` | Seven unsmoothed task GLMs, LR/RL fixed effects, and group display maps |
| Candidates | `01_prepare_candidates.py` | HCP-MMP1 and Tian S2 candidate regions, displayed composites, and gray-matter support |
| Masks | `01_prepare_masks.py` | CIT168 amygdala probability masks and functionally defined BVR targets |
| Emotion | `03_fit_internal_emotion.py` | Four-run Dense Amygdala Emotion fixed effects |
| Movies | `04_fit_movie_faces.py` | Face-presence fixed effects across 29 movie runs |
| Task summaries | `05_make_task_summaries.py` | ROI, peri-stimulus, trimming, distance-gradient, subnuclear, Figure 8, and working-memory summaries |
| Smoothing | `06_run_smoothing.py` | Task GLMs refit after smoothing the 4D BOLD series |
| Rest | `07_run_rapidtide.py` | Rapidtide delay maps, participant consensus maps, and lag summaries |
| Sources | `08_make_source_profiles.py` | 23-condition source-profile correlations and sensitivity analyses |
| Atlas overlap | `09_summarize_atlas_overlap.py` | CIT168 and Harvard-Oxford overlap with the peri-amygdalar trace |
| Figure tables | `10_export_figure_tables.py` | Analysis summaries in the schemas read by the quantitative figure scripts |

## Inputs

The HCP task workflow expects the Task3T Recommended volumetric directory
layout and cleaned files ending in
`_hp0_clean_rclean_tclean.nii.gz`. The resting-state workflow expects the HCP
S1200 minimally preprocessed directory layout. Dense Amygdala commands expect
the BIDS dataset together with its ICA-FIX-cleaned `slabpreproc` derivatives.

Atlas preparation requires the CIT168 probabilistic atlas, the volumetric
HCP-MMP1 registration-fusion probability atlas and its left/right annotation
files, the deterministic Tian S2 atlas with its labels, a brain-extracted 2-mm
MNI152NLin6Asym T1 template, and the standard HCP subcortical segmentation.
FSL 6.0.7.21 `flirt` and `fast` implement the resampling and tissue
classification operations described in the Methods.

For gray-matter support, use FSL's `MNI152_T1_2mm_brain.nii.gz` and
TemplateFlow's `tpl-MNI152NLin6Asym_res-02_atlas-HCP_dseg.nii.gz`. Supply these
files through `--mni-template-brain` and `--hcp-subcortical`; they are not
included in this repository.

Rapidtide 3.1.6 was used for the resting-state delay analysis and is invoked
as an external command by `07_run_rapidtide.py`.

## Execution order

1. Select an authorized HCP cohort with `01_select_hcp_cohort.py`, or supply an
   existing authorized subject table. Supply `--behavior`, `--hcp-data-root`,
   and `--output`; HCP accuracy columns are percentages (threshold 80), and the
   script requires `3T_Full_Task_fMRI` and an empty `QC_Issue`. Without
   `--hcp-data-root`, only behavioral/quality eligibility is checked.
2. Run `02_fit_hcp_tasks.py --group-maps` for the seven HCP tasks.
3. Run `01_prepare_candidates.py` to write `source_candidates.tsv`, the
   prepared atlas images, and `gray_matter_support.nii.gz`. The support is the
   union of FAST pGM >= 0.50 and nonzero HCP subcortical labels.
4. Run `01_prepare_masks.py`. It accepts either an existing six-label BVR image
   or the group fear-minus-shape effect map produced in step 2, and writes
   target tables for the later source and lag analyses. Tian S2 and the
   candidate-preparation outputs from step 3 also create the Supplementary
   Figure S4 target and positive-control specifications.
5. Use `05_make_task_summaries.py` for the ROI, peri-stimulus, and spatial
   summaries. Its commands can read the HCP inputs or fixed-effects tree
   directly. Mean ROI mode also writes the Figure 8 condition profile and the
   working-memory load analysis; median mode is used for source-region
   profiles. For the mean analysis, name the four required ROIs `amygdala`,
   `target`, `ffc`, and `v1`. For the Figure 4 PSTC, name the ROIs
   `cit168_amygdala`, `striate_bvr`, and `peduncular_bvr`; the latter two use
   the pAmy >= 0.50-excluded striate and peduncular masks. Its spatial
   summaries use `bvr_striate_bilateral_pamy50_excluded.nii.gz`.
6. Run `06_run_smoothing.py` for the HCP-YA and Dense Amygdala smoothing
   analyses.
7. Use `07_run_rapidtide.py select-cohort --subjects <task-subjects.tsv>
   --hcp-root <S1200-root> --output <rest-subjects.tsv>
   --audit-output <availability.tsv>` to retain participants with four complete
   resting-state runs (305 of the 311 task participants in the paper). Use that
   list for `plan` or `run`, followed by `summarize`. Pass
   `cit168_amygdala_p50.nii.gz` as the anatomical-amygdala mask. The
   `summarize --existing-consensus-only` route can reuse saved cluster consensus
   maps without the original run maps; it omits optional run-level summaries.
8. Run `08_make_source_profiles.py` with the HCP fixed-effects root, candidate
   table, and gray-matter support from step 3. The support is applied to all
   195 candidate source regions, not to the peri-amygdalar/BVR targets. Use
   `figure7_targets.tsv` for Figure 7,
   `source_targets.tsv` for the segment and exclusion analysis in Figure S5,
   and `s4_targets.tsv` for the mesencephalic-BVR analysis in Figure S4. The
   optional `--s4-positive-controls` input runs the V2-to-V1 and
   auditory-belt-to-A1 control screens alongside the Figure 7 invocation. The
   Figure 7 run also uses `--point-spread-target
   bvr_striate_peduncular_pamy50` together with the CIT168 probability
   map supplied through `--amygdala-probability`. Pass
   `cit168_amygdala_p50.nii.gz` as `--amygdala-mask`.
   Figure 7D normalizes each Gaussian blur prediction to the modeled response
   in the same amygdala reference ROI (participant voxel medians, then their
   mean, separately by hemisphere), before taking the maximum across kernels
   at each target voxel and averaging by distance shell.
9. Run `09_summarize_atlas_overlap.py` with the CIT168 and Harvard-Oxford
   amygdala masks to reproduce the Supplementary Figure S7 overlap table.
10. Run `10_export_figure_tables.py` with the completed stage directories to
    create all 26 manuscript-facing tables. Supply the Figure 7, Figure S5,
    Figure S4 target-ranking, and Figure S4 positive-control runs through their
    separate named options.
    Write them to `outputs/source_data/tables/`; `reproduce.py` reads its parent
    directory by default.

The stage-directory options for the final export are:

| Option | Stage outputs used |
|---|---|
| `--roi-dir` | Figure 1, Figure 8, Figure S1, and working-memory summaries from `05_make_task_summaries.py roi` |
| `--pstc-dir` | Figure 4 peri-stimulus summaries from `05_make_task_summaries.py pstc` |
| `--spatial-dir` | Figure 4 trimming, gradient, and subnucleus summaries from `05_make_task_summaries.py spatial` |
| `--smoothing-dir` | Figure 5 HCP-YA summary from `06_run_smoothing.py` |
| `--rapidtide-dir` | Figure 6 summaries from `07_run_rapidtide.py summarize` |
| `--figure7-source-profiles-dir` | Figure 7 run from `08_make_source_profiles.py` |
| `--s5-source-profiles-dir` | segment-by-exclusion run from `08_make_source_profiles.py` |
| `--s4-source-profiles-dir` | mesencephalic-target run from `08_make_source_profiles.py` |
| `--s4-positive-controls-dir` | Figure S4 positive controls from the Figure 7 source-profile run |
| `--atlas-overlap` | Figure S7 table from `09_summarize_atlas_overlap.py` |

The Dense Amygdala Emotion and movie models are independent of the HCP task
sequence and can be run with `03_fit_internal_emotion.py` and
`04_fit_movie_faces.py` whenever their inputs are available.

Participant-level analysis outputs contain controlled-data identifiers and
must be stored and shared under the corresponding dataset terms.

## Model summaries

Task models use unsmoothed, run-mean-scaled BOLD data, an SPM canonical HRF
with temporal derivative, cosine drift terms with a 1/128-Hz high-pass, and an
AR(1) noise model. Run estimates are combined within participant by
precision-weighted fixed effects. Smoothing analyses apply Gaussian smoothing
to each 4D BOLD run before refitting the GLM.

Source profiles use participant-, condition-, and hemisphere-specific median
effect estimates after a seven-task common-support intersection. Correlations
give equal total weight to each task and each hemisphere, and uncertainty is
estimated by participant bootstrap. The V2 and auditory-belt profiles used as
Supplementary Figure S4 control targets retain their unrestricted atlas masks;
V1, A1, and every source-screen candidate use gray-matter support.

These profile analyses rank correspondence with candidate upstream
territories. They do not establish causal source direction or quantify how
much venous signal originated in each territory. The BVR targets represent a
functionally defined peri-amygdalar signal.

Figure 1, Figure 8, and the working-memory analysis use participant-level
voxelwise mean effect estimates within the specified bilateral ROIs.
