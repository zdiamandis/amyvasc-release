# Vascular image panels: Figure 3 and Supplements S2/S3

These optional renderers start from prepared, registered images. They reproduce
the displayed anatomical comparisons, including the selected slices, projection
thicknesses, color ranges, and amygdala contours.

```bash
uv run python analysis/11_render_vascular_panels.py \
  --inputs vascular_inputs.json --figures 3 S2 S3 \
  --output-dir outputs/vascular
```

Copy `vascular_inputs.example.json` and replace its paths with your local inputs.
Relative paths are resolved from the manifest's directory. The command writes
PNG/PDF figures and a copy of the resolved input manifest. The three panels can
be rendered separately; only the fields needed by the selected figure are read.

## Inputs and preparation

The original Dense Amygdala movie acquisitions are in OpenNeuro ds006947 v1.0.1.
The follow-up Emotion, TOF and multi-echo GRE acquisitions used here are available
from the authors on reasonable request, as stated in the manuscript. Request the
matching subject templates and prepared vascular derivatives when reproducing
these image panels. Image reconstruction and subject-template registration are
upstream prerequisites; they are not performed by this renderer.

| Manifest field | Prepared image |
|---|---|
| `hcp.effect` | HCP group fear-minus-shape effect map from `02_fit_hcp_tasks.py --group-maps` |
| `hcp.amygdala` | Binary CIT168 pAmy >= 0.50 mask on that exact 2-mm grid |
| `hcp.venat` | VENAT partial-volume atlas transformed from MNI152NLin2009cAsym into MNI152NLin6Asym, at 0.5 mm |
| `hcp.anatomy` | Anatomical background; the manuscript renderer used Nilearn's `load_mni152_template(resolution=1)` |
| `emotion_effect` | Subject-specific fear-minus-shape fixed-effects effect from `03_fit_internal_emotion.py` |
| `movie_effect` | Subject-specific face-presence fixed-effects effect from `04_fit_movie_faces.py` |
| `frangi` | Registered 0.5-mm QSM-derived Frangi venogram with the preparation below |
| `amygdala` | Binary CIT168 pAmy >= 0.50 mask on the exact Emotion effect grid |
| `movie_amygdala_pseg` | Original subject-space 0.5-mm CIT168 probability segmentation used by S2 |
| `cit168_labels` | CIT168 label names in probability-volume order |
| `t2w` | Subject-specific 0.5-mm T2-weighted template |
| `tof` | Registered, brain-masked, session-averaged 0.5-mm TOF arteriogram |

All subject images must already share the same subject-template world-coordinate
space. Different voxel sizes are allowed; registration cannot be inferred from
similar NIfTI dimensions or performed by changing an affine header. Keep the
prepared files and their registration provenance together.

The manuscript TOF images were affine-registered to the subject's 0.5-mm T1
template, brain-masked, and averaged across two acquisitions. QSM reconstruction
used QSMxT 8.2.1, ROMEO unwrapping, and rapid two-step dipole inversion. QSM maps
were affine-registered to that same template using the multi-echo magnitude
image. The registered QSMs were filtered for positive ridges using six
logarithmically spaced Frangi scales from 0.5 to 2.5 mm, then gamma-corrected with
exponent 0.5. The renderer expects this completed vesselness image; it does not
apply that correction again. The original processing outputs and settings must
be obtained with the follow-up data to reproduce this upstream reconstruction.

Figure 3/S3 binary contours use the selected subject-space CIT168 segmentation:
clamp each selected amygdala probability label to [0,1], sum and clamp the union,
resample with FSL FLIRT sinc interpolation to the Emotion effect grid, clamp
again, and threshold at 0.50. The reusable functions are
`amyvasc_release.masks.build_cit168_probability`,
`resample_probability_sinc`, and `save_mask`. S2 retains its original continuous
0.5-mm CIT168 union: the renderer sums the selected probability channels, clamps
the union, and samples the 0.50 contour on the display grid. These are separate
inputs because the finalized panels used those distinct contour representations.

VENAT requires an inter-template spatial transform before rendering. Obtain the
original partial-volume atlas and TemplateFlow's 1-mm T1w templates and brain
masks for MNI152NLin6Asym and MNI152NLin2009cAsym, then run the finalized SyN recipe:

```bash
uv run --python 3.12 --script analysis/13_prepare_venat.py \
  --venat <VENAT_PartialVolume.nii.gz> \
  --fixed-template <NLin6Asym-T1w-1mm> --moving-template <NLin2009cAsym-T1w-1mm> \
  --fixed-mask <NLin6Asym-brain-mask> --moving-mask <NLin2009cAsym-brain-mask> \
  --output-dir outputs/venat_nlin6
```

The script uses an isolated ANTsPy 0.6.3 environment, leaving the core analysis
environment unchanged. It saves the spatial transforms, the 0.5-mm registered
atlas, and provenance with before/after brain-mask Dice. Use that atlas as
`hcp.venat`; the original NLin2009cAsym atlas is not interchangeable. No VENAT or
other atlas files are redistributed in the release. Re-running template
registration can introduce small optimizer variation; retain the transforms and
registered atlas with the figure inputs.

## Fixed display settings

- Neurological orientation; subject sampling z coordinates are -8.977964,
  -9.969137, and -10.46914 mm for Damy001–003; displayed coordinates are rounded
  to 0.5 mm. The HCP row is z = -12 mm in MNI space.
- Figure 3: HCP single-slice activation with a 3-mm VENAT MIP; individual
  single-slice activation with a 5-mm Frangi MIP. HCP activation maximum 1,
  internal effect maximum 3, VENAT maximum 0.4, Frangi maximum 0.25.
- S2: the same subject coordinates and 5-mm Frangi MIP; movie effect maximum 0.8.
- S3: T2w single slices and 8-mm TOF/Frangi MIPs in an expanded field of view.
  TOF opacity uses the 94.5th–99.7th percentiles of positive projected values;
  venous opacity uses vesselness 0.025–0.25. These are visualization parameters.

The numeric effects, movie GLMs, and vascular reconstruction are not recomputed
by this rendering stage. The manuscript's third-party Figure 2 anatomical
artwork remains a separate manually assembled illustration.
