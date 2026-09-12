# Rendering manuscript figures

`reproduce.py all` renders the quantitative panels from exported source tables.
The commands below add the image panels from the corresponding prepared NIfTI
inputs. They can be run independently of the table renderers.

## HCP anatomical and task maps

```bash
uv run python analysis/12_render_hcp_maps.py \
  --group-root <stage02-root> --background <MNI6-T1> \
  --amygdala-mask <p50-binary> --amygdala-probability <pAmy> \
  --bvr-labels <six-label-BVR> --data-dir outputs/source_data \
  --output-dir outputs/image_panels \
  --ho50 <HarvardOxford-p50-binary> --ho25 <HarvardOxford-p25-binary>
```

This writes standalone image panels for Figures 1B, 4A, 8B, S1, S4B, S5A,
S6, and S7A. Use the same prepared atlas masks and task maps as the quantitative
analyses. The optional Harvard–Oxford masks are needed for S7A.
To create them from FSL's distributed subcortical maxprob50/25 atlases, supply
those atlas files and `--harvard-oxford-labels <HarvardOxford-Subcortical.xml>`
to `analysis/09_summarize_atlas_overlap.py`. It writes both binary amygdala masks
beside the overlap table; use these files with `--ho50` and `--ho25` above.

## Vascular alignment and movie viewing

```bash
uv run python analysis/11_render_vascular_panels.py \
  --inputs vascular_inputs.json --figures 3 S2 S3 --output-dir outputs/vascular
```

Copy and fill in [the input example](../analysis/vascular_inputs.example.json).
The [input guide](../analysis/vascular_inputs.md) lists every required image,
coordinate, and preparation step. Original Dense Amygdala movie data are in
OpenNeuro; follow-up Emotion/TOF/QSM data and matching prepared vascular
derivatives are available on reasonable request to the authors. These renderers
retain the manuscript's projection thicknesses, effect ranges, and contour
sampling and write a resolved input manifest alongside the PNG/PDF files.

Figure 3 additionally needs VENAT in MNI152NLin6Asym space. The optional
`analysis/13_prepare_venat.py` implements the finalized ANTsPy template-registration
recipe; see the input guide for its separate environment and command. Its
outputs include the atlas, transformation files, and brain-mask overlap QC.

Figure 2 is a manually assembled anatomical illustration; its third-party
artwork is not redistributed here.

The image commands produce standalone panels for assembly with the quantitative
plots. Figure 5A uses the internal Emotion maps from the 0/2/4/6-mm smoothing
workflow, and Figure 6A is an anatomical target illustration. These layouts and
Figure 6D's anatomical thumbnails remain manual assembly steps. The Figure 6
table renderer writes B as paired participant lags, C as the participant slope
histogram, and D as the quantitative exclusion-sensitivity summary. Its B/C
inputs use anonymous display IDs to preserve pairing; the renderer checks the
participant distributions against the exported group statistics.
