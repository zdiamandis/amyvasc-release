# Source data

This directory records the table contracts underlying
the quantitative figure panels and reported results. The tables themselves are
generated locally from authorized inputs and are not distributed in this
repository.

`manifest.tsv` records the output filename, figure, panel, description, and data
level for each generated table.

Run `analysis/10_export_figure_tables.py` after the preceding analysis stages to
write all 31 tables to `outputs/source_data/tables/`. The figure scripts read
that directory by default. Additional tables record numerical results reported
in the manuscript text. The Figure 7 and Supplementary Figure S4 mask
inventories report source-candidate voxel counts before gray-matter restriction,
after restriction, and after target-overlap removal.

Source-correlation tables distinguish `n_cohort` from
`n_paired_cell_participants_min`, `_median`, and `_max`: the latter summarize
finite target/candidate participant pairs across condition-by-hemisphere cells.
They describe support; each ROI group mean and bootstrap draw still uses its
own available observations, as in the analysis. A cohort of 311 does not imply
311 usable observations for every candidate and cell.

The Figure 6 participant tables retain pairing through sequential `display_id`
values and omit the original dataset identifiers. They remain participant-level
derived data subject to the original data-use terms. Other exported tables are
aggregate or group-voxel summaries. No generated tables are included here.

The participant distance-slope summary reports inference across participants;
the separate Figure 4D group-voxel fit describes the displayed spatial trend.
The mean-estimator sensitivity table reports correlations and intervals only;
it does not imply a full-atlas ranking if a restricted candidate set was used.
The held-out prediction table supplies Q² results for all tasks and with Emotion
excluded. Figure 7C commonality intervals are in the exported table, although
the panel displays point estimates.

Users must obtain HCP-YA, Dense Amygdala, and atlas inputs separately and comply
with their original terms.

The MIT License applies to repository code only. HCP data used to generate the
local tables remain subject to the
[WU-Minn HCP Consortium Open Access Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).
