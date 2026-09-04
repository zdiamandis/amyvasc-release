# Source data

This directory records the aggregate and group-level table contracts underlying
the quantitative figure panels and reported results. The tables themselves are
generated locally from authorized inputs and are not distributed in this
repository.

`manifest.tsv` records the output filename, figure, panel, description, and data
level for each generated table.

Run `analysis/10_export_figure_tables.py` after the preceding analysis stages to
write all 26 tables to `outputs/source_data/tables/`. The figure scripts read
that directory by default. Additional tables record numerical results reported
in the manuscript text. The Figure 7 and Supplementary Figure S4 mask
inventories report source-candidate voxel counts before gray-matter restriction,
after restriction, and after target-overlap removal.

Users must obtain HCP-YA, Dense Amygdala, and atlas inputs separately and comply
with their original terms.

The MIT License applies to repository code only. HCP data used to generate the
local tables remain subject to the
[WU-Minn HCP Consortium Open Access Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms).
