# HCP-YA analysis cohort

The HCP-YA task analysis cohort contained 311 participants and required at
least 80% accuracy on every task with accuracy scoring, together with complete
required imaging inputs. The behavioral CSV must also indicate complete task
fMRI (`3T_Full_Task_fMRI`) and have no recorded `QC_Issue`. Accuracy columns use
percentage units, so the selection threshold is 80. Subject identifiers are not included here because
access to HCP participant-level data requires registration and acceptance of
the HCP data-use terms. The stated criteria identify 316 eligible participants;
five lacked one or more imaging inputs required for the analyses, yielding the
reported cohort of 311.

`analysis/01_select_hcp_cohort.py` implements the behavioral and
imaging-completeness filters. Users with authorized HCP access can apply it to
the corresponding behavioral and imaging data.

The resting-state analysis uses the subset with four complete resting-state
runs (305 participants); `analysis/07_run_rapidtide.py select-cohort` generates
that list and an availability audit locally.
