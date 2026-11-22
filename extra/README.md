# Extra material

Nothing here is needed to follow the main page. It is what I would look at next if a number
there seemed odd.

| File | What it is |
|---|---|
| `details.md` | Start here: all eight tables in one place, the other checks, baselines, timings, and what the names in the code mean. |
| `all_methods.md` | One table per dataset with every variant that was run, including the ones that did not work, with raw and column-normalised Qini, fold spread and seconds per fold. Also the full synthetic-benchmark table. |
| `figures/qini_<table>.png` | The same as a bar chart per table. |
| `figures/batch_levels_<table>_cpfn_default.png` | Default pipeline, one dot per query batch. Left: the raw difference in outcome means between treated and control rows inside the batch's context, against the effect measured on the batch's own test rows. Right: the batch's mean prediction against that raw difference. The prediction for a batch follows the raw difference in its context closely. |
| `figures/poly_pehe_vs_n.png` | Synthetic polynomial benchmark from the paper (true effects known): PEHE against the number of training rows. |
| `results/decomposition_<table>_<method>.json` | For the default pipeline: how much of the ranking comes from the order of the batches and how much from the order inside a batch. |

Other variant names that appear in `all_methods.md`:

| Name | What it is |
|---|---|
| `cpfn_A_B4k`, `cpfn_A_B40k`, `cpfn_A_B2k` | random chunks of 4K / 40K / 2K rows, averaged |
| `cpfn_wide` | default pipeline with a wider neighbourhood per batch (`num_neighbours=12_000`) |
| `cpfn_A_B10k_strat` | random chunks stratified by treatment and outcome |
| `cpfn_A_B10k_perm` | a different column order in every chunk |
| `cpfn_A_B10k_resid` | outcome replaced by its residual from a cross-fitted outcome model |
| `cpfn_Aplus_B10k` | the three previous options together |
| `cpfn_A_B10k_stack`, `cpfn_K1_B10k_stack` | cross-fitted estimates of the weak learner and of the S, T, X, DA learners added as input columns |
| `cpfn_A_B10k_stackonly` | the model sees only those estimate columns, no original columns |
