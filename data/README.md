# Data

`SiouxFalls/` — the SiouxFalls transportation network (TNTP format), the prototype's test instance
(24 nodes, 76 directed arcs). Tiny and committed for reproducibility. `Anaheim/` likewise.
(A copy of these road networks also lives in the shared hub under `$AIER_DATA/tntp/`.)

For larger experiments over the SuiteSparse corpus, use the consolidated data hub referenced by the
`AIER_DATA` environment variable (default `~/code/aier/data`; corpus under `$AIER_DATA/suitesparse`,
which is a symlink to the once-on-disk `~/code/data`). See `~/code/aier/data/README.md`. The legacy
`GRAPH_DATA` variable is still honored where present, but `AIER_DATA` is the unified reference.
