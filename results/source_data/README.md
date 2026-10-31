# Numerical source data for the current manuscript

Files are keyed to the current figure/table numbers. `record_type` distinguishes observations, per-battery summaries, means and conditional intervals in mixed figure files. The corresponding scripts read the complete reference predictions and recalculate the values.

| File | Contents |
|---|---|
| Table1.csv | Eight full-pipeline comparison configurations, with pooled and battery-macro estimands separated |
| Fig2.csv.gz | Main query predictions, native-pipeline queries, all battery metrics, domain intervals, support observations and representative-battery IDs |
| Fig3.csv.gz | Mapping and control battery metrics, mapping-domain means, paired battery changes and intervals, and the federated factorial contrasts |
| Fig4.csv.gz | NASA and Oxford query-level observations/predictions at the reported label budgets |
| Results_2_4.csv.gz | Query predictions for complete method, uniform support adaptation and removal of running minima |

Figs1 and5 are conceptual diagrams and have no numerical source table. The full per-component predictions remain in `results/reference/predictions/`; the final ensembles are used for reported accuracy. Source Data files do not add supplementary figures or new results.
