# Training Report Reference

Current two-stage reports expose:

- Top-level thresholds: `stage1_threshold`, `up_threshold`, `down_threshold`, `margin_threshold`
- Legacy alias during transition: `buy_threshold = up_threshold`
- Search outputs: `stage1_threshold_search`, `stage2_threshold_search`
- Window stats: `train_window` and `validation_window` include `stage1_class_ratio`, `stage2_selected_ratio`, `stage2_row_count`, `stage2_class_ratio`, `stage1_filter_purity`, `stage1_precision`, `stage1_recall`
- Metrics:
  - `train_metrics.stage1`
  - `train_metrics.stage2`
  - `train_metrics.end_to_end`
  - `validation_metrics.stage1`
  - `validation_metrics.stage2`
  - `validation_metrics.end_to_end`

Stage 2 metrics are three-class metrics on the Stage 1 selected subset.
End-to-end metrics are evaluated on the full validation window using the Stage 1 gate plus Stage 2 decision thresholds.
