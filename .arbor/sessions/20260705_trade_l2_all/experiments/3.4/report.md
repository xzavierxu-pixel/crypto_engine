# Experiment 3.4

**Hypothesis**: Mechanism: T5 explicit UP/DOWN path and relative liquidity-pressure features
Hypothesis: selected-only features miss opposite-token pressure and complement deviations that predict adverse selection
Observable: ablations attribute B_test gain to q quality versus fill selection and report side-specific diagnostics
Conflicts: prior path signatures improved Brier but not PnL; relative side state attacks omitted representation

**Score**: 276.83640326898956

**Insight**: Side-specific feature ablations did not beat the legacy trade/L2 feature set; the frozen selector retained legacy.

**Result**: B_dev robust score 276.8364; frozen legacy candidate; B_test 8.47 versus 42.43 anchor.
