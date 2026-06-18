# Validation Set Analysis by p_side Buckets

This report provides a detailed breakdown of the safe_lowest_price_gap model's performance on the validation set, segmented by the baseline probability (p_side) in increments of 0.1.

## Metrics Definitions
- **Active Count**: Number of samples where the model made a prediction rather than falling back to the baseline.
- **Coverage Overall**: Proportion of total samples successfully covered by the prediction.
- **Coverage Feasible**: Proportion of *theoretically feasible* samples successfully covered.
- **Covered Gap Norm Mean**: The mean normalized gap between the predicted price and the safe target (lower is better, 1.0 means no gap compression).

## Bucket Breakdown

| Bucket (p_side)   |   Total Count |   Active Count | Active Share   |   Feasible Count | Coverage Overall   | Coverage Feasible   |   Covered Gap Norm Mean |   Mean p_side |   Mean p_pred |
|:------------------|--------------:|---------------:|:---------------|-----------------:|:-------------------|:--------------------|------------------------:|--------------:|--------------:|
| [0.0, 0.1)        |             2 |              0 | 0.00%          |                2 | 100.00%            | 100.00%             |                  1      |        0.0904 |        0.0904 |
| [0.1, 0.2)        |            53 |              3 | 5.66%          |               43 | 81.13%             | 100.00%             |                  0.9912 |        0.1659 |        0.1652 |
| [0.2, 0.3)        |           318 |             69 | 21.70%         |              177 | 54.09%             | 97.18%              |                  0.912  |        0.2621 |        0.2573 |
| [0.3, 0.4)        |           977 |            200 | 20.47%         |              460 | 45.24%             | 96.09%              |                  0.9219 |        0.357  |        0.3488 |
| [0.4, 0.5)        |          1517 |            633 | 41.73%         |              837 | 52.93%             | 95.94%              |                  0.8547 |        0.4512 |        0.4285 |
| [0.5, 0.6)        |          1468 |           1012 | 68.94%         |             1049 | 66.49%             | 92.95%              |                  0.7607 |        0.5492 |        0.4982 |
| [0.6, 0.7)        |          1335 |           1167 | 87.42%         |             1132 | 76.70%             | 90.46%              |                  0.6161 |        0.6492 |        0.5382 |
| [0.7, 0.8)        |          1041 |            942 | 90.49%         |              954 | 80.88%             | 88.26%              |                  0.6256 |        0.7469 |        0.6272 |
| [0.8, 0.9)        |           576 |            528 | 91.67%         |              538 | 75.00%             | 80.30%              |                  0.6758 |        0.842  |        0.7726 |
| [0.9, 1.0)        |           145 |            132 | 91.03%         |              139 | 70.34%             | 73.38%              |                  0.6171 |        0.932  |        0.8776 |
