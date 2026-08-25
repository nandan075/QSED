# Experiment Results

## Experiment 1: Edge map only vs keypoint detection
- Edge pixels: 262
- Keypoints: 12
- Runtime: 0.0000s

## Experiment 2: 3x3 vs 5x5 neighborhood
- Avg Entropy (3x3): 2.5830
- Avg Entropy (5x5): 3.8787
- Runtime: 0.0024s

## Experiment 3: Method A vs Method B
- Method A: Min 0.0000, Max 3.1699, Avg 2.8160
- Runtime: 0.0000s

## Experiment 4: Ranking overlap
- Overlap in top 50: 12/50
- Runtime: 0.0000s

## Experiment 5: Threshold sweep
- Threshold 0.05: 12 keypoints (Runtime: 0.0007s)
- Threshold 0.1: 12 keypoints (Runtime: 0.0007s)
- Threshold 0.15: 12 keypoints (Runtime: 0.0006s)
- Threshold 0.2: 12 keypoints (Runtime: 0.0006s)
- Threshold 0.25: 12 keypoints (Runtime: 0.0008s)

## Experiment 6: Alpha sweep
- Alpha 0.0: Top 5 avg combined score = 0.8654 (Runtime: 0.0000s)
- Alpha 0.25: Top 5 avg combined score = 0.8191 (Runtime: 0.0000s)
- Alpha 0.5: Top 5 avg combined score = 0.8135 (Runtime: 0.0000s)
- Alpha 0.75: Top 5 avg combined score = 0.8144 (Runtime: 0.0000s)
- Alpha 1.0: Top 5 avg combined score = 0.8235 (Runtime: 0.0000s)
