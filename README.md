# HOTC 2026 Spectral Signature Baseline

A small, reproducible baseline for the
[Hyperspectral Object Tracking Challenge 2026](https://www.kaggle.com/competitions/hyperspectral-object-tracking-challenge-2026).

Unlike an RGB-only tracker, it represents the initialized target by its robust
median spectrum and searches locally for the closest spectral signature. A
motion prior stabilizes ambiguous regions and a conservative template update
adapts to gradual appearance change.

## Included

- Hyperspectral `H × W × bands` tracker.
- One-pass evaluation: precision at 20 px, success AUC and mean IoU.
- Submission writer and strict validator.
- Deterministic moving-target regression test.

## Test

```bash
python -m unittest discover -s tests -v
```

## Current status

Core evaluation is runnable without competition credentials. The official data
adapter and exact CSV column mapping will be added after Kaggle OAuth and rules
acceptance; no synthetic score is presented as a competition result.

## Research path

1. Validate spectral signature matching by camera family (16/25/15 bands).
2. Add target-aware band weighting using foreground/background Fisher score.
3. Fuse the spectral tracker with a strong spatial tracker.
4. Use sequence-level splits to avoid leakage when tuning update thresholds.

## License

MIT.

