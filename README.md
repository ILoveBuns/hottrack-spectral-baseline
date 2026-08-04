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
- Official `ID,x,y,width,height` submission writer and template-order validator.
- Deterministic moving-target regression test.

## Test

```bash
python -m unittest discover -s tests -v
```

The bounded tracker passes, initialization provenance, hardened checkpoint
contract, and current no-submit decision are documented in
[`HOTC_TRACKING_AUDIT.md`](HOTC_TRACKING_AUDIT.md).

## Current status

The official CSV bundle is integrated and the first reproducible submission,
`label_transfer_v1.csv`, scored **0.05855** on the public leaderboard
(submission ref `55143805`). As of the 2026-08-01 15:10 Beijing-time patrol,
Kaggle reports a deadline of 2026-09-09 16:00 UTC and rank 69 of 88 teams. The
public bundle contains labels and a submission template but no test imagery, so
the submitted fallback uses
cross-modality trajectory transfer where possible and modality-level robust
median boxes elsewhere. No synthetic score is presented as a competition
result. A materially independent image-tracking candidate is complete for 50/75
sequences with organizer initial boxes; the remaining 25 are ready for retry
after Drive throttling clears. It has not been submitted.

## Research path

1. Validate spectral signature matching by camera family (16/25/15 bands).
2. Add target-aware band weighting using foreground/background Fisher score.
3. Fuse the spectral tracker with a strong spatial tracker.
4. Use sequence-level splits to avoid leakage when tuning update thresholds.

## License

MIT.
