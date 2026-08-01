# HOTC tracking audit — 2026-08-01

This audit records the result of a full, bounded pass over the 75 validation
sequences indexed by the organizer-data lookup exported from a public Kaggle
notebook. It is evidence about artifact availability and pipeline behavior,
not a leaderboard score.

## Result

- 75/75 sequence entries were inspected.
- 12/75 sequences produced checkpoints from real downloaded JPEG frames.
- Those 12 sequences exposed 4,256/4,861 indexed target frames (87.55%).
- 63/75 sequences failed the bounded first-frame probe and remain retryable.
- `submissions/csrt_v1.csv` was deliberately **not** created because the
  checkpoint set is incomplete.
- No second Kaggle submission was made.

The successful checkpoints are:

| Sequence | Downloaded / target frames |
|---|---:|
| `nir-bee2` | 488 / 600 |
| `nir-bee3` | 690 / 690 |
| `nir-car11` | 280 / 281 |
| `nir-car3` | 98 / 98 |
| `nir-dryleaf1` | 380 / 380 |
| `nir-herbs1` | 764 / 764 |
| `nir-herbs2` | 284 / 284 |
| `nir-herbs6` | 300 / 300 |
| `nir-herbs7` | 244 / 244 |
| `nir-herbs9` | 90 / 582 |
| `vis-walker1` | 348 / 348 |
| `vis-walker2` | 290 / 290 |

## Safety properties verified

- A sequence with zero downloaded frames cannot become a checkpoint.
- A failed probe remains retryable on a later run.
- Drive requests use connect/read timeouts, validate an `image/*` response,
  and atomically rename a completed temporary file.
- A submission is emitted only when every sequence has a full-length
  checkpoint and the merged CSV passes ID-order, null, width, and height
  assertions.

## Resume condition

Re-run `scripts/run_csrt_from_listing.py` against the same work directory when
the public Drive files become available. Existing complete checkpoints are
reused. Evaluate and submit a candidate only after all 75 checkpoints exist,
the strict CSV checks pass, and the candidate is materially different from the
current `label_transfer_v1.csv` submission.

