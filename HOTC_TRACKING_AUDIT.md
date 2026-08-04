# HOTC tracking audit — updated 2026-08-04

This audit records the result of a full, bounded pass over the 75 validation
sequences indexed by the organizer-data lookup exported from a public Kaggle
notebook. It is evidence about artifact availability and pipeline behavior,
not a leaderboard score.

## 2026-08-04 recovery result

- A full 75-sequence pipeline pass proved that all 26,860 target rows are
  addressable. That first output is **not submission eligible** because it used
  V1-estimated initial boxes and legacy checkpoints without a bound manifest.
- The hardened rerun uses organizer `init_rect.txt` boxes, requires every target
  frame, and binds checkpoints to the tracker and initialization source.
- 50/75 sequences (18,548 rows) now have complete KCF checkpoints using verified
  organizer initial boxes. Their first-frame coordinate conversion has zero
  maximum error against the organizer rectangles.
- All 18,548 predictions differ from V1; median center displacement is 99.45 px
  (p95 240.71 px), so this is a materially independent candidate.
- Drive throttling began after sequence 50. Incomplete sequences were rejected
  and no full submission was emitted.
- Initialization provenance is explicit: 53 locally downloaded organizer boxes,
  3 public-notebook mirrors validated against 51 exact local matches, and 19
  clearly labeled V1 cross-modality fallbacks.
- The remaining 25-sequence run is checkpoint-ready but currently has 0/25
  because both bounded Drive endpoints are throttled.

No new Kaggle submission has been made. Resume only the names in
`config/missing_official_kcf_sequences.txt`, then assemble and audit the hybrid
candidate before an account-holder-approved scored submission.

## 2026-08-01 result

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
- Legacy checkpoints without a configuration manifest are rejected; changing
  tracker or initialization source also causes a hard failure.
- Text metadata accepts only text/plain or octet-stream responses; frames accept
  only image content. HTML quota/error pages are never checkpointed.

## Resume condition

Re-run `scripts/run_csrt_from_listing.py` against the same work directory when
the public Drive files become available. Existing complete checkpoints are
reused. Evaluate and submit a candidate only after all 75 checkpoints exist,
the strict CSV checks pass, and the candidate is materially different from the
current `label_transfer_v1.csv` submission.
