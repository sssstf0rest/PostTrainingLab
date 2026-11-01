# Mac post-training course — session plan

## Goal
Audit PostTrainingLab, map a complete Mac-first interview course, create and execute lessons 00 and 01 with paired percent-format Python sources, and start the first learning section.

## Phases
1. Acquire and audit the repository, instructions, environment, and evidence — complete.
2. Verify primary references and create minimal teaching environment — complete.
3. Create course roadmap, evidence audit, and learner progress record — complete.
4. Author complete notebooks 00 and 01 with editable Python sources — complete.
5. Execute fresh-kernel CPU paths, probe MPS, review outputs and invariants — complete.
6. Report exact verification and guide the first learning section — complete (delivery below).

## Decisions
- The supplied workspace was empty and was not a Git repository; fetch the requested repository here without touching other checkouts.
- Required lessons use only local synthetic data and tiny float32 PyTorch models; no model downloads.
- Preserve historical experiment records and separate the Mac course under notebooks/mac_learning/.

## Errors encountered
| Error | Resolution |
|---|---|
| Initial git status: not a Git repository | Workspace is empty; initialize and fetch requested repository. |
| Source hash changed after concurrent lesson author edit during notebook build | Stale-source guard stopped execution before lesson 01; wait for author handoff, finalize source, rebuild that lesson and rerun. |
| README patch context mismatch | No edit applied; reapplied the single intended Python-version line with exact context. |

## Acceptance criteria
- Audited branch/commit and distinction between implemented, recorded, and planned work.
- Complete roadmap 00–13 with priorities, prerequisites, file/milestone mapping.
- Two valid, executed notebooks and matching # %% sources; offline default path.
- Minimal reproducible dependencies and honest CPU/MPS/runtime evidence.
- Learner understanding remains unassessed until demonstrated.

## Delivery
- Course files, paired sources/notebooks, minimal setup and dependency snapshot complete.
- Both notebooks validated and fully executed on this Mac using CPU and MPS; source/evidence hashes match.
- Full report in notebooks/mac_learning/VERIFICATION.md; 7 upstream utility tests passed, M2 module skipped.
- Existing tracked historical files unchanged. Begin lesson00 section1; ask whether retrieval changes model parameters.
