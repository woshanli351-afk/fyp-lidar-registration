# Working agreement

Read README.md, PROJECT_CONTEXT.md, NEXT_STEPS.md and CHANGELOG.md before work.
This is the LiDAR/CVTNet/GICP companion to James-5757/fyp-place-recognition.

- Preserve attribution and frozen research parameters and published results.
- Keep GT out of runtime candidate selection, registration and online decisions.
- Transform convention: candidate is source, query is target; T_query_from_candidate.
- Keep raw datasets, model weights, credentials and full query-level outputs outside Git.
- Make data locations configurable; do not add personal/server absolute paths.
- Run relevant tests and CLI checks. State clearly when real data/GPU validation is unavailable.
- Update project context, next steps and changelog when the project state changes.
- User requests ongoing synchronization: after completing a requested change, test,
  commit the task's files and push to the configured origin. Do not commit unrelated
  edits, force-push, or modify the teammate's repository. Report commit and push status.
- Stop publication if a credential or unresolved third-party redistribution issue is found.
  Complete unaffected local work and explain the specific blocker.
