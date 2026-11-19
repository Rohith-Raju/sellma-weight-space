# Single-question edit experiment (stopped)

Stopped at 2026-11-19T06:47Z at the owner request, in favour of the group edit (one change trained on many
wrong answers at once; see experiments/group/).

Each folder under qwen-9b-full/edits and gemma-4b-full/edits is one rank-1 edit that tried to fix
one wrong dev question: search.json (sizes tried), the edit and its random control (.json and
.safetensors), and full run folders for every measurement. Edits cut off mid-way have incomplete
run folders (no finished_at in run.json): do not read them as results.

logs/ holds the terminal output of the experiment and of the overnight chain.
The temporary edited model files (work/) were deleted; any edit can be rebuilt from its saved .safetensors.
