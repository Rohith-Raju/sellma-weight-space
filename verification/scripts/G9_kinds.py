"""G9 independent check: five kinds of question, transitions, option-A pick rates, per-position table.

Reads raw per-question files only:
  results/experiments/analysis/rotation_choices.json   (per question, per position of the right answer:
                                                        chosen ORIGINAL letter and chosen POSITION)
  results/experiments/analysis/rotation_proof.json     (dev 900, ok per position)
  results/experiments/group/gemma-4b-full/split.json   (hidden papers)
  results/runs/*qwen-9b-full_dev/answers.jsonl         (Qwen original-order answers)
Run: cd <repo> && python3 verification/scripts/G9_kinds.py
"""
import os as _os
# Repository root (two folders up from this script) and the saved results: gpu_results/ when present
# (full private copy), otherwise results/ (the published copy). SELMA_ROOT overrides the root.
_REPO = _os.environ.get("SELMA_ROOT") or _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_RES = _os.path.join(_REPO, "gpu_results") if _os.path.isdir(_os.path.join(_REPO, "gpu_results")) else _os.path.join(_REPO, "results")
import glob
import json
from collections import Counter

R = "results/"
RC = json.load(open(R + "experiments/analysis/rotation_choices.json"))
RP = json.load(open(R + "experiments/analysis/rotation_proof.json"))
GS = json.load(open(R + "experiments/group/gemma-4b-full/split.json"))
P = "ABCD"


def right(rec, pos):
    return rec["orders"][pos]["chosen"] == rec["answer"]


def kind(rec):
    r = {p: right(rec, p) for p in P}
    n = sum(r.values())
    if n == 4:
        return "knows"
    if not r["A"] and r["B"] and r["C"] and r["D"]:
        return "habit_A"
    if n > 0:
        return "unsure"
    if len({rec["orders"][p]["chosen"] for p in P}) == 1:
        return "wrong_belief"
    if len({rec["orders"][p]["chosen_position"] for p in P}) == 1:
        return "position_guess"
    return "no_idea"


# sanity: check that chosen_position is consistent with a cyclic rotation of the options
def check_rotation(rec):
    ans = rec["answer"]
    ci = P.index(ans)
    for pos, o in rec["orders"].items():
        shift = (ci - P.index(pos)) % 4  # displayed position i holds original letter P[(i+shift)%4]
        orig_at = {P[i]: P[(i + shift) % 4] for i in range(4)}
        if orig_at[o["chosen_position"]] != o["chosen"]:
            return False
        if max(o["probs"], key=o["probs"].get) != o["chosen"]:
            return False
    return True


U, E = RC["unedited"], RC["editor3"]
bad = [q for q in U if not (check_rotation(U[q]) and check_rotation(E[q]))]
print("rotation consistency failures:", len(bad))

dev = sorted(q for q in U if U[q]["split"] == "dev")
test = sorted(q for q in U if U[q]["split"] == "test")
hid_dates = set(GS["dates"]["hidden"])
hidden = [q for q in dev if q.split("/")[0] in hid_dates]
print("sizes dev/hidden/test:", len(dev), len(hidden), len(test), "hidden papers:", len(hid_dates))
print("dev sections:", Counter(U[q]["section"] for q in dev))

KINDS = ["knows", "habit_A", "unsure", "wrong_belief", "position_guess", "no_idea"]
for name, ids in (("dev", dev), ("hidden", hidden), ("test", test)):
    ku = {q: kind(U[q]) for q in ids}
    ke = {q: kind(E[q]) for q in ids}
    cu, ce = Counter(ku.values()), Counter(ke.values())
    print(f"\n== {name} ({len(ids)})")
    for k in KINDS:
        print(f"  {k:15s} unedited {cu[k]:4d}  edited {ce[k]:4d}")
    T = Counter((ku[q], ke[q]) for q in ids if ku[q] != ke[q])
    print("  transitions:", dict(T))
    never_u = [q for q in ids if ku[q] in ("wrong_belief", "position_guess", "no_idea")]
    print("  never right (unedited):", len(never_u), " of them knows after edit:",
          sum(ke[q] == "knows" for q in never_u), " right at least once after:",
          sum(ke[q] not in ("wrong_belief", "position_guess", "no_idea") for q in never_u),
          " kinds after:", dict(Counter(ke[q] for q in never_u)))
    never_e = sum(ke[q] in ("wrong_belief", "position_guess", "no_idea") for q in ids)
    print("  never right (edited):", never_e)
    # confidence of wrong beliefs / prob on right for never-right
    wb = [q for q in ids if ku[q] == "wrong_belief"]
    if wb:
        pc = [U[q]["orders"][p]["probs"][U[q]["orders"][p]["chosen"]] for q in wb for p in P]
        print(f"  wrong_belief mean prob on chosen option: {sum(pc)/len(pc):.4f}")
    if never_u:
        pr = [U[q]["orders"][p]["probs"][U[q]["answer"]] for q in never_u for p in P]
        print(f"  never-right mean prob on right option (all 4 orders): {100*sum(pr)/len(pr):.2f}%")
        # also original order only
        pr0 = [U[q]["orders"][U[q]["answer"]]["probs"][U[q]["answer"]] for q in never_u]
        print(f"  never-right mean prob on right option (original order): {100*sum(pr0)/len(pr0):.2f}%")
    hab = [q for q in ids if ku[q] == "habit_A"]
    fixed = [q for q in hab if right(E[q], "A")]
    other = [q for q in hab if not right(E[q], "A")]
    print(f"  habit_A {len(hab)}: edited right at A {len(fixed)}; other {len(other)}; "
          f"other still habit_A {sum(ke[q]=='habit_A' for q in other)}; other kinds after {dict(Counter(ke[q] for q in other))}; "
          f"other: same chosen at A as before {sum(E[q]['orders']['A']['chosen']==U[q]['orders']['A']['chosen'] for q in other)}")
    still = [q for q in ids if not right(E[q], "A")]
    same = sum(E[q]["orders"]["A"]["chosen"] == U[q]["orders"]["A"]["chosen"] for q in still)
    print(f"  still wrong at A after edit: {len(still)}; same wrong ORIGINAL option as unedited at A: {same}")
    # position table
    for nm, D in (("unedited", U), ("editor3", E)):
        c = {p: sum(right(D[q], p) for q in ids) for p in P}
        tot = sum(c.values())
        print(f"  {nm} right by position: " + "  ".join(f"{p} {c[p]} ({100*c[p]/len(ids):.1f}%)" for p in P)
              + f" | total {tot}/{4*len(ids)} ({100*tot/(4*len(ids)):.1f}%) | wrong at A {len(ids)-c['A']}"
              + f" wrong at B,C,D {[len(ids)-c[p] for p in 'BCD']}")
    g = sum((not right(U[q], "A")) and right(E[q], "A") for q in ids)
    l = sum(right(U[q], "A") and not right(E[q], "A") for q in ids)
    print(f"  right at A: fixed {g} broken {l}")
    onlyA_u = sum(ku[q] == "habit_A" for q in ids); onlyA_e = sum(ke[q] == "habit_A" for q in ids)
    print(f"  wrong only at A: {onlyA_u} -> {onlyA_e}")
    # option-A pick rates (positions)
    for nm, D in (("unedited", U), ("editor3", E)):
        pick_when_right = sum(D[q]["orders"]["A"]["chosen_position"] == "A" for q in ids)
        tries_wrong = [(q, p) for q in ids for p in "BCD"]
        pick_when_wrong = sum(D[q]["orders"][p]["chosen_position"] == "A" for q, p in tries_wrong)
        picks = pick_when_right + pick_when_wrong
        print(f"  {nm}: A right -> picks A {100*pick_when_right/len(ids):.1f}% ({pick_when_right}/{len(ids)}); "
              f"A wrong -> picks A {100*pick_when_wrong/len(tries_wrong):.1f}% ({pick_when_wrong}/{len(tries_wrong)}); "
              f"picks of A right {100*pick_when_right/picks:.1f}% ({pick_when_right}/{picks})")

# rotation_proof cross-check of the per-position table (dev 900)
print("\n== rotation_proof.json (dev)")
for k in ("unedited_1", "unedited_2", "unedited_after", "editor3", "random"):
    D = RP[k]
    c = {p: sum(D[q]["ok"][p] for q in D) for p in P}
    print(f"  {k:15s} n={len(D)} " + " ".join(f"{p} {c[p]}" for p in P) + f" total {sum(c.values())}")
same = all(RP["unedited_1"][q]["ok"] == RP["unedited_2"][q]["ok"] == RP["unedited_after"][q]["ok"] for q in RP["unedited_1"])
print("  unedited_1 == unedited_2 == unedited_after on all 3,600 answers:", same)
rc_same = all(RP["unedited_1"][q]["ok"][p] == right(U[q], p) for q in RP["unedited_1"] for p in P)
rc_same_e = all(RP["editor3"][q]["ok"][p] == right(E[q], p) for q in RP["editor3"] for p in P)
print("  rotation_choices agrees with rotation_proof (unedited, editor3):", rc_same, rc_same_e)
for k in ("editor3", "random"):
    g = sum((not RP["unedited_1"][q]["ok"]["A"]) and RP[k][q]["ok"]["A"] for q in RP[k])
    l = sum(RP["unedited_1"][q]["ok"]["A"] and not RP[k][q]["ok"]["A"] for q in RP[k])
    print(f"  {k} vs unedited at A: gained {g} lost {l}")

# Qwen on the 109 never-right questions (original order)
qf = [f for f in glob.glob(R + "runs/*qwen-9b-full_dev/answers.jsonl")]
print("\nqwen dev files:", qf)
qa = {}
for l in open(qf[0]):
    r = json.loads(l)
    qa[r["question_id"]] = r["is_correct"]
never = [q for q in dev if kind(U[q]) in ("wrong_belief", "position_guess", "no_idea")]
nq = sum(qa[q] for q in never)
print(f"Qwen right on {nq} of {len(never)} never-right questions ({100*nq/len(never):.1f}%)")
