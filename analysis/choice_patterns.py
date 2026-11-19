"""Sort every question by what Gemma picks across the four orders (reads rotation_choices.json, no model).

Categories, per question, for unedited Gemma and for Gemma with editor 3's weight edit:
  knows            right option picked in all four orders
  habit_A          right option at B, C and D, something else when the right answer is at A
  unsure           right in 1 to 3 orders, but not the habit_A pattern
  wrong_belief     never right, and the SAME wrong option (same text) picked in all four orders
  position_guess   never right, and the SAME letter picked in all four orders (whatever text is there)
  no_idea          never right, choice changes with no fixed text or letter
Prints the counts, the before -> after transitions, and what the edited model picks when it is still
wrong at A. Sets: dev (900, 23 older papers), hidden (the 8 older papers no edit was built on), test (120).
"""
import json
from collections import Counter
from _paths import RESULTS

R = json.load(open(RESULTS / "experiments/analysis/rotation_choices.json"))
hidden_papers = set(json.load(open(RESULTS / "experiments/group/gemma-4b-full/split.json"))["dates"]["hidden"])
CATS = ["knows", "habit_A", "unsure", "wrong_belief", "position_guess", "no_idea"]


def category(r):
    o, ans = r["orders"], r["answer"]
    right = {L: o[L]["chosen"] == ans for L in "ABCD"}
    k = sum(right.values())
    if k == 4:
        return "knows"
    if not right["A"] and right["B"] and right["C"] and right["D"]:
        return "habit_A"
    if k > 0:
        return "unsure"
    if len({o[L]["chosen"] for L in "ABCD"}) == 1:
        return "wrong_belief"
    if len({o[L]["chosen_position"] for L in "ABCD"}) == 1:
        return "position_guess"
    return "no_idea"


def paper(qid):
    return qid.split("/")[0]


def report(name, ids):
    print(f"\n=== {name}: {len(ids)} questions ===")
    cu = {i: category(R["unedited"][i]) for i in ids}
    ce = {i: category(R["editor3"][i]) for i in ids}
    cnt_u, cnt_e = Counter(cu.values()), Counter(ce.values())
    print(f"{'category':16s} {'unedited':>9s} {'edited':>7s}")
    for c in CATS:
        print(f"{c:16s} {cnt_u[c]:9d} {cnt_e[c]:7d}")
    print("\ntransitions (unedited -> edited), non-zero only:")
    T = Counter((cu[i], ce[i]) for i in ids)
    for (a, b), n in sorted(T.items(), key=lambda x: (CATS.index(x[0][0]), CATS.index(x[0][1]))):
        if a != b:
            print(f"  {a:15s} -> {b:15s} {n}")
    # Confidence on never-right questions (unedited): probability on the chosen option, mean over orders.
    for c in ("wrong_belief", "position_guess", "no_idea"):
        sel = [i for i in ids if cu[i] == c]
        if sel:
            pch = [R["unedited"][i]["orders"][L]["probs"][R["unedited"][i]["orders"][L]["chosen"]] for i in sel for L in "ABCD"]
            pok = [R["unedited"][i]["orders"][L]["probs"][R["unedited"][i]["answer"]] for i in sel for L in "ABCD"]
            print(f"  unedited {c}: mean prob on chosen option {sum(pch)/len(pch):.3f}, on right option {sum(pok)/len(pok):.3f}")
    # Still wrong at A after the edit: same wrong option as before, or a new one?
    still = [i for i in ids if R["editor3"][i]["orders"]["A"]["chosen"] != R["editor3"][i]["answer"]]
    same = sum(R["editor3"][i]["orders"]["A"]["chosen"] == R["unedited"][i]["orders"]["A"]["chosen"] for i in still)
    print(f"\nstill wrong at A after the edit: {len(still)}; same wrong option as unedited at A: {same}; "
          f"different wrong option: {len(still) - same}")
    hab = [i for i in ids if cu[i] == "habit_A"]
    fixed = sum(R["editor3"][i]["orders"]["A"]["chosen"] == R["editor3"][i]["answer"] for i in hab)
    print(f"habit_A questions (unedited): {len(hab)}; edited picks the right option at A: {fixed}")


dev = [i for i, r in R["unedited"].items() if r["split"] == "dev"]
test = [i for i, r in R["unedited"].items() if r["split"] == "test"]
report("dev (23 older papers)", dev)
if hidden_papers:
    report("hidden (8 older papers no edit was built on)", [i for i in dev if paper(i) in hidden_papers])
report("test (3 newest papers)", test)
