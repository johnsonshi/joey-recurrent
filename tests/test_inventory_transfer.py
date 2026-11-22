"""In-memory checks; no study inputs, labels, receipts, or predictions are written."""

from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import unicodedata
import unittest


path = Path(__file__).resolve().parents[1] / "benchmarks" / "inventory_transfer.py"
spec = importlib.util.spec_from_file_location("inventory_transfer", path)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

VERIFY, LEVEL, CHOICE = "stock-eligibility", "stock-level", "stock-choice"
BINARY = ["Yes", "No"]
LEVELS = ["Empty (0 units)", "Low (1 to 3 units)", "Medium (4 to 6 units)", "High (7 or more units)"]
DEPOTS = ["Aster", "Birch", "Kelm", "Rill"]
TIE = "No unique depot"
MENUS = {VERIFY: BINARY, LEVEL: LEVELS, CHOICE: [*DEPOTS, TIE]}

# Hand ledger with different names, interleaved facts, all operations, and
# void quantities that could not be funded. It closes Ash/Fir/Pine/Yew = 5/5/4/3.
HAND = """Warehouse inventory ledger (units; records unordered)
R-00000BA1 | POSTED | transfer 3 from Pine to Fir
Opening Ash: 4
R-00000BA2 | VOID | remove 31 from Ash
Focus depot: Ash
Opening Fir: 0
R-00000BA3 | POSTED | add 2 to Fir
R-00000BA4 | POSTED | remove 2 from Yew
Opening Pine: 7
R-00000BA5 | VOID | transfer 20 from Fir to Pine
Opening Yew: 6
R-00000BA6 | POSTED | transfer 1 from Yew to Ash"""
HAND_MENU = ["Fir", TIE, "Yew", "Ash", "Pine"]


def hand(task=CHOICE, prompt=HAND):
    return {"task": task, "id": "invented-é", "group_id": "arbitrary", "prompt": prompt,
            "choices": (HAND_MENU if task == CHOICE else MENUS[task]).copy()}


def digest(row):
    payload = json.dumps(row, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reference(prompt):
    """Independent forward accountant: sum incoming/outgoing columns by account."""
    opening, entries = {}, {}
    for line in prompt.splitlines()[1:]:
        if line.startswith("Opening "):
            name, units = line.removeprefix("Opening ").split(": ")
            opening[name] = int(units)
        elif line.startswith("Focus depot:"):
            focus = line.split()[-1]
        else:
            ticket, status, body = line.split(" | ")
            units = int(body.split()[1])
            source = re.findall(r"from ([A-Za-z]+)", body)
            dest = re.findall(r"to ([A-Za-z]+)", body)
            entries[ticket] = status, units, next(iter(source), ""), next(iter(dest), "")
    posted = [(q, src, dst) for status, q, src, dst in entries.values() if status == "POSTED"]
    stock = {d: x + sum(q for q, _, dst in posted if d == dst) - sum(q for q, src, _ in posted if d == src)
             for d, x in opening.items()}
    return focus, opening, entries, stock


def decision(task, focus, stock):
    if task == VERIFY:
        return "Yes" if stock[focus] >= 5 else "No"
    if task == LEVEL:
        for text, upper in zip(LEVELS, (0, 3, 6, float("inf"))):
            if stock[focus] <= upper:
                return text
    ranked = sorted(stock, key=stock.get, reverse=True)
    return TIE if stock[ranked[0]] == stock[ranked[1]] else ranked[0]


def pairs(seed, size):
    rows = audit.iter_inputs(seed, size)
    return zip(rows, rows, strict=True)


class AdjudicationTests(unittest.TestCase):
    def test_thresholds(self):
        for units, expected_level in enumerate([0, 1, 1, 1, 2, 2, 2, 3, 3]):
            # Opening is on the *other* side of several cutpoints.
            prompt = "\n".join([
                HAND.splitlines()[0], "Focus depot: Ash", f"Opening Ash: {units + 1}",
                "Opening Yew: 1", "Opening Fir: 0", "Opening Pine: 8",
                "R-00000F40 | VOID | add 50 to Ash", "R-00000F41 | POSTED | remove 1 from Ash",
            ])
            with self.subTest(units=units):
                self.assertEqual(audit.adjudicate(hand(LEVEL, prompt))["gold_index"], expected_level)
                index = audit.adjudicate(hand(VERIFY, prompt))["gold_index"]
                self.assertIs(type(index), int)
                self.assertEqual(index, 1 if units in range(5) else 0)

    def test_binding_and_counterfactuals(self):
        for row, answer in [
            (hand(), TIE),
            (hand(VERIFY), "Yes"),
            (hand(LEVEL), LEVELS[2]),
            (hand(LEVEL, HAND.replace("Focus depot: Ash", "Focus depot: Yew")), LEVELS[1]),
            (hand(prompt=HAND.replace("POSTED | add 2", "VOID | add 2")), "Ash"),
            (hand(prompt=HAND.replace("remove 31", "remove 2")), TIE),
            (hand(prompt=HAND.replace("transfer 3 from Pine", "transfer 4 from Pine")), "Fir"),
        ]:
            with self.subTest(answer=answer, task=row["task"]):
                self.assertEqual(audit.adjudicate(row), {"id": row["id"], "input_sha256": digest(row),
                                                         "gold_index": row["choices"].index(answer)})
        row = hand()
        shuffled = dict(row, choices=row["choices"][1:] + row["choices"][:1])
        self.assertEqual(audit.adjudicate(shuffled)["gold_index"], 0)
        for edited in (shuffled, dict(row, id="elsewhere"), dict(row, group_id="elsewhere"),
                       dict(row, prompt=HAND.replace("add 2", "add  2"))):
            self.assertEqual(audit.adjudicate(edited)["input_sha256"], digest(edited))
            self.assertNotEqual(digest(row), digest(edited))
        self.assertEqual(audit.adjudicate(dict(reversed(row.items()))), audit.adjudicate(row))

    def test_fail_closed(self):
        for old, new in [
            (" (units; records unordered)", ""), ("Focus depot: Ash", "Focus depot: Elm"),
            ("Focus depot: Ash", ""), ("Focus depot: Ash", "Focus depot: Ash\nFocus depot: Ash"),
            ("Opening Fir: 0", "Opening Ash: 4"), ("Opening Fir: 0", ""), ("Opening Fir: 0", "Opening Fir: -1"),
            ("Opening Fir: 0", "Opening Fir: 0\nOpening Elm: 2"), ("Opening Ash: 4", "Opening Ash: 04"),
            ("R-00000BA1", "R-00000BA2"), ("R-00000BA1", "undocketed"), ("POSTED", "PENDING"),
            ("POSTED |", "POSTED | extra |"), ("VOID |", ""), ("remove 31", "remove -31"),
            ("remove 31", "remove 0"), ("remove 31", "return 31"), ("remove 31", "remove 3.1"),
            ("31 from Ash", "31 from Elm"), ("20 from Fir to Pine", "20 from Fir to Fir"),
            ("add 2 to Fir", "add 2 to Fir and Yew"), ("transfer 3 from Pine", "transfer 30 from Pine"),
            ("VOID", "POSTED"), ("POSTED", "VOID"),
        ]:
            with self.subTest(old=old, new=new), self.assertRaises(ValueError):
                audit.adjudicate(hand(prompt=HAND.replace(old, new)))
        for tail in ("extra note", "R-00000BA7 | POSTED | add 2 to Ash"):
            with self.subTest(tail=tail), self.assertRaises(ValueError):
                audit.adjudicate(hand(prompt=HAND + "\n" + tail))
        row = hand()
        for malformed in (dict(row, gold_index=3), {k: v for k, v in row.items() if k != "id"},
                          dict(row, id=" "), dict(row, group_id=4), dict(row, prompt=[]), dict(row, task="score"),
                          dict(row, choices=tuple(HAND_MENU)), dict(row, choices=[*HAND_MENU, "Elm"]),
                          dict(row, choices=["Ark", *HAND_MENU[1:]]), dict(row, choices=["Ash", *HAND_MENU[1:]]),
                          dict(row, choices=[["Fir"], *HAND_MENU[1:]]), dict(row, choices=HAND_MENU[:-1]),
                          dict(row, choices=["Tie" if c == TIE else c for c in HAND_MENU]),
                          dict(hand(VERIFY), choices=["No", "Yes"]), dict(hand(LEVEL), choices=LEVELS[::-1])):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                audit.adjudicate(malformed)


class DesignTests(unittest.TestCase):
    def test_two_seed_stream_against_forward_reference(self):
        all_ids, states, books = set(), set(), set()
        for seed in (81403, 94217):
            manifest = audit.group_manifest(seed, 96)
            self.assertEqual(len(manifest), 864)
            counts = defaultdict(Counter)
            for a, b in pairs(seed, 96):
                group, task = a["group_id"], a["task"]
                self.assertEqual((group, task), (b["group_id"], b["task"]))
                for opaque_id in (group, a["id"], b["id"]):
                    self.assertRegex(opaque_id, r"\A[0-9a-f]{32}\Z")
                    self.assertNotIn(opaque_id, a["prompt"])
                    self.assertNotIn(opaque_id, b["prompt"])
                self.assertNotIn(group, all_ids)
                all_ids.add(group)
                focus, opening, entries, stock = reference(a["prompt"])
                n = len(entries)
                self.assertEqual(manifest.pop(group), {"seed": seed, "task": task, "difficulty": n,
                                                       "renderings": {a["id"]: 0, b["id"]: 1}})
                focus2, opening2, entries2, stock2 = reference(b["prompt"])
                self.assertEqual((focus, opening, entries, stock), (focus2, opening2, entries2, stock2))
                self.assertNotEqual(list(opening), list(opening2))
                self.assertNotEqual(list(entries), list(entries2))
                self.assertEqual(set(opening), set(DEPOTS))
                self.assertEqual({s for s, _, _, _ in entries.values()}, {"POSTED", "VOID"})
                posted = [(q, s, d) for status, q, s, d in entries.values() if status == "POSTED"]
                focal = sum(focus in (s, d) for _, s, d in posted)
                self.assertGreaterEqual(focal, {2: 1, 6: 2, 12: 4}[n])
                if n > 2:
                    self.assertGreater(len(posted), focal)
                # Transfers cancel exactly in the network-wide total.
                external = sum(q for q, s, _ in posted if not s) - sum(q for q, _, d in posted if not d)
                _, reconciled = audit._read_stock(a["prompt"])
                self.assertEqual(reconciled, stock)
                self.assertEqual(sum(reconciled.values()), sum(opening.values()) + external)
                self.assertGreaterEqual(min(stock.values()), 0)
                self.assertGreaterEqual(min(opening.values()), 0)
                answer = decision(task, focus, stock)
                counts[task, n][answer] += 1  # Groups, not views or shuffled indices.
                # Stronger than text deduplication: ignore ticket IDs, ordering, task and focus.
                book = (tuple(sorted(opening.items())), tuple(sorted(entries.values())))
                self.assertNotIn(book, books)
                books.add(book)
                if task == CHOICE:
                    self.assertNotEqual(a["choices"], b["choices"])
                    leaders = {d for d in opening if opening[d] == max(opening.values())
                               or stock[d] == max(stock.values())}
                    self.assertTrue(any(src in leaders or dst in leaders for _, src, dst in posted))
                for row in (a, b):
                    self.assertEqual(set(row), {"id", "group_id", "task", "prompt", "choices"})
                    self.assertNotIn(row["id"], all_ids)
                    all_ids.add(row["id"])
                    normalized = " ".join(unicodedata.normalize("NFKC", row["prompt"]).casefold().split())
                    state = hashlib.sha256(normalized.encode()).digest()
                    self.assertNotIn(state, states)
                    states.add(state)
                    if task == CHOICE:
                        self.assertCountEqual(row["choices"], [*opening, TIE])
                    else:
                        self.assertEqual(row["choices"], MENUS[task])
                    self.assertEqual(audit.adjudicate(row), {"id": row["id"], "input_sha256": digest(row),
                                                             "gold_index": row["choices"].index(answer)})
            self.assertFalse(manifest)
            self.assertEqual(len(counts), 9)
            for (task, _), classes in counts.items():
                self.assertEqual(set(classes), set(MENUS[task]))
                self.assertEqual(sum(classes.values()), 96)
                self.assertLessEqual(max(classes.values()) - min(classes.values()), 1)
        self.assertEqual((len(books), len(states), len(all_ids)), (1728, 3456, 5184))

    def test_reproducibility_and_partial_cells(self):
        for n in (1, 7):
            rows = list(audit.iter_inputs(-19, n))
            self.assertEqual(rows, list(audit.iter_inputs(-19, n)))
            manifest = audit.group_manifest(-19, n)
            self.assertEqual(manifest, audit.group_manifest(-19, n))
            counts = defaultdict(Counter)
            for row in rows[::2]:
                task = row["task"]
                value = row["choices"][audit.adjudicate(row)["gold_index"]]
                counts[task, manifest[row["group_id"]]["difficulty"]][value] += 1
            self.assertEqual(len(rows), 18 * n)
            self.assertEqual(len(counts), 9)
            for (task, _), classes in counts.items():
                freqs = [classes[v] for v in MENUS[task]]
                self.assertLessEqual(max(freqs) - min(freqs), 1)
        for args in ((0, 0), (0, -1), (True, 2), (12, True), (1.5, 2), (1, 3.5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                list(audit.iter_inputs(*args))
            with self.assertRaises(ValueError):
                audit.group_manifest(*args)

    def test_no_opening_or_display_order_rule(self):
        opening_hits = defaultdict(list)
        opening_values, first_status, ops = defaultdict(set), defaultdict(set), defaultdict(set)
        menu_slots, book_slots, focus_names = defaultdict(set), defaultdict(set), defaultdict(set)
        menu_permutations, same_slot = set(), set()
        for a, b in pairs(2039, 64):
            task = a["task"]
            focus, opening, entries, stock = reference(a["prompt"])
            key = task, len(entries)
            answer = decision(task, focus, stock)
            opening_hits[key].append(decision(task, focus, opening) == answer)
            opening_values[key].add(opening[focus])
            first_status[key].add(next(iter(entries.values()))[0])
            for status, _, src, dst in entries.values():
                ops[key, status].add((bool(src), bool(dst)))
            if task == CHOICE:
                i, j = a["choices"].index(answer), b["choices"].index(answer)
                menu_slots[answer].add(i)
                same_slot.add(i == j)
                menu_permutations.add(tuple(a["choices"].index(c) for c in b["choices"]))
                if answer != TIE:
                    book_slots[answer].add(list(opening).index(answer))
                    focus_names[answer].add(focus)
        for key, hits in opening_hits.items():
            task, _ = key
            # Catch both "copy the opening" and "always invert the opening".
            self.assertGreater(sum(hits) / len(hits), 0.20 if task == VERIFY else 0.03)
            self.assertLess(sum(hits) / len(hits), 0.80 if task == VERIFY else 0.60)
            self.assertGreaterEqual(len(opening_values[key]), 8)
            self.assertEqual(first_status[key], {"POSTED", "VOID"})
            for status in first_status[key]:
                self.assertEqual(ops[key, status], {(True, True), (False, True), (True, False)})
        for winner in [*DEPOTS, TIE]:
            self.assertEqual(menu_slots[winner], set(range(5)))
            if winner != TIE:
                self.assertEqual(book_slots[winner], set(range(4)))
                self.assertEqual(focus_names[winner], set(DEPOTS))
        self.assertEqual(same_slot, {False, True})
        self.assertGreater(len(menu_permutations), 10)  # Not just a fixed rotation for view two.


if __name__ == "__main__":
    unittest.main()
