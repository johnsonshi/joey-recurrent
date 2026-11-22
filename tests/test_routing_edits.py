"""CPU-only checks on invented routing fixtures; no frozen inputs or labels."""
from collections import Counter, defaultdict
import json
import unittest
from unittest.mock import patch

from benchmarks import fixed_horizon, routing_edits as routing
from benchmarks.routing_edits_metrics import compare_edits
from joey_research.evaluation import EvaluationError, evaluate, input_sha256


_TOY_SEED, _SECOND_TOY_SEED = 0x6D3, 0x6D7
_ROLES = ("target_swap", "other_swap")


def _shown(row):
    # Independent reader of the required public grammar. Never use its labeler.
    return fixed_horizon.shown({**row, "task": fixed_horizon.TASK})


def _swap_printed_receivers(row, a, b):
    _, w, edges, _ = _shown(row)
    receivers = dict(edges)
    replacement = {a: receivers[b], b: receivers[a]}
    lines = row["prompt"].split("\n")
    for i, (s, _) in enumerate(edges, 2):
        if s in replacement:
            lines[i] = lines[i][:-3] + replacement[s] if w == 0 else replacement[s] + lines[i][3:]
    return {**row, "prompt": "\n".join(lines)}


def _rekey(row, meta):
    """Re-issue IDs for a deliberately damaged fixture to test structure, not just stale IDs."""
    goal, w, _, _ = _shown(row)
    case = routing._case_id(row["group_id"], meta["seed"], goal, meta["hop"])
    key = routing._row_id(case, w, meta["condition"], row["prompt"], row["choices"])
    return {**row, "id": key}, {**meta, "case_id": case}


def _audit(samples):
    return routing.check_blind_panel((r for r, _ in samples), {r["id"]: m for r, m in samples})


class RoutingEditsTests(unittest.TestCase):
    def setUp(self):
        self.samples = list(routing.iter_cases(_TOY_SEED, 1))
        self.cases = defaultdict(dict)
        for row, meta in self.samples:
            self.cases[meta["case_id"]][meta["rendering"], meta["condition"]] = row, meta

    def test_deterministic_blind_api(self):
        with patch.object(routing, "adjudicate", side_effect=AssertionError("blind solve")), \
             patch.object(routing, "label_row", side_effect=AssertionError("blind label")):
            two = list(routing.iter_cases(_TOY_SEED, 2))
            self.assertEqual(two, list(routing.iter_cases(_TOY_SEED, 2)))
            self.assertEqual(two[:48], self.samples)
            pooled = two + list(routing.iter_cases(_SECOND_TOY_SEED, 1))
            counts = _audit(pooled[::2] + pooled[1::2])
        self.assertEqual(json.loads(json.dumps(counts)), {"rows": 144, "groups": 3, "cases": 24})
        self.assertTrue(all(type(n) is int for n in counts.values()))
        self.assertNotEqual(routing.TASK, fixed_horizon.TASK)
        for row, meta in self.samples:
            self.assertEqual(set(row), {"id", "group_id", "task", "prompt", "choices"})
            self.assertEqual(row["task"], routing.TASK)
            self.assertEqual(set(meta), {"case_id", "hop", "rendering", "condition", "seed"})
            self.assertEqual(meta["seed"], _TOY_SEED)

    def test_sampling_bounds(self):
        for bad_seed in (-1, 2**32, True, 3.0, "4"):
            with self.subTest(seed=bad_seed), self.assertRaises(EvaluationError):
                next(routing.iter_cases(bad_seed, 1))
        for bad_count in (-1, 0, 1001, False, 1.0, "1"):
            with self.subTest(graphs=bad_count), self.assertRaises(EvaluationError):
                next(routing.iter_cases(_TOY_SEED, bad_count))
        self.assertEqual(next(routing.iter_cases(_TOY_SEED, 1000)), self.samples[0])

    def test_exact_display_repairs_and_balanced_reuse(self):
        orders, menus = defaultdict(set), defaultdict(set)
        uses = defaultdict(Counter)
        for slots in self.cases.values():
            goals, role_graphs = set(), defaultdict(set)
            for w in (0, 1):
                base, meta = slots[w, "base"]
                goal, wording, edges, entries = _shown(base)
                self.assertEqual(wording, w)
                goals.add(goal)
                orders[w].add(edges)
                menus[w].add(entries)
                pairs = []
                for role in _ROLES:
                    row, _ = slots[w, role]
                    new_goal, new_wording, changed, _ = _shown(row)
                    self.assertEqual((new_goal, new_wording, row["choices"]), (goal, w, base["choices"]))
                    self.assertEqual([s for s, _ in edges], [s for s, _ in changed])
                    pair = [s for (s, t), (_, u) in zip(edges, changed) if t != u]
                    self.assertEqual(len(pair), 2)
                    self.assertLessEqual(set(pair), set(entries))
                    repaired = _swap_printed_receivers(row, *pair)
                    self.assertEqual(repaired["prompt"], base["prompt"])  # every byte, not just the edge set
                    pairs.append(set(pair))
                    graph = frozenset(changed)
                    role_graphs[role].add(graph)
                    uses[w, meta["hop"], role][graph] += 1
                self.assertEqual(pairs[0] | pairs[1], set(entries))
            self.assertEqual(len(goals), 1)
            self.assertTrue(all(len(graphs) == 1 for graphs in role_graphs.values()))

        (a,), (b,) = orders[0], orders[1]
        (m,), (n,) = menus[0], menus[1]
        self.assertEqual(set(a), set(b))
        self.assertNotEqual(a, b)
        self.assertEqual(set(m), set(n))
        self.assertNotEqual(m, n)
        alternates = set.union(*(set(counts) for counts in uses.values()))
        self.assertEqual(len(alternates), 2)
        for w in (0, 1):
            for hop in (2, 6):
                for role in _ROLES:
                    self.assertEqual(uses[w, hop, role], {g: 2 for g in alternates})

    def test_independent_routes_and_real_row_labels(self):
        coverage = Counter()
        for slots in self.cases.values():
            for w in (0, 1):
                answers = {}
                for condition in ("base", *_ROLES):
                    row, meta = slots[w, condition]
                    goal, _, edges, roots = _shown(row)
                    parents = {t: s for s, t in edges}
                    coordinates = {}
                    for station in parents.keys() | set(roots):
                        node, depth = station, 0
                        while node in parents:  # independent reverse walk on this fixture only
                            node, depth = parents[node], depth + 1
                            self.assertLessEqual(depth, 6)
                        coordinates[station] = node, depth
                    self.assertCountEqual(coordinates.values(), [(r, h) for r in roots for h in range(7)])
                    entry, depth = coordinates[goal]
                    self.assertEqual(depth, meta["hop"])
                    answers[condition] = entry
                    if w == 0 and condition == "base":
                        coverage[entry, depth] += 1
                    label, info = routing.adjudicate(row)
                    self.assertEqual(info, {"entry": entry, "hop": depth, "rendering": w})
                    self.assertEqual(label, {"id": row["id"], "input_sha256": input_sha256(row),
                                             "gold_index": roots.index(entry)})
                self.assertNotEqual(answers["target_swap"], answers["base"])
                self.assertEqual(answers["other_swap"], answers["base"])
        self.assertEqual(coverage, {(r, h): 1 for r in roots for h in (2, 6)})

    def test_standalone_adjudication(self):
        base, _ = next(iter(self.cases.values()))[0, "base"]
        goal, _, edges, entries = _shown(base)
        # No design, generator identity or original menu order available at adjudication.
        row = {**base, "id": "invented-item", "group_id": "invented-group", "choices": base["choices"][::-1]}
        label, info = routing.adjudicate(row)
        self.assertEqual(label["input_sha256"], input_sha256(row))
        self.assertEqual(row["choices"][label["gold_index"]], "Enter at " + info["entry"])
        outgoing = dict(edges)
        wrong_depth = row["prompt"].replace(goal, outgoing[entries[0]], 1)
        uneven = _swap_printed_receivers(row, entries[0], outgoing[entries[1]])
        for bad in ({**row, "prompt": wrong_depth}, uneven, {**row, "task": fixed_horizon.TASK}):
            with self.assertRaises(EvaluationError):
                routing.adjudicate(bad)

    def test_evaluator_rejects_cross_condition_pairing(self):
        slots = next(iter(self.cases.values()))
        base, _ = slots[1, "base"]
        other, _ = slots[1, "other_swap"]
        label, _ = routing.adjudicate(base)
        # Even when the correct choice is preserved, transplanting the ID cannot rebind the label.
        with self.assertRaisesRegex(EvaluationError, "stale input digest"):
            evaluate([other], [{**label, "id": other["id"]}], {"fixture": []}, baseline="fixture",
                     bootstrap_reps=0, bootstrap_seed=_TOY_SEED)

    def test_strict_membership_and_design_binding(self):
        rows = [r for r, _ in self.samples]
        design = {r["id"]: m for r, m in self.samples}
        row, meta = self.samples[0]
        key = row["id"]
        for wrong_rows, wrong_design in (
            ([], {}), (rows[:-1], design), (rows + [row], design), (rows, {**design, "extra": meta}),
            (rows[1:], {k: v for k, v in design.items() if k != key}),
            ([{**row, "gold_index": 0}, *rows[1:]], design),
            ([{**row, "group_id": "another-group"}, *rows[1:]], design),
        ):
            with self.assertRaises(EvaluationError):
                routing.check_blind_panel(wrong_rows, wrong_design)
        for change in ({"entry": "ABC"}, {"case_id": "different"}, {"rendering": False}, {"hop": 2.0},
                       {"hop": 4}, {"condition": "unknown"}, {"seed": _SECOND_TOY_SEED}):
            with self.subTest(change=change), self.assertRaises(EvaluationError):
                routing.check_blind_panel(rows, {**design, key: {**meta, **change}})
        other, other_meta = self.samples[1]
        with self.assertRaises(EvaluationError):
            routing.check_blind_panel(rows, {**design, key: other_meta, other["id"]: meta})

    def test_rejects_local_damage_even_with_reissued_ids(self):
        slots = next(iter(self.cases.values()))
        base, _ = slots[0, "base"]
        target, meta = slots[0, "target_swap"]
        other, _ = slots[0, "other_swap"]
        _, _, edges, roots = _shown(base)
        nonroots = [s for s, _ in edges if s not in roots]
        lines = target["prompt"].split("\n")
        lines[2], lines[3] = lines[3], lines[2]
        different_goal = next(r for r, m in self.samples if m["case_id"] != meta["case_id"]
                              and m["rendering"] == 0)
        header = different_goal["prompt"].split("\n", 2)[:2]
        for name, bad, design in (
            ("duplicate cell", target, {**meta, "condition": "base"}),
            ("changed menu order", {**target, "choices": target["choices"][::-1]}, meta),
            ("changed record order", {**target, "prompt": "\n".join(lines)}, meta),
            ("extra newline", {**target, "prompt": target["prompt"] + "\n"}, meta),
            ("changed goal", {**target, "prompt": "\n".join(header + target["prompt"].split("\n")[2:])}, meta),
            ("no swap", base, meta), ("non-root swap", _swap_printed_receivers(base, *nonroots[:2]), meta),
            ("same swap twice", other, meta),
        ):
            replacement = _rekey(bad, design)
            with self.subTest(name=name), self.assertRaises(EvaluationError):
                _audit([replacement if r == target else (r, m) for r, m in self.samples])

    def test_rejects_unbalanced_or_additional_alternates(self):
        case, slots = next(iter(self.cases.items()))
        _, _, base_edges, _ = _shown(slots[0, "base"][0])
        original = dict(base_edges)
        _, _, target_edges, _ = _shown(slots[0, "target_swap"][0])
        _, _, other_edges, _ = _shown(slots[0, "other_swap"][0])
        left, right = ([s for s, t in changed if original[s] != t]
                       for changed in (target_edges, other_edges))
        for mode in ("one rendering", "both renderings", "different partition"):
            samples = []
            for r, m in self.samples:
                w, role = m["rendering"], m["condition"]
                if m["case_id"] == case and role in _ROLES:
                    if mode == "different partition":
                        i = _ROLES.index(role)
                        r = _swap_printed_receivers(slots[w, "base"][0], left[i], right[i])
                    elif mode == "both renderings" or w == 1:
                        r = slots[w, _ROLES[1 - _ROLES.index(role)]][0]
                    r, m = _rekey(r, m)
                samples.append((r, m))
            with self.subTest(mode=mode), self.assertRaises(EvaluationError):
                _audit(samples)

    def test_rejects_graph_reuse_across_groups(self):
        group = next(routing.iter_cases(_SECOND_TOY_SEED, 1))[0]["group_id"]
        duplicate = [_rekey({**r, "group_id": group}, {**m, "seed": _SECOND_TOY_SEED})
                     for r, m in self.samples]
        with self.assertRaises(EvaluationError):
            _audit(self.samples + duplicate)


class EditScoreTests(unittest.TestCase):
    def streams(self, samples, correct, *, reps=0):
        rows, labels, arms, design, tokens = [], [], [[], []], {}, {}
        for r, d in samples:
            rows.append(r)
            y, _ = routing.adjudicate(r)
            labels.append(y)
            for stream, right in zip(arms, correct(r, d), strict=True):
                top = (y["gold_index"] + int(not right)) % 4
                stream.append({"id": r["id"], "input_sha256": y["input_sha256"],
                               "probs": [float(i == top) for i in range(4)]})
            design[r["id"]] = d
            tokens[r["id"]] = 42 + int(d["condition"] == "other_swap")
        return dict(rows=rows, labels=labels, baseline=arms[0], candidate=arms[1], design_by_id=design,
                    baseline_tokens=tokens, candidate_tokens=tokens.copy(), reps=reps)

    def test_switch_and_preserve_are_not_three_correct_decisions(self):
        data = self.streams(routing.iter_cases(_TOY_SEED, 2), lambda r, d: (False, True), reps=100)
        z = compare_edits(**data)
        a = z["all"]
        self.assertEqual((z["rows"], a["triples"], z["cases"]), (96, 32, 16))
        self.assertEqual((a["baseline"]["switch_on_target"], a["baseline"]["preserve_on_other"]), (32, 32))
        self.assertEqual((a["baseline"]["all_three_correct"], a["candidate"]["all_three_correct"]), (0, 32))
        self.assertEqual((a["triple_corrections"], a["triple_corruptions"], a["all_three_equal_tokens"]), (32, 0, 0))
        self.assertEqual(a["paired"]["all_three"], {"delta": 1.0, "ci95": [1.0, 1.0]})
        self.assertEqual(z["by_hop"]["6"]["paired"]["target_swap_brier"]["delta"], -2)

    def test_draws_whole_graphs_and_uses_every_triple(self):
        rows = list(routing.iter_cases(_TOY_SEED, 2))
        first = rows[0][0]["group_id"]
        d = self.streams(rows, lambda r, m: (r["group_id"] == first, r["group_id"] != first), reps=200)
        a = compare_edits(**d)["all"]
        self.assertEqual(a["paired"]["all_three"], {"delta": 0.0, "ci95": [-1.0, 1.0]})
        self.assertEqual((a["triple_corrections"], a["triple_corruptions"]), (16, 16))
        singleton = self.streams(rows[:48], lambda r, m: (True, True), reps=50)
        self.assertIsNone(compare_edits(**singleton)["all"]["paired"]["all_three"]["ci95"])

    def test_structurally_balanced_but_semantically_inverted_design_is_rejected(self):
        samples = []
        for r, m in routing.iter_cases(_TOY_SEED, 1):
            if m["condition"] in _ROLES:
                r, m = _rekey(r, {**m, "condition": _ROLES[1 - _ROLES.index(m["condition"])]})
            samples.append((r, m))
        self.assertEqual(_audit(samples)["rows"], 48)
        with self.assertRaisesRegex(EvaluationError, "semantic adjudication"):
            compare_edits(**self.streams(samples, lambda r, m: (True, True)))

    def test_strict_binding_and_recorded_token_lengths(self):
        for key in ("labels", "candidate", "baseline_tokens"):
            d = self.streams(routing.iter_cases(_TOY_SEED, 1), lambda r, m: (True, True))
            if key == "labels":
                d[key][0]["gold_index"] = (d[key][0]["gold_index"] + 1) % 4
            elif key == "candidate":
                d[key].reverse()
            else:
                d[key][d["rows"][0]["id"]] += 1
            with self.subTest(key=key), self.assertRaises(EvaluationError):
                compare_edits(**d)


if __name__ == "__main__":
    unittest.main()
