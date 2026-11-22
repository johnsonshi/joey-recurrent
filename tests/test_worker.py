import contextlib
import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from joey_research import worker
from joey_research.loop import LoopPlan
from joey_research.loop_analysis import Policy, Stop
from joey_research.worker import _packet, choice_request, compact_recipe, measure_choice


class WorkerTest(unittest.TestCase):
    row = {"id":"one", "group_id":"graph", "task":"choice", "prompt":"Which fits?", "choices":["Red","Green","Blue"]}

    def test_constructed_compact_plan_matches_its_independent_export_recipe(self):
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        for name in ("select-20","select-16","merge-16",*worker.REFINEMENT_RECIPES):
            self.assertEqual(compact_recipe(name),json.loads((root/"configs/compact"/(name+".json")).read_bytes()))
        packet={"schema":1,"split":"discovery","inputs_sha256":"0"*64,"loop":{"k":1},"window":[8,12],
                "reverse":False,"deadline_utc":100,"compact_recipe":"select-20"}
        with self.assertRaisesRegex(ValueError,"independent"):_packet(json.dumps(packet).encode())
        packet["compact_manifest_sha256"]="1"*64
        self.assertEqual(_packet(json.dumps(packet).encode())[0]["compact_recipe"],"select-20")
        settings={k:v for k,v in packet.items() if k not in worker._ENVELOPE}
        for name in worker.REFINEMENT_RECIPES:
            extra={**settings,"compact_recipe":name};mid=worker.method_sha256(extra)
            with self.assertRaisesRegex(ValueError,"refinement"):_packet(json.dumps({**packet,**extra}).encode())
            for split in ("selection","calibration","final","public-development"):
                envelope={**extra,"method_sha256":mid,"split":split}
                with self.assertRaisesRegex(ValueError,"refinement"):
                    worker._nominated(envelope,{mid:extra},{"method_sha256","split"})
            ok={**extra,"method_sha256":mid,"split":"refinement"}
            self.assertEqual(worker._nominated(ok,{mid:extra},{"method_sha256","split"})[1],(8,12))

    def test_reverse_is_mapped_back_and_does_not_send_bookkeeping(self):
        class Native:
            def branches(inner, state, questions, *, window):
                self.assertEqual(state, "Which fits?")
                self.assertEqual(list(questions["decision"]["criteria"].values()), ["Blue","Green","Red"])
                self.assertEqual(set(questions["decision"]), {"type","instructions","criteria"})
                branch = SimpleNamespace(labels=("opt_2","opt_1","opt_0"), layout=SimpleNamespace(input_ids=(1,2)),
                                         stock=lambda: [3,2,1], probabilities=lambda _: (.7,.2,.1),
                                         inference_context=contextlib.nullcontext)
                return {"decision":branch}
        pred, trace = measure_choice(Native(), self.row, (8,12), LoopPlan(), reverse=True)
        self.assertEqual(pred["probs"], [.1,.2,.7])
        self.assertEqual(trace["layer_calls"], 24)

    def test_reject_label_leak_and_final_packet(self):
        with self.assertRaises(ValueError):
            choice_request({**self.row, "gold_index":0}, (0,1,2))
        packet = {"schema":1,"split":"final","inputs_sha256":"0"*64,"window":[4,8],"loop":{"k":2},
                  "reverse":False,"deadline_utc":1}
        with self.assertRaisesRegex(ValueError, "discovery"):
            _packet(json.dumps(packet).encode())
        for k in (0,9):
            with self.assertRaises(ValueError):
                _packet(json.dumps({**packet,"split":"discovery","loop":{"k":k}}).encode())

    def test_same_compact_counterpart_charges_physical_coda(self):
        class Compact:
            physical_to_source = tuple(range(20))
            def branches(self, state, questions, *, window):
                return {"decision":SimpleNamespace(labels=tuple(questions["decision"]["criteria"]),
                       layout=SimpleNamespace(input_ids=(1,)), stock=lambda: 0, prelude=lambda: 0,
                       block=lambda h,s:h+1, coda=lambda h,s:h, fresh_state=lambda: object(),
                       probabilities=lambda h: (.5,.25,.25), inference_context=contextlib.nullcontext)}
        for k, expected in ((1,20),(2,24)):
            _,trace = measure_choice(Compact(), self.row, (8,12), LoopPlan(k=k))
            self.assertEqual(trace["layer_calls"],expected)

    def test_live_policy_pays_for_stop_readout_and_restores_original_order(self):
        class Native:
            def branches(self,state,questions,*,window):
                return {"decision":SimpleNamespace(labels=tuple(questions["decision"]["criteria"]),
                        layout=SimpleNamespace(input_ids=(1,2)), temperature=1., prelude=lambda:0,
                        block=lambda h,s:h+1,coda=lambda h,s:[4.,0.,-2.] if h==2 else [0.,0.,0.],
                        fresh_state=lambda:object(),inference_context=contextlib.nullcontext)}
        with patch.object(worker, "SlotOrder", wraps=worker.SlotOrder) as order:
            pred,trace=measure_choice(Native(),self.row,(8,12),LoopPlan(k=5),reverse=True,
                                      policy=Policy(stop=Stop("margin",0.5)),split="selection")
        self.assertEqual({c.args[0] for c in order.call_args_list}, {"selection"})
        self.assertGreater(pred["probs"][2],.9)
        self.assertEqual((trace["policy_decision_pass"],trace["layer_calls"],trace["callback_counts"]["coda_calls"]),(2,40,2))

    def test_selection_permit_verifies_complete_declared_settings_and_both_byte_pins(self):
        sha = lambda value: hashlib.sha256(value).hexdigest()
        settings = {"reverse": False, "window": [4, 8], "loop": {"k": 2}}
        mid = sha(b'{"loop":{"k":2},"reverse":false,"window":[4,8]}')
        self.assertEqual(worker.method_sha256(settings), mid)
        permit = {"schema": "joey-promotion-permit/v1", "split": "selection", "split_manifest_sha256": "a"*64,
                  "inputs_sha256": "b"*64, "methods": {mid: settings}}

        def admission(*, packet_updates=None, permit_updates=None, extra_methods=None,
                      expected_nom=None, expected_packet=None):
            doc = dict(permit, methods={**permit["methods"], **(extra_methods or {})})
            nom = json.dumps(dict(doc, **(permit_updates or {})), indent=2).encode()
            pkt = {"schema": 1, "split": "selection", "deadline_utc": 123, "inputs_sha256": permit["inputs_sha256"],
                   "permit_sha256": sha(nom), "method_sha256": mid, **settings, **(packet_updates or {})}
            raw = json.dumps(pkt).encode()
            return worker.selection_packet(raw, nom, packet_sha256=expected_packet or sha(raw),
                                           nomination_sha256=expected_nom or sha(nom))

        valid, _, _ = admission()
        with self.assertRaises(ValueError):
            _packet(json.dumps(valid).encode())
        for wrong in ({"window": [8, 12]}, {"loop": {"k": 2, "readout": "final"}}, {"reverse": True},
                      {"policy": {}}, {"compact_recipe": "select-20", "compact_manifest_sha256": "c"*64},
                      {"order_ensemble": {"numerator": 1, "denominator": 2, "seed": 3}},
                      {"method_sha256": "d"*64}, {"permit_sha256": "c"*64}, {"inputs_sha256": "f"*64},
                      {"family": "not-settings"}, {"split": "final"}, {"split": "discovery"}, {"split": "calibration"}):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                admission(packet_updates=wrong)
        for wrong in ({"permit_updates": {"split": "discovery"}}, {"permit_updates": {"split_manifest_sha256": ""}},
                      {"permit_updates": {"methods": {}}}, {"permit_updates": {"notes": "not allowed"}},
                      {"expected_nom": "0"*64}, {"expected_packet": "0"*64}, {"extra_methods": {"0"*64: settings}}):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                admission(**wrong)

        # Use independently pinned (even valid) optional settings, not just core fields.
        for method in (dict(settings, compact_recipe="select-20", compact_manifest_sha256="e"*64),
                       dict(settings, policy={"stop": {"kind": "margin", "threshold": 1.0}}),
                       dict(settings, loop={"k": 2, "method": "zero_extra"},
                            order_ensemble={"numerator": 1, "denominator": 2, "seed": 7})):
            key = worker.method_sha256(method)
            admission(extra_methods={key: method}, packet_updates={**method, "method_sha256": key})
            for optional in ("policy", "compact_recipe", "compact_manifest_sha256", "order_ensemble"):
                if optional in method:
                    altered = {k: v for k, v in method.items() if k != optional}
                    with self.assertRaises(ValueError):
                        admission(extra_methods={key: method}, packet_updates={**altered, "method_sha256": key})

    def test_spend_is_by_whole_groups_stratified_without_using_rows_or_scores(self):
        # At 2/3, A:1+1/3, B:1+1/3, C:0+2/3; three of five, not four.
        groups = [("A", "a1"), ("A", "a2"), ("B", "b1"), ("B", "b2"), ("C", "c1")]
        cfg = {"numerator": 2, "denominator": 3, "seed": 42}
        choose = lambda pairs, c=cfg: worker.ensemble_groups(pairs, c, split="selection")
        def ranked(task, group):
            raw = json.dumps(["joey-order-ensemble/v1", "selection", 42, task, group], separators=(",", ":")).encode()
            return hashlib.sha256(raw).hexdigest()
        expected = {"c1", min(("a1", "a2"), key=lambda g: ranked("A", g)),
                    min(("b1", "b2"), key=lambda g: ranked("B", g))}
        self.assertEqual(choose(groups), expected)
        self.assertEqual(choose(groups[::-1] + groups[:2] * 10), expected)
        self.assertEqual(choose(groups, dict(cfg, numerator=0)), set())
        self.assertEqual(choose(groups, dict(cfg, numerator=3)), {g for _, g in groups})
        with self.assertRaises(ValueError):
            choose(groups + [("C", "a1")])

    def test_ensemble_is_a_bounded_split_only_control(self):
        cfg = {"numerator": 1, "denominator": 4, "seed": 0}
        base = {"schema": 1, "split": "discovery", "inputs_sha256": "d"*64, "window": [8, 12],
                "loop": {"k": 2, "method": "zero_extra"}, "reverse": False, "deadline_utc": 123, "order_ensemble": cfg}
        self.assertEqual(_packet(json.dumps(base).encode())[2], LoopPlan(k=2, method="zero_extra"))
        for wrong in ({"reverse": True}, {"policy": {}}, {"compact_manifest_sha256": "e"*64},
                      {"loop": {"k": 1, "method": "zero_extra"}}, {"loop": {"k": 2}},
                      {"loop": {"k": 2, "method": "zero_extra", "readout": "diagnostic"}},
                      *({"order_ensemble": dict(cfg, **edit)} for edit in (
                          {"seed": True}, {"seed": 2**32}, {"numerator": 1.0}, {"numerator": -1}, {"numerator": 5},
                          {"denominator": 0}, {"denominator": 10001}, {"rank_by": "confidence"}))):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                _packet(json.dumps(dict(base, **wrong)).encode())

    def test_public_native_frozen_six_are_genuine_measured_loops(self):
        common = {"window": [8, 12], "reverse": False}
        compact = {"compact_recipe": "select-20", "compact_manifest_sha256": "a"*64}
        arms = [(dict(common, loop={"k": k, "method": m}, **extra), layer_calls) for k,m,extra,layer_calls in (
            (1,"naive",{},24), (2,"zero_extra",{},24), (2,"naive",{},28), (3,"euler",{},32),
            (1,"naive",compact,20), (2,"naive",compact,24))]
        methods = {worker.method_sha256(s): s for s,_ in arms}
        def admit(s, extra=None):
            mid = worker.method_sha256(s)
            all_methods = methods if extra is None else {**methods, worker.method_sha256(extra): extra}
            return worker._nominated(dict(s, split="public-native", method_sha256=mid), all_methods, {"split","method_sha256"})
        row = dict(self.row, choices=[f"c{i}" for i in range(77)])
        for s, charged in arms:
            with self.subTest(settings=s):
                _, window, plan = admit(s)
                observed, vectors, decoded = [], [], []
                def invoke(op, value):
                    observed.append(op)
                    return value
                p = [1/77]*77
                def decode(z):
                    decoded.append(z)
                    return {"type": "choice", "choice": "option_4", "probabilities": dict(zip(worker.NATIVE_OPTIONS,p))}
                b = SimpleNamespace(labels=worker.NATIVE_OPTIONS, layout=SimpleNamespace(input_ids=(1,)),
                    stock=lambda: invoke("stock_calls", 1.), prelude=lambda: invoke("prelude_calls", 0.),
                    block=lambda h,state: invoke("block_calls", h+1.), coda=lambda h,state: invoke("coda_calls", h),
                    probabilities=lambda z: vectors.append(z) or p, decode=decode, fresh_state=object,
                    inference_context=contextlib.nullcontext)
                native = SimpleNamespace(branches=lambda *args,**kw: {"q1": b})
                if "compact_recipe" in s:
                    native.physical_to_source = tuple(range(20))
                _, trace = measure_choice(native, row, window, plan, split="public-native")
                self.assertEqual(len(decoded), 1)
                self.assertEqual(len(vectors), 1)
                self.assertIs(decoded[0], vectors[0])
                self.assertEqual(trace["layer_calls"], charged)
                self.assertEqual(trace["callback_counts"], {k: observed.count(k) for k in trace["callback_counts"]})
        base = arms[0][0]
        for prohibited in (dict(base, reverse=True), dict(base, window=[4,8]), dict(base, policy={}),
                           dict(base, loop={"k": 3, "method": "naive"}), dict(base, loop={"readout":"diagnostic"}),
                           dict(base, compact_manifest_sha256="a"*64), dict(base, **dict(compact,compact_recipe="select-16")),
                           dict(arms[3][0], **compact),
                           dict(arms[1][0], order_ensemble={"numerator":1,"denominator":2,"seed":0})):
            # Even an unused out-of-scope entry invalidates the permit.
            with self.subTest(prohibited=prohibited), self.assertRaises(ValueError):
                admit(base, prohibited)


class TypedTransferTest(unittest.TestCase):
    def test_one_loop_two_reads_and_exact_native_request_for_each_invented_case(self):
        from joey_research.evaluation import input_sha256
        from joey_research.native import BranchLayout

        levels = ["Empty (0 units)", "Low (1 to 3 units)", "Medium (4 to 6 units)", "High (7 or more units)"]
        states = ("Focus: Aster. Opening units: Aster 3, Beryl 4, Umber 0, Coral 2. Aster +1 posted; Beryl +4 pending.",
                  "Focus: Umber. Opening units: Aster 3, Beryl 1, Umber 0, Coral 5. Umber +8 posted; Aster −3 posted.")
        for case, state in enumerate(states):
            depots = ["Aster", "No unique depot", "Beryl", "Umber", "Coral"][::1 if case else -1]
            for task, menu, kind, instruction, criteria, labels, p in (
                ("stock-eligibility", ["Yes", "No"], "noul", "After posted entries only, does the focus depot have at least five units?",
                 {"true": "Yes", "false": "No"}, ("yes", "no"), (.64321089, .35678911)),
                ("stock-level", levels, "score", "After posted entries only, rate the final stock of the focus depot using the ordered levels.",
                 levels, ("0", "1", "2", "3"), (.12345671, .2, .3, .37654329)),
                ("stock-choice", depots, "choice", "After posted entries only, which depot has the most units? "
                 "If there is no unique maximum, choose No unique depot.",
                 {f"option_{i}": x for i, x in enumerate(depots)}, tuple(f"option_{i}" for i in range(5)),
                 (.42345671, .1, .2, .2, .07654329)),
            ):
                row = dict(id=f"opaque-{task}-{case}", task=task, group_id="never-forward", prompt=state, choices=menu)
                q = {"type": kind, "instructions": instruction, "criteria": criteria}
                raw_pin = input_sha256(row)
                for depth in (20, 24):
                    for k, method in ((1, "naive"), (2, "zero_extra"), (2, "naive"), (4, "naive"), (6, "naive")):
                        with self.subTest(case=case, task=task, depth=depth, k=k, method=method):
                            logits = object()
                            calls, reads = [], []
                            def call(name, value):
                                calls.append(name)
                                return value
                            def read(name, z, value):
                                self.assertIs(z, logits)
                                reads.append(name)
                                return value
                            answer = {"type": kind}
                            if kind == "noul":
                                answer["noul"] = round(p[0], 6)
                            else:
                                answer["probabilities"] = {s: round(v, 6) for s, v in zip(labels, p, strict=True)}
                                answer[kind] = (round(sum(i*v for i, v in enumerate(p)), 6) if kind == "score" else "option_3")
                            def branches(text, qs, *, window):
                                self.assertEqual((text, qs, window), (state, {"audit": q}, (8, 12)))
                                if kind == "choice":
                                    self.assertEqual(list(qs["audit"]["criteria"].items()), list(criteria.items()))
                                layout = BranchLayout("audit", kind, labels, tuple(range(len(labels))), 1, (8, 9), ())
                                return {"audit": SimpleNamespace(labels=layout.labels, layout=layout, fresh_state=object,
                                    inference_context=contextlib.nullcontext, stock=lambda: call("stock", logits),
                                    prelude=lambda: call("prelude", 0), block=lambda h, s: call("block", h + 1),
                                    coda=lambda h, s: call("coda", logits),
                                    probabilities=lambda z: read("vector", z, p), decode=lambda z: read("decode", z, answer))}
                            native = SimpleNamespace(branches=branches)
                            if depth == 20:
                                native.physical_to_source = (*range(12), *range(16, 24))
                            with patch.object(worker, "run_loop", wraps=worker.run_loop) as loop:
                                pred, trace = measure_choice(native, row, (8, 12), LoopPlan(k=k, method=method), split="typed-transfer")
                            loop.assert_called_once()
                            blocks = 1 if method == "zero_extra" else k
                            self.assertEqual(calls, ["stock"] if k == 1 else ["prelude", *(["block"] * blocks), "coda"])
                            self.assertEqual(trace["callback_counts"], {op + "_calls": calls.count(op) for op in ("stock", "prelude", "block", "coda")})
                            self.assertEqual((trace["layer_calls"], trace["tokens"], trace["readouts"]), (depth + 4*(blocks - 1), 2, []))
                            self.assertEqual(reads, ["vector", "decode"])
                            self.assertIs(trace["transfer_answer"], answer)
                            self.assertNotIn("native_answer", trace)
                            self.assertEqual(pred, {"id": row["id"], "input_sha256": raw_pin, "probs": list(p)})
                            self.assertEqual(input_sha256(row), raw_pin)

    def test_transfer_rejects_wrong_rendered_slots_and_compact_map_before_loop(self):
        row = dict(id="toy", group_id="g", task="stock-eligibility", prompt="Toy state", choices=["Yes", "No"])
        branch = SimpleNamespace(labels=("true", "false"))  # criteria keys aren't the rendered noul labels.
        native = SimpleNamespace(branches=lambda *args, **kw: {"audit": branch})
        with patch.object(worker, "run_loop") as loop:
            for changed in ({"reverse": True}, {"policy": Policy()}, {}):
                with self.assertRaises(ValueError):
                    measure_choice(native, row, (8, 12), LoopPlan(), split="typed-transfer", **changed)
            branch.labels = ("yes", "no")
            native.physical_to_source = tuple(range(20))  # correct depth, wrong retained cycle
            with self.assertRaises(ValueError):
                measure_choice(native, row, (8, 12), LoopPlan(), split="typed-transfer")
            loop.assert_not_called()

    def test_transfer_checks_even_unused_nominations_and_one_compact_pin(self):
        base = {"window": [8, 12], "loop": {"k": 1}, "reverse": False}
        compact = {"compact_recipe": "select-20", "compact_manifest_sha256": "a"*64}
        mid = worker.method_sha256(base)
        def admit(*settings):
            methods = {mid: base, **{hashlib.sha256(worker._canonical(s)).hexdigest(): s for s in settings}}
            return worker._nominated(dict(base, split="typed-transfer", method_sha256=mid), methods, {"split", "method_sha256"})
        approved = [dict(base, loop={"k": k, "method": method, "readout": "final"}, **extras)
                    for extras in ({}, compact) for k, method in ((1, "naive"), (2, "zero_extra"), (2, "naive"),
                                                                   (4, "naive"), (6, "naive"))
                    if extras or k != 1]
        self.assertEqual(admit(*approved)[2], LoopPlan())
        for change in ({"compact_manifest_sha256": "a"*64}, {"compact_recipe": "select-20"},
                       {**compact, "compact_recipe": "select-20-late"}, {**compact, "compact_recipe": "select-16"},
                       {**compact, "compact_manifest_sha256": "b"*64, "loop": {"k": 4}},
                       {"reverse": True}, {"window": [4, 8]}, {"policy": {}}, {"ignored": None},
                       {"loop": {"k": 3}}, {"loop": {"k": 6, "method": "zero_extra"}},
                       {"loop": {"k": 3, "method": "euler"}}, {"loop": {"k": 2, "readout": "diagnostic"}},
                       {"loop": {"k": 2, "alpha": None}}, {"loop": {"k": 2, "beta": None}},
                       {"loop": {"k": 2, "method": "zero_extra"}, "order_ensemble": {"numerator": 0, "denominator": 1, "seed": 0}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                admit(approved[-1], dict(base, **change))
