"""Offline, single-question decision execution. The worker never loads labels or cloud clients."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from io import BytesIO

from .evaluation import input_sha256
from .loop import LoopPlan, run_loop
from .loop_analysis import Policy, SlotOrder, Stop, actuate

TRANSFER_MODE = "typed-transfer"
PROBE_MODE = "one-probe"
HORIZON_MODE = "routing-horizon"
EDIT_MODE = "routing-edits"
CROSS_MODE = "routing-crossover"
PHASES = {"public-development": "banking77-public-train-research-77/v1",
          "public-native": "banking77-original-train-native77/v1",
          TRANSFER_MODE: "joey-native-inventory-development/v1",
          PROBE_MODE: "joey-fresh-single-probe-routing/v1",
          HORIZON_MODE: "joey-constant-evidence-routing/v1",
          EDIT_MODE: "joey-matched-route-edits/v1",
          CROSS_MODE: "joey-two-device-timing/v1",
          "refinement": "joey-discovery-remainder/v1",
          "calibration": "joey-calibration-research/v1", "final": "joey-final-research/v1"}
REFINEMENT_RECIPES = ("select-20-late", "merge-20-mid")
NATIVE_OPTIONS = tuple(f"option_{i}" for i in range(77))
SELECT20_LAYERS = (*range(12), *range(16, 24))
TRANSFER_LOOPS = ((1, "naive"), (2, "naive"), (4, "naive"), (6, "naive"), (2, "zero_extra"))
HORIZON_LOOPS = (*TRANSFER_LOOPS, (8, "naive"))
EDIT_LOOPS = ((1, "naive"), (2, "zero_extra"), (2, "naive"), (4, "naive"))
_LEVELS = ("Empty (0 units)", "Low (1 to 3 units)", "Medium (4 to 6 units)", "High (7 or more units)")
_TRANSFER_TASKS = {
    "stock-eligibility": ("noul", ("yes", "no"), "After posted entries only, does the focus depot have at least five units?"),
    "stock-level": ("score", ("0", "1", "2", "3"),
                    "After posted entries only, rate the final stock of the focus depot using the ordered levels."),
    "stock-choice": ("choice", tuple(f"option_{i}" for i in range(5)),
                     "After posted entries only, which depot has the most units? "
                     "If there is no unique maximum, choose No unique depot."),
}


@dataclass(frozen=True)
class _FixedRoute:
    task: str
    loops: tuple[tuple[int, str], ...]


_FIXED_ROUTES = {HORIZON_MODE: _FixedRoute("fixed_evidence_route", HORIZON_LOOPS),
                 EDIT_MODE: _FixedRoute("parcel_entry_route", EDIT_LOOPS),
                 CROSS_MODE: _FixedRoute("parcel_entry_route", ((1, "naive"), (2, "zero_extra"), (2, "naive")))}
_SEALED_STREAM_MODES = (TRANSFER_MODE, PROBE_MODE, *_FIXED_ROUTES)


def compact_recipe(name: str) -> dict:
    if name == "select-20":
        groups, repeat = [{"cycles":[g]} for g in (0,1,2,4,5)], [2,3]
    elif name == "select-20-late":
        groups, repeat = [{"cycles":[g]} for g in (0,1,3,4,5)], [2,3]
    elif name == "merge-20-mid":
        groups = [{"cycles":[0]}, {"cycles":[1]}, {"cycles":[2,3],"weights":[0.5,0.5],"anchor":2},
                  {"cycles":[4]}, {"cycles":[5]}]
        repeat = [2,3]
    elif name == "select-16":
        groups, repeat = [{"cycles":[g]} for g in (0,2,3,5)], [1,3]
    elif name == "merge-16":
        groups = [{"cycles":[0]}, {"cycles":[1,2],"weights":[0.25,0.75],"anchor":2},
                  {"cycles":[3,4],"weights":[0.75,0.25]}, {"cycles":[5]}]
        repeat = [1,3]
    else:
        raise ValueError("unknown deterministic compact topology")
    return {"format":"joey-compact/v1","groups":groups,"repeat_cycles":repeat}


def _checked_input_sha256(row, split):
    digest = input_sha256(row)
    route = _FIXED_ROUTES.get(split)
    if route is not None:
        if (row["task"] != route.task or len(row["choices"]) != 4
                or not all(re.fullmatch(r"Enter at [A-Z]{3}", c) for c in row["choices"])):
            raise ValueError("fixed routing requires its own task and four-entry menu")
    elif row["task"] == _FIXED_ROUTES[EDIT_MODE].task:
        raise ValueError("edited routing input requires its own phase")
    return digest


def choice_request(row: dict, order: tuple[int, ...], *, split="discovery") -> tuple[str, dict]:
    _checked_input_sha256(row, split)
    choices = row["choices"]
    if len(choices) > 255 or sorted(order) != list(range(len(choices))):
        raise ValueError("unsupported choice count or ordering")
    criteria = {f"opt_{i}": choices[i] for i in order}
    if split in _FIXED_ROUTES and tuple(criteria) != ("opt_0", "opt_1", "opt_2", "opt_3"):
        raise ValueError("fixed routing requires forward slots")
    return row["prompt"], {"decision": {"type": "choice", "instructions": "Select the best option.", "criteria": criteria}}


def public_native_payload(row):
    """Reconstruct only the q1 payload whose equivalence was checked at admission."""
    _checked_input_sha256(row, "public-native")
    criteria = dict(zip(NATIVE_OPTIONS, row["choices"], strict=True))
    return {"state": {}, "questions": {"q1": {"type": "choice", "instructions": row["prompt"], "criteria": criteria}}}


def typed_transfer_payload(row: dict) -> dict:
    """Validate a generic input without changing its digest; return only the native request."""
    input_sha256(row)
    if row["task"] not in _TRANSFER_TASKS:
        raise ValueError("unknown typed-transfer task")
    kind, labels, instructions = _TRANSFER_TASKS[row["task"]]
    choices = row["choices"]
    if kind == "noul":
        if choices != ["Yes", "No"]:
            raise ValueError("invalid eligibility menu")
        criteria = {"true": "Yes", "false": "No"}
    elif kind == "score":
        if tuple(choices) != _LEVELS:
            raise ValueError("invalid ordered stock levels")
        criteria = list(choices)
    else:
        if len(choices) != len(labels) or choices.count("No unique depot") != 1:
            raise ValueError("invalid depot menu")
        criteria = dict(zip(labels, choices, strict=True))
    return {"state": row["prompt"], "questions": {"audit": {"type": kind, "instructions": instructions, "criteria": criteria}}}


def validate_transfer_answer(task: str, answer: dict, probs) -> None:
    """Check public native decoding against the *unrounded*, ordered branch vector."""
    def in_range(x, hi):
        return type(x) in (int, float) and 0 <= x <= hi

    def rounded(x, raw, hi):
        # Half a unit in Joey's last public decimal, plus binary arithmetic at a tie.
        return in_range(x, hi) and round(x, 6) == x and abs(x - raw) <= 0.5e-6 + 1e-14

    if task not in _TRANSFER_TASKS:
        raise ValueError("unknown typed-transfer task")
    kind, labels, _ = _TRANSFER_TASKS[task]
    width = len(labels)
    if (not isinstance(probs, (tuple, list)) or len(probs) != width
            or not all(in_range(p, 1) for p in probs) or abs(math.fsum(probs) - 1) > 1e-6):
        raise ValueError("invalid transfer prediction")
    fields = {"type", kind} if kind == "noul" else {"type", kind, "probabilities"}
    if type(answer) is not dict or set(answer) != fields or answer["type"] != kind:
        raise ValueError("invalid public transfer answer")
    if kind == "noul":
        if not rounded(answer["noul"], probs[0], 1):
            raise ValueError("invalid native P(yes)")
        return
    values = answer["probabilities"]
    if (type(values) is not dict or values.keys() != set(labels)
            or not all(rounded(values[label], p, 1) for label, p in zip(labels, probs, strict=True))
            or abs(math.fsum(values.values()) - 1) > 1e-5):
        raise ValueError("invalid native transfer probability map")
    if kind == "score":
        if not rounded(answer["score"], math.fsum(i * p for i, p in enumerate(probs)), width - 1):
            raise ValueError("invalid native expected level")
    elif answer["choice"] not in labels:
        raise ValueError("invalid declared native choice")


def validate_native_answer(answer):
    if (type(answer) is not dict or set(answer) != {"type", "choice", "probabilities"}
            or answer["type"] != "choice" or answer["choice"] not in NATIVE_OPTIONS):
        raise ValueError("invalid typed q1 choice")
    p = answer["probabilities"]
    if (type(p) is not dict or p.keys() != set(NATIVE_OPTIONS)
            or not all(type(v) in (float, int) and 0 <= v <= 1 for v in p.values())
            or abs(math.fsum(p.values()) - 1) > 0.01):
        raise ValueError("invalid typed q1 probability map")


def _native_arm(window, plan, compact):
    allowed = {(1, "naive"), (2, "naive")}
    if not compact:
        allowed.update({(2, "zero_extra"), (3, "euler")})
    if tuple(window) != (8, 12) or plan.readout != "final" or (plan.k, plan.method) not in allowed:
        raise ValueError("outside the frozen public-native comparison")
    return compact, plan.k, plan.method


def _transfer_arm(window, plan, compact):
    if tuple(window) != (8, 12) or plan.readout != "final" or (plan.k, plan.method) not in TRANSFER_LOOPS:
        raise ValueError("outside the frozen typed-transfer comparison")
    return compact, plan.k, plan.method


def _fixed_route_arm(settings, mode):
    compact = "compact_recipe" in settings
    keys = {"window", "loop", "reverse"} | ({"compact_recipe", "compact_manifest_sha256"} if compact else set())
    loop = settings["loop"]
    if (set(settings) != keys or settings["window"] != [8, 12] or settings["reverse"] is not False
            or type(loop) is not dict or set(loop) != {"k", "method"} or type(loop["k"]) is not int
            or (loop["k"], loop["method"]) not in _FIXED_ROUTES[mode].loops
            or (compact and settings["compact_recipe"] != "select-20")):
        raise ValueError(f"outside the fixed {mode} comparison")
    return compact, loop["k"], loop["method"]


def _horizon_arm(settings):
    return _fixed_route_arm(settings, HORIZON_MODE)


def measure_choice(native, row: dict, window: tuple[int, int], plan: LoopPlan, *, reverse=False,
                   policy=None, split="discovery", probe_margin=None) -> tuple[dict, dict]:
    if split not in ("discovery", "selection", *PHASES):
        raise ValueError("unsupported inference split")
    if probe_margin is not None and (split!=PROBE_MODE or reverse or policy is not None or tuple(window)!=(8,12)
                                     or plan.method!="naive" or plan.readout!="final"):
        raise ValueError("first-margin policy requires its own stage and forward order")
    order = tuple(range(len(row["choices"])))
    if reverse:
        order = order[::-1]
    if split == "public-native":
        if reverse or policy is not None:
            raise ValueError("original q1 requires the approved forward method")
        _native_arm(window, plan, hasattr(native, "physical_to_source"))
        payload = public_native_payload(row)
        state, questions, qid = payload["state"], payload["questions"], "q1"
        labels = NATIVE_OPTIONS
    elif split == TRANSFER_MODE:
        if reverse is not False or policy is not None:
            raise ValueError("typed-transfer requires a single forward-order method")
        compact = hasattr(native, "physical_to_source")
        _transfer_arm(window, plan, compact)
        if compact and tuple(native.physical_to_source) != SELECT20_LAYERS:
            raise ValueError("typed-transfer requires the verified select-20 layers")
        payload = typed_transfer_payload(row)
        state, questions, qid = payload["state"], payload["questions"], "audit"
        # Noul criteria keys and score descriptions are not the native slot labels.
        labels = _TRANSFER_TASKS[row["task"]][1]
    else:
        route = _FIXED_ROUTES.get(split)
        if route is not None and (reverse is not False or policy is not None or plan.readout != "final"
                or tuple(window) != (8, 12) or (plan.k, plan.method) not in route.loops
                or hasattr(native, "physical_to_source") and tuple(native.physical_to_source) != SELECT20_LAYERS):
            raise ValueError("nonfixed method or payload in fixed routing comparison")
        state, questions = choice_request(row, order, split=split)
        qid = "decision"
        labels = tuple(questions[qid]["criteria"])
    branch = native.branches(state, questions, window=window)[qid]
    if tuple(branch.labels) != labels:
        raise ValueError("native option order differs from the request")

    def original_order(p):
        if (len(p) != len(order) or (split in _FIXED_ROUTES and any(type(v) not in (int, float) for v in p))
                or any(not math.isfinite(v) or not 0 <= v <= 1 for v in p) or abs(math.fsum(p)-1) > 1e-6):
            raise ValueError("invalid native distribution")
        return [v for _, v in sorted(zip(order, p, strict=True))]
    lo, hi = window
    depth = len(native.physical_to_source) if hasattr(native, "physical_to_source") else 24
    digest = input_sha256(row)
    more = {}
    if probe_margin is not None:
        from .one_probe import run_one_probe
        if branch.temperature!=1:raise ValueError("first margin requires native raw probabilities")
        result=run_one_probe(branch,cap=plan.k,threshold=probe_margin)
        counts=result.counts
        if result.labels!=tuple(branch.labels):raise ValueError("native probe keys changed")
        p=original_order(result.first_probabilities if result.exit_depth==1 else branch.probabilities(result.logits))
        more["probe"]={"first_probs":original_order(result.first_probabilities),"first_margin":result.first_margin,
                       "exit_pass":result.exit_depth,"reason":"margin_met" if result.exit_depth==1 else "continue_to_cap"}
        trace=[]
    elif policy is None:
        result = run_loop(plan, branch)
        counts = result.counts
        p = original_order(branch.probabilities(result.logits))
        if split == "public-native":
            # Native typed decoding is a second read of these logits, not another model invocation.
            answer = branch.decode(result.logits)
            validate_native_answer(answer)
            more["native_answer"] = answer
        elif split == TRANSFER_MODE:
            answer = branch.decode(result.logits)
            validate_transfer_answer(row["task"], answer, p)
            more["transfer_answer"] = answer
        trace = [{"block_evaluations": r.block_evaluations, "kind": r.kind,
                  "probs": original_order(branch.probabilities(r.logits))} for r in result.readouts]
    else:
        if branch.temperature != 1:
            raise ValueError("raw-logit policy requires the pinned uncalibrated readout")
        decision = actuate(plan, branch, lambda: SlotOrder(split,digest,tuple(branch.labels)), policy)
        counts = decision.executed
        p = original_order(decision.probabilities)
        trace = []
        more = {"policy_decision_pass":decision.decision_pass,"policy_reason":decision.reason,
                "policy_fallback":decision.fallback_used, "mode":"live-actuated"}
    return (
        {"id": row["id"], "input_sha256": digest, "probs": p},
        {"id": row["id"], "tokens": len(branch.layout.input_ids), "layer_calls": counts.layer_evaluations(lo, hi-lo, depth-hi),
         "callback_counts": asdict(counts), "readouts": trace, **more},
    )


def _load(path: Path, digest: str) -> bytes:
    data = path.read_bytes()
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("staged input digest differs")
    return data


def _is_gated_shard(path, gated):
    if path is None:
        return False
    if path in gated or path.resolve() in gated:
        return True
    # Discovery accepts arbitrary paths. A hard link can name the same
    # staged file without resolving to its path; compare identities, not bytes.
    try:
        st = path.stat()
    except FileNotFoundError:
        return False
    return st.st_nlink > 1 and any(p.is_file() and os.path.samestat(st, p.stat()) for p in gated)


def _policy(obj):
    if not isinstance(obj,dict) or set(obj)-{"pool","stop","fallback","first_weight"}:
        raise ValueError("invalid discovery policy")
    cfg=dict(obj)
    if "stop" in cfg:
        if not isinstance(cfg["stop"],dict) or set(cfg["stop"])!={"kind","threshold"}:
            raise ValueError("invalid stopping rule")
        cfg["stop"]=Stop(**cfg["stop"])
    return Policy(**cfg)


_ENVELOPE = {"schema", "split", "inputs_sha256", "deadline_utc"}
_SELECTION_ENVELOPE = _ENVELOPE | {"permit_sha256", "method_sha256"}
_PHASE_ENVELOPE = _SELECTION_ENVELOPE | {"view_sha256"}
_REQUIRED_SETTINGS = {"window", "loop", "reverse"}
_SETTINGS = _REQUIRED_SETTINGS | {"policy", "compact_recipe", "compact_manifest_sha256", "order_ensemble", "probe_margin"}
PERMIT_LIMIT = 128 * 1024


def _pin(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _fraction(cfg):
    if (type(cfg) is not dict or set(cfg) != {"numerator", "denominator", "seed"}
            or any(type(v) is not int for v in cfg.values())
            or not 1 <= cfg["denominator"] <= 10000 or not 0 <= cfg["numerator"] <= cfg["denominator"]
            or not 0 <= cfg["seed"] < 2**32):
        raise ValueError("invalid group spend budget")
    return cfg["numerator"], cfg["denominator"], cfg["seed"]


def ensemble_groups(task_groups, config, *, split):
    """Choose whole groups from (task, group_id) pairs, independent of rows, labels and scores."""
    numerator, denominator, seed = _fraction(config)
    if split not in ("discovery", "selection", *PHASES):
        raise ValueError("unsupported spend split")
    owners, tasks = {}, {}
    for task, group in task_groups:
        if owners.setdefault(group, task) != task:
            raise ValueError("one group spans tasks")
        tasks.setdefault(task, set()).add(group)
    quotas = {t: divmod(len(groups) * numerator, denominator) for t, groups in tasks.items()}
    left = len(owners) * numerator // denominator - sum(q for q, _ in quotas.values())
    extra = set(sorted(tasks, key=lambda t: (-quotas[t][1], t))[:left])
    selected = set()
    for task, groups in tasks.items():
        def rank(group):
            key = ["joey-order-ensemble/v1", split, seed, task, group]
            return hashlib.sha256(_canonical(key)).digest(), group
        selected.update(sorted(groups, key=rank)[:quotas[task][0] + (task in extra)])
    return selected


def _settings(obj):
    if type(obj) is not dict or not _REQUIRED_SETTINGS <= obj.keys() <= _SETTINGS or type(obj["reverse"]) is not bool:
        raise ValueError("unexpected inference settings")
    if "probe_margin" in obj:
        from .probe_runtime import method
        method(obj)
    if "compact_manifest_sha256" in obj and not _pin(obj["compact_manifest_sha256"]):
        raise ValueError("invalid compact receipt")
    if "compact_recipe" in obj:
        compact_recipe(obj["compact_recipe"])
        if "compact_manifest_sha256" not in obj:
            raise ValueError("derived compact payload needs an independent coordinator pin")
    w = obj["window"]
    if not isinstance(w, list) or len(w) != 2 or not all(type(i) is int for i in w) or not 0 <= w[0] < w[1] < 24:
        raise ValueError("invalid stock model window")
    if not isinstance(obj["loop"], dict) or set(obj["loop"]) - {"k", "method", "alpha", "beta", "readout"}:
        raise ValueError("unrecognized loop configuration")
    plan = LoopPlan(**obj["loop"])
    if plan.k > 8:
        raise ValueError("pass budget exceeded")
    if "policy" in obj:
        _policy(obj["policy"])
        if plan.method not in ("naive","damped_extra") or plan.readout != "final":
            raise ValueError("live pooling needs completed repeats; use separate diagnostic packets for recording")
    if "order_ensemble" in obj:
        _fraction(obj["order_ensemble"])
        if (plan != LoopPlan(k=2, method="zero_extra") or obj["reverse"]
                or set(obj["loop"]) - {"k", "method", "readout"}
                or {"compact_recipe", "compact_manifest_sha256", "policy"} & obj.keys()):
            raise ValueError("order ensemble needs forward-order full-model zero-extra split passes")
    return (w[0], w[1]), plan


def method_sha256(settings: dict) -> str:
    """Digest of canonical *declared* worker settings, without default substitution."""
    _settings(settings)
    return hashlib.sha256(_canonical(settings)).hexdigest()


def _packet(raw: bytes) -> tuple[dict, tuple[int, int], LoopPlan]:
    """Legacy discovery entry point; never route on a self-declared packet split."""
    obj = json.loads(raw)
    if not isinstance(obj, dict) or not _ENVELOPE <= obj.keys() or obj["schema"] != 1:
        raise ValueError("unexpected worker packet")
    if obj["split"] != "discovery":
        raise ValueError("worker only accepts explicit discovery packets")
    if "probe_margin" in obj:raise ValueError("first-margin method requires an independent phase release")
    if obj.get("compact_recipe") in REFINEMENT_RECIPES:
        raise ValueError("this compact recipe requires a separate refinement permit")
    return obj, *_settings({k: v for k, v in obj.items() if k not in _ENVELOPE})


def selection_packet(raw: bytes, nomination: bytes, *, packet_sha256: str, nomination_sha256: str):
    """Verify the two *independent* byte pins and bind the entire selection method."""
    from .transport import _json
    if (len(raw) > 8192 or len(nomination) > PERMIT_LIMIT or not _pin(packet_sha256) or not _pin(nomination_sha256)
            or hashlib.sha256(raw).hexdigest() != packet_sha256
            or hashlib.sha256(nomination).hexdigest() != nomination_sha256):
        raise ValueError("unbound selection admission")
    obj, permit = _json(raw), _json(nomination)
    if (type(permit) is not dict or set(permit) != {"schema", "split", "inputs_sha256", "split_manifest_sha256", "methods"}
            or permit["schema"] != "joey-promotion-permit/v1" or permit["split"] != "selection"
            or not _pin(permit["inputs_sha256"]) or not _pin(permit["split_manifest_sha256"])
            or type(permit["methods"]) is not dict or not 1 <= len(permit["methods"]) <= 256):
        raise ValueError("invalid selection permit")
    if (type(obj) is not dict or not _SELECTION_ENVELOPE <= obj.keys()
            or type(obj["schema"]) is not int or obj["schema"] != 1 or obj["split"] != "selection"
            or type(obj["deadline_utc"]) is not int or obj["deadline_utc"] <= 0
            or permit["inputs_sha256"] != obj["inputs_sha256"] or obj["permit_sha256"] != nomination_sha256
            or not _pin(obj["method_sha256"]) or obj["method_sha256"] not in permit["methods"]):
        raise ValueError("selection packet does not match its permit")
    return _nominated(obj, permit["methods"], _SELECTION_ENVELOPE)


def _nominated(obj, methods, envelope):
    if type(methods) is not dict or not 1 <= len(methods) <= 256 or obj["method_sha256"] not in methods:
        raise ValueError("no authorized method")
    arms, compact_pins = set(), set()
    for key, settings in methods.items():
        if key != method_sha256(settings):
            raise ValueError("invalid nominated method ID")
        if obj["split"] != "refinement" and settings.get("compact_recipe") in REFINEMENT_RECIPES:
            raise ValueError("refinement-only recipe outside its authorized phase")
        if obj["split"] == PROBE_MODE:
            from .probe_runtime import method
            arm=method(settings)
            if arm[0]:
                compact_pins.add(settings["compact_manifest_sha256"])
            if arm in arms or len(compact_pins)>1:raise ValueError("duplicate probe arm or different compact payloads")
            arms.add(arm)
        elif "probe_margin" in settings:raise ValueError("first-margin method outside its independent phase")
        if obj["split"] in _FIXED_ROUTES:
            arm = _fixed_route_arm(settings, obj["split"])
            if arm[0]:
                compact_pins.add(settings["compact_manifest_sha256"])
            if arm in arms or len(compact_pins) > 1:
                raise ValueError("duplicate routing arm or inconsistent physical payload")
            arms.add(arm)
        if obj["split"] in ("public-native", TRANSFER_MODE):
            window, plan = _settings(settings)
            compact = "compact_manifest_sha256" in settings
            if compact:
                compact_pins.add(settings["compact_manifest_sha256"])
            if obj["split"] == TRANSFER_MODE:
                if set(settings["loop"]) - {"k", "method", "readout"}:
                    raise ValueError("unused typed-transfer loop parameter")
                arm = _transfer_arm(window, plan, compact)
            else:
                arm = _native_arm(window, plan, compact)
            if (settings["reverse"] or {"policy", "order_ensemble"} & settings.keys()
                    or (compact and settings.get("compact_recipe") != "select-20")
                    or arm in arms or len(compact_pins) > 1):
                raise ValueError(f"outside the frozen {obj['split']} comparison")
            arms.add(arm)
    declared = {k: v for k, v in obj.items() if k not in envelope}
    window, plan = _settings(declared)
    # Comparing canonical bytes also distinguishes true/1, 1/1.0, and a
    # missing field vs an explicitly supplied default.
    if _canonical(declared) != _canonical(methods[obj["method_sha256"]]):
        raise ValueError("selection settings differ from the nominated method")
    return obj, window, plan


@dataclass(frozen=True)
class PhaseGate:
    """Coordinator-held pins, not claims populated by parsing the mounted files."""
    mode: str
    admit_before: int
    hard_stop: int
    packet_sha256: str
    permit_sha256: str
    run_sha256: str
    view_sha256: str
    approval_sha256: str | None = None
    final_freeze_sha256: str | None = None

    def __post_init__(self):
        if (self.mode not in PHASES
                or not all(map(_pin, (self.packet_sha256, self.permit_sha256, self.run_sha256, self.view_sha256)))
                or type(self.admit_before) is not int or type(self.hard_stop) is not int
                or not 0 < self.admit_before <= self.hard_stop < 2**53):
            raise ValueError("invalid phase admission")
        final = self.mode == "final"
        if not (_pin(self.final_freeze_sha256) and self.approval_sha256 is None if final
                else _pin(self.approval_sha256) and self.final_freeze_sha256 is None):
            raise ValueError("phase requires its own release")

    def evidence(self):
        release = ("final_freeze_sha256" if self.mode == "final" else "approval_sha256")
        return {"run_sha256": self.run_sha256, "view_sha256": self.view_sha256, release: getattr(self, release)}

    def check_start(self, now):
        if now >= min(self.admit_before, self.hard_stop - 15):
            raise TimeoutError
        if self.hard_stop - now > 3600:
            raise ValueError("unbounded phase")


def view_sha256(view):
    """Hash the complete JSON view descriptor; no whitespace, newline or omitted fields."""
    return hashlib.sha256(_canonical(view)).hexdigest()


def phase_packet(raw, permit_bytes, gate: PhaseGate):
    """Read only a permit of the independently admitted role. Returns packet, window, plan, view."""
    from .transport import _json
    if (len(raw) > 8192 or len(permit_bytes) > PERMIT_LIMIT
            or hashlib.sha256(raw).hexdigest() != gate.packet_sha256
            or hashlib.sha256(permit_bytes).hexdigest() != gate.permit_sha256):
        raise ValueError("unbound phase packet or permit")
    obj, permit = _json(raw), _json(permit_bytes)
    release = gate.evidence()
    expected = {"schema": f"joey-{gate.mode}-permit/v1", "split": gate.mode, "admit_before": gate.admit_before,
                "hard_stop": gate.hard_stop, **{k: v for k, v in release.items() if k != "view_sha256"}}
    if (type(permit) is not dict or set(permit) != expected.keys() | {"not_before", "view", "methods"}
            or any(type(permit[k]) is not type(v) or permit[k] != v for k, v in expected.items())):
        raise ValueError("phase release or interval differs")
    earliest = permit["not_before"]
    if type(earliest) is not int or not 0 < earliest < gate.admit_before or gate.hard_stop - earliest > 3600:
        raise ValueError("invalid phase release window")
    if time.time() < earliest:
        raise TimeoutError
    view = permit["view"]
    fields = {"schema", "split", "projection", "inputs_sha256", "split_manifest_sha256", "rows", "bytes"}
    if gate.mode in ("public-development", "public-native"):
        fields.add("menu_sha256")
    if gate.mode == "public-native":
        fields.update(("research_inputs_sha256", "native_requests_sha256"))
    if (type(view) is not dict or set(view) != fields or view_sha256(view) != gate.view_sha256
            or view["schema"] != "joey-label-free-view/v1" or view["split"] != gate.mode
            or view["projection"] != PHASES[gate.mode]
            or not all(_pin(view[k]) for k in fields if k.endswith("_sha256"))
            or type(view["rows"]) is not int or not 0 < view["rows"] <= (512 if gate.mode == "public-native" else 20000)
            or type(view["bytes"]) is not int or not 0 < view["bytes"] <= 64 * 2**20):
        raise ValueError("invalid phase view")
    if (type(obj) is not dict or not _PHASE_ENVELOPE <= obj.keys() or type(obj["schema"]) is not int or obj["schema"] != 1
            or obj["split"] != gate.mode or obj["view_sha256"] != gate.view_sha256
            or obj["inputs_sha256"] != view["inputs_sha256"] or obj["permit_sha256"] != gate.permit_sha256
            or not _pin(obj["method_sha256"]) or type(obj["deadline_utc"]) is not int
            or obj["deadline_utc"] != gate.hard_stop):
        raise ValueError("phase packet differs from frozen view")
    return (*_nominated(obj, permit["methods"], _PHASE_ENVELOPE), view)


def _upstream_line(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _original_pair(pair):
    if type(pair) is not dict or set(pair) != {"input", "native"}:
        raise ValueError("expected a blinded research/native pair")
    row, wrapper = pair["input"], pair["native"]
    payload = public_native_payload(row)
    if (type(wrapper) is not dict or set(wrapper) != {"id", "input_sha256", "payload"} or wrapper["id"] != row["id"]
            or wrapper["input_sha256"] != input_sha256(row) or wrapper["payload"] != payload
            or tuple(wrapper["payload"]["questions"]["q1"]["criteria"]) != NATIVE_OPTIONS):
        raise ValueError("original q1 and research row disagree")
    return row, wrapper


def phase_rows(raw, view):
    from .transport import _json
    if len(raw) != view["bytes"] or hashlib.sha256(raw).hexdigest() != view["inputs_sha256"]:
        raise ValueError("phase shard differs")
    seen, rows = set(), []
    paired = view["split"] == "public-native"
    if paired:
        research, native = hashlib.sha256(), hashlib.sha256()
    for line in BytesIO(raw):
        row = _json(line)
        if paired:
            row, wrapper = _original_pair(row)
            # Hash the originals in their source order, never the reconstructed request.
            research.update(_upstream_line(row))
            native.update(_upstream_line(wrapper))
        elif view["split"] == TRANSFER_MODE:
            typed_transfer_payload(row)
        else:
            _checked_input_sha256(row, view["split"])
            if view["split"]==PROBE_MODE and (len(row["choices"])!=4 or row["task"] not in ("routing_2hop","routing_3hop","routing_4hop")):
                raise ValueError("single-probe view needs four-way routed decisions")
        if len(rows) >= view["rows"] or row["id"] in seen or len(row["choices"]) > 255:
            raise ValueError("phase row inventory differs")
        if view["split"] in ("public-development", "public-native") and (
                len(row["choices"]) != 77 or hashlib.sha256(_canonical(row["choices"])).hexdigest() != view["menu_sha256"]):
            raise ValueError("public train requires the approved ordered 77-choice research menu")
        seen.add(row["id"])
        rows.append(row)
    if len(rows) != view["rows"]:
        raise ValueError("phase row count differs")
    if paired and (research.hexdigest() != view["research_inputs_sha256"]
                   or native.hexdigest() != view["native_requests_sha256"]):
        raise ValueError("recovered original JSONL digests differ")
    return rows


def _measure_ensemble(native, row, window, plan, split, *, second):
    if split == TRANSFER_MODE or split in _FIXED_ROUTES:
        raise ValueError(f"no order ensemble in {split}")
    pred, trace = measure_choice(native, row, window, plan, split=split)
    if second:
        rev, other = measure_choice(native, row, window, plan, split=split, reverse=True)
        pred["probs"] = [(a + b) / 2 for a, b in zip(pred["probs"], rev["probs"], strict=True)]
        for key in ("tokens", "layer_calls"):
            trace[key] += other[key]
        for key, count in other["callback_counts"].items():
            trace["callback_counts"][key] += count
    trace["ensemble_calls"] = trace["forward_calls"] = 2 if second else 1
    return pred, trace


def run(args) -> None:
    from . import transport
    if any(os.environ.get(k) != "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("offline loader flags are required")
    selection = getattr(args, "selection", False)
    nomination_pin = getattr(args, "nomination_sha256", None)
    phase = getattr(args, "phase", None)
    gate = None
    evidence_names = ("run_sha256", "view_sha256", "approval_sha256", "final_freeze_sha256")
    if phase is not None:
        if (selection or nomination_pin is not None or phase not in PHASES
                or args.packet != transport.ADMISSION / "packet.json" or args.model_dir != transport.STAGED / "model"
                or args.inputs != transport.STAGED / "inputs" / (phase + ".jsonl")
                or (phase in _SEALED_STREAM_MODES and (args.output != transport.WORK / "run/out" or args.compact_dir is not None))
                or args.compact_dir not in (None, transport.STAGED / "compact")):
            raise ValueError("wrong phase launch")
        gate = PhaseGate(phase, args.admit_before, args.hard_stop, args.packet_sha256, args.phase_permit_sha256,
                         **{k: getattr(args, k) for k in evidence_names})
        obj, window, plan, rows = transport._phase_inputs(gate)
    elif any(getattr(args, k, None) is not None for k in (*evidence_names, "phase_permit_sha256", "admit_before", "hard_stop")):
        raise ValueError("phase fields on a legacy launch")
    elif selection:
        if (not _pin(nomination_pin) or args.packet != transport.ADMISSION / "packet.json"
                or args.inputs != transport.STAGED / "inputs/selection.jsonl"
                or args.model_dir != transport.STAGED / "model"
                or args.compact_dir not in (None, transport.STAGED / "compact")):
            raise ValueError("selection requires fixed verified mounts")
        obj, window, plan = transport._read_selection_packet(args.packet_sha256, nomination_pin)
    else:
        if nomination_pin is not None:
            raise ValueError("discovery cannot consume a nomination")
        gated = {transport.STAGED / "inputs" / (n + ".jsonl") for n in _SEALED_STREAM_MODES}
        if any(_is_gated_shard(path, gated) for path in (args.packet, getattr(args, "inputs", None))):
            raise ValueError("staged shard requires its phase gate")
        obj, window, plan = _packet(_load(args.packet, args.packet_sha256))
    if type(obj["deadline_utc"]) is not int or not 0 < obj["deadline_utc"] - time.time() <= 30 * 3600:
        raise ValueError("expired or unbounded worker deadline")
    # The coordinator binds a stricter absolute cutoff to this exact packet.
    # The OS alarm also covers initialization and time spent inside native code.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    remaining = obj["deadline_utc"] - (15 if gate else 0) - time.time()
    if remaining <= 0:
        raise ValueError("worker deadline elapsed before start")
    signal.setitimer(signal.ITIMER_REAL, remaining)
    if selection:
        shard = transport._shard_bytes("selection")
        if hashlib.sha256(shard).hexdigest() != obj["inputs_sha256"]:
            raise ValueError("selection shard differs from permit")
        rows = [transport._json(s) for s in shard.splitlines()]
        del shard
    elif gate is None:
        rows = [json.loads(s) for s in _load(args.inputs, obj["inputs_sha256"]).splitlines()]
    if not 0 < len(rows) <= 20000 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("invalid tier or duplicate decision IDs")
    if phase not in ("public-native", TRANSFER_MODE):  # phase_rows already checked every native request.
        for row in rows:
            choice_request(row, tuple(range(len(row["choices"]))), split=obj["split"])
    ensemble = obj.get("order_ensemble")
    split = obj["split"]
    if ensemble is not None:
        selected = ensemble_groups(((r["task"], r["group_id"]) for r in rows), ensemble, split=split)
        spend = {"groups": len({r["group_id"] for r in rows}), "extra_groups": len(selected),
                 "forward_calls": 0, "layer_calls": 0, "coda_calls": 0}
    if os.environ.get("TMPDIR") == "/work/tmp":
        Path("/work/tmp").mkdir(mode=0o700, exist_ok=True)
    import torch
    from .native import NativeJoey
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise ValueError("one BF16-capable CUDA device is required")
    props = torch.cuda.get_device_properties(0)
    if not re.search(r"\bA100\b", props.name) or (props.major, props.minor) != (8, 0):
        raise ValueError("device class mismatch")
    os.umask(0o077)
    args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
    begin = time.perf_counter()
    if "compact_manifest_sha256" in obj:
        if "compact_recipe" in obj:
            from .compact import export_compact
            if args.compact_dir is not None:
                raise ValueError("choose a staged or a derived compact directory, not both")
            compact_dir = args.output.parent / "derived-compact"
            export_compact(args.model_dir, compact_dir, compact_recipe(obj["compact_recipe"]))
            _load(compact_dir / "compact.json", obj["compact_manifest_sha256"])
        elif args.compact_dir is None:
            raise ValueError("verified compact directory required")
        else:
            compact_dir = args.compact_dir
        from .native_compact import CompactJoey
        native = CompactJoey.load(compact_dir, source_model_dir=args.model_dir,
                                  expected_manifest_sha256=obj["compact_manifest_sha256"])
        if native.repeat_window != window:
            raise ValueError("compact window differs from the verified physical mapping")
    else:
        if args.compact_dir is not None:
            raise ValueError("compact directory without compact manifest pin")
        native = NativeJoey.load(args.model_dir)
    report = native.verify_parity(window=window)
    policy = _policy(obj["policy"]) if "policy" in obj else None
    # Parity uses fixed public cases. Don't observe any extra study row for
    # promotion or spend controls; retain the existing discovery warmup otherwise.
    frozen = selection or gate is not None
    if not frozen and ensemble is None:
        measure_choice(native, rows[0], window, plan, reverse=obj["reverse"], policy=policy, split=split)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - begin
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    times = []
    outputs = [args.output / "predictions.partial", args.output / "trace.partial"]
    names = ("predictions.jsonl", "trace.jsonl")
    digests = {name: hashlib.sha256() for name in names} if frozen else {}
    with outputs[0].open("xb") as predictions, outputs[1].open("xb") as traces:
        for row in rows:
            torch.cuda.synchronize()
            wall = time.perf_counter_ns()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            if ensemble is None:
                pred, diagnostic = measure_choice(native, row, window, plan, reverse=obj["reverse"],
                                                  policy=policy, split=split, **({"probe_margin":obj["probe_margin"]} if "probe_margin" in obj else {}))
            else:
                pred, diagnostic = _measure_ensemble(native, row, window, plan, split, second=row["group_id"] in selected)
                spend["forward_calls"] += diagnostic["forward_calls"]
                spend["layer_calls"] += diagnostic["layer_calls"]
                spend["coda_calls"] += diagnostic["callback_counts"]["coda_calls"]
            if frozen:
                diagnostic["packet_sha256"] = args.packet_sha256
            end.record()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns() - wall) / 1e6
            diagnostic.update(wall_ms=elapsed, cuda_event_ms=start.elapsed_time(end))
            times.append(elapsed)
            for name, f, rec in zip(names, (predictions, traces), (pred, diagnostic), strict=True):
                line = (json.dumps(rec, allow_nan=False, separators=(",", ":")) + "\n").encode("ascii")
                f.write(line)
                if frozen:
                    digests[name].update(line)
                f.flush()
                os.fsync(f.fileno())
    duration = time.perf_counter() - t0
    ordered_times = sorted(times)
    info = {"schema": 1, "packet_sha256": args.packet_sha256, "batch_size": 1, "decisions": len(rows),
            "parity": asdict(report), "hardware": "A100", "total_device_bytes": props.total_memory,
            "warmup_load_and_parity_s": warmup_s, "wall_including_persistence_s": duration,
            "decisions_per_wall_s": len(rows)/duration,
            "warm_latency_ms": {str(p): ordered_times[math.ceil(len(times)*p/100)-1] for p in (50,95,99)},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "steady_allocated_bytes": torch.cuda.memory_allocated(), "steady_reserved_bytes": torch.cuda.memory_reserved()}
    if "compact_manifest_sha256" in obj:
        info.update(compact_manifest_sha256=obj["compact_manifest_sha256"], reload_parity=asdict(native.reload_report),
                    physical_layers=len(native.physical_to_source))
    if frozen:
        info.update((k, obj[k]) for k in ("split", "inputs_sha256", "permit_sha256", "method_sha256"))
        info["result_sha256"] = {name: digest.hexdigest() for name, digest in digests.items()}
    if gate is not None:
        info.update(gate.evidence())
    if phase==PROBE_MODE:
        info["probe_settings"]={k:v for k,v in obj.items() if k in _SETTINGS}
    if ensemble is not None:
        info["compute_spend"] = spend
    for f, name in zip(outputs, names, strict=True):
        f.rename(args.output/name)
    with (args.output/"complete.json").open("x") as f:
        json.dump(info, f, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())


@contextmanager
def _quiet_native():
    # Native libraries can print outside Python's sys.stderr. Jobs return results
    # through their authorized private volume, not through shared container logs.
    sys.stdout.flush()
    sys.stderr.flush()
    backup = os.dup(1), os.dup(2)
    try:
        with open(os.devnull, "wb") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        for destination, original in enumerate(backup, start=1):
            os.dup2(original, destination)
            os.close(original)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("packet", "inputs", "model-dir", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--compact-dir", type=Path)
    p.add_argument("--packet-sha256", required=True)
    p.add_argument("--selection", action="store_true")
    p.add_argument("--nomination-sha256")
    p.add_argument("--phase", choices=tuple(PHASES))
    for name in ("phase-permit", "run", "view", "approval", "final-freeze"):
        p.add_argument("--" + name + "-sha256")
    p.add_argument("--admit-before", type=int)
    p.add_argument("--hard-stop", type=int)
    args = p.parse_args(argv)
    try:
        with _quiet_native():
            run(args)
    except (OSError, ValueError, RuntimeError, KeyError, ImportError, TypeError) as e:
        # Never log the prompt, provider message, hostname, path or local bindings.
        p.exit(2, f"worker refused trial ({type(e).__name__}); incomplete output is not scoreable\n")


if __name__ == "__main__":
    main()
