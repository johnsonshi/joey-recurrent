"""Offline SVG figures from one released aggregate group. See docs/decision-figures.md."""

import argparse
import json
import math
import re
from bisect import bisect_right
from pathlib import Path
from xml.etree import ElementTree as ET


THRESHOLDS = ("0.80", "0.90", "0.95", "0.99")
CUTS = tuple(i / 15 for i in range(1, 15))
MAX_INT = 2**53 - 1
COLORS = ("#006aa6", "#b54915", "#00765c", "#7453a0", "#916400", "#ac4178")
LABEL = re.compile(r"""[A-Za-z0-9 .,_+'"&<>()-]{1,24}""")
MODES = {"ordinary": "Ordinary", "stock_split": "Stock split"}


def need(ok, where, message):
    if not ok:
        raise ValueError(f"{where}: {message}")


def fields(v, where, required, optional=""):
    req = set(required.split())
    need(type(v) is dict and req <= v.keys() <= req.union(optional.split()),
         where, "missing or unexpected fields")


def count(v, where, lo=0, hi=MAX_INT):
    need(type(v) is int and lo <= v <= hi, where, f"expected integer in [{lo}, {hi}]")
    return v


def number(v, where, lo=0, hi=1):
    # Bounds first: don't coerce a huge Python integer to float.
    need(type(v) in (float, int) and lo <= v <= hi and math.isfinite(v),
         where, f"expected finite number in [{lo}, {hi}]")
    return v


def array(v, where, lo, hi):
    need(type(v) is list and lo <= len(v) <= hi, where, f"expected {lo}..{hi} entries")
    return v


def label(v, where):
    need(type(v) is str and LABEL.fullmatch(v) and v == v.strip()
         and any(c.isalnum() for c in v), where, "expected short public display label")


def choice(v, where, options):
    need(type(v) is str and v in options, where, "unknown category")


def accepted(v, where, n):
    fields(v, where, "accepted wrong")
    a = count(v["accepted"], where + ".accepted", hi=n)
    return a, count(v["wrong"], where + ".wrong", hi=a)


def validate_confidence(r, d, w):
    n, correct = d["n"], r["correct"]
    bins = {}
    if "bins" in r:
        for b in array(r["bins"], w + ".bins", 0, 15):
            bw = w + ".bins[]"
            fields(b, bw, "index n", "correct mean_confidence accuracy_interval")
            i = count(b["index"], bw + ".index", hi=14)
            size = count(b["n"], bw + ".n", hi=n)
            need(i not in bins, bw, "duplicate bin")
            bins[i] = b
            if not size:
                fields(b, bw, "index n")
                continue
            fields(b, bw, "index n correct mean_confidence", "accuracy_interval")
            count(b["correct"], bw + ".correct", hi=size)
            mean = number(b["mean_confidence"], bw + ".mean_confidence")
            need(bisect_right(CUTS, mean) == i, bw, "mean confidence outside bin")
            if "accuracy_interval" in b:
                lo, hi = array(b["accuracy_interval"], bw + ".accuracy_interval", 2, 2)
                need("interval_level" in d, bw, "interval requires interval_level")
                number(lo, bw + ".lower")
                number(hi, bw + ".upper", lo=lo)
        hits = sum(b.get("correct", 0) for b in bins.values())
        total = sum(b["n"] for b in bins.values())
        need(hits <= correct and total - hits <= n - correct, w, "bin counts exceed marginals")
        need(len(bins) < 15 or total == n, w, "complete bins must cover the split")

    if "thresholds" in r:
        ts = r["thresholds"]
        fields(ts, w + ".thresholds", " ".join(THRESHOLDS))
        last_hits, last_errors = correct, n - correct
        for t in THRESHOLDS:
            a, e = accepted(ts[t], w + ".thresholds." + t, n)
            need(a - e <= last_hits and e <= last_errors, w, "threshold subsets not nested")
            last_hits, last_errors = a - e, e

    if "ties" in r:
        last = 2
        mass, errors, weighted = [0] * 15, [0] * 15, [0.] * 15
        totals = {t: [0, 0] for t in THRESHOLDS}
        for g in array(r["ties"], w + ".ties", 1, n):
            gw = w + ".ties[]"
            fields(g, gw, "confidence n wrong")
            score = number(g["confidence"], gw + ".confidence")
            size = count(g["n"], gw + ".n", 1, n)
            e = count(g["wrong"], gw + ".wrong", hi=size)
            need(score < last, gw, "require one whole tie per distinct score, descending")
            last = score
            i = bisect_right(CUTS, score)
            mass[i] += size
            errors[i] += e
            weighted[i] += size * score
            for t in THRESHOLDS:
                if score >= float(t):
                    totals[t][0] += size
                    totals[t][1] += e
        need(sum(mass) == n and sum(errors) == n - correct, w, "tie totals disagree with split")
        for i, b in bins.items():
            need((mass[i], mass[i] - errors[i]) == (b["n"], b.get("correct", 0)), w,
                 "bin counts disagree with ties")
            if mass[i]:
                need(math.isclose(weighted[i] / mass[i], b["mean_confidence"], rel_tol=0,
                                  abs_tol=1e-9), w, "bin mean disagrees with ties")
        for t, row in r.get("thresholds", {}).items():
            need(totals[t] == [row["accepted"], row["wrong"]], w, "threshold disagrees with ties")


def validate(d):
    fields(d, "group", "version split projection n depth_unit series", "timing interval_level")
    count(d["version"], "version", 1, 1)
    n = count(d["n"], "n", 1)
    for key in ("split", "projection"):
        label(d[key], key)
    choice(d["depth_unit"], "depth_unit", ("passes", "layers", "blocks"))
    if "interval_level" in d:
        level = number(d["interval_level"], "interval_level")
        need(0 < level < 1, "interval_level", "must be strictly between 0 and 1")
    if "timing" in d:
        tm = d["timing"]
        fields(tm, "timing", "statistic per batch_size")
        choice(tm["statistic"], "timing.statistic", ("mean", "median", "p95"))
        choice(tm["per"], "timing.per", ("batch", "decision"))
        count(tm["batch_size"], "timing.batch_size", 1, 1_000_000)

    seen, families = set(), {}
    for j, s in enumerate(array(d["series"], "series", 1, len(COLORS))):
        w = f"series[{j}]"
        fields(s, w, "model family passes", "path adaptive")
        for k in ("model", "family"):
            label(s[k], w + "." + k)
        if "path" in s:
            choice(s["path"], w + ".path", MODES)
        key = s["model"], s.get("path")
        need(key not in seen and families.setdefault(key[0], s["family"]) == s["family"],
             w, "duplicate series or inconsistent family")
        seen.add(key)
        if "adaptive" in s:
            accepted(s["adaptive"], w + ".adaptive", n)
        prev = None
        for k, r in enumerate(array(s["passes"], w + ".passes", 0 if "adaptive" in s else 1, 32)):
            rw = f"{w}.passes[{k}]"
            fields(r, rw, "pass depth correct",
                   "brier bins thresholds ties from_previous cuda_ms stored_bytes")
            p = count(r["pass"], rw + ".pass", 1, 1_000_000)
            depth = count(r["depth"], rw + ".depth", 1, 1_000_000)
            c = count(r["correct"], rw + ".correct", hi=n)
            need(prev is None or (p > prev["pass"] and depth > prev["depth"]), rw,
                 "passes and completed depths must increase")
            need(d["depth_unit"] != "passes" or p == depth, rw, "pass depth must equal pass")
            if "brier" in r:
                number(r["brier"], rw + ".brier", hi=2)
            if "cuda_ms" in r:
                number(r["cuda_ms"], rw + ".cuda_ms", 1e-9, 1e12)
                need("timing" in d, rw, "measured CUDA time requires group timing settings")
            if "stored_bytes" in r:
                count(r["stored_bytes"], rw + ".stored_bytes", 1)
            validate_confidence(r, d, rw)
            if "from_previous" in r:
                tr = r["from_previous"]
                tw = rw + ".from_previous"
                fields(tr, tw, "matched corrections corruptions", "overthought")
                need(prev is not None and p == prev["pass"] + 1, tw, "requires adjacent pass")
                count(tr["matched"], tw + ".matched", n, n)
                before = prev["correct"]
                gain = count(tr["corrections"], tw + ".corrections", hi=n - before)
                loss = count(tr["corruptions"], tw + ".corruptions", hi=before)
                need(before + gain - loss == c, tw, "matched counts disagree with marginals")
                if "overthought" in tr:
                    h = count(tr["overthought"], tw + ".overthought", loss, n - c)
                    old = 0 if p == 2 else prev.get("from_previous", {}).get("overthought")
                    if old is not None:
                        need(loss + max(0, old - gain) <= h <= loss + old, tw,
                             "overthinking inconsistent with earlier history")
            prev = r


# A small SVG vocabulary. ElementTree escapes every text node; no input becomes markup.
WIDTH, PAD, PLOT_H = 1060, 30, 186
INK, MUTED, GRID = "#172f40", "#415867", "#d6e0e4"
PCT = [(i / 4, str(i * 25)) for i in range(5)]


def element(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k.replace("_", "-"): str(v) for k, v in attrs.items()})


def text(parent, x, y, value, size=14, **attrs):
    node = element(parent, "text", x=x, y=y, font_size=size, **attrs)
    node.text = str(value)
    return node


def line(parent, x1, y1, x2, y2, color, **attrs):
    return element(parent, "line", x1=f"{x1:.3f}", y1=f"{y1:.3f}",
                   x2=f"{x2:.3f}", y2=f"{y2:.3f}", stroke=color, **attrs)


def tip(node, value):
    element(node, "title").text = value


def mark(parent, x, y, color, shape="circle", kind="point", info=None):
    a = {"fill": color, "stroke": "white", "stroke_width": 1, "class": kind}
    if shape == "diamond":
        node = element(parent, "path", d=f"M{x:.3f},{y-6:.3f}l6,6 -6,6 -6,-6Z", **a)
    elif shape == "square":
        node = element(parent, "rect", x=f"{x-4:.3f}", y=f"{y-4:.3f}", width=8, height=8, **a)
    else:
        node = element(parent, "circle", cx=f"{x:.3f}", cy=f"{y:.3f}", r=4.5, **a)
    if info is not None:
        tip(node, info)


def pct(fraction):
    return f"{100 * fraction:.4g}%"


def compact(v):
    for scale, suffix in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(v) >= scale:
            return f"{v/scale:.3g}{suffix}"
    return f"{v:.4g}"


def name(s):
    return s["model"] + (" · " + MODES[s["path"]] if "path" in s else "")


class Figure:
    def __init__(self, title, d):
        self.svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg", role="img",
                              fill=INK, attrib={"font-family": "sans-serif"})
        tip(self.svg, title)
        element(self.svg, "rect", width="100%", height="100%", fill="white")
        text(self.svg, PAD, 38, title, 26, font_weight="bold")
        text(self.svg, WIDTH-PAD, 38, f"Matched N = {d['n']:,}", 15, text_anchor="end")
        text(self.svg, PAD, 66, f"Split: {d['split']}", 16)
        text(self.svg, WIDTH//2, 66, f"Projection: {d['projection']}", 16)
        line(self.svg, PAD, 82, WIDTH-PAD, 82, GRID)
        self.y = 92

    def reserve(self, h):
        top = self.y
        self.y += h
        return top

    def note(self, value, bold=False):
        text(self.svg, PAD, self.reserve(25) + 16, value, font_weight="bold" if bold else "normal")

    def legend(self, items, cols=2):
        for i, (caption, color, shape) in enumerate(items):
            if i % cols == 0:
                top = self.reserve(26)
            x = PAD + (i % cols) * ((WIDTH - 2*PAD) / cols)
            mark(self.svg, x+6, top+11, color, shape, "legend-key")
            text(self.svg, x+20, top+16, caption)

    def row(self, xs, values, header=False):
        top = self.reserve(24)
        group = element(self.svg, "g", **{"class": "table-header" if header else "table-row"})
        if header:
            element(group, "rect", x=xs[0]-5, y=top, width=WIDTH-PAD-xs[0]+10, height=24,
                    fill="#eaf0f2")
        for x, v in zip(xs, values):
            short, detail = v if isinstance(v, tuple) else (v, None)
            cell = text(group, x, top+17, short, 13, font_weight="bold" if header else "normal")
            if detail:
                tip(cell, detail)

    def finish(self):
        h = self.y + 16
        self.svg.attrib.update(width=str(WIDTH), height=str(h), viewBox=f"0 0 {WIDTH} {h}")
        return ET.tostring(self.svg, encoding="utf-8", xml_declaration=True)


def discrete(values):
    values = sorted(set(values))
    lo, hi = values[0], values[-1]
    if lo == hi:
        return (lo-1, hi+1), [(lo, str(lo))]
    # Use only measured ticks; don't crowd nearby depths on irregular grids.
    step = (hi-lo) / 5
    ticks = [lo]
    for v in values[1:-1]:
        if min(v-ticks[-1], hi-v) >= step:
            ticks.append(v)
    return (lo, hi), [(v, str(v)) for v in ticks + [hi]]


def error_scale(rates):
    """Keep rare errors legible; share the zero-based scale across comparable panels."""
    peak = max(rates, default=0)
    if not peak:
        return 1, PCT
    unit = 10.0 ** math.floor(math.log10(peak))
    high = min(1, next(k*unit for k in (1, 2, 5, 10) if k*unit >= peak))
    return high, [(high*i/4, f"{100*high*i/4:.4g}") for i in range(5)]


class Axes:
    """A linear plotting area; all chart code supplies data coordinates."""
    def __init__(self, f, top, col, cols, x_axis, xlabel, ylabel, ymax=1, yticks=PCT, height=PLOT_H):
        self.svg = element(f.svg, "g", role="group", **{"class": "axes", "aria-label": f"{ylabel} by {xlabel}"})
        cell = (WIDTH-2*PAD) / cols
        margin = max(70, 38 + 8*max(len(lab) for _, lab in yticks))
        self.left = PAD + col*cell + margin
        self.width = cell - margin - 22
        self.height = height
        self.bottom = top + 12 + height
        (self.lo, self.hi), xticks = x_axis
        self.ymax = ymax
        for v, lab in yticks:
            y = self.y(v)
            line(self.svg, self.left, y, self.left+self.width, y, GRID)
            text(self.svg, self.left-9, y+4, lab, 12, text_anchor="end", fill=MUTED)
        for v, lab in xticks:
            x = self.x(v)
            line(self.svg, x, self.bottom, x, self.bottom+5, MUTED)
            text(self.svg, x, self.bottom+20, lab, 12, text_anchor="middle", fill=MUTED)
        line(self.svg, self.left, self.bottom-height, self.left, self.bottom, MUTED)
        line(self.svg, self.left, self.bottom, self.left+self.width, self.bottom, MUTED)
        text(self.svg, self.left+self.width/2, self.bottom+42, xlabel, text_anchor="middle")
        x, y = PAD+col*cell+18, self.bottom-height/2
        text(self.svg, x, y, ylabel, text_anchor="middle", transform=f"rotate(-90 {x} {y})")

    def x(self, v):
        return self.left + (v-self.lo)/(self.hi-self.lo)*self.width

    def y(self, v):
        return self.bottom - (v/self.ymax)*self.height

    def series(self, points, color, kind):
        path, marks = [], []
        cmd = "M"
        for x, y, info in points:
            if y is None:
                cmd = "M"
                continue
            px, py = self.x(x), self.y(y)
            path.append(f"{cmd}{px:.3f},{py:.3f}")
            marks.append((px, py, info))
            cmd = "L"
        if path:
            element(self.svg, "path", d=" ".join(path), fill="none", stroke=color,
                    stroke_width=2, **{"class": kind + "-line"})
        for px, py, info in marks:
            mark(self.svg, px, py, color, kind=kind, info=info)


PLOT_ROW = PLOT_H + 70
UNIT_AXIS = ((0, 1), PCT)


def indexed(d):
    return [(s, COLORS[i]) for i, s in enumerate(d["series"])]


def model_key(f, series):
    f.legend([(name(s), color, "circle") for s, color in series])


def reliability(d):
    runs = [(s, color, r) for s, color in indexed(d) for r in s["passes"]
            if any(b["n"] for b in r.get("bins", []))]
    if not runs:
        return
    f = Figure("Top-label reliability", d)
    f.note("15 equal-width bins. Dashed line: ideal calibration. Support: 0 = empty; — = not released.")
    if any("accuracy_interval" in b for _, _, r in runs for b in r["bins"]):
        f.note(f"Whiskers: precomputed {pct(d['interval_level'])} paired-group bootstrap accuracy intervals, where supplied.")
    for s, color, r in runs:
        f.legend([(f"{name(s)}   ·   pass {r['pass']}   ·   depth {r['depth']} {d['depth_unit']}",
                   color, "circle")], 1)
        a = Axes(f, f.y, 0, 2, UNIT_AXIS, "Mean top-label confidence (%)", "Bin accuracy (%)",
                 height=302)
        line(a.svg, a.x(0), a.y(0), a.x(1), a.y(1), MUTED, stroke_dasharray="6 5",
             **{"class": "identity"})
        xs = (552, 607, 787, 918)
        f.row(xs, ("Bin", "Support (n)", "Mean conf.", "Accuracy"), True)
        bins = {b["index"]: b for b in r["bins"]}
        for i in range(15):
            b = bins.get(i, {})
            size = b.get("n")
            if not size:
                f.row(xs, (i+1, "—" if size is None else 0, "—", "—"))
                continue
            conf, acc = b["mean_confidence"], b["correct"] / size
            x = a.x(conf)
            if "accuracy_interval" in b:
                lo, hi = b["accuracy_interval"]
                whisker = line(a.svg, x, a.y(lo), x, a.y(hi), color, stroke_width=2,
                               **{"class": "accuracy-interval"})
                tip(whisker, f"Accuracy interval: {pct(lo)} to {pct(hi)}")
                for edge in (lo, hi):
                    line(a.svg, x-5, a.y(edge), x+5, a.y(edge), color, stroke_width=2)
            mark(a.svg, x, a.y(acc), color, kind="bin-accuracy",
                 info=f"Bin {i+1}: {b['correct']} / {size} correct")
            f.row(xs, (i+1, size, pct(conf), pct(acc)))
        f.reserve(12)
    return f.finish()


def depth_scores(d):
    series = [(s, c) for s, c in indexed(d) if s["passes"]]
    if not series:
        return
    axis = discrete(r["depth"] for s, _ in series for r in s["passes"])
    has_brier = any("brier" in r for s, _ in series for r in s["passes"])
    f = Figure("Decision quality by completed depth", d)
    f.note("Accuracy: correct / N. Multiclass Brier: mean sum of squared class-probability errors (0–2; lower is better).")
    if not has_brier:
        f.note("No Brier scores supplied for this group.")
    model_key(f, series)
    top = f.reserve(PLOT_ROW)
    xlabel = f"Completed depth ({d['depth_unit']})"
    a = Axes(f, top, 0, 2 if has_brier else 1, axis, xlabel, "Decision accuracy (%)")
    for s, color in series:
        a.series([(r["depth"], r["correct"]/d["n"], f"{r['correct']} / {d['n']} correct")
                  for r in s["passes"]], color, "depth-accuracy")
    if has_brier:
        b = Axes(f, top, 1, 2, axis, xlabel, "Multiclass Brier score", 2,
                 [(i/2, f"{i/2:g}") for i in range(5)])
        for s, color in series:
            b.series([(r["depth"], r.get("brier"), f"Brier: {r.get('brier')}")
                      for r in s["passes"]], color, "depth-brier")
    return f.finish()


def confident_wrong(d):
    series = [s for s, _ in indexed(d) if any("thresholds" in r for r in s["passes"])]
    if not series:
        return
    f = Figure("Confident errors by completed pass", d)
    f.note("Left: wrong and confident / N. Right: wrong / accepted. No conditional point when none are accepted.")
    f.note("Each rate has its own labeled scale, shared across models.")
    f.legend([(f"Confidence ≥ {t}", c, "circle") for t, c in zip(THRESHOLDS, COLORS)], 4)
    n = d["n"]
    axis = discrete(r["pass"] for s in series for r in s["passes"])
    values = [v for s in series for r in s["passes"] for v in r.get("thresholds", {}).values()]
    joint_scale = error_scale(v["wrong"]/n for v in values)
    conditional_scale = error_scale(v["wrong"]/v["accepted"] for v in values if v["accepted"])
    for s in series:
        f.note(name(s), bold=True)
        top = f.reserve(PLOT_ROW)
        a = Axes(f, top, 0, 2, axis, "Completed passes", "Wrong / all decisions (%)", *joint_scale)
        b = Axes(f, top, 1, 2, axis, "Completed passes", "Wrong / accepted (%)", *conditional_scale)
        for t, color in zip(THRESHOLDS, COLORS):
            joint, conditional = [], []
            for r in s["passes"]:
                v = r.get("thresholds", {}).get(t)
                if joint and r["pass"] != joint[-1][0]+1:
                    joint.append((r["pass"]-1, None, ""))
                    conditional.append((r["pass"]-1, None, ""))
                joint.append((r["pass"], v["wrong"]/n if v else None,
                              f"{v['wrong']} / {n} wrong and confident" if v else ""))
                supported = v and v["accepted"] > 0
                conditional.append((r["pass"], v["wrong"]/v["accepted"] if supported else None,
                                    f"{v['wrong']} wrong / {v['accepted']} accepted" if supported else ""))
            a.series(joint, color, "joint-error")
            b.series(conditional, color, "conditional-error")
        f.note("Cells: wrong count / accepted count (coverage = accepted / N). A 0 / 0 count pair has undefined conditional risk.")
        xs = (PAD, 129, 361, 593, 825)
        f.row(xs, ("Pass", *("≥ " + t for t in THRESHOLDS)), True)
        for r in s["passes"]:
            cells = []
            for t in THRESHOLDS:
                v = r.get("thresholds", {}).get(t)
                if v is None:
                    cells.append("—")
                    continue
                e, a_n = v["wrong"], v["accepted"]
                exact = f"{e} / {a_n} ({pct(a_n/n)})"
                if not a_n:
                    cells.append("0 / 0 (0%); undefined")
                else:
                    cells.append(exact if len(exact) <= 24 else
                                 (f"{compact(e)} / {compact(a_n)} ({pct(a_n/n)})", exact))
            f.row(xs, (r["pass"], *cells))
        f.reserve(10)
    return f.finish()


def transitions(d):
    series = [s for s, _ in indexed(d) if any("from_previous" in r for r in s["passes"])]
    if not series:
        return
    f = Figure("Matched pass transitions", d)
    f.note("Bars use the same 0–N scale. Labels: count and share of matched decisions. — = not released.")
    f.note("Overthinking includes adjacent corruptions: correct at any earlier pass, wrong now. Do not add these columns.")
    xs = (PAD, 153, 452, 751)
    for s in series:
        f.note(name(s), bold=True)
        f.row(xs, ("Pass pair", "Corrections: wrong → right", "Corruptions: right → wrong",
                   "Overthinking: earlier right → wrong"), True)
        for r in s["passes"]:
            if "from_previous" not in r:
                continue
            tr = r["from_previous"]
            y = f.reserve(32)+17
            text(f.svg, PAD, y, f"{r['pass']-1} → {r['pass']}", 13)
            for x, k, color in zip(xs[1:], ("corrections", "corruptions", "overthought"),
                                   (COLORS[2], COLORS[1], COLORS[3])):
                if k not in tr:
                    text(f.svg, x, y, "—")
                    continue
                v = tr[k]
                share = v / tr["matched"]
                element(f.svg, "rect", x=x, y=y-10, width=96, height=9, fill=GRID)
                element(f.svg, "rect", x=x, y=y-10, width=f"{96*share:.4f}", height=9,
                        fill=color, **{"class": k})
                text(f.svg, x+104, y, f"{v} ({pct(share)})", 13)
    return f.finish()


def cost_accuracy(d, field):
    series = [(i+1, s) for i, s in enumerate(d["series"]) if any(field in r for r in s["passes"])]
    if not series:
        return
    timing = field == "cuda_ms"
    f = Figure("Accuracy versus measured CUDA time" if timing else "Accuracy versus physical stored bytes", d)
    if timing:
        t = d["timing"]
        xlabel = f"{t['statistic'].capitalize()} measured CUDA time (ms/{t['per']})"
        f.note(f"Matched timing settings   ·   Batch size: {t['batch_size']}   ·   Each point uses accuracy at the timed setting.")
    else:
        xlabel = "Physical stored bytes (B)"
        f.note("Stored artifact bytes, not active parameters or peak VRAM. Ticks use decimal prefixes; the table gives exact bytes.")
    f.note("Color: model family. Shape: circle = ordinary; diamond = stock split; square = execution path not supplied.")
    families = dict(zip(dict.fromkeys(s["family"] for s in d["series"]), COLORS))
    shapes = {None: "square", "ordinary": "circle", "stock_split": "diamond"}
    f.legend([(f"{i}. {name(s)}   ·   {s['family']}", families[s["family"]], shapes[s.get("path")])
              for i, s in series], 1)
    high = max(r[field] for _, s in series for r in s["passes"] if field in r)
    ticks = sorted({high*i/4 if timing else high*i//4 for i in range(5)})
    a = Axes(f, f.reserve(PLOT_ROW), 0, 1, ((0, high), [(v, compact(v)) for v in ticks]),
             xlabel, "Decision accuracy (%)")
    xs = (PAD, 128, 254, 490, 728, 945)
    f.row(xs, ("Series", "Pass", f"Depth ({d['depth_unit']})", f"ms/{t['per']}" if timing else "Bytes (B)",
               "Correct", "Accuracy"), True)
    for i, s in series:
        for r in s["passes"]:
            if field not in r:
                continue
            v, c = r[field], r["correct"]
            acc = c/d["n"]
            mark(a.svg, a.x(v), a.y(acc), families[s["family"]], shapes[s.get("path")],
                 "cost-point", f"Series {i}, pass {r['pass']}: {v} {'ms' if timing else 'B'}; {c} / {d['n']} correct")
            f.row(xs, (i, r["pass"], r["depth"], v, c, pct(acc)))
    return f.finish()


def tie_totals(ties):
    size = errors = 0
    for group in ties:
        size += group["n"]
        errors += group["wrong"]
        yield size, errors


def selective_risk(d):
    series = [(s, c) for s, c in indexed(d)
              if "adaptive" in s or any("ties" in r for r in s["passes"])]
    if not series:
        return
    f = Figure("Selective risk versus coverage", d)
    f.note("Coverage = accepted / N. Risk = wrong / accepted. Risk at zero coverage is undefined and is never plotted.")
    f.note("Fixed-pass dots admit whole confidence ties in descending order. No interpolation or within-tie ordering.")
    model_key(f, series)
    fixed, adaptive = {}, []
    for s, color in series:
        for r in s["passes"]:
            if "ties" in r:
                fixed.setdefault(r["pass"], []).append((color, r))
        if "adaptive" in s:
            adaptive.append((s, color))
    panels = sorted(fixed)
    policies = [s["adaptive"] for s, _ in adaptive if s["adaptive"]["accepted"]]
    if policies:
        panels.append(None)
    peaks = [max(e/n for n, e in tie_totals(r["ties"])) for runs in fixed.values() for _, r in runs]
    risk_scale = error_scale(peaks + [v["wrong"]/v["accepted"] for v in policies])
    for offset in range(0, len(panels), 2):
        top = f.reserve(PLOT_ROW+26)
        cols = min(2, len(panels))
        for col, p in enumerate(panels[offset:offset+2]):
            x = PAD + col*((WIDTH-2*PAD)/cols)
            text(f.svg, x, top+15, "Observed adaptive policy" if p is None else f"Fixed pass {p}",
                 15, font_weight="bold")
            a = Axes(f, top+24, col, cols, UNIT_AXIS, "Coverage (%)", "Selective risk (%)", *risk_scale)
            if p is None:
                for s, color in adaptive:
                    v = s["adaptive"]
                    if v["accepted"]:
                        mark(a.svg, a.x(v["accepted"]/d["n"]), a.y(v["wrong"]/v["accepted"]),
                             color, "diamond", "adaptive-point", f"{v['wrong']} wrong / {v['accepted']} accepted")
                continue
            for color, r in fixed[p]:
                dots = [f"M{a.x(size/d['n']):.3f},{a.y(errors/size):.3f}h0"
                        for size, errors in tie_totals(r["ties"])]
                # Round caps on disjoint zero-length segments: one dot per attainable coverage.
                node = element(a.svg, "path", d=" ".join(dots), fill="none", stroke=color,
                               stroke_width=5, stroke_linecap="round", **{"class": "empirical-ties"})
                tip(node, f"{len(dots)} whole ties; full coverage: {d['n']-r['correct']} / {d['n']} wrong")
    if adaptive:
        f.note("Actual adaptive policy outcomes (not points selected from the fixed-pass curves).", bold=True)
        xs = (PAD, 432, 595, 733, 870)
        f.row(xs, ("Model / path", "Accepted", "Wrong", "Coverage", "Conditional risk"), True)
        for s, _ in adaptive:
            v = s["adaptive"]
            size, e = v["accepted"], v["wrong"]
            f.row(xs, (name(s), size, e, pct(size/d["n"]), pct(e/size) if size else "— (undefined)"))
    return f.finish()


def render_evidence(data) -> dict[str, bytes]:
    """Validate one v1 group; return only supported views, without file I/O or mutation."""
    validate(data)
    charts = (("reliability", reliability), ("depth", depth_scores), ("confident_wrong", confident_wrong),
              ("transitions", transitions), ("cuda_time", lambda d: cost_accuracy(d, "cuda_ms")),
              ("storage", lambda d: cost_accuracy(d, "stored_bytes")), ("selective_risk", selective_risk))
    return {key + ".svg": svg for key, render in charts if (svg := render(data)) is not None}


def load_summary(stream):
    """Read strict JSON. Ordinary json.load silently accepts repeated keys and NaN."""
    def unique(pairs):
        obj = {}
        for k, v in pairs:
            need(k not in obj, "JSON", "duplicate object key")
            obj[k] = v
        return obj

    def invalid(_):
        raise ValueError("JSON: non-finite number")

    return json.load(stream, object_pairs_hook=unique, parse_constant=invalid)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="released v1 aggregate JSON")
    parser.add_argument("output", type=Path, help="new directory (must not already exist)")
    args = parser.parse_args(argv)
    try:
        with args.summary.open(encoding="utf-8") as stream:
            figures = render_evidence(load_summary(stream))
        args.output.mkdir()
        for filename, svg in figures.items():
            with (args.output / filename).open("xb") as stream:
                stream.write(svg)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
