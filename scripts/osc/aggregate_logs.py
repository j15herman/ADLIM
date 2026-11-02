"""
Seed-aggregate plasticity_mlp / baseline sweep logs: for each distinct config,
average EVERY statistic that appears in the per-run logs across its seeds, and
re-print them in (almost) the same format as a single run -- so a 5-seed config
reads like one run whose numbers are the seed means.

What it reads. Each array task writes `logs/slurm_<job>_<N>.out` (the [INFO]
logging lines: Accuracy, Raw accuracy, and the `plast NCM ORACLE / PROBE /
GEO PROBE / DEATH / TURNOVER` probes) and a sibling `.out`-mate `.err` (the
stderr-only Forgetting / Backward Transfer / Forward Transfer lines). Config
identity is the `Run:  <args>` line the sbatch echoes into the .out, with
`--seed N` stripped -- so the 5 files that differ only in seed are one group.

What it prints.
  1. A ranked SUMMARY table: per config, final 5-task Class-IL and Task-IL as
     mean +/- std over seeds (this is the number the method-rule cares about),
     plus n (seeds found).
  2. A DETAIL block per config that reproduces every statistic line with its
     seed MEAN (identical format). Pass --std to append +/-std to every number.
  3. A LEGEND describing each statistic (also via --legend-only).

Generic by design: each stat line is split into a fixed text skeleton + its
numbers; numbers are averaged position-wise across seeds and substituted back,
so ANY statistic in the logs is carried through without being enumerated here
(digit-bearing labels like fc1/fc2 and the p[.1/.5/.9] percentile labels are
masked so they are never mistaken for data).

Usage:
    # a whole sweep's logs (a dir, a glob, or explicit files):
    python aggregate_logs.py ../../logs
    python aggregate_logs.py '../../logs/slurm_*_6131234_*.out'
    python aggregate_logs.py run1.out run2.out ...
    python aggregate_logs.py ../../logs --std          # mean +/- std everywhere
    python aggregate_logs.py --legend-only             # just the stat glossary
"""

import argparse
import glob
import math
import os
import re
import sys
from collections import OrderedDict

# A number: int / float / scientific, optionally signed. Masked label sentinels
# (below) carry no digits, so they are never matched here.
NUM = re.compile(r'-?\d*\.\d+(?:[eE][-+]?\d+)?|-?\d+(?:[eE][-+]?\d+)?')

# Only these lines carry statistics (anything else in the log is ignored).
STAT_MARKERS = (
    'Accuracy for', 'Raw accuracy values',
    'plast NCM ORACLE', 'plast PROBE', 'plast GEO PROBE', 'plast DEATH',
    'plast TURNOVER',
    'Backward Transfer', 'Forgetting', 'Forward Transfer',
    'Raw Backward Transfer', 'Raw Forgetting', 'Raw Forward Transfer',
)
# These repeat at every task boundary with no task index of their own, so they
# are tagged with the current boundary N (the last 'Accuracy for N task(s)').
BOUNDARY_TAGGED = (
    'Raw accuracy values', 'Backward Transfer', 'Forgetting', 'Forward Transfer',
    'Raw Backward Transfer', 'Raw Forgetting', 'Raw Forward Transfer',
)
# Digit-bearing LABELS (not data) -> masked to a digit-free sentinel before the
# numbers are pulled out, then restored verbatim when rendering.
MASKS = [('[.1/.5/.9]', '①'), ('fc1', '②'), ('fc2', '③')]

LEGEND = """\
STATISTIC GLOSSARY (every line the runs log; values shown are seed means)
  Accuracy for N task(s): [Class-IL] [Task-IL]
      The headline. Class-IL = accuracy with NO task id at test (the hard
      setting); Task-IL = accuracy given the task id. Printed after each of the
      N seen tasks; the N=5 row is the final result.
  Raw accuracy values: Class-IL [...] | Task-IL [...]
      Per-task accuracy vector behind that mean (entry t = acc on task t). The
      tell for head-collapse: [0,0,0,~s,~96] = only the last task survives.
  Backward Transfer
      Mean change in earlier tasks' accuracy after learning later ones
      (negative = forgetting; ~0 = retained).
  Forgetting
      Mean drop from each task's PEAK accuracy to its final (higher = worse).
  Forward Transfer
      Zero-shot benefit on not-yet-seen tasks vs a random net (>0 = helps).
  plast NCM ORACLE task T: [Class-IL] [Task-IL] | per-task CIL [...]
      Head-FREE ceiling: nearest-class-mean on the current fc2 features. The
      oracle-minus-head Class-IL gap localizes loss to the readout vs features.
  plast PROBE task T: <layer>: mean std frozen plastic
      Per-layer plasticity p map: mean/std, frozen fraction (p<freeze_thresh),
      fully-plastic fraction (p>=target). How you SEE freezing vs lost plasticity.
  plast GEO PROBE task T: deaths | <layer>: P+/-std barriers u_range eff_comp
      Dendrite geometry: per-neuron mean plasticity P, barriers/neuron, mean
      dendrite u-span, and EFFECTIVE compartments (kernel-disconnected breaks;
      ~1 = geometry inert). deaths = neurons recycled this task.
  plast DEATH task T: deaths | <layer>P
      Neuron-recycle count this task + per-layer mean plasticity.
  plast TURNOVER task T: <layer>: frozen ever new thaw p_froz p_plas p[.1/.5/.9] when
      frozen = currently frozen frac; ever = ever-frozen frac; new = froze FIRST
      this task; thaw = was frozen last boundary, plastic now (re-plasticized);
      p_froz/p_plas = mean p among frozen vs still-plastic (freeze depth vs how
      labile the rest are); p[.1/.5/.9] = plasticity percentiles; when = mean
      within-task position of this task's fresh freezes (0=early, 1=late).
"""


def strip_prefix(line):
    """Drop the '[INFO] DD-Mon-YY HH:MM:SS - ' logging prefix (absent on the
    stderr-only lines) and any leading tabs/spaces."""
    line = re.sub(r'^\[[A-Z]+\]\s+\d\d-[A-Za-z]+-\d\d\s+[\d:]+\s+-\s+', '', line)
    return line.strip().lstrip('\t ').strip()


def mask(s):
    for lit, sent in MASKS:
        s = s.replace(lit, sent)
    return s


def unmask(s):
    for lit, sent in MASKS:
        s = s.replace(sent, lit)
    return s


def parse_stat_line(raw, cur_n):
    """-> (key, skeleton, tokens) or None. key groups the same stat across seeds;
    skeleton is the line text with each number replaced by '{}'; tokens are the
    number strings (kept for decimal-faithful re-rendering)."""
    s = strip_prefix(raw)
    if not any(m in s for m in STAT_MARKERS):
        return None
    if ': ' not in s:
        return None
    key, rest = s.split(': ', 1)
    key = key.strip()
    if any(key.startswith(b) for b in BOUNDARY_TAGGED) and cur_n is not None:
        key = f'{key} @t{cur_n}'
    rest = mask(rest)
    tokens = NUM.findall(rest)
    if not tokens:
        return None
    skeleton = NUM.sub('{}', rest)
    return key, skeleton, tokens


# Short display names + a preferred ordering for the flags a label may show.
SHORT = {
    '--plast_importance_mode': 'mode', '--plast_beta': 'beta',
    '--plast_budget': 'budget', '--plast_budget_frac': 'frac',
    '--plast_lambda': 'lam', '--plast_lambda_end': 'lam_end',
    '--plast_layers': 'layers', '--plast_geo_n_barriers': 'nbar',
    '--plast_consistency': 'cons', '--plast_signal': 'sig',
    '--plast_normalize': 'norm', '--plast_p_min': 'pmin', '--plast_p_max': 'pmax',
    '--plast_accum_floor': 'floor', '--plast_accum_center': 'center',
    '--plast_consol_stiffness': 'stiff', '--plast_stiff_min': 'smin',
    '--plast_stiff_max': 'smax', '--plast_spread': 'spread',
    '--plast_spread_axis': 'spax', '--plast_spread_kernel': 'sker',
    '--plast_coact_thresh': 'cthr', '--plast_geometry': 'geo',
    '--plast_geo_sigma': 'gsig', '--plast_geo_barrier_height': 'gbh',
    '--plast_geo_barrier_width': 'gbw', '--plast_lr_schedule': 'lrsch',
    '--plast_lr_min': 'lrmin', '--plast_warmup_epochs': 'warm',
    '--plast_death': 'death', '--plast_death_thresh': 'dthr',
    '--plast_death_rate_cap': 'drate', '--e_lambda': 'e_lambda',
}
ORDER = ['--plast_importance_mode', '--plast_beta', '--plast_budget',
         '--plast_budget_frac', '--plast_lambda', '--plast_lambda_end',
         '--plast_layers', '--plast_geo_n_barriers', '--plast_consistency',
         '--plast_signal', '--plast_p_min', '--plast_accum_floor',
         '--plast_death', '--plast_consol_stiffness', '--plast_geometry',
         '--plast_geo_sigma', '--plast_spread', '--plast_normalize']


def parse_flags(args_line):
    """Args string -> {flag: value} (single-token values; valueless flags -> '')."""
    toks = args_line.split()
    d, i = {}, 0
    while i < len(toks):
        if toks[i].startswith('--'):
            if i + 1 < len(toks) and not toks[i + 1].startswith('--'):
                d[toks[i]] = toks[i + 1]; i += 2
            else:
                d[toks[i]] = ''; i += 1
        else:
            i += 1
    return d


def distinguishing_flags(plast_cfgs):
    """The plast flags that take >1 value across the plasticity-mlp configs --
    these are exactly what tells the runs apart, so the label shows all of them
    (ordered preferred-first). Empty -> a single config / nothing varies."""
    ds = [parse_flags(c) for c in plast_cfgs]
    if not ds:
        return []
    flags = set().union(*(set(d) for d in ds))
    varying = [f for f in flags
               if f not in ('--model', '--seed') and len({d.get(f) for d in ds}) > 1]
    return [f for f in ORDER if f in varying] + sorted(f for f in varying if f not in ORDER)


def make_labeler(varying):
    """Label = model + every distinguishing flag (so two runs never share a label
    unless they ARE the same config). Baselines get a plain model label."""
    def label(args_line):
        d = parse_flags(args_line)
        model = d.get('--model', '?')
        if model == 'mas':
            return f"mas e_lambda={d.get('--e_lambda')}"
        if model in ('sgd', 'si'):
            return model
        if model == 'plasticity-mlp' and d.get('--plast_mlp') == '0':
            return 'plasticity-mlp OFF'
        keys = varying or ['--plast_importance_mode', '--plast_beta',
                           '--plast_budget_frac', '--plast_lambda']
        parts = [f"{SHORT.get(k, k[2:].replace('plast_', ''))}={d.get(k, '-')}"
                 for k in keys]
        return 'plast ' + ' '.join(parts)
    return label


def _final_stat_key(lines, name):
    """The highest-task `<name> task T` key in lines (e.g. name='plast NCM ORACLE')."""
    best = None
    for k in lines:
        m = re.match(re.escape(name) + r' task (\d+)$', k)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), k)
    return best[1] if best else None


def _seed_mean_token(lines, key, idx, sel=None, n_full=None):
    """Seed-mean of the idx-th number on `key`'s line, or None if absent/ragged.
    If `sel` (seed indices) is given AND this key has one entry per seed
    (len == n_full), restrict the mean to those seeds; otherwise fall back to
    averaging every seed that reports the key (best effort when ragged)."""
    if not key or key not in lines:
        return None
    per = lines[key]
    if sel is not None and n_full is not None and len(per) == n_full:
        per = [per[i] for i in sel]
    try:
        return sum(float(t[idx]) for _, t in per) / len(per)
    except (IndexError, ValueError, ZeroDivisionError):
        return None


def collect_units(paths):
    """Expand dirs/globs to files, group .out/.err that share a stem into one
    'array unit' -> ordered list of its source files."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, '*.out')))
            files += sorted(glob.glob(os.path.join(p, '*.err')))
        else:
            files += sorted(glob.glob(p)) or ([p] if os.path.exists(p) else [])
    units = OrderedDict()
    for f in files:
        stem = re.sub(r'\.(out|err)$', '', f)
        units.setdefault(stem, [])
        # .out before .err so the [INFO] copy of a shared line wins on dedup.
        (units[stem].insert(0, f) if f.endswith('.out') else units[stem].append(f))
    return units


def parse_unit(unit_files):
    """-> (config_args_or_None, OrderedDict key -> (skeleton, tokens)). Dedupes a
    stat key within the unit (the .out [INFO] line and its stderr twin collapse)."""
    config = None
    stats = OrderedDict()
    cur_n = None
    for path in unit_files:
        try:
            with open(path, encoding='utf-8', errors='replace') as fh:
                lines = fh.readlines()
        except OSError:
            continue
        for raw in lines:
            mrun = re.match(r'^Run:\s+(.*\S)\s*$', raw)
            if mrun:
                config = re.sub(r'\s*--seed\s+\S+', '', mrun.group(1)).strip()
                continue
            mn = re.search(r'Accuracy for (\d+) task\(s\)', raw)
            if mn:
                cur_n = int(mn.group(1))
            parsed = parse_stat_line(raw, cur_n)
            if parsed is None:
                continue
            key, skeleton, tokens = parsed
            if key not in stats:           # first occurrence wins (dedupe out/err)
                stats[key] = (skeleton, tokens)
    return config, stats


def fmt_num(mean, ref_tok, std=None):
    """Format a mean like its reference token (same decimals), optional +/-std."""
    if 'e' in ref_tok or 'E' in ref_tok:
        body = f'{mean:.3g}'
        dec = 3
    else:
        dec = len(ref_tok.split('.')[1]) if '.' in ref_tok else 0
        body = f'{mean:.{dec}f}'
    if std is not None:
        body += f'+/-{std:.{max(dec, 2)}f}'
    return body


def mean_std(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, var ** 0.5


def render_line(key, per_seed, show_std):
    """per_seed: list of (skeleton, tokens) over seeds for this key. Average
    tokens position-wise and substitute back into the (shared) skeleton."""
    skeleton = per_seed[0][0]
    ref_tokens = per_seed[0][1]
    counts = {len(t) for _, t in per_seed}
    if len(counts) != 1:                   # ragged (shouldn't happen per key)
        return f'{key}: [varies across seeds: {sorted(counts)} numbers] ' \
               f'{unmask(skeleton).format(*ref_tokens)}'
    cols = list(zip(*[[float(x) for x in t] for _, t in per_seed]))
    pieces = []
    for j, col in enumerate(cols):
        m, sd = mean_std(list(col))
        pieces.append(fmt_num(m, ref_tokens[j], sd if show_std else None))
    return f'{key}: ' + unmask(skeleton).format(*pieces)


def final_acc(stats):
    """(cil_tokens, til_tokens) from the largest 'Accuracy for N task(s)' key, or
    None. Tokens are per-seed lists already merged at the call site."""
    best = None
    for key in stats:
        m = re.match(r'Accuracy for (\d+) task\(s\)$', key)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), key)
    return best[1] if best else None


def summary_row(c, mode):
    """One SUMMARY row for config c computed over a SUBSET of its seeds.
    mode in {'all','best','worst'}: rank the seeds by their final Class-IL and
    take all / the top half / the bottom half (with an odd seed count the median
    seed lands in both halves, so each is a true 50%). Returns the same tuple the
    summary table consumes: ((cil_m,cil_s),(til_m,til_s),orc,frz,n_sel,label)."""
    fk = final_acc(c['lines'])
    if fk is None:
        return None
    per_seed = c['lines'][fk]
    cil = [float(t[0]) for _, t in per_seed]
    til = [float(t[1]) for _, t in per_seed] if len(per_seed[0][1]) > 1 else cil
    n = len(cil)
    order = sorted(range(n), key=lambda i: -cil[i])     # best (high CIL) first
    half = math.ceil(n / 2)
    sel = {'all': list(range(n)), 'best': order[:half],
           'worst': order[-half:]}[mode]
    cil_sel = [cil[i] for i in sel]
    til_sel = [til[i] for i in sel]
    # oracle/fc1frz restricted to the same seeds (idx 0 on ORACLE, idx 1 on
    # TURNOVER: frozen, EVER, new, thaw, ...), at the last task.
    orc = _seed_mean_token(c['lines'], _final_stat_key(c['lines'], 'plast NCM ORACLE'), 0, sel, n)
    frz = _seed_mean_token(c['lines'], _final_stat_key(c['lines'], 'plast TURNOVER'), 1, sel, n)
    return (mean_std(cil_sel), mean_std(til_sel), orc, frz, len(sel), c['label'])


def print_summary(rows, title):
    """Print one ranked SUMMARY table (rows are summary_row tuples)."""
    rows = sorted(rows, key=lambda r: -r[0][0])
    print('=' * 104)
    print(title)
    print('  oracle = head-free NCM Class-IL ceiling on fc2 feats; fc1frz = fc1 '
          'ever-frozen fraction (final task)')
    print('=' * 104)
    print(f'{"Class-IL":>13} {"Task-IL":>13} {"oracle":>7} {"fc1frz":>7} {"n":>2}  config')
    for (cm, cs), (tm, ts), orc, frz, n, label in rows:
        orc_s = f'{orc:6.1f}' if orc is not None else '     -'
        frz_s = f'{frz:6.3f}' if frz is not None else '     -'
        print(f'{cm:6.2f}+/-{cs:4.2f} {tm:6.2f}+/-{ts:4.2f} {orc_s} {frz_s} {n:>2}  {label}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('paths', nargs='*', default=['logs'],
                    help='log files, globs, or dirs (default: logs/).')
    ap.add_argument('--std', action='store_true',
                    help='append +/-std (over seeds) to every averaged number.')
    ap.add_argument('--legend-only', action='store_true',
                    help='print only the statistic glossary and exit.')
    ap.add_argument('--no-legend', action='store_true',
                    help='omit the trailing glossary.')
    args = ap.parse_args()

    if args.legend_only:
        print(LEGEND)
        return

    # config -> { 'label':, 'lines': OrderedDict(key -> list of (skel, tokens)),
    #             'nseeds': int }
    configs = OrderedDict()
    for stem, unit_files in collect_units(args.paths).items():
        cfg, stats = parse_unit(unit_files)
        if not stats:
            continue
        cfg = cfg or f'[no Run line] {os.path.basename(stem)}'
        c = configs.setdefault(cfg, {'label': '', 'lines': OrderedDict(),
                                     'nseeds': 0})
        c['nseeds'] += 1
        for key, (skel, tokens) in stats.items():
            c['lines'].setdefault(key, []).append((skel, tokens))

    if not configs:
        print('No stat lines found. Point me at the sweep .out files (and their '
              '.err siblings).', file=sys.stderr)
        sys.exit(1)

    # Label each config by the flags that VARY across the plasticity-mlp runs, so
    # runs differing only in (say) beta are distinguishable in the table.
    plast_cfgs = [cfg for cfg in configs
                  if parse_flags(cfg).get('--model') == 'plasticity-mlp'
                  and parse_flags(cfg).get('--plast_mlp') != '0']
    labeler = make_labeler(distinguishing_flags(plast_cfgs))
    for cfg, c in configs.items():
        c['label'] = labeler(cfg)

    # ---- summary tables: ranked by final Class-IL, with extra distinguishing
    # columns (head-free NCM-oracle ceiling + fc1 ever-frozen fraction). The same
    # table is printed three times -- over ALL seeds, then over each config's
    # best-50% and worst-50% of seeds (seeds ranked within the config by final
    # Class-IL) -- so the seed spread of every config is visible at a glance. ----
    nc = len(configs)
    for mode, title in [
        ('all', f'SEED-AGGREGATED SUMMARY  ({nc} configs)   '
                'ranked by final Class-IL (mean+/-std over ALL seeds)'),
        ('best', f'BEST-50% SEEDS SUMMARY  ({nc} configs)   '
                 'each config over its top-half seeds by Class-IL (mean+/-std)'),
        ('worst', f'WORST-50% SEEDS SUMMARY  ({nc} configs)   '
                  'each config over its bottom-half seeds by Class-IL (mean+/-std)'),
    ]:
        rows = [r for r in (summary_row(c, mode) for c in configs.values()) if r]
        print_summary(rows, title)
        print()

    # ---- per-config detail: every statistic, seed mean (same format) ----
    for cfg, c in configs.items():
        print('\n' + '-' * 96)
        print(f'{c["label"]}   (n={c["nseeds"]} seeds)')
        print(f'  args: {cfg}')
        print('-' * 96)
        for key, per_seed in c['lines'].items():
            print('  ' + render_line(key, per_seed, args.std))

    if not args.no_legend:
        print('\n' + '=' * 96)
        print(LEGEND)


if __name__ == '__main__':
    main()
