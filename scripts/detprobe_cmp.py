"""Compare determinism-probe dumps (ds41h/detprobe.py) of repeated forwards of the same prompt.

  python scripts/detprobe_cmp.py <probe_dir> [--max 12]

<probe_dir>/<tag>/rep<r>_r<rank>.pt (written by scratch driver dp_drive.py). For every tag and rank, rep 0 vs rep r:
* first differing record (module, in/out, tensor index, shape, differing rows),
* CULPRITS: module calls whose inputs are bit-identical between the two reps but whose outputs differ (a module that
  reads state outside its arguments, e.g. a KV cache, can also show up here), in call order,
* cross-rank: records whose digests differ between ranks in the same rep (replicated tensors should agree).
"""

from __future__ import annotations

import argparse
import glob
import os

import torch


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def same(a, b) -> bool:
    if len(a) != len(b):
        return False
    for (sa, _, da), (sb, _, db) in zip(a, b):
        if sa != sb or not torch.equal(da, db):
            return False
    return True


def diff_desc(a, b) -> str:
    out = []
    for j, ((sa, dta, da), (sb, _, db)) in enumerate(zip(a, b)):
        if sa != sb:
            out.append(f"t{j} shape {sa} vs {sb}")
        elif not torch.equal(da, db):
            rows = (da != db).nonzero().flatten()
            out.append(f"t{j} {dta}{list(sa)} rows {rows.numel()}/{da.numel()} first {rows[:6].tolist()}")
    return "; ".join(out)


def compare(base: dict, other: dict, maxn: int) -> list[str]:
    ra, rb = base["recs"], other["recs"]
    lines = []
    if len(ra) != len(rb):
        lines.append(f"  record count differs: {len(ra)} vs {len(rb)}")
    first = None
    stack: dict[str, list[int]] = {}
    culprits = []
    ndiff = 0
    for i, (a, b) in enumerate(zip(ra, rb)):
        if a[0] != b[0] or a[1] != b[1]:
            lines.append(f"  call order differs at record {i}: {a[0]}/{a[1]} vs {b[0]}/{b[1]}")
            break
        eq = same(a[2], b[2])
        if not eq:
            ndiff += 1
            if first is None:
                first = (i, a[0], a[1], diff_desc(a[2], b[2]))
        if a[1] == "in":
            stack.setdefault(a[0], []).append(i)
        elif a[1] == "out":
            st = stack.get(a[0])
            j = st.pop() if st else None
            if j is not None and not eq and same(ra[j][2], rb[j][2]):
                culprits.append((i, a[0], diff_desc(a[2], b[2])))
    lines.append(f"  records {len(ra)}, differing {ndiff}, first: {first}")
    for c in culprits[:maxn]:
        lines.append(f"  CULPRIT rec {c[0]} {c[1]}: {c[2]}")
    if len(culprits) > maxn:
        lines.append(f"  ... {len(culprits)} culprit calls in total; modules: "
                     f"{sorted({'.'.join(x for x in c[1].split('.') if not x.isdigit()) for c in culprits})}")
    return lines


def cross_rank(d0: dict, d1: dict, maxn: int) -> list[str]:
    lines = []
    n = 0
    for i, (a, b) in enumerate(zip(d0["recs"], d1["recs"])):
        if a[0] != b[0] or a[1] != b[1]:
            break
        if not same(a[2], b[2]):
            n += 1
            if n <= maxn:
                lines.append(f"    rec {i} {a[0]}/{a[1]}: {diff_desc(a[2], b[2])}")
    lines.insert(0, f"  ranks differ in {n} records")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--cross", action="store_true", help="also compare ranks (replicated tensors)")
    args = ap.parse_args()
    for tag in sorted(os.listdir(args.dir)):
        td = os.path.join(args.dir, tag)
        if not os.path.isdir(td):
            continue
        files = glob.glob(os.path.join(td, "rep*_r*.pt"))
        reps = sorted({int(os.path.basename(f)[3:].split("_")[0]) for f in files})
        ranks = sorted({int(f.rsplit("_r", 1)[1].split(".")[0]) for f in files})
        if not reps:
            continue
        print(f"== {tag}: reps {reps} ranks {ranks}")
        for k in ranks:
            base = load(os.path.join(td, f"rep{reps[0]}_r{k}.pt"))
            for r in reps[1:]:
                p = os.path.join(td, f"rep{r}_r{k}.pt")
                if not os.path.exists(p):
                    continue
                print(f" rank {k} rep {reps[0]} vs rep {r} (T={base['T']}):")
                for line in compare(base, load(p), args.max):
                    print(line)
        if args.cross and len(ranks) > 1:
            d0 = load(os.path.join(td, f"rep{reps[0]}_r{ranks[0]}.pt"))
            for k in ranks[1:]:
                print(f" rep {reps[0]} rank {ranks[0]} vs rank {k}:")
                for line in cross_rank(d0, load(os.path.join(td, f"rep{reps[0]}_r{k}.pt")), args.max):
                    print(line)


if __name__ == "__main__":
    main()
