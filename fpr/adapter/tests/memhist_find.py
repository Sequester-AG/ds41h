"""Which allocation lived at a faulting address (Xid 31 FAULT_PDE under D13).

From fpr/:
  python adapter/tests/memhist_find.py <memsnap.pickle> 0x74db_b0000000 [0x... ...]
  python adapter/tests/memhist_find.py <memsnap.pickle> --journal "<since time>"   # addresses from journalctl -k

The pickle is torch's allocator snapshot with history (adapter/debug_patches.py PHYS_MEMHIST). For each address, on
every device trace: the last segment event covering it (segment_alloc / segment_free / map / unmap), the last block
alloc covering it and that block's free, each with its size and the top Python frames (file:line:function); and whether
a live segment still covers it in the snapshot's state. Frames only, no tensor contents.
"""
import argparse
import pickle
import re
import subprocess


def frames(ev, n=7):
    out = []
    for f in ev.get("frames") or []:
        fn = f.get("filename", "?")
        if "/torch/" in fn and not out:
            continue
        out.append(f"{fn.split('/')[-1]}:{f.get('line')}:{f.get('name')}")
        if len(out) >= n:
            break
    return out


def find(snap, a):
    res = []
    for d, tr in enumerate(snap.get("device_traces") or []):
        seg_last, blk_alloc, blk_free, n = None, None, None, 0
        for i, ev in enumerate(tr):
            act, addr, size = ev.get("action"), int(ev.get("addr", 0) or 0), int(ev.get("size", 0) or 0)
            if not (addr <= a < addr + max(size, 1)):
                continue
            n += 1
            if act.startswith("segment"):
                seg_last = (i, ev)
            elif act == "alloc":
                blk_alloc, blk_free = (i, ev), None
            elif act in ("free_requested", "free_completed") and blk_alloc is not None:
                blk_free = (i, ev)
        if n:
            res.append((d, len(tr), seg_last, blk_alloc, blk_free))
    live = []
    for sg in snap.get("segments") or []:
        b, s = int(sg["address"]), int(sg["total_size"])
        if b <= a < b + s:
            live.append(dict(device=sg.get("device"), base=hex(b), size_mb=round(s / 2**20, 1),
                             pool=sg.get("segment_pool_id"), type=sg.get("segment_type")))
    return res, live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickle")
    ap.add_argument("addr", nargs="*")
    ap.add_argument("--journal", help="journalctl -k --since this time: take the Xid 31 fault addresses from it")
    a = ap.parse_args()
    addrs = [int(x.replace("_", ""), 16) for x in a.addr]
    if a.journal:
        txt = subprocess.run(["journalctl", "-k", "--since", a.journal], capture_output=True, text=True).stdout
        addrs += [int(m.replace("_", ""), 16) for m in re.findall(r"faulted @ (0x[0-9a-f_]+)", txt)]
    with open(a.pickle, "rb") as f:
        snap = pickle.load(f)
    print(f"{a.pickle}: {len(snap.get('segments') or [])} live segments, "
          f"trace lengths {[len(t) for t in snap.get('device_traces') or []]}")
    for x in addrs:
        res, live = find(snap, x)
        print(f"\n== {hex(x)}  live segment now: {live or 'NONE (unmapped)'}")
        for d, ntr, seg, al, fr in res:
            print(f"  device trace {d} ({ntr} events)")
            for lab, it in (("last segment event", seg), ("last block alloc", al), ("its free", fr)):
                if it is None:
                    print(f"    {lab}: -")
                    continue
                i, ev = it
                print(f"    {lab}: #{i} {ev['action']} addr {hex(int(ev['addr']))} size {int(ev['size']) / 2**20:.2f} MiB "
                      f"stream {ev.get('stream')} | {' < '.join(frames(ev))}")
        if not res:
            print("  not in any device trace (allocated before the history started, or outside torch's allocator)")


if __name__ == "__main__":
    main()
