import pickle, sys
snap = pickle.load(open(sys.argv[1], "rb"))
A = int(sys.argv[2].replace("_", ""), 16)
dev = int(sys.argv[3])
tr = snap["device_traces"][dev]
def fr(ev, n=10):
    return " < ".join(f"{f['filename'].split('/')[-1]}:{f['line']}:{f['name']}" for f in (ev.get("frames") or [])[:n])
for i, ev in enumerate(tr):
    a, s = int(ev.get("addr", 0) or 0), int(ev.get("size", 0) or 0)
    if a <= A < a + max(s, 1):
        print(i, ev["action"], hex(a), s >> 20, "MiB stream", ev.get("stream"), "|", fr(ev))
