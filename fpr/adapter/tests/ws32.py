import pickle, sys
snap = pickle.load(open(sys.argv[1], "rb"))
dev = int(sys.argv[2])
tr = snap["device_traces"][dev]
SZ = 32 << 20
def fr(ev, n=4):
    return " < ".join(f"{f['filename'].split('/')[-1]}:{f['line']}:{f['name']}" for f in (ev.get("frames") or [])[:n])
for i, ev in enumerate(tr):
    if int(ev.get("size", 0) or 0) == SZ and ev["action"] in ("alloc", "free_requested", "segment_alloc", "segment_free"):
        print(i, ev["action"], hex(int(ev["addr"])), "stream", ev.get("stream"), "|", fr(ev))
