import pickle, sys
snap = pickle.load(open(sys.argv[1], "rb"))
dev, lo, hi = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
tr = snap["device_traces"][dev]
for i in range(lo, hi):
    e = tr[i]
    fr = [f"{f['filename'].split('/')[-1]}:{f['line']}:{f['name']}" for f in (e.get("frames") or [])[:3]]
    print(i, e["action"], hex(int(e.get("addr", 0) or 0)), round(int(e.get("size", 0) or 0) / 2**20, 3), e.get("stream"), fr)
