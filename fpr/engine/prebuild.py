#!/usr/bin/env python3
"""Engine prebuild (CPU only, no GPU needed): the wcache entries of the REAL 40-layer model per rank, with the
serve/weights.py builders and keys (shared with serve: attn_L{L}_v1, L{L}_r{r}_raw_experts_v1, cos_sin_c{c}_v1).

  CUDA_VISIBLE_DEVICES= python engine/prebuild.py --part 0 --nparts 4      (layers L with L % nparts == part)
"""
import argparse
import os
import sys
import time

PHYS = os.environ.get("PHYS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PHYS, "serve"))
import weights as Wt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--layers", default="all")
    ap.add_argument("--ranks", default="0,1,2,3")
    ap.add_argument("--idx", action="store_true", help="only the indexer / compressor weights of the 8 index layers")
    a = ap.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "CPU only: CUDA_VISIBLE_DEVICES= (empty)"
    Wt._paths()
    from wcache import prebuild
    import bench2b as B
    _orig = B.stock_cos_sin
    # ratio-0 layers (0, 1): attn_half.load_attn_weights calls stock_cos_sin(0, dev, 0), which has no table for
    # max_pos 0 (math domain error). The cached entry drops the table (built on the GPU at load), so any max_pos works.
    B.stock_cos_sin = lambda ratio, dev, max_pos: _orig(ratio, dev, max_pos if (ratio or max_pos) else 16)
    if a.idx:
        sys.path.insert(0, PHYS)
        from engine.hooks_idx import LAYERS, build_idx_weights
        for L in LAYERS:
            t1 = time.time()
            prebuild(f"eng_idxw_L{L}_v1", lambda L=L: build_idx_weights(L))
            print(f"prebuilt eng_idxw_L{L}_v1 in {time.time() - t1:.1f} s", flush=True)
        return
    layers = list(range(40)) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    layers = [L for L in layers if L % a.nparts == a.part]
    ranks = [int(x) for x in a.ranks.split(",")]
    t0 = time.time()
    # cos_sin is NOT cached: vLLM computes the RoPE table on the GPU and CPU bits differ (sp_layer.c_attn_weights);
    # the engine builds it on the device at load (engine/weights.py)
    ents = [e for e in Wt.entries(layers, ranks, small_all=False, globals_=(a.part == 0), drafter=False)
            if not e[0].startswith("cos_sin")]
    for key, b in ents:
        t1 = time.time()
        prebuild(key, b)
        print(f"prebuilt {key} in {time.time() - t1:.1f} s", flush=True)
    print(f"part {a.part}: done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
