"""GPU test: ds41h fused SWA decode prep (ds41h/patch_glue.py) == stock ge + tail fill + _compute_swa_indices_and_lens.

  DS41H=1 DS41H_PREP_FUSE=1 python tests/test_prep_fuse.py        (one GPU, < 1 min)

1. The AST rewrite of DeepseekSparseSWAMetadataBuilder.build applies (hash + statement counts) and compiles.
2. Kernel equivalence on random batches: decode rows with / without invalid (padded) slots, several requests,
   seq lens below / above the window, replay_start bounds, block sizes 64 / 256, slot_mapping longer than the decode
   rows (padding), stale garbage in the output buffers before the call: is_valid, decode_swa_lens (whole buffer incl.
   the zero tail) and the decode window indices must be bit-identical.
"""

import os
import sys
from types import SimpleNamespace

os.environ.setdefault("DS41H", "1")
os.environ.setdefault("DS41H_PREP_FUSE", "1")


def main() -> int:
    from ds41h import hook

    hook.install()
    import torch

    from ds41h import patch_glue
    from vllm.v1.attention.backends.mla import sparse_swa as m

    assert getattr(m.DeepseekSparseSWAMetadataBuilder.build, "_ds41h_fused", False), "build() not rewritten"
    dev = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(0)
    bad = cases = 0
    for case in range(300):
        block_size = [64, 256][case % 2]
        window = 128
        max_tokens = int(torch.randint(64, 20000, (1,), generator=g))
        num_reqs = int(torch.randint(1, 9, (1,), generator=g))
        q_lens = torch.randint(1, 9, (num_reqs,), generator=g)
        n_dec = int(q_lens.sum())
        n_slot = n_dec + int(torch.randint(0, 9, (1,), generator=g))
        qsl = torch.zeros(num_reqs + 1, dtype=torch.int32)
        qsl[1:] = torch.cumsum(q_lens, 0)
        seq_lens = (q_lens + torch.randint(0, 3000, (num_reqs,), generator=g)).to(torch.int32)
        replay = torch.where(torch.rand(num_reqs, generator=g) < 0.3,
                             (seq_lens.float() * torch.rand(num_reqs, generator=g)).to(torch.int32),
                             torch.zeros(num_reqs, dtype=torch.int32))
        nblk = (int(seq_lens.max()) + block_size - 1) // block_size + 2
        block_table = torch.randint(0, 50000, (num_reqs, nblk), generator=g, dtype=torch.int32)
        t2r = torch.repeat_interleave(torch.arange(num_reqs, dtype=torch.int32), q_lens)
        t2r = torch.cat([t2r, torch.zeros(max_tokens - n_dec, dtype=torch.int32)])
        slot = torch.randint(0, 10**7, (n_slot,), generator=g, dtype=torch.int64)
        slot[torch.rand(n_slot, generator=g) < 0.15] = -1
        slot[n_dec:] = -1  # padded rows
        to = dict(device=dev)
        args = [x.to(**to) for x in (qsl, seq_lens, t2r, block_table, slot, replay)]
        qsl_d, seq_d, t2r_d, bt_d, slot_d, replay_d = args
        outs = []
        for fused in (False, True):
            is_valid_buf = torch.randint(0, 2, (max_tokens,), dtype=torch.bool, device=dev)
            lens = torch.randint(-5, 999, (max_tokens,), dtype=torch.int32, device=dev)
            idx = torch.randint(-5, 999, (max_tokens, 1, window), dtype=torch.int32, device=dev)
            is_valid = is_valid_buf[:n_slot]
            if fused:
                b = SimpleNamespace(decode_swa_lens=lens, decode_swa_indices=idx, window_size=window,
                                    block_size=block_size)
                patch_glue._prep_swa(b, slot_d, is_valid, idx, n_dec, qsl_d, seq_d, t2r_d, bt_d, replay_d)
            else:
                torch.ge(slot_d, 0, out=is_valid)
                lens[n_dec:] = 0
                m._COMPUTE_SWA_INDICES_AND_LENS_KERNEL(idx, lens, window, idx.shape[-1], lens, lens, qsl_d, seq_d,
                                                       t2r_d, is_valid, bt_d, block_size, replay_d,
                                                       num_tokens=n_dec, token_offset=0)
            torch.cuda.synchronize()
            outs.append((is_valid.clone(), lens.clone(), idx[:n_dec].clone()))
        cases += 1
        ok = all(torch.equal(a, b) for a, b in zip(*outs))
        if not ok:
            bad += 1
            print(f"MISMATCH case {case}: n_dec {n_dec} n_slot {n_slot} max_tokens {max_tokens} reqs {num_reqs}")
    print(f"{cases} cases, {bad} mismatches; stats {patch_glue.STATS}")
    print("PASS" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
