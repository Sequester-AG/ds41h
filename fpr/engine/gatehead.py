"""The head for exactness-check sequences ("gate" sequences): forced decode and score.

forced (production's ds41h/patch_force.py semantics): the rows' input ids are the forced stream (tables.tokens), every
  verify row is accepted, and the RAW target logits of every verify row (+ the last prefill row) are returned:
  StepResult.logits[rid] = (positions, fp32 [rows, V], vocab_start 0) -- the full vocabulary from the owner rank.
score (teacher-forced prompt logprobs): every prompt row runs all 40 layers (tail_rows = all rows) and returns the
  top-k logprobs: StepResult.logprobs[rid] = [(position, [(id, logprob)] x k)].
Logits = bf16(h @ head^T) with fp32 accumulation (production's compute_logits returns bf16 logits), then fp32; logprobs
= log_softmax in fp32. The head weight is replicated (1.3 GB per GPU) so the owner computes its rows alone (no exchange).
Normal sequences keep HSD (or the stub); a step with only gate sequences skips HSD.
"""
from __future__ import annotations

import torch

V = 129_280


class GateHead:
    def __init__(self, eng, topk: int = 20, chunk: int = 512):
        self.eng, self.topk, self.chunk = eng, topk, chunk
        w = eng.reader.get("head.weight").to(torch.bfloat16)
        self.W = []
        for d in eng.devs:
            self.W.append(w.to(d))
        del w

    @staticmethod
    def flags(tab):
        sq = getattr(tab, "seq", None) or {}
        f = sq.get("forced")
        sc = sq.get("score")
        n = len(tab.seq_rows)
        return ([int(f[k]) for k in range(n)] if f is not None else [0] * n,
                [int(sc[k]) for k in range(n)] if sc is not None else [0] * n)

    def any_gate(self, tables) -> tuple:
        """(any gate sequence in the step, all sequences are gate sequences)."""
        anyg, allg, n = False, True, 0
        for t in tables:
            f, sc = self.flags(t)
            for a, b in zip(f, sc):
                n += 1
                g = bool(a or b)
                anyg |= g
                allg &= g
        return anyg, (allg and n > 0)

    def logits(self, r: int, h: torch.Tensor) -> torch.Tensor:
        return torch.mm(h, self.W[r].t()).float()

    def fill(self, r: int, s: dict, tab, res) -> None:
        """Overwrite res for the bucket's forced / score sequences (rows index the bucket = tab rows)."""
        f, sc = self.flags(tab)
        if not any(f) and not any(sc):
            return
        h_all = s["h"]
        for k, (rid, row0, nr, kind, last) in enumerate(tab.seq_rows):
            if f[k]:
                rows = list(range(row0, row0 + nr)) if kind == "decode" else ([row0 + nr - 1] if last else [])
                if rows:
                    ix = torch.tensor(rows, dtype=torch.int64, device=s["dev"])
                    lg = self.logits(r, h_all.index_select(0, ix)).cpu()
                    res.logits[rid] = ([int(tab.positions[x]) for x in rows], lg, 0)
                tg = [int(tab.targets[x]) for x in rows] if tab.targets is not None else []
                res.accepted.pop(rid, None)
                res.tokens.pop(rid, None)
                if kind == "decode":
                    res.accepted[rid] = nr
                    res.tokens[rid] = tg
                elif last:
                    res.tokens[rid] = tg[-1:]
            elif sc[k]:
                out = []
                for c0 in range(row0, row0 + nr, self.chunk):
                    c1 = min(row0 + nr, c0 + self.chunk)
                    ix = torch.arange(c0, c1, device=s["dev"])
                    lp = torch.log_softmax(self.logits(r, h_all.index_select(0, ix)), -1)
                    v, i = lp.topk(self.topk, -1)
                    v, i = v.cpu().tolist(), i.cpu().tolist()
                    for j in range(c1 - c0):
                        out.append((int(tab.positions[c0 + j]), list(zip(i[j], v[j]))))
                res.logprobs[rid] = out
                res.accepted.pop(rid, None)
                res.tokens.pop(rid, None)
                if last:
                    res.tokens[rid] = [out[-1][1][0][0]]
