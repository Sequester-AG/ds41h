"""Peer-store all-gathers for the head / sampler / drafter (single-process 4-GPU harness, like kernels/sp_layer).

Builds kernels/sp_layer/comm.cu (snapshot in kernels/head/vendor) as the extension `head_comm`, and wraps
its k_ag_push + k_wait into a gather with one flag word + one epoch counter per collective SITE (each site = one fixed
point of the step program: "h" final hidden, "g1", "g2", "d0".."d2", "hd" draft hidden).

  gather = PeerGather(devices, ncta={"h": 8})          # default ncta 1 for the word exchanges
  gather(site, inps, outs, streams)                    # inps[r] [R, ...] contiguous -> outs[s][r] on every rank s
"""
from __future__ import annotations

import os

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.join(os.path.dirname(HERE), "sp_layer")
_EXT = {}


def ext():
    import vendor_ext

    return vendor_ext.comm()


class PeerGather:
    """Flags / epochs of every site are created up front and the devices synchronized: a lazily created flag is
    zero-filled on the device's default stream, which can land AFTER a peer's release-add on the collective's stream
    (lost signal -> the waiter spins forever; inside a graph capture the fill would even be replayed every step)."""

    SITES = ("h", "g1", "g2", "g3", "hd", "d0", "d1", "d2", "d3", "d4", "d5", "d6")

    def __init__(self, devices, ncta: dict | None = None, sites=SITES):
        self.devs = [torch.device(d) for d in devices]
        self.W = len(self.devs)
        assert self.W == 4, "comm.cu is built for W = 4"
        self.e = ext()
        self.e.enable_peers(self.W)
        self.ncta = dict(ncta or {})
        self.sites: dict[str, tuple[list, list]] = {}
        for site in sites:
            flags = [torch.zeros(4, dtype=torch.int32, device=d) for d in self.devs]
            epochs = [torch.zeros(1, dtype=torch.int32, device=d) for d in self.devs]
            self.sites[site] = (flags, epochs)
        for d in self.devs:
            torch.cuda.synchronize(d)

    def _site(self, site):
        if site not in self.sites:
            raise KeyError(f"PeerGather: unknown site {site!r} (create it in the constructor)")
        return self.sites[site]

    def __call__(self, site, inps, outs, streams):
        flags, epochs = self._site(site)
        nc = self.ncta.get(site, 1)
        for r in range(self.W):
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(streams[r]):
                self.e.ag_push(inps[r], r, outs, flags, epochs[r], nc)
        for r in range(self.W):
            with torch.cuda.device(self.devs[r]), torch.cuda.stream(streams[r]):
                self.e.wait(flags[r], epochs[r], self.W * nc)
