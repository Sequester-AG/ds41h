# Engram table modes

DeepSeek-V4.1's two Engram layers read n-gram tables of about 190 GiB in total. `ENGRAM=` on `scripts/serve_ds41h.sh`
picks where they live. The rows served are bit-identical in all three modes.

| Mode | Where | Effect |
|---|---|---|
| `nvme` (default) | Files under `$DS41H_HOME/engram_nvme` (`DS41H_ENGRAM_NVME_DIR`), about 210 GB, built on first start in about 2 min | Frees the GPU memory for KV. About 4.2 GiB of pinned RAM and one busy poller thread per GPU worker. Prefill of novel text is limited by NVMe reads |
| `host` | One pinned copy in `/dev/shm` (`DS41H_ENGRAM_HOST_DIR`), about 195 GiB | Frees the GPU memory for KV, with no NVMe read limit |
| `hbm` | GPU memory, as in stock vLLM | Leaves little room for KV: about 86K tokens at `MAXLEN=36864` |

With `ENGRAM=host`, the KV cache was 6.6M tokens at k=5 with `MAX_SEQS=256`, `MAXLEN=262144`, `GPU_UTIL=0.91`, and 11.3M
tokens at k=3 with `MAX_SEQS=128`, `MAXLEN=655360`, `GPU_UTIL=0.91`. The server logs the size at start.

## Host mode setup

The server creates and fills the table file at start, loading it cooperatively across the ranks. It removes the file on
exit unless `DS41H_ENGRAM_HOST_KEEP=1`. To inspect or remove it:

```bash
python -m ds41h.engram_host status
```

```bash
python -m ds41h.engram_host drop
```

**RAM.** The tables need 194.6 GiB in the default `packed` layout, on top of the page cache used while loading the
checkpoint.

**`/dev/shm` size.** The tmpfs default is 50% of RAM. If `df -h /dev/shm` shows less than the tables need, raise it as
root:

```bash
mount -o remount,size=300G /dev/shm
```

To keep the size across reboots, add it to `/etc/fstab`:

```
tmpfs /dev/shm tmpfs defaults,size=300G 0 0
```

If `/dev/shm` has a per-user quota, the tables count against it, along with the `KVRAM` region.

**memlock.** Not needed in our tests: registering 1 GiB and 8 GiB mappings worked with `RLIMIT_MEMLOCK` at 8 MiB, because
the NVIDIA driver pins them without memlock accounting.

**IOMMU.** Recommended. If `/sys/kernel/iommu_groups/*/type` shows `DMA-FQ`, each zero-copy read is translated twice, by
the GPU MMU and by the IOMMU. We saw random-row reads fall from about 40 GB/s on a 64 MiB working set to 8 GB/s on 8 GiB.
Adding `iommu=pt` to the kernel command line and rebooting is the expected fix; we did not measure the result. It raises
the bandwidth headroom but is not needed for correctness.

**Huge pages.** Optional. Backing the tables with transparent huge pages did not change the random-row rate in our tests,
because the driver keeps 4 KiB GPU page table entries for registered memory. To try them, point `DS41H_ENGRAM_HOST_DIR` at
a `huge=always` tmpfs or a hugetlbfs mount.

**NUMA.** Tested on a single NUMA node only. On several nodes, consider creating the tables under
`numactl --interleave=all`.
