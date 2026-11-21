c1d_src3: a frozen copy of
fpr/kernels/c1mega/srv8m minus its two wrappers, for the DSpark drafter kind c1d=s8m_nox2 (../c1d_s8m3_nox2.cu, ../c1d3.py,
../hsdt_c1m3e.py). Every file byte-identical to srv8m EXCEPT c1mega.cu: the counter-layout
static_assert is scoped to TOPK 6 (`NRM != 4 || TOPK != 6 || (...)`), which the drafter (TOPK 3) needs.
guard is present (c1mega.cu expert-bitmap / slot paths). Never edited in place: a new srv8m port is a new c1d_src<N>.
