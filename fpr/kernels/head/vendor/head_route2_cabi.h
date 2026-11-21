// head_route2_cabi.h -- C ABI of kernels/head/vendor/route2.cu, the HSD head's head_route2, per
// fpr/engine/mixed_cxx/fpr_cabi.h. vendor/route2.cu was byte-identical to moe_stream/route2.cu before its C ABI
// and carries the same conversion, so the Args structs are moe_stream/route2_cabi.h's (FprRoute2InitArgs, FprRoute2Args,
// FprRoute2rArgs). The drafter (drafter.py moe_mode "route2" / "n8") calls route2 / route2x = "head_route2.route2" (ec8
// NULL / non-NULL); route2r comes with the shared conversion. route2_max_ec(T) / route2_scratch_ints() are host constants
// for buffer sizing (route2_cabi.h; ws = "head_route2.route2"'s ws_bytes).
// Table: fpr_head_route2_table(); "head_route2.init" (per device: the k_fill / k_one dynamic-SMEM opt-ins, atomic per-device
// mask; the SM count), "head_route2.route2", "head_route2.route2r". route2_config's tuning globals: init-time only.
#pragma once
#include "../../moe_stream/route2_cabi.h"

#ifdef __cplusplus
extern "C" {
#endif

const FprExtTable* fpr_head_route2_table(void);

#ifdef __cplusplus
}
#endif
