// fe::Json (fe_common) <-> fe::mp::Value (engine wire DOM). Key order and int/float kinds are preserved,
// so a request body converted here feeds sampling_from_request() exactly like Python's json.loads() dict.
#pragma once
#ifdef FE_HAVE_COMMON
#include "common/json.h"
#include "engine/mp.h"

namespace fe::engine {
mp::Value to_mp(const fe::Json& j);   // BigInt -> UInt when it fits uint64, else Float (msgspec would reject it)
fe::Json to_json(const mp::Value& v); // Bin/Ext -> string of the raw bytes
}  // namespace fe::engine
#endif
