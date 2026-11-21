// fpr_core.h
// Golden-trace method: fpr/core/GTRACE.md.
//
// Every type mirrors today's Python 1:1 (same field names, same semantics, same units) so golden traces compare field by
// field: serve/scheduler.py (Request, SchedConfig, SeqWork, StepPlan), serve/kv_manager.py (CopyOp, SeqKV, AdmitResult,
// OwnedKVManager), serve/interfaces.py (SeqEntry, StepBatch, StepContext, StepResult), serve/rowexp_ref.py
// (CompactTables), engine/d13.py (the D13 host side). A field Python has and C++ does not = a spec bug: report it.
//
// Build rule for EVERY component: -O2 -std=c++17 -ffp-contract=off (no FMA contraction: float parity
// with Python's est_us / loads in the last bit). Shared trace writer: fpr/core/gtrace.py (next_seq + per-component files).
// Rules: C++17, no exceptions across component boundaries (Status return), no Python, deterministic (no unordered
// iteration where Python iterates in insertion order: use insertion-ordered maps), no wall-clock reads inside components
// (time comes in as `now` arguments, as in Python), all randomness seeded from the request (sampling is on the device).
// v3: SPLIT into per-component headers, each owned by its porting session: common.h, kv.h + tables.h, scheduler.h
// + geometry.h + cost_model.h, d13.h, core.h, engine.h. Hash is 128-bit (common.h).
#pragma once
#include "common.h"
#include "geometry.h"
#include "kv.h"
#include "scheduler.h"
#include "tables.h"
#include "engine.h"
#include "d13.h"
#include "core.h"
