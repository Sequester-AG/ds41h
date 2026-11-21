// Do not edit: re-run the generator.
// Build: -O2 -std=c++17 -ffp-contract=off.
#pragma once
#include <cstdint>
#include <limits>
#include <string>

namespace fpr {

// serve/scheduler.py SchedConfig, field for field (same names, same defaults)
struct SchedConfig {
  std::string placement = "balanced";
  int64_t migrate_min_blocks = 16;
  int64_t max_copy_bytes_per_step = 600000000;
  int64_t horizon_steps = 12;
  double min_slack_us = 3000.0;
  double max_slack_frac = 0.8;
  double prefill_floor_us = 60000.0;
  double steal_decode_tol = 0.05;
  double placement_tol = 0.02;
  int64_t backlog_drain_steps = 10;
  int64_t quantum = 256;
  bool rebalance = true;
  double rebalance_threshold = 0.1;
  int64_t rebalance_patience = 8;
  int64_t rebalance_every = 8;
  int64_t max_waiting_admit_per_step = 64;
  bool async_mode = false;
  std::string prefill_order = "srpt";
  double prefill_aging_tok_per_s = 40000.0;
  int64_t small_prefill_tokens = 4352;
  int64_t large_min_rows = 0;
  double large_share_after_s = 0.0;
  double large_share_frac = 0.25;
  double starve_wait_s = 4.0;
  double starve_slack_frac = 1.2;
  double amortize_mixed = 32.0;
  int64_t mixed_target_tokens = 0;
  double mixed_target_wait_s = 1.0;
  double mixed_target_full_wait_s = std::numeric_limits<double>::infinity();
  bool mixed_target_fill = false;
  bool mixed_target_total = false;
  std::string long_order = "srpt";
  std::string backlog_drain_until = "first_chunk";
  double backlog_drain_wait_s = std::numeric_limits<double>::infinity();
  int64_t mixed_adapt_lo = 0;
  int64_t mixed_adapt_hi = 0;
  double mixed_adapt_min = 0.0;
  int64_t mixed_adapt_force_min = 0;
  int64_t mixed_min_decode_steps = 2;
  double amortize_min_fixed_us = 5000.0;
  double max_mixed_step_us = 600000.0;
  double max_prefill_wait_s = std::numeric_limits<double>::infinity();
  int64_t cp_async_settle_steps = 2;
  bool cp_prefill = false;
  int64_t cp_min = 8192;
  int64_t cp_force_min = 0;
  double cp_force_room_frac = 0.5;
  int64_t cp_force_srpt_min_left = 0;
  double cp_force_srpt_hold = 1.0;
  std::string cp_force_order = "fcfs";
  int64_t cp_force_decode_steps = -1;
  bool cp_replay_first = true;
  double cp_exchange_us = 200.0;
  double cp_light_frac = 0.5;
  bool image_pin = true;
  bool steal = true;
  double steal_min_backlog_steps = 2.0;
  int64_t max_steals_per_step = 2;
  bool bucket_aware = false;
  std::string decode_buckets = "";
  int64_t bucket_min_rows = 64;

  // every field with its Python name, in declaration order: v(const char* name, T& member)
  template <class C, class V> static void visit(C& c, V&& v) {
    v("placement", c.placement);
    v("migrate_min_blocks", c.migrate_min_blocks);
    v("max_copy_bytes_per_step", c.max_copy_bytes_per_step);
    v("horizon_steps", c.horizon_steps);
    v("min_slack_us", c.min_slack_us);
    v("max_slack_frac", c.max_slack_frac);
    v("prefill_floor_us", c.prefill_floor_us);
    v("steal_decode_tol", c.steal_decode_tol);
    v("placement_tol", c.placement_tol);
    v("backlog_drain_steps", c.backlog_drain_steps);
    v("quantum", c.quantum);
    v("rebalance", c.rebalance);
    v("rebalance_threshold", c.rebalance_threshold);
    v("rebalance_patience", c.rebalance_patience);
    v("rebalance_every", c.rebalance_every);
    v("max_waiting_admit_per_step", c.max_waiting_admit_per_step);
    v("async_mode", c.async_mode);
    v("prefill_order", c.prefill_order);
    v("prefill_aging_tok_per_s", c.prefill_aging_tok_per_s);
    v("small_prefill_tokens", c.small_prefill_tokens);
    v("large_min_rows", c.large_min_rows);
    v("large_share_after_s", c.large_share_after_s);
    v("large_share_frac", c.large_share_frac);
    v("starve_wait_s", c.starve_wait_s);
    v("starve_slack_frac", c.starve_slack_frac);
    v("amortize_mixed", c.amortize_mixed);
    v("mixed_target_tokens", c.mixed_target_tokens);
    v("mixed_target_wait_s", c.mixed_target_wait_s);
    v("mixed_target_full_wait_s", c.mixed_target_full_wait_s);
    v("mixed_target_fill", c.mixed_target_fill);
    v("mixed_target_total", c.mixed_target_total);
    v("long_order", c.long_order);
    v("backlog_drain_until", c.backlog_drain_until);
    v("backlog_drain_wait_s", c.backlog_drain_wait_s);
    v("mixed_adapt_lo", c.mixed_adapt_lo);
    v("mixed_adapt_hi", c.mixed_adapt_hi);
    v("mixed_adapt_min", c.mixed_adapt_min);
    v("mixed_adapt_force_min", c.mixed_adapt_force_min);
    v("mixed_min_decode_steps", c.mixed_min_decode_steps);
    v("amortize_min_fixed_us", c.amortize_min_fixed_us);
    v("max_mixed_step_us", c.max_mixed_step_us);
    v("max_prefill_wait_s", c.max_prefill_wait_s);
    v("cp_async_settle_steps", c.cp_async_settle_steps);
    v("cp_prefill", c.cp_prefill);
    v("cp_min", c.cp_min);
    v("cp_force_min", c.cp_force_min);
    v("cp_force_room_frac", c.cp_force_room_frac);
    v("cp_force_srpt_min_left", c.cp_force_srpt_min_left);
    v("cp_force_srpt_hold", c.cp_force_srpt_hold);
    v("cp_force_order", c.cp_force_order);
    v("cp_force_decode_steps", c.cp_force_decode_steps);
    v("cp_replay_first", c.cp_replay_first);
    v("cp_exchange_us", c.cp_exchange_us);
    v("cp_light_frac", c.cp_light_frac);
    v("image_pin", c.image_pin);
    v("steal", c.steal);
    v("steal_min_backlog_steps", c.steal_min_backlog_steps);
    v("max_steals_per_step", c.max_steals_per_step);
    v("bucket_aware", c.bucket_aware);
    v("decode_buckets", c.decode_buckets);
    v("bucket_min_rows", c.bucket_min_rows);
  }
};

}  // namespace fpr
