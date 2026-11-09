"""FPRv1 engine: the StepBatch-driven step program.

serve's worker loads `engine.Engine` (WorkerConfig.engine_cls). Imported lazily so CPU-only tools (prebuild scripts) can
import submodules without torch / CUDA work at package import."""


def __getattr__(name):
    if name in ("Engine", "EngineConfig", "tail_batch"):
        from . import engine as _e
        return getattr(_e, name)
    raise AttributeError(name)
