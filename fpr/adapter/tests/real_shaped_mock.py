"""serve's MockEngine with the real engine's row geometry (engine.py EngineConfig: max_mixed_rows 4,096 + 320, tail
rows rounded up to 256), so CPU runs plan the same prefill chunks as the GPU runs (the plain mock caps steps at 320)."""
from mock_engine import MockEngine


class RealShapedMock(MockEngine):
    max_step_rows = 4096 + 320
    max_tail_rows = -(-(4096 + 320) // 256) * 256

    def __init__(self, devices, model_dir=None, max_rows: int = 512, **kw):
        super().__init__(devices, model_dir, max_rows=max(max_rows, self.max_step_rows), **kw)
