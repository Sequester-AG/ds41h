"""RealShapedMock under a name without "mock": serve's OfflineEngine and the adapter's Stage4Worker then treat it as the
real engine (engine-measured step costs by default), as in the GPU runs the exactness gate compares."""
from real_shaped_mock import RealShapedMock as ShapedEngine  # noqa: F401
