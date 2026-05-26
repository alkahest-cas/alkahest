# Stub for the compiled Rust extension module (alkahest._core / alkahest-py).
#
# ty and other static analysers cannot inspect compiled extension modules
# (.so / .pyd files).  This stub satisfies the import resolver while keeping
# the rest of python/alkahest/ type-checkable.
#
# Full per-symbol stubs are tracked in the type-annotations roadmap.
from typing import Any

def __getattr__(name: str) -> Any: ...
