from __future__ import annotations

import re
from pathlib import Path


def _extract_td_value_ops(td_text: str) -> set[str]:
    # Capture each op block and keep only value-producing ops (blocks with `let results =`).
    op_blocks = re.findall(
        r"def\s+MiniTriton_[A-Za-z0-9_]+Op\s*:\s*MiniTriton_Op<\"([^\"]+)\">\s*\{(.*?)\}",
        td_text,
        flags=re.DOTALL,
    )
    value_ops: set[str] = set()
    for op_name, block_body in op_blocks:
        if "let results" in block_body:
            value_ops.add(op_name)
    return value_ops


def _extract_cpp_supported_ops(cpp_text: str) -> set[str]:
    # Extract quoted entries inside the kOps initializer list.
    array_match = re.search(r"kOps\s*=\s*\{(?P<body>.*?)\};", cpp_text, flags=re.DOTALL)
    if array_match is None:
        raise AssertionError("could not find kOps initializer in dialect_ops.cpp")
    return set(re.findall(r'"([^\"]+)"', array_match.group("body")))


def test_td_and_cpp_supported_ops_are_consistent():
    repo_root = Path(__file__).resolve().parents[2]
    td_path = repo_root / "tools" / "mtc-lower" / "MiniTritonOps.td"
    cpp_path = repo_root / "tools" / "mtc-lower" / "dialect_ops.cpp"

    td_ops = _extract_td_value_ops(td_path.read_text(encoding="utf-8"))
    cpp_ops = _extract_cpp_supported_ops(cpp_path.read_text(encoding="utf-8"))

    assert td_ops == cpp_ops, (
        "TD/C++ op schema drift detected: "
        f"td_only={sorted(td_ops - cpp_ops)}, cpp_only={sorted(cpp_ops - td_ops)}"
    )
