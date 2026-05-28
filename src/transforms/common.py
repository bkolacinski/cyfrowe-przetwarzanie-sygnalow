from __future__ import annotations

import math


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def ensure_power_of_two(value: int, param_name: str = "N") -> None:
    if not is_power_of_two(value):
        raise ValueError(f"{param_name} must be a power of two, got {value}.")


def bit_reverse(value: int, bits: int) -> int:
    reversed_value = 0
    for _ in range(bits):
        reversed_value = (reversed_value << 1) | (value & 1)
        value >>= 1
    return reversed_value


def bit_reversal_indices(length: int) -> list[int]:
    ensure_power_of_two(length, "length")
    bit_count = int(math.log2(length))
    return [bit_reverse(i, bit_count) for i in range(length)]
