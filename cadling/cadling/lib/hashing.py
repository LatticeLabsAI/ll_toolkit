"""Deterministic, cross-process-stable hashing.

Python's builtin ``hash()`` for ``str``/``bytes`` is salted per process via
``PYTHONHASHSEED``, so any feature value or token id derived from it changes
between runs and machines. When such values are written into training data (face
features, command/op tokens), that silently makes the data non-reproducible.

:func:`stable_hash` provides a deterministic alternative (BLAKE2b) that yields
identical results across runs, processes and machines.
"""

from __future__ import annotations

import hashlib
from typing import Optional, Union


def stable_hash(value: Union[str, bytes], modulo: Optional[int] = None) -> int:
    """Return a deterministic, non-negative hash of ``value``.

    Args:
        value: A ``str`` (UTF-8 encoded) or ``bytes`` to hash.
        modulo: If given, the result is reduced modulo this value.

    Returns:
        A non-negative integer hash, reproducible across runs/machines (unlike
        the builtin ``hash()``).
    """
    data = value.encode("utf-8") if isinstance(value, str) else value
    digest = hashlib.blake2b(data, digest_size=8).digest()
    result = int.from_bytes(digest, "big")
    return result % modulo if modulo else result
