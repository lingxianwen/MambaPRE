PAD_BYTE = 256

ROLES = (
    "constant",
    "length",
    "sequence",
    "opcode",
    "address",
    "count",
    "payload",
    "checksum",
)

TYPES = ("uint8", "uint16_be", "uint16_le", "uint32_be", "bytes")

ROLE_TO_ID = {name: idx for idx, name in enumerate(ROLES)}
TYPE_TO_ID = {name: idx for idx, name in enumerate(TYPES)}

LENGTH_BUCKETS = (
    ("0009-0032", 0, 32),
    ("0033-0128", 33, 128),
    ("0129-0256", 129, 256),
    ("0257-0512", 257, 512),
    ("0513-1024", 513, 1024),
    ("1025+", 1025, 10**9),
)


def length_bucket(length: int) -> str:
    for name, low, high in LENGTH_BUCKETS:
        if low <= length <= high:
            return name
    raise ValueError(f"invalid message length: {length}")

