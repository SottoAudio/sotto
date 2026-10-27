"""Minimal protobuf wire-format reader/writer, enough to edit ONNX graphs without the
`onnx` package. A message is a list of [field_number, wire_type, value]; value is an int
(varint / fixed) or bytes (length-delimited). Field order and unknown fields are kept."""
import struct


def _varint(buf, i):
    shift = res = 0
    while True:
        b = buf[i]
        i += 1
        res |= (b & 0x7F) << shift
        if not b & 0x80:
            return res, i
        shift += 7


def _enc_varint(v):
    if v < 0:
        v += 1 << 64
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def decode(buf):
    msg, i, n = [], 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(buf, i)
        elif wt == 1:
            v = buf[i:i + 8]; i += 8
        elif wt == 2:
            ln, i = _varint(buf, i)
            v = bytes(buf[i:i + ln]); i += ln
        elif wt == 5:
            v = buf[i:i + 4]; i += 4
        else:
            raise ValueError(f"wire type {wt}")
        msg.append([f, wt, v])
    return msg


def encode(msg):
    out = bytearray()
    for f, wt, v in msg:
        out += _enc_varint((f << 3) | wt)
        if wt == 0:
            out += _enc_varint(v)
        elif wt == 2:
            if isinstance(v, list):
                v = encode(v)
            out += _enc_varint(len(v)) + v
        else:
            out += v
    return bytes(out)


def get(msg, f):
    return [v for ff, _, v in msg if ff == f]


def get1(msg, f, default=None):
    r = get(msg, f)
    return r[0] if r else default


def s(b):
    return b.decode() if isinstance(b, (bytes, bytearray)) else b


def packed_ints(b):
    out, i = [], 0
    while i < len(b):
        v, i = _varint(b, i)
        if v >= 1 << 63:
            v -= 1 << 64
        out.append(v)
    return out


# ------------------------------------------------------------------ ONNX helpers
# ModelProto.graph=7; GraphProto: node=1 initializer=5 input=11 output=12 value_info=13
# NodeProto: input=1 output=2 name=3 op_type=4 attribute=5 domain=7
# AttributeProto: name=1 f=2 i=3 s=4 t=5 floats=7 ints=8 type=20
# TensorProto: dims=1 data_type=2 int64_data=7 name=8 raw_data=9 float_data=4


class Node:
    def __init__(self, raw):
        self.raw = decode(raw) if isinstance(raw, (bytes, bytearray)) else raw

    @property
    def op(self):
        return s(get1(self.raw, 4, b""))

    @property
    def name(self):
        return s(get1(self.raw, 3, b""))

    @property
    def inputs(self):
        return [s(x) for x in get(self.raw, 1)]

    @property
    def outputs(self):
        return [s(x) for x in get(self.raw, 2)]

    def set_io(self, inputs=None, outputs=None):
        rest = [e for e in self.raw if e[0] not in ((1,) if inputs is not None else ()) + ((2,) if outputs is not None else ())]
        new = []
        if inputs is not None:
            new += [[1, 2, x.encode()] for x in inputs]
        if outputs is not None:
            new += [[2, 2, x.encode()] for x in outputs]
        self.raw = new + rest

    def attrs(self):
        out = {}
        for a in get(self.raw, 5):
            am = decode(a)
            name = s(get1(am, 1))
            t = get1(am, 20)
            if t == 2:
                out[name] = get1(am, 3)
                if out[name] >= 1 << 63:
                    out[name] -= 1 << 64
            elif t == 7:
                vals = []
                for x in get(am, 8):
                    vals += packed_ints(x) if isinstance(x, bytes) else [x]
                out[name] = vals
            elif t == 1:
                out[name] = struct.unpack("<f", get1(am, 2))[0]
            elif t == 3:
                out[name] = s(get1(am, 4))
            else:
                out[name] = ("type", t)
        return out

    def encode(self):
        return encode(self.raw)


def make_node(op, inputs, outputs, name, attrs=None):
    raw = [[1, 2, x.encode()] for x in inputs] + [[2, 2, x.encode()] for x in outputs]
    raw += [[3, 2, name.encode()], [4, 2, op.encode()]]
    for k, v in (attrs or {}).items():
        if isinstance(v, int):
            a = [[1, 2, k.encode()], [3, 0, v], [20, 0, 2]]
        elif isinstance(v, (list, tuple)):
            a = [[1, 2, k.encode()]] + [[8, 0, x] for x in v] + [[20, 0, 7]]
        else:
            raise TypeError(v)
        raw.append([5, 2, encode(a)])
    return Node(raw)


def make_int64_tensor(name, values, dims=None):
    dims = [len(values)] if dims is None else dims
    raw = [[1, 0, d] for d in dims] + [[2, 0, 7], [8, 2, name.encode()],
                                       [9, 2, struct.pack(f"<{len(values)}q", *values)]]
    return encode(raw)


def make_value_info(name, elem_type, shape):
    """shape: list of int or str (symbolic)."""
    dims = []
    for d in shape:
        dims.append([1, 2, encode([[1, 0, d]] if isinstance(d, int) else [[2, 2, d.encode()]])])
    tensor = [[1, 0, elem_type], [2, 2, encode(dims)]]
    typ = [[1, 2, encode(tensor)]]
    return encode([[1, 2, name.encode()], [2, 2, encode(typ)]])


def value_info_shape(raw):
    vi = decode(raw)
    name = s(get1(vi, 1))
    typ = decode(get1(vi, 2))
    tt = decode(get1(typ, 1))
    shape = []
    sh = get1(tt, 2)
    if sh is not None:
        for d in get(decode(sh), 1):
            dm = decode(d)
            v = get1(dm, 1)
            shape.append(v if v is not None else s(get1(dm, 2, b"?")))
    return name, get1(tt, 1), shape


def tensor_info(raw):
    t = decode(raw)
    dims = get(t, 1)
    return s(get1(t, 8, b"")), get1(t, 2), dims, t


def tensor_values(raw):
    import numpy as np
    name, dt, dims, t = tensor_info(raw)
    rd = get1(t, 9)
    if rd is not None:
        np_dt = {1: np.float32, 7: np.int64, 6: np.int32, 11: np.float64, 9: np.bool_}[dt]
        return np.frombuffer(rd, dtype=np_dt).reshape(dims)
    if dt == 7:
        vals = []
        for x in get(t, 7):
            vals += packed_ints(x) if isinstance(x, bytes) else [x]
        return np.array(vals, dtype=np.int64).reshape(dims)
    if dt == 1:
        vals = []
        for x in get(t, 4):
            vals += list(struct.unpack(f"<{len(x)//4}f", x)) if isinstance(x, bytes) else [x]
        return np.array(vals, dtype=np.float32).reshape(dims)
    raise ValueError(dt)
