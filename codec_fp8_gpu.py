"""SplitZip for FP8 KV caches: an extension of the BF16 codec in codec_gpu.py to E4M3 and E5M2
with 2-, 3- and 4-bit exponent codes.

Same scheme as the BF16 codec: a calibrated global top-k exponent codebook with fixed-length codes, sign and
mantissa stored verbatim, rare exponents routed to a chunk-local escape stream, two-pass encode
(dense pack + per-chunk escape count, prefix sum, compact escape write).

What changes for FP8
  * One FP8 value is one byte, so a GPU lane owns 8 values in a single 64-bit register and all
    field extraction, coding and packing is done with SWAR arithmetic (no per-element work).
  * 8 values x (sign+mantissa bits + code bits) is always a whole number of bytes:
        E4M3 (4 sm bits): 2-bit -> 48 bits, 3-bit -> 56 bits
        E5M2 (3 sm bits): 2-bit -> 40 bits, 3-bit -> 48 bits, 4-bit -> 56 bits
    The packed word is stored as coalesced planes: a u32 plane plus a u16 and/or u8 plane.
  * If the calibrated top-k set is a contiguous exponent window, code = exponent - base, computed
    for 8 values at once; decode is an add. Otherwise the codec falls back to lookup tables ("lut" mode), which is also available via force_lut.
  * An escape is one u16 record: 10-bit chunk-local position | 5-bit raw exponent.

Payload = planes (5/6/7 bytes per 8 values) + 2 bytes per escape + 2 bytes per 1024-value chunk.
The kernels code whole groups of 8 chunks (8192 values); a shorter remainder is stored raw.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

CHUNK = 1024          # values per chunk; escape position needs 10 bits
LANES = CHUNK // 8    # one lane = 8 FP8 values = one u64


# ----------------------------------------------------------------------------- SWAR helpers
@triton.jit
def _pack(y, BITS: tl.constexpr):
    """Compress eight BITS-wide fields held in the low bits of each byte into one contiguous field."""
    if BITS == 4:
        y = (y | (y >> 4)) & 0x00FF00FF00FF00FF
        y = (y | (y >> 8)) & 0x0000FFFF0000FFFF
        y = (y | (y >> 16)) & 0x00000000FFFFFFFF
    elif BITS == 3:
        y = (y | (y >> 5)) & 0x003F003F003F003F
        y = (y | (y >> 10)) & 0x00000FFF00000FFF
        y = (y | (y >> 20)) & 0x0000000000FFFFFF
    else:
        y = (y | (y >> 6)) & 0x000F000F000F000F
        y = (y | (y >> 12)) & 0x000000FF000000FF
        y = (y | (y >> 24)) & 0x000000000000FFFF
    return y


@triton.jit
def _unpack(w, BITS: tl.constexpr):
    """Inverse of _pack: spread eight BITS-wide fields to the low bits of each byte."""
    if BITS == 4:
        w = (w | (w << 16)) & 0x0000FFFF0000FFFF
        w = (w | (w << 8)) & 0x00FF00FF00FF00FF
        w = (w | (w << 4)) & 0x0F0F0F0F0F0F0F0F
    elif BITS == 3:
        w = (w | (w << 20)) & 0x00000FFF00000FFF
        w = (w | (w << 10)) & 0x003F003F003F003F
        w = (w | (w << 5)) & 0x0707070707070707
    else:
        w = (w | (w << 24)) & 0x000000FF000000FF
        w = (w | (w << 12)) & 0x000F000F000F000F
        w = (w | (w << 6)) & 0x0303030303030303
    return w


@triton.jit
def _split_fields(x, E4M3: tl.constexpr):
    """x: 8 FP8 bytes. Returns (packed sign|mantissa field, exponent in each byte)."""
    if E4M3:      # S EEEE MMM
        smb = (x & 0x0707070707070707) | ((x >> 4) & 0x0808080808080808)
        e = (x >> 3) & 0x0F0F0F0F0F0F0F0F
        sm = _pack(smb, 4)
    else:         # S EEEEE MM
        smb = (x & 0x0303030303030303) | ((x >> 5) & 0x0404040404040404)
        e = (x >> 2) & 0x1F1F1F1F1F1F1F1F
        sm = _pack(smb, 3)
    return sm, e


@triton.jit
def _window_codes(e, base_vec, hi_mask, lo_mask):
    """e: exponent per byte. Returns (code per byte, escape flag in bit 7 of each byte)."""
    r = (e | 0x8080808080808080) - base_vec            # per byte: 0x80 + e - base, never borrows
    t = (r ^ 0x8080808080808080) & hi_mask              # nonzero byte <=> outside [0, K)
    z = (((t & 0x7F7F7F7F7F7F7F7F) + 0x7F7F7F7F7F7F7F7F) | t) & 0x8080808080808080
    return r & lo_mask, z


@triton.jit
def _lut_codes(e, lut, lo_mask):
    """Table lookup, one value at a time. lut[e] = code, or 0x80 for a non-member."""
    c = tl.zeros_like(e)
    for k in tl.static_range(8):
        ck = tl.load(lut + ((e >> (8 * k)) & 0x1F).to(tl.int32)).to(tl.uint64)
        c |= ck << (8 * k)
    return c & lo_mask, c & 0x8080808080808080


# ----------------------------------------------------------------------------- encode
@triton.jit
def _fp8_encode_pass1(inp64, p32, p16, p8, counts, lut, base_vec, hi_mask, lo_mask,
                      E4M3: tl.constexpr, CB: tl.constexpr, TB: tl.constexpr, WINDOW: tl.constexpr,
                      LANES: tl.constexpr, CPP: tl.constexpr):
    """One program packs CPP chunks (CPP * LANES u64 lanes) and writes one escape count per chunk.
    counts is padded to a multiple of CPP; the input is always a whole number of programs."""
    pid = tl.program_id(0)
    lane = pid * (LANES * CPP) + tl.arange(0, LANES * CPP)
    x = tl.load(inp64 + lane).to(tl.uint64)
    sm, e = _split_fields(x, E4M3)
    if WINDOW:
        c, z = _window_codes(e, base_vec, hi_mask, lo_mask)
    else:
        c, z = _lut_codes(e, lut, lo_mask)
    code = _pack(c, CB)
    if E4M3:
        w = sm | (code << 32)
    else:
        w = sm | (code << 24)
    tl.store(p32 + lane, (w & 0xFFFFFFFF).to(tl.int32))
    if TB == 40:
        tl.store(p8 + lane, ((w >> 32) & 0xFF).to(tl.uint8))
    if TB == 48:
        tl.store(p16 + lane, ((w >> 32) & 0xFFFF).to(tl.int16))
    if TB == 56:
        tl.store(p16 + lane, ((w >> 32) & 0xFFFF).to(tl.int16))
        tl.store(p8 + lane, ((w >> 48) & 0xFF).to(tl.uint8))
    cnt = (((z >> 7) * 0x0101010101010101) >> 56).to(tl.int32)
    per_chunk = tl.sum(tl.reshape(cnt, (CPP, LANES)), axis=1)
    tl.store(counts + pid * CPP + tl.arange(0, CPP), per_chunk)


@triton.jit
def _fp8_encode_pass2(inp64, counts, starts, esc16, lut, base_vec, hi_mask, lo_mask,
                      E4M3: tl.constexpr, WINDOW: tl.constexpr, LANES: tl.constexpr, CPP: tl.constexpr):
    pid = tl.program_id(0)
    lofs = tl.arange(0, LANES)
    for j in tl.static_range(CPP):
        ch = pid * CPP + j
        total = tl.load(counts + ch)
        if total > 0:                               # block-uniform: escape-free chunks are never re-read
            x = tl.load(inp64 + ch * LANES + lofs).to(tl.uint64)
            if E4M3:
                e = (x >> 3) & 0x0F0F0F0F0F0F0F0F
            else:
                e = (x >> 2) & 0x1F1F1F1F1F1F1F1F
            if WINDOW:
                c, z = _window_codes(e, base_vec, hi_mask, lo_mask)
            else:
                c, z = _lut_codes(e, lut, lo_mask)
            lane_cnt = (((z >> 7) * 0x0101010101010101) >> 56).to(tl.int32)
            pos = tl.load(starts + ch) + tl.cumsum(lane_cnt, 0) - lane_cnt
            for k in tl.static_range(8):
                fk = ((z >> (8 * k + 7)) & 1) != 0
                rec = (((lofs * 8 + k) << 5) | ((e >> (8 * k)) & 0x1F).to(tl.int32))
                tl.store(esc16 + pos, rec.to(tl.int16), mask=fk)
                pos += fk.to(tl.int32)


# ----------------------------------------------------------------------------- decode
@triton.jit
def _fp8_decode_dense(p32, p16, p8, out64, dlut, base_vec,
                      E4M3: tl.constexpr, CB: tl.constexpr, TB: tl.constexpr, WINDOW: tl.constexpr,
                      LANES: tl.constexpr):
    lane = tl.program_id(0) * LANES + tl.arange(0, LANES)
    w = tl.load(p32 + lane).to(tl.uint64) & 0xFFFFFFFF
    if TB == 40:
        w |= (tl.load(p8 + lane).to(tl.uint64) & 0xFF) << 32
    if TB == 48:
        w |= (tl.load(p16 + lane).to(tl.uint64) & 0xFFFF) << 32
    if TB == 56:
        w |= (tl.load(p16 + lane).to(tl.uint64) & 0xFFFF) << 32
        w |= (tl.load(p8 + lane).to(tl.uint64) & 0xFF) << 48
    if E4M3:
        smb = _unpack(w & 0xFFFFFFFF, 4)
        cb = _unpack(w >> 32, CB)
    else:
        smb = _unpack(w & 0xFFFFFF, 3)
        cb = _unpack(w >> 24, CB)
    if WINDOW:
        eb = cb + base_vec
    else:
        eb = tl.zeros_like(cb)
        for k in tl.static_range(8):
            ek = tl.load(dlut + ((cb >> (8 * k)) & 0xFF).to(tl.int32)).to(tl.uint64)
            eb |= ek << (8 * k)
    if E4M3:
        out = ((smb & 0x0808080808080808) << 4) | (eb << 3) | (smb & 0x0707070707070707)
    else:
        out = ((smb & 0x0404040404040404) << 5) | (eb << 2) | (smb & 0x0303030303030303)
    tl.store(out64 + lane, out)


@triton.jit
def _fp8_fix_escapes(esc16, counts, starts, out8, E_SHIFT: tl.constexpr, KEEP: tl.constexpr,
                     CHUNK: tl.constexpr, BLOCK: tl.constexpr, CPP: tl.constexpr):
    pid = tl.program_id(0)
    for j in tl.static_range(CPP):
        ch = pid * CPP + j
        cnt = tl.load(counts + ch)
        if cnt > 0:
            start = tl.load(starts + ch)
            for i in range(0, cnt, BLOCK):
                idx = i + tl.arange(0, BLOCK)
                m = idx < cnt
                rec = tl.load(esc16 + start + idx, mask=m, other=0).to(tl.int32) & 0xFFFF
                pos = ch * CHUNK + (rec >> 5)
                b = tl.load(out8 + pos, mask=m, other=0).to(tl.int32)
                tl.store(out8 + pos, ((b & KEEP) | ((rec & 31) << E_SHIFT)).to(tl.uint8), mask=m)


# ----------------------------------------------------------------------------- python API
@dataclass
class FP8Encoded:
    p32: torch.Tensor
    p16: Optional[torch.Tensor]
    p8: Optional[torch.Tensor]
    counts: torch.Tensor      # int32 per chunk (transmitted as u16)
    starts: torch.Tensor      # prefix sum of counts; recomputed by the receiver, not transmitted
    esc: torch.Tensor         # int16 records: (local position << 5) | raw exponent
    tail: torch.Tensor        # < CHUNK trailing bytes, stored raw
    n: int
    n_esc: int

    @property
    def compressed_bytes(self) -> int:
        b = self.p32.numel() * 4 + self.esc.numel() * 2 + self.counts.numel() * 2 + self.tail.numel()
        if self.p16 is not None:
            b += self.p16.numel() * 2
        if self.p8 is not None:
            b += self.p8.numel()
        return b


class SplitZipFP8GPU:
    """SplitZip codec for FP8 tensors. fmt in {"e4m3","e5m2"}, code_bits in {2,3,4} (4-bit: e5m2 only).

    encode() accepts any one-byte-per-value CUDA tensor (torch.float8_e4m3fn, torch.float8_e5m2 or
    their uint8 view); decode() returns the original bytes as a flat uint8 tensor."""

    def __init__(self, fmt: str, code_bits: int, device: str = "cuda", force_lut: bool = False,
                 num_warps: int = 4, p1_cpp: int = 8, p2_cpp: int = 8, fix_cpp: int = 8,
                 dec_lanes: int = 1024, dec_warps: int = 4):
        if fmt not in ("e4m3", "e5m2"):
            raise ValueError(fmt)
        self.e4m3 = fmt == "e4m3"
        self.exp_bits = 4 if self.e4m3 else 5
        self.man_bits = 3 if self.e4m3 else 2
        if code_bits not in (2, 3, 4) or code_bits >= self.exp_bits:
            raise ValueError(f"{code_bits}-bit codes do not compress the {self.exp_bits}-bit {fmt} exponent")
        self.fmt, self.cb, self.K = fmt, code_bits, 1 << code_bits
        self.device, self.force_lut, self.num_warps = device, force_lut, num_warps
        self.p1_cpp, self.p2_cpp, self.fix_cpp = p1_cpp, p2_cpp, fix_cpp
        self.dec_lanes, self.dec_warps = dec_lanes, dec_warps
        self.super = 8                      # the kernel-coded body is a whole number of 8-chunk groups
        for v in (p1_cpp, p2_cpp, fix_cpp):
            assert self.super % v == 0
        assert (self.super * LANES) % dec_lanes == 0
        self.total_bits = (8 - self.exp_bits) + code_bits            # per value
        self.plane16 = self.total_bits in (6, 7)
        self.plane8 = self.total_bits in (5, 7)
        self.codebook = None

    # -- calibration: top-k exponents by frequency, as in the BF16 codec --
    def calibrate(self, sample: torch.Tensor) -> float:
        b = sample.contiguous().view(torch.uint8).reshape(-1).to(self.device)
        e = (b.to(torch.int32) >> self.man_bits) & ((1 << self.exp_bits) - 1)
        hist = torch.bincount(e, minlength=1 << self.exp_bits)
        top = torch.topk(hist, self.K).indices
        members = sorted(top.tolist())
        self.codebook = members
        self.window = (members == list(range(members[0], members[0] + self.K))) and not self.force_lut
        self.base = members[0]
        rep = lambda v: int.from_bytes(bytes([v]) * 8, "little")
        self.base_vec = rep(self.base)
        self.hi_mask = rep(0xFF ^ (self.K - 1))
        self.lo_mask = rep(self.K - 1)
        enc = torch.full((32,), 0x80, dtype=torch.uint8)
        dec = torch.zeros(256, dtype=torch.uint8)
        for code, val in enumerate(members):
            enc[val] = code
            dec[code] = val
        self.enc_lut, self.dec_lut = enc.to(self.device), dec.to(self.device)
        return float(hist[top].sum() / hist.sum())

    def _consts(self):
        return dict(E4M3=self.e4m3, WINDOW=self.window, LANES=LANES, num_warps=self.num_warps)

    def _nch(self, n: int) -> int:
        return (n // (CHUNK * self.super)) * self.super

    def encode(self, tensor: torch.Tensor) -> FP8Encoded:
        if self.codebook is None:
            raise RuntimeError("codec must be calibrated")
        flat = tensor.contiguous().view(torch.uint8).reshape(-1)
        n = flat.numel()
        nch = self._nch(n)
        body = flat[: nch * CHUNK]
        dev = self.device
        nl = nch * LANES
        p32 = torch.empty(nl, dtype=torch.int32, device=dev)
        p16 = torch.empty(nl, dtype=torch.int16, device=dev) if self.plane16 else None
        p8 = torch.empty(nl, dtype=torch.uint8, device=dev) if self.plane8 else None
        counts = torch.empty(nch, dtype=torch.int32, device=dev)
        inp64 = body.view(torch.int64)
        if nch:
            _fp8_encode_pass1[(nch // self.p1_cpp,)](
                inp64, p32, p16 if p16 is not None else p32, p8 if p8 is not None else p32,
                counts, self.enc_lut, self.base_vec, self.hi_mask, self.lo_mask,
                CB=self.cb, TB=8 * self.total_bits, CPP=self.p1_cpp, **self._consts())
        offsets = torch.cumsum(counts, 0)
        starts = (offsets - counts).to(torch.int32)
        n_esc = int(offsets[-1].item()) if nch else 0
        esc = torch.empty(n_esc, dtype=torch.int16, device=dev)
        if n_esc:
            _fp8_encode_pass2[(nch // self.p2_cpp,)](inp64, counts, starts, esc, self.enc_lut, self.base_vec,
                                                     self.hi_mask, self.lo_mask, CPP=self.p2_cpp, **self._consts())
        return FP8Encoded(p32, p16, p8, counts, starts, esc, flat[nch * CHUNK:].clone(), n, n_esc)

    def decode(self, enc: FP8Encoded) -> torch.Tensor:
        out = torch.empty(enc.n, dtype=torch.uint8, device=self.device)
        nch = self._nch(enc.n)
        if nch:
            _fp8_decode_dense[(nch * LANES // self.dec_lanes,)](
                enc.p32, enc.p16 if enc.p16 is not None else enc.p32,
                enc.p8 if enc.p8 is not None else enc.p32,
                out[: nch * CHUNK].view(torch.int64), self.dec_lut, self.base_vec,
                CB=self.cb, TB=8 * self.total_bits, E4M3=self.e4m3, WINDOW=self.window,
                LANES=self.dec_lanes, num_warps=self.dec_warps)
            if enc.n_esc:
                keep = 0xFF ^ (((1 << self.exp_bits) - 1) << self.man_bits)
                _fp8_fix_escapes[(nch // self.fix_cpp,)](enc.esc, enc.counts, enc.starts, out,
                                                         E_SHIFT=self.man_bits, KEEP=keep, CHUNK=CHUNK,
                                                         BLOCK=128, CPP=self.fix_cpp, num_warps=2)
        if enc.tail.numel():
            out[nch * CHUNK:] = enc.tail
        return out
