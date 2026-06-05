from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import triton
import triton.language as tl

@triton.jit
def _enc_4bit_quad64(inp64, lut, pk16, sm32, n_quads, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_quads
    raw_pack = tl.load(inp64 + offs, mask=mask, other=0).to(tl.uint64)
    raw0 = (raw_pack & 0xFFFF).to(tl.int32)
    raw1 = ((raw_pack >> 16) & 0xFFFF).to(tl.int32)
    raw2 = ((raw_pack >> 32) & 0xFFFF).to(tl.int32)
    raw3 = ((raw_pack >> 48) & 0xFFFF).to(tl.int32)

    code0 = tl.load(lut + ((raw0 >> 7) & 0xFF), mask=mask, other=0).to(tl.int32)
    code1 = tl.load(lut + ((raw1 >> 7) & 0xFF), mask=mask, other=0).to(tl.int32)
    code2 = tl.load(lut + ((raw2 >> 7) & 0xFF), mask=mask, other=0).to(tl.int32)
    code3 = tl.load(lut + ((raw3 >> 7) & 0xFF), mask=mask, other=0).to(tl.int32)

    pk0 = ((code0 << 4) | code1) & 0xFF
    pk1 = ((code2 << 4) | code3) & 0xFF
    tl.store(pk16 + offs, (pk0 | (pk1 << 8)).to(tl.uint16), mask=mask)

    sm0 = ((raw0 >> 8) & 0x80) | (raw0 & 0x7F)
    sm1 = ((raw1 >> 8) & 0x80) | (raw1 & 0x7F)
    sm2 = ((raw2 >> 8) & 0x80) | (raw2 & 0x7F)
    sm3 = ((raw3 >> 8) & 0x80) | (raw3 & 0x7F)
    sm_pack = sm0 | (sm1 << 8) | (sm2 << 16) | (sm3 << 24)
    tl.store(sm32 + offs, sm_pack, mask=mask)


@triton.jit
def _enc_4bit_quad64_count_chunk(inp64, marked_lut, pk16, sm32, counts,
                                 n: tl.constexpr, CHUNK: tl.constexpr, BLOCK_QUADS: tl.constexpr):
    pid = tl.program_id(0)
    q_base = pid * BLOCK_QUADS
    qoffs = q_base + tl.arange(0, BLOCK_QUADS)
    elem_base = qoffs * 4
    mask0 = elem_base < n
    mask1 = elem_base + 1 < n
    mask2 = elem_base + 2 < n
    mask3 = elem_base + 3 < n
    mask_pack = mask0

    raw_pack = tl.load(inp64 + qoffs, mask=mask_pack, other=0).to(tl.uint64)
    raw0 = (raw_pack & 0xFFFF).to(tl.int32)
    raw1 = ((raw_pack >> 16) & 0xFFFF).to(tl.int32)
    raw2 = ((raw_pack >> 32) & 0xFFFF).to(tl.int32)
    raw3 = ((raw_pack >> 48) & 0xFFFF).to(tl.int32)

    exp0 = (raw0 >> 7) & 0xFF
    exp1 = (raw1 >> 7) & 0xFF
    exp2 = (raw2 >> 7) & 0xFF
    exp3 = (raw3 >> 7) & 0xFF
    code0 = tl.load(marked_lut + exp0, mask=mask0, other=16).to(tl.int32)
    code1 = tl.load(marked_lut + exp1, mask=mask1, other=16).to(tl.int32)
    code2 = tl.load(marked_lut + exp2, mask=mask2, other=16).to(tl.int32)
    code3 = tl.load(marked_lut + exp3, mask=mask3, other=16).to(tl.int32)

    pk0 = (((code0 & 0x0F) << 4) | (code1 & 0x0F)) & 0xFF
    pk1 = (((code2 & 0x0F) << 4) | (code3 & 0x0F)) & 0xFF
    tl.store(pk16 + qoffs, (pk0 | (pk1 << 8)).to(tl.uint16), mask=mask_pack)

    sm0 = ((raw0 >> 8) & 0x80) | (raw0 & 0x7F)
    sm1 = ((raw1 >> 8) & 0x80) | (raw1 & 0x7F)
    sm2 = ((raw2 >> 8) & 0x80) | (raw2 & 0x7F)
    sm3 = ((raw3 >> 8) & 0x80) | (raw3 & 0x7F)
    sm_pack = sm0 | (sm1 << 8) | (sm2 << 16) | (sm3 << 24)
    tl.store(sm32 + qoffs, sm_pack, mask=mask_pack)

    esc_count = (
        tl.sum(((code0 > 15) & mask0).to(tl.int32), axis=0)
        + tl.sum(((code1 > 15) & mask1).to(tl.int32), axis=0)
        + tl.sum(((code2 > 15) & mask2).to(tl.int32), axis=0)
        + tl.sum(((code3 > 15) & mask3).to(tl.int32), axis=0)
    )
    tl.store(counts + pid, esc_count)


@triton.jit
def _enc_4bit_quad64_count2_chunks(inp64, marked_lut, pk16, sm32, counts,
                                   n: tl.constexpr, n_chunks: tl.constexpr, BLOCK_QUADS: tl.constexpr):
    pid = tl.program_id(0)
    q_base = pid * BLOCK_QUADS
    po = tl.arange(0, BLOCK_QUADS)
    qoffs = q_base + po
    elem_base = qoffs * 4
    mask0 = elem_base < n
    mask1 = elem_base + 1 < n
    mask2 = elem_base + 2 < n
    mask3 = elem_base + 3 < n

    raw_pack = tl.load(inp64 + qoffs, mask=mask0, other=0).to(tl.uint64)
    raw0 = (raw_pack & 0xFFFF).to(tl.int32)
    raw1 = ((raw_pack >> 16) & 0xFFFF).to(tl.int32)
    raw2 = ((raw_pack >> 32) & 0xFFFF).to(tl.int32)
    raw3 = ((raw_pack >> 48) & 0xFFFF).to(tl.int32)

    exp0 = (raw0 >> 7) & 0xFF
    exp1 = (raw1 >> 7) & 0xFF
    exp2 = (raw2 >> 7) & 0xFF
    exp3 = (raw3 >> 7) & 0xFF
    code0 = tl.load(marked_lut + exp0, mask=mask0, other=16).to(tl.int32)
    code1 = tl.load(marked_lut + exp1, mask=mask1, other=16).to(tl.int32)
    code2 = tl.load(marked_lut + exp2, mask=mask2, other=16).to(tl.int32)
    code3 = tl.load(marked_lut + exp3, mask=mask3, other=16).to(tl.int32)

    pk0 = (((code0 & 0x0F) << 4) | (code1 & 0x0F)) & 0xFF
    pk1 = (((code2 & 0x0F) << 4) | (code3 & 0x0F)) & 0xFF
    tl.store(pk16 + qoffs, (pk0 | (pk1 << 8)).to(tl.uint16), mask=mask0)

    sm0 = ((raw0 >> 8) & 0x80) | (raw0 & 0x7F)
    sm1 = ((raw1 >> 8) & 0x80) | (raw1 & 0x7F)
    sm2 = ((raw2 >> 8) & 0x80) | (raw2 & 0x7F)
    sm3 = ((raw3 >> 8) & 0x80) | (raw3 & 0x7F)
    sm_pack = sm0 | (sm1 << 8) | (sm2 << 16) | (sm3 << 24)
    tl.store(sm32 + qoffs, sm_pack, mask=mask0)

    esc = (
        ((code0 > 15) & mask0).to(tl.int32)
        + ((code1 > 15) & mask1).to(tl.int32)
        + ((code2 > 15) & mask2).to(tl.int32)
        + ((code3 > 15) & mask3).to(tl.int32)
    )
    first = po < (BLOCK_QUADS // 2)
    count0 = tl.sum(tl.where(first, esc, 0), axis=0)
    count1 = tl.sum(tl.where(first, 0, esc), axis=0)
    chunk0 = pid * 2
    tl.store(counts + chunk0, count0, mask=chunk0 < n_chunks)
    tl.store(counts + chunk0 + 1, count1, mask=chunk0 + 1 < n_chunks)


@triton.jit
def _enc_4bit(inp, lut, pk, sm, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    po = tl.arange(0, BLOCK)
    base = pid * BLOCK * 4
    for step in range(4):
        pi = base + step * BLOCK + po
        ei = pi * 2
        oi = ei + 1
        em = ei < n
        om = oi < n
        v0 = tl.load(inp + ei, mask=em, other=0).to(tl.int16)
        v1 = tl.load(inp + oi, mask=om, other=0).to(tl.int16)
        i0 = tl.load(lut + ((v0 >> 7) & 0xFF).to(tl.int32), mask=em, other=0).to(tl.uint8)
        i1 = tl.load(lut + ((v1 >> 7) & 0xFF).to(tl.int32), mask=om, other=0).to(tl.uint8)
        tl.store(pk + pi, (i0 << 4) | i1, mask=em)
        tl.store(sm + ei, (((v0 >> 8) & 0x80) | (v0 & 0x7F)).to(tl.uint8), mask=em)
        tl.store(sm + oi, (((v1 >> 8) & 0x80) | (v1 & 0x7F)).to(tl.uint8), mask=om)


@triton.jit
def _dec_4bit(pk, sm, dlut, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    po = tl.arange(0, BLOCK)
    base = pid * BLOCK * 4
    for step in range(4):
        pi = base + step * BLOCK + po
        ei = pi * 2
        oi = ei + 1
        em = ei < n
        om = oi < n
        packed = tl.load(pk + pi, mask=em, other=0)
        e0 = tl.load(dlut + ((packed >> 4) & 0x0F).to(tl.int32), mask=em, other=0).to(tl.int16)
        e1 = tl.load(dlut + (packed & 0x0F).to(tl.int32), mask=om, other=0).to(tl.int16)
        s0 = tl.load(sm + ei, mask=em, other=0).to(tl.int16)
        s1 = tl.load(sm + oi, mask=om, other=0).to(tl.int16)
        tl.store(out + ei, ((s0 & 0x80) << 8) | (e0 << 7) | (s0 & 0x7F), mask=em)
        tl.store(out + oi, ((s1 & 0x80) << 8) | (e1 << 7) | (s1 & 0x7F), mask=om)


@triton.jit
def _dec_4bit_pair32(pk, sm, dlut, out32, n_pairs, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs
    packed = tl.load(pk + offs, mask=mask, other=0).to(tl.int32)
    pos0 = offs * 2
    pos1 = pos0 + 1

    exp0 = tl.load(dlut + ((packed >> 4) & 0x0F), mask=mask, other=0).to(tl.int32)
    exp1 = tl.load(dlut + (packed & 0x0F), mask=mask, other=0).to(tl.int32)
    sm0 = tl.load(sm + pos0, mask=mask, other=0).to(tl.int32)
    sm1 = tl.load(sm + pos1, mask=mask, other=0).to(tl.int32)

    raw0 = ((sm0 & 0x80) << 8) | (exp0 << 7) | (sm0 & 0x7F)
    raw1 = ((sm1 & 0x80) << 8) | (exp1 << 7) | (sm1 & 0x7F)
    pair = (raw0 & 0xFFFF) | ((raw1 & 0xFFFF) << 16)
    tl.store(out32 + offs, pair, mask=mask)


@triton.jit
def _dec_4bit_quad64(pk16, sm32, dlut, out64, n_quads, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_quads
    packed = tl.load(pk16 + offs, mask=mask, other=0).to(tl.int32)
    p0 = packed & 0xFF
    p1 = (packed >> 8) & 0xFF

    sm_pack = tl.load(sm32 + offs, mask=mask, other=0).to(tl.int32)
    sm0 = sm_pack & 0xFF
    sm1 = (sm_pack >> 8) & 0xFF
    sm2 = (sm_pack >> 16) & 0xFF
    sm3 = (sm_pack >> 24) & 0xFF

    exp0 = tl.load(dlut + ((p0 >> 4) & 0x0F), mask=mask, other=0).to(tl.int32)
    exp1 = tl.load(dlut + (p0 & 0x0F), mask=mask, other=0).to(tl.int32)
    exp2 = tl.load(dlut + ((p1 >> 4) & 0x0F), mask=mask, other=0).to(tl.int32)
    exp3 = tl.load(dlut + (p1 & 0x0F), mask=mask, other=0).to(tl.int32)

    raw0 = ((sm0 & 0x80) << 8) | (exp0 << 7) | (sm0 & 0x7F)
    raw1 = ((sm1 & 0x80) << 8) | (exp1 << 7) | (sm1 & 0x7F)
    raw2 = ((sm2 & 0x80) << 8) | (exp2 << 7) | (sm2 & 0x7F)
    raw3 = ((sm3 & 0x80) << 8) | (exp3 << 7) | (sm3 & 0x7F)

    quad = (
        raw0.to(tl.uint64)
        | (raw1.to(tl.uint64) << 16)
        | (raw2.to(tl.uint64) << 32)
        | (raw3.to(tl.uint64) << 48)
    )
    tl.store(out64 + offs, quad, mask=mask)


@triton.jit
def _decode_exp_pair16(pk, dlut, exp16_out, n_pairs, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs
    packed = tl.load(pk + offs, mask=mask, other=0).to(tl.int32)
    exp0 = tl.load(dlut + ((packed >> 4) & 0x0F), mask=mask, other=0).to(tl.int32)
    exp1 = tl.load(dlut + (packed & 0x0F), mask=mask, other=0).to(tl.int32)
    pair = (exp0 & 0xFF) | ((exp1 & 0xFF) << 8)
    tl.store(exp16_out + offs, pair.to(tl.uint16), mask=mask)


@triton.jit
def _decode_exp_scalar(pk, dlut, exp_out, n, n_pairs, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs
    packed = tl.load(pk + offs, mask=mask, other=0).to(tl.int32)
    pos0 = offs * 2
    pos1 = pos0 + 1
    exp0 = tl.load(dlut + ((packed >> 4) & 0x0F), mask=mask, other=0)
    exp1 = tl.load(dlut + (packed & 0x0F), mask=mask, other=0)
    tl.store(exp_out + pos0, exp0, mask=pos0 < n)
    tl.store(exp_out + pos1, exp1, mask=pos1 < n)


@triton.jit
def _decode_exp_quad32(pk16, dlut, exp32_out, n_quads, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_quads
    packed = tl.load(pk16 + offs, mask=mask, other=0).to(tl.int32)
    p0 = packed & 0xFF
    p1 = (packed >> 8) & 0xFF

    exp0 = tl.load(dlut + ((p0 >> 4) & 0x0F), mask=mask, other=0).to(tl.int32)
    exp1 = tl.load(dlut + (p0 & 0x0F), mask=mask, other=0).to(tl.int32)
    exp2 = tl.load(dlut + ((p1 >> 4) & 0x0F), mask=mask, other=0).to(tl.int32)
    exp3 = tl.load(dlut + (p1 & 0x0F), mask=mask, other=0).to(tl.int32)

    quad = (exp0 & 0xFF) | ((exp1 & 0xFF) << 8) | ((exp2 & 0xFF) << 16) | ((exp3 & 0xFF) << 24)
    tl.store(exp32_out + offs, quad, mask=mask)


@triton.jit
def _fix_escape_exponents_local_linear(chunk_id, local_pos, esc_val, exp_out,
                                       n_esc, CHUNK: tl.constexpr, BLOCK_ESC: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_ESC + tl.arange(0, BLOCK_ESC)
    mask = offs < n_esc
    chunk = tl.load(chunk_id + offs, mask=mask, other=0).to(tl.int32)
    local = tl.load(local_pos + offs, mask=mask, other=0).to(tl.int32)
    exp = tl.load(esc_val + offs, mask=mask, other=0)
    pos = chunk * CHUNK + local
    tl.store(exp_out + pos, exp, mask=mask)


@triton.jit
def _merge_exp_sm_pair32(exp16, sm, out32, n_pairs, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_pairs
    exp_pack = tl.load(exp16 + offs, mask=mask, other=0).to(tl.int32)
    pos0 = offs * 2
    pos1 = pos0 + 1
    exp0 = exp_pack & 0xFF
    exp1 = (exp_pack >> 8) & 0xFF
    sm0 = tl.load(sm + pos0, mask=mask, other=0).to(tl.int32)
    sm1 = tl.load(sm + pos1, mask=mask, other=0).to(tl.int32)

    raw0 = ((sm0 & 0x80) << 8) | (exp0 << 7) | (sm0 & 0x7F)
    raw1 = ((sm1 & 0x80) << 8) | (exp1 << 7) | (sm1 & 0x7F)
    pair = (raw0 & 0xFFFF) | ((raw1 & 0xFFFF) << 16)
    tl.store(out32 + offs, pair, mask=mask)


@triton.jit
def _merge_exp_sm_scalar(exp, sm, out, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    e = tl.load(exp + offs, mask=mask, other=0).to(tl.int32)
    s = tl.load(sm + offs, mask=mask, other=0).to(tl.int32)
    raw = ((s & 0x80) << 8) | (e << 7) | (s & 0x7F)
    tl.store(out + offs, raw.to(tl.int16), mask=mask)


@triton.jit
def _merge_exp_sm_quad64(exp32, sm32, out64, n_quads, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_quads
    exp_pack = tl.load(exp32 + offs, mask=mask, other=0).to(tl.int32)
    sm_pack = tl.load(sm32 + offs, mask=mask, other=0).to(tl.int32)

    exp0 = exp_pack & 0xFF
    exp1 = (exp_pack >> 8) & 0xFF
    exp2 = (exp_pack >> 16) & 0xFF
    exp3 = (exp_pack >> 24) & 0xFF
    sm0 = sm_pack & 0xFF
    sm1 = (sm_pack >> 8) & 0xFF
    sm2 = (sm_pack >> 16) & 0xFF
    sm3 = (sm_pack >> 24) & 0xFF

    raw0 = ((sm0 & 0x80) << 8) | (exp0 << 7) | (sm0 & 0x7F)
    raw1 = ((sm1 & 0x80) << 8) | (exp1 << 7) | (sm1 & 0x7F)
    raw2 = ((sm2 & 0x80) << 8) | (exp2 << 7) | (sm2 & 0x7F)
    raw3 = ((sm3 & 0x80) << 8) | (exp3 << 7) | (sm3 & 0x7F)

    quad = (
        raw0.to(tl.uint64)
        | (raw1.to(tl.uint64) << 16)
        | (raw2.to(tl.uint64) << 32)
        | (raw3.to(tl.uint64) << 48)
    )
    tl.store(out64 + offs, quad, mask=mask)


@triton.jit
def _count_escapes_chunk(inp, common_lut, counts, n: tl.constexpr, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * CHUNK + tl.arange(0, CHUNK)
    mask = offs < n
    raw = tl.load(inp + offs, mask=mask, other=0).to(tl.int32)
    exp = (raw >> 7) & 0xFF
    common = tl.load(common_lut + exp, mask=mask, other=1).to(tl.int32)
    esc = (common == 0) & mask
    tl.store(counts + pid, tl.sum(esc.to(tl.int32), axis=0))


@triton.jit
def _write_escapes_chunk(inp, common_lut, starts, chunk_id, local_pos, esc_val,
                         n: tl.constexpr, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * CHUNK
    offs = base + tl.arange(0, CHUNK)
    mask = offs < n
    raw = tl.load(inp + offs, mask=mask, other=0).to(tl.int32)
    exp = (raw >> 7) & 0xFF
    common = tl.load(common_lut + exp, mask=mask, other=1).to(tl.int32)
    esc = (common == 0) & mask
    rank = tl.cumsum(esc.to(tl.int32), 0) - 1
    start = tl.load(starts + pid)
    out = start + rank
    tl.store(chunk_id + out, pid, mask=esc)
    tl.store(local_pos + out, (offs - base).to(tl.uint16), mask=esc)
    tl.store(esc_val + out, exp.to(tl.uint8), mask=esc)


@triton.jit
def _write_escapes_chunk_split4(inp, common_lut, starts, chunk_id, local_pos, esc_val,
                                n: tl.constexpr, CHUNK: tl.constexpr, SUB: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * CHUNK
    po = tl.arange(0, SUB)
    start = tl.load(starts + pid)
    running = tl.full((), 0, tl.int32)
    for step in tl.static_range(0, 4):
        local_base = step * SUB
        offs = base + local_base + po
        local_mask = local_base + po < CHUNK
        mask = (offs < n) & local_mask
        raw = tl.load(inp + offs, mask=mask, other=0).to(tl.int32)
        exp = (raw >> 7) & 0xFF
        common = tl.load(common_lut + exp, mask=mask, other=1).to(tl.int32)
        esc = (common == 0) & mask
        esc_i = esc.to(tl.int32)
        rank = running + tl.cumsum(esc_i, 0) - 1
        out = start + rank
        tl.store(chunk_id + out, pid, mask=esc)
        tl.store(local_pos + out, (local_base + po).to(tl.uint16), mask=esc)
        tl.store(esc_val + out, exp.to(tl.uint8), mask=esc)
        running += tl.sum(esc_i, axis=0)


@triton.jit
def _collect_escapes_chunk_atomic(inp, common_lut, chunk_id, local_pos, esc_val, esc_count,
                                  n: tl.constexpr, CHUNK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * CHUNK
    offs = base + tl.arange(0, CHUNK)
    mask = offs < n
    raw = tl.load(inp + offs, mask=mask, other=0).to(tl.int32)
    exp = (raw >> 7) & 0xFF
    common = tl.load(common_lut + exp, mask=mask, other=1).to(tl.int32)
    esc = (common == 0) & mask
    esc_i = esc.to(tl.int32)
    n_total = tl.sum(esc_i, axis=0)
    start = tl.atomic_add(esc_count, n_total)
    rank = tl.cumsum(esc_i, 0) - 1
    out = start + rank
    tl.store(chunk_id + out, pid, mask=esc)
    tl.store(local_pos + out, (offs - base).to(tl.uint16), mask=esc)
    tl.store(esc_val + out, exp.to(tl.uint8), mask=esc)


@triton.jit
def _fix_escapes_local_linear(chunk_id, local_pos, esc_val, sm, out,
                              n_esc, CHUNK: tl.constexpr, BLOCK_ESC: tl.constexpr):
    offs = tl.program_id(0) * BLOCK_ESC + tl.arange(0, BLOCK_ESC)
    mask = offs < n_esc
    chunk = tl.load(chunk_id + offs, mask=mask, other=0).to(tl.int32)
    local = tl.load(local_pos + offs, mask=mask, other=0).to(tl.int32)
    exp = tl.load(esc_val + offs, mask=mask, other=0).to(tl.int32)
    pos = chunk * CHUNK + local
    s = tl.load(sm + pos, mask=mask, other=0).to(tl.int32)
    fixed = ((s & 0x80) << 8) | (exp << 7) | (s & 0x7F)
    tl.store(out + pos, fixed.to(tl.int16), mask=mask)

@dataclass
class ChunkLocalGPUEncoded:
    pk: torch.Tensor
    sm: torch.Tensor
    counts: torch.Tensor
    starts: torch.Tensor
    chunk_id: torch.Tensor
    local_pos: torch.Tensor
    esc_val: torch.Tensor
    n: int
    n_esc: int
    chunk_size: int

    @property
    def compressed_bytes(self) -> int:
        # counts/starts are construction scratch data; decode uses compact chunk_id
        # plus local offsets, so they are not part of the transmitted payload.
        return (
            self.pk.numel()
            + self.sm.numel()
            + self.chunk_id.numel() * 4
            + self.local_pos.numel() * 2
            + self.esc_val.numel()
        )


class ChunkLocalSplitZipGPU:
    """Triton chunk-local SplitZip implementation.

    Escape collection is two-pass and lock-free at the global level:
    per-chunk count, prefix sum, per-chunk scatter into disjoint ranges.
    """

    def __init__(self, device: str = "cuda", chunk_size: int = 1024):
        if chunk_size <= 0 or chunk_size > 65536:
            raise ValueError("chunk_size must be in [1, 65536] for uint16 offsets")
        if chunk_size & (chunk_size - 1):
            raise ValueError("chunk_size must be a power of two for Triton blocks")
        self.device = device
        self.chunk_size = int(chunk_size)
        self.enc_lut: Optional[torch.Tensor] = None
        self.enc_lut_marked: Optional[torch.Tensor] = None
        self.dec_lut: Optional[torch.Tensor] = None
        self.common_lut: Optional[torch.Tensor] = None

    def calibrate(self, sample: torch.Tensor) -> float:
        flat = sample.contiguous().view(torch.int16)
        exp = ((flat >> 7) & 0xFF).to(torch.uint8)
        vals, counts = torch.unique(exp, return_counts=True)
        order = torch.argsort(counts, descending=True)
        self.enc_lut = torch.zeros(256, dtype=torch.uint8, device=self.device)
        self.enc_lut_marked = torch.full((256,), 16, dtype=torch.uint8, device=self.device)
        self.dec_lut = torch.zeros(16, dtype=torch.uint8, device=self.device)
        self.common_lut = torch.zeros(256, dtype=torch.uint8, device=self.device)
        top = min(16, vals.numel())
        for code in range(top):
            value = vals[order[code]].item()
            self.enc_lut[value] = code
            self.enc_lut_marked[value] = code
            self.dec_lut[code] = value
            self.common_lut[value] = 1
        return float(counts[order[:top]].sum().item() / counts.sum().item())

    def _check_ready(self):
        if (
            self.enc_lut is None
            or self.enc_lut_marked is None
            or self.dec_lut is None
            or self.common_lut is None
        ):
            raise RuntimeError("codec must be calibrated")

    def encode(self, tensor: torch.Tensor) -> ChunkLocalGPUEncoded:
        self._check_ready()
        flat = tensor.contiguous().view(torch.int16)
        n = int(flat.numel())
        n_pairs = (n + 1) // 2
        pk = torch.empty(n_pairs, dtype=torch.uint8, device=self.device)
        sm = torch.empty(n, dtype=torch.uint8, device=self.device)
        n_chunks = (n + self.chunk_size - 1) // self.chunk_size
        counts = torch.empty(n_chunks, dtype=torch.int32, device=self.device)

        if n % 4 == 0 and self.chunk_size == 1024:
            _enc_4bit_quad64_count2_chunks[((n_chunks + 1) // 2,)](
                flat.view(torch.int64),
                self.enc_lut_marked,
                pk.view(torch.int16),
                sm.view(torch.int32),
                counts,
                n,
                n_chunks,
                BLOCK_QUADS=self.chunk_size // 2,
                num_warps=4,
            )
        elif n % 4 == 0:
            block = 512
            _enc_4bit_quad64[((n // 4 + block - 1) // block,)](
                flat.view(torch.int64),
                self.enc_lut,
                pk.view(torch.int16),
                sm.view(torch.int32),
                n // 4,
                BLOCK=block,
                num_warps=4,
            )
            _count_escapes_chunk[(n_chunks,)](
                flat, self.common_lut, counts, n, CHUNK=self.chunk_size
            )
        else:
            block = 256
            _enc_4bit[((n_pairs + block * 4 - 1) // (block * 4),)](
                flat, self.enc_lut, pk, sm, n, BLOCK=block
            )
            _count_escapes_chunk[(n_chunks,)](
                flat, self.common_lut, counts, n, CHUNK=self.chunk_size
            )

        offsets = torch.cumsum(counts, dim=0)
        starts = offsets - counts
        n_esc = int(offsets[-1].item()) if offsets.numel() else 0
        chunk_id = torch.empty(n_esc, dtype=torch.int32, device=self.device)
        local_pos = torch.empty(n_esc, dtype=torch.uint16, device=self.device)
        esc_val = torch.empty(n_esc, dtype=torch.uint8, device=self.device)
        if n_esc:
            _write_escapes_chunk[(n_chunks,)](
                flat, self.common_lut, starts, chunk_id, local_pos, esc_val, n, CHUNK=self.chunk_size
            )
        return ChunkLocalGPUEncoded(pk, sm, counts, starts, chunk_id, local_pos, esc_val, n, n_esc, self.chunk_size)

    def decode(self, encoded: ChunkLocalGPUEncoded) -> torch.Tensor:
        self._check_ready()
        n_pairs = (encoded.n + 1) // 2
        out = torch.empty(encoded.n, dtype=torch.int16, device=self.device)
        if encoded.n % 4 == 0:
            block = 512
            n_quads = encoded.n // 4
            _dec_4bit_quad64[((n_quads + block - 1) // block,)](
                encoded.pk.view(torch.int16),
                encoded.sm.view(torch.int32),
                self.dec_lut,
                out.view(torch.int64),
                n_quads,
                BLOCK=block,
                num_warps=4,
            )
        elif encoded.n % 2 == 0:
            block = 1024
            _dec_4bit_pair32[((n_pairs + block - 1) // block,)](
                encoded.pk, encoded.sm, self.dec_lut, out.view(torch.int32), n_pairs, BLOCK=block, num_warps=4
            )
        else:
            block = 1024
            _dec_4bit[((n_pairs + block * 4 - 1) // (block * 4),)](
                encoded.pk, encoded.sm, self.dec_lut, out, encoded.n, BLOCK=block, num_warps=4
            )
        if encoded.n_esc:
            block_esc = 128
            _fix_escapes_local_linear[((encoded.n_esc + block_esc - 1) // block_esc,)](
                encoded.chunk_id,
                encoded.local_pos,
                encoded.esc_val,
                encoded.sm,
                out,
                encoded.n_esc,
                CHUNK=encoded.chunk_size,
                BLOCK_ESC=block_esc,
            )
        return out.view(torch.bfloat16)

    def decode_escape_first(self, encoded: ChunkLocalGPUEncoded) -> torch.Tensor:
        self._check_ready()
        n_pairs = (encoded.n + 1) // 2
        exp = torch.empty(encoded.n, dtype=torch.uint8, device=self.device)
        out = torch.empty(encoded.n, dtype=torch.int16, device=self.device)
        if encoded.n % 4 == 0:
            block = 512
            n_quads = encoded.n // 4
            _decode_exp_quad32[((n_quads + block - 1) // block,)](
                encoded.pk.view(torch.int16),
                self.dec_lut,
                exp.view(torch.int32),
                n_quads,
                BLOCK=block,
                num_warps=4,
            )
        elif encoded.n % 2 == 0:
            block = 1024
            _decode_exp_pair16[((n_pairs + block - 1) // block,)](
                encoded.pk,
                self.dec_lut,
                exp.view(torch.int16),
                n_pairs,
                BLOCK=block,
                num_warps=4,
            )
        else:
            block = 1024
            _decode_exp_scalar[((n_pairs + block - 1) // block,)](
                encoded.pk,
                self.dec_lut,
                exp,
                encoded.n,
                n_pairs,
                BLOCK=block,
                num_warps=4,
            )
        if encoded.n_esc:
            block_esc = 128
            _fix_escape_exponents_local_linear[((encoded.n_esc + block_esc - 1) // block_esc,)](
                encoded.chunk_id,
                encoded.local_pos,
                encoded.esc_val,
                exp,
                encoded.n_esc,
                CHUNK=encoded.chunk_size,
                BLOCK_ESC=block_esc,
            )
        if encoded.n % 4 == 0:
            block = 512
            n_quads = encoded.n // 4
            _merge_exp_sm_quad64[((n_quads + block - 1) // block,)](
                exp.view(torch.int32),
                encoded.sm.view(torch.int32),
                out.view(torch.int64),
                n_quads,
                BLOCK=block,
                num_warps=4,
            )
        elif encoded.n % 2 == 0:
            block = 1024
            _merge_exp_sm_pair32[((n_pairs + block - 1) // block,)](
                exp.view(torch.int16),
                encoded.sm,
                out.view(torch.int32),
                n_pairs,
                BLOCK=block,
                num_warps=4,
            )
        else:
            block = 1024
            _merge_exp_sm_scalar[((encoded.n + block - 1) // block,)](
                exp,
                encoded.sm,
                out,
                encoded.n,
                BLOCK=block,
                num_warps=4,
            )
        return out.view(torch.bfloat16)
