"""Generic ground-truth labeler that drives tshark with -T pdml and converts
each frame's per-field byte positions into our (off,len,type,role,name,value)
schema -- the same shape that Modbus's hand-written dissector emits.

This avoids writing one dissector per protocol: we leverage Wireshark's
already-mature dissectors for S7comm, DNP3, IEC60870-5-104, etc.

Key transformations:
  1. tshark `pos`/`size` are absolute frame offsets. We rebase them to be
     relative to the application payload (i.e. start of TCP/UDP payload).
  2. PDML emits hierarchical fields (containers + leaves + bit-mask siblings).
     We keep only LEAVES that lie fully inside the app-layer slice, are not
     marked hide="yes", have size>0, and don't have any nested children
     covering the same span.
  3. Bit-mask siblings (same pos/size) are collapsed into one field --
     we keep the parent byte-level field, not the per-bit decompositions.
  4. Each field's name is mapped to one of our roles via keyword heuristics
     (consistent with LIVA's 4-class taint-mapping vocabulary, extended).
"""
from __future__ import annotations

import os
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Iterator, Optional

# ---------------------------------------------------------------------------
# Role inference: name keyword -> role label
# Order matters: first match wins.
# ---------------------------------------------------------------------------
_ROLE_KEYWORDS: list[tuple[re.Pattern, str]] = [
    # checksum first: highly specific
    (re.compile(r"(crc|checksum|chksum|\bfcs\b)", re.I),                            "checksum"),
    # length: matches apdulen, asdulen, parlg, datlg, byte_count, header_length, ...
    (re.compile(r"(length|\blen\b|\blen$|len_|_len|datlg|parlg|byte_count|apdulen|asdulen)", re.I), "length"),
    # opcode / function code / message type 鈥?also IEC104 'type' (TypeID)
    (re.compile(r"(funccode|func_code|funct|\bfunc\b|opcode|cmd_code|type_?id|typeid|asduty|message_type|rosctr|application_function|^type$|_type$|\btype\b)", re.I), "opcode"),
    # cause of transmission (IEC104) 鈥?treat as opcode-like routing info
    (re.compile(r"(\bcot\b|cause|origaddr)", re.I),                                  "opcode"),
    # address
    (re.compile(r"(address|\baddr\b|^src$|^dst$|destination|source|\bioa\b|common_address|comaddr|unit_id|^unit$|station|sta_addr)", re.I), "address"),
    # count / quantity / item count
    (re.compile(r"(qty|quant|\bcount\b|num_items|itemcount|num_objs|num_obj|number_of|numix|^nbr$)", re.I), "count"),
    # sequence / transaction / fragment
    (re.compile(r"(trans_id|transaction|pduref|pdu_ref|\bseq\b|seqnum|sequence|sendseq|recvseq|tx_seq|rx_seq|frame_seq|fragment|sendcounter|receivecounter)", re.I), "sequence"),
    # constant header markers: magic / version / reserved / etc.
    (re.compile(r"(protocol_id|prot_id|protid|magic|version|reserved|filler|padding|^tpkt$|tpkt\.|cotp\.type|^start$|start_byte|signature|preamble)", re.I), "constant"),
]


# Per-protocol overrides: raw PDML field name (with prefix) -> role.
# These take precedence over the generic keyword regex when present, fixing
# protocol-specific fields whose role can't be inferred from the name alone.
_ROLE_OVERRIDES: dict[str, str] = {
    # ------- S7comm (TPKT + COTP + S7comm) -------
    "tpkt.version": "constant",
    "tpkt.reserved": "constant",
    "tpkt.length": "length",
    "cotp.li": "length",                           # length indicator (1 byte)
    "cotp.type": "opcode",                          # COTP PDU type
    "cotp.tpdu-number": "sequence",
    "s7comm.header.protid": "constant",
    "s7comm.header.rosctr": "opcode",
    "s7comm.header.redid": "constant",
    "s7comm.header.pduref": "sequence",
    "s7comm.header.parlg": "length",
    "s7comm.header.datlg": "length",
    "s7comm.header.errcls": "opcode",
    "s7comm.header.errcod": "opcode",
    "s7comm.param.func": "opcode",
    "s7comm.param.itemcount": "count",
    "s7comm.param.item.varspec": "constant",       # 0x12 item marker
    "s7comm.param.item.varspec_length": "length",
    "s7comm.param.item.syntaxid": "constant",      # 0x10 S7ANY marker
    "s7comm.param.item.transp_size": "opcode",     # data type encoding
    "s7comm.param.item.length": "count",           # quantity to read
    "s7comm.param.item.db": "address",
    "s7comm.param.item.area": "address",
    "s7comm.param.item.address": "address",
    "s7comm.data.returncode": "opcode",
    "s7comm.data.transportsize": "opcode",
    "s7comm.data.length": "length",
    "s7comm.resp.data": "payload",
    # ------- DNP3 -------
    "dnp3.start": "constant",
    "dnp3.len": "length",
    "dnp3.ctl": "opcode",                           # link-layer ctl byte holds FC
    "dnp3.dst": "address",
    "dnp3.src": "address",
    "dnp3.hdr.CRC": "checksum",
    "dnp3.hdr_CRC": "checksum",
    "dnp3.tr.ctl": "opcode",                        # transport ctl has FIN/FIR/SEQ
    "dnp3.tr_ctl": "opcode",
    "dnp3.al.ctl": "opcode",
    "dnp3.al.func": "opcode",
    "dnp3.al.iin": "constant",
    "dnp3.al.objq.prefix": "constant",
    "dnp3.al.objq.code": "constant",
    "dnp3.al.range.quantity": "count",
    "dnp3.al.range.start": "address",
    "dnp3.al.range.stop": "address",
    "dnp3.al.index": "address",
    "dnp3.al.cnt": "count",
    "dnp3.al.aoq.b7": "constant",
    "dnp3.al.ctrq.b7": "constant",
    "dnp3.data_chunk": "payload",
    "dnp3.data_chunk_CRC": "checksum",
    # ------- IEC60870-5-104 APCI + ASDU -------
    "iec60870_104.start": "constant",
    "iec60870_104.apdulen": "length",
    "iec60870_104.type": "opcode",                  # I/S/U APCI type
    "iec60870_104.utype": "opcode",
    "iec60870_104.stype": "opcode",
    "iec60870_104.numrx": "sequence",
    "iec60870_104.numtx": "sequence",
    "iec60870_asdu.typeid": "opcode",
    "iec60870_asdu.sq": "constant",                 # structure qualifier bit
    "iec60870_asdu.numix": "count",
    "iec60870_asdu.causetx": "opcode",              # cause of transmission
    "iec60870_asdu.nega": "constant",
    "iec60870_asdu.test": "constant",
    "iec60870_asdu.oa": "address",                  # originator address
    "iec60870_asdu.addr": "address",                # ASDU common address
    "iec60870_asdu.ioa": "address",                 # information object address
    "iec60870_asdu.qds": "constant",                # quality descriptor
    "iec60870_asdu.qos": "constant",
    "iec60870_asdu.qpm": "constant",
    "iec60870_asdu.siq": "constant",
    "iec60870_asdu.diq": "constant",
    "iec60870_asdu.vti": "constant",
    "iec60870_asdu.sco": "opcode",                  # single command
    "iec60870_asdu.dco": "opcode",                  # double command
    "iec60870_asdu.normval": "payload",
    "iec60870_asdu.scalval": "payload",
    "iec60870_asdu.float": "payload",
    "iec60870_asdu.bitstring": "payload",
    "iec60870_asdu.cp56time.ms": "payload",
    "iec60870_asdu.cp56time.min": "payload",
    "iec60870_asdu.cp56time.hour": "payload",
    "iec60870_asdu.cp56time.day": "payload",
    "iec60870_asdu.cp56time.month": "payload",
    "iec60870_asdu.cp56time.year": "payload",
    # ------- CIP PCCC (EtherNet/IP + CIP + PCCC for Allen-Bradley PLCs) -------
    "enip.command": "opcode",
    "enip.length": "length",
    "enip.session": "sequence",
    "enip.status": "constant",
    "enip.context": "sequence",
    "enip.options": "constant",
    "enip.timeout": "constant",
    "enip.cpf.itemcount": "count",
    "enip.cpf.typeid": "opcode",
    "enip.cpf.length": "length",
    "enip.cpf.cai.connid": "sequence",
    "enip.sud.iface": "constant",
    "enip.fwd_open_in": "constant",
    "enip.response_to": "constant",
    "enip.time": "constant",
    "cip.service": "opcode",
    "cip.sc": "opcode",
    "cip.path_segment": "constant",
    "cip.path_segment.type": "constant",
    "cip.request_path_size": "length",
    "cip.class": "address",
    "cip.instance": "address",
    "cip.connection": "sequence",
    "cip.connid": "address",
    "cip.epath": "address",
    "cip.logical_segment.format": "constant",
    "cip.logical_segment.type": "constant",
    "cip.genstat": "constant",
    "cip.addstat_size": "length",
    "cip.seq": "sequence",
    "cip.rr": "constant",
    "cip.cm.otapi": "constant",
    "cip.cm.toapi": "constant",
    "cip.analysis.request_no_response": "constant",
    "cip.pccc.cmd.code": "opcode",
    "cip.pccc.fnc.code_0f": "opcode",
    "cip.pccc.resp.code": "opcode",
    "cip.pccc.sc": "opcode",
    "cip.pccc.cip.vend.id": "constant",
    "cip.pccc.cip.serial.num": "address",
    "cip.pccc.gs.status": "constant",
    "cip.pccc.tns.code": "sequence",
    "cip.pccc.req.id.len": "length",
    "cip.pccc.file.num": "address",
    "cip.pccc.file.type": "constant",
    "cip.pccc.element.num": "address",
    "cip.pccc.subelement.num": "address",
    "cip.pccc.byte.size": "length",
    "cip.pccc.data": "payload",
    # ------- Omron FINS -------
    "omron.tcp.magic": "constant",
    "omron.tcp.length": "length",
    "omron.tcp.command": "opcode",
    "omron.tcp.error_code": "opcode",
    "omron.icf": "constant",
    "omron.rsv": "constant",
    "omron.gct": "constant",
    "omron.dna": "address",
    "omron.da1": "address",
    "omron.da2": "address",
    "omron.sna": "address",
    "omron.sa1": "address",
    "omron.sa2": "address",
    "omron.sid": "sequence",
    "omron.command": "opcode",
    "omron.response.code": "opcode",
    "omron.response.data": "payload",
    "omron.memory.area.read": "opcode",
    "omron.memory.address": "address",
    "omron.memory.address.bits": "address",
    "omron.memory.numitems": "count",
    "omron.status": "constant",
    "omron.mode_code": "opcode",
    "omron.message": "payload",
    "omron.fals": "constant",
    "omron.error_message": "payload",
    "omron.numwords": "count",
    "omron.parameter": "payload",
    "omron.program_number": "address",
    "omron.fatal_error_data": "payload",
}


def infer_role(raw_name: str, short_name_: str, raw_size: int) -> str:
    """Resolve field role using the per-protocol override table first,
    falling back to the generic keyword regex.
    """
    if raw_name in _ROLE_OVERRIDES:
        return _ROLE_OVERRIDES[raw_name]
    for pat, role in _ROLE_KEYWORDS:
        if pat.search(short_name_):
            return role
    return "payload"


def short_name(field_name: str) -> str:
    """Strip the protocol prefix (e.g. 'mbtcp.trans_id' -> 'trans_id')."""
    if "." in field_name:
        field_name = field_name.split(".", 1)[1]
    # collapse remaining dots into underscores
    return field_name.replace(".", "_")


def infer_type(raw_size: int) -> str:
    if raw_size == 1: return "uint8"
    if raw_size == 2: return "uint16_be"
    if raw_size == 4: return "uint32_be"
    return "bytes"


# ---------------------------------------------------------------------------
# PDML parsing
# ---------------------------------------------------------------------------
@dataclass
class PdmlField:
    off: int
    length: int
    name: str          # raw PDML field name (e.g. "s7comm.header.protid")
    value: str         # PDML 'show' or 'value' attribute, prefer human-readable
    children: list["PdmlField"]   # filled during tree assembly


def _walk_fields(node: ET.Element, app_start: int, app_end: int) -> Iterator[ET.Element]:
    """Yield every <field> element under `node` that lies inside [app_start, app_end)."""
    for child in node.findall(".//field"):
        if child.get("hide") == "yes":
            continue
        try:
            pos = int(child.get("pos", "-1"))
            size = int(child.get("size", "0"))
        except ValueError:
            continue
        if size <= 0:
            continue
        if pos < app_start or pos + size > app_end:
            continue
        if child.get("name", "").startswith("_ws."):
            continue
        if not child.get("name"):
            continue
        yield child


def _extract_leaves(packet: ET.Element, app_start: int, app_end: int) -> list[PdmlField]:
    """Return leaf fields covering [app_start, app_end), in offset order.

    "Leaf" = no nested <field> with smaller size that shares the same span,
    AND not a bitfield (we keep one entry per byte span).
    """
    raw: list[tuple[int, int, str, str]] = []  # (pos, size, name, value)
    for proto in packet.findall("proto"):
        try:
            proto_pos = int(proto.get("pos", "-1"))
            proto_size = int(proto.get("size", "0"))
        except ValueError:
            continue
        # only protos whose range overlaps the app layer
        if proto_size <= 0:
            continue
        if proto_pos + proto_size <= app_start:
            continue
        if proto_pos >= app_end:
            continue

        for f in _walk_fields(proto, app_start, app_end):
            pos = int(f.get("pos"))
            size = int(f.get("size"))
            name = f.get("name") or "anon"
            value = f.get("show") or f.get("value") or ""
            raw.append((pos, size, name, value))

    if not raw:
        return []

    # Deduplicate: drop fields whose span is exactly equal to another field
    # AND has a longer name (bitfield child like x.y.z over byte x.y).
    by_span: dict[tuple[int, int], tuple[int, int, str, str]] = {}
    for entry in raw:
        pos, size, name, value = entry
        key = (pos, size)
        if key not in by_span:
            by_span[key] = entry
            continue
        existing = by_span[key]
        # Prefer the entry with the SHORTER name (fewer dots)
        if name.count(".") < existing[2].count("."):
            by_span[key] = entry

    items = sorted(by_span.values(), key=lambda x: (x[0], x[1]))

    # Keep only leaves: a field is a leaf if no STRICTLY-NARROWER field
    # is fully contained within it (i.e. there's a more specific child).
    # We do this by: for each interval, check if any other interval is
    # strictly inside it.
    intervals = [(p, p + s) for (p, s, _, _) in items]
    keep = []
    for i, (a, b) in enumerate(intervals):
        is_container = False
        for j, (c, d) in enumerate(intervals):
            if i == j:
                continue
            if a <= c and d <= b and (b - a) > (d - c):
                is_container = True
                break
        if not is_container:
            keep.append(items[i])

    # Final pass: collapse exact-overlap duplicates again (some bitfield
    # siblings may have survived; pick the shortest-name one)
    by_span2: dict[tuple[int, int], tuple[int, int, str, str]] = {}
    for entry in keep:
        key = (entry[0], entry[1])
        if key not in by_span2 or entry[2].count(".") < by_span2[key][2].count("."):
            by_span2[key] = entry
    return sorted(by_span2.values(), key=lambda x: x[0])


def _packet_payload_span(packet: ET.Element) -> Optional[tuple[int, int]]:
    """Return (app_start, app_end) byte positions in the frame for the
    application-layer payload, or None if there isn't one.
    """
    # tcp.payload or udp.payload field has pos+size = app payload range
    for f in packet.findall(".//field"):
        nm = f.get("name", "")
        if nm in ("tcp.payload", "udp.payload"):
            try:
                pos = int(f.get("pos"))
                size = int(f.get("size", "0"))
            except ValueError:
                continue
            if size > 0:
                return (pos, pos + size)
    return None


def _packet_payload_hex(packet: ET.Element, app_start: int, app_end: int) -> str:
    """Reconstruct the application payload bytes from tcp.payload's 'value'."""
    for f in packet.findall(".//field"):
        nm = f.get("name", "")
        if nm in ("tcp.payload", "udp.payload"):
            v = f.get("value", "")
            return v.replace(":", "").lower()
    return ""


def _packet_meta(packet: ET.Element) -> dict:
    meta = {}
    for f in packet.findall(".//field"):
        nm = f.get("name", "")
        if nm == "tcp.srcport":
            meta["src_port"] = int(f.get("show"))
        elif nm == "tcp.dstport":
            meta["dst_port"] = int(f.get("show"))
        elif nm == "udp.srcport":
            meta["src_port"] = int(f.get("show"))
            meta["proto"] = "udp"
        elif nm == "udp.dstport":
            meta["dst_port"] = int(f.get("show"))
            meta["proto"] = "udp"
        elif nm == "ip.src":
            meta["src_ip"] = f.get("show")
        elif nm == "ip.dst":
            meta["dst_ip"] = f.get("show")
        elif nm == "frame.number":
            meta["pkt_idx"] = int(f.get("show"))
    meta.setdefault("proto", "tcp")
    return meta


def run_tshark_pdml(
    pcap_path: str,
    display_filter: str,
    decode_as: list[str] | None = None,
    tshark_exe: str | None = None,
    max_packets: int | None = None,
) -> bytes:
    """Run tshark and return PDML XML bytes.

    tshark binary resolution: explicit arg > $TSHARK_EXE env > Windows default.
    (Set TSHARK_EXE=tshark when running under WSL/Linux.)"""
    if tshark_exe is None:
        tshark_exe = os.environ.get("TSHARK_EXE")
    if not tshark_exe:
        # Default by platform: bare `tshark` on Linux/macOS (relies on PATH),
        # Wireshark install path on Windows. Avoids depending on a shell env
        # var that some login shells (csh/tcsh) won't export.
        tshark_exe = r"C:\Program Files\Wireshark\tshark.exe" if os.name == "nt" else "tshark"
    args = [tshark_exe, "-r", pcap_path]
    for da in decode_as or []:
        args += ["-d", da]
    if display_filter:
        args += ["-Y", display_filter]
    if max_packets:
        args += ["-c", str(max_packets)]
    args += ["-T", "pdml"]
    out = subprocess.run(args, capture_output=True, check=True)
    return out.stdout


@dataclass
class PdmlPacketResult:
    schema: dict      # {"fields":[...]}
    payload_hex: str
    meta: dict        # src/dst port, direction, pkt_idx, ...


def dissect_pcap(
    pcap_path: str,
    display_filter: str,
    decode_as: list[str] | None = None,
    server_ports: list[int] | None = None,
    tshark_exe: str | None = None,
    max_packets: int | None = None,
) -> Iterator[PdmlPacketResult]:
    """Yield (schema, payload_hex, meta) per dissected packet.
    tshark binary: explicit arg > $TSHARK_EXE env > Windows default."""
    pdml = run_tshark_pdml(pcap_path, display_filter, decode_as, tshark_exe, max_packets)
    root = ET.fromstring(pdml)

    server_port: Optional[int] = None
    if server_ports:
        server_port = server_ports[0]

    for packet in root.findall("packet"):
        span = _packet_payload_span(packet)
        if span is None:
            continue
        app_start, app_end = span
        if app_end <= app_start:
            continue

        meta = _packet_meta(packet)
        # direction
        sp, dp = meta.get("src_port"), meta.get("dst_port")
        if server_port is None and (sp in (server_ports or []) or dp in (server_ports or [])):
            server_port = dp if dp in (server_ports or []) else sp
        if server_port is not None:
            meta["direction"] = "c2s" if dp == server_port else "s2c"
        else:
            meta["direction"] = "c2s"

        raw_leaves = _extract_leaves(packet, app_start, app_end)
        if not raw_leaves:
            continue

        # rebase to app payload (subtract app_start)
        fields: list[dict] = []
        for (pos, size, raw_name, value) in raw_leaves:
            sname = short_name(raw_name)
            field = {
                "off": pos - app_start,
                "len": size,
                "type": infer_type(size),
                "role": infer_role(raw_name, sname, size),
                "name": sname,
                "value": value,
            }
            fields.append(field)

        # Drop fields whose value is empty AND name is opaque container.
        # Skip frames whose tiled coverage is < 50% (likely under-dissected).
        covered = sum(f["len"] for f in fields)
        total = app_end - app_start
        if covered < total * 0.5:
            continue

        # Optionally fill remaining gaps with "payload" placeholders so the
        # schema tiles the full frame. Important so the model learns the
        # whole payload, not just dissected bits.
        fields = _fill_gaps(fields, total)

        payload_hex = _packet_payload_hex(packet, app_start, app_end)
        yield PdmlPacketResult(
            schema={"fields": fields, "frame_end": total},
            payload_hex=payload_hex,
            meta=meta,
        )


def split_iec104_apdus(result: "PdmlPacketResult") -> list["PdmlPacketResult"]:
    """Split an IEC104 frame containing multiple concatenated APDUs into one
    PdmlPacketResult per APDU. Each APDU starts at a `start` field (0x68
    marker) and runs for `apdulen + 2` bytes.

    Frames with a single APDU pass through unchanged.
    """
    fields = result.schema["fields"]
    frame_end = result.schema["frame_end"]
    payload_hex = result.payload_hex

    # Locate every 0x68 start marker by name == 'start' & len == 1.
    starts = sorted(
        f["off"] for f in fields
        if f.get("name") == "start" and f.get("len") == 1
    )
    if len(starts) <= 1:
        return [result]

    out: list[PdmlPacketResult] = []
    # Pair each start with its following apdulen value to determine the slice.
    boundaries: list[tuple[int, int]] = []
    for off in starts:
        # Find the apdulen field immediately following this start (off + 1)
        apdulen_field = next(
            (f for f in fields if f.get("name") == "apdulen" and f["off"] == off + 1),
            None,
        )
        if apdulen_field is None:
            return [result]  # malformed, give up splitting
        try:
            apdulen = int(str(apdulen_field.get("value", "0")))
        except ValueError:
            return [result]
        boundaries.append((off, off + 2 + apdulen))

    for slice_start, slice_end in boundaries:
        if slice_end > frame_end:
            slice_end = frame_end
        # Filter fields contained in this slice, rebase offsets to 0.
        sub_fields_raw = [
            {**f, "off": f["off"] - slice_start}
            for f in fields
            if f["off"] >= slice_start and f["off"] + f["len"] <= slice_end
        ]
        if not sub_fields_raw:
            continue
        sub_len = slice_end - slice_start
        sub_fields = _fill_gaps(sub_fields_raw, sub_len)
        sub_hex = payload_hex[slice_start * 2 : slice_end * 2]
        out.append(PdmlPacketResult(
            schema={"fields": sub_fields, "frame_end": sub_len},
            payload_hex=sub_hex,
            meta=dict(result.meta),
        ))
    return out if out else [result]


def _fill_gaps(fields: list[dict], total: int) -> list[dict]:
    """Fill any uncovered byte spans with role='payload' placeholders."""
    out: list[dict] = []
    cursor = 0
    for f in sorted(fields, key=lambda x: x["off"]):
        if f["off"] > cursor:
            out.append({
                "off": cursor, "len": f["off"] - cursor,
                "type": "bytes", "role": "payload",
                "name": "undissected_bytes", "value": "",
            })
        out.append(f)
        cursor = max(cursor, f["off"] + f["len"])
    if cursor < total:
        out.append({
            "off": cursor, "len": total - cursor,
            "type": "bytes", "role": "payload",
            "name": "undissected_bytes", "value": "",
        })
    return out


