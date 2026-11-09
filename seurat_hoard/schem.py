"""Minecraft schematics (WorldEdit / Sponge ``.schem``, versions 1-3) as voxel models.

The file is gzip-compressed NBT: the size (Width = x, Height = y, Length = z), a palette of block states and the blocks
as varints (index = x + z * Width + y * Width * Length). Each block becomes the average colour of its texture (table
below, with rules for the dye-coloured families); air and invisible blocks are empty. Minecraft's axes match Seurat's:
x east (right), y up, z south (towards the viewer).
"""

from __future__ import annotations

import gzip
import io
import struct
import zlib
from typing import Any

import numpy as np

from .errors import SeuratError
from .voxels import MAX_SIDE, VoxelModel

EMPTY = {"air", "cave_air", "void_air", "structure_void", "barrier", "light", "moving_piston"}

DYES = ("white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray", "light_gray", "cyan", "purple", "blue",
        "brown", "green", "red", "black")
_FAMILIES = {
    "wool": "e9ecec f07613 bd44b3 3aafd9 f8c527 70b919 ed8dac 3e4447 8e8e86 158991 792aac 35399d 724728 546d1b a12722 141519",
    "carpet": "e9ecec f07613 bd44b3 3aafd9 f8c527 70b919 ed8dac 3e4447 8e8e86 158991 792aac 35399d 724728 546d1b a12722 141519",
    "concrete": "cfd5d6 e06100 a9309f 2389c6 f1af15 5ea818 d5658e 36393d 7d7d73 157788 64209c 2c2e8f 603b1f 495b24 8e2020 080a0f",
    "concrete_powder": "e1e3e3 e3831f c053b8 4aa4d5 e8c736 7dbd29 e499b5 4c5154 9a9a94 249499 8337b1 464ba6 7d5435 61772c a83632 191a1f",
    "terracotta": "d1b2a1 a15325 95586c 716c89 ba8523 677534 a14e4e 392a23 876a61 575b5b 764656 4a3b5b 4d3323 4c532a 8e3c2e 251610",
    "glazed_terracotta": "bcd4ca a39552 d064bf 5ea4d0 eac058 a3c537 eb9aaa 535a5d 90a6a7 34777d 6d3098 2f408b 77654f 75922f b53b35 431e20",
    "stained_glass": "ffffff d87f33 b24cd8 6699d8 e5e533 7fcc19 f27fa5 4c4c4c 999999 4c7f99 7f3fb2 334cb2 664c33 667f33 993333 191919",
    "shulker_box": "d7dcdd e96a0c ad38a4 3197cf f8bc1d 64ad17 e780a0 373a3e 7c7c73 147a88 6a2a9b 2c2e8c 6a4224 4f6520 8c1f1e 19191d",
    "bed": "e9ecec f07613 bd44b3 3aafd9 f8c527 70b919 ed8dac 3e4447 8e8e86 158991 792aac 35399d 724728 546d1b a12722 141519",
}
FAMILY = {fam: dict(zip(DYES, cols.split())) for fam, cols in _FAMILIES.items()}
BLOCKS = {
    "terracotta": "985e43", "stone": "7e7e7e", "cobblestone": "7f7f7f", "andesite": "888888", "polished_andesite": "848686",
    "diorite": "bcbcbc", "polished_diorite": "c0c1c2", "granite": "956755", "polished_granite": "9a6a59", "tuff": "6c6d66",
    "deepslate": "505053", "cobbled_deepslate": "4d4d50", "gravel": "838180", "sand": "dbcfa3", "red_sand": "be6621",
    "sandstone": "d8cb9b", "smooth_sandstone": "e0d6aa", "red_sandstone": "ba631d", "smooth_red_sandstone": "b5621f",
    "clay": "a0a6b3", "dirt": "866043", "coarse_dirt": "77553b", "grass_block": "5d913a", "podzol": "5c3f18", "mud": "3c393d",
    "oak_planks": "a2834f", "spruce_planks": "735531", "birch_planks": "c0af79", "jungle_planks": "a07351", "acacia_planks": "a85a32",
    "dark_oak_planks": "432b14", "mangrove_planks": "763631", "cherry_planks": "e2b2ac", "bamboo_planks": "c2ad50",
    "crimson_planks": "653147", "warped_planks": "2b6963", "oak_log": "6d5532", "spruce_log": "3a2610", "birch_log": "d8d7d2",
    "jungle_log": "55441a", "acacia_log": "676157", "dark_oak_log": "3c2e1a", "oak_wood": "6d5532", "spruce_wood": "3a2610",
    "birch_wood": "d8d7d2", "jungle_wood": "55441a", "acacia_wood": "676157", "dark_oak_wood": "3c2e1a",
    "stripped_oak_wood": "b19057", "stripped_oak_log": "b19057", "stripped_spruce_wood": "73593a", "stripped_spruce_log": "73593a",
    "stripped_birch_wood": "c5b07b", "stripped_birch_log": "c5b07b", "stripped_jungle_wood": "ab8452", "stripped_jungle_log": "ab8452",
    "stripped_acacia_wood": "ae5c3b", "stripped_acacia_log": "ae5c3b", "stripped_dark_oak_wood": "604c31", "stripped_dark_oak_log": "604c31",
    "oak_leaves": "4e7c2c", "spruce_leaves": "3d5e3d", "birch_leaves": "6a8a4c", "jungle_leaves": "3f8a1c", "acacia_leaves": "4e7c2c",
    "dark_oak_leaves": "3d6b1f", "azalea_leaves": "5a7428", "melon": "6f911f", "pumpkin": "c67519", "hay_block": "a68b0c",
    "red_mushroom_block": "c82e2d", "brown_mushroom_block": "957051", "mushroom_stem": "cbc4b9",
    "copper_block": "c06c50", "exposed_copper": "a17e68", "weathered_copper": "6c996e", "oxidized_copper": "52a385",
    "cut_copper": "bf6b51", "gold_block": "f6d03d", "iron_block": "dcdcdc", "diamond_block": "62edea", "emerald_block": "2acb57",
    "lapis_block": "1e438c", "redstone_block": "af1805", "coal_block": "101010", "netherite_block": "423d3f",
    "quartz_block": "ece6df", "smooth_quartz": "ece6df", "quartz_pillar": "ebe6e0", "chiseled_quartz_block": "e7e2da",
    "bricks": "976253", "stone_bricks": "7a7a7a", "mossy_stone_bricks": "737960", "nether_bricks": "2c1519", "red_nether_bricks": "461013",
    "netherrack": "622626", "glowstone": "ac8354", "obsidian": "0f0b19", "crying_obsidian": "200a3c", "blackstone": "2a2328",
    "basalt": "505155", "polished_basalt": "636366", "end_stone": "dbde9e", "purpur_block": "a97ea9", "prismarine": "639c97",
    "dark_prismarine": "335b4b", "prismarine_bricks": "63ab9e", "sea_lantern": "acc7be", "snow_block": "f9fefe", "snow": "f9fefe",
    "ice": "91b7fd", "packed_ice": "8db4fa", "blue_ice": "74a8fd", "glass": "dbe9eb", "slime_block": "6fc05b", "honey_block": "fbbc39",
    "bone_block": "e5e1cf", "dried_kelp_block": "323b27", "moss_block": "596d2d", "sponge": "c4c04b", "wet_sponge": "aab43f",
    "calcite": "dfe0dc", "dripstone_block": "866b5c", "amethyst_block": "8562bf", "white_wool": "e9ecec", "bookshelf": "755f3c",
    "crafting_table": "77593a", "note_block": "583b29", "jukebox": "5d3f2d", "tnt": "db4430", "target": "e5b0a8",
    "shroomlight": "f19246", "magma_block": "8e3f1f", "soul_sand": "51402f", "soul_soil": "4b3a2e", "nether_wart_block": "720b0b",
    "warped_wart_block": "167779", "crimson_nylium": "831e1e", "warped_nylium": "2b7265", "mycelium": "6f6265",
}


def block_color(name: str) -> str | None:
    """``#rrggbb`` for a block state (``minecraft:red_wool[...]``), None for air and invisible blocks."""
    base = name.split("[", 1)[0].split(":", 1)[-1]
    if base in EMPTY:
        return None
    if base in BLOCKS:
        return "#" + BLOCKS[base]
    for fam in sorted(FAMILY, key=len, reverse=True):
        if base.endswith("_" + fam):
            dye = base[: -len(fam) - 1]
            if dye in FAMILY[fam]:
                return "#" + FAMILY[fam][dye]
    # slabs, stairs, walls, fences... take the colour of their material
    for suffix in ("_slab", "_stairs", "_wall", "_fence", "_fence_gate", "_trapdoor", "_door", "_button", "_pressure_plate",
                   "_pane", "_sign", "_wall_sign"):
        if base.endswith(suffix):
            stem = base[: -len(suffix)]
            for cand in (stem, stem + "s", stem + "_planks", stem + "_block", stem.replace("brick", "bricks")):
                if cand in BLOCKS:
                    return "#" + BLOCKS[cand]
            if stem.endswith("_stained_glass"):
                return "#" + FAMILY["stained_glass"].get(stem[: -len("_stained_glass")], "dbe9eb")
    for dye in sorted(DYES, key=len, reverse=True):          # any other dyed block
        if base.startswith(dye + "_"):
            return "#" + FAMILY["concrete"][dye]
    h = zlib.crc32(base.encode()) & 0x3F                       # unknown: a grey, stable per block
    return "#%02x%02x%02x" % (110 + h, 110 + h, 110 + h)


# ---------------------------------------------------------------------------------------------------------- NBT
def _nbt(data: bytes) -> dict[str, Any]:
    b = io.BytesIO(data)

    def rd(fmt: str):
        n = struct.calcsize(fmt)
        chunk = b.read(n)
        if len(chunk) < n:
            raise ValueError("truncated NBT")
        return struct.unpack(">" + fmt, chunk)[0]

    def name() -> str:
        return b.read(rd("H")).decode("utf-8", "replace")

    def payload(t: int):
        if t == 1: return rd("b")
        if t == 2: return rd("h")
        if t == 3: return rd("i")
        if t == 4: return rd("q")
        if t == 5: return rd("f")
        if t == 6: return rd("d")
        if t == 7: return b.read(rd("i"))
        if t == 8: return name()
        if t == 9:
            et, n = rd("b"), rd("i")
            return [payload(et) for _ in range(n)]
        if t == 10:
            out = {}
            while True:
                tt = rd("b")
                if tt == 0:
                    return out
                k = name()
                out[k] = payload(tt)
        if t == 11:
            n = rd("i"); return [rd("i") for _ in range(n)]
        if t == 12:
            n = rd("i"); return [rd("q") for _ in range(n)]
        raise ValueError(f"unknown NBT tag {t}")

    t = rd("b")
    if t != 10:
        raise ValueError("not an NBT compound")
    name()
    return payload(10)


def _varints(raw: bytes, count: int) -> np.ndarray:
    out = np.zeros(count, dtype=np.int32)
    i = n = 0
    while n < count and i < len(raw):
        value, shift = 0, 0
        while True:
            byte = raw[i]
            i += 1
            value |= (byte & 0x7F) << shift
            if not byte & 0x80:
                break
            shift += 7
        out[n] = value
        n += 1
    return out


def from_schem(data: bytes, model_id: str, name: str) -> VoxelModel:
    try:
        raw = gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
        root = _nbt(raw)
    except Exception:  # noqa: BLE001
        if data[:15].lower().startswith(b"<!doctype") or data[:5].lower() == b"<html":
            raise SeuratError("invalid", "This file is a web page, not a schematic (the browser saved the viewer page).",
                              "Download the .schem itself (the site's direct download link).") from None
        raise SeuratError("invalid", "Not a Minecraft .schem file (gzip-compressed NBT).") from None
    s = root.get("Schematic", root)
    w, h, l = int(s.get("Width", 0)), int(s.get("Height", 0)), int(s.get("Length", 0))
    if "Blocks" in s and isinstance(s["Blocks"], dict):           # version 3
        palette, blockdata = s["Blocks"].get("Palette") or {}, s["Blocks"].get("Data") or b""
    else:
        palette, blockdata = s.get("Palette") or {}, s.get("BlockData") or b""
    if not (w and h and l and palette and blockdata):
        raise SeuratError("invalid", "Unsupported schematic: only Sponge/WorldEdit .schem (versions 1-3) can be read.",
                          "Old MCEdit .schematic files: open and save them as .schem in a viewer first.")
    if max(w, h, l) > MAX_SIDE:
        raise SeuratError("invalid", f"The schematic is {w}x{h}x{l}; Seurat models are at most {MAX_SIDE} per side.")
    ids = _varints(blockdata, w * h * l).reshape(h, l, w)          # [y, z, x]
    colours: list[str] = []
    lut = np.zeros(max(palette.values()) + 1, dtype=np.uint8)
    for state, idx in palette.items():
        col = block_color(state)
        if col is None:
            continue
        if col not in colours:
            if len(colours) >= 255:
                continue
            colours.append(col)
        lut[idx] = colours.index(col) + 1
    ids = np.where((ids >= 0) & (ids < len(lut)), ids, 0)          # an index outside the palette is empty
    grid = lut[ids].transpose(2, 0, 1)                              # -> [x, y, z]
    model = VoxelModel.empty(model_id, name, [w, h, l], colours or ["#888888"])
    model.grid = np.ascontiguousarray(grid)
    model.source = {"kind": "imported", "format": "schem", "blocks": len(palette)}
    return model
