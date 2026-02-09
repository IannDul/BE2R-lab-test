# Скрипт генерации разреженного датасета (ChatGPT-generated)
# Актуально для датасетов формата TUM

from __future__ import annotations

import argparse
import bisect
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class AssocEntry:
    ts: float
    rel_path: str


@dataclass(frozen=True)
class PoseEntry:
    ts: float
    rest: str


def read_assoc(path: Path) -> Tuple[List[str], List[AssocEntry]]:
    header: List[str] = []
    entries: List[AssocEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            header.append(line)
            continue
        parts = s.split()
        # формат: timestamp filename
        entries.append(AssocEntry(ts=float(parts[0]), rel_path=parts[1]))
    entries.sort(key=lambda e: e.ts)
    return header, entries


def write_assoc(header: List[str], entries: List[AssocEntry], out_path: Path) -> None:
    out_lines = []
    out_lines.extend(header)
    for e in entries:
        out_lines.append(f"{e.ts:.7f} {e.rel_path}")
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def read_groundtruth(path: Path) -> Tuple[List[str], List[PoseEntry]]:
    header: List[str] = []
    poses: List[PoseEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            header.append(line)
            continue
        parts = s.split()
        ts = float(parts[0])
        rest = " ".join(parts[1:])
        poses.append(PoseEntry(ts=ts, rest=rest))
    poses.sort(key=lambda p: p.ts)
    return header, poses


def write_groundtruth(header: List[str], poses: List[PoseEntry], out_path: Path) -> None:
    out_lines = []
    out_lines.extend(header)
    for p in poses:
        out_lines.append(f"{p.ts:.7f} {p.rest}")
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def select_stride(entries: List[AssocEntry], stride: int, offset: int = 0) -> List[AssocEntry]:
    return entries[offset::stride]


def nearest_index(sorted_ts: List[float], ts: float) -> int:
    i = bisect.bisect_left(sorted_ts, ts)
    if i == 0:
        return 0
    if i >= len(sorted_ts):
        return len(sorted_ts) - 1
    before = sorted_ts[i - 1]
    after = sorted_ts[i]
    return i if abs(after - ts) < abs(ts - before) else i - 1


def copy_entries(src_root: Path, dst_root: Path, entries: List[AssocEntry]) -> None:
    for e in entries:
        src = src_root / e.rel_path
        dst = dst_root / e.rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def match_depth_by_timestamp(
    rgb_sel: List[AssocEntry],
    depth_all: List[AssocEntry],
    max_dt: float,
) -> List[AssocEntry]:
    depth_by_ts: Dict[float, AssocEntry] = {d.ts: d for d in depth_all}
    depth_ts = [d.ts for d in depth_all]

    matched: List[AssocEntry] = []
    for r in rgb_sel:
        if r.ts in depth_by_ts:
            d = depth_by_ts[r.ts]
            matched.append(AssocEntry(ts=r.ts, rel_path=d.rel_path))
            continue

        # если вдруг нет точного совпадения — берём ближайший по времени
        j = nearest_index(depth_ts, r.ts)
        d = depth_all[j]
        if abs(d.ts - r.ts) <= max_dt:
            # Важно: timestamp в depth.txt ставим как у RGB (чтобы синхронизация была 1-1)
            matched.append(AssocEntry(ts=r.ts, rel_path=d.rel_path))
    return matched


def filter_groundtruth_by_timestamps(
    target_ts: List[float],
    gt_all: List[PoseEntry],
    max_dt: float,
) -> List[PoseEntry]:
    if not gt_all:
        return []
    gt_ts = [p.ts for p in gt_all]
    out: List[PoseEntry] = []
    for t in target_ts:
        j = nearest_index(gt_ts, t)
        p = gt_all[j]
        if abs(p.ts - t) <= max_dt:
            # timestamp пишем как у RGB, чтобы ATE считался ровно на выбранных кадрах
            out.append(PoseEntry(ts=t, rest=p.rest))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="путь к датасету")
    ap.add_argument("--dst", type=Path, required=True, help="куда писать вариант (например umbrella_stride2)")
    ap.add_argument("--stride", type=int, default=2, help="stride (2 = каждый второй кадр)")
    ap.add_argument("--offset", type=int, default=0, help="смещение (0 или 1) для выбора кадров")
    ap.add_argument("--max-dt", type=float, default=0.02, help="макс. рассинхронизация по времени (сек)")
    ap.add_argument("--copy-nvs", action="store_true", help="скопировать папку nvs целиком")
    args = ap.parse_args()

    src = args.src
    dst = args.dst
    dst.mkdir(parents=True, exist_ok=True)

    # читаем ассоциации
    rgb_header, rgb_all = read_assoc(src / "rgb.txt")
    depth_header, depth_all = read_assoc(src / "depth.txt")

    # выбираем RGB по stride
    rgb_sel = select_stride(rgb_all, args.stride, args.offset)

    # сопоставляем depth по timestamp
    depth_sel = match_depth_by_timestamp(rgb_sel, depth_all, max_dt=args.max_dt)

    # копируем картинки
    copy_entries(src, dst, rgb_sel)
    copy_entries(src, dst, depth_sel)

    # пишем новые assoc-файлы
    # Важно: пути остаются вида rgb/frame_XXXXX.png и depth/depth_XXXXX.png
    write_assoc(rgb_header, [AssocEntry(e.ts, e.rel_path) for e in rgb_sel], dst / "rgb.txt")
    write_assoc(depth_header, [AssocEntry(e.ts, e.rel_path) for e in depth_sel], dst / "depth.txt")

    # groundtruth (если есть)
    gt_path = src / "groundtruth.txt"
    if gt_path.exists():
        gt_header, gt_all = read_groundtruth(gt_path)
        ts_sel = [e.ts for e in rgb_sel]
        gt_sel = filter_groundtruth_by_timestamps(ts_sel, gt_all, max_dt=args.max_dt)
        write_groundtruth(gt_header, gt_sel, dst / "groundtruth.txt")

    # intrinsics.json (если есть)
    intr = src / "intrinsics.json"
    if intr.exists():
        shutil.copy2(intr, dst / "intrinsics.json")

    # nvs (опционально)
    if args.copy_nvs and (src / "nvs").exists():
        out_nvs = dst / "nvs"
        if out_nvs.exists():
            shutil.rmtree(out_nvs)
        shutil.copytree(src / "nvs", out_nvs)

    print(f"Done. RGB: {len(rgb_sel)} frames, Depth matched: {len(depth_sel)} frames.")
    if (dst / "groundtruth.txt").exists():
        print(f"GT: written {sum(1 for _ in (dst / 'groundtruth.txt').read_text(encoding='utf-8').splitlines() if _.strip() and not _.startswith('#'))} poses.")


if __name__ == "__main__":
    main()
