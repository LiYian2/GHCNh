#!/usr/bin/env python3
"""Run resumable GHCNh weather shards through the Google Colab CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_MEMBERS = (
    "src/pipelinev1.py",
    "src/weather_pipeline.py",
    "docs/ghcnh-station-list.csv",
    "docs/ghcnh-inventory.txt",
)


def run(
    *args: str,
    check: bool = True,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args), flush=True)
    if timeout_seconds is not None:
        process = subprocess.Popen(args, text=True)
        deadline = time.time() + timeout_seconds
        while process.poll() is None:
            if time.time() >= deadline:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise subprocess.TimeoutExpired(args, timeout_seconds)
            time.sleep(0.5)
        if check and process.returncode:
            raise subprocess.CalledProcessError(process.returncode, args)
        return subprocess.CompletedProcess(args, process.returncode)
    return subprocess.run(
        args,
        check=check,
        text=True,
    )


def parse_years(value: str) -> list[int]:
    if "-" in value:
        start, end = (int(part) for part in value.split("-", 1))
        if end < start:
            raise argparse.ArgumentTypeError("year range must be ascending")
        return list(range(start, end + 1))
    return [int(part) for part in value.split(",")]


def parse_indexes(value: str) -> list[int]:
    indexes: list[int] = []
    for segment in value.split(","):
        if "-" in segment:
            start, end = (int(part) for part in segment.split("-", 1))
            if end < start:
                raise argparse.ArgumentTypeError("shard range must be ascending")
            indexes.extend(range(start, end + 1))
        else:
            indexes.append(int(segment))
    return list(dict.fromkeys(indexes))


def safe_extract(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"unsafe archive member: {member.name}")
        archive.extractall(destination, filter="data")


def valid_output(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        metadata = pq.ParquetFile(path).metadata
    except Exception:  # noqa: BLE001
        return False
    return metadata.num_rows > 0 and metadata.num_columns == 58


def build_bundle(path: Path) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for relative in BUNDLE_MEMBERS:
            archive.add(ROOT / relative, arcname=relative)


def session_ready(session: str) -> bool:
    command = ("colab", "status", "-s", session)
    print("+", " ".join(command), flush=True)
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        print("colab status timed out", flush=True)
        return False
    output = result.stdout + result.stderr
    print(output, end="", flush=True)
    return result.returncode == 0 and "Status:" in output and "not found" not in output


def create_session(session: str, gpu: str, retries: int = 5) -> None:
    command = ["colab", "new", "-s", session]
    if gpu:
        command.extend(["--gpu", gpu])
    for attempt in range(1, retries + 1):
        try:
            run(*command, timeout_seconds=90)
            return
        except subprocess.CalledProcessError:
            if attempt == retries:
                raise
            time.sleep(min(5 * attempt, 20))


def upload_runtime(session: str, bundle: Path, retries: int = 5) -> None:
    command = (
        "colab",
        "upload",
        "-s",
        session,
        str(bundle),
        "/content/ghcnh_weather_bundle.tgz",
    )
    for attempt in range(1, retries + 1):
        try:
            run(*command, timeout_seconds=90)
            return
        except subprocess.CalledProcessError:
            if attempt == retries:
                raise
            time.sleep(min(5 * attempt, 20))


def drive_mounted(session: str) -> bool:
    result = subprocess.run(
        ("colab", "ls", "-s", session, "/content/drive"),
        check=False,
        text=True,
        capture_output=True,
        timeout=30,
    )
    return result.returncode == 0


def ensure_session(
    session: str,
    gpu: str,
    bundle: Path,
    require_drive: bool = False,
) -> None:
    if not session_ready(session):
        create_session(session, gpu)
    if require_drive and not drive_mounted(session):
        raise RuntimeError(
            "Google Drive is not mounted in this Colab session; run "
            f"`colab drivemount -s {session} /content/drive` first"
        )
    upload_runtime(session, bundle)


def process_shard(
    session: str,
    bundle: Path,
    job_path: Path,
    output_root: Path,
    year: int,
    shard: int,
    num_shards: int,
    timeout: int,
    drive_root: str | None = None,
) -> None:
    job_path.write_text(
        json.dumps(
            {
                "year": year,
                "shard_index": shard,
                "num_shards": num_shards,
                "drive_root": drive_root,
            }
        )
    )
    run(
        "colab",
        "upload",
        "-s",
        session,
        str(job_path),
        "/content/weather_job.json",
        timeout_seconds=90,
    )
    run(
        "colab",
        "exec",
        "-s",
        session,
        "--timeout",
        str(timeout),
        "-f",
        str(ROOT / "scripts/colab_weather_shard.py"),
        timeout_seconds=timeout + 120,
    )

    if drive_root:
        return

    remote = f"/content/ghcnh_weather_{year}_shard_{shard:03d}.tar"
    local = job_path.parent / Path(remote).name
    run(
        "colab",
        "download",
        "-s",
        session,
        remote,
        str(local),
        timeout_seconds=180,
    )
    safe_extract(local, output_root)

    expected = output_root / "weather_by_year" / f"year={year}" / f"part-{shard:03d}.parquet"
    if not valid_output(expected):
        raise RuntimeError(f"missing or empty result after extraction: {expected}")
    local.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", type=parse_years, default=parse_years("2010-2025"))
    parser.add_argument("--num-shards", type=int, default=180)
    parser.add_argument(
        "--shard-indexes",
        type=parse_indexes,
        default=None,
        help="optional comma/range subset, e.g. 3-8,12-17,21-26",
    )
    parser.add_argument(
        "--start-at",
        default=None,
        metavar="YEAR:SHARD",
        help="skip year/shard pairs before this point, e.g. 2011:45",
    )
    parser.add_argument("--session", default="ghcnh-weather-batch")
    parser.add_argument(
        "--gpu",
        default="",
        help="GPU type; omitted/empty uses a CPU runtime (recommended)",
    )
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/final")
    parser.add_argument(
        "--drive-root",
        default=None,
        help=(
            "remote mounted Drive directory, e.g. /content/drive/MyDrive/GHCNh; "
            "results stay in Drive instead of being downloaded locally"
        ),
    )
    parser.add_argument(
        "--keep-session",
        action="store_true",
        help="leave the Colab session running after this invocation",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    shard_indexes = args.shard_indexes or list(range(args.num_shards))
    if any(index < 0 or index >= args.num_shards for index in shard_indexes):
        raise ValueError("each shard index must satisfy 0 <= index < num-shards")
    start_at: tuple[int, int] | None = None
    if args.start_at:
        try:
            start_year, start_shard = args.start_at.split(":", 1)
            start_at = (int(start_year), int(start_shard))
        except ValueError as exc:
            raise ValueError("--start-at must use YEAR:SHARD") from exc
        if start_at[1] < 0 or start_at[1] >= args.num_shards:
            raise ValueError("--start-at shard must satisfy 0 <= shard < num-shards")
    args.output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="ghcnh-colab-") as temporary:
        temp = Path(temporary)
        bundle = temp / "ghcnh_weather_bundle.tgz"
        job_path = temp / "weather_job.json"
        build_bundle(bundle)

        ensure_session(
            args.session,
            args.gpu,
            bundle,
            require_drive=bool(args.drive_root),
        )
        try:
            for year in args.years:
                for shard in shard_indexes:
                    if start_at and (year, shard) < start_at:
                        continue
                    expected = (
                        args.output_root
                        / "weather_by_year"
                        / f"year={year}"
                        / f"part-{shard:03d}.parquet"
                    )
                    if not args.drive_root and valid_output(expected):
                        print(f"SKIP existing {expected}", flush=True)
                        continue
                    for attempt in range(1, args.retries + 2):
                        try:
                            process_shard(
                                args.session,
                                bundle,
                                job_path,
                                args.output_root,
                                year,
                                shard,
                                args.num_shards,
                                args.timeout,
                                args.drive_root,
                            )
                            break
                        except (
                            subprocess.CalledProcessError,
                            subprocess.TimeoutExpired,
                            RuntimeError,
                        ):
                            if attempt > args.retries:
                                raise
                            if args.drive_root:
                                ensure_session(
                                    args.session,
                                    args.gpu,
                                    bundle,
                                    require_drive=True,
                                )
                            else:
                                run(
                                    "colab",
                                    "stop",
                                    "-s",
                                    args.session,
                                    check=False,
                                    timeout_seconds=90,
                                )
                                create_session(args.session, args.gpu)
                                upload_runtime(args.session, bundle)
        finally:
            if not args.keep_session:
                run(
                    "colab",
                    "stop",
                    "-s",
                    args.session,
                    check=False,
                    timeout_seconds=90,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
