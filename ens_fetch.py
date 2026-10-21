#!/usr/bin/env python
"""Read the ECMWF IFS ensemble (ENS, 51 members) at a few points from ECMWF's open data on AWS.

For each day's 00 UTC run and each 3-hourly step 18..48 h it fetches only the GRIB messages of the
wanted parameters (byte ranges from the .index file: 51 members x ~1.1 MB each, nothing else), decodes
the values at the cells nearest to the points in points.yaml and appends rows to a monthly parquet:

    run (YYYY-MM-DD), step (h), param, point, member (0 = control, 1..50), value, published_at (UTC
    upload time of the step file on the bucket, i.e. when the step was first available)

Usage: python ens_fetch.py --start 2024-06-15 --end 2024-06-16 --out data [--params ssrd,tcc]
A day already complete in the output is skipped. Exit code 0 even if some days are missing (they are
listed); the caller decides what to do.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

BUCKET = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
STEPS = list(range(18, 49, 3))          # 18, 21, ..., 48 h: brackets the local day D of a 00z run on D-1
PARAMS = ("ssrd", "2t")                 # accumulated J/m2 since the start of the run; 2 m temperature K
# (the open-data ENS surface set has no cloud cover: 2t, tcwv, tp, 10u/10v, sp, cape would be the other options)
MISSING = 9.999e20


class Slow(Exception):
    pass


def get(session, url, headers=None, tries=10, timeout=(30, 300)):
    """GET with backoff on 429/5xx and connection errors; None on 404. The body is read in full
    before returning (a range of 56 MB), so a stalled transfer counts as a failed attempt."""
    for attempt in range(tries):
        try:
            r = session.get(url, headers=headers or {}, timeout=timeout)
            if r.status_code in (200, 206):
                r.content  # noqa: B018 -- force the full read inside the try
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        wait = min(180, 3 * 2 ** attempt + random.random() * 3)
        time.sleep(wait)
    raise Slow(f"gave up on {url}")


def head(session, url, tries=6):
    for attempt in range(tries):
        try:
            r = session.head(url, timeout=60)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(min(60, 2 ** attempt + random.random()))
    raise Slow(f"gave up on HEAD {url}")


def step_url(day: dt.date, step: int, ext: str) -> str:
    d = day.strftime("%Y%m%d")
    return f"{BUCKET}/{d}/00z/ifs/0p25/enfo/{d}000000-{step}h-enfo-ef.{ext}"


def wanted_entries(index_text: str, params) -> list[dict]:
    out = []
    for line in index_text.splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        if e.get("param") in params and e.get("levtype") == "sfc" and e.get("type") in ("cf", "pf"):
            e["member"] = 0 if e["type"] == "cf" else int(e["number"])
            out.append(e)
    return out


def contiguous_runs(entries: list[dict]) -> list[list[dict]]:
    """Group entries into runs of adjacent byte ranges so each run is one range request."""
    entries = sorted(entries, key=lambda e: int(e["_offset"]))
    runs: list[list[dict]] = []
    for e in entries:
        if runs and int(runs[-1][-1]["_offset"]) + int(runs[-1][-1]["_length"]) == int(e["_offset"]):
            runs[-1].append(e)
        else:
            runs.append([e])
    return runs


_GRID = {}


def decode_points(message: bytes, coords) -> np.ndarray:
    import eccodes

    handle = eccodes.codes_new_from_message(message)
    try:
        g = eccodes.codes_get
        key = (g(handle, "Ni"), g(handle, "Nj"))
        if key not in _GRID:
            if g(handle, "gridType") != "regular_ll" or g(handle, "iScansNegatively") or g(handle, "jPointsAreConsecutive"):
                raise ValueError("unexpected grid")
            dlat = float(g(handle, "jDirectionIncrementInDegrees"))
            if not g(handle, "jScansPositively"):
                dlat = -dlat
            _GRID[key] = (float(g(handle, "latitudeOfFirstGridPointInDegrees")),
                          float(g(handle, "longitudeOfFirstGridPointInDegrees")), dlat,
                          float(g(handle, "iDirectionIncrementInDegrees")), int(g(handle, "Nj")), int(g(handle, "Ni")))
        lat0, lon0, dlat, dlon, nj, ni = _GRID[key]
        eccodes.codes_set(handle, "missingValue", MISSING)
        values = eccodes.codes_get_values(handle).reshape(nj, ni)
    finally:
        eccodes.codes_release(handle)
    out = []
    for lat, lon in coords:
        row = int(round((lat - lat0) / dlat))
        col = int(round(((lon - lon0) % 360.0) / dlon)) % ni
        out.append(values[row, col])
    out = np.array(out, dtype=float)
    out[out >= MISSING] = np.nan
    return out


def fetch_step(session, day, step, params, coords, point_ids) -> list[dict] | None:
    idx = get(session, step_url(day, step, "index"))
    if idx is None:
        return None
    entries = wanted_entries(idx.text, params)
    if not entries:
        return None
    h = head(session, step_url(day, step, "grib2"))
    published = h.headers.get("Last-Modified", "") if h is not None else ""
    published_at = pd.Timestamp(published).tz_convert("UTC").isoformat() if published else ""
    rows = []
    for run in contiguous_runs(entries):
        start = int(run[0]["_offset"]); end = int(run[-1]["_offset"]) + int(run[-1]["_length"]) - 1
        r = get(session, step_url(day, step, "grib2"), headers={"Range": f"bytes={start}-{end}"})
        if r is None:
            return None
        blob = r.content
        for e in run:
            a = int(e["_offset"]) - start; b = a + int(e["_length"])
            vals = decode_points(blob[a:b], coords)
            for pid, v in zip(point_ids, vals):
                rows.append({"run": day.isoformat(), "step": step, "param": e["param"], "point": pid,
                             "member": e["member"], "value": float(v), "published_at": published_at})
    return rows


def month_path(out: Path, day: dt.date) -> Path:
    return out / f"ens_{day:%Y-%m}.parquet"


def load_month(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["run", "step", "param", "point", "member", "value", "published_at"])


def complete_days(frame: pd.DataFrame, params, n_points) -> set[str]:
    if frame.empty:
        return set()
    need = len(STEPS) * len(params) * n_points * 51
    counts = frame.groupby("run").size()
    return set(counts[counts >= need].index)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True, help="inclusive")
    ap.add_argument("--out", default="data")
    ap.add_argument("--params", default=",".join(PARAMS))
    ap.add_argument("--points", default=str(Path(__file__).with_name("points.yaml")))
    ap.add_argument("--pause", type=float, default=0.3, help="seconds between requests")
    args = ap.parse_args(argv)
    params = tuple(args.params.split(","))
    pts = yaml.safe_load(open(args.points))["points"]
    coords = [(float(p["lat"]), float(p["lon"])) for p in pts]
    point_ids = [p["id"] for p in pts]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    start = dt.date.fromisoformat(args.start); end = dt.date.fromisoformat(args.end)
    session = requests.Session()
    session.headers["User-Agent"] = "hrsolar-ens/1.0 (research; ECMWF open data)"
    missing = []
    day = start
    t0 = time.time()
    cache: dict[Path, pd.DataFrame] = {}
    while day <= end:
        path = month_path(out, day)
        if path not in cache:
            cache[path] = load_month(path)
        done = complete_days(cache[path], params, len(point_ids))
        if day.isoformat() in done:
            day += dt.timedelta(days=1); continue
        rows = []
        ok = True
        for step in STEPS:
            try:
                got = fetch_step(session, day, step, params, coords, point_ids)
            except Slow as exc:            # a day that will not come today: skip it, a later run fills it
                print(f"{day} step {step}: {exc}", flush=True)
                got = None
            time.sleep(args.pause)
            if got is None:
                ok = False; break
            rows.extend(got)
        if ok and rows:
            new = pd.DataFrame(rows)
            frame = cache[path]
            frame = frame[frame["run"] != day.isoformat()]
            frame = new if frame.empty else pd.concat([frame, new], ignore_index=True)
            cache[path] = frame
            frame.to_parquet(path, index=False)
            print(f"{day} ok {len(new)} rows {time.time() - t0:.0f}s", flush=True)
        else:
            missing.append(day.isoformat()); print(f"{day} missing", flush=True)
        day += dt.timedelta(days=1)
    print("missing days:", missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
