#!/usr/bin/env python3
"""Mirror BRACS_RoI/latest_version (+ BRACS.xlsx) from the BRACS FTP server.

Usage: python3 -I bracs_download.py <dest_dir> [workers]

Resumable: complete files are skipped, partial files (.part) continue where
they stopped, and every file is checked against the server's size.
Progress goes to <dest>/_download.log and <dest>/_download_status.json.
"""
import ftplib
import json
import pathlib
import queue
import sys
import threading
import time

# Public read-only login published on the BRACS Download page.
HOST, USER, PASS = "histoimage.na.icar.cnr.it", "utente_reader", "ICAR-CNR"
ROOT = "BRACS_RoI/latest_version"
DEST = pathlib.Path(sys.argv[1]).expanduser().resolve()
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
DEST.mkdir(parents=True, exist_ok=True)
LOG = DEST / "_download.log"
STATUS = DEST / "_download_status.json"

lock = threading.Lock()
done_bytes = 0
done_files = 0
failed = []


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    with lock:
        with open(LOG, "a") as fh:
            fh.write(line + "\n")
    print(line, flush=True)


def connect():
    delay = 5
    for _ in range(10):
        try:
            ftp = ftplib.FTP(HOST, timeout=120)
            ftp.login(USER, PASS)
            ftp.voidcmd("TYPE I")
            return ftp
        except Exception as e:  # noqa: BLE001 - retry any network error
            log(f"connect failed ({e!r}); retry in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 300)
    raise RuntimeError("could not connect to the BRACS server")


def listing(ftp, path):
    rows = []
    ftp.retrlines(f"LIST {path}" if path else "LIST", rows.append)
    out = []
    for r in rows:
        p = r.split(None, 8)
        if len(p) == 9 and p[0][0] in "-d":
            out.append((p[0][0] == "d", int(p[4]), p[8]))
    return out


def build_manifest():
    ftp = connect()
    files = [("BRACS.xlsx", next(s for d, s, n in listing(ftp, "") if n == "BRACS.xlsx"))]
    for split in ("test", "val", "train"):  # small splits first
        for is_dir, _, cls in listing(ftp, f"{ROOT}/{split}"):
            if is_dir:
                for d, size, name in listing(ftp, f"{ROOT}/{split}/{cls}"):
                    if not d:
                        files.append((f"{ROOT}/{split}/{cls}/{name}", size))
    ftp.quit()
    return files


def worker(q):
    global done_bytes, done_files
    ftp = None
    while True:
        try:
            rpath, size = q.get_nowait()
        except queue.Empty:
            break
        local = DEST / rpath
        local.parent.mkdir(parents=True, exist_ok=True)
        part = local.with_name(local.name + ".part")
        counted = 0
        ok = False
        for attempt in range(6):
            try:
                if ftp is None:
                    ftp = connect()
                offset = part.stat().st_size if part.exists() else 0
                if offset > size:
                    part.unlink()
                    offset = 0
                with lock:
                    done_bytes += offset - counted
                counted = offset
                if offset < size:
                    with open(part, "ab") as fh:
                        def write(chunk):
                            global done_bytes
                            nonlocal counted
                            fh.write(chunk)
                            counted += len(chunk)
                            with lock:
                                done_bytes += len(chunk)
                        ftp.retrbinary(f"RETR {rpath}", write, blocksize=1 << 16, rest=offset or None)
                got = part.stat().st_size
                if got != size:
                    raise IOError(f"got {got} bytes, expected {size}")
                part.rename(local)
                ok = True
                break
            except Exception as e:  # noqa: BLE001 - reconnect and resume
                log(f"retry {attempt + 1}/6 {rpath}: {e!r}")
                try:
                    ftp.close()
                except Exception:  # noqa: BLE001
                    pass
                ftp = None
                time.sleep(min(10 * 2 ** attempt, 300))
        with lock:
            if ok:
                done_files += 1
            else:
                failed.append(rpath)
    if ftp is not None:
        try:
            ftp.quit()
        except Exception:  # noqa: BLE001
            pass


def reporter(total_files, total_bytes, base, t0, stop):
    while not stop.wait(30):
        b = done_bytes
        speed = (b - base) / max(time.time() - t0, 1)
        eta = (total_bytes - b) / speed / 3600 if speed > 0 else -1
        log(f"{done_files}/{total_files} files, {b / 1e9:.1f}/{total_bytes / 1e9:.1f} GB, "
            f"{speed / 1e6:.1f} MB/s average, ETA {eta:.1f} h, failed {len(failed)}")
        STATUS.write_text(json.dumps({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "files_done": done_files,
            "files_total": total_files, "gb_done": round(b / 1e9, 2),
            "gb_total": round(total_bytes / 1e9, 2), "mb_per_s": round(speed / 1e6, 2),
            "eta_hours": round(eta, 2), "failed": list(failed)}))


def main():
    global done_bytes, done_files
    files = build_manifest()
    (DEST / "_manifest.json").write_text(json.dumps([{"path": p, "bytes": s} for p, s in files], indent=0))
    total_files, total_bytes = len(files), sum(s for _, s in files)
    log(f"manifest: {total_files} files, {total_bytes / 1e9:.1f} GB -> {DEST} ({WORKERS} connections)")
    q = queue.Queue()
    for rpath, size in files:
        local = DEST / rpath
        if local.exists() and local.stat().st_size == size:
            done_files += 1
            done_bytes += size
        else:
            q.put((rpath, size))
    log(f"already complete: {done_files} files; to fetch: {q.qsize()}")
    stop = threading.Event()
    threading.Thread(target=reporter, args=(total_files, total_bytes, done_bytes, time.time(), stop),
                     daemon=True).start()
    threads = [threading.Thread(target=worker, args=(q,)) for _ in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    bad = [p for p, s in files if not ((DEST / p).exists() and (DEST / p).stat().st_size == s)]
    per_split = {}
    for p, _ in files:
        parts = p.split("/")
        if len(parts) == 5:
            per_split[parts[2]] = per_split.get(parts[2], 0) + 1
    log(f"DONE: {total_files - len(bad)}/{total_files} files verified by size; per split {per_split}; "
        f"missing or wrong size: {len(bad)}")
    for p in bad[:20]:
        log(f"  not ok: {p}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
