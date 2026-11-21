# Step 1 · Get the data

**Code:** [`tools/bracs_download.py`](../tools/bracs_download.py) · **Ran:** 2026-11-13 20:21 to 2026-11-14 00:03 · **Milestone:** M0

## In one sentence

We copied the BRACS regions (4,539 images, 55.6 GB) and its summary spreadsheet from the BRACS server to your Mac, then checked that every file arrived complete.

## A real example

The server has a file `BRACS_RoI/latest_version/train/1_PB/BRACS_1926_PB_5.png`.
- At 22:02 the server refused one attempt to download it ("425 Security: Bad IP connecting").
- The script waited, connected again, finished the file, and checked its size against the server's list. It matched.
- That was the only problem in the whole 3 h 42 min download.

## Why this way

| Choice | Why | Other option |
|---|---|---|
| BRACS regions, not whole slides | The brief's version 1 uses regions; whole slides are huge and need extra steps | whole slides: a later version |
| Only `latest_version` | That is the current release; `previous_versions` is older | download both (about 48 GB more) |
| A script, not clicking in an FTP app | 4,540 files: the script resumes, retries and checks every file | FileZilla (manual, no size check) |
| 3 downloads at once | It barely helped (4.2 MB/s vs 3.9 with one); the server caps total speed | 1 connection |

## The code, part by part

FTP is a protocol for copying files from a server; Python has it built in (`ftplib`). The login is the public read-only one from the BRACS download page.

1. **`connect()`:** log in, and if that fails, wait and try again: 5 s, then 10, 20 … up to 300 s, 10 tries. This is "retry with backoff".
2. **`listing()`:** ask the server for a folder listing. Each line looks like `-rw-r--r-- 1 owner group 12345678 Jan 1 2022 BRACS_1003658_PB_1.png`; the script keeps the size (5th column) and the name (last).
3. **`build_manifest()`:** walk `test`, `val`, `train` × the 7 type folders and make a **manifest**, a list of all 4,540 files with their sizes. It is saved as `data/bracs/_manifest.json`.
4. **`worker()`:** three workers take files from a shared queue. Each file is written to `name.png.part` first.
   - If a `.part` already exists, the download continues from where it stopped (`rest=offset`, the FTP "restart" command).
   - When the size matches the manifest, the file is renamed to `name.png`. So a finished-looking file is always a complete file.
5. **`reporter()`:** every 30 s, writes progress and an estimated finish time to `_download.log`.
6. **`main()`:** skips files that are already complete. At the end it checks every file's size against the manifest, logs `DONE: 4540/4540 files verified by size`, and exits with code 1 if anything is missing.

## Result

`DONE: 4540/4540 files verified by size; per split {'test': 570, 'val': 312, 'train': 3657}; missing or wrong size: 0`

The data is in `data/bracs/`: 52 GB on disk.

**For later (M3):** the script contains the public login. Before the code goes into a public repository, move it into a setting outside the code.

## Rebuild it yourself

Goal: list one folder and download one file, then check its size.

```python
import ftplib
ftp = ftplib.FTP("histoimage.na.icar.cnr.it", timeout=60)
ftp.login(USER, PASSWORD)   # the read-only login from the BRACS download page (also in the script)
rows = []
ftp.retrlines("LIST BRACS_RoI/latest_version/test/0_N", rows.append)
print(len(rows))            # predict first: how many normal test images are there?
# Your turn:
# 1. find the smallest file in rows (size = 5th column)
# 2. download it with ftp.retrbinary("RETR " + path, file.write)
# 3. compare the size on disk with the size in the listing
```

Expected: 81 rows. The sizes match.

## Check yourself

1. Why write to `.part` first, and rename only after the size check?
2. Your internet drops at 60%. What happens when you run the script again?
3. Why is "the file exists" not enough proof that a download worked?

<details><summary>Answers</summary>

1. So that an unfinished file can never look finished. Anything without `.part` has been checked.
2. Complete files are skipped. Each `.part` file continues from its current size, and nothing is downloaded twice.
3. A broken connection can leave a file that exists but is cut short. Comparing its size with the server's list catches that.
</details>

## Words to know

- **FTP:** a way to copy files from a server.
- **Manifest:** the list of every file with its expected size.
- **Resumable:** stopping and restarting loses no work.
- **Retry with backoff:** try again, waiting longer each time.
- **Exit code:** 0 means success and anything else means failure; other programs can check it.
