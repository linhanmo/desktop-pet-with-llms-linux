import argparse
import base64
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from http.client import HTTPSConnection
from pathlib import Path


def _git_github_basic_auth() -> str:
    p = subprocess.run(
        ["git", "credential", "fill"],
        input=b"protocol=https\nhost=github.com\n\n",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError("git credential fill failed")
    creds = {}
    for line in p.stdout.decode().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            creds[k.strip()] = v.strip()
    user = creds.get("username")
    password = creds.get("password")
    if not user or not password:
        raise RuntimeError("no github credentials from git credential helper")
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


def _have_curl() -> bool:
    return shutil.which("curl") is not None


def _curl_json(method: str, url: str, auth: str, body: dict | None = None, timeout_s: int = 60) -> dict:
    cmd = [
        "curl",
        "-sS",
        "--fail-with-body",
        "--connect-timeout",
        "30",
        "--max-time",
        str(timeout_s),
        "-X",
        method,
        "-H",
        "Accept: application/vnd.github+json",
        "-H",
        "User-Agent: desktop-pet-with-llms-linux-release-uploader",
        "-H",
        f"Authorization: {auth}",
        url,
    ]
    if body is not None:
        cmd.extend(["-H", "Content-Type: application/json", "--data-binary", json.dumps(body)])
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or p.stdout.strip() or f"curl failed: {method} {url}")
    return json.loads(p.stdout)


def _api_request_json(method: str, url: str, auth: str, body: dict | None = None) -> dict:
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "desktop-pet-with-llms-linux-release-uploader")
    req.add_header("Authorization", auth)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _list_assets(repo: str, release_id: int, auth: str) -> dict[str, dict]:
    url = f"https://api.github.com/repos/{repo}/releases/{release_id}/assets"
    if _have_curl():
        assets = _curl_json("GET", url, auth, timeout_s=60)
    else:
        req = urllib.request.Request(url, method="GET")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("User-Agent", "desktop-pet-with-llms-linux-release-uploader")
        req.add_header("Authorization", auth)
        with urllib.request.urlopen(req, timeout=60) as resp:
            assets = json.loads(resp.read().decode("utf-8"))
    out = {}
    for a in assets:
        if isinstance(a, dict) and a.get("name"):
            out[a["name"]] = a
    return out


def _upload_asset_curl(repo: str, release_id: int, auth: str, file_path: Path, timeout_s: int) -> None:
    qs = urllib.parse.urlencode({"name": file_path.name})
    url = f"https://uploads.github.com/repos/{repo}/releases/{release_id}/assets?{qs}"
    cmd = [
        "curl",
        "-sS",
        "--fail-with-body",
        "--retry",
        "5",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--max-time",
        str(timeout_s),
        "-X",
        "POST",
        "-H",
        "Accept: application/vnd.github+json",
        "-H",
        "User-Agent: desktop-pet-with-llms-linux-release-uploader",
        "-H",
        f"Authorization: {auth}",
        "-H",
        "Content-Type: application/octet-stream",
        "--data-binary",
        f"@{str(file_path)}",
        url,
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if p.returncode != 0:
        msg = (p.stderr.strip() or p.stdout.strip() or "").strip()
        raise RuntimeError(f"upload failed: {file_path.name}: {msg}")


def _upload_asset_streaming(repo: str, release_id: int, auth: str, file_path: Path, timeout_s: int) -> None:
    qs = urllib.parse.urlencode({"name": file_path.name})
    url = f"https://uploads.github.com/repos/{repo}/releases/{release_id}/assets?{qs}"
    parsed = urllib.parse.urlparse(url)
    conn = HTTPSConnection(parsed.netloc, timeout=timeout_s)
    content_length = file_path.stat().st_size
    headers = {
        "Authorization": auth,
        "Accept": "application/vnd.github+json",
        "User-Agent": "desktop-pet-with-llms-linux-release-uploader",
        "Content-Type": "application/octet-stream",
        "Content-Length": str(content_length),
    }
    conn.putrequest("POST", parsed.path + ("?" + parsed.query if parsed.query else ""))
    for k, v in headers.items():
        conn.putheader(k, v)
    conn.endheaders()

    sent = 0
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            conn.send(chunk)
            sent += len(chunk)
            if sent % (256 * 1024 * 1024) == 0:
                sys.stdout.write(f"[{file_path.name}] {sent}/{content_length}\n")
                sys.stdout.flush()

    resp = conn.getresponse()
    data = resp.read()
    if resp.status >= 300:
        msg = data.decode("utf-8", errors="replace")
        raise RuntimeError(f"upload failed: {file_path.name} status={resp.status} body={msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tag", required=False, default="")
    ap.add_argument("--release-id", type=int, default=0)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pattern", action="append", default=[])
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--retries", type=int, default=3)
    args = ap.parse_args()

    auth = _git_github_basic_auth()
    release_id = int(args.release_id)
    if release_id <= 0:
        if not args.tag:
            raise SystemExit("--tag or --release-id is required")
        release_url = f"https://api.github.com/repos/{args.repo}/releases/tags/{urllib.parse.quote(args.tag)}"
        last_err: Exception | None = None
        for attempt in range(1, 6):
            try:
                if _have_curl():
                    release = _curl_json("GET", release_url, auth, timeout_s=60)
                else:
                    release = _api_request_json("GET", release_url, auth)
                release_id = int(release["id"])
                break
            except Exception as e:
                last_err = e
                time.sleep(min(15, 2 * attempt))
        if release_id <= 0:
            raise SystemExit(f"failed to resolve release id: {last_err}")
    existing = _list_assets(args.repo, release_id, auth)

    base_dir = Path(args.dir).resolve()
    patterns = args.pattern or ["*"]
    files: list[Path] = []
    for pat in patterns:
        for p in glob.glob(str(base_dir / pat)):
            fp = Path(p)
            if fp.is_file():
                files.append(fp)
    files = sorted(files, key=lambda p: p.name)

    for fp in files:
        if fp.name in existing:
            print(f"skip: {fp.name}")
            continue
        ok = False
        for attempt in range(1, args.retries + 1):
            try:
                print(f"upload: {fp.name} ({fp.stat().st_size} bytes) attempt {attempt}/{args.retries}")
                if _have_curl():
                    _upload_asset_curl(args.repo, release_id, auth, fp, args.timeout)
                else:
                    _upload_asset_streaming(args.repo, release_id, auth, fp, args.timeout)
                ok = True
                break
            except Exception as e:
                print(f"failed: {fp.name}: {e}")
                time.sleep(min(30, 3 * attempt))
        if not ok:
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
