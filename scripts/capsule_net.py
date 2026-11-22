"""Anonymous, opt-in HTTPS, with per-object redirect policy (including signed CDN hops)."""
from __future__ import annotations

import http.client
import os
import re
import ssl
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

from capsule_io import identity, length, require


HF = "https://huggingface.co/baselabs/joey-1"
ONE = "https://github.com/OmniJev/OneJev"
RAW = "https://raw.githubusercontent.com"
KERNEL = "https://api.github.com/repos/Dao-AILab/causal-conv1d/releases"
FLA_REV = "9c8e42e762fce087c27b673af4922795d9edb85e"
FLA_LICENSE = f"{RAW}/fla-org/flash-linear-attention/{FLA_REV}/LICENSE"


def parsed(url):
    require(url.isascii() and len(url) <= 16384 and not re.search(r"[\x00-\x20\x7f\\#]", url),
            "unsafe transport URL")
    u = urlsplit(url)
    require(u.scheme == "https" and u.hostname and u.netloc == u.hostname and u.path.startswith("/"),
            "only uncredentialed standard-port HTTPS is admitted")
    require(not re.search("%(?:2f|5c|00|25)", u.path, re.I)
            and all(p not in (".", "..") for p in unquote(u.path).split("/")), "unsafe transport path")
    return u


def redirect_ok(initial, target):
    a, b = parsed(initial), parsed(target)
    if target == initial:
        return True
    if a.hostname == "huggingface.co" and a.path.startswith("/baselabs/joey-1/resolve/"):
        suffix = a.path.split("/resolve/", 1)[1]
        if b.hostname == a.hostname:
            return b.path == "/api/resolve-cache/models/baselabs/joey-1/" + suffix
        if b.hostname == "cas-bridge.xethub.hf.co":
            return b.path.startswith("/xet-bridge-")
        return b.path.startswith("/repos/") and b.hostname in {
            "cdn-lfs.hf.co", "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co",
            "cdn-lfs.huggingface.co", "cdn-lfs-us-1.huggingface.co", "cdn-lfs-eu-1.huggingface.co",
        }
    if initial.startswith(KERNEL + "/assets/"):
        return (b.hostname == "release-assets.githubusercontent.com"
                and b.path.startswith("/github-production-release-asset/"))
    return False


class Network:
    def __init__(self, sources):
        require(os.geteuid() != 0, "run network ingress as an unprivileged user")
        self.sources = sources
        # Do not consult proxies, .netrc, cookies, SSLKEYLOGFILE or user-selected CA files.
        self.tls = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.tls.minimum_version = ssl.TLSVersion.TLSv1_2
        for ca in ("/etc/ssl/certs/ca-certificates.crt", "/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/cert.pem"):
            if Path(ca).is_file():
                self.tls.load_verify_locations(cafile=ca)
                break
        else:
            require(False, "no supported system CA bundle on ingestion host")

    @contextmanager
    def get(self, initial, *, offset=0, accept=None):
        u = parsed(initial)
        require(initial in self.sources and (not u.query or (
            u.hostname == "api.github.com" and u.path == "/repos/OmniJev/OneJev/contents/LICENSE"
            and re.fullmatch("ref=[0-9a-f]{40}", u.query))), "unapproved public source")
        url = initial
        headers = {"Accept-Encoding": "identity", "User-Agent": "capsule-ingress/1",
                   "Accept": accept or "*/*"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        for _ in range(6):
            u = parsed(url)
            conn = http.client.HTTPSConnection(u.hostname, context=self.tls, timeout=90)
            try:
                conn.request("GET", u.path + ("?" + u.query if u.query else ""), headers=headers)
                with conn.getresponse() as resp:
                    if resp.status not in (301, 302, 303, 307, 308):
                        yield resp
                        return
                    location = resp.getheader("Location")
                    require(location and not re.search(r"[\x00-\x20\x7f\\]", location), "invalid redirect")
                    url = urljoin(url, location)
                    require(redirect_ok(initial, url), "redirect outside the object's delivery policy")
            finally:
                conn.close()
        require(False, "excessive redirects")


def fetch(net, url, limit):
    with net.get(url) as r:
        require(r.status == 200 and r.getheader("Content-Range") is None, "metadata/source GET refused")
        identity(r)
        n = length(r)
        require(n is None or 0 < n <= limit, "metadata/source exceeds limit")
        data = r.read(limit + 1)
        require(0 < len(data) <= limit and n in (None, len(data)), "incomplete or oversized metadata/source")
        return data
