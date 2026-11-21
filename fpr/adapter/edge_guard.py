"""Edge guard for a public FPR / vLLM endpoint.

Pure ASGI middleware in front of vLLM's API server (`--middleware adapter.edge_guard.EdgeGuard`), two checks:

1. Paths. Only the OpenAI / Anthropic API (/v1/...) and the monitoring routes /health, /metrics, /version are served.
   vLLM also mounts admin and debug routes (/scale_elastic_ep, /is_scaling_elastic_ep, /pause, /sleep, /collective_rpc,
   /openapi.json, /docs, /load, /invocations, /tokenize, ...).

2. Media URLs (blind SSRF). vLLM fetches http(s) image / audio / video URLs from the request on the server. A request
   whose media URL points at a loopback, private, link-local, multicast, reserved or otherwise non-global address
   (literal, or as the host resolves now) is refused with 400 before it reaches vLLM, and so is a host that does not
   resolve. Redirects are switched off separately (VLLM_MEDIA_URL_ALLOW_REDIRECTS=0), so a public URL cannot bounce to
   an internal one. data: URLs are untouched; vLLM refuses file: URLs unless --allowed-local-media-path is set.
"""
import asyncio
import ipaddress
import json
import posixpath
import socket
from urllib.parse import urlsplit

ALLOWED_EXACT = {"/health", "/metrics", "/version"}
URL_KEYS = {"url", "image_url", "audio_url", "video_url", "file_url"}
MAX_URLS = 32  # media URLs per request (each one is resolved)
DNS_TIMEOUT_S = 3.0


def path_ok(path: str) -> bool:
    if path in ALLOWED_EXACT:
        return True
    if not path.startswith("/v1/"):
        return False
    norm = posixpath.normpath(path)
    return norm == path or norm + "/" == path


def bad_ip(addr: str) -> bool:
    ip = ipaddress.ip_address(addr.split("%", 1)[0])
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not ip.is_global or ip.is_multicast or ip.is_reserved or ip.is_loopback or ip.is_link_local


async def url_blocked(url: str) -> bool:
    try:
        parts = urlsplit(url.strip())
        host, port = parts.hostname, parts.port
    except ValueError:
        return True
    if parts.scheme.lower() not in ("http", "https"):
        return False
    if not host:
        return True
    try:
        return bad_ip(host)
    except ValueError:
        pass
    h = host.rstrip(".").lower()
    if h == "localhost" or h.endswith((".localhost", ".local", ".internal", ".lan", ".home.arpa")):
        return True
    try:
        infos = await asyncio.wait_for(
            asyncio.get_running_loop().getaddrinfo(h, port or (443 if parts.scheme.lower() == "https" else 80),
                                                   type=socket.SOCK_STREAM), DNS_TIMEOUT_S)
    except Exception:
        return True
    return not infos or any(bad_ip(i[4][0]) for i in infos)


def media_urls(obj, out: list, depth: int = 0) -> None:
    if depth > 64:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                if k in URL_KEYS and v.lstrip()[:8].lower().startswith(("http://", "https://")):
                    out.append(v)
            else:
                media_urls(v, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            media_urls(v, out, depth + 1)


async def reply(send, status: int, obj) -> None:
    body = json.dumps(obj).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


def invalid(msg: str) -> dict:
    return {"error": {"message": msg, "type": "invalid_request_error", "param": None, "code": 400}}


class EdgeGuard:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        if not path_ok(path):
            return await reply(send, 404, {"detail": "Not Found"})
        if scope.get("method") != "POST" or not path.startswith("/v1/"):
            return await self.app(scope, receive, send)
        chunks, more = [], True
        while more:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            chunks.append(msg.get("body", b""))
            more = msg.get("more_body", False)
        body = b"".join(chunks)
        if b"ttp" in body:  # a media URL needs "http" (JSON may escape the slashes); skip the parse otherwise
            try:
                obj = json.loads(body)
            except Exception:
                obj = None
            if obj is not None:
                urls: list = []
                media_urls(obj, urls)
                if len(urls) > MAX_URLS:
                    return await reply(send, 400, invalid(f"too many media URLs (max {MAX_URLS})"))
                for u in urls:
                    if await url_blocked(u):
                        return await reply(send, 400, invalid("media URL not allowed: it points at an internal or "
                                                              "unresolvable address"))
        replayed = False

        async def replay():
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        return await self.app(scope, replay, send)
