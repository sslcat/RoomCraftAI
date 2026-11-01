"""Existing Runpod credentials only; no login, key creation, or disk-token cache.

The Windows deadline controller must run in the normal user's logon context.
This module does not arm a guard, start a Pod, or expose a mutation command.
"""

import ctypes
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def owner_settings():
    """Nonsecret local bindings; never include this ignored profile in a release."""
    path = Path(__file__).resolve().parents[1] / ".cache/runpod-owner.json"
    return json.loads(path.read_text()) if path.is_file() else {}


WINDOWS_TARGET = os.environ.get("ROOMCRAFT_RUNPOD_CREDENTIAL_TARGET") or owner_settings().get(
    "credential_target", "Runpod/RoomCraftAI"
)


def ssh_identity():
    value = os.environ.get("ROOMCRAFT_SSH_KEY") or owner_settings().get("ssh_key")
    return Path(value).expanduser() if value else Path.home() / ".ssh/roomcraft_runpod"


class CredentialSourceError(RuntimeError):
    pass


class RunpodCredential:
    __slots__ = ("source", "_value")

    def __init__(self, source, value):
        value = value.strip()
        if (
            not value
            or not value.isascii()
            or any(c.isspace() for c in value)
            or value.startswith(('"', "'"))
        ):
            raise CredentialSourceError("Invalid credential format; value withheld")
        self.source, self._value = source, value

    def __repr__(self):
        return f"RunpodCredential(source={self.source!r}, value=[redacted])"

    def authorization_header(self):
        # Return only to an HTTPS request object, never to a logger or shell.
        return {"Authorization": "Bearer " + self._value}


def read_windows_credential():
    if os.name != "nt":
        raise CredentialSourceError("The existing Windows credential source requires Windows")
    from ctypes import wintypes

    class CREDENTIALW(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    pointer = ctypes.POINTER(CREDENTIALW)()
    api = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
    api.CredReadW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.POINTER(CREDENTIALW)),
    ]
    api.CredReadW.restype = wintypes.BOOL
    api.CredFree.argtypes = [ctypes.c_void_p]
    api.CredFree.restype = None
    if not api.CredReadW(WINDOWS_TARGET, 1, 0, ctypes.byref(pointer)):
        code = ctypes.get_last_error()
        if code == 1312:
            raise CredentialSourceError(
                "Windows logon session unavailable; run in the normal user context (1312)"
            )
        raise CredentialSourceError(
            f"Existing Runpod Windows credential unavailable ({code}); no stale-file fallback"
        )
    try:
        blob = ctypes.string_at(
            pointer.contents.CredentialBlob, pointer.contents.CredentialBlobSize
        )
        return blob.decode("utf-16-le").rstrip("\0")
    finally:
        api.CredFree(pointer)


def load_runpod_credential():
    value = os.environ.get("RUNPOD_API_KEY")
    if value:
        return RunpodCredential("process environment: RUNPOD_API_KEY", value)
    return RunpodCredential(
        "Windows Credential Manager: " + WINDOWS_TARGET, read_windows_credential()
    )


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def get_known_resource(credential, kind, resource_id):
    if kind not in ("pods", "global-volumes") or not resource_id.isalnum():
        raise ValueError("Only a known Runpod Pod or Global Volume read is allowed")
    url = f"https://api.runpod.io/v2/{kind}/{resource_id}"
    request = urllib.request.Request(
        url,
        method="GET",
        headers={**credential.authorization_header(), "Accept": "application/json"},
    )
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=20) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        # Never log error bodies, headers, or the credential-bearing Request.
        raise CredentialSourceError(
            f"Runpod read rejected: HTTP {error.code}; source={credential.source}"
        ) from None
