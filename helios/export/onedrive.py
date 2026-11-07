"""OneDrive and external disk preservation mirroring engine for Helios.

Backward compatibility module re-exporting disk preservation engine configured for
external storage (primary: `/Volumes/One Touch/Helios/`).

Author: Arron Kian Parejas <arronkianparejas@gmail.com>
"""

from __future__ import annotations

from helios.export.disk import (
    DEFAULT_DISK_PATH,
    DEFAULT_ONEDRIVE_PATH,
    FALLBACK_DISK_PATH,
    PRIMARY_DISK_PATH,
    DiskMirrorManager,
    DiskSyncIntegrityError,
    MirroredItem,
    OneDriveMirrorManager,
    OneDriveSyncIntegrityError,
    compute_sha256,
    mirror_to_disk,
    mirror_to_onedrive,
)

__all__ = [
    "DEFAULT_DISK_PATH",
    "DEFAULT_ONEDRIVE_PATH",
    "FALLBACK_DISK_PATH",
    "PRIMARY_DISK_PATH",
    "DiskMirrorManager",
    "DiskSyncIntegrityError",
    "MirroredItem",
    "OneDriveMirrorManager",
    "OneDriveSyncIntegrityError",
    "compute_sha256",
    "mirror_to_disk",
    "mirror_to_onedrive",
]
