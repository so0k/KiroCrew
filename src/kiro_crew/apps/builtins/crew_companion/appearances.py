"""Appearance packs: the companion's avatar library.

The desktop app kept these in Electron's ``userData`` and served them over IPC. Here
they live in the app's own data directory and are served over HTTP, because the
renderer is a page rather than a privileged window. The pack SHAPE is unchanged — a
manifest plus animation files, as ``appearanceTypes.ts`` defines it — so packs the
user already made remain loadable.

Three properties this has to keep, all learned from the desktop version:

* **A pack the user made is precious.** Custom art is unrecoverable if lost, so writes
  go through a temp file and a rename, and a delete only ever touches a custom pack's
  own directory.
* **A malformed pack must not take the companion down.** A pack is third-party content,
  possibly hand-edited. Anything unreadable is skipped with a warning and the others
  still load, rather than one bad manifest emptying the library.
* **The built-in ghost is not a file.** It ships with the app and cannot be deleted or
  renamed, so it is registered from code and always present even when the custom
  directory is empty.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kiro_crew.platform_compat import chmod_safe

logger = logging.getLogger(__name__)

#: The built-in ghost's id. Referenced by the renderer, so it is a contract.
DEFAULT_PACK = "kiro-ghost"

#: Custom packs live one directory each, named by id, under this subdirectory.
PACKS_DIRNAME = "appearances"

#: Per-file ceiling for pack content. Generous for art, small enough that a
#: hand-edited manifest claiming a gigabyte cannot exhaust memory on read.
MAX_FILE_BYTES = 8 * 1024 * 1024

#: Animation formats the renderer knows how to draw.
FORMATS = ("svg", "lottie", "sprite")


@dataclass(frozen=True)
class PackMeta:
    """A pack as the gallery lists it — metadata only, no art."""

    id: str
    name: str
    author: str
    description: str
    #: "builtin" or "custom". Only custom packs can be deleted.
    type: str
    format: str
    #: True when the user has recoloured this pack.
    recoloured: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "author": self.author,
            "description": self.description,
            "type": self.type,
            "format": self.format,
            "recoloured": self.recoloured,
        }


def _safe_id(raw: Any) -> str | None:
    """Validate a pack id as a single safe path segment.

    A pack id becomes a directory name, so this is the boundary that stops
    ``../`` or an absolute path from escaping the packs directory. Rejecting is
    correct here rather than sanitising: a caller sending a traversal is not making
    a typo, and silently rewriting it would hide that.
    """
    if not isinstance(raw, str):
        return None
    ident = raw.strip()
    if not ident or len(ident) > 64:
        return None
    if ident in (".", ".."):
        return None
    # Letters, digits, dash and underscore only — no separators, no dots.
    if not all(c.isalnum() or c in "-_" for c in ident):
        return None
    return ident


class AppearanceStore:
    """Reads and writes the companion's appearance packs."""

    def __init__(self, data_dir: Path) -> None:
        self._root = Path(data_dir) / PACKS_DIRNAME
        #: id -> colour map, for packs the user has recoloured.
        self._colour_maps: dict[str, dict[str, str]] = {}
        self._colour_path = Path(data_dir) / "crew-companion-colours.json"

    # ── setup ───────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Prepare the packs directory and read any saved colour maps."""
        try:
            self._root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning("crew-companion: cannot create packs dir: %s", exc)
        try:
            if self._colour_path.exists():
                raw = json.loads(self._colour_path.read_text("utf-8"))
                if isinstance(raw, dict):
                    self._colour_maps = {
                        k: v for k, v in raw.items() if isinstance(v, dict)
                    }
        except (OSError, ValueError) as exc:
            # A corrupt colour file costs the user their recolouring, not their art,
            # so carrying on with defaults beats refusing to start.
            logger.warning("crew-companion: colour maps unreadable: %s", exc)

    # ── reads ───────────────────────────────────────────────────────────────

    def list_packs(self) -> list[dict[str, Any]]:
        """Every pack, built-in first.

        One unreadable pack is skipped rather than failing the list: the gallery
        showing four of five packs is recoverable, showing none is not.
        """
        packs: list[dict[str, Any]] = [self._builtin_meta().to_dict()]
        if not self._root.exists():
            return packs
        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir():
                continue
            meta = self._read_meta(entry)
            if meta is not None:
                packs.append(meta.to_dict())
            else:
                logger.warning("crew-companion: skipping unreadable pack %s", entry.name)
        return packs

    def pack_detail(self, pack_id: str) -> dict[str, Any] | None:
        """A pack with its animation content inlined, ready to render.

        Content is returned inline rather than as URLs because a custom pack's files
        are user data outside the served tree; handing back paths would mean opening
        a file-serving route over that directory.
        """
        ident = _safe_id(pack_id)
        if ident is None:
            return None
        if ident == DEFAULT_PACK:
            return {
                "meta": self._builtin_meta().to_dict(),
                # The built-in ghost's art is bundled with the frontend, so the
                # renderer already has it and needs no content here.
                "animations": {},
                "colorMap": self._colour_maps.get(DEFAULT_PACK) or {},
            }

        pack_dir = self._root / ident
        manifest = self._read_manifest(pack_dir)
        if manifest is None:
            return None

        animations: dict[str, Any] = {}
        states = manifest.get("states")
        if isinstance(states, dict):
            for slot, filename in states.items():
                content = self._read_pack_file(pack_dir, filename)
                if content is None:
                    continue
                animations[slot] = {
                    "content": content,
                    "format": manifest.get("meta", {}).get("format", "svg"),
                }
        return {
            "meta": (self._read_meta(pack_dir) or self._builtin_meta()).to_dict(),
            "animations": animations,
            "sprite": manifest.get("sprite") or {},
            "colorMap": self._colour_maps.get(ident) or {},
        }

    def colour_map(self, pack_id: str) -> dict[str, str]:
        ident = _safe_id(pack_id)
        return dict(self._colour_maps.get(ident or "", {}))

    # ── writes ──────────────────────────────────────────────────────────────

    def set_colour_map(self, pack_id: str, colours: Any) -> bool:
        """Record a recolouring. Returns False when the input is unusable."""
        ident = _safe_id(pack_id)
        if ident is None or not isinstance(colours, dict):
            return False
        # Only string→string pairs; anything else would break the SVG rewrite that
        # consumes this on the renderer side.
        clean = {
            str(k): str(v)
            for k, v in colours.items()
            if isinstance(k, str) and isinstance(v, str)
        }
        self._colour_maps[ident] = clean
        self._save_colours()
        return True

    def delete_pack(self, pack_id: str) -> bool:
        """Delete a CUSTOM pack. The built-in is refused."""
        ident = _safe_id(pack_id)
        if ident is None or ident == DEFAULT_PACK:
            return False
        pack_dir = self._root / ident
        # Resolve and re-check containment: the id is already validated, and this is
        # the second belt on an irreversible recursive delete.
        try:
            resolved = pack_dir.resolve()
            if self._root.resolve() not in resolved.parents:
                return False
            if not resolved.is_dir():
                return False
            shutil.rmtree(resolved)
        except OSError as exc:
            logger.warning("crew-companion: pack delete failed: %s", exc)
            return False
        self._colour_maps.pop(ident, None)
        self._save_colours()
        return True

    def save_pack(self, pack_id: str, manifest: Any, files: Any) -> bool:
        """Create or replace a custom pack.

        ``files`` maps filename to content (text for svg/lottie, base64 for a sprite
        sheet). Written to a temp directory and moved into place, so an interrupted
        save cannot leave a half-written pack that lists in the gallery and then
        fails to render.
        """
        ident = _safe_id(pack_id)
        if ident is None or ident == DEFAULT_PACK:
            return False
        if not isinstance(manifest, dict) or not isinstance(files, dict):
            return False
        # The read path needs a `meta` dict to list a pack at all. Without this check a
        # save could report success and then be invisible in the gallery — present on
        # disk, skipped on read — which is far harder to diagnose than a refusal here.
        if not isinstance(manifest.get("meta"), dict):
            logger.warning("crew-companion: pack manifest has no meta: %s", ident)
            return False

        staging = self._root / f".tmp-{ident}-{os.getpid()}"
        target = self._root / ident
        try:
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2), "utf-8"
            )
            for name, content in files.items():
                safe = _safe_filename(name)
                if safe is None or not isinstance(content, str):
                    continue
                if len(content.encode("utf-8")) > MAX_FILE_BYTES:
                    logger.warning("crew-companion: pack file too large: %s", safe)
                    continue
                (staging / safe).write_text(content, "utf-8")
            if target.exists():
                shutil.rmtree(target)
            os.replace(staging, target)
            return True
        except OSError as exc:
            logger.warning("crew-companion: pack save failed: %s", exc)
            try:
                if staging.exists():
                    shutil.rmtree(staging)
            except OSError:
                pass
            return False

    # ── internals ───────────────────────────────────────────────────────────

    def _builtin_meta(self) -> PackMeta:
        return PackMeta(
            id=DEFAULT_PACK,
            name="Kiro",
            author="Kiro Crew",
            description="The default companion.",
            type="builtin",
            format="svg",
            recoloured=bool(self._colour_maps.get(DEFAULT_PACK)),
        )

    def _read_manifest(self, pack_dir: Path) -> dict[str, Any] | None:
        path = pack_dir / "manifest.json"
        try:
            if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                return None
            raw = json.loads(path.read_text("utf-8"))
            return raw if isinstance(raw, dict) else None
        except (OSError, ValueError):
            return None

    def _read_meta(self, pack_dir: Path) -> PackMeta | None:
        manifest = self._read_manifest(pack_dir)
        if manifest is None:
            return None
        meta = manifest.get("meta")
        if not isinstance(meta, dict):
            return None
        ident = _safe_id(meta.get("id")) or pack_dir.name
        fmt = meta.get("format")
        return PackMeta(
            id=ident,
            name=str(meta.get("name") or ident),
            author=str(meta.get("author") or ""),
            description=str(meta.get("description") or ""),
            type="custom",
            format=fmt if fmt in FORMATS else "svg",
            recoloured=bool(self._colour_maps.get(ident)),
        )

    def _read_pack_file(self, pack_dir: Path, filename: Any) -> str | None:
        safe = _safe_filename(filename)
        if safe is None:
            return None
        path = pack_dir / safe
        try:
            if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                return None
            return path.read_text("utf-8")
        except (OSError, ValueError):
            return None

    def _save_colours(self) -> None:
        try:
            tmp = self._colour_path.with_suffix(f".json.tmp.{os.getpid()}")
            tmp.write_text(json.dumps(self._colour_maps, indent=2), "utf-8")
            # chmod_safe, not os.chmod: the root AGENTS.md mandates the
            # platform_compat shim, which is a no-op where POSIX modes mean
            # nothing (Windows) instead of raising or silently misleading.
            chmod_safe(tmp, 0o600)
            os.replace(tmp, self._colour_path)
        except OSError as exc:
            logger.warning("crew-companion: colour map write failed: %s", exc)


def _safe_filename(raw: Any) -> str | None:
    """Validate a filename as a single segment inside a pack directory.

    Same reasoning as ``_safe_id``: these names come from a manifest that may be
    hand-edited, and a name containing a separator would read or write outside the
    pack.
    """
    if not isinstance(raw, str):
        return None
    name = raw.strip()
    if not name or len(name) > 128:
        return None
    if name.startswith(".") or "/" in name or "\\" in name or ".." in name:
        return None
    return name
