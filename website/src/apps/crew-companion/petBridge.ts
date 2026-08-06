/**
 * The API surface `useDrag` and `PetContextMenu` expect, backed by Kiro Crew.
 *
 * In the desktop app these were Electron IPC calls into the main process, which owned
 * the pet window and persisted its position to userData. Here the companion is a
 * full-display overlay and its state lives in the gateway, so the same method names
 * are re-implemented over HTTP and the preload bridge instead.
 *
 * The shape is deliberately identical (`getWindowPosition`, `savePosition`,
 * `dragStart`, …) so the ported hook needs one changed line — where it reads its api
 * — and none of its logic. Its drag maths, edge-snap thresholds and click-vs-drag
 * disambiguation stay exactly as written.
 */
import { i18nT } from '../../i18n/t'
import { OPTIONAL_STATES, REQUIRED_STATES } from './appearanceTypes'
import type { AnimationFormat, PackMeta, SpriteConfig } from './appearanceTypes'
import type { Reminder } from './types'
import type { SpriteImportResult } from './spriteImportTypes'
import { REMOVE_PATH, APPEARANCE_EXPORT_PATH, APPEARANCE_IMPORT_PATH, PETDEX_FETCH_PATH, APPEARANCES_PATH, APPEARANCE_COLOURS_PATH, APPEARANCE_DETAIL_PATH, APPEARANCE_SAVE_PATH, APPEARANCE_DELETE_PATH, CONFIG_PATH, REMINDERS_PATH } from './constants'

/** The preload bridge, absent when this page is opened in an ordinary browser. */
type Preload = {
  setFocusable?(v: boolean): void
  /**
   * Report the companion's and bubble's hitboxes to the main process, which polls
   * the cursor and toggles ignore-mouse itself. Passing rects rather than a
   * pointer-enter/leave boolean is what removes the IPC round-trip that let a click
   * on the companion body fall through to the window behind it.
   */
  updateHitbox?(
    pet: { x: number; y: number; w: number; h: number } | null,
    bubble: { x: number; y: number; w: number; h: number } | null,
  ): void
  /**
   * Report the context menu's rect while it is open, or null when it closes.
   *
   * The overlay is click-through everywhere except the reported hitboxes, so the
   * menu's own rect is added as one — making the menu clickable while the rest of
   * the desktop stays click-through. This replaces the old hack of making the whole
   * window interactive for the menu's lifetime.
   */
  setMenuHitbox?(rect: { x: number; y: number; w: number; h: number } | null): void
  contextMenuAction?(action: string): void
  appearanceChanged?(): void
  onAppearanceChanged?(cb: () => void): () => void
  openExternal?(url: string): void
  galleryOpen?(): void
  galleryClose?(): void
}

function preload(): Preload | undefined {
  return (window as unknown as { crewCompanion?: Preload }).crewCompanion
}

/** The detail payload, named so call sites need no indexed-access type. */
/**
 * What the gallery's write calls answer with.
 *
 * An object rather than a boolean because these refusals have reasons the dialog
 * shows — "already installed", "too large" — and a boolean would leave the user with a
 * failure and no cause. `value` carries the saved pack's metadata back, which the
 * editor uses to update its list without a re-read.
 */
export interface GalleryResult {
  ok: boolean
  /** The saved pack's id, which the editor uses to select it after a save. */
  packId?: string
  error?: string
  value?: unknown
  id?: string
}

/** The editor's authoring shape: art grouped by purpose rather than as files. */
export interface PackInput {
  meta: Record<string, unknown>
  states: Record<string, string>
  moods?: Record<string, string>
  random?: Record<string, string>
}


export interface PackDetail {
  animations?: Record<string, { content?: string; format?: AnimationFormat } | string>
  sprite?: Partial<SpriteConfig>
  colorMap?: Record<string, string>
  [k: string]: unknown
}

export interface PetBridge {
  getWindowPosition?: () => Promise<{ x: number; y: number } | null>
  savePosition?: (x: number, y: number) => void
  dragStart?: (offsetX: number, offsetY: number) => void
  dragEnd?: () => void
  onDragUpdate?: (cb: (x: number, y: number) => void) => (() => void) | undefined
  onDragEnded?: (cb: (x: number, y: number) => void) => (() => void) | undefined
  /**
   * Report the companion's and bubble's hitboxes to the main process.
   *
   * The main process polls the cursor at ~60fps and toggles this overlay's
   * ignore-mouse from these rects directly, with no pointer-enter/leave round-trip
   * — which is what stopped clicks on the companion body falling through to the
   * window behind it.
   */
  updateHitbox?: (
    pet: { x: number; y: number; w: number; h: number } | null,
    bubble: { x: number; y: number; w: number; h: number } | null,
  ) => void
  /**
   * Report the context menu's rect while it is open, or null when it closes.
   *
   * The overlay is click-through except over the reported hitboxes, so the menu's
   * rect is reported as one and the cursor poll makes the menu region interactive.
   * Everything else stays click-through — no more making the whole window
   * interactive for the menu's lifetime.
   */
  setMenuHitbox?: (rect: { x: number; y: number; w: number; h: number } | null) => void
  /** The full reminder list, for the panel's "see all" view. */
  remindersList?: () => Promise<Reminder[]>
  remindersRemove?: (id: string) => Promise<boolean>
  /** Colour presets the user saved by hand. */
  presetsLoadCustom?: () => Promise<unknown[]>
  presetsSaveCustom?: (presets: unknown[]) => Promise<boolean>
  contextMenuAction?: (action: string) => void

  // ── Walk commands (main-process steered) ───────────────────────────────────
  // In the desktop app these carried the agent's "go here" / "tuck at that edge"
  // commands from the main process to useWalking. The autonomous wander never used
  // them — it called walkPath directly in the renderer. They are declared so the
  // ported useWalking type-checks and can adopt them later; left undefined for now
  // (like dragStart above), so the hook's api?.on*?.() guards make them no-ops.
  onWalk?: (cb: (x: number, y: number) => void) => (() => void) | undefined
  onWalkPath?: (cb: (points: Array<{ x: number; y: number }>) => void) => (() => void) | undefined
  onWalkCancel?: (cb: () => void) => (() => void) | undefined
  onWalkAppend?: (cb: (points: Array<{ x: number; y: number }>) => void) => (() => void) | undefined
  onHide?: (cb: (edge: 'left' | 'right') => void) => (() => void) | undefined
  walkDone?: () => void

  // ── Appearance packs ──────────────────────────────────────────────────────
  getCrewCompanionConfig?: () => Promise<{ activeAppearance?: string; [k: string]: unknown } | null>
  galleryListPacks?: () => Promise<PackMeta[]>
  galleryGetPackDetail?: (packId: string) => Promise<PackDetail | null>
  gallerySetActive?: (packId: string) => Promise<GalleryResult>
  galleryDeletePack?: (packId: string) => Promise<boolean>
  /** The gallery's name for the same call. */
  galleryDelete?: (packId: string) => Promise<boolean>
  galleryExport?: (packId: string) => Promise<unknown | null>
  /** Both import entry points pick a file; the desktop app had two channels. */
  galleryImportBundle?: () => Promise<GalleryResult>
  galleryImportFile?: () => Promise<GalleryResult>
  gallerySaveSpritePack?: (data: SpriteImportResult) => Promise<GalleryResult>
  petdexFetch?: (input: string) => Promise<Record<string, unknown>>
  /** Read one file out of an installed pack, base64-encoded. */
  galleryReadPackFile?: (packId: string, filename: string) => Promise<string | null>
  /** Ask the user for a sprite sheet and hand back its bytes. */
  importSpriteFile?: () => Promise<GalleryResult & { value?: { content: string } }>
  updateConfig?: (patch: Record<string, unknown>) => Promise<boolean>
  openExternal?: (url: string) => void
  closeGallery?: () => void
  onGalleryPacksChanged?: (cb: () => void) => () => void
  onConfigUpdated?: (cb: () => void) => () => void
  gallerySavePack?: (data: PackInput) => Promise<GalleryResult>
  presetsGetColorMap?: (packId: string) => Promise<Record<string, string> | null>
  gallerySetColorMap?: (packId: string, colorMap: Record<string, string>) => Promise<GalleryResult>
  /** Change notifications. Polled here rather than pushed — see the note below. */
  onGalleryActiveChanged?: (cb: () => void) => () => void
  onColorMapChanged?: (cb: (d: { packId: string; colorMap: Record<string, string> }) => void) => () => void
}

/** Detail URL for a pack. URLSearchParams encodes the id correctly by construction. */
function detailUrl(packId: string): string {
  const q = new URLSearchParams()
  q.set('id', packId)
  return `${APPEARANCE_DETAIL_PATH}?${q.toString()}`
}

async function getJson<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url, { credentials: 'same-origin' })
    if (!r.ok) return null
    return (await r.json()) as T
  } catch {
    return null
  }
}

/**
 * POST and read the response body.
 *
 * Separate from `postJson` because the transfer calls answer with a reason when they
 * refuse — "already installed", "too large" — and the dialog shows it. Collapsing that
 * to a boolean would leave the user with a failure and no cause.
 */
async function postForJson<T>(url: string, body: unknown): Promise<T | null> {
  try {
    const r = await fetch(url, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    // Read the body either way: a refusal carries the reason.
    return (await r.json()) as T
  } catch {
    return null
  }
}

/**
 * Ask the user for a bundle file.
 *
 * The desktop app opened the OS dialog through Electron. This page is web content, so
 * a file input is the only way to read a file the user chose — and it keeps the read
 * inside the sandbox rather than granting the page a path.
 */
function pickJsonFile(): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'application/json,.json'
    input.style.display = 'none'
    // Resolve on cancel too, via focus returning to the window, so a dismissed dialog
    // does not leave the caller awaiting forever.
    let settled = false
    const done = (f: File | null) => {
      if (settled) return
      settled = true
      input.remove()
      resolve(f)
    }
    input.addEventListener('change', () => done(input.files?.[0] ?? null))
    input.addEventListener('cancel', () => done(null))
    document.body.appendChild(input)
    input.click()
  })
}

/** Ask the user for an image, for the sprite importer. */
function pickImageFile(): Promise<File | null> {
  return new Promise((resolve) => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = 'image/png,image/webp,image/gif,.png,.webp,.gif'
    input.style.display = 'none'
    let settled = false
    const done = (f: File | null) => {
      if (settled) return
      settled = true
      input.remove()
      resolve(f)
    }
    input.addEventListener('change', () => done(input.files?.[0] ?? null))
    input.addEventListener('cancel', () => done(null))
    document.body.appendChild(input)
    input.click()
  })
}

async function postJson(url: string, body: unknown): Promise<boolean> {
  try {
    const r = await fetch(url, {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    return r.ok
  } catch {
    return false
  }
}

/**
 * Local change fan-out for appearance edits.
 *
 * The desktop app pushed these from the main process, which owned the packs. Here the
 * gallery and the companion are separate pages against one gateway, and there is no
 * server-push channel for an app's own windows — so a change made in this page
 * notifies this page directly, and another window picks it up on its next read. That
 * covers the case that matters (recolour or switch pack, see the companion update)
 * without inventing a socket.
 */
const listeners = {
  active: new Set<() => void>(),
  colour: new Set<(d: never) => void>(),
  packs: new Set<() => void>(),
  config: new Set<() => void>(),
}

/**
 * Coalesce position writes.
 *
 * A drag ends in a single save, but an edge-snap can produce a second one a frame
 * later; without this the two race and the loser is whichever the gateway happens to
 * write last. Also keeps a fast drag from turning into a burst of requests.
 */
let saveTimer: number | null = null
let pendingSave: { x: number; y: number } | null = null

function flushSave(): void {
  const p = pendingSave
  pendingSave = null
  saveTimer = null
  if (!p) return
  void fetch(CONFIG_PATH, {
    method: 'POST',
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    // Both axes together: the backend refuses half a position, because landing on
    // an axis the user never chose is worse than not saving at all.
    body: JSON.stringify({ petX: Math.round(p.x), petY: Math.round(p.y) }),
  }).catch(() => {
    /* the companion stays where it is on screen; only persistence missed */
  })
}


/** Delete a pack. Standalone so the bridge can expose it under two names. */
async function deletePack(packId: string): Promise<boolean> {
  const ok = await postJson(APPEARANCE_DELETE_PATH, { id: packId })
  if (ok) for (const cb of listeners.packs) cb()
  return ok
}

/**
 * Pick a bundle file and install it.
 *
 * Standalone for the same reason: the desktop app exposed two channels for this and
 * the ported call sites use both names.
 */
async function importBundle(): Promise<GalleryResult> {
  const file = await pickJsonFile()
  // Cancelling is not a failure, so there is no message to show.
  if (!file) return { ok: false, error: '' }
  let bundle: unknown
  try {
    bundle = JSON.parse(await file.text())
  } catch {
    return { ok: false, error: 'That file is not valid JSON' }
  }
  const result = await postForJson<GalleryResult>(APPEARANCE_IMPORT_PATH, { bundle })
  if (result?.ok) for (const cb of listeners.packs) cb()
  return result ?? { ok: false, error: 'Import failed' }
}

export const petBridge: PetBridge = {
  async getWindowPosition() {
    try {
      const r = await fetch(REMINDERS_PATH, { credentials: 'same-origin' })
      if (!r.ok) return null
      // The snapshot is FLAT — config fields sit at the top level, not nested
      // under `config`. Reading the wrong level fails silently as "never moved",
      // which is how the position appeared not to persist at all.
      const d = (await r.json()) as { petX?: number | null; petY?: number | null }
      const x = d.petX
      const y = d.petY
      // Null means never moved. Returning null (rather than 0,0) is what lets the
      // renderer keep its own default placement instead of jumping to the corner.
      if (typeof x !== 'number' || typeof y !== 'number') return null
      return { x, y }
    } catch {
      return null
    }
  },

  savePosition(x, y) {
    pendingSave = { x, y }
    if (saveTimer !== null) return
    saveTimer = window.setTimeout(flushSave, 250)
  },

  /**
   * Cross-display drag hooks.
   *
   * In the desktop app the main process polled the cursor so a drag could carry the
   * pet onto another monitor. Here each display has its own overlay, so a drag is
   * handled entirely within the one the pointer is in — these stay undefined and the
   * hook's optional calls (`api?.dragStart?.()`) simply do nothing, which is the
   * single-display path it already supports.
   *
   * Left named and documented rather than deleted so the gap is visible: without
   * them, dragging the companion to a second monitor stops at the screen edge.
   */
  dragStart: undefined,
  dragEnd: undefined,
  onDragUpdate: undefined,
  onDragEnded: undefined,

  // Walk commands are renderer-driven for now (the wander calls walkPath directly),
  // so the main process emits none of these yet. Left undefined and documented, like
  // the drag hooks above, so useWalking's api?.on*?.() guards resolve to no-ops.
  onWalk: undefined,
  onWalkPath: undefined,
  onWalkCancel: undefined,
  onWalkAppend: undefined,
  onHide: undefined,
  walkDone: undefined,

  async getCrewCompanionConfig() {
    return getJson<Record<string, unknown>>(REMINDERS_PATH)
  },


  async galleryExport(packId: string) {
    return getJson<unknown>(`${APPEARANCE_EXPORT_PATH}?id=${encodeURIComponent(packId)}`)
  },



  /**
   * Pick a bundle file and install it.
   *
   * The file dialog is the browser's own rather than the OS dialog the desktop app
   * opened through Electron: this page is web content, so an <input type=file> is the
   * only way to read a file the user chose, and it keeps the read inside the sandbox.
   */

  /**
   * Save a pack whose art arrived as sprite strips.
   *
   * Mirrors the desktop handler's structure, which three details of my first attempt
   * got wrong:
   *
   *  - Slots are split into `states` and `moods` by whether the key is a known state.
   *    Putting everything in `states` mislabels a mood as a state.
   *  - `randomAssignments` are open-ended CLIPS, stored under a `random-` prefix with
   *    the name sanitised and capped, not merged in beside the states.
   *  - The id is the overwrite target or a fresh UUID. Slugifying the display name
   *    meant importing the same pet twice silently overwrote the first one.
   */
  async gallerySaveSpritePack(data: SpriteImportResult) {
    const bare = (v: string) => (v.includes(',') ? v.slice(v.indexOf(',') + 1) : v)
    const id = data.overwriteId || crypto.randomUUID()

    const files: Record<string, string> = {}
    const states: Record<string, string> = {}
    const moods: Record<string, string> = {}
    const random: Record<string, string> = {}

    const known = new Set<string>([...REQUIRED_STATES, ...OPTIONAL_STATES])
    for (const [slot, strip] of Object.entries(data.assignments ?? {})) {
      if (typeof strip !== 'string' || !strip) continue
      const file = `${slot}.png`
      files[file] = bare(strip)
      if (known.has(slot)) states[slot] = file
      else moods[slot] = file
    }

    for (const [name, strip] of Object.entries(data.randomAssignments ?? {})) {
      if (typeof strip !== 'string' || !strip) continue
      // Sanitised and capped: the name is user data and becomes a filename.
      const safe = name.replace(/[^a-z0-9]+/gi, '_').toLowerCase().slice(0, 24) || 'clip'
      const file = `random-${safe}.png`
      files[file] = bare(strip)
      random[name] = file
    }

    // A whole sheet, kept only for re-editing later — PetDex omits it because its
    // sheet is webp, and the per-slot strips are what actually render.
    const sheet = bare(data.sourceImage ?? '')
    if (sheet) files['source.png'] = sheet

    if (Object.keys(states).length === 0 && Object.keys(moods).length === 0) {
      return { ok: false, error: i18nT('apps.crewCompanion.gallery.noArt') }
    }

    const result = await postForJson<GalleryResult>(APPEARANCE_SAVE_PATH, {
      id,
      manifest: {
        meta: {
          id,
          name: data.name,
          author: data.author,
          description: data.description,
          format: 'sprite',
        },
        states,
        moods,
        random,
        sprite: {
          frameWidth: data.frameWidth,
          frameHeight: data.frameHeight,
          fps: data.fps,
          flipX: data.flipX,
          offsetY: data.offsetY,
          rowAssignments: data.rowAssignments,
          source: sheet ? 'source.png' : undefined,
        },
      },
      files,
    })
    if (result?.ok) for (const cb of listeners.packs) cb()
    return result ?? { ok: false, error: i18nT('apps.crewCompanion.gallery.saveFailed') }
  },

  async petdexFetch(input: string) {
    const r = await postForJson<Record<string, unknown>>(PETDEX_FETCH_PATH, { input })
    return r ?? { ok: false, error: 'PetDex import failed' }
  },

  async updateConfig(patch: Record<string, unknown>) {
    const r = await postForJson<{ ok?: boolean }>(CONFIG_PATH, patch)
    if (r) for (const cb of listeners.config) cb()
    return Boolean(r)
  },

  /**
   * Open a link in the user's real browser.
   *
   * Handed to the main process rather than opened here: `window.open` inside an
   * Electron window opens ANOTHER APP WINDOW, which is not what following a link to
   * petdex.dev means. The main process passes it to the OS, and re-checks the scheme
   * on its side because this page is web content and cannot be trusted with it.
   */
  openExternal(url: string) {
    if (!/^https:\/\//i.test(url)) return
    const bridge = preload()
    if (bridge?.openExternal) {
      bridge.openExternal(url)
      return
    }
    // Opened in an ordinary browser tab rather than its window — a new tab is right.
    window.open(url, '_blank', 'noopener,noreferrer')
  },

  closeGallery() {
    preload()?.galleryClose?.()
  },

  /** Packs changed. Polled, like the other gallery notifications. */
  onGalleryPacksChanged(cb: () => void) {
    listeners.packs.add(cb)
    return () => listeners.packs.delete(cb)
  },

  onConfigUpdated(cb: () => void) {
    listeners.config.add(cb)
    return () => listeners.config.delete(cb)
  },

  /** The gallery's name for the same delete the companion's menu uses. */
  galleryDelete: deletePack,
  /** Both import entry points do the same thing; the desktop app had two channels. */
  galleryImportFile: importBundle,
  galleryImportBundle: importBundle,

  /**
   * Read one file out of an installed pack.
   *
   * The sprite importer re-opens a saved pack to re-slice it, so it needs the ORIGINAL
   * sheet rather than the per-state strips derived from it.
   */
  async galleryReadPackFile(packId: string, filename: string) {
    const detail = await getJson<PackDetail>(detailUrl(packId))
    const entry = detail?.animations?.[filename.replace(/\.[^.]+$/, '')]
    if (typeof entry === 'string') return entry
    return entry?.content ?? null
  },

  /**
   * Ask the user for a sprite sheet.
   *
   * Read as a data URL and split rather than handed over as a path: this page is web
   * content and never sees the filesystem, which is the right boundary for art coming
   * from wherever the user got it.
   */
  async importSpriteFile() {
    const file = await pickImageFile()
    if (!file) return { ok: false, error: '' }
    const dataUrl = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(String(reader.result ?? ''))
      reader.onerror = () => reject(new Error('read failed'))
      reader.readAsDataURL(file)
    }).catch(() => '')
    const content = dataUrl.includes(',') ? dataUrl.slice(dataUrl.indexOf(',') + 1) : ''
    if (!content) return { ok: false, error: 'That image could not be read' }
    return { ok: true, value: { content } }
  },

  async galleryListPacks(): Promise<PackMeta[]> {
    const d = await getJson<{ packs?: PackMeta[] }>(APPEARANCES_PATH)
    return Array.isArray(d?.packs) ? d.packs : []
  },

  async galleryGetPackDetail(packId) {
    return getJson<PackDetail>(detailUrl(packId))
  },

  async gallerySetActive(packId) {
    const ok = await postJson(CONFIG_PATH, { activeAppearance: packId })
    if (ok) {
      for (const cb of listeners.active) cb()
      // The gallery is its own window, so the in-page fan-out above cannot reach the
      // companion overlay. Only the main process spans both.
      preload()?.appearanceChanged?.()
    }
    return ok ? { ok: true, packId } : { ok: false, error: 'Could not switch avatar' }
  },

  async galleryDeletePack(packId) {
    const ok = await postJson(APPEARANCE_DELETE_PATH, { id: packId })
    if (ok) for (const cb of listeners.active) cb()
    return ok
  },

  /**
   * Save an authored pack.
   *
   * The editor thinks in purposes — required states, optional moods, random extras —
   * while storage thinks in files plus a manifest that names them. Flattening happens
   * here so the editor keeps its own vocabulary and the store keeps its contract; a
   * slot is skipped when it has no art rather than written as an empty file, which
   * would list as present and then fail to render.
   */
  async gallerySavePack(data: PackInput) {
    const meta = data.meta ?? {}
    const id = String(meta.id ?? '')
    if (!id) return { ok: false, error: 'That pack needs a name' }

    const files: Record<string, string> = {}
    // `states` maps a slot to its FILENAME; the file contents travel separately. That
    // is the store's contract, and a manifest in any other shape saves but then reads
    // back as unlistable.
    const states: Record<string, string> = {}
    let format = 'svg'
    const groups = [data.states ?? {}, data.moods ?? {}, data.random ?? {}]
    for (const group of groups) {
      for (const [slot, content] of Object.entries(group)) {
        if (typeof content !== 'string' || !content.trim()) continue
        // Lottie is JSON; everything else the editor produces is markup.
        const isJson = content.trimStart().startsWith('{')
        if (isJson) format = 'lottie'
        const file = `${slot}.${isJson ? 'json' : 'svg'}`
        files[file] = content
        states[slot] = file
      }
    }
    if (Object.keys(files).length === 0) {
      return { ok: false, error: 'That pack has no art in it' }
    }

    const result = await postForJson<GalleryResult>(APPEARANCE_SAVE_PATH, {
      id,
      manifest: { meta: { ...meta, id, format }, states },
      files,
    })
    if (result?.ok) {
      for (const cb of listeners.packs) cb()
      for (const cb of listeners.active) cb()
    }
    return result ?? { ok: false, error: 'Save failed' }
  },

  async presetsGetColorMap(packId) {
    const d = await galleryDetailColours(packId)
    return d
  },

  async gallerySetColorMap(packId, colorMap) {
    const ok = await postJson(APPEARANCE_COLOURS_PATH, { id: packId, colorMap })
    if (ok) for (const cb of listeners.colour) (cb as (d: unknown) => void)({ packId, colorMap })
    return ok ? { ok: true, packId } : { ok: false, error: 'Could not save those colours' }
  },

  onGalleryActiveChanged(cb) {
    listeners.active.add(cb)
    // Also listen for the cross-window broadcast, so a switch made in the gallery
    // window repaints the companion instead of waiting for a reload.
    const offBridge = preload()?.onAppearanceChanged?.(cb)
    return () => {
      listeners.active.delete(cb)
      offBridge?.()
    }
  },

  onColorMapChanged(cb) {
    listeners.colour.add(cb as (d: never) => void)
    return () => listeners.colour.delete(cb as (d: never) => void)
  },

  updateHitbox(pet, bubble) {
    preload()?.updateHitbox?.(pet, bubble)
  },

  setMenuHitbox(rect) {
    // Report the menu's rect as an interactive hitbox; the main process's cursor
    // poll makes that region accept input while the rest of the desktop stays
    // click-through. Passing null on close drops it again.
    preload()?.setMenuHitbox?.(rect)
  },

  async remindersList() {
    const d = await getJson<{ reminders?: Reminder[] }>(REMINDERS_PATH)
    return Array.isArray(d?.reminders) ? d.reminders : []
  },

  async remindersRemove(id: string) {
    return postJson(REMOVE_PATH, { id })
  },

  /**
   * Colour presets the user saved.
   *
   * Kept in the companion's own config rather than a new store: they are a handful of
   * small objects and they belong to the same settings the rest of this file reads.
   */
  async presetsLoadCustom() {
    const d = await getJson<{ customPresets?: unknown[] }>(CONFIG_PATH)
    return Array.isArray(d?.customPresets) ? d.customPresets : []
  },

  async presetsSaveCustom(presets: unknown[]) {
    return postJson(CONFIG_PATH, { customPresets: presets })
  },

  contextMenuAction(action) {
    if (action === 'quit') {
      // The desktop app quit its own process. Here the companion IS an app, so the
      // equivalent is disabling it: the overlay goes, the reminders stay, and the
      // user can bring it back from the Apps page. Quitting Kiro Crew itself would
      // close the dashboard too, which is not what "dismiss the companion" means.
      void fetch('/api/apps/crew-companion/disable', {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
      }).catch(() => {})
      return
    }
    if (action === 'gallery') {
      // Its own window, like the panel — the overlay is click-through and cannot host
      // a surface the user needs to type and drag inside.
      preload()?.galleryOpen?.()
      return
    }
    // Anything else is a window-level concern for the main process.
    preload()?.contextMenuAction?.(action)
  },
}

/** A pack's saved colour map, read from its detail payload. */
async function galleryDetailColours(packId: string): Promise<Record<string, string> | null> {
  const d = await getJson<PackDetail>(detailUrl(packId))
  return d?.colorMap ?? null
}

/**
 * The gallery's view of the bridge, with nothing optional.
 *
 * Every method on `PetBridge` is optional because the companion's own pages run
 * against a preload that may not be there — in a plain browser tab there is no
 * Electron. The gallery is different: it only ever renders inside its own window,
 * where the bridge is always present, and its ported call sites invoke these
 * directly. Asserting once here beats an optional-chain on 40 call sites, each of
 * which would silently do nothing if the assumption were ever wrong.
 */
export const galleryApi = petBridge as Required<
  Pick<
    PetBridge,
    | 'galleryListPacks'
    | 'galleryGetPackDetail'
    | 'gallerySetActive'
    | 'galleryDelete'
    | 'galleryDeletePack'
    | 'gallerySavePack'
    | 'gallerySaveSpritePack'
    | 'galleryExport'
    | 'galleryImportBundle'
    | 'galleryImportFile'
    | 'gallerySetColorMap'
    | 'presetsGetColorMap'
    | 'petdexFetch'
    | 'galleryReadPackFile'
    | 'importSpriteFile'
    | 'updateConfig'
    | 'getCrewCompanionConfig'
    | 'openExternal'
    | 'closeGallery'
    | 'onGalleryPacksChanged'
    | 'onGalleryActiveChanged'
    | 'onColorMapChanged'
    | 'onConfigUpdated'
  >
>
