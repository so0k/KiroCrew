/**
 * The pending-fire cursor must survive a reload, or history replays.
 *
 * Reading `/pending` is non-destructive by design, so the backend keeps every fire
 * and answers "everything after `since`". That is what lets a lost response or a
 * second display's overlay still see a reminder. It also means the queue outlives
 * the page — and starting from 0 on every load re-delivered the whole history, so
 * each restart of the desktop shell put an already-seen, already-fired reminder back
 * on screen. From the user's side it read as a bubble that could not be closed.
 *
 * The desktop app this was ported from never hit this: its queue lived in the
 * Electron main process and died with the app.
 */
import { describe, it, expect, beforeEach } from 'vitest'

const CURSOR_KEY = 'cc:pendingCursor'

/**
 * The two helpers as pet.tsx defines them. Duplicated rather than exported: the
 * module pulls in the whole overlay (bridge, rAF loops, CSS) on import, and the
 * behaviour under test is this pair of rules, not the wiring.
 */
function readStoredCursor(): number {
  try {
    const n = Number(window.localStorage.getItem(CURSOR_KEY))
    return Number.isFinite(n) && n > 0 ? n : 0
  } catch {
    return 0
  }
}

function writeStoredCursor(n: number): void {
  try {
    window.localStorage.setItem(CURSOR_KEY, String(n))
  } catch {
    /* ignored */
  }
}

beforeEach(() => {
  window.localStorage.clear()
})

describe('pending cursor persistence', () => {
  it('starts at 0 the very first time, so nothing is missed', () => {
    expect(readStoredCursor()).toBe(0)
  })

  it('resumes where the last run left off instead of replaying', () => {
    writeStoredCursor(7)
    expect(readStoredCursor()).toBe(7)
  })

  it('a fire newer than the cursor is still delivered late', () => {
    // The property that forbids the tempting "just start at the current cursor"
    // shortcut: a reminder that fired while the companion was off must arrive on
    // the user's return, which only works if the cursor is BEHIND that fire.
    writeStoredCursor(7)
    const fires = [{ seq: 7 }, { seq: 8 }].filter((f) => f.seq > readStoredCursor())
    expect(fires.map((f) => f.seq)).toEqual([8])
  })

  it('falls back to 0 on a corrupt value rather than muting everything', () => {
    // Failing towards "replay once" is recoverable; failing towards a huge cursor
    // would silently swallow every future reminder.
    window.localStorage.setItem(CURSOR_KEY, 'not-a-number')
    expect(readStoredCursor()).toBe(0)
    window.localStorage.setItem(CURSOR_KEY, '-5')
    expect(readStoredCursor()).toBe(0)
  })

  it('survives a simulated restart: the same fire is not shown twice', () => {
    const history = [{ seq: 1 }, { seq: 2 }]
    // First run drains everything and records the cursor.
    const firstRun = history.filter((f) => f.seq > readStoredCursor())
    expect(firstRun).toHaveLength(2)
    writeStoredCursor(2)
    // Restart: same backend history, nothing new to show.
    const secondRun = history.filter((f) => f.seq > readStoredCursor())
    expect(secondRun).toHaveLength(0)
  })
})
