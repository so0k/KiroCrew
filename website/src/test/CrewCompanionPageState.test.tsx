/**
 * The app page's two states, as the mainline page defined them.
 *
 * The page decides from its own READS, not from a separate presence probe: when the
 * companion's endpoints answer, it shows the live sections; when they do not, it shows
 * one card with a title, a line of copy and a single action to bring the companion back.
 *
 * A previous revision replaced this with a three-state model plus an extra "Companion"
 * card carrying its own Open / Change avatar buttons. That duplicated the card this page
 * already had, so it was reverted — these tests pin the original shape so it does not
 * drift again. The one thing the card must never lose is its button: an earlier version
 * kept the copy and dropped the action, leaving the user told that the companion was
 * away with no way to bring it back.
 */
import { describe, it, expect, vi } from 'vitest'
import { render, waitFor } from '@testing-library/react'

/** Per-test switch: does the companion's backend answer, or not? */
const state = vi.hoisted(() => ({ reachable: true }))

vi.mock('../apps/crew-companion/api', () => ({
  apiGet: vi.fn(async (path: string) => {
    if (!state.reachable) throw new Error('offline')
    if (path.includes('/reminders')) {
      return {
        reminders: [],
        breakNudgesEnabled: true,
        breakReminderMins: 45,
        sessionNotificationsEnabled: true,
      }
    }
    if (path.includes('/stats')) {
      return { stats: { companionSeconds: 0, breathingSessions: 0, remindersFired: 0 } }
    }
    return {}
  }),
  apiPost: vi.fn(async () => ({})),
}))

// Imported after the mock so it binds to the mocked api module.
const { default: CrewCompanionPage } = await import(
  '../apps/crew-companion/CrewCompanionPage'
)

describe('CrewCompanionPage: reachable vs not', () => {
  it('shows the live sections when the companion answers', async () => {
    state.reachable = true
    const { container } = render(<CrewCompanionPage />)
    await waitFor(() => expect(container.querySelector('.cc-quit-tip')).not.toBeNull())
    // The away card is not showing.
    expect(container.querySelector('.cc-offline')).toBeNull()
  })

  it('shows the away card when the companion does not answer', async () => {
    state.reachable = false
    const { container } = render(<CrewCompanionPage />)
    await waitFor(() => expect(container.querySelector('.cc-offline')).not.toBeNull())
    expect(container.querySelector('.cc-quit-tip')).toBeNull()
  })

  it('keeps an action on the away card', async () => {
    state.reachable = false
    const { container } = render(<CrewCompanionPage />)
    await waitFor(() => expect(container.querySelector('.cc-offline')).not.toBeNull())
    // The regression this pins: copy without a way to act on it.
    const cta = container.querySelector('.cc-offline .cc-cta')
    expect(cta).not.toBeNull()
    expect((cta as HTMLElement).textContent?.trim().length).toBeGreaterThan(0)
  })

  it('does not add a second card duplicating that action', async () => {
    // The reverted revision put Open panel / Change avatar in their own card while the
    // away card already existed. One entry point, not two.
    state.reachable = true
    const { container } = render(<CrewCompanionPage />)
    await waitFor(() => expect(container.querySelector('.cc-quit-tip')).not.toBeNull())
    const buttons = Array.from(container.querySelectorAll('button')).map((b) =>
      (b.textContent ?? '').toLowerCase(),
    )
    expect(buttons.some((t) => t.includes('change avatar'))).toBe(false)
  })
})
