import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { screen } from '@testing-library/react'

import { renderWithProviders } from './helpers'
import type { NormalizedUsage } from '../providers'

// Mock the API client so the real AcpAdapter reads our canned /api/usage/kiro.
vi.mock('../api/client', () => ({
  api: { kiroUsage: vi.fn() },
}))

// The Usage tab reads its provider through the seam; hand it a stub whose
// fetchUsage resolves whatever the current test staged.
const h = vi.hoisted(() => ({ usage: null as NormalizedUsage | null }))
vi.mock('../providers', () => ({
  useProvider: () => ({
    id: 'acp',
    displayName: 'ACP',
    capabilities: { usageBilling: true },
    fetchUsage: () => Promise.resolve(h.usage),
  }),
}))

import { api } from '../api/client'
import { AcpAdapter } from '../providers/adapters/acp'
import UsageTab from '../pages/overview/UsageTab'

const SESSIONS: NormalizedUsage['sessions'] = {
  total: 3,
  today: { sessions: 1, messages: 4, toolCalls: 2 },
  thisWeek: { sessions: 2, messages: 8, toolCalls: 5 },
  thisMonth: { sessions: 3, messages: 12, toolCalls: 7 },
  avgMsgsPerSession: 4,
  dailyHistory: [],
}

// A raw sessions block shaped as GET /api/usage/kiro serves it, so the real
// adapter's session mapping does not throw while we exercise the billing branch.
const RAW_SESSIONS = {
  total_sessions: 3,
  today: { sessions: 1, messages: 4, tool_calls: 2 },
  this_week: { sessions: 2, messages: 8, tool_calls: 5 },
  this_month: { sessions: 3, messages: 12, tool_calls: 7 },
  avg_msgs_per_session: 4,
  daily_history: [],
}

describe('AcpAdapter.fetchUsage — codex billing mapping', () => {
  beforeEach(() => vi.clearAllMocks())

  it('maps a codex billing block by provider, not by the plan key', async () => {
    ;(api.kiroUsage as any).mockResolvedValue({
      sessions: RAW_SESSIONS,
      billing: {
        provider: 'codex',
        plan_type: 'plus',
        primary: { used_percent: 13.0, window_minutes: 10080, resets_at: 1787011261 },
        secondary: { used_percent: 4.5, window_minutes: 300, resets_at: 1786000000 },
        credits: { has_credits: false, unlimited: false, balance: '0' },
        captured_at: '2026-08-12T10:39:00.000Z',
      },
    })
    const usage = await new AcpAdapter().fetchUsage()
    expect(usage.billing).toEqual({
      provider: 'codex',
      planType: 'plus',
      primary: { usedPercent: 13.0, windowMinutes: 10080, resetsAt: 1787011261 },
      secondary: { usedPercent: 4.5, windowMinutes: 300, resetsAt: 1786000000 },
      credits: { hasCredits: false, unlimited: false, balance: '0' },
      capturedAt: '2026-08-12T10:39:00.000Z',
    })
  })

  it('preserves independent nullability of primary/secondary/credits', async () => {
    ;(api.kiroUsage as any).mockResolvedValue({
      sessions: RAW_SESSIONS,
      billing: {
        provider: 'codex',
        plan_type: '',
        primary: { used_percent: 20, window_minutes: 10080, resets_at: null },
        secondary: null,
        credits: null,
        captured_at: '',
      },
    })
    const usage = await new AcpAdapter().fetchUsage()
    const b = usage.billing
    expect(b && b.provider).toBe('codex')
    if (b && b.provider === 'codex') {
      expect(b.primary).toEqual({ usedPercent: 20, windowMinutes: 10080, resetsAt: null })
      expect(b.secondary).toBeNull()
      expect(b.credits).toBeNull()
      expect(b.planType).toBe('')
      expect(b.capturedAt).toBe('')
    }
  })

  it('maps {} billing to null (no regression from the kiro empty case)', async () => {
    ;(api.kiroUsage as any).mockResolvedValue({ sessions: RAW_SESSIONS, billing: {} })
    const usage = await new AcpAdapter().fetchUsage()
    expect(usage.billing).toBeNull()
  })

  it('still maps the kiro credit-plan shape byte-for-byte', async () => {
    ;(api.kiroUsage as any).mockResolvedValue({
      sessions: RAW_SESSIONS,
      billing: { plan: 'Pro', credits_used: 30, credits_plan: 120, resets: '2026-09-01' },
    })
    const usage = await new AcpAdapter().fetchUsage()
    expect(usage.billing).toEqual({
      plan: 'Pro',
      used: 30,
      limit: 120,
      unit: 'credits',
      resets: '2026-09-01',
      percentUsed: 25,
    })
  })
})

describe('UsageTab — codex billing render', () => {
  afterEach(() => {
    h.usage = null
  })

  it('renders plan, both windows, credits and staleness', async () => {
    // A reset one week out and captured recently, so relative formatting is
    // deterministic enough to assert the presence of the reset/staleness lines.
    const now = Math.floor(Date.now() / 1000)
    h.usage = {
      sessions: SESSIONS,
      billing: {
        provider: 'codex',
        planType: 'plus',
        primary: { usedPercent: 13.0, windowMinutes: 10080, resetsAt: now + 7 * 24 * 3600 },
        secondary: { usedPercent: 4.5, windowMinutes: 300, resetsAt: now + 3600 },
        credits: { hasCredits: true, unlimited: false, balance: '12.50' },
        capturedAt: new Date().toISOString(),
      },
    }
    renderWithProviders(<UsageTab />)

    expect(await screen.findByText('Weekly limit')).toBeInTheDocument()
    expect(screen.getByText('5-hour limit')).toBeInTheDocument()
    expect(screen.getByText('Weekly reset')).toBeInTheDocument()
    // Plan value + credit balance rendered.
    expect(screen.getByText('plus')).toBeInTheDocument()
    expect(screen.getByText('12.50')).toBeInTheDocument()
    // Percent is locale-formatted (once, as the colored badge), never a raw
    // float or "null%".
    expect(screen.getByText('13%')).toBeInTheDocument()
    expect(screen.getByText('4.5%')).toBeInTheDocument()
    // Staleness line from captured_at.
    expect(screen.getByText(/Usage as of/)).toBeInTheDocument()

    const card = screen.getByText('Weekly limit').closest('.card, div')
    expect(document.body.textContent).not.toMatch(/NaN|null%|undefined/)
    expect(card).toBeTruthy()
  })

  it('omits the shorter window and credits when they are null', async () => {
    h.usage = {
      sessions: SESSIONS,
      billing: {
        provider: 'codex',
        planType: 'pro',
        primary: { usedPercent: 55, windowMinutes: 10080, resetsAt: null },
        secondary: null,
        credits: null,
        capturedAt: '',
      },
    }
    renderWithProviders(<UsageTab />)

    expect(await screen.findByText('Weekly limit')).toBeInTheDocument()
    // secondary null -> no 5-hour rows; credits null -> no credits row.
    expect(screen.queryByText('5-hour limit')).not.toBeInTheDocument()
    expect(screen.queryByText('Credits')).not.toBeInTheDocument()
    // resetsAt null -> no reset row, and no NaN/placeholder leaks anywhere.
    expect(screen.queryByText('Weekly reset')).not.toBeInTheDocument()
    expect(document.body.textContent).not.toMatch(/NaN|null%|undefined/)
    // capturedAt '' -> no staleness line.
    expect(screen.queryByText(/Usage as of/)).not.toBeInTheDocument()
  })

  it('shows "Unlimited" instead of a balance when credits are unlimited', async () => {
    h.usage = {
      sessions: SESSIONS,
      billing: {
        provider: 'codex',
        planType: '',
        primary: null,
        secondary: null,
        credits: { hasCredits: false, unlimited: true, balance: '0' },
        capturedAt: '',
      },
    }
    renderWithProviders(<UsageTab />)

    expect(await screen.findByText('Credits')).toBeInTheDocument()
    expect(screen.getByText('Unlimited')).toBeInTheDocument()
    // planType '' -> no Plan row.
    expect(screen.queryByText('Plan')).not.toBeInTheDocument()
  })
})
