import { describe, expect, it } from 'vitest'

import { assembleSkillContent, parseFrontmatter, parseSkillContent } from '../components/SkillForm'

/**
 * The structured skill editor rebuilds the frontmatter block from its own
 * fields, so any key it does not model is destroyed on save. Two runtime keys
 * live in that blind spot: `repo_scope` (the matcher's repo guard) and
 * `inject_on_trigger` (the full-body opt-out). These pin that a round-trip
 * carries them through.
 */
describe('structured editor preserves unmodelled frontmatter', () => {
  const RAW = [
    '---',
    'name: worktree-dev',
    'description: Develop in a worktree',
    'triggers: worktree, build gate',
    'repo_scope: src/kiro_crew',
    'inject_on_trigger: false',
    '---',
    '',
    '# Body',
    'Steps here.',
  ].join('\n')

  it('round-trips repo_scope, which gates where the skill may match', () => {
    const out = assembleSkillContent(parseSkillContent(RAW, 'kirocrew-dev/worktree-dev'))
    expect(parseFrontmatter(out).meta.repo_scope).toBe('src/kiro_crew')
  })

  it('round-trips inject_on_trigger, so editing a description cannot re-enable injection', () => {
    const out = assembleSkillContent(parseSkillContent(RAW, 'kirocrew-dev/worktree-dev'))
    expect(parseFrontmatter(out).meta.inject_on_trigger).toBe('false')
  })

  it('still writes the keys the form owns', () => {
    const data = parseSkillContent(RAW, 'kirocrew-dev/worktree-dev')
    const meta = parseFrontmatter(assembleSkillContent({ ...data, description: 'Changed' })).meta
    expect(meta.name).toBe('worktree-dev')
    expect(meta.description).toBe('Changed')
    expect(meta.triggers).toBe('worktree, build gate')
  })

  it('does not duplicate a managed key that also appears in extra', () => {
    const data = parseSkillContent(RAW, 'kirocrew-dev/worktree-dev')
    const out = assembleSkillContent({ ...data, extra: { ...data.extra, name: 'smuggled' } })
    expect(out.match(/^name:/gm)).toHaveLength(1)
    expect(parseFrontmatter(out).meta.name).toBe('worktree-dev')
  })

  it('carries a multi-line unmodelled value without corrupting the block', () => {
    const data = parseSkillContent(RAW, 'k/w')
    const out = assembleSkillContent({ ...data, extra: { note: 'line one\nline two' } })
    expect(parseFrontmatter(out).meta.note).toBe('line one\nline two')
    expect(parseFrontmatter(out).body).toContain('# Body')
  })

  it('leaves a skill with no frontmatter alone', () => {
    const data = parseSkillContent('# Just a body\n', 'plain')
    expect(data.extra ?? {}).toEqual({})
    expect(assembleSkillContent(data)).toContain('# Just a body')
  })
})
