# Codex ACP backend with ChatGPT-subscription OAuth (fork-only)

Task spec + friction tracking for `feat/codex-acp-oauth` on the `so0k/KiroCrew`
fork. This work is a **deliberate divergence from upstream policy** — upstream's
`AGENTS.md` fixes `agent.provider` to `acp`/kiro-cli and forbids re-adding
provider registration glue. This document records why we diverge, where the
seam has caused friction upstream, and the design we implement.

## Goal

Run a Kiro Crew instance whose LLM turns are served by **OpenAI Codex** using an
existing **ChatGPT subscription via OAuth** (`codex login` device/browser flow,
tokens in `$CODEX_HOME/auth.json`). **No API key anywhere** — the crucial
constraint. The backend rides the existing dormant alternate-ACP-backend seam
(`AcpClient(acp_backend=...)`), exactly the shape the preserved
`ACP_BACKEND_CLAUDE` seam was built for.

## Friction log: where the KiroACP-only seam has hurt (upstream evidence)

The single-provider policy is enforced in three places — `agent.provider`
schema `enum=["acp"]` (`config/loader.py`), `AGENTS.md` ("do NOT re-add the
public registration glue"), and
`docs/system-specs/features/claude-code-provider.md` ("Standalone provider —
removed"). Every provider-shaped request upstream has collided with it:

| Ref | What was asked | Outcome / friction |
|---|---|---|
| [#1693](https://github.com/kirodotdev/KiroCrew/issues/1693) | `agent.provider` ∈ {acp, ollama, bedrock, openai_compatible} with `base_url`/`api_key`; per-role provider targeting | **Open, canonical tracking issue.** No maintainer ruling. All later asks get folded into it. |
| [#1872](https://github.com/kirodotdev/KiroCrew/pull/1872) | Full LiteLLM-based multi-provider implementation | **Closed unmerged.** Implementation shelved on author's fork (tag `litellm-provider-impl-shelved`). API-key-shaped — wrong auth model for subscription reuse anyway. |
| [#2107](https://github.com/kirodotdev/KiroCrew/pull/2107) | Docs-only RFC asking maintainers to *rule* on provider pluggability (accept / accept narrowly / reaffirm-and-close) | **Open since 2026-08-07 with clean reviews, unanswered.** The policy question itself is stuck. |
| [#2463](https://github.com/kirodotdev/KiroCrew/issues/2463) | BYOM: "users with existing **ChatGPT**, GLM, Kimi… **subscriptions**" | **Closed NOT_PLANNED within hours** as dup of #1693 — the *subscription-OAuth* framing (our exact use case) was flattened into the API-key adapter scope and lost. |
| [#2573](https://github.com/kirodotdev/KiroCrew/issues/2573) | OpenAI-compatible / Gemini / Groq / Mistral endpoints | Open, `needs-human`; triage: "decide first whether an OpenAI-compatible passthrough is in scope at all". |
| [#2033](https://github.com/kirodotdev/KiroCrew/issues/2033) | Pi coding agent as a selectable harness (RPC/SDK mode) | Open, `needs-human`; "strategic decision", no code. The only other-*harness* (vs other-model) ask. |
| [#1570](https://github.com/kirodotdev/KiroCrew/issues/1570) | Surface active harness auth/session info in the Crew panel | Open. Read-only, blocked on nothing, but shows demand for harness visibility. |

Pattern: **every ask is normalized to an API-key `base_url` adapter and then
stalls on an unanswered policy question.** Nobody upstream has proposed
subscription-OAuth auth (no mention of Codex CLI, `codex-acp`, or
`~/.codex/auth.json` anywhere in the tracker). Meanwhile the codebase itself
keeps a live, maintained alternate-backend seam (`ACP_BACKEND_CLAUDE`,
`_is_claude`, `_resolve_claude_acp_bin`, `AcpProvider.start()`'s legacy-client
branch, and `build_provider_factory`'s documented ProviderRegistry extension
point "e.g. re-registering an extra ACP backend through the dormant
`ACP_BACKEND_*` seam") — the seam exists precisely so a companion edition can
do what the public tracker is told cannot be done.

## Design

Ride the seam; keep the diff minimal and rebase-friendly.

1. **`ACP_BACKEND_CODEX = "codex"`** (`acp/types.py`), next to
   `ACP_BACKEND_CLAUDE`.
2. **Backend classes in `AcpClient`**: `_is_codex`; `_is_kiro`
   (`backend == ""`); `_is_spec_adapter` (`claude or codex` — adapters speaking
   the public ACP spec rather than the kiro dialect). Guard audit:
   - kiro-dialect-only sites (`set_mode`, kiro session-file resume check, jsonl
     seek, entitlement `model_is_unusable`, `is_kiro_cli=` sandbox flag,
     effort/tool-search overlay) key on `_is_kiro`.
   - spec-adapter sites (integer `protocolVersion` 1, model via
     `session/set_config_option`, `mcpServers` passed in `session/new|load`
     params, permission `optionId`/`name` shape — already dual-read) key on
     `_is_spec_adapter`.
   - claude-only sites stay `_is_claude` (`CLAUDE_CODE_EXECUTABLE`,
     `settings.local.json` seed, `_meta.claudeCode`, substitution-advisory
     retry).
3. **Binary resolution** `_resolve_codex_acp_argv()` ladder:
   `CODEX_ACP_BIN` env override → `codex-acp` standalone adapter binary
   (mise → augmented PATH) → `codex` CLI + `acp` subcommand
   (`["codex", "acp"]`). Covers both the Zed-style standalone adapter and
   native `codex acp`, without betting on either distribution. **Live-probe
   TODO**: verify against the installed codex version which form serves ACP and
   which config options (`model`, `effort`) it advertises.
4. **MCP servers**: codex-acp reads no kiro agent config, so
   `_codex_session_mcp_servers()` injects **all three** managed stdio servers —
   kirocrew-core, kirocrew-cron, **and kirocrew-computer** — reshaped to the ACP
   array form, into `session/new` / `session/load` params. Without them the crew
   has no cron, memory, or core tools. The computer shim is injected
   **unconditionally and deliberately**, matching the kiro agent spec: its stdio
   shim returns an EMPTY `tools/list` while the keystone `computer_use.json`
   primary enable is off, so a disabled feature costs the model no context and
   needs no per-server enable check here. It carries no `autoApprove` key (the
   managed-server table forbids one for exactly this server), so its calls stay
   on the in-band `tools._dispatch` refusal path.
   `_session_mcp_servers()` then merges that list with the MCP-gateway broker
   stubs (`_pooled_mcp_servers`), **deduped by name with the broker stub
   winning** — the rewriter wraps every stdio server in the materialized agent
   spec, so all three managed names appear on both sides, and two elements with
   one name is undefined in the ACP schema. On a spec adapter each surviving
   element is reduced to `{name, command, args, env}`: the stub carries kiro-cli
   passthrough keys (`autoApprove`, `timeout`, operator vendor keys) that a
   strict Rust serde deserializer rejects, which would fail the whole
   `session/new`.
5. **Registration glue (the deliberate fork divergence)**:
   `agent.acp_backend` config field (enum `["", "codex"]`, default `""` =
   kiro-cli), threaded `create_provider_factory` → `AcpProvider` →
   `AcpClient`. Kiro-specific model-id normalization (`to_acp_id`) and the
   effort cli.json overlay are skipped for codex. Model default stays `"auto"`
   → for codex, no model call is sent (backend default) unless the user picked
   one AND the adapter advertises a `model` config option.
6. **OAuth**: nothing stored by Kiro Crew. `codex login` owns the ChatGPT OAuth
   flow (browser/device, localhost:1455 callback) and persists
   `$CODEX_HOME/auth.json` (default `~/.codex/auth.json`); the spawned adapter
   inherits it from disk. Kiro Crew adds: (a) auth preflight surfaced through
   `kirocrew doctor`; (b) backend-aware `AcpAuthRequired` message ("Run
   `codex login`") from the shared not-logged-in stderr classifier. Headless
   host options: SSH/SSM port-forward 1455 during login, or copy an existing
   `auth.json` to the host.
7. **Approval-gate boundary (known, surfaced, not enforced)**: Kiro Crew's
   PreToolUse gate (139 denied-command rules, the `~/.aws`/`~/.ssh` path block,
   the governance ceiling) only runs on a `session/request_permission`. kiro-cli
   is made to ask by `--agent <spec>`, claude-agent-acp by the seeded
   `defaultMode: default`; a codex adapter is governed by its OWN
   `approval_policy` in `$CODEX_HOME/config.toml`. Under `never` / `on-failure`
   it auto-approves sandboxed commands and the gate is never consulted. v1
   **reports** this rather than forcing it: a spawn-time warning plus a
   `kirocrew doctor` `approvals:` row (⚠️ + an issue for a bypassing policy).
   Not a refusal, because the value read is the top-level one and a
   `[profiles.*]` selection inside the adapter can override it. Forcing the
   policy on the wire (a `-c approval_policy=...` argv override) needs the live
   probe first — the standalone `codex-acp` form may not accept CLI overrides at
   all, and guessing would break spawn for one of the two distribution forms.
8. **Model picker**: no model-registry entries for GPT ids. `GET /api/models`
   is made backend-aware instead: when `agent.acp_backend` is non-empty it
   returns the live session's advertised models
   (`_advertised_alt_backend_models`, sourced from `_capture_available_models`)
   and never shells out to `kiro-cli chat --list-models`. That spawn is both
   impossible (the binary may be absent) and wrong (kiro-namespace ids, which
   `_wire_model_id`'s codex branch would pass verbatim into `set_model`) on a
   codex host. With no live session yet it 503s `acp_backend_models_unavailable`,
   the existing "degraded, keep polling" contract.
9. **Kiro readiness gate**: `reject_if_kiro_unverified` opens unconditionally on
   a non-kiro backend. Every probe behind that latch asks about kiro-cli, so on a
   codex host it would 503 resume / regenerate / rewind / `/v1/chat/completions`
   forever while ordinary sends (deliberately ungated) kept working.
10. **`$CODEX_HOME/auth.json` is a protected path.** It holds a live ChatGPT
    OAuth access + refresh pair, so `.codex/auth.json` (the leaf, not the
    directory — `config.toml` stays readable) is on
    `security._SENSITIVE_HOME_DIRS`. That one entry arms both the fs_read/write
    gate and the bash matcher (verb-independent catch-all + relative-traversal),
    which is wider than per-verb `DeniedCommandRule` entries. The adapter reads
    the file in-process, never through the gate, and the sandbox path-hiding
    lists are separate, so the spawn is unaffected.
11. **Out of scope (v1)**: session sharing/multiplexed subagents (codex uses the
    per-session `AcpClient` path like claude), kiro credit metering, codex-labelled
    subagent usage telemetry (`_is_cc_provider` folds codex into the `acp` label —
    see its docstring for why that is a two-way question).

## Status

- [x] Seam read (specs + code audit)
- [x] This tracking doc
- [x] types + client dialect branches
- [x] binary resolution ladder
- [x] MCP injection
- [x] config field + factory threading (`agent.acp_backend`, enum `["", "codex"]`,
      captured once in `create_provider_factory` → `AcpProvider` → `AcpClient`)
- [x] auth classification + doctor (spawn preflight and the mid-handshake stderr
      banner both raise `AcpAuthRequired`; `_doctor_codex_backend` reports the
      adapter ladder, `auth.json`, and the approval-gate row)
- [x] approval-gate boundary surfaced (`codex_approval_policy()` +
      `CODEX_PERMISSION_BYPASS_POLICIES`, spawn warning, doctor row, spec text)
- [x] spec docs (providers.md alternate-backend seam + config, acp-client.md
      backend-selection bullet + approval-gate boundary)
- [x] tests (`test/test_acp_backend_codex.py`: resolution ladder, `auth.json`
      location, backend classes, spawn auth preflight, the stderr-banner →
      `AcpAuthRequired` translation, MCP injection incl. the `KIROCREW_HOME` pin
      and arrival in the `session/new` params, the `model` config-option guard on
      both the startup and explicit-pick paths, approval-policy parsing, the
      doctor rows, the dashboard `_wire_model_id` / `_pinned_model_withheld`
      codex branches, config + factory threading, provider/session predicates)
- [~] live probe (kirocrew-vm, Ubuntu 24.04, codex-cli 0.147.0 +
      `@agentclientprotocol/codex-acp` 1.1.14, 2026-08-12):
      - **`codex acp` subcommand does NOT exist** — codex-cli 0.147.0 forwards
        `acp` as a *prompt*, it is not an ACP server. Only the standalone
        `codex-acp` adapter (resolver path 2) serves ACP. **Latent bug**:
        resolver path 3 (`[codex, "acp"]`) would spawn codex in prompt mode on a
        host lacking the standalone adapter — it never fires here (path 2 wins)
        but should be dropped or guarded, and acp-client.md's "or the Codex CLI's
        own `codex acp` subcommand" wording is wrong.
      - `_doctor_codex_backend` validated live: adapter ✅ resolved, missing
        `auth.json` ❌ with the headless remedy, `approval_policy=untrusted` ✅.
      - still TODO once authed: advertised config options (`model`, `effort`),
        `mcpServers` stdio shape acceptance, whether approval policy can be
        forced on the wire.

## Polyfill round (post-audit)

A claude-parity + subsystem audit adjudicated every codex-absent site; the
gaps where Kiro Crew itself owns the interface were closed (each with tests and
same-commit spec updates; validated live on kirocrew-vm over LAN):

- **Session resume/prune** — `session_map` treats `"codex"` as SDK-managed
  (`_SDK_MANAGED_PROVIDERS`), so codex mappings resume instead of being swept.
- **Background one-liners + knowledge pool** — `_bg_provider_is_kiro` and
  `AcpWorker.start()` read `agent.acp_backend`, so neither spawns kiro-cli on
  a codex-only host.
- **Provider labels end-to-end** — `session_map.configured_provider_label()`
  replaces the pinned `cfg.agent.provider` at every `provider_type=`/usage-row
  call site (slack gateway, chat_runner, task_executor, dashboard hooks,
  subagent), so codex sessions reach context.py and telemetry as `"codex"`.
- **Context/steering/skills parity** — context.py's `is_cc` gates widened to
  `is_spec_adapter` (steering docs, mapped `skill://` globs, project group,
  post-compaction re-injection); the persona branding rewrite stays claude-only.
- **Usage analytics same-shape** — `/api/usage/kiro` falls back to
  `_sessions_from_own_records()` (Kiro Crew's own token shards) on a non-kiro
  backend instead of `{"error": "No sessions directory"}`; chat_runner's
  persist gate keeps a completed spec-adapter turn even with zero
  tokens/credits (codex-acp forwards neither per-turn), so the rows exist to
  count. Billing stays `{}` until upstream forwards ChatGPT rate limits.
- **Agent-profile fail-closed guard** — spec adapters never `set_mode`, so a
  custom agent whose kiro config withholds `execute_bash` would silently gain
  the adapter's own shell; `_assert_spec_adapter_agent_permitted()` refuses
  exactly that case (agents Kiro Crew itself authors, `OWNED_KIRO_AGENT_FILES`, exempt —
  their shell-less profiles are Kiro Crew's own scope choice, and refusing
  `kirocrew-lite` bricked the background session on first live deploy).
- **PID lifecycle** — `_MANAGED_AGENT_MARKERS` gains `"codex"` (full-cmdline
  pre-kill re-validation covers all three codex-acp spawn shapes).
- **Verified non-gaps** — `_model_rejected_reason` needs no codex exemption
  (codex ids are never canonical registry keys; pinned by test).
