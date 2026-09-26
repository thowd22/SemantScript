export const meta = {
  name: 'semantscript-task-worktree',
  description: 'Implement one SemantScript Backlog task end to end: plan, implement, adversarial review loop, finalize and push',
  phases: [
    { title: 'Understand', detail: 'read the task and the code it touches, produce a plan', model: 'opus' },
    { title: 'Implement', detail: 'code, tests, docs, Backlog plan and notes', model: 'opus' },
    { title: 'Verify', detail: 'three reviewers try to refute completion', model: 'opus' },
    { title: 'Fix', detail: 'address blocking findings', model: 'opus' },
    { title: 'Finalize', detail: 'acceptance criteria with evidence, commit, push', model: 'opus' },
  ],
}

const task = args.task
const extra = args.extra || ''
const root = args.root
const branch = args.branch

const CONTEXT = `
Repository worktree for this task: ${root} on branch ${branch} (a git worktree of /home/admin2/SemantScript; remote origin = git@github.com:thowd22/SemantScript.git; gh CLI is authenticated as thowd22). Work ONLY inside ${root}: cd there first in every shell command, never modify /home/admin2/SemantScript or any other worktree, and run backlog, npm, python and git commands from ${root}. Other tasks are being implemented in other worktrees at the same time, so keep your changes to what this task needs. node_modules, dist and .python-packages (a symlink) are already set up in the worktree; .env is copied there.
You are working on Backlog task ${task}. Read it first with: backlog task ${task} --plain
Project rules (AGENTS.md): every task lifecycle action goes through the backlog CLI; run "backlog instructions task-execution" before planning or changing status, "backlog instructions task-finalization" before checking acceptance criteria or moving to Done; never edit backlog/*.md files directly; never create new Backlog tasks (report follow-ups in your result instead); check an acceptance criterion only with verification evidence (a command you ran and its output).
Environment: Node 22 (npm workspaces: compiler, runtime, cli, framework, benchmarks/refund; build with "npm run build"; lint with "npm run lint:node"; tests with "npm test -w <workspace>" or "npm run test:node"). Python 3.12 with deps in .python-packages: run Python as "PYTHONNOUSERSITE=1 HSA_ENABLE_DXG_DETECTION=1 PYTHONPATH=.:trainer/src:model/src:.python-packages python3 ..."; ruff as "PYTHONNOUSERSITE=1 PYTHONPATH=.python-packages python3 -m ruff check/format"; NEVER run pip with --upgrade --target (it deletes .python-packages/bin); prettier via "npx prettier --write <files>". "tsc -b" may not rebuild a deleted dist file (use --force). GPU: AMD RX 9070 XT via ROCm (needs the two env vars above); Ollama on 127.0.0.1:11434 has qwen2.5 1.5B/7B and qwen3:14b.
Secrets: /home/admin2/SemantScript/.env holds OPENROUTER_API_KEY (git-ignored). Never print, log, commit or paste it. Do not spend more than USD 0.50 of it in this task without the user's approval; if a criterion needs a paid run, do everything else, run the smallest probe that proves the mechanism, and report the estimated cost of the full run instead of running it. ANTHROPIC_API_KEY is unset. Held-out benchmark inputs never enter training; judge labels are never recorded as human-authored.
Existing docs to keep consistent: README.md, docs/ (index.md links every page; cli-reference.md, diagnostics.md, teachers.md, build-cache.md, getting-started.md, tutorial-refund-decision.md, components.md, architecture.md), package READMEs. Prettier must pass on any markdown you touch (npx prettier --check docs README.md).
Git: commit only when your stage says so, on branch ${branch}; end every commit message with the attribution trailer lines your session's attribution reminder gives you (Co-Authored-By and Claude-Session). After every commit, push the branch with: git push -u origin ${branch}. NEVER push to main, never merge, never rebase, never switch branches; the orchestrator merges the branch into main afterwards. CI runs on every push (.github/workflows/ci.yml); when you push, watch it with gh run watch and treat a red run as a blocking problem.
Your final text is data returned to an orchestrator, not a message to a person.
${extra}
`

const PLAN_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    files: { type: 'array', items: { type: 'string' } },
    steps: { type: 'array', items: { type: 'string' } },
    risks: { type: 'array', items: { type: 'string' } },
    needs_user: { type: 'array', items: { type: 'string' }, description: 'things only the user can provide (credentials, spend approval, decisions); empty if none' },
  },
  required: ['summary', 'files', 'steps', 'risks', 'needs_user'],
}

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    changed_files: { type: 'array', items: { type: 'string' } },
    checks_run: { type: 'array', items: { type: 'string' }, description: 'each command and its outcome' },
    open_issues: { type: 'array', items: { type: 'string' } },
    blocked_on_user: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'changed_files', 'checks_run', 'open_issues', 'blocked_on_user'],
}

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    lens: { type: 'string' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          blocking: { type: 'boolean' },
          file: { type: 'string' },
          summary: { type: 'string' },
          evidence: { type: 'string', description: 'the command you ran or the code you read that proves it' },
        },
        required: ['blocking', 'file', 'summary', 'evidence'],
      },
    },
    verdict: { type: 'string', enum: ['complete', 'incomplete'] },
  },
  required: ['lens', 'findings', 'verdict'],
}

const FINAL_SCHEMA = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['done', 'blocked'] },
    commit: { type: 'string' },
    pushed: { type: 'boolean' },
    acceptance: { type: 'array', items: { type: 'string' }, description: 'each criterion with checked/unchecked and its evidence' },
    follow_ups: { type: 'array', items: { type: 'string' } },
    blocked_on_user: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['status', 'commit', 'pushed', 'acceptance', 'follow_ups', 'blocked_on_user', 'summary'],
}

phase('Understand')
log(`${task}: planning`)
const plan = await agent(`${CONTEXT}
Stage: UNDERSTAND. Do not modify any file. Read the task, its acceptance criteria and references, then read the code, tests and docs it touches (grep and read; do not rely on assumptions). Look at how neighbouring features are built and tested so the plan follows the codebase's conventions (typed errors, closed contracts, tests beside features, docs updated with code). Produce a concrete plan: files to create or change, steps in order, risks, and anything only the user can provide. Mark the task In Progress and record the plan with the backlog CLI as the execution guide says (backlog task edit ${task} -s "In Progress" -a @claude --plan "...").`, { model: 'opus', schema: PLAN_SCHEMA, label: `plan ${task}` })

if (plan.needs_user.length) log(`${task}: needs user: ${plan.needs_user.join('; ')}`)

phase('Implement')
log(`${task}: implementing`)
let impl = await agent(`${CONTEXT}
Stage: IMPLEMENT. Plan from the previous stage:
${JSON.stringify(plan, null, 1)}
Implement the whole task: code, tests that exercise the new behaviour, and documentation (the docs index, the CLI reference, the diagnostics catalogue or the relevant guide, whichever the change touches). Follow the codebase's conventions you saw. Run the relevant lint and test suites and fix what they surface; run prettier on markdown you touch. Record progress with backlog task edit ${task} --append-notes "...". Do not check acceptance criteria and do not mark the task Done; do not commit unless the task requires pushing to observe CI (then commit with the trailers and a "${task}: ..." message, push the branch, watch the run with gh run watch, and iterate). Report exactly what you changed, every check you ran with its result, what is still open, and what only the user can resolve.`, { model: 'opus', schema: IMPL_SCHEMA, label: `implement ${task}` })

const LENSES = [
  { name: 'correctness', focus: 'Run the code paths and tests yourself. Look for behaviour that contradicts the acceptance criteria, unhandled failure modes, contracts the change breaks elsewhere (search for callers), and tests that do not actually exercise the new behaviour. Run "npm run lint:node", the affected npm test workspaces, ruff and the affected pytest directories yourself and report failures as blocking.' },
  { name: 'acceptance', focus: 'Take each acceptance criterion of the task literally and try to show it is NOT met: run the command or reproduce the scenario it describes and record what happened. A criterion that cannot be demonstrated with a command or a reproducible observation is not met. Also confirm the Backlog task has a plan and notes recorded through the CLI.' },
  { name: 'developer-experience', focus: 'Judge it as the developer this task is for. Are the error messages, command output, defaults and docs what a new user needs? Are the docs (index, CLI reference, diagnostics catalogue, guides, package READMEs) updated and consistent with the code, and does prettier pass on them? Is anything new undocumented or any old doc now wrong?' },
]

let round = 0
let blocking = []
while (round < 3) {
  phase('Verify')
  log(`${task}: verify round ${round + 1}`)
  const reviews = (await parallel(LENSES.map((lens) => () => agent(`${CONTEXT}
Stage: VERIFY (${lens.name} lens). Do not modify any tracked file (you may run builds and tests). The implementer reports:
${JSON.stringify(impl, null, 1)}
Your job is to refute the claim that ${task} is complete. ${lens.focus}
Be skeptical: prefer finding a real problem to approving. Mark a finding blocking only when it must be fixed before the task can be Done; non-blocking findings are advice. Every finding needs evidence you produced yourself.`, { model: 'opus', effort: 'high', schema: REVIEW_SCHEMA, phase: 'Verify', label: `review ${lens.name} ${task}` })))).filter(Boolean)
  blocking = reviews.flatMap((r) => r.findings.filter((f) => f.blocking).map((f) => ({ lens: r.lens, ...f })))
  const advice = reviews.flatMap((r) => r.findings.filter((f) => !f.blocking).map((f) => ({ lens: r.lens, ...f })))
  log(`${task}: ${blocking.length} blocking, ${advice.length} advisory findings`)
  if (!blocking.length) break
  phase('Fix')
  round += 1
  impl = await agent(`${CONTEXT}
Stage: FIX (round ${round}). Reviewers found these blocking problems with your implementation of ${task}:
${JSON.stringify(blocking, null, 1)}
Advisory findings you may also address:
${JSON.stringify(advice, null, 1)}
Previous implementation report:
${JSON.stringify(impl, null, 1)}
Fix every blocking finding (verify each reviewer claim yourself first; if one is wrong, say so with evidence instead of changing code). Re-run the lint and test suites. Append notes to the Backlog task. Report as before.`, { model: 'opus', schema: IMPL_SCHEMA, label: `fix ${task} r${round}` })
}

phase('Finalize')
log(`${task}: finalizing`)
const final = await agent(`${CONTEXT}
Stage: FINALIZE. Implementation report:
${JSON.stringify(impl, null, 1)}
Blocking findings still open after ${round} fix round(s): ${JSON.stringify(blocking)}
Read "backlog instructions task-finalization" and follow it: run the objective verification for each acceptance criterion yourself (the actual commands), check only the criteria the evidence proves (backlog task edit ${task} --check-ac N), append notes naming the validation, write the final summary, and set the status to Done only if every criterion is checked and the working tree passes lint and the relevant tests. If a criterion is genuinely blocked on something only the user can provide, leave it unchecked, leave the task In Progress, and say exactly what is needed. Then run "git status", stage everything the task changed (never .env or anything under .semantscript/artifact), commit with the message "${task} done: <one line>" (or "${task}: <one line>" when not Done) followed by a short body and the attribution trailer lines, push the branch (git push -u origin ${branch}), and wait for the CI run to pass (gh run watch). Report the commit hash, whether the push succeeded, each criterion's state with its evidence, and any follow-ups you would propose (do not create tasks).`, { model: 'opus', schema: FINAL_SCHEMA, label: `finalize ${task}` })

return { task, plan: plan.summary, needs_user: plan.needs_user, rounds: round, final }