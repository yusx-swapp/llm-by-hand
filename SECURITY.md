# Security and local execution

LLM by Hand is a **local, single-user learning application**. It runs submitted Python with the execution process's permissions. The source code, subprocess boundary, AST checks and timeouts are **not a security sandbox**.

## Supported use

- Run on your own machine and execute code you trust.
- Keep the service bound to loopback. The default server rejects untrusted Host headers and cross-origin writes.
- Keep progress databases, exported implementations, checkpoints and environment files private. They are not repository assets and are ignored by Git.
- Treat a downloaded implementation or fork as code that needs review before execution.

## Not supported as-is

Do not expose this runner as a public multi-user service. Authentication alone does not make arbitrary Python execution safe. Public deployment would require separate per-user progress, isolated execution environments, CPU/memory/process/output/time limits, restricted filesystem/network access, and separation from application/cloud credentials.

The application does not ship a hosted execution service or guarantee anti-cheating isolation. The curriculum is intended for learning, not adversarial assessment.

## Reporting concerns

Do not put credentials, private student code or exploitable service details into public issues. If the repository has GitHub private vulnerability reporting enabled, use that channel. Otherwise ask the maintainer for a private reporting route before sharing sensitive details.
