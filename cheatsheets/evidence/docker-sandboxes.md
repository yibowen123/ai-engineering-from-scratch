# Evidence: cheatsheets/docker-sandboxes.md

Paths are relative to /private/tmp/aiefs-cheatsheets/manuals/docker-sandboxes-101/. S = sections/, O = capture/out/.

## Front matter
- pin sbx 0.47.0, docker-agent 1.149.0: manual.json lines 9-15

## What runs where (architecture, 8 nodes, 7 edges)
- sbx CLI talks to sandboxd over a socket: S/1-1:13, O/02-daemon-status.txt:3
- sandboxd.sock, daemon.log: O/02-daemon-status.txt:3-4, S/2-4:21-22
- Secret store, macOS Keychain: S/1-1:15, S/3-5:13
- microVM with own kernel and private Docker Engine: S/1-1:11, S/r-8:48 (concept only)
- agent user uid 1000: S/2-2:11
- forward proxy gateway.docker.internal:3128 via HTTPS_PROXY: S/3-4:19,33
- proxy swaps key after the request leaves the VM: S/1-1:15
- workspace virtiofs at same absolute path: S/2-1:11
- allowed hosts by policy rules: S/3-4:33 (both paths enforce policies)

## One request leaves the sandbox (flow, 7)
- 1: S/2-1 create listing lines 19-31 (image pull, created)
- 2: S/3-4:19,33
- 3: S/3-3:84 (deny wins), S/3-3:118 (implicit names no rule)
- 4: S/3-4:47-64 (403 inside TLS, approval body)
- 5: S/3-4:64-66 (forward-bypass after allow)
- 6: S/3-5:74 (swap of placeholder on bound host)
- 7: S/3-4:70,104 (one row per host, proxy_type, rule)

## Sandbox lifecycle (task-table, 7)
- create shell <path> --name: O/04-create.txt:1
- run --name re-attach: O/04-create.txt:34, S/2-1:58
- exec -u root sh -c: O/07-template-save.txt:1
- cp host to SANDBOX:PATH: O/06-cp.txt:1
- ports --publish 18081:8080: O/05-ports.txt:1
- stop keeps state: O/04-stop.txt:1-2
- prune --force: O/04-prune.txt:5

## Secrets, MCP and health (task-table, 7)
- secret set-custom --host --env --value: O/10-set-custom.txt:1
- secret import --dry-run: O/10-import-dry-run.txt:1
- mcp add --url: O/11-mcp-add.txt:1
- mcp load --sandbox: O/11-load.txt:1
- diagnose (13 checks): O/02-diagnose.txt:1, S/3-1:9 (13 checks, not in from; count also in O/02-diagnose.txt "13 passed")
- template save --output tar: O/07-template-save.txt:14
- docker-agent run --sandbox --exec: O/27-sandbox-run.txt:1

## Clone mode round trip (annotated-code, 6 lines)
- create --clone shell --name: O/08-clone-create.txt:1. source ro at /run/sandbox/source: O/08-clone-create.txt:17
- agent commits in clone: S/3-2:39, O/08-clone-commit.txt (via S/3-2)
- git -C remote -v shows sandbox-<name>: O/08-clone-host.txt:1
- git -C fetch sandbox-<name>: O/08-clone-host.txt:25, S/3-2:58-60
- rm --force removes VM, state, scoped secrets: O/08-clone-rm.txt:1, S/2-1:125

## Debug a blocked host (annotated-code, 9 lines)
- policy init balanced once: O/03-policy-init.txt:1
- create shell --name m101-policy, no mount: O/09-create.txt:1
- exec curl, Approval required: O/09-blocked-by-rule.txt:1 (command), Approval line in O/09-blocked.txt:12
- policy log: O/09-policy-log.txt:1,4 (forward, default deny)
- policy allow network --sandbox: O/09-allow.txt:1-2
- policy check network --sandbox --verbose --json: C/run.py:854

## What the agent sees inside (command-table, 7)
- HTTPS_PROXY: O/04-env.txt (proxy lines), S/3-4:19
- ANTHROPIC_API_KEY=proxy-managed: O/10-env-sentinel.txt:2
- M101_RECV_KEY=sbx-cs-<rand>: O/10-env-sentinel.txt:4
- SSH_AUTH_SOCK=/run/ssh-agent.sock, on by default: O/04-env.txt, S/3-5:80
- forward-bypass, no inspection: S/3-4:66,135
- transparent, no injection: S/3-4:135
- deny_kind explicit vs implicit: S/3-3:110,118

## Defaults (8)
- --name <agent>-<workdir>: S/r-1:21
- -m, --memory 50% of host memory, 512 MiB to 32 GiB: S/r-1:21
- --pull always: S/r-1:21
- --skills readonly: S/r-1:21
- secret set --refresh 55m: S/r-1:63
- ssh.agentForwardingEnabled true: S/r-3:35
- sandbox.disk.dockerVolume 10g: S/r-3:33
- kit.allowedSources ["docker.io/"]: S/r-3:17

## Key concepts (6)
- Sandbox: S/r-8:48. Clone mode: S/r-8:20. Preset: S/r-8:42. Policy scope: S/r-8:41. Sentinel: S/r-8:51
- Deny wins: S/3-3:84

## Troubleshooting (6)
- Approval required, no allow rule: O/09-blocked.txt:12, S/3-4:106
- ::1 exit 7, tcp4 default, publish /tcp: O/05-ports-curl.txt:5-8, S/2-2:72,115
- copying between sandboxes: O/06-cp.txt:62-65
- deny conflicts with allow, rm first: O/09-deny-conflict.txt:2, S/3-3:86, O/09-rm-allow.txt:1
- no key from ~/.zshrc, host env not read since v0.35.0: S/3-5:5,91
- ensure daemon error, daemon status, daemon.log: S/2-4:5,100

## Q&A (4)
- no Docker Desktop, sbx login, docker ps: S/1-1:11,13,126
- stop vs rm: S/2-1:110,125
- host allow passes any request, gist: S/3-4:112
- agent sees proxy-managed, proxy adds key after VM: S/1-1:15

## Tips (5)
- set at creation, re-attach ignores: S/2-1:94,182
- prune --dry-run first: S/2-1:183
- narrow rules with --sandbox: S/3-3:130
- secret set, never -e KEY: S/3-5:91
- bind custom secret to exact hosts: S/3-5:93
