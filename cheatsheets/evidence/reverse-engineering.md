# Evidence: reverse-engineering

M = manuals/reverse-engineering-101, S = M/sections, O = M/capture/out, P13 = phases/13-tools-and-protocols

## Front matter
- subtitle pin rea-agents 6.5.0, Ghidra 12.1.4: M/manual.json:9-12

## rea CLI tasks (8)
- rea doctor --provider ghidra --format json: S/r-1-commands-and-tools.md:11; O/00-commands.txt:1
- rea capabilities --format json: S/r-1:12; O/00-commands.txt:4
- rea inspect-artifact <path>, no Java: S/r-1:13; S/1-1:65
- rea inspect-macho <path>: S/r-1:15
- rea analyze <path> --provider ghidra: S/r-1:16
- rea function <path> <name> --provider ghidra: S/r-1:17
- rea decompile <path> <name> --provider ghidra: S/r-1:18
- rea address-to-file-offset <path> <address> --provider ghidra: S/r-1:19

## Evidence and MCP tasks (8)
- inspect-evidence-bundle '{"path":..., "detail": "summary"}': S/r-1:20
- evidence-export <bundle> <out>: S/r-1:21
- evidence-import <bundle>: S/r-1:22
- compare <left> <right>: S/r-1:23
- npx -y rea-agents@6.5.0 mcp: S/8-1:23
- rea setup, rea doctor --client <id>: S/8-1:29
- python3 run.py --check: M/capture/README.md:7

## One investigation from the shell (14 lines)
- GHIDRA_INSTALL_DIR, JAVA_HOME absolute paths: S/r-1:57-58; JDK 21: S/2-2 (header and line 88)
- doctor, find first ok: false: S/2-2:135
- analyze arm64 binary_overview: O/00-commands.txt:16; S/r-1:16
- function _percent_change, analyze_function: O/00-commands.txt:22; S/r-1:17
- decompile x86_64 grade, procedure_pseudo_code: O/00-commands.txt:40; S/r-1:18
- address-to-file-offset 0x100000518, offset 1304: O/00-commands.txt:43; S/3-3:35,51
- compare O1 O2, exit 0, status unknown: O/00-commands.txt:58-59; S/r-3:91

## How rea is built (7 nodes)
- rea CLI 153 commands, rea mcp 207 tools: S/1-1:19
- BinarySession one active target, shared by CLI and MCP: S/1-1:19
- Evidence record 16 fields, ev_ id: S/1-1:11-13
- Provider registry selects ghidra: S/1-1:17; S/1-2:34
- Ghidra client private copy, bridge.sock: S/1-2:36
- ReaGhidraBridge.java post-script, JSON line per request: S/1-1:57
- stdio JSON-RPC edge: S/8-1:33

## One rea function call (7)
1. parse header, architecture, hash: S/1-2:33
2. selection_source "request": S/1-2:34
3. analysis profile hashed: S/1-2:35
4. copy and java -readOnly -deleteProject: S/1-2:36
5. ping checks run_id, token, digests, analysis_complete: S/1-2:37
6. one JSON line, one Evidence record: S/1-2:38-39
7. shutdown 10 s, project deleted: S/1-2:40

## MCP tools (8)
- open_binary binds provider, starts no process: S/r-1:37; S/8-1 ("open_binary took 39 ms and started no process")
- binary_session tool availability: S/r-1:38
- list_procedures: S/r-1:39. procedure_assembly: S/r-1:40. xrefs: S/r-1:41. analyze_function: S/r-1:43
- export_evidence_bundle before close_binary: S/r-1:44; S/7-1:180
- verify_reconstruction pass/fail/unknown: S/r-1:47

## Native tools from the lessons (8)
- llvm-objdump -d -r linear sweep with relocations: P13/37/docs/en.md:71; P13/39/docs/en.md:35
- llvm-objdump -t value and size: P13/32/docs/en.md:45
- llvm-objdump -h section sizes: P13/38/docs/en.md:146
- nm -m library per symbol: P13/40/docs/en.md:187
- wasm-objdump -d: P13/43/docs/en.md:15
- clang -target x86_64-unknown-linux-gnu -c: P13/34/docs/en.md:460
- break bump / breakpoint set -n bump: P13/42/docs/en.md:289 (gdb and lldb table)
- python3 -m unittest discover -s code/tests -v: P13/36/docs/en.md:403

## Evidence fields (8)
- evidence_id, subject, provider, analysis_profile: S/r-2:13-16
- confidence, authority: S/r-2:22-23,36-37; S/5-1:91
- limitations: S/r-2:25. relationship_coverage: S/r-2:26

## Defaults (8)
- REA_GHIDRA_STARTUP_TIMEOUT_MS 330,000 ms: S/r-1:59; S/8-1 startup deadline table
- client tool call 60,000 ms, SDK 2.3.1: S/8-1 deadline table
- bridge shutdown 10 s: S/1-2:40
- TOON without --format: S/1-2:21
- confidence observed, authority shipped-artifact when the producer names none: S/5-1:87
- relationship_coverage null, out of the id hash: S/r-2:5,49
- answer cache 10,000 for one target: S/r-4:10

## Troubleshooting (8)
- capability_unavailable, Linux x64 only, inspect-macho: S/r-3:9-30; M/capture/README.md:115
- invalid_request unknown procedure, fix name: S/r-3:56; S/4-6:266
- client timeout, Java keeps running: S/8-1:5 and takeaways
- provider_timeout first call: S/r-3:106
- target_unavailable "No app is open.": S/r-3:106
- compare unknown, exit 0, not an error: S/r-3:91
- healthy vs environment_healthy: S/2-2:121,137
- masked compare passes wrong constant: S/7-2:176,250

## Glossary (8)
- Dossier: S/r-4:23. Provider: S/r-4:38. Bridge: S/r-4:16. Thunk: S/r-4:50. Jump table: S/r-4:35
- Relocation masking: S/r-4:41. Residual unknown: S/r-4:42. File offset: S/r-4:28

## Q&A (5)
- rea adds adapters, session, Evidence. Ghidra does import, analysis, decompilation: S/1-1:3,49
- 7 to 9 s per CLI call, under 100 ms later: S/1-1:59
- observed means engine output recorded: S/5-1:180
- id leaves out subject.name and subject.local_path: S/r-2:47
- offset = address - vmaddr + fileoff, 0x100000518 to 1304: S/3-3:51,57

## Tips (6)
- one exact version: S/8-1:29, takeaways
- read confidence, authority, limitations first: S/5-1:179
- export before close_binary: S/7-1:180; S/8-2:204
- passed structural claim is same Ghidra text: S/7-1:179
- quote target strings with address: S/8-2:203
- original flags, mask relocations: S/7-2:249
