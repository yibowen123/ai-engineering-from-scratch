# Evidence: cheatsheets/sglang-serving.md (after trim pass)

Paths are relative to /private/tmp/aiefs-cheatsheets/manuals/sglang-101/. S = sections/, C = capture/, O = capture/out/.

## Front matter
- subtitle pin 0.5.22: manual.json lines 9-15

## Tasks (task-table, 8)
- sglang serve --model-path --port 30000: O/server-default.command.txt:1
- POST /generate {"text": ...}: C/one_request.py:13
- POST /v1/chat/completions: C/one_request.py:17
- GET /ready readiness: S/7-1:49,57
- GET /metrics: C/run.py:94, S/5-2:11
- fixed concurrency cap --max-concurrency 8: C/run.py:34, S/5-1:29
- fixed request rate --request-rate (default inf): S/5-1:42
- docker stop sends SIGTERM to main process: S/7-1:118

## Start the server (terminal 1) (annotated-code, 5 lines)
- sglang serve --model-path --port 30000 --enable-cache-report --enable-metrics --mem-fraction-static 0.4: O/server-default.command.txt:1
- cache report and metrics meaning: S/r-1:35,57. fraction meaning: S/3-2:55

## Benchmark it (terminal 2) (annotated-code, 5 lines)
- benchmark --backend sglang --port 30000 --dataset-name random --random-input-len 512 --random-output-len 128 --random-range-ratio 1.0: C/run.py:23,26, S/5-1:27-29
- default range ratio 0.0 halves prompts: S/5-1:40,45
- --num-prompts 32 --max-concurrency 8: C/run.py:34. 255.63 tok/s: O/bench-random-c8.txt:23

## Key flags (command-table, 8)
- --mem-fraction-static: S/3-2:55
- --max-total-tokens: S/r-1:12
- --max-running-requests at the knee: S/r-1:13, S/5-1:131
- --chunked-prefill-size: S/r-1:14, meaning S/2-1:75
- --schedule-policy lpm: S/2-2:62-71
- --schedule-conservativeness 1.3: S/2-3:114
- --disable-radix-cache: S/r-1:31
- --tp-size: S/6-1:13

## Three processes (architecture, 6 nodes, 6 edges)
- TokenizerManager main process: S/2-1:19. Scheduler owns GPU: S/2-1:21. DetokenizerManager: S/2-1:23
- ZMQ links: S/2-1:25. port 30000: S/2-1:46
- UnifiedRadixCache, match_prefix: S/3-3:60,77
- KV pool one slot per token: S/3-1:59

## One request, end to end (flow, 7)
- 1: S/2-1:19,25. 2: S/2-2:21,62. 3: S/3-3:60, S/2-2:53. 4: S/2-3:19
- 5: S/2-2:22,26. 6: S/2-2:22,24. 7: S/2-1:23

## Defaults in the code (8, all S/r-1)
- mem-fraction-static None: r-1:11. max-running-requests None: r-1:13. chunked-prefill-size None: r-1:14
- schedule-conservativeness 1.0: r-1:16. schedule-policy fcfs: r-1:23. enable-cache-report False: r-1:35
- enable-metrics None, resolves to off: r-1:57. watchdog-timeout 300 s: r-1:58

## Key metrics (concepts, 8)
- TTFT, ITL, queue time histograms: S/r-2:11,12,15. p99 over 5 min alert: S/5-2:81
- num_queue_reqs, token_usage, num_retracted_reqs alerts: S/5-2:83-85
- prefill_effective_tokens_total windowed hit rate: S/5-2:47,66
- cache_hit_rate last interval: S/5-2:39

## Log lines to read (command-table, 8)
- bytes_per_slot=114688, pool_size=181574: O/server-default.log:61
- max_total_num_tokens=181574, max_running_requests=4096: O/server-default.log:70
- #new-token: 3, #cached-token: 2: O/server-default.log:94, S/2-1 listing via 1-2:87
- #running-req: 5 ... #queue-req: 11: O/server-small-pool.log:118
- KV cache pool is full. Retract requests.: S/2-3:82
- crashed with exit code -15: O/server-default.log:566, S/2-1:111
- Insufficient Memory: O/oom-exploratory.log:15, S/3-2:43 (memory outside the pool)

## Troubleshooting (8)
- Insufficient Memory, 87 percent, 0.4: S/r-3:11
- RuntimeError, --disable-overlap-schedule, keep overlap on: S/r-3:12
- retractions, raise conservativeness: S/2-3:114, S/r-3:21
- high TTFT fast TPOT, past knee, cap: S/2-3:65-74, S/5-1:123,131
- no cached_tokens without --enable-cache-report: S/1-2:133
- #cached-token near 0, id at start, shared text first: S/3-4:29-33
- health check failed, 20 s, /ready: S/r-3:70
- crash line at shutdown, process group: S/r-3:13

## Glossary (8)
- TTFT: S/r-4:71, expansion S/3-3:16. TPOT: S/r-4:69, O/bench-shared-prefix.txt:40. ITL: S/r-4:34, O/bench-shared-prefix.txt:46
- KV cache: S/r-4:35. TP: S/r-4:65. DP: S/r-4:20. EP: S/r-4:25. PD: S/r-4:55

## Q&A (5)
- GIL: S/2-1:3,13. Five requests, 768 each: S/2-3:38. 2,998 of 2,999: S/3-3:65
- 0.942 vs 0.636, trust counters: S/5-2:56,111. Knee at or below 8: S/5-1:123

## Tips (6)
- --random-range-ratio 1.0: S/5-1:47. p50/p99 TTFT and TPOT: S/5-1:129
- steps of 0.02: S/3-2:88. radix cache on, measure hit rate per feature: S/3-3:98-102
- pin lmsysorg/sglang:v0.5.22: S/7-1:24,161. mount ~/.cache/sglang: S/7-1:41,162
