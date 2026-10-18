# Writing a cheat sheet

A cheat sheet is a reference that you use while you work. It is not a summary of a lesson. One file holds one topic. The build turns the file into a poster: one landscape page, two at most.

Each topic is one file, `cheatsheets/<id>.md`. The id is lower case with hyphens.

## File shape

```text
---
id: sglang-serving
title: LLM Serving with SGLang
subtitle: Start, size, measure and fix an SGLang server. SGLang 0.5.22.
hue: 200
sources: manuals/sglang-101, phases/17-infrastructure-and-production/03-vllm-serving
---

## Start a server
type: annotated-code
lang: bash
from: manuals/sglang-101/sections/2-1-sglang-serve-and-the-three-processes.md

    ```bash
    sglang serve --model-path <model> --port 30000   # start the HTTP server
    ```
```

The front matter has five keys. `id`, `title` and `subtitle` are text. The subtitle has a limit of 120 characters, so that it fits on one line. `hue` is a number from 0 to 359. It sets the colour of the poster. `sources` lists the lessons and manuals that the topic covers. Lesson pages link to the sheet from these paths. The optional key `code` sets the two letters on the tile. The default is the initials of the title. Each topic needs a different code.

Each block starts with a `## ` heading. The heading is the label on the coloured bar. Lines of `key: value` follow the heading. A blank line ends them. The body comes after the blank line.

| Key | Required | Meaning |
|---|---|---|
| `type` | yes | one of the block types below |
| `from` | yes | comma-separated repository paths that hold the facts of this block |
| `lang` | for code | the language of an `annotated-code` block |
| `col` | no | the poster column, 1 to 4. Without it, the build puts the blocks in order and balances the columns |
| `span` | no | how many columns the block uses, 1 to 4. Annotated code, architecture, defaults and troubleshooting use 2 by default. The others use 1 |

## Block types

| Type | Colour | Body |
|---|---|---|
| `concepts` | violet | bullets: `- **Name**: one line` |
| `glossary` | violet | bullets: `- **ABBR**: what it stands for and one line` |
| `task-table` | blue | table: `\| Task \| Code \|`. Put the code cell in backticks |
| `command-table` | blue | table: `\| Command \| What it does \|` |
| `annotated-code` | ink | one fenced code block. Put a comment after two or more spaces. The build aligns the comments in a column |
| `flow` | teal | numbered list: `1. **Actor**: what happens`. The build draws numbered steps |
| `architecture` | teal | a fenced `arch` block with `node` and `edge` lines, see below |
| `defaults` | amber | table: `\| Setting \| Default \| Note \|` |
| `troubleshooting` | red | table: `\| Symptom \| Cause \| Fix \|` |
| `qa` | green | bullets: `- **Question?** Answer.` |
| `tips` | green | bullets |

Mark a value that the reader replaces with angle brackets: `<model>`, `<port>`. The build colours them.

### Architecture blocks

```text
node | api | 1,1 | HTTP server | tokenizer manager
node | sched | 2,1 | Scheduler | batches and KV pool
edge | api | sched | ZMQ
```

A node line is `node | <id> | <column>,<row> | <label> | <second line> | <kind>`. Columns and rows start at 1. The second line can be empty. The kind is optional. It draws a small mark in the corner of the box: `client`, `process`, `store`, `queue`, `gpu` or `model`. An edge line is `edge | <from id> | <to id> | <label>`. The label can be empty.

## Facts

Every fact must come from a file in the `from` list of its block: a lesson, a lesson's code, a manual section or a manual capture. Do not write a command, flag, API name, default or number that is not in those files. `node site/build-cheatsheets.js --check` enforces part of this rule. Each flag (`--name`), each name with an underscore, a dot or a call, and each default value must appear in at least one `from` file of its block.

## Prose

Use short sentences in Simplified Technical English. Use one line for each row or bullet. Do not use dashes, semicolons or figurative words. A cheat sheet tells the reader what to type and what to expect. It does not tell a story.
