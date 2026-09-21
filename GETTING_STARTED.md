# Promptscope Quick Start

A plain-language guide. No coding experience needed. (Developers: see [README.md](README.md) for the architecture and API.)

## What Promptscope does

Promptscope checks the instructions ("prompts") you plan to give an AI and makes sure they are professional and on-topic for the CSAW hardware project. You give it a prompt, and it answers with one of three verdicts:

- **ACCEPT** — the prompt is fine as written.
- **REWRITE** — the prompt was on-topic but sloppy or vague, so Promptscope cleaned it up and hands you the improved version.
- **REJECT** — the prompt is off-topic (an essay, the weather, anything aimed at real hardware outside the competition), so it is thrown out.

It can check one prompt or a whole list at once. It works with no AI account at all; connecting a real AI model (see the optional section below) makes the rewrites much better.

## Before you start

You need two things: Python (a free program that runs the code) and a terminal (the window where you type commands). Nothing else needs installing.

**Open a terminal.**

- Mac: press Cmd + Space, type `Terminal`, press Enter.
- Windows: press the Windows key, type `PowerShell`, press Enter.

**Check that Python is installed.** In the terminal, type the line for your computer and press Enter:

- Mac: `python3 --version`
- Windows: `python --version`

If you see a version number such as `Python 3.12.4`, you are ready. If you see "command not found" or a version starting with 2, install Python from [python.org/downloads](https://www.python.org/downloads/) (pick the latest 3.x version; on Windows, tick "Add python.exe to PATH" in the installer), then close and reopen the terminal.

In the rest of this guide, "type a command" means: type it into that window and press Enter. Every command starts with `python3` on a Mac and `python` on Windows. That is the only difference between the two.

## Get the code

1. On this repository's GitHub page, click the green **Code** button, then **Download ZIP**.
2. Find the downloaded file (usually in Downloads) and double-click it. A folder named `promptscope-main` appears.
3. Move that folder somewhere you can find it, such as your Desktop or Documents.

That is the whole installation. There is nothing to run or set up.

## Run it

All commands are typed inside the folder you downloaded, so point the terminal there first.

**Step 1 — Go into the folder.** Type `cd ` (with a space after it), drag the `promptscope-main` folder from Finder or File Explorer into the terminal window, and press Enter. The terminal now works inside that folder. You do this once each time you open a new terminal.

**Step 2 — Check one prompt.** Type this on a Mac (on Windows, swap `python3` for `python`):

```
python3 -m promptscope "Lint alu.v with verilator and list every warning; keep the module interface unchanged."
```

Keep the quotation marks around the prompt. The verdict appears in under a second. Now try a sloppy one and watch it get cleaned up:

```
python3 -m promptscope "hey can u make the alu module pass its testbench asap lol"
```

**Step 3 — Check a whole list.** Put one prompt per line in a plain text file and save it inside the `promptscope-main` folder as `my_prompts.txt` (Mac: TextEdit, then Format → Make Plain Text; Windows: Notepad). Then type:

```
python3 -m promptscope -f my_prompts.txt
```

A sample list is included, so `python3 -m promptscope -f examples/prompts.txt` works right away. To keep the results, add `--jsonl results.jsonl` to the end of any command and a results file appears in the folder.

## Reading the results

Each prompt gets a verdict line, then the prompt you typed (`in`) and, for rewrites, the improved version (`out`):

```
[REWRITE ] score=0.85  id=b2fe79502288  iters=0
   in : hey can u make the alu module pass its testbench asap lol
   out: Can you make the alu module pass its testbench. Preserve functional equivalence with the original design and keep the module interface unchanged. Return a unified diff and a short explanation.
```

- The word in brackets is the verdict: **ACCEPT** (use the prompt as is), **REWRITE** (use the `out` line instead), or **REJECT** (do not use it; it is off-topic or outside the project's rules).
- `score` is how confident Promptscope is that the prompt fits the project, from 0.00 to 1.00. Above 0.60 is solid.
- `id` and `iters` are bookkeeping. You can ignore them.

After the list, one summary line counts how many prompts were accepted, rewritten and rejected. To see *why* a verdict was reached, add `-v` to the end of the command and the reasons appear under each prompt.

Without a real AI model connected, rewrites come from a built-in stand-in: it tidies the wording and appends standard engineering constraints, which is enough to see how the tool works. Connect a real model (next section) for rewrites that properly reshape the prompt.

## Connect a real AI model (optional)

Promptscope can hand the hard cases to a real AI model: Claude, OpenAI/Codex, or a model running on your own computer. For the online ones you need an **API key**, a long code from the AI company that lets a program use their service. Keys are usually pay-per-use, so treat one like a password.

**Claude (Anthropic).** Create a key at [console.anthropic.com](https://console.anthropic.com/) under API Keys. Then type these two lines in the terminal, pasting your key inside the quotes:

Mac:

```
export ANTHROPIC_API_KEY="paste-your-key-here"
python3 -m promptscope -f my_prompts.txt --backend anthropic --model claude-sonnet-4-5
```

Windows (PowerShell):

```
$env:ANTHROPIC_API_KEY="paste-your-key-here"
python -m promptscope -f my_prompts.txt --backend anthropic --model claude-sonnet-4-5
```

**OpenAI / Codex.** Create a key at [platform.openai.com](https://platform.openai.com/api-keys). Same two lines, but with `OPENAI_API_KEY` in the first and `--backend openai --model gpt-4o-mini` in the second. If you use the Codex command-line app instead, no key is needed: add `--backend subprocess --cmd "codex exec -"` to the command.

**A model on your own computer (free, private).** If you have [Ollama](https://ollama.com/) installed and have run `ollama run llama3.1` once, add `--backend openai --base-url http://localhost:11434/v1 --model llama3.1` to the command.

Two things to know: the key line only lasts for that terminal window, so type it again after reopening the terminal; and if you get a "model not found" error, the model name is out of date, so swap in a current one from the company's model list (for Claude, [docs.claude.com](https://docs.claude.com/en/docs/about-claude/models)). Everything else in this guide works the same. Only the rewrites get smarter.

## If something goes wrong

Almost every problem is one of these six. Match the message you see, then apply the fix.

| What you see | What it means | Fix |
| --- | --- | --- |
| `command not found: python3` or `'python' is not recognized` | Python is not installed, or the terminal was opened before you installed it | Install Python from [python.org](https://www.python.org/downloads/), then close and reopen the terminal. On Windows, re-run the installer and tick "Add python.exe to PATH". |
| `No module named promptscope` | The terminal is not inside the `promptscope-main` folder | Repeat Step 1 of "Run it": type `cd `, drag the folder in, press Enter. |
| `No such file or directory: 'my_prompts.txt'` | The prompt file is not in that folder, or is named differently | Save it inside `promptscope-main` and check the name. TextEdit sometimes saves as `.rtf`; Notepad sometimes adds a second `.txt`. |
| `ANTHROPIC_API_KEY not set` or `HTTP 401` | The key was not set in this terminal window, or was pasted with a typo | Run the `export` / `$env:` line again in the same window, then the command. |
| `HTTP 404` mentioning the model | The model name is out of date | Replace the name after `--model` with a current one from the company's model list. |
| Every prompt says REJECT | The prompts do not mention the design, files or tools, so they read as off-topic | That is the tool working as intended. Rephrase around the actual hardware work: name the file or module and say what you want back. |

Still stuck? Type `python3 tests/test_promptscope.py` (Windows: `python`) inside the folder. It should end with `0 failed`. If it does, the program is fine and the problem is in the command you typed; if not, send the output to whoever set the tool up.
