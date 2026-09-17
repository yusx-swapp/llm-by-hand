# LLM by Hand

**Learn to build and train LLMs, one line at a time.**

A hands-on curriculum built on real source code. Type along, fill the gaps, and rebuild—from attention to training and MoE.

The interface and teaching notes are currently primarily in Chinese. Reference implementations retain their original upstream code.

> **Local, trusted-code use only.** This application executes the Python you submit. Subprocesses and timeouts are not a security sandbox. Do not expose this runner to untrusted users or the public internet. See [SECURITY.md](SECURITY.md).

## How learning works

Every lesson has three sequential stages:

1. **Type along** — an unedited upstream reference next to your own blank editor. Hover over a statement or variable for a contextual explanation and recorded tensor shapes, dtypes and values.
2. **Fill the gaps** — meaningful implementation regions are hidden while the surrounding context remains. Run the completed code to verify it.
3. **Rebuild independently** — start from the interface, not the answer. Passing unlocks the next lesson and saves a verified implementation snapshot.

Hover observations come from real reference execution on explicit small inputs. They are not fabricated states of an unexecuted draft. Unmatched learner code gets concept explanations without misleading runtime values.

## Curriculum

**22 ordered lessons**, starting at the foundations rather than post-training:

| Lessons | Topics |
|---|---|
| 001–006 | MHA math and projections, causal masks, RMSNorm, RoPE, SwiGLU |
| 007–009 | Llama attention, decoder layers and causal-LM output |
| 010–015 | Qwen3 QK-Norm, Gemma softcap, MoE routing/experts, Mixtral and DeepSeek MLA |
| 016–019 | Causal-LM loss, data collation, learning-rate scheduling and pretraining steps |
| 020–021 | SFT supervision masks and DPO preference objectives |
| 022 | Top-p generation and the integrated training experiment |

See the [full course list](docs/curriculum.md).

## Run locally

The pinned environment was tested with **Python 3.14**, PyTorch 2.10.0, Transformers 5.1.0 and TRL 1.1.0. Dependency installation needs internet access; normal lessons do not fetch model weights, datasets or AI explanations.

```sh
git clone https://github.com/yusx-swapp/llm-by-hand.git
cd llm-by-hand
python -m venv .venv
```

Activate the environment:

```sh
# macOS / Linux
source .venv/bin/activate
```

```cmd
:: Windows Command Prompt
.venv\Scripts\activate
```

Then install and start:

```sh
python -m pip install -r requirements.txt
python -X utf8 app.py
```

Open **http://127.0.0.1:8017/#/001/follow**. Windows users can also run `start.cmd` after installation.

There is no frontend build, npm server, external CDN or required AI-provider key. Monaco is served locally.

Keyboard shortcuts: `Ctrl/⌘ + Enter` to verify, `Ctrl/⌘ + S` to save. The editor also provides a button to explain the current cursor position.

### Saving and recovery

- Edits are written to the local SQLite database after a short debounce and survive application restarts.
- The editor header shows the latest saved time. **Save progress** (`Ctrl/⌘ + S`) creates a named recovery point; continued work also receives periodic snapshots.
- **Save history** lists recovery points for the current lesson and stage. Restoring one first preserves the current code as another backup.
- Closing or hiding the page attempts a final beacon write, while every keystroke also keeps a browser recovery copy in case the server was stopped before the request completed.
- Drafts, checkpoints and pass records remain separate per lesson and per stage.

## Connect the code to training

Once the required independent implementations pass, the course can run a tiny CPU experiment using those verified snapshots:

- Assemble learner MHA, mask, normalization, RoPE, MLP, attention, decoder and LM code into a small Llama model.
- Perform real pretraining updates, SFT with prompt labels masked, and a DPO update against a frozen reference.
- Generate tokens with KV cache and learner Top-p code.
- Install learner experts and a learner MoE block in a small Mixtral model, then train and check expert gradients.
- Save/reload checkpoints and export verified source, runtime helpers and checkpoints in a replayable ZIP.

Integration helpers are project-owned glue, clearly separate from upstream lesson references. The experiment uses **passed source**, never a later broken draft.

These are small educational runs, not claims of production model quality or large-scale training. Full distributed/FSDP, LoRA/PEFT, multimodal and RL-rollout curricula are outside the current scope. Not every branch of each upstream implementation is covered by the supplied scenarios.

## Project structure

```text
app.py                    Local FastAPI entry point
qk/                       Small source, execution, progress and training modules
lessons/curriculum.json    Ordered curriculum and prerequisites
lessons/<id>/             Exact source, provenance, notes, blanks and observations
web/course/               Native HTML/CSS/JavaScript classroom
web/vendor/monaco/        Offline editor and its license notices
tests/                    Source, API, hover and real-execution regression tests
```

No ORM, plugin framework or frontend state-management library is required.

## Local state and configuration

- Progress defaults to `data/progress.sqlite3`.
- `QK_DB` selects another database; its experiment artifacts live beside it in a `*-runs/` directory.
- `QK_PORT` selects the local port (default `8017`). The server remains loopback-only.
- Drafts, databases, checkpoints, local environments and experiment outputs are ignored by Git. No personal practice history is shipped.
- Failed saves retain a browser recovery copy. Verification does not overwrite newer edits. Draft history is stored locally and is never part of the Git repository.

This repository contains a runnable application, not a GitHub Pages deployment: Pages cannot execute its Python/PyTorch validation backend.

## Development checks

```sh
python -X utf8 -m pytest
node --test tests/test_course_hover.mjs
python -X utf8 qk/course_worker.py check-all --output reference-check.json
```

Node is optional and used only for the frontend's pure-logic tests. Python tests use temporary databases and real small-model execution.

After changing source plumbing, fixtures or the observer, rebuild the recorded observations:

```sh
python -X utf8 qk/course_worker.py observe-all --output observation-build.json
```

Source/observer fingerprints are checked before showing observations. Do not hand-edit tensor values to make an example look correct.

## Source attribution and license

Application code is licensed under **Apache-2.0**. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Lesson excerpts come from pinned Transformers and TRL releases. Each lesson retains its source path, line range, hashes, license and copyright notice. Method carriers and other teaching adaptations are separate from the reference files.

Monaco retains its original license and third-party notices. Installed dependencies remain subject to their own licenses. See [source attribution](docs/attribution.md).
