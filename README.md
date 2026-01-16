# logical_attribution_module

This directory contains a modularised version of the logical attribution
pipeline that was previously implemented in a single `logical_attribution.py`
script.

## Modules and stages

The pipeline is split into the following modules:

- `config.py` &mdash; LLM configuration and low-level API helpers.
- `preprocess.py` &mdash; optional dataset loading and normalisation helpers,
  `pre_process(args)`. This step is **not** invoked automatically by the
  pipeline any more; Stage&nbsp;1 now assumes that the input data is already in
  the unified pre-processed format described below.
- `qa.py` &mdash; prompts and helpers used across stages (reasoning generation,
  answer checking, etc.).
- `stage1.py` &mdash; question answering with citations,
  `stage_1_processing(total_data, args)`.
- `stage2.py` &mdash; extraction of logical answers from the generated
  statements, `stage_2_processing(total_data, args, num_workers=10)`.
- `stage3.py` &mdash; reasoning-path analysis and scoring,
  `stage_3_processing(total_data, args, ...)`.
- `stage4.py` &mdash; citation relevance / support evaluation utilities,
  including `is_support`, `is_relevant` and `need_citation`, plus
  `stage_4_processing(...)` for batch processing when needed.
- `stage5.py` &mdash; FActScore-style factuality evaluation of generated
  statements, `stage_5_processing(args, total_data, num_workers=4)`.

## Stage 1 input format

Stage&nbsp;1 expects its input to be a JSON array of records, where each record has
at least the following fields:

- `id` *(optional)* &mdash; original example identifier.
- `query` *(str)* &mdash; the question text.
- `context` *(list)* &mdash; a list of document entries, each of the form
  `[title, [sent_1, sent_2, ...]]`, where `title` is a string and the inner
  list contains sentence strings.
- `answer` *(str)* &mdash; the gold answer string.

In Python terms, the structure looks like:

```json
[
  {
    "id": "example-id-1",
    "query": "...question text...",
    "context": [
      ["Doc title 1", ["Sentence 1.", "Sentence 2."]],
      ["Doc title 2", ["Another sentence."]]
    ],
    "answer": "...gold answer..."
  },
  {
    "id": "example-id-2",
    "query": "...",
    "context": [
      ["Title", ["..."]]
    ],
    "answer": "..."
  }
]
```

## Entry points

The main entry points you are likely to use are:

- `logical_attribution_module.pipeline.main(args)` &mdash; run the full
  multi-stage pipeline from Python.
- `logical_attribution_module.cli.main(argv=None)` &mdash; command-line
  interface used by `python -m logical_attribution_module.cli`.
- `logical_attribution_module.run_pipeline` &mdash; alias exported from
  `__init__.py` for convenience.

## Using from Python

A minimal example of running the pipeline programmatically, starting from
Stage&nbsp;1 (i.e. assuming you already have a pre-processed Stage&nbsp;1 input
file):

```python
from types import SimpleNamespace
from logical_attribution_module import run_pipeline

args = SimpleNamespace(
	    dataset="2wiki",
	    file_path="data/2wiki_stage1_input.json",
	    save_path="results/2wiki",
	    generate_model="Qwen-8B",
	    num_workers=40,
	    start_stage=1,
	)

run_pipeline(args)
```

## Running from the command line

```bash
python -m logical_attribution_module.cli \
	  --dataset 2wiki \
	  --generate_model Qwen-8B \
	  --start_stage 1
```

If `--file_path` and `--save_path` are not provided, the CLI falls back to
per-dataset **relative** defaults under the project root (for example,
`data/2wiki_stage1_input.json` and `results/2wiki`).

