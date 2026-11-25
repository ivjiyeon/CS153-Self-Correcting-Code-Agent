# Self-Correcting Coding Agent with Execution Feedback

Compares, on MBPP+ (loaded with `evalplus.data.get_mbpp_plus()`), with a small local Qwen model:
single-shot, self-reflection (no execution), best-of-N (fixed and token-matched), and
test-feedback repair (up to 5 rounds) with three feedback levels (`pass_fail`, `traceback`,
`expected_vs_actual`), under three prompting styles (`base`, `cot`, `react`).
Optional extra condition: `--use_all_tests` (feedback from visible AND hidden cases).

## Data layout (created by `python prepare_data.py`)
| folder | content |
|---|---|
| `data/raw/` | exact `get_mbpp_plus()` output (`mbpp_plus_raw.pkl`) + manifest |
| `data/processed/` | **dev split (90%)**: task text, 3 visible tests, visible cases |
| `data/private/` | hidden cases for all problems, **eval split (10%)**, canonical solutions, split manifest |

Visible cases = first 3 original MBPP inputs. Hidden cases = remaining original inputs + all MBPP+
extra inputs, minus duplicates of visible cases, with expected outputs produced by running the
canonical solution. Never show `data/private/` to the model.

## Results layout
`results/base/`, `results/cot/`, `results/react/`, each with `<split>_shard<N>.jsonl`.

## Run
```bash
pip install -r requirements.txt
python download_assets.py                   # login node (internet): model + MBPP+
python prepare_data.py                      # builds data/raw, data/processed, data/private
python -m src.data_loading --split dev      # sanity: canonical passes, a stub fails
python -m src.data_loading --split eval
mkdir -p logs results
sbatch run_experiment.sbatch                # 3 styles x 4 shards; edit SPLIT / USE_ALL_TESTS inside
python analyze_results.py --results_root results --split dev
```
Use `SPLIT="dev"` for development and the main comparison; run `SPLIT="eval"` once at the end.

## Notes
- Test cases are pickled Python objects, so tuples, dicts and complex numbers are restored exactly.
  Only plain text (problem descriptions, assert strings) is stored as JSON.
- The sandbox fixes `PYTHONHASHSEED=0` so set/dict iteration order is reproducible between the
  canonical run and candidate runs. Hidden cases whose pickled size exceeds 1 MB are dropped
  (some MBPP+ outputs are hundreds of MB).
- Metrics: `visible_pass_rate`, `hidden_pass_rate` (unseen cases only), `standard_pass_rate`
  (visible + hidden, EvalPlus-style), `generalization_gap`, `hidden_fail_given_visible_pass`,
  `hardcoding_rate`. With `use_all_tests`, the hidden score is no longer independent
  (flagged by `hidden_tests_used_as_feedback`).
