# Self-Correcting Coding Agent with Execution Feedback

Compares, on MBPP+ with a small local Qwen model:
single-shot, self-reflection (no execution), best-of-N (fixed and token-matched),
and test-feedback repair (up to 5 rounds) with three feedback levels
(`pass_fail`, `traceback`, `expected_vs_actual`). Each problem shows 3 tests to the
model; all other tests (extended MBPP+ tests) are hidden and used only for scoring.

## Layout
- `src/data_loading.py` – MBPP+ loading, visible/hidden split (`python -m src.data_loading` = sanity check with reference solutions)
- `src/sandbox.py` – subprocess execution with time/memory limits, per-test expected/actual
- `src/feedback.py` – the three feedback levels
- `src/prompts.py` – ReAct-style prompts, code extraction
- `src/strategies.py` – all strategies and token accounting
- `src/hardcoding_detection.py` – AST heuristic for test hard-coding
- `run_experiment.py`, `run_experiment.sbatch`, `analyze_results.py`, `download_assets.py`

## Run on HPC
```bash
pip install -r requirements.txt
python download_assets.py --model_name Qwen/Qwen2.5-Coder-3B-Instruct   # login node (internet)
python -m src.data_loading                                              # reference solutions should pass 3/3 and hidden=True
mkdir -p logs results
sbatch run_experiment.sbatch                                            # 4 array shards -> results/results_shard*.jsonl
python analyze_results.py --results_dir results --output_dir analysis
```

## Design notes
- Repair and self-reflection start from the single-shot attempt (round 0 is shared), and its tokens are counted in their budgets.
- `best_of_n_matched_<level>` keeps sampling (T=0.8) until it has spent as many tokens (prompt+completion) as the matching repair run, giving the equal-compute comparison.
- Repair stops as soon as all visible tests pass; the final answer is the best candidate on *visible* tests. Hidden tests never influence the agent.
- Overfitting metrics: `generalization_gap` (visible − hidden pass rate), `hidden_fail_given_visible_pass`, and `hardcoding_rate` (visible inputs and outputs both embedded as literals inside functions).
- The jobs resume automatically: re-submitting skips finished (task, method) pairs.
