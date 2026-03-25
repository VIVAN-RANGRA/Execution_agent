"""Batch evaluation across multiple episodes and agents."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from evaluation.counterfactual_regret import CounterfactualRegretTracker
from evaluation.episode_runner import run_episode
from evaluation.metrics import mann_whitney_p_value, win_rate_vs_baseline
from simulator.data_classes import EpisodeResult

BASE_DIR = Path(__file__).resolve().parent.parent


def _evaluate_agent_batch(agent, env, seeds: List[int]) -> Dict:
    """Run one agent across all seeds in an isolated worker/process."""
    results: List[EpisodeResult] = []
    errors = []
    for seed in seeds:
        try:
            results.append(run_episode(agent, env, episode_seed=seed))
        except Exception as exc:  # pragma: no cover - surfaced in parent summary
            errors.append({"seed": seed, "error": str(exc)})
    return {
        "agent_name": agent.name,
        "results": results,
        "errors": errors,
    }


def _compute_summary(all_results: Dict[str, List[EpisodeResult]]) -> tuple[Dict, Dict, Dict]:
    """Compute summary tables and counterfactual regret trackers."""
    summary = {}
    twap_is = [r.implementation_shortfall_bps for r in all_results.get("TWAP", [])]
    twap_is_by_seed = {
        result.episode_seed: result.implementation_shortfall_bps
        for result in all_results.get("TWAP", [])
    }
    regret_trackers: Dict[str, CounterfactualRegretTracker] = {}

    ac_is = [r.implementation_shortfall_bps for r in all_results.get("AC_Optimal", [])]

    for agent_name, results in all_results.items():
        if not results:
            continue

        is_values = [r.implementation_shortfall_bps for r in results]
        vwap_slip = [r.vwap_slippage_bps for r in results]
        part_rates = [r.participation_rate for r in results]

        arr = np.array(is_values)
        mean_is = float(np.mean(arr))
        std_is = float(np.std(arr))
        p25 = float(np.percentile(arr, 25))
        p75 = float(np.percentile(arr, 75))

        wr_vs_twap = win_rate_vs_baseline(is_values, twap_is) if twap_is and agent_name != "TWAP" else 0.5
        p_val = mann_whitney_p_value(is_values, twap_is) if twap_is and agent_name != "TWAP" else 1.0
        wr_vs_ac = (
            win_rate_vs_baseline(is_values, ac_is)
            if ac_is and agent_name not in ("TWAP", "AC_Optimal")
            else None
        )

        summary[agent_name] = {
            "mean_IS_bps": mean_is,
            "std_IS_bps": std_is,
            "median_IS_bps": float(np.median(arr)),
            "p25_IS_bps": p25,
            "p75_IS_bps": p75,
            "iqr_IS_bps": p75 - p25,
            "max_IS_bps": float(np.max(arr)),
            "min_IS_bps": float(np.min(arr)),
            "information_ratio": float(mean_is / (std_is + 1e-8)),
            "mean_vwap_slippage_bps": float(np.mean(vwap_slip)),
            "mean_participation_rate": float(np.mean(part_rates)),
            "win_rate_vs_TWAP": float(wr_vs_twap),
            "win_rate_vs_AC": float(wr_vs_ac) if wr_vs_ac is not None else None,
            "p_value_vs_TWAP": float(p_val),
            "n_episodes": len(results),
            "is_values": is_values,
        }

        if agent_name != "TWAP" and twap_is_by_seed:
            tracker = CounterfactualRegretTracker()
            for result in results:
                twap_val = twap_is_by_seed.get(result.episode_seed)
                if twap_val is not None:
                    tracker.record(result.implementation_shortfall_bps, twap_val)
            regret_trackers[agent_name] = tracker

    return summary, regret_trackers, twap_is_by_seed


def _write_results_snapshot(
    output_path: Path,
    n_episodes: int,
    split: str,
    random_seed: int,
    seeds: List[int],
    summary: Dict,
    regret_trackers: Dict[str, CounterfactualRegretTracker],
    completed_agents: List[str],
) -> None:
    output = {
        "timestamp": datetime.now(UTC).isoformat(),
        "n_episodes": n_episodes,
        "split": split,
        "random_seed": random_seed,
        "completed_agents": completed_agents,
        "summary": {
            name: {key: value for key, value in values.items() if key != "is_values"}
            for name, values in summary.items()
        },
        "is_distributions": {name: values["is_values"] for name, values in summary.items()},
        "seeds": seeds,
        "regret_tracking": {
            agent_name: tracker.get_summary()
            for agent_name, tracker in regret_trackers.items()
        },
        "regret_curves": {
            agent_name: tracker.get_regret_curve().tolist()
            for agent_name, tracker in regret_trackers.items()
        },
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def _print_summary(summary: Dict, regret_trackers: Dict[str, CounterfactualRegretTracker]) -> None:
    print("\n" + "=" * 100)
    print(
        f"{'Agent':<18} {'Mean IS':>10} {'Std IS':>10} {'Win vs TWAP':>12} "
        f"{'P-value':>10} {'Cum Regret':>12} {'Degraded':>10}"
    )
    print("-" * 100)
    for agent_name, values in summary.items():
        if agent_name in regret_trackers:
            regret_summary = regret_trackers[agent_name].get_summary()
            cum_regret_str = f"{regret_summary['total_cumulative_regret']:>12.2f}"
            degraded_str = f"{'YES' if regret_summary['degradation_detected'] else 'no':>10}"
        else:
            cum_regret_str = f"{'--':>12}"
            degraded_str = f"{'--':>10}"
        print(
            f"{agent_name:<18} {values['mean_IS_bps']:>10.2f} {values['std_IS_bps']:>10.2f} "
            f"{values['win_rate_vs_TWAP']:>12.3f} {values['p_value_vs_TWAP']:>10.4f} "
            f"{cum_regret_str} {degraded_str}"
        )
    print("=" * 100)


def run_batch_evaluation(
    agents: list,
    env,
    n_episodes: int = 50,
    split: str = "test",
    random_seed: int = 42,
    results_dir: str = None,
    max_workers: int = 1,
) -> Dict:
    """
    Evaluate all agents on the same episode seeds.

    Parallelism is applied at the agent level so each worker owns its own agent
    and environment copy. That avoids the shared mutable state bug that occurs
    when parallelizing episodes against the same env/agent instances.
    """
    results_dir = Path(results_dir) if results_dir else BASE_DIR / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(random_seed)
    seeds = rng.randint(0, int(1e6), n_episodes).tolist()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    partial_path = results_dir / f"{timestamp}.partial.json"
    out_path = results_dir / f"{timestamp}.json"

    all_results: Dict[str, List[EpisodeResult]] = {}
    agent_errors: Dict[str, List[Dict]] = {}
    pending_agents = list(agents)

    if len(pending_agents) > 1 and max_workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=min(max_workers, len(pending_agents))) as executor:
                future_to_agent = {
                    executor.submit(_evaluate_agent_batch, agent, env, seeds): agent.name
                    for agent in pending_agents
                }
                with tqdm(total=len(future_to_agent), desc="Agents") as pbar:
                    for future in as_completed(future_to_agent):
                        agent_name = future_to_agent[future]
                        result = future.result()
                        all_results[agent_name] = result["results"]
                        agent_errors[agent_name] = result["errors"]
                        summary, regret_trackers, _ = _compute_summary(all_results)
                        _write_results_snapshot(
                            partial_path,
                            n_episodes,
                            split,
                            random_seed,
                            seeds,
                            summary,
                            regret_trackers,
                            sorted(all_results.keys()),
                        )
                        pbar.update(1)
        except Exception as exc:
            print(f"Parallel agent evaluation failed ({exc}); falling back to sequential execution.")
            all_results = {}
            agent_errors = {}

    if not all_results:
        for agent in pending_agents:
            agent_name = agent.name
            print(f"\nEvaluating {agent_name} on {n_episodes} episodes...")
            result = _evaluate_agent_batch(agent, env, seeds)
            all_results[agent_name] = result["results"]
            agent_errors[agent_name] = result["errors"]
            summary, regret_trackers, _ = _compute_summary(all_results)
            _write_results_snapshot(
                partial_path,
                n_episodes,
                split,
                random_seed,
                seeds,
                summary,
                regret_trackers,
                sorted(all_results.keys()),
            )

    summary, regret_trackers, _ = _compute_summary(all_results)
    _print_summary(summary, regret_trackers)
    _write_results_snapshot(
        out_path,
        n_episodes,
        split,
        random_seed,
        seeds,
        summary,
        regret_trackers,
        sorted(all_results.keys()),
    )
    if partial_path.exists():
        partial_path.unlink()

    if any(agent_errors.values()):
        print("\n--- Episode Failures ---")
        for agent_name, errors in agent_errors.items():
            if errors:
                print(f"{agent_name}: {len(errors)} failed episodes")

    print(f"\nResults saved to {out_path}")
    return {
        "summary": summary,
        "results": all_results,
        "output_path": str(out_path),
        "regret_trackers": regret_trackers,
        "agent_errors": agent_errors,
    }
