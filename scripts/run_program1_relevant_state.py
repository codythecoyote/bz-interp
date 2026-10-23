#!/usr/bin/env python3
"""Run exact full-frontier and hierarchical spatial-state interventions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "software" / "baskerville" / "src"))

from borzoi_circuits.artifacts import (  # noqa: E402
    ShardManifest,
    validate_completed_shard,
    write_shard_manifest,
)
from borzoi_circuits.graph_surgery import FunctionalGraphReplay  # noqa: E402
from borzoi_circuits.interventions import InterventionResult  # noqa: E402
from borzoi_circuits.locking import read_lock, sha256_file, stable_json_hash  # noqa: E402
from borzoi_circuits.relevant_state import (  # noqa: E402
    bidirectional_recovery,
    coverage_delta_consistency,
    patch_frontier_positions,
    spatial_intervals_to_genomic,
)
from borzoi_circuits.search import hierarchical_state_search  # noqa: E402
from borzoi_circuits.sequence import one_hot_allele_pair  # noqa: E402
from borzoi_circuits.targets import (  # noqa: E402
    choose_joint_window,
    intervals_to_bins,
    native_summary_tensor,
    parse_exon_intervals,
    target_spec_from_registry,
)


class CoverageInstabilityError(RuntimeError):
    """A model-response exclusion, distinct from instrumentation failure."""

    def __init__(self, details: dict[str, object]):
        self.details = details
        super().__init__(
            "coverage_delta_instability:"
            f"absolute={details['coverage_delta_absolute_error']:.8g},"
            f"normalized={details['coverage_delta_normalized_error']:.8g},"
            f"sign_match={details['coverage_delta_sign_match']},"
            f"above_noise={details['coverage_delta_above_noise']}"
        )


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def compress_positions(positions: Sequence[int]) -> tuple[tuple[int, int], ...]:
    values = sorted(set(int(value) for value in positions))
    if not values:
        return ()
    intervals: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            intervals.append((start, previous + 1))
            start = value
        previous = value
    intervals.append((start, previous + 1))
    return tuple(intervals)


def selected_cohort(frame: pd.DataFrame, subset: str) -> pd.DataFrame:
    if set(frame["split"].astype(str)) != {"discovery"}:
        raise RuntimeError("cohort contains a non-discovery record")
    if subset == "smoke":
        selected = frame[frame["smoke_selected"].astype(bool)].copy()
        return selected.sort_values("smoke_index").reset_index(drop=True)
    return frame.sort_values("cohort_index").reset_index(drop=True)


def load_model(config: dict):
    from baskerville import seqnn

    params = json.loads(resolve(config["model"]["params"]).read_text())
    network = seqnn.SeqNN(params["model"])
    network.restore(str(resolve(config["model"]["primary"])), 0)
    network.model.trainable = False
    return network.model


def build_scalar_replay(model, routes: dict, cut: str, target, bins: np.ndarray):
    import tensorflow as tf

    frontier_names = tuple(routes["cuts"][cut])
    head = model.get_layer(routes["output_head"]).output
    summary = native_summary_tensor(head, bins, target)
    public_summary = native_summary_tensor(model.output, bins, target)
    summary_model = tf.keras.Model(model.input, summary, name="relevant_state_summary")
    frontier_tensors = [model.get_layer(name).output for name in frontier_names]
    trace_model = tf.keras.Model(
        model.input,
        [*frontier_tensors, summary, public_summary],
        name="relevant_state_trace",
    )
    replay = FunctionalGraphReplay(summary_model).build(
        frontier_names, outputs=summary, name=f"replay__{cut}"
    )
    return trace_model, replay, frontier_names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/program1.json"))
    parser.add_argument("--routes", type=Path, default=Path("config/borzoi_routes.json"))
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--cut")
    parser.add_argument("--subset", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--minimum-span-bins", type=int)
    parser.add_argument("--maximum-evaluations", type=int)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import tensorflow as tf
    from pyfaidx import Fasta

    devices = tf.config.list_physical_devices("GPU")
    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)
    if args.require_gpu and not devices:
        raise RuntimeError("TensorFlow cannot see a GPU")

    config_path = resolve(args.config)
    routes_path = resolve(args.routes)
    config = json.loads(config_path.read_text())
    routes = json.loads(routes_path.read_text())
    relevant = config["relevant_state"]
    cut = args.cut or relevant["primary_cut"]
    if cut not in routes["cuts"]:
        raise KeyError(f"unknown graph cut: {cut}")
    cohort_path = resolve(args.cohort or relevant["cohort"])
    cohort_lock_path = resolve(relevant["cohort_lock"])
    cohort_lock = read_lock(cohort_lock_path)
    if sha256_file(cohort_path) != cohort_lock["cohort_sha256"]:
        raise RuntimeError("cohort does not match its immutable lock")
    cohort = selected_cohort(pd.read_csv(cohort_path, sep="\t"), args.subset)
    if not 0 <= args.task_index < len(cohort):
        raise ValueError(f"task index must lie in [0, {len(cohort)})")
    row = cohort.iloc[args.task_index]
    output_root = resolve(args.output_directory or relevant["output_directory"])
    task_root = output_root / cut / args.subset
    result_path = task_root / "results" / f"task-{args.task_index:03d}.json"
    manifest_path = task_root / "manifests" / f"task-{args.task_index:03d}.json"
    outputs = {"result": result_path}
    if validate_completed_shard(manifest_path, outputs):
        if not args.force:
            print(f"already_complete={manifest_path}", flush=True)
            return 0
        attempts = task_root / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        attempt_tag = sha256_file(result_path)[:12]
        archived_result = attempts / f"task-{args.task_index:03d}-{attempt_tag}.json"
        archived_manifest = attempts / f"task-{args.task_index:03d}-{attempt_tag}.manifest.json"
        if not archived_result.exists():
            shutil.copy2(result_path, archived_result)
            shutil.copy2(manifest_path, archived_manifest)

    minimum_span = int(args.minimum_span_bins or relevant["minimum_span_bins"])
    maximum_evaluations = int(
        args.maximum_evaluations or relevant["maximum_evaluations"]
    )
    model_path = resolve(config["model"]["primary"])
    model_sidecar = Path(str(model_path) + ".sha256")
    manifest = ShardManifest(
        task="program1_relevant_state",
        shard_index=args.task_index,
        num_shards=len(cohort),
        config_hash=stable_json_hash(
            {
                "config": config,
                "routes": routes,
                "cut": cut,
                "subset": args.subset,
                "minimum_span": minimum_span,
                "maximum_evaluations": maximum_evaluations,
            }
        ),
        input_hashes={
            "cohort": sha256_file(cohort_path),
            "cohort_lock": sha256_file(cohort_lock_path),
            "coverage_lock": sha256_file(
                ROOT / "artifacts/program1/preregistration/coverage_noise_v1.lock.json"
            ),
            "model_sidecar": sha256_file(model_sidecar),
        },
    )
    write_shard_manifest(manifest, manifest_path)
    started = time.time()
    base_payload = {
        "schema_version": "1",
        "task_index": args.task_index,
        "subset": args.subset,
        "cut": cut,
        "record_id": str(row["record_id"]),
        "variant_id": str(row["variant_id"]),
        "gene_id": str(row["gene_id"]),
        "split": str(row["split"]),
        "evidence_tier": str(row["evidence_tier"]),
        "model": str(model_path),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        target = target_spec_from_registry(
            resolve(config["model"]["targets"]),
            config["targets"]["whole_blood_rna_identifiers"],
            stride_bp=int(config["targets"]["native_stride_bp"]),
            crop_bp=int(config["targets"]["native_crop_bp"]),
        )
        model = load_model(config)
        fasta = Fasta(
            str(resolve(config["model"]["fasta"])),
            as_raw=True,
            sequence_always_upper=True,
        )
        chrom = str(row["chrom"])
        geometry = choose_joint_window(
            [int(row["position"]) - 1, int(row["tss"]) - 1],
            len(fasta[chrom]),
            input_length=int(routes["input_length"]),
            crop_bp=target.crop_bp,
        )
        if not geometry.eligible:
            raise ValueError(f"geometry:{geometry.reason}")
        sequence = str(fasta[chrom][geometry.window_start : geometry.window_end])
        pair = one_hot_allele_pair(
            sequence,
            int(row["position"]) - 1 - int(geometry.window_start),
            row["ref"],
            row["alt"],
        )
        bins = intervals_to_bins(
            parse_exon_intervals(row["exon_intervals"]),
            output_start_0based=int(geometry.window_start) + target.crop_bp,
            output_length=int(routes["native_output_length"]),
            stride_bp=target.stride_bp,
        )
        trace_model, replay, frontier_names = build_scalar_replay(
            model, routes, cut, target, bins
        )
        traced = trace_model(pair, training=False)
        states = [np.asarray(value) for value in traced[:-2]]
        natural = np.asarray(traced[-2], dtype=float)
        public_natural = np.asarray(traced[-1], dtype=float)
        head_public_absolute_error = float(
            np.max(np.abs(natural - public_natural), initial=0.0)
        )
        head_public_scale = max(float(np.max(np.abs(public_natural))), 1e-6)
        head_public_relative_error = head_public_absolute_error / head_public_scale
        if head_public_absolute_error > 1e-6 and head_public_relative_error > 1e-6:
            raise RuntimeError(
                "pre-switch output head differs from public inference output:"
                f"absolute={head_public_absolute_error:.8g},"
                f"relative={head_public_relative_error:.8g}"
            )
        natural_ref, natural_alt = map(float, natural)
        natural_delta = natural_alt - natural_ref
        noise_floor = float(row["noise_floor"])
        coverage_delta_error = abs(natural_delta - float(row["natural_delta"]))
        (
            coverage_delta_pass,
            coverage_delta_normalized_error,
            coverage_delta_sign_match,
            coverage_delta_above_noise,
        ) = coverage_delta_consistency(
            natural_delta,
            float(row["natural_delta"]),
            noise_floor,
            normalized_tolerance=0.01,
        )
        if not coverage_delta_pass:
            raise CoverageInstabilityError(
                {
                    "window_start": int(geometry.window_start),
                    "target_bins": bins.tolist(),
                    "natural_ref": natural_ref,
                    "natural_alt": natural_alt,
                    "natural_delta": natural_delta,
                    "coverage_delta": float(row["natural_delta"]),
                    "coverage_delta_absolute_error": coverage_delta_error,
                    "coverage_delta_normalized_error": coverage_delta_normalized_error,
                    "coverage_delta_sign_match": coverage_delta_sign_match,
                    "coverage_delta_above_noise": coverage_delta_above_noise,
                    "noise_floor": noise_floor,
                    "head_public_absolute_error": head_public_absolute_error,
                    "head_public_relative_error": head_public_relative_error,
                }
            )
        ref_states = [value[0:1] for value in states]
        alt_states = [value[1:2] for value in states]
        spatial_length = max(value.shape[-2] for value in states)
        cache: dict[tuple[int, ...], dict[str, object]] = {}

        def summarize(frontier: Sequence[np.ndarray]) -> float:
            return float(np.asarray(replay(list(frontier), training=False)).reshape(-1)[0])

        no_op_ref = FunctionalGraphReplay.verify(
            replay, ref_states, natural[0:1]
        )
        no_op_alt = FunctionalGraphReplay.verify(
            replay, alt_states, natural[1:2]
        )

        def evaluate_positions(positions: tuple[int, ...]) -> dict[str, object]:
            key = tuple(positions)
            if key in cache:
                return cache[key]
            patched_ref, mapped = patch_frontier_positions(
                ref_states, alt_states, key, spatial_length=spatial_length
            )
            patched_alt, _ = patch_frontier_positions(
                alt_states, ref_states, key, spatial_length=spatial_length
            )
            forward_output = summarize(patched_ref)
            reverse_output = summarize(patched_alt)
            sufficiency = forward_output - natural_ref
            necessity = natural_alt - reverse_output
            value = {
                "forward_output": forward_output,
                "reverse_output": reverse_output,
                "sufficiency_effect": sufficiency,
                "necessity_effect": necessity,
                "bidirectional_recovery": bidirectional_recovery(
                    sufficiency, necessity, natural_delta, noise_floor
                ),
                "mapped_positions": mapped,
            }
            cache[key] = value
            return value

        full_positions = tuple(range(spatial_length))
        full = evaluate_positions(full_positions)
        full_result = InterventionResult(
            intervention_id=f"full_{cut}",
            natural_ref=natural_ref,
            natural_alt=natural_alt,
            forward_output=float(full["forward_output"]),
            reverse_output=float(full["reverse_output"]),
        )
        full_forward_error, full_reverse_error = full_result.relative_errors(noise_floor)
        frontier_threshold = float(config["circuit_gates"]["exact_frontier_recovery"])
        relevant_threshold = float(config["circuit_gates"]["relevant_state_recovery"])
        full_frontier_pass = (
            no_op_ref.passed
            and no_op_alt.passed
            and full_forward_error <= 1 - frontier_threshold
            and full_reverse_error <= 1 - frontier_threshold
        )
        search_payload: dict[str, object] = {}
        status = "frontier_failure"
        if full_frontier_pass:
            search = hierarchical_state_search(
                spatial_length,
                lambda positions: float(
                    evaluate_positions(positions)["bidirectional_recovery"]
                ),
                1.0,
                recovery_threshold=relevant_threshold,
                minimum_span=minimum_span,
                noise_floor=1e-12,
                maximum_evaluations=maximum_evaluations,
            )
            selected = evaluate_positions(search.positions)
            selected_recovery = float(selected["bidirectional_recovery"])
            relevant_state_pass = (
                not search.budget_exhausted
                and selected_recovery >= relevant_threshold
            )
            status = "relevant_state_pass" if relevant_state_pass else "diffuse_effect"
            mapped = selected["mapped_positions"]
            search_payload = {
                "relevant_state_pass": relevant_state_pass,
                "search_budget_exhausted": search.budget_exhausted,
                "search_evaluations": search.evaluations,
                "spatial_length": spatial_length,
                "minimum_span_bins": minimum_span,
                "selected_position_count": len(search.positions),
                "selected_intervals": [list(value) for value in search.intervals],
                "selected_genomic_intervals": [
                    list(value)
                    for value in spatial_intervals_to_genomic(
                        search.intervals,
                        window_start=int(geometry.window_start),
                        input_length=int(routes["input_length"]),
                        spatial_length=spatial_length,
                    )
                ],
                "selected_layer_position_counts": {
                    name: len(layer_positions)
                    for name, layer_positions in zip(frontier_names, mapped)
                },
                "selected_layer_intervals": {
                    name: [list(value) for value in compress_positions(layer_positions)]
                    for name, layer_positions in zip(frontier_names, mapped)
                },
                "selected_forward_output": float(selected["forward_output"]),
                "selected_reverse_output": float(selected["reverse_output"]),
                "selected_sufficiency_effect": float(selected["sufficiency_effect"]),
                "selected_necessity_effect": float(selected["necessity_effect"]),
                "selected_bidirectional_recovery": selected_recovery,
            }
        payload = {
            **base_payload,
            "status": status,
            "window_start": int(geometry.window_start),
            "target_bins": bins.tolist(),
            "frontier_layers": list(frontier_names),
            "frontier_shapes": {
                name: list(value.shape)
                for name, value in zip(frontier_names, states)
            },
            "natural_ref": natural_ref,
            "natural_alt": natural_alt,
            "natural_delta": natural_delta,
            "head_public_absolute_error": head_public_absolute_error,
            "head_public_relative_error": head_public_relative_error,
            "coverage_delta": float(row["natural_delta"]),
            "coverage_delta_absolute_error": coverage_delta_error,
            "coverage_delta_normalized_error": coverage_delta_normalized_error,
            "coverage_delta_sign_match": coverage_delta_sign_match,
            "coverage_delta_above_noise": coverage_delta_above_noise,
            "noise_floor": noise_floor,
            "noop_ref": no_op_ref.__dict__,
            "noop_alt": no_op_alt.__dict__,
            "full_forward_output": full_result.forward_output,
            "full_reverse_output": full_result.reverse_output,
            "full_sufficiency_effect": full_result.sufficiency_effect,
            "full_necessity_effect": full_result.necessity_effect,
            "full_forward_recovery": 1.0 - full_forward_error,
            "full_reverse_recovery": 1.0 - full_reverse_error,
            "full_bidirectional_recovery": float(full["bidirectional_recovery"]),
            "full_frontier_pass": full_frontier_pass,
            **search_payload,
            "runtime_seconds": time.time() - started,
        }
    except CoverageInstabilityError as error:
        manifest.exclude("coverage_instability")
        payload = {
            **base_payload,
            "status": "coverage_instability",
            **error.details,
            "detail": str(error),
            "runtime_seconds": time.time() - started,
        }
    except Exception as error:  # task failures become inspectable, resumable artifacts
        manifest.exclude("runtime_failure")
        payload = {
            **base_payload,
            "status": "runtime_failure",
            "reason": type(error).__name__,
            "detail": str(error),
            "runtime_seconds": time.time() - started,
        }
    atomic_json(payload, result_path)
    manifest.records_completed = 1
    manifest.complete(outputs)
    write_shard_manifest(manifest, manifest_path)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
