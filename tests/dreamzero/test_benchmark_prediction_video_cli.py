# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pathlib import Path

import pytest

from examples.offline_inference.dreamzero import benchmark_prediction_video


pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_parse_args_defaults_to_two_generate_requests() -> None:
    args = benchmark_prediction_video.parse_args(
        [
            "--deploy-config",
            "vllm_omni/deploy/dreamzero_tp1_cfg2.yaml",
        ]
    )

    assert args.model == "GEAR-Dreams/DreamZero-DROID"
    assert args.deploy_config == Path("vllm_omni/deploy/dreamzero_tp1_cfg2.yaml")
    assert args.num_requests == 2
    assert args.output_dir == Path("outputs/dreamzero/benchmark")
    assert args.output_stem == "dreamzero_benchmark"
    assert args.fps == 5
    assert args.accuracy_atol == pytest.approx(1e-3)


def test_parse_args_rejects_non_positive_request_count() -> None:
    with pytest.raises(SystemExit):
        benchmark_prediction_video.parse_args(
            [
                "--deploy-config",
                "vllm_omni/deploy/dreamzero.yaml",
                "--num-requests",
                "0",
            ]
        )
