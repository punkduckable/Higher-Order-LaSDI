import copy
import os
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC)

from HLaSDI.Schemas import validate_experiment_config


EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _load_example(name: str) -> dict:
    with open(EXAMPLES / name, "r") as f:
        return yaml.safe_load(f)


def test_all_example_configs_validate():
    for path in sorted(EXAMPLES.glob("*.yml")):
        with open(path, "r") as f:
            validated = validate_experiment_config(yaml.safe_load(f))
        assert validated.trainer.type
        assert validated.latent_dynamics.type
        assert validated.physics.type


def test_telegraphers_profile_config_captures_mature_rollout_in_short_run():
    config = validate_experiment_config(_load_example("Telegraphers_Profile.yml"))
    profiler = config.trainer.profiler
    rollout = config.trainer.Second_Order_Rollout

    assert config.trainer.max_iter == 4
    assert config.trainer.n_iter == 4
    assert profiler.enabled is True
    assert profiler.wait_epochs == 1
    assert profiler.warmup_epochs == 1
    assert profiler.active_epochs == 1
    assert profiler.num_repeats == 1
    assert rollout.p_rollout_init == rollout.max_p_rollout == 0.5


def test_schema_rejects_unknown_keys():
    config = _load_example("Burgers2D.yml")
    config["trainer"]["typo_learning_rate"] = 1.0

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        validate_experiment_config(config)


def test_schema_rejects_nonpositive_learning_rate():
    config = _load_example("Burgers2D.yml")
    config["trainer"]["First_Order_Rollout"]["lr"] = 0.0

    with pytest.raises(ValidationError, match="greater than 0"):
        validate_experiment_config(config)


def test_schema_rejects_missing_explicit_noise_ratio():
    config = _load_example("Burgers2D.yml")
    del config["trainer"]["noise_ratio"]

    with pytest.raises(ValidationError, match="Field required"):
        validate_experiment_config(config)


def test_schema_rejects_missing_interpolator_config():
    config = _load_example("Burgers2D.yml")
    del config["latent_dynamics"]["interpolator"]

    with pytest.raises(ValidationError, match="Field required"):
        validate_experiment_config(config)


def test_schema_accepts_rbf_interpolator_kernel():
    config = _load_example("Burgers2D.yml")
    kernel = config["latent_dynamics"]["interpolator"]["GP"]["kernel"]
    kernel["type"] = "RBF"
    del kernel["nu"]

    validated = validate_experiment_config(config)

    assert validated.latent_dynamics.interpolator.GP.kernel.type == "RBF"


def test_schema_rejects_incompatible_trainer_and_latent_dynamics_order():
    config = _load_example("Burgers2D.yml")
    config["trainer"]["type"] = "Second_Order_Rollout"
    config["trainer"]["Second_Order_Rollout"] = copy.deepcopy(
        config["trainer"].pop("First_Order_Rollout")
    )
    config["trainer"]["Second_Order_Rollout"]["loss_weights"]["chain_rule"] = 1.0
    config["trainer"]["Second_Order_Rollout"]["loss_weights"]["consistency"] = 1.0
    config["trainer"]["Second_Order_Rollout"]["loss_types"]["chain_rule"] = "MSE"
    config["trainer"]["Second_Order_Rollout"]["loss_types"]["consistency"] = "MSE"

    with pytest.raises(ValidationError, match="requires n_IC=2"):
        validate_experiment_config(config)
