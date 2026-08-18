import os
import sys

import yaml

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(SRC)

from HLaSDI.Physics.Explicit import Explicit
from HLaSDI.Schemas import ExplicitPhysicsConfig, validate_experiment_config


def test_physics_load_restores_validated_config_object():
    config_path = os.path.join(os.path.dirname(__file__), "..", "examples", "Explicit.yml")
    with open(config_path, "r") as f:
        experiment_config = validate_experiment_config(yaml.safe_load(f))

    physics = Explicit(experiment_config.physics, ["A", "w"])
    state = physics.export()

    assert isinstance(state["config"], dict)

    physics.load(state)

    assert isinstance(physics.config, ExplicitPhysicsConfig)
    assert physics.config.type == "Explicit"
    assert physics.config.Explicit.n_t == experiment_config.physics.Explicit.n_t


def test_physics_load_accepts_already_validated_config_object():
    config_path = os.path.join(os.path.dirname(__file__), "..", "examples", "Explicit.yml")
    with open(config_path, "r") as f:
        experiment_config = validate_experiment_config(yaml.safe_load(f))

    physics = Explicit(experiment_config.physics, ["A", "w"])
    state = physics.export()
    state["config"] = experiment_config.physics

    physics.load(state)

    assert isinstance(physics.config, ExplicitPhysicsConfig)
    assert physics.config.type == "Explicit"
