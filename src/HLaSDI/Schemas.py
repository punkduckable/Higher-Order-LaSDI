"""Validated configuration schemas for Higher-Order LaSDI experiments.

The schemas intentionally mirror the existing YAML shape used by the codebase: each top-level
section has a ``type`` discriminator and a nested subclass-specific section with the same key used
by the corresponding implementation. Unknown fields are rejected so misspelled YAML keys fail before
any training, sampling, or physics code runs.
"""

from    __future__  import  annotations

from    typing      import  Annotated, Any, Literal

from    pydantic    import  BaseModel, ConfigDict, Field, TypeAdapter, model_validator


# -------------------------------------------------------------------------------------------------
# General helpers
# -------------------------------------------------------------------------------------------------


class ConfigBase(BaseModel):
    """Base class for all validated config schemas."""

    model_config = ConfigDict(extra = "forbid");



# Some commonly used constrained datatypes
PositiveInt         = Annotated[int,   Field(ge = 1)]
NonNegativeInt      = Annotated[int,   Field(ge = 0)]
PositiveFloat       = Annotated[float, Field(gt = 0.0)]
NonNegativeFloat    = Annotated[float, Field(ge = 0.0)]
Probability         = Annotated[float, Field(ge = 0.0, le = 1.0)]
ActivationSpec      = str | list[str]
Conv3DParam         = int | tuple[int, int, int] | list[int] | list[tuple[int, int, int]] | list[list[int]]
LossType            = Literal["MSE", "MAE"]

_ALLOWED_ACTIVATIONS = {
    "elu",
    "hardshrink",
    "hardsigmoid",
    "hardtanh",
    "hardswish",
    "leakyrelu",
    "logsigmoid",
    "relu",
    "relu6",
    "rrelu",
    "selu",
    "celu",
    "sin",
    "cos",
    "gelu",
    "sigmoid",
    "silu",
    "mish",
    "softplus",
    "softshrink",
    "tanh",
    "tanhshrink",
}


def _check_activation_spec(value: ActivationSpec, *, n_layers: int, field_name: str) -> None:
    """Validate an activation string or per-layer activation list."""

    activations = [value] * n_layers if isinstance(value, str) else value
    if len(activations) != n_layers:
        raise ValueError(
            f"{field_name} must contain one activation per hidden layer. "
            f"Got {len(activations)} activations for {n_layers} layers."
        )
    for activation in activations:
        if activation.lower() not in _ALLOWED_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation {activation!r}. Allowed activations are "
                f"{sorted(_ALLOWED_ACTIVATIONS)}."
            )


def _check_min_less_than_max(min_value: float, max_value: float, *, name: str) -> None:
    if min_value >= max_value:
        raise ValueError(f"{name}.min must be smaller than {name}.max.")


def _check_sequence_3tuple_param(value: Any, *, n_layers: int, field_name: str) -> None:
    """Validate Conv3d scalar/3-tuple/per-layer 3-tuple configuration."""

    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{field_name} must be positive.")
        return

    if not isinstance(value, list | tuple) or len(value) == 0:
        raise ValueError(f"{field_name} must be an int, 3-tuple, or per-layer list.")

    # Shared 3-tuple case.
    if len(value) == 3 and all(isinstance(x, int) for x in value):
        if any(x <= 0 for x in value):
            raise ValueError(f"{field_name} entries must be positive.")
        return

    # Per-layer sequence of ints or 3-tuples.
    if len(value) != n_layers:
        raise ValueError(
            f"{field_name} must have length {n_layers} when specified per layer; "
            f"got length {len(value)}."
        )
    for entry in value:
        if isinstance(entry, int):
            if entry <= 0:
                raise ValueError(f"{field_name} entries must be positive.")
        elif isinstance(entry, list | tuple) and len(entry) == 3 and all(isinstance(x, int) for x in entry):
            if any(x <= 0 for x in entry):
                raise ValueError(f"{field_name} entries must be positive.")
        else:
            raise ValueError(f"{field_name} entries must be positive ints or positive 3-tuples.")


# -------------------------------------------------------------------------------------------------
# Workflow schemas
# -------------------------------------------------------------------------------------------------


class WorkflowConfig(ConfigBase):
    """Workflow/restart settings."""

    # Should we load the encoder/decoder/LD/optimizer from file or train from scratch?
    use_restart: bool

    # If using a restart, this is name of the file in `results/` holding the serialized state. 
    restart_file: str | None

    @model_validator(mode = "after")
    def validate_restart_file(self) -> "WorkflowConfig":
        if self.use_restart and not self.restart_file:
            raise ValueError("workflow.use_restart = true requires workflow.restart_file.")
        return self


# -------------------------------------------------------------------------------------------------
# Parameter-space schemas
# -------------------------------------------------------------------------------------------------


class BaseParameterConfig(ConfigBase):
    """Base config for one scalar parameter."""

    # Variable name
    name: str = Field(min_length = 1)

    # How we specify the allowed variables.
    test_space_type: str


class UniformParameterConfig(BaseParameterConfig):
    """Parameter sampled by a linear/log-uniform grid."""

    test_space_type: Literal["uniform"]

    # Minimum of parameter range
    min: float

    # Maximum of parameter range
    max: float

    # Number of values in parameter range.
    sample_size: PositiveInt

    # If true, the logs of the values are spread uniformly (from log(min) to log(max)).
    log_scale: bool

    @model_validator(mode = "after")
    def validate_bounds(self) -> "UniformParameterConfig":
        # Make sure the min < max
        _check_min_less_than_max(self.min, self.max, name = f"parameter {self.name!r}")

        # If using log scale, the range needs to be positive. This is automatically true if the 
        # min is positive.
        if self.log_scale and self.min <= 0.0:
            raise ValueError(f"parameter {self.name!r} uses log_scale = true, so min must be > 0.")

        return self


class ListParameterConfig(BaseParameterConfig):
    """Parameter sampled from an explicit list."""

    test_space_type: Literal["list"]

    # Specific values to use for this parameter.
    value_list: list[float] = Field(alias = "list", min_length = 1)


class FileParameterConfig(BaseParameterConfig):
    """Parameter sampled from a metadata file."""

    test_space_type: Literal["file"]

    # File housing the parameter values.
    file: str = Field(min_length = 1)


ParameterConfig = Annotated[
    UniformParameterConfig | ListParameterConfig | FileParameterConfig,
    Field(discriminator = "test_space_type"),
]


class GridTestSpaceConfig(ConfigBase):
    """Cartesian-product testing space."""

    type: Literal["grid"]


class ParameterSpaceConfig(ConfigBase):
    """Top-level parameter-space configuration."""

    # List holding the parameters
    parameters: list[ParameterConfig] = Field(min_length = 1)

    # How we specify which parameter values are in the testing space.
    test_space: GridTestSpaceConfig

    @model_validator(mode = "after")
    def validate_unique_names(self) -> "ParameterSpaceConfig":

        # Make sure there are no duplicated names (set of names has same length as list of names)
        names = [param.name for param in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(f"parameter names must be unique; got {names}.")
        return self


# -------------------------------------------------------------------------------------------------
# Encoder/decoder schemas
# -------------------------------------------------------------------------------------------------


class DenseAutoencoderSettings(ConfigBase):
    """Fully-connected autoencoder settings shared by first-/second-order dense models."""

    # How many latent space -> FOM space models should our `decoder` contain (for mLaSDI)?
    n_Decoders: PositiveInt

    # Hidden widths of the encoder (input dim implicit from FOM dimension, latent dim specified 
    # elsewhere). Note that the decoder hidden widths are the same but in reverse order.
    hidden_widths: list[PositiveInt] = Field(min_length = 1)

    # Latent dimension of the model.
    latent_dimension: PositiveInt

    # Activations to use; the i'th activation is applied just after the i'th layer. Note
    # that there is no activation after the final layer (to latent space).
    activations: ActivationSpec

    @model_validator(mode = "after")
    def validate_activations(self) -> "DenseAutoencoderSettings":
        # Ensure the number of activations matches the number of hidden layers.
        _check_activation_spec(
            self.activations,
            n_layers    = len(self.hidden_widths),
            field_name  = "activations",
        )
        return self


class CNN3DAutoencoderSettings(ConfigBase):
    """3D convolutional autoencoder settings."""

    # How many latent space -> FOM space models should our `decoder` contain (for mLaSDI)?
    n_Decoders: PositiveInt

    # Hidden widths of the fully connected portion of the encoder. Note that the decoder hidden 
    # widths are the same but in reverse order.
    hidden_widths_fc: list[PositiveInt] = Field(min_length = 1)

    # Activations to apply after each FC layer.
    activations_fc: ActivationSpec

    # Latent dimension of the model.
    latent_dimension: PositiveInt

    # List holding the number of convolutional channels. i'th element specifies the number of CNN
    # channels after the i-1'th convolution.
    conv_channels: list[PositiveInt] = Field(min_length = 2)

    # Kernel sizes, strides, paddings, and activations. Each can be an int (use as the 
    # height/width/depth in each layer) a 3-tuple of ints (specify height/width/depth spec used in 
    # each layer), a list of ints (i'th element used as height/width/depth for i'th layer), or list
    # of 3-tuples of ints (specify exact height/width/depth in each layer).
    conv_kernel_sizes   : Conv3DParam
    conv_strides        : Conv3DParam
    conv_paddings       : Conv3DParam
    conv_activations    : ActivationSpec

    @model_validator(mode = "after")
    def validate_cnn_settings(self) -> "CNN3DAutoencoderSettings":
        _check_activation_spec(
            self.activations_fc,
            n_layers    = len(self.hidden_widths_fc),
            field_name  = "activations_fc",
        )
        n_conv_layers = len(self.conv_channels) - 1
        _check_activation_spec(
            self.conv_activations,
            n_layers    = n_conv_layers,
            field_name  = "conv_activations",
        )
        _check_sequence_3tuple_param(
            self.conv_kernel_sizes,
            n_layers    = n_conv_layers,
            field_name  = "conv_kernel_sizes",
        )
        _check_sequence_3tuple_param(
            self.conv_strides,
            n_layers    = n_conv_layers,
            field_name  = "conv_strides",
        )
        _check_sequence_3tuple_param(
            self.conv_paddings,
            n_layers    = n_conv_layers,
            field_name  = "conv_paddings",
        )
        return self


class BaseEncoderDecoderConfig(ConfigBase):
    """Base encoder/decoder config."""

    type: str

    # Should the encoder/decoder parameters be learned during training?
    trainable: bool


class AEEncoderDecoderConfig(BaseEncoderDecoderConfig):
    type                : Literal["ae"]
    ae                  : DenseAutoencoderSettings


class AutoencoderEncoderDecoderConfig(BaseEncoderDecoderConfig):
    type                : Literal["autoencoder"]
    autoencoder         : DenseAutoencoderSettings


class PairEncoderDecoderConfig(BaseEncoderDecoderConfig):
    type                : Literal["pair"]
    pair                : DenseAutoencoderSettings


class AutoencoderPairEncoderDecoderConfig(BaseEncoderDecoderConfig):
    type                : Literal["autoencoder_pair"]
    autoencoder_pair    : DenseAutoencoderSettings


class CNN3DEncoderDecoderConfig(BaseEncoderDecoderConfig):
    type                : Literal["cnn_3d"]
    cnn_3d              : CNN3DAutoencoderSettings


class CNN3DAEEncoderDecoderConfig(BaseEncoderDecoderConfig):
    type                : Literal["cnn_3d_ae"]
    cnn_3d_ae           : CNN3DAutoencoderSettings


class CNN3DAutoencoderEncoderDecoderConfig(BaseEncoderDecoderConfig):
    type                : Literal["cnn_3d_autoencoder"]
    cnn_3d_autoencoder  : CNN3DAutoencoderSettings


EncoderDecoderConfig = Annotated[
    AEEncoderDecoderConfig
    | AutoencoderEncoderDecoderConfig
    | PairEncoderDecoderConfig
    | AutoencoderPairEncoderDecoderConfig
    | CNN3DEncoderDecoderConfig
    | CNN3DAEEncoderDecoderConfig
    | CNN3DAutoencoderEncoderDecoderConfig,
    Field(discriminator = "type"),
]


# -------------------------------------------------------------------------------------------------
# Latent-dynamics schemas
# -------------------------------------------------------------------------------------------------


class LatentDynamicsBaseConfig(ConfigBase):
    """Base latent-dynamics config shared by currently implemented subclasses."""

    type: str

    # How should we determine the latent dynamics at testing parameter combinations?
    interpolator_type: Literal["GP"]

    # Should we learn the latent coefficients during training, or keep them fixed?
    trainable: bool

    # Which losses are computed by the latent dynamics and what are their weights? Note that
    # compute_losses MUST return scalar losses whose keys match the losses you list here.
    loss_weights: dict[str, NonNegativeFloat]

    @model_validator(mode = "after")
    def validate_loss_weights(self) -> "LatentDynamicsBaseConfig":
        expected_keys_by_type : dict[str, set[str]] = {
            "sindy"    : {"LD", "coef", "stab"},
            "sindy_w"  : {"LD", "coef", "stab"},
            "spring"   : {"LD", "coef", "stab"},
            "spring_w" : {"LD", "coef", "stab"},
            "switch"   : {"LD", "coef", "stab"},
            "switch_w" : {"LD", "coef", "stab"},
            "cable"    : {"LD", "coef", "diversity", "tail"},
        };

        expected_keys = expected_keys_by_type.get(self.type);
        if expected_keys is not None and set(self.loss_weights.keys()) != expected_keys:
            raise ValueError("latent_dynamics.loss_weights for type `%s` must include exactly %s. Got %s." % (
                self.type,
                sorted(expected_keys),
                sorted(self.loss_weights.keys()),
            ));
        return self;


class InterpolatableLatentDynamicsSettings(ConfigBase):
    """Latent dynamics settings for strong-form least-squares coefficient initialization."""

    # What L2 regularization penalty should we apply when solving for initial coefficients?
    lstsq_reg: NonNegativeFloat


class WeakLatentDynamicsSettings(ConfigBase):
    """Weak-form latent dynamics test-function settings."""

    # Should the test functions be polynomials or bumps? If polynomial, the polynomial order 
    # is always 2 more than n_IC.
    test_func_type: Literal["bump", "PC-poly"]

    # How wide should the test functions be?
    test_func_width: PositiveFloat

    # How much overlap (as a proportion of width) should there be between successive 
    # bumps?
    overlap: Annotated[float, Field(ge = 0.0, lt = 1.0)]


class CABLELatentDynamicsSettings(ConfigBase):
    """CABLE mixture-of-experts latent dynamics settings."""

    # How many experts should we use?
    n_experts: PositiveInt 

    # Roughly how many experts do we want to be active each step? CABLE imposes a series of soft
    # penalties to concentrate all weight in <= n_active experts at each time/parameter. Must 
    # also be <= n_experts
    n_active : PositiveInt 

    # Hidden widths of the gate network (input dim is 1 + n_param and the output is n_experts).
    hidden_widths: list[PositiveInt] = Field(min_length = 1)

    # Activations to use; the i'th activation is applied just after the i'th layer. Note that
    # we use a soft-max on the final layer.
    activations: ActivationSpec

    # Should each expert include a bias?
    use_biases : bool

    # Which (vector) norm should we use for the coefficient loss? This is applied to the flattened
    # expert coefficients for each expert.
    coef_norm : Literal['l1', 'l2']

    # Should we periodically mask out coefficients that get too small? If so, once a coefficient
    # is masked, it will never be unmasked.
    use_mask : bool

    # If masking is enabled, below what threshold should we mask out a value?
    mask_threshold : PositiveFloat | None = None

    # If masking is enabled, when should we start computing the mask?
    first_mask_step : PositiveInt | None = None

    # If masking is enabled, how frequently should we apply it after enabling it?
    mask_update_freq : PositiveInt | None = None  

    @model_validator(mode = "after")
    def validate_activations_and_active_count(self) -> "CABLELatentDynamicsSettings":
        # Ensure the number of activations matches the number of hidden layers.
        _check_activation_spec(
            self.activations,
            n_layers    = len(self.hidden_widths),
            field_name  = "activations",
        )

        # Check that n_active <= n_experts
        if self.n_active > self.n_experts:
            raise ValueError("n_active = %d, but n_experts = %d; the target number of active experts can not exceed the number of experts" % (self.n_active, self.n_experts));

        # If masking is enabled, make sure `mask_threshold`, `first_mask_step`, and 
        # `mask_update_freq` are enabled
        if self.use_mask:
            if self.mask_threshold is None:
                raise ValueError("self.mask_threshold is None, even though masking is enabled.");
            if self.first_mask_step is None:
                raise ValueError("self.first_mask_step is None, even though masking is enabled.");
            if self.mask_update_freq is None:
                raise ValueError("self.mask_update_freq is None, even though masking is enabled.");

        # All done :) 
        return self;

class SINDyLatentDynamicsConfig(LatentDynamicsBaseConfig):
    type        : Literal["sindy"]
    sindy       : InterpolatableLatentDynamicsSettings


class SINDyWeakLatentDynamicsConfig(LatentDynamicsBaseConfig):
    type        : Literal["sindy_w"]
    sindy_w     : WeakLatentDynamicsSettings


class DampedSpringLatentDynamicsConfig(LatentDynamicsBaseConfig):
    type        : Literal["spring"]
    spring      : InterpolatableLatentDynamicsSettings


class DampedSpringWeakLatentDynamicsConfig(LatentDynamicsBaseConfig):
    type        : Literal["spring_w"]
    spring_w    : WeakLatentDynamicsSettings


class SwitchSINDyLatentDynamicsConfig(LatentDynamicsBaseConfig):
    type        : Literal["switch"]
    switch      : InterpolatableLatentDynamicsSettings


class SwitchSINDyWeakLatentDynamicsConfig(LatentDynamicsBaseConfig):
    type        : Literal["switch_w"]
    switch_w    : WeakLatentDynamicsSettings


class CABLELatentDynamicsConfig(LatentDynamicsBaseConfig):
    type        : Literal["cable"]
    cable       : CABLELatentDynamicsSettings


class WeakCABLELatentDynamicsConfig(LatentDynamicsBaseConfig):
    type        : Literal["cable_w"]
    cable       : CABLELatentDynamicsSettings
    weak        : WeakLatentDynamicsSettings


LatentDynamicsConfig = Annotated[
    SINDyLatentDynamicsConfig
    | SINDyWeakLatentDynamicsConfig
    | DampedSpringLatentDynamicsConfig
    | DampedSpringWeakLatentDynamicsConfig
    | SwitchSINDyLatentDynamicsConfig
    | SwitchSINDyWeakLatentDynamicsConfig
    | CABLELatentDynamicsConfig
    | WeakCABLELatentDynamicsConfig,
    Field(discriminator = "type"),
]


# -------------------------------------------------------------------------------------------------
# Physics schemas
# -------------------------------------------------------------------------------------------------


class TimedPhysicsSettings(ConfigBase):
    """Common time-grid settings used by several external/MFEM physics wrappers."""

    # Number of time steps
    n_t: PositiveInt

    # Maximum time value (minimum is 0)
    t_max: PositiveFloat

    # Are time steps uniformly spaced?
    uniform_t_grid: bool


class BurgersSettings(TimedPhysicsSettings):
    """1D Burgers spatial/time settings."""

    # Number of grid points along the spatial axis in each FOM frame.
    n_x: Annotated[int, Field(ge = 2)]

    # Minimum, maximum values of spatial variables.
    x_min: float
    x_max: float

    # The maximum number of corrections we are willing to make at each time step when solving
    # Burgers equation.
    maxk: PositiveInt

    # the maximum allowed relative residual at each time step when solving Burgers equation.
    convergence_threshold: PositiveFloat

    @model_validator(mode = "after")
    def validate_domain(self) -> "BurgersSettings":
        _check_min_less_than_max(self.x_min, self.x_max, name = "Burgers.x")
        return self


class Burgers2DSettings(TimedPhysicsSettings):
    """2D Burgers spatial/time settings."""

    # Number of grid points along the x axis in each FOM frame.
    n_x: Annotated[int, Field(ge = 2)]

    # Minimum, maximum x value in each FOM frame.
    x_min: float
    x_max: float

    # Number of grid points along the y axis in each FOM frame.
    n_y: Annotated[int, Field(ge = 2)]

    # Minimum, maximum y value in each FOM frame.
    y_min: float
    y_max: float

    # The frequency of sinusoids in the initial condition 
    # u(0, (x, y)) exp(-k (x^2 + y^2)) * sin(pi * w * x) * sin(pi * w * y)
    w: float

    @model_validator(mode = "after")
    def validate_domain(self) -> "Burgers2DSettings":
        _check_min_less_than_max(self.x_min, self.x_max, name = "Burgers2D.x")
        _check_min_less_than_max(self.y_min, self.y_max, name = "Burgers2D.y")
        return self


class ExplicitSettings(TimedPhysicsSettings):
    """Explicit analytic-physics settings."""

    # Number of points along the x and y axes in the spatial portion of the solution.
    n_positions: PositiveInt

    # Minimum and maximum x/y values in each FOM frame; we define an n_positions x n_positions 
    # grid of points over the rectangle defined by these setting.s
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @model_validator(mode = "after")
    def validate_domain(self) -> "ExplicitSettings":
        _check_min_less_than_max(self.x_min, self.x_max, name = "Explicit.x")
        _check_min_less_than_max(self.y_min, self.y_max, name = "Explicit.y")
        return self


class ThermalSettings(ConfigBase):
    """Thermal hdf5-dataset physics settings."""

    # Directory housing the metadata and simulation-specific hdf5 files. 
    hdf5_dir: str = Field(min_length = 1)

    # Should we ensure the data in each frame lies on a 3D grid? We need to ensure this holds if we 
    # are using a CNN for the encoder/decoder.
    use_cnn: bool

    # Animations will only plot nodes whose temperature lies above this value.
    threshold: float


class BasePhysicsConfig(ConfigBase):
    type: str


class BurgersPhysicsConfig(BasePhysicsConfig):
    type                : Literal["Burgers"]
    Burgers             : BurgersSettings


class BurgersSecondOrderPhysicsConfig(BasePhysicsConfig):
    type                : Literal["BurgersSecondOrder"]
    Burgers             : BurgersSettings


class Burgers2DPhysicsConfig(BasePhysicsConfig):
    type                : Literal["Burgers2D"]
    Burgers2D           : Burgers2DSettings


class ExplicitPhysicsConfig(BasePhysicsConfig):
    type                : Literal["Explicit"]
    Explicit            : ExplicitSettings


class ExplicitSecondOrderPhysicsConfig(BasePhysicsConfig):
    type                : Literal["ExplicitSecondOrder"]
    Explicit            : ExplicitSettings


class ThermalPhysicsConfig(BasePhysicsConfig):
    type                : Literal["Thermal"]
    Thermal             : ThermalSettings


class AdvectionPhysicsConfig(BasePhysicsConfig):
    type                : Literal["Advection"]
    Advection           : TimedPhysicsSettings


class KleinGordonPhysicsConfig(BasePhysicsConfig):
    type                : Literal["KleinGordon"]
    KleinGordon         : TimedPhysicsSettings


class NonlinearElasticityPhysicsConfig(BasePhysicsConfig):
    type                : Literal["NonlinearElasticity"]
    NonlinearElasticity : TimedPhysicsSettings


class TelegraphersPhysicsConfig(BasePhysicsConfig):
    type                : Literal["Telegraphers"]
    Telegraphers        : TimedPhysicsSettings


class WaveEquationPhysicsConfig(BasePhysicsConfig):
    type                : Literal["WaveEquation"]
    WaveEquation        : TimedPhysicsSettings


PhysicsConfig = Annotated[
    BurgersPhysicsConfig
    | BurgersSecondOrderPhysicsConfig
    | Burgers2DPhysicsConfig
    | ExplicitPhysicsConfig
    | ExplicitSecondOrderPhysicsConfig
    | ThermalPhysicsConfig
    | AdvectionPhysicsConfig
    | KleinGordonPhysicsConfig
    | NonlinearElasticityPhysicsConfig
    | TelegraphersPhysicsConfig
    | WaveEquationPhysicsConfig,
    Field(discriminator = "type"),
]


# -------------------------------------------------------------------------------------------------
# Sampler schemas
# -------------------------------------------------------------------------------------------------


class BaseSamplerConfig(ConfigBase):
    type: str


class FOMVarianceSettings(ConfigBase):
    # How many samples do should we draw for each testing parameter combination? Each sample 
    # results in a set of latent dynamics for that testing value; we solve it/decode. Doing this 
    # for each sample gives us an empirical distribution for the FOM variance for this parameter 
    # value.
    n_samples: PositiveInt


class FOMRolloutSettings(ConfigBase):
    # Should we draw samples of the latent dynamics, or use the mean? Note that if our LD model 
    # is not stochastic, may not be supported (at the very best, it will do nothing).
    sample_test_LD: bool

    # How many samples should we draw for each testing parameter combination? Only required 
    # if `sample_test_LD = True`. Ignored when `sample_test_LD = False`.
    n_samples: PositiveInt | None = None

    # If True, compute errors in normalized units, if False, compute errors in physical 
    # units (requires trainer normalization stats).
    normalized_FOM: bool

    # How should we normalize the error?
    error_normalization: Literal["none", "global_std", "trajectory_std"]

    # Small divisor for STD normalization to ensure we don't divide by zero.
    eps: PositiveFloat

    @model_validator(mode = "after")
    def validate_sampling(self) -> "FOMRolloutSettings":
        if self.sample_test_LD and self.n_samples is None:
            raise ValueError("n_samples must be set if sample_test_LD = True.");
        return self


class FOMVarianceSamplerConfig(BaseSamplerConfig):
    type            : Literal["FOM_Variance"]
    FOM_Variance    : FOMVarianceSettings


class FOMRolloutSamplerConfig(BaseSamplerConfig):
    type            : Literal["FOM_Rollout"]
    FOM_Rollout     : FOMRolloutSettings


class ROMDiscrepancySamplerConfig(BaseSamplerConfig):
    type            : Literal["ROM_Discrepancy"]


SamplerConfig = Annotated[
    FOMVarianceSamplerConfig | FOMRolloutSamplerConfig | ROMDiscrepancySamplerConfig,
    Field(discriminator = "type"),
]


# -------------------------------------------------------------------------------------------------
# Trainer schemas
# -------------------------------------------------------------------------------------------------


class FirstOrderLossWeights(ConfigBase):
    # Weight applied to the frame-wise reconstruction loss.
    recon: NonNegativeFloat

    # Weight applied to rollout loss from sampled rollout start frames.
    rollout: NonNegativeFloat

    # Weight applied to rollout loss from initial-condition frames.
    IC_rollout: NonNegativeFloat


class SecondOrderLossWeights(FirstOrderLossWeights):
    # Weight applied to the decoder/latent chain-rule consistency loss.
    chain_rule: NonNegativeFloat

    # Weight applied to position/velocity latent consistency loss.
    consistency: NonNegativeFloat


class FirstOrderLossTypes(ConfigBase):
    # Reconstruction loss function.
    recon: LossType

    # Rollout loss function.
    rollout: LossType

    # Initial-condition rollout loss function.
    IC_rollout: LossType


class SecondOrderLossTypes(FirstOrderLossTypes):
    # Chain-rule loss function.
    chain_rule: LossType

    # Position/velocity latent consistency loss function.
    consistency: LossType


class RolloutTrainerSettings(ConfigBase):
    """Shared hyperparameters for rollout/weak trainers."""

    # Optimizer learning rate.
    lr: PositiveFloat

    # Maximum gradient norm/magnitude used for gradient clipping.
    gradient_clip: PositiveFloat

    # Initial fraction of each trajectory used for rollout loss.
    p_rollout_init: Probability

    # Number of training epochs/iterations between rollout-curriculum updates.
    rollout_update_freq: PositiveInt

    # Amount added to p_rollout at each rollout-curriculum update.
    dp_per_update: PositiveFloat

    # Maximum fraction of each trajectory used for rollout loss.
    max_p_rollout: Probability

    # Number of rollout start frames sampled per training parameter per epoch.
    n_rollouts: PositiveInt

    # Initial fraction of each trajectory used for initial-condition rollout loss.
    p_IC_rollout_init: Probability

    # Number of training epochs/iterations between IC-rollout curriculum updates.
    IC_rollout_update_freq: PositiveInt

    # Amount added to p_IC_rollout at each IC-rollout curriculum update.
    IC_dp_per_update: PositiveFloat

    # Maximum fraction of each trajectory used for initial-condition rollout loss.
    max_p_IC_rollout: Probability

    # Number of warmup epochs/iterations after greedy sampling before checkpointing resumes.
    warmup_epochs: NonNegativeInt

    @model_validator(mode = "after")
    def validate_curriculum_bounds(self) -> "RolloutTrainerSettings":
        if self.p_rollout_init > self.max_p_rollout:
            raise ValueError("p_rollout_init must be <= max_p_rollout.")
        if self.p_IC_rollout_init > self.max_p_IC_rollout:
            raise ValueError("p_IC_rollout_init must be <= max_p_IC_rollout.")
        return self


class FirstOrderRolloutSettings(RolloutTrainerSettings):
    # Weights for all first-order trainer loss terms.
    loss_weights: FirstOrderLossWeights

    # Loss functions for all first-order trainer loss terms.
    loss_types: FirstOrderLossTypes


class SecondOrderRolloutSettings(RolloutTrainerSettings):
    # Weights for all second-order trainer loss terms.
    loss_weights: SecondOrderLossWeights

    # Loss functions for all second-order trainer loss terms.
    loss_types: SecondOrderLossTypes


class BaseTrainerConfig(ConfigBase):
    """Base trainer configuration."""

    # Trainer implementation selected by the discriminator.
    type: str

    # Total global training-iteration budget.
    max_iter: PositiveInt

    # Number of optimizer iterations in one train/greedy-sampling round.
    n_iter: PositiveInt

    # Last global iteration at which greedy sampling may add new training points.
    max_greedy_iter: PositiveInt

    # If true, normalize all generated FOM trajectories using training-data statistics.
    normalize: bool

    # Device string used for PyTorch training, e.g. "cpu", "cuda", "cuda:0", or "mps".
    device: str = Field(min_length = 1)

    # Ratio of Gaussian noise standard deviation to signal RMS for training-data noise injection.
    noise_ratio: NonNegativeFloat

    @model_validator(mode = "after")
    def validate_iteration_bounds(self) -> "BaseTrainerConfig":
        if self.n_iter > self.max_iter:
            raise ValueError("trainer.n_iter must be <= trainer.max_iter.")
        if self.max_greedy_iter > self.max_iter:
            raise ValueError("trainer.max_greedy_iter must be <= trainer.max_iter.")
        return self


class FirstOrderRolloutTrainerConfig(BaseTrainerConfig):
    type                    : Literal["First_Order_Rollout"]
    First_Order_Rollout     : FirstOrderRolloutSettings


class FirstOrderWeakTrainerConfig(BaseTrainerConfig):
    type                    : Literal["First_Order_Weak"]
    First_Order_Weak        : FirstOrderRolloutSettings


class SecondOrderRolloutTrainerConfig(BaseTrainerConfig):
    type                    : Literal["Second_Order_Rollout"]
    Second_Order_Rollout    : SecondOrderRolloutSettings


class SecondOrderWeakTrainerConfig(BaseTrainerConfig):
    type                    : Literal["Second_Order_Weak"]
    Second_Order_Weak       : SecondOrderRolloutSettings


TrainerConfig = Annotated[
    FirstOrderRolloutTrainerConfig
    | FirstOrderWeakTrainerConfig
    | SecondOrderRolloutTrainerConfig
    | SecondOrderWeakTrainerConfig,
    Field(discriminator = "type"),
]


# -------------------------------------------------------------------------------------------------
# Experiment schema and validation helpers
# -------------------------------------------------------------------------------------------------


_ENCODER_DECODER_N_IC = {
    "ae": 1,
    "autoencoder": 1,
    "cnn_3d": 1,
    "cnn_3d_ae": 1,
    "cnn_3d_autoencoder": 1,
    "pair": 2,
    "autoencoder_pair": 2,
}

_LATENT_DYNAMICS_N_IC = {
    "sindy": 1,
    "sindy_w": 1,
    "switch": 1,
    "switch_w": 1,
    "cable": 1,
    "spring": 2,
    "spring_w": 2,
}

_PHYSICS_N_IC = {
    "Advection": 1,
    "Burgers": 1,
    "Burgers2D": 1,
    "Explicit": 1,
    "Thermal": 1,
    "BurgersSecondOrder": 2,
    "ExplicitSecondOrder": 2,
    "KleinGordon": 2,
    "NonlinearElasticity": 2,
    "Telegraphers": 2,
    "WaveEquation": 2,
}

_EXPECTED_PHYSICS_PARAMETERS = {
    "Advection": {"w", "g"},
    "Burgers": {"a", "w"},
    "BurgersSecondOrder": {"a", "w"},
    "Burgers2D": {"k", "nu"},
    "Explicit": {"A", "w"},
    "ExplicitSecondOrder": {"A", "w"},
    "KleinGordon": {"m", "w"},
    "NonlinearElasticity": {"s", "mu"},
    "Telegraphers": {"alpha", "w"},
    "WaveEquation": {"c", "k"},
}


class ExperimentConfig(ConfigBase):
    """Complete experiment configuration."""

    # The trainer we use to train the encoder/decoder parameters and LD coefficients.
    trainer: TrainerConfig

    # Settings to control if we should train from scratch or load from a save.
    workflow: WorkflowConfig

    # Settings to control how we sample/select new training points after each round of greedy
    # sampling.
    sampler: SamplerConfig

    # Settings to define the parameter space (test/train)
    parameter_space: ParameterSpaceConfig

    # Settings to configure the encoder/decoder objects.
    EncoderDecoder: EncoderDecoderConfig

    # Specifies the latent dynamics model to use.
    latent_dynamics: LatentDynamicsConfig

    # Specifies which physics model to get FOM data from.
    physics: PhysicsConfig

    @model_validator(mode = "after")
    def validate_cross_section_consistency(self) -> "ExperimentConfig":
        trainer_type            = self.trainer.type
        encoder_type            = self.EncoderDecoder.type
        latent_dynamics_type    = self.latent_dynamics.type
        physics_type            = self.physics.type

        expected_n_ic = 1 if trainer_type.startswith("First_Order") else 2
        actual_n_ic = {
            "EncoderDecoder"    : _ENCODER_DECODER_N_IC[encoder_type],
            "latent_dynamics"   : _LATENT_DYNAMICS_N_IC[latent_dynamics_type],
            "physics"           : _PHYSICS_N_IC[physics_type],
        }
        mismatches = {
            section: n_ic for section, n_ic in actual_n_ic.items() if n_ic != expected_n_ic
        }
        if mismatches:
            raise ValueError(
                f"trainer.type = {trainer_type!r} requires n_IC={expected_n_ic}, but got "
                f"{mismatches}."
            )

        if trainer_type.endswith("_Weak") and not latent_dynamics_type.endswith("_w"):
            raise ValueError(
                f"trainer.type = {trainer_type!r} requires a weak latent_dynamics type ending in "
                f"'_w', got {latent_dynamics_type!r}."
            )

        if latent_dynamics_type in {"switch", "switch_w"} and physics_type != "Thermal":
            raise ValueError(
                "switch/switch_w latent dynamics require a physics model that defines "
                "switch_time; currently only physics.type = 'Thermal' provides this."
            )

        expected_param_names = _EXPECTED_PHYSICS_PARAMETERS.get(physics_type)
        if expected_param_names is not None:
            configured_param_names = {param.name for param in self.parameter_space.parameters}
            if configured_param_names != expected_param_names:
                raise ValueError(
                    f"physics.type = {physics_type!r} expects parameter names "
                    f"{sorted(expected_param_names)}, got {sorted(configured_param_names)}."
                )

        return self

    def to_runtime_dict(self) -> dict[str, Any]:
        """Return a validated plain dict with the same shape expected by existing code."""

        return self.model_dump(mode = "python", by_alias = True)


ExperimentConfigAdapter = TypeAdapter(ExperimentConfig)


def validate_experiment_config(config: dict[str, Any]) -> ExperimentConfig:
    """Validate and normalize a raw YAML-loaded experiment configuration."""

    return ExperimentConfigAdapter.validate_python(config)
