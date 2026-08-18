# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  torch;
import  numpy;

from    HLaSDI.Interpolate                     import Interpolate, GPInterpolate;
from    HLaSDI.LatentDynamics.LatentDynamics   import LatentDynamics;
from    HLaSDI.Schemas                         import LatentDynamicsBaseConfig;

# Logger setup.
LOGGER : logging.Logger = logging.getLogger(__name__);


# -------------------------------------------------------------------------------------------------
# InterpolatableLatentDynamics class
# -------------------------------------------------------------------------------------------------


class InterpolatableLatentDynamics(LatentDynamics):
    """
    A sub-class of LatentDynamics which assumes the latent dynamics are `interpolatable`. 

    In this case, this merely means that the latent dynamics at any parameter value is defined 
    by a set of named tensors, each one of which can change with the parameter. 
     
    This means that the set of tensors defining the latent dynamics at each parameter value
    has the same names/shapes, but the values within those named tensors can vary by parameter 
    value.

    All of this is stored in a `train_coefs` dictionary (see below). To define samples at new 
    parameter values, 
    
    The LatentDynamics object holds the learnedLatentDynamics coefficients for the training set,
    while an Interpolate object samples LatentDynamics coefficients for testing parameter 
    combinations. 

    Note that all Interpolatable latent dynamics objects are stochastic,.


    -----------------------------------------------------------------------------------------------
    Class/instance variables
    -----------------------------------------------------------------------------------------------

    train_coefs : dict[tuple[float, ...], dict[str, torch.Tensor]]
        Trainable, native coefficient dictionaries indexed by parameter tuple. The training 
        parameter value (as returned by the _param_key method) is the key, while the value is a 
        dictionary housing the associated coefficients. The dictionary for a particular parameter 
        value should use string keys (corresponding to the symbols used for various matrices and
        vectors in the latent dynamics model) and tensor value. For instance, for each combination
        of parameter values in the SINDy class, the associated coefficient dictionary has two 
        keys, "A" and "b", whose values correspond to the system matrix and bias vector in the
        SINDy latent dynamics model (z' = Az + b). This should only be used to store the TRAINING
        coefficients; test values are determined by the `interpolator` (see below). 

    n_coefs : int
        An integer housing the number of coefficients in the latent dynamics model; typically 
        (# of matrices in the LD model)*n_z^2 + (# of vectors in the LD model)*n_z


    interpolator : Interpolate
        An interpolator object used to sample the coefficients at testing parameter values. See
        the Interpolate class definition for details.
    """

    train_coefs     : dict[tuple[float, ...], dict[str, torch.Tensor]];
    n_coefs         : int;
    interpolator    : Interpolate 


    def __init__(   self, 
                    n_z             : int,
                    n_coefs         : int,
                    n_IC            : int, 
                    n_p             : int,
                    Uniform_t_Grid  : bool, 
                    trainable       : bool,
                    config          : dict) -> None:
        r"""
        Initializes a LatentDynamics object. Each LatentDynamics object needs to have a 
        dimensionality (n_z), a number of time steps, a model for the latent space dynamics, and 
        set of coefficients for that model. The model should describe a set of ODEs in 
        \mathbb{R}^{n_z}. These ODEs should contain a set of unknown coefficients. We learn those 
        coefficients using the compute_losses function. Once we have learned the coefficients, we 
        can solve the corresponding set of ODEs forward in time using the simulate function.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        n_z : int
            The number of dimensions in the latent space, where the latent dynamics takes place.

        n_coefs : int
            An integer housing the number of coefficients in the latent dynamics model; typically 
            (# of matrices in the LD model)*n_z^2 + (# of vectors in the LD model)*n_z

        n_IC : int
            Number of latent initial-condition components required to start the dynamics. For 
            example, first-order dynamics typically use `n_IC = 1`, while second-order dynamics use 
            position and velocity components with `n_IC = 2`.
    
        n_p : int 
            The number of (scalar) parameters in the parameter space.

        Uniform_t_Grid : bool 
            If True, then for each parameter value, the times corresponding to the frames of the 
            solution for that parameter value will be uniformly spaced. In other words, the first 
            frame corresponds to time t0, the second to t0 + h, the k'th to t0 + (k - 1)h, etc 
            (note that h may depend on the parameter value, but it needs to be constant for a 
            specific parameter value). The value of this setting determines which finite difference 
            method we use to compute time derivatives. 
        
        trainable : bool
            Indicates if the trainer should train the latent dynamics parameters. If false, 
            `parameters` should return an empty list.

        config : dict
            The "latent_dynamics" sub-dictionary of the config file. If `type == "weak"`, the
            model-specific sub-dictionary `config[config["type"]]` must contain `overlap`,
            `test_func_width`, and `test_func_type`.

            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        # Call base-class initializer directly. Several concrete weak classes explicitly call both
        # InterpolatableLatentDynamics.__init__ and WeakLatentDynamics.__init__, so we avoid
        # cooperative MRO here.
        LatentDynamics.__init__(self,
                                n_z                = n_z,
                                n_IC               = n_IC,
                                n_p                = n_p,
                                Uniform_t_Grid     = Uniform_t_Grid,
                                trainable          = trainable,
                                stochastic         = True,
                                config             = config)

        self.n_coefs = n_coefs;

        # Set up the Interpolate object. GP is the only implemented interpolator at the moment.
        assert isinstance(config, LatentDynamicsBaseConfig), "config must be a LatentDynamicsBaseConfig, got %s" % str(type(config));
        interpolator_type : str = config.interpolator.type;
        assert interpolator_type in {"GP"}, "Allowed interpolator types are `GP`, got %s" % interpolator_type;
        self.interpolator = GPInterpolate(config.interpolator);

        # Finally, set a dummy `train_coefs` dict.
        self.train_coefs     : dict[tuple[float, ...], dict[str, torch.Tensor]] = {};



    # ---------------------------------------------------------------------------------------------
    # Interpolatable specific methods. 
    # ---------------------------------------------------------------------------------------------

    def get_train_coefs(self, params_row : numpy.ndarray | torch.Tensor | list | tuple) -> dict[str, torch.Tensor]:
        r"""
        Fetch the native coefficient dictionary for one parameter combination.

        This method deliberately performs a direct dictionary lookup using `_param_key(...)`. If the
        requested parameter is missing, Python raises a KeyError. This is intentional: all training
        coefficients should be initialized before training starts.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        params_row : numpy.ndarray or torch.Tensor or list or tuple
            The parameter values whose coefficient dictionary we want to fetch.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        coefs : dict[str, torch.Tensor]
            A native coefficient dictionary for the requested parameter. The exact keys depend on
            the concrete LatentDynamics subclass. For example, SINDy uses `A` and `b`, while the
            damped-spring models use `K`, `C`, and `b`.
        """

        key = self._param_key(params_row);
        return self.train_coefs[key];



    def set_train_coefs(
            self, 
            params_row  : numpy.ndarray | torch.Tensor | list | tuple, 
            coefs       : dict[str, torch.Tensor], 
            device      : torch.device) -> None:
        r"""
        Store a native coefficient dictionary for one parameter combination.

        The values in `coefs` are converted to detached leaf tensors whose `requires_grad` flag
        matches `self.trainable`, unless they are already leaf tensors with the correct gradient
        setting. This ensures that `parameters()` can pass these exact tensor objects
        to a torch optimizer when training is enabled, and frozen latent dynamics do not accumulate
        coefficient gradients.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        params_row : numpy.ndarray or torch.Tensor or list or tuple
            The parameter values associated with the coefficient dictionary. These values are
            converted to a tuple key using `param_key(...)`.

        coefs : dict[str, torch.Tensor]
            Native coefficient dictionary. Keys must be strings and values must be tensors. The
            expected keys are subclass-specific.

        device : torch.device
            The device that tensors in this object should live on.

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        assert isinstance(coefs, dict), "coefs must be a dict[str, torch.Tensor]";
        device = torch.device(device);
        for name, value in coefs.items():
            assert isinstance(name, str), "coefficient names must be strings";
            assert isinstance(value, torch.Tensor), "coefficient %s must be a torch.Tensor" % name;
            if value.is_leaf and (value.requires_grad == self.trainable) and (value.device == device):
                coefs[name] = value;
            else:
                coefs[name] = value.detach().to(dtype = value.dtype, device = device).clone().requires_grad_(self.trainable);
        self.train_coefs[self._param_key(params_row)] = coefs;

        # All done :)
        return;


    def move_parameters_to_device(self, device : torch.device | str) -> None:
        r"""
        Move LD-owned parameters to the requested device as trainable leaves.

        Interpolatable latent dynamics store their LD-owned trainable tensor state in
        `self.train_coefs`. Moving these tensors requires replacing the values in that dictionary;
        a generic trainer cannot do that safely because it does not own the storage layout.

        This should be called before optimizer construction. The replacement tensors are detached
        leaves whose `requires_grad` flags match `self.trainable`, preserving the optimizer-facing
        behavior used by `set_train_coefs(...)` and checkpoint reloads.
        """

        device = torch.device(device);
        for coef_dict in self.train_coefs.values():
            assert isinstance(coef_dict, dict), "train_coefs values must be dictionaries";
            for name, tensor in list(coef_dict.items()):
                assert isinstance(name, str), "coefficient names must be strings";
                assert isinstance(tensor, torch.Tensor), "coefficient %s must be a torch.Tensor" % name;
                coef_dict[name] = tensor.detach().to(device = device).clone().requires_grad_(self.trainable);
        return;



    def update_interpolator(self) -> None:
        r"""Update the interpolator from the current LD-owned training coefficients."""

        if(len(self.train_coefs) > 0):
            self.interpolator.update_train_coefs(self.train_coefs);
        return;



    def _coefs_for_params(self,
                          params : numpy.ndarray,
                          sample : bool) -> list[dict[str, torch.Tensor]]:
        r"""
        Fetch training coefficients or interpolated coefficients for each parameter row.

        Training parameters always use the exact tensors stored in `self.train_coefs`. Parameters
        outside the training set use either an interpolated posterior mean or posterior sample.
        """

        assert isinstance(params, numpy.ndarray);
        assert len(params.shape) == 2;

        param_keys : list[tuple[float, ...]] = [self._param_key(params[i, :]) for i in range(params.shape[0])];
        needs_interpolation : bool = any(key not in self.train_coefs for key in param_keys);
        if(needs_interpolation == True):
            assert len(self.train_coefs) > 0, "Cannot interpolate coefficients before train_coefs are initialized";
            # Optimizer steps update the tensors stored in train_coefs in place. Refit the
            # interpolator before non-training rollouts so it reflects the current coefficients.
            self.update_interpolator();

        coefs_list : list[dict[str, torch.Tensor]] = [];
        for i, key in enumerate(param_keys):
            if(key in self.train_coefs):
                coefs_list.append(self.train_coefs[key]);
            else:
                if(sample == True):
                    coefs_list.append(self.interpolator.sample(params[i, :]));
                else:
                    coefs_list.append(self.interpolator.mean(params[i, :]));

        return coefs_list;


    # ---------------------------------------------------------------------------------------------
    # Serialization methods
    # ---------------------------------------------------------------------------------------------

    def export(self) -> dict:
        r"""
        Export latent-dynamics metadata and LD-owned training coefficients.

        Coefficients are detached and moved to CPU for portable checkpoint/restart files. Loading
        re-creates leaf tensors whose `requires_grad` flags match `self.trainable`, so optimizer
        construction after load still works when training is enabled and frozen latent dynamics
        remain frozen.
        """

        train_coefs_cpu : dict[tuple[float, ...], dict[str, torch.Tensor]] = {};
        for key, coef_dict in self.train_coefs.items():
            assert isinstance(coef_dict, dict), "train_coefs values must be dictionaries";
            train_coefs_cpu[key] = {};
            for name, tensor in coef_dict.items():
                assert isinstance(name, str);
                assert isinstance(tensor, torch.Tensor);
                train_coefs_cpu[key][name] = tensor.detach().cpu().clone();

        param_dict = {'n_z'             : self.n_z, 
                      'n_coefs'         : self.n_coefs, 
                      'n_IC'            : self.n_IC,
                      'config'          : self.config.model_dump(mode = "python", by_alias = True),
                      'Uniform_t_Grid'  : self.Uniform_t_Grid,
                      'train_coefs'     : train_coefs_cpu};
        return param_dict;



    def load(self, dict_ : dict) -> None:
        r"""
        Load latent-dynamics metadata and replace `self.train_coefs`.

        Shape/model metadata are checked against the already-constructed object. Coefficients are
        restored as leaf tensors whose `requires_grad` flags match `self.trainable`.
        """

        assert(self.n_z             == dict_['n_z']);
        assert(self.n_coefs         == dict_['n_coefs']);
        assert(self.n_IC            == dict_['n_IC']);
        assert(self.Uniform_t_Grid  == dict_['Uniform_t_Grid']);

        loaded_train_coefs = dict_.get('train_coefs', {});
        assert isinstance(loaded_train_coefs, dict), "train_coefs must be a dictionary";
        self.train_coefs = {};
        for key, coef_dict in loaded_train_coefs.items():
            assert isinstance(key, tuple), "train_coefs keys must be parameter tuples";
            assert isinstance(coef_dict, dict), "train_coefs values must be dictionaries";
            self.train_coefs[key] = {};
            for name, tensor in coef_dict.items():
                assert isinstance(name, str), "coefficient names must be strings";
                assert isinstance(tensor, torch.Tensor), "coefficient values must be tensors";
                self.train_coefs[key][name] = tensor.detach().clone().requires_grad_(self.trainable);
        self.update_interpolator();
        return;
