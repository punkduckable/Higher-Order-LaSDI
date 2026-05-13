# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;
from    sklearn.gaussian_process    import  GaussianProcessRegressor;

from    GaussianProcess             import  fit_gps, eval_gp, sample_coefs;

LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Interpolate class
# -------------------------------------------------------------------------------------------------

class Interpolate:
    r"""
    GP-backed interpolation over native latent-dynamics coefficient dictionaries.

    This class fits one independent Gaussian process for each scalar component of each named
    coefficient tensor. The public methods return coefficient dictionaries with the same native
    keys and tensor shapes as the latent-dynamics model's `train_coefs` entries.


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    train_coefs : dict[tuple[float, ...], dict[str, torch.Tensor]]
        LD-owned training coefficient dictionary. The outer key is an exact parameter tuple; the
        inner dictionary maps coefficient tensor names to tensors. Every inner dictionary must have
        the same string keys and tensor shapes.


    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Nothing!
    """

    def __init__(self, train_coefs : dict[tuple[float, ...], dict[str, torch.Tensor]]) -> None:
        r"""
        Build one collection of GPs for each named coefficient tensor.

        For a fixed tensor name (for example "A" or "K"), every training parameter must have a
        tensor with the same shape. We flatten that tensor component-wise and fit one independent GP
        per scalar component, using the parameter tuple as GP input.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        train_coefs : dict[tuple[float, ...], dict[str, torch.Tensor]]
            LD-owned training coefficient dictionary. The outer key is an exact parameter tuple;
            the inner dictionary maps coefficient tensor names to tensors.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Check the top-level object first. The class is intentionally narrow for now: it only
        # supports LD train_coefs dictionaries with tensor-valued inner dictionaries.
        assert isinstance(train_coefs, dict), "train_coefs must be a dictionary";
        assert len(train_coefs) > 0, "train_coefs must be non-empty";

        # Store parameter keys in a deterministic list. This order is used to build both the GP
        # input array X and the corresponding target rows for every coefficient tensor.
        self.train_coefs = train_coefs;
        self.param_keys : list[tuple[float, ...]] = list(train_coefs.keys());
        for key in self.param_keys:
            assert isinstance(key, tuple), "train_coefs keys must be parameter tuples";
            assert all(isinstance(x, float) for x in key), "parameter tuple entries must be floats";

        # Use the first coefficient dictionary as the schema, then verify that every other
        # parameter has exactly the same names and shapes.
        first = train_coefs[self.param_keys[0]];
        assert isinstance(first, dict), "train_coefs values must be dictionaries";
        self.coef_names : list[str] = list(first.keys());
        assert len(self.coef_names) > 0, "coefficient dictionaries must be non-empty";
        for name in self.coef_names:
            assert isinstance(name, str), "coefficient names must be strings";
            assert isinstance(first[name], torch.Tensor), "coefficient values must be tensors";

        self.coef_shapes : dict[str, torch.Size] = {name: first[name].shape for name in self.coef_names};
        for key, coef_dict in train_coefs.items():
            assert isinstance(coef_dict, dict), "train_coefs[%s] must be a dictionary" % str(key);
            assert set(coef_dict.keys()) == set(self.coef_names), "coefficient keys differ for parameter %s" % str(key);
            for name in self.coef_names:
                assert isinstance(coef_dict[name], torch.Tensor), "coefficient %s for parameter %s must be a tensor" % (name, str(key));
                assert coef_dict[name].shape == self.coef_shapes[name], "coefficient %s shape mismatch for parameter %s" % (name, str(key));

        # GP inputs: one row per training parameter.
        self.X : numpy.ndarray = numpy.array(self.param_keys, dtype = numpy.float64);
        assert len(self.X.shape) == 2, "parameter keys must form a 2D array";

        # For each named tensor, flatten it across components and fit one GP per component. The
        # existing GaussianProcess utilities handle scaling, kernel construction, and sampling.
        self.gps : dict[str, list[GaussianProcessRegressor]] = {};
        for name in self.coef_names:
            Y_rows : list[numpy.ndarray] = [];
            for key in self.param_keys:
                Y_rows.append(train_coefs[key][name].detach().cpu().numpy().reshape(1, -1));
            Y : numpy.ndarray = numpy.concatenate(Y_rows, axis = 0);
            self.gps[name] = fit_gps(self.X, Y);
            LOGGER.info("Fit %d GPs for coefficient tensor '%s' with shape %s" % (Y.shape[1], name, tuple(self.coef_shapes[name])));
        return;



    @staticmethod
    def _param_array(param : numpy.ndarray | torch.Tensor | list | tuple) -> numpy.ndarray:
        r"""
        Normalize a parameter input to a one-dimensional NumPy array for GP evaluation.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray or torch.Tensor or list or tuple
            Parameter values for one requested point.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        param_np : numpy.ndarray, shape = (n_p,)
            One-dimensional float64 NumPy array containing the parameter values.
        """

        if isinstance(param, torch.Tensor):
            param = param.detach().cpu().numpy();
        elif isinstance(param, (list, tuple)):
            param = numpy.array(param);
        assert isinstance(param, numpy.ndarray), "param must be numpy.ndarray, torch.Tensor, list, or tuple";
        return param.reshape(-1).astype(numpy.float64);



    def sample(self, param : numpy.ndarray | torch.Tensor | list | tuple) -> dict[str, torch.Tensor]:
        r"""
        Draw one native coefficient sample at a requested parameter value.

        The returned dictionary has the same keys and tensor shapes as each item in `train_coefs`,
        so it can be passed directly to `LatentDynamics.simulate(...)`.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray or torch.Tensor or list or tuple
            Parameter values at which to sample the coefficient posterior.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        out : dict[str, torch.Tensor]
            Native coefficient dictionary containing one posterior sample for each coefficient
            tensor.
        """

        x = self._param_array(param);
        out : dict[str, torch.Tensor] = {};
        for name in self.coef_names:
            sample_np : numpy.ndarray = sample_coefs(self.gps[name], x, 1)[0, :].reshape(tuple(self.coef_shapes[name]));
            out[name] = torch.tensor(sample_np, dtype = torch.float32);
        return out;



    def mean(self, param : numpy.ndarray | torch.Tensor | list | tuple) -> dict[str, torch.Tensor]:
        r"""
        Return the posterior mean coefficient dictionary at a requested parameter value.

        This is used for deterministic plotting/rollouts where drawing random coefficient samples
        would make figures nondeterministic.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray or torch.Tensor or list or tuple
            Parameter values at which to evaluate the posterior mean.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        out : dict[str, torch.Tensor]
            Native coefficient dictionary containing the posterior mean for each coefficient
            tensor.
        """

        x = self._param_array(param);
        out : dict[str, torch.Tensor] = {};
        for name in self.coef_names:
            mean_np, _ = eval_gp(self.gps[name], x.reshape(1, -1));
            out[name] = torch.tensor(mean_np[0, :].reshape(tuple(self.coef_shapes[name])), dtype = torch.float32);
        return out;



    def std(self, param : numpy.ndarray | torch.Tensor | list | tuple) -> dict[str, torch.Tensor]:
        r"""
        Return the posterior standard-deviation coefficient dictionary at a requested parameter.

        The returned tensors use the same native keys and shapes as `mean(...)`. Each entry holds
        the marginal GP posterior standard deviation for the corresponding scalar coefficient.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray or torch.Tensor or list or tuple
            Parameter values at which to evaluate posterior standard deviations.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        out : dict[str, torch.Tensor]
            Native coefficient dictionary containing posterior standard deviations for each
            coefficient tensor.
        """

        x = self._param_array(param);
        out : dict[str, torch.Tensor] = {};
        for name in self.coef_names:
            _, std_np = eval_gp(self.gps[name], x.reshape(1, -1));
            out[name] = torch.tensor(std_np[0, :].reshape(tuple(self.coef_shapes[name])), dtype = torch.float32);
        return out;
