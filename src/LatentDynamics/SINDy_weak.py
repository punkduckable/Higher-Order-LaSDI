# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

from    LatentDynamics.Weak             import  WeakLatentDynamics;
from    LatentDynamics.Interpolatable   import  InterpolatableLatentDynamics;
from    Schemas                         import  SINDyWeakLatentDynamicsConfig;
from    Utilities.FirstOrderSolvers     import  RK4;

LOGGER  : logging.Logger    = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# SINDy_weak class
# -------------------------------------------------------------------------------------------------

class SINDy_weak(WeakLatentDynamics, InterpolatableLatentDynamics):
    def __init__(   self,
                    n_z             : int,
                    Uniform_t_Grid  : bool,
                    config          : SINDyWeakLatentDynamicsConfig) -> None:
        r"""
        Initializes a SINDy_weak latent-dynamics object.

        This class is the weak-form version of the affine SINDy latent dynamics

            z'(t) = A z(t) + b.

        Here, z is the latent state, A is an n_z x n_z matrix, and b is an n_z-vector. There is a
        separate set of coefficients for each combination of parameter values. We store the native
        coefficient tensors in `self.train_coefs` as

            {"A": A, "b": b}.

        The weak-form latent-dynamics residual is based on

            - \int phi'(t) z(t) dt = \int phi(t) (A z(t) + b) dt,

        where phi is one of the compactly supported test functions owned by the base class.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        n_z : int
            The number of dimensions in the latent space.

        Uniform_t_Grid : bool
            Whether each trajectory has uniform time spacing. This argument is kept for API
            consistency with other latent-dynamics classes; weak compute_losses uses stored test
            functions rather than finite differences.

        config : dict
            The latent-dynamics configuration dictionary. It must three keys: `type`, `trainable`,
            and `sindy_w`. It must have `config["type"] == "sindy_w"` and `config["sindy_w"]` 
            should be a weak-form sub-dictionary containing the following keys:
                - test_func_type: Specifies the kind of bump function. Either "bump" or "PC-poly".
                - test_func_width: The width of each bump.
                - overlap: The amount of overlap between successive bumps.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        assert isinstance(config, SINDyWeakLatentDynamicsConfig), "config must be a SINDyWeakLatentDynamicsConfig, got %s" % str(type(config));

        # Run the base class initializer. Since A has n_z^2 entries and b has n_z entries, there
        # are n_z*(n_z + 1) scalar coefficients.
        InterpolatableLatentDynamics.__init__(
            self,
            n_z            = n_z,
            n_coefs        = n_z*(n_z + 1),
            n_IC           = 1,
            Uniform_t_Grid = Uniform_t_Grid,
            trainable      = config["trainable"],
            config         = config);

        WeakLatentDynamics.__init__(
            self,
            n_z            = n_z,
            n_coefs        = n_z*(n_z + 1),
            n_IC           = 1,
            Uniform_t_Grid = Uniform_t_Grid,
            trainable      = config["trainable"],
            config         = config);

        LOGGER.info("Initializing a SINDy_weak object with n_z = %d, Uniform_t_Grid = %s" % (
            self.n_z,
            str(self.Uniform_t_Grid),
        ));

        # Setup the loss functions used by compute_losses.
        self.MSE = torch.nn.MSELoss(reduction = 'mean');
        self.MAE = torch.nn.L1Loss(reduction = 'mean');
        return;



    def trainable_tensors(self) -> list[torch.Tensor]:
        r"""Return the actual weak-form SINDy coefficient tensors to optimize."""

        if self.trainable == False:
            return [];

        tensors : list[torch.Tensor] = [];
        for coef_dict in self.train_coefs.values():
            tensors.extend([coef_dict["A"], coef_dict["b"]]);
        return tensors;



    # ---------------------------------------------------------------------------------------------
    # initialize_coefficients
    # ---------------------------------------------------------------------------------------------

    def initialize_coefficients(
            self,
            Latent_States   : list[list[torch.Tensor]],
            t_Grid          : list[torch.Tensor],
            device          : torch.device,
            params          : numpy.ndarray) -> None:
        r"""
        Initialize weak-form SINDy coefficients to zero.

        This method intentionally does not solve a weak-form least-squares system. For weak
        training, coefficients computed from a randomly initialized encoder can be a poor starting
        point. Instead, each requested parameter receives trainable zero tensors for `A` and `b`;
        the optimizer learns them jointly with the encoder/decoder.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element contains one latent trajectory tensor with shape (n_t(i), n_z).
            This method uses the tensor dtype to initialize coefficients with matching precision.

        t_Grid : list[torch.Tensor], len = n_param
            Time grids corresponding to the latent trajectories. These are checked for length
            consistency but are not otherwise used because weak coefficients are zero-initialized.

        device : torch.device
            Device on which the new coefficient tensors should be stored.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used as keys in `self.train_coefs`.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        None. Zero coefficient dictionaries are stored in `self.train_coefs`, and the interpolator
        is updated from the full training-coefficient dictionary.
        """

        assert params is not None, "SINDy_weak.initialize_coefficients requires `params`";
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

        for i in range(params.shape[0]):
            assert isinstance(Latent_States[i], list);
            assert len(Latent_States[i]) == self.n_IC;
            assert isinstance(Latent_States[i][0], torch.Tensor);
            dtype  = Latent_States[i][0].dtype;

            A : torch.Tensor = torch.zeros((self.n_z, self.n_z), device = device, dtype = dtype, requires_grad = True);
            b : torch.Tensor = torch.zeros((self.n_z,),          device = device, dtype = dtype, requires_grad = True);
            self.set_train_coefs(params[i, :], {"A": A, "b": b}, device);

        # Finally, update the interpolator using the new training coefficients!
        self.update_interpolator();

        # All done :) 
        return None;



    # ---------------------------------------------------------------------------------------------
    # compute_losses
    # ---------------------------------------------------------------------------------------------

    def compute_losses(  
        self,
        Latent_States   : list[list[torch.Tensor]],
        loss_type       : str,
        t_Grid          : list[torch.Tensor],
        params          : numpy.ndarray | None = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        r"""
        Compute weak-form SINDy latent-dynamics, coefficient, and stability losses.

        For each parameter combination, this method fetches the native coefficient dictionary from
        `self.train_coefs` and the weak-form test functions from the base class. Missing
        coefficients or test functions are intentional hard errors because the sampler/training
        path should initialize both before optimization starts.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element contains one latent trajectory tensor of shape (n_t(i), n_z).

        loss_type : str
            The type of loss function to use. Must be either "MSE" or "MAE".

        t_Grid : list[torch.Tensor], len = n_param
            Time grids corresponding to the latent trajectories.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used to fetch weak-form test functions and coefficient dictionaries.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        loss_LD_list : list[torch.Tensor], len = n_param
            Per-parameter weak-form SINDy residual losses.

        loss_coef_list : list[torch.Tensor], len = n_param
            Per-parameter coefficient regularization values.

        loss_stab_list : list[torch.Tensor], len = n_param
            Per-parameter stability penalties from the linear system matrix.
        """

        # Checks.
        assert params is not None, "SINDy_weak.compute_losses requires params";
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];
        assert loss_type in ["MSE", "MAE"];

        loss_LD_list   : list[torch.Tensor] = [];
        loss_coef_list : list[torch.Tensor] = [];
        loss_stab_list : list[torch.Tensor] = [];

        # -----------------------------------------------------------------------------------------
        # Loop over parameter combinations.
        # -----------------------------------------------------------------------------------------

        for i in range(len(t_Grid)):
            assert isinstance(Latent_States[i], list);
            assert len(Latent_States[i]) == self.n_IC;

            # Fetch this parameter's latent trajectory.
            Z : torch.Tensor = Latent_States[i][0];
            assert isinstance(Z, torch.Tensor);
            assert len(Z.shape) == 2;
            assert Z.shape[-1] == self.n_z;

            # Fetch weak test functions and match their device/dtype to Z.
            Phis0, dPhis0 = self.get_test_functions(params[i, :]);
            Phis   : torch.Tensor = Phis0.to(device = Z.device, dtype = Z.dtype);
            dPhis  : torch.Tensor = dPhis0.to(device = Z.device, dtype = Z.dtype);

            # Fetch native trainable coefficients for this parameter.
            coef_dict = self.get_train_coefs(params[i, :]);
            A = coef_dict["A"].to(device = Z.device, dtype = Z.dtype);
            b = coef_dict["b"].to(device = Z.device, dtype = Z.dtype);

            # Compute the weak residual. We follow the existing weak-form convention in this repo:
            # matrix multiplication with the sampled test functions approximates the time integral.
            weak_LHS : torch.Tensor = -torch.matmul(dPhis, Z);
            RHS      : torch.Tensor = torch.matmul(Z, A.T) + b.reshape(1, -1);
            weak_RHS : torch.Tensor = torch.matmul(Phis, RHS);

            # Normalize each test-function residual by the norm of phi' to keep losses comparable
            # across support locations and widths.
            scale : torch.Tensor = torch.linalg.norm(dPhis, dim = 1, keepdim = True).clamp(min = 1.0e-10);
            if(loss_type == "MSE"):
                loss_LD = self.MSE(weak_LHS / scale, weak_RHS / scale);
            else:
                loss_LD = self.MAE(weak_LHS / scale, weak_RHS / scale);

            # Compute regularization terms.
            loss_coef = torch.norm(A, 'fro') + torch.norm(b);
            loss_stab = self.stability_penalty(A);

            loss_LD_list.append(loss_LD);
            loss_coef_list.append(loss_coef);
            loss_stab_list.append(loss_stab);

        return loss_LD_list, loss_coef_list, loss_stab_list;



    def simulate(   self,
                    IC      : list[list[numpy.ndarray | torch.Tensor]],
                    t_Grid  : list[numpy.ndarray      | torch.Tensor],
                    params  : numpy.ndarray,
                    sample  : bool = False) -> list[list[numpy.ndarray | torch.Tensor]]:
        r"""
        Time-integrate the native SINDy latent dynamics.

        The weak formulation only changes the LD loss; rollouts still solve

            z'(t) = A z(t) + b.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        IC : list[list[numpy.ndarray | torch.Tensor]], len = n_param
            Initial latent states for each coefficient set. SINDy_weak has one IC component.

        t_Grid : list[numpy.ndarray | torch.Tensor], len = n_param
            Time grids for simulation.

        params : numpy.ndarray, shape = (n_param, n_p)
            The i'th row holds the i'th combination of parameter values.

        sample : bool 
            If self is stochastic, setting this to true will sample from the posterior distribution 
            of the latent dynamics at each parameter value, then solve the latent dynamics using 
            the resulting sample. Otherwise, setting this to true will use the mean of that 
            posterior distribution. If self is not stochastic, this does nothing.
            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Z : list[list[numpy.ndarray | torch.Tensor]]
            Simulated latent trajectories. Z[i][0] has shape (n_t(i), n_initial_conditions, n_z).
        """

        # Checks.
        assert isinstance(params, numpy.ndarray);
        assert len(params.shape) == 2;
        n_param : int = params.shape[0];
        assert isinstance(t_Grid, list) and isinstance(IC, list);
        assert len(IC) == n_param and len(t_Grid) == n_param;


        # -----------------------------------------------------------------------------------------
        # Fetch coefficient dictionaries for the passed parameters.
        # -----------------------------------------------------------------------------------------

        coefs_list : list[dict[str, torch.Tensor]] = self._coefs_for_params(params = params, sample = sample);


        # -----------------------------------------------------------------------------------------
        # Loop through parameter combinations.
        # -----------------------------------------------------------------------------------------

        Z : list[list[numpy.ndarray | torch.Tensor]] = [];
        LOGGER.debug("Simulating with %d parameter combinations" % n_param);
        for i in range(n_param):
            # Fetch the i'th set of coefficients, initial conditions, and time grid.
            ith_coefs  : dict[str, numpy.ndarray | torch.Tensor] = coefs_list[i];
            ith_IC     : list[numpy.ndarray | torch.Tensor]      = IC[i];
            ith_t_Grid : numpy.ndarray | torch.Tensor            = t_Grid[i];

            # Set up the i'th single-parameter solve.
            assert isinstance(ith_coefs, dict) and set(ith_coefs.keys()) == {"A", "b"};
            assert isinstance(ith_IC, list) and len(ith_IC) == 1;
            assert len(ith_t_Grid.shape) == 1 or len(ith_t_Grid.shape) == 2;
            if(isinstance(ith_t_Grid, torch.Tensor)):
                ith_t_Grid = ith_t_Grid.detach().cpu().numpy();
            Same_t_Grid : bool = (len(ith_t_Grid.shape) == 1);
            ith_Z0 : numpy.ndarray | torch.Tensor = ith_IC[0];
            n_i    : int                          = ith_Z0.shape[0];
            assert len(ith_Z0.shape) == 2 and ith_Z0.shape[1] == self.n_z;
            if(Same_t_Grid == False):
                assert ith_t_Grid.shape[0] == n_i;

            # Fetch native coefficients for this parameter.
            A = ith_coefs["A"];
            b = ith_coefs["b"];

            # Match the coefficient backend to the initial-condition backend. This keeps the solver
            # purely NumPy for NumPy inputs and differentiable PyTorch for tensor inputs.
            if isinstance(ith_Z0, numpy.ndarray):
                if isinstance(A, torch.Tensor):
                    A = A.detach().cpu().numpy();
                    b = b.detach().cpu().numpy();
                b = b.reshape(1, -1);
                f = lambda t, z: b + numpy.matmul(z, A.T);
            else:
                if isinstance(A, numpy.ndarray):
                    A = torch.tensor(A, dtype = ith_Z0.dtype, device = ith_Z0.device);
                    b = torch.tensor(b, dtype = ith_Z0.dtype, device = ith_Z0.device);
                else:
                    A = A.to(device = ith_Z0.device, dtype = ith_Z0.dtype);
                    b = b.to(device = ith_Z0.device, dtype = ith_Z0.dtype);
                b = b.reshape(1, -1);
                f = lambda t, z: b + torch.matmul(z, A.T);

            # Solve the ODE. If all ICs share the same time grid we integrate them as a batch;
            # otherwise, integrate each initial condition separately and concatenate the results.
            if(Same_t_Grid == True):
                ith_Z = RK4(f = f, y0 = ith_Z0, t_Grid = ith_t_Grid);
            else:
                Z_list : list[torch.Tensor | numpy.ndarray] = [];
                for j in range(n_i):
                    Z_j = RK4(f = f, y0 = ith_Z0[j, :].reshape(1, -1), t_Grid = ith_t_Grid[j, :]);
                    Z_list.append(Z_j);
                if(isinstance(ith_Z0, numpy.ndarray)):
                    ith_Z = numpy.concatenate(Z_list, axis = 1);
                else:
                    ith_Z = torch.cat(Z_list, dim = 1);

            # Add this parameter's trajectory to the output list.
            Z.append([ith_Z]);

        # All done!
        return Z;
