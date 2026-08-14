# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

from    LatentDynamics.Weak             import  WeakLatentDynamics;
from    LatentDynamics.Interpolatable   import  InterpolatableLatentDynamics;
from    Schemas                         import  SwitchSINDyWeakLatentDynamicsConfig;
from    Utilities.FirstOrderSolvers     import  RK4;

LOGGER  : logging.Logger    = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# SwitchSINDy_weak class
# -------------------------------------------------------------------------------------------------

class SwitchSINDy_weak(WeakLatentDynamics, InterpolatableLatentDynamics):
    def __init__(   self,
                    n_z             : int,
                    Uniform_t_Grid  : bool,
                    switch_time     : callable,
                    config          : SwitchSINDyWeakLatentDynamicsConfig) -> None:
        r"""
        Initializes a SwitchSINDy_weak latent-dynamics object.

        This class is the weak-form version of the switching affine SINDy model. For a parameter
        value theta, the latent dynamics are

            z'(t) = A_before(theta) z(t) + b_before(theta),  t <  switch_time(theta),
            z'(t) = A_after(theta)  z(t) + b_after(theta),   t >= switch_time(theta).

        Coefficients are stored natively in `self.train_coefs` using the keys `A_before`,
        `b_before`, `A_after`, and `b_after`.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        n_z : int
            The number of dimensions in the latent space.

        Uniform_t_Grid : bool
            Whether each trajectory has uniform time spacing. This argument is kept for API
            consistency with other latent-dynamics classes; weak compute_losses uses stored test
            functions rather than finite differences.

        switch_time : callable
            A function that takes a numpy.ndarray of parameter values and returns the switch time
            for those parameter values.

        config : dict
            The latent-dynamics configuration dictionary. It must three keys: `type`, `trainable`,
            and `switch_w`. It must have `config["type"] == "switch_w"` and `config["switch_w"]` 
            should be a weak-form sub-dictionary containing the following keys:
                - test_func_type: Specifies the kind of bump function. Either "bump" or "PC-poly".
                - test_func_width: The width of each bump.
                - overlap: The amount of overlap between successive bumps.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        assert isinstance(config, SwitchSINDyWeakLatentDynamicsConfig), "config must be a SwitchSINDyWeakLatentDynamicsConfig, got %s" % str(type(config));

        # Run the base class initializer. There are two affine systems, each with n_z*(n_z + 1)
        # scalar coefficients.
        InterpolatableLatentDynamics.__init__(   
            self,
            n_z             = n_z,
            n_coefs         = n_z*(n_z + 1)*2,
            n_IC            = 1,
            Uniform_t_Grid  = Uniform_t_Grid,
            trainable       = config["trainable"],
            config          = config);

        WeakLatentDynamics.__init__(   
            self,
            n_z             = n_z,
            n_coefs         = n_z*(n_z + 1)*2,
            n_IC            = 1,
            Uniform_t_Grid  = Uniform_t_Grid,
            trainable       = config["trainable"],
            config          = config);

        # Class-specific initialization.
        self.switch_time : callable = switch_time;

        # Setup the loss functions used by compute_losses.
        self.MSE = torch.nn.MSELoss(reduction = 'mean');
        self.MAE = torch.nn.L1Loss(reduction = 'mean');

        LOGGER.info("Initializing a SwitchSINDy_weak object with n_z = %d, Uniform_t_Grid = %s" % (
            self.n_z,
            str(self.Uniform_t_Grid),
        ));
        return;



    def trainable_tensors(self) -> list[torch.Tensor]:
        r"""Return all trainable weak-form switching-SINDy coefficient tensors."""

        if self.trainable == False:
            return [];

        tensors : list[torch.Tensor] = [];
        for coef_dict in self.train_coefs.values():
            tensors.extend([coef_dict["A_before"], coef_dict["b_before"], coef_dict["A_after"], coef_dict["b_after"]]);
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
        Initialize weak-form switching-SINDy coefficients to zero.

        This method intentionally does not solve a weak-form least-squares system. Each requested
        parameter receives trainable zero tensors for `A_before`, `b_before`, `A_after`, and
        `b_after`; the optimizer learns them jointly with the encoder/decoder.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element contains one latent trajectory tensor with shape (n_t(i), n_z).
            This method uses the tensor dtype/device to initialize coefficients with matching
            precision and placement.

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

        assert params is not None, "SwitchSINDy_weak.initialize_coefficients requires `params`";
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

        for i in range(params.shape[0]):
            assert isinstance(Latent_States[i], list);
            assert len(Latent_States[i]) == self.n_IC;
            assert isinstance(Latent_States[i][0], torch.Tensor);
            dtype  = Latent_States[i][0].dtype;

            A_before : torch.Tensor = torch.zeros((self.n_z, self.n_z), device = device, dtype = dtype, requires_grad = True);
            b_before : torch.Tensor = torch.zeros((self.n_z,),          device = device, dtype = dtype, requires_grad = True);
            A_after  : torch.Tensor = torch.zeros((self.n_z, self.n_z), device = device, dtype = dtype, requires_grad = True);
            b_after  : torch.Tensor = torch.zeros((self.n_z,),          device = device, dtype = dtype, requires_grad = True);
            self.set_train_coefs(params[i, :], {
                "A_before": A_before,
                "b_before": b_before,
                "A_after":  A_after,
                "b_after":  b_after,
            }, device);

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
        Compute weak-form switching-SINDy latent-dynamics, coefficient, and stability losses.

        For each parameter combination, this method fetches the native coefficient dictionary from
        `self.train_coefs`, splits the weak-form right-hand side into before/after-switch
        contributions, and compares it against the weak first-derivative term.


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
            Parameter rows used to compute switch times and fetch coefficient dictionaries.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        loss_LD_list : list[torch.Tensor], len = n_param
            Per-parameter weak-form switching-SINDy residual losses.

        loss_coef_list : list[torch.Tensor], len = n_param
            Per-parameter coefficient regularization values.

        loss_stab_list : list[torch.Tensor], len = n_param
            Per-parameter stability penalties from the before and after systems.
        """

        # Checks.
        assert params is not None, "SwitchSINDy_weak.compute_losses requires params";
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

            # Fetch this parameter's latent trajectory and time grid.
            Z       : torch.Tensor = Latent_States[i][0];
            t_Grid0 : torch.Tensor = t_Grid[i];
            assert isinstance(Z, torch.Tensor);
            assert isinstance(t_Grid0, torch.Tensor);
            assert len(Z.shape) == 2;
            assert Z.shape[-1] == self.n_z;

            # Fetch weak test functions and match their device/dtype to Z.
            Phis0, dPhis0 = self.get_test_functions(params[i, :]);
            Phis   : torch.Tensor = Phis0.to(device = Z.device, dtype = Z.dtype);
            dPhis  : torch.Tensor = dPhis0.to(device = Z.device, dtype = Z.dtype);

            # Fetch native trainable coefficients for this parameter.
            coef_dict = self.get_train_coefs(params[i, :]);
            A_before = coef_dict["A_before"].to(device = Z.device, dtype = Z.dtype);
            b_before = coef_dict["b_before"].to(device = Z.device, dtype = Z.dtype);
            A_after  = coef_dict["A_after"].to(device = Z.device, dtype = Z.dtype);
            b_after  = coef_dict["b_after"].to(device = Z.device, dtype = Z.dtype);

            # Split the trajectory into before/after-switch samples.
            switch_time_theta : float = self.switch_time(params[i, :].reshape(1, -1));
            mask_before = (t_Grid0 < switch_time_theta).to(device = Z.device);
            mask_after  = ~mask_before;
            mask_before = mask_before.to(dtype = Z.dtype).reshape(1, -1);
            mask_after  = mask_after.to(dtype = Z.dtype).reshape(1, -1);

            # Compute the weak residual. The before/after masks restrict the test-function rows to
            # the corresponding switch regime.
            weak_LHS   : torch.Tensor = -torch.matmul(dPhis, Z);
            RHS_before : torch.Tensor = torch.matmul(Z, A_before.T) + b_before.reshape(1, -1);
            RHS_after  : torch.Tensor = torch.matmul(Z, A_after.T)  + b_after.reshape(1, -1);
            weak_RHS   : torch.Tensor = torch.matmul(Phis * mask_before, RHS_before) + torch.matmul(Phis * mask_after, RHS_after);

            # Normalize each test-function residual by the norm of phi' to keep losses comparable
            # across support locations and widths.
            scale : torch.Tensor = torch.linalg.norm(dPhis, dim = 1, keepdim = True).clamp(min = 1.0e-10);
            if(loss_type == "MSE"):
                loss_LD = self.MSE(weak_LHS / scale, weak_RHS / scale);
            else:
                loss_LD = self.MAE(weak_LHS / scale, weak_RHS / scale);

            # Compute regularization terms.
            loss_coef = torch.norm(A_before, 'fro') + torch.norm(b_before) + torch.norm(A_after, 'fro') + torch.norm(b_after);
            loss_stab = self.stability_penalty(A_before) + self.stability_penalty(A_after);

            loss_LD_list.append(loss_LD);
            loss_coef_list.append(loss_coef);
            loss_stab_list.append(loss_stab);

        return loss_LD_list, loss_coef_list, loss_stab_list;



    def simulate(   self,
                    IC      : list[list[numpy.ndarray | torch.Tensor]],
                    t_Grid  : list[numpy.ndarray      | torch.Tensor],
                    params  : numpy.ndarray,
                    sample  : bool  = False) -> list[list[numpy.ndarray | torch.Tensor]]:
        r"""
        Time integrates the switching SINDy latent dynamics.

        The weak formulation only changes the LD loss; rollouts still solve the native
        before/after switching affine ODE.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        IC : list[list[numpy.ndarray | torch.Tensor]], len = n_param
            Initial latent states for each parameter/coefficient set. SwitchSINDy_weak has one IC
            component.

        t_Grid : list[numpy.ndarray | torch.Tensor], len = n_param
            Time grids at which to solve the latent dynamics.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used to compute the switch time for each simulation.

        sample : bool 
            If self is stochastic, setting this to true will sample from the posterior distribution 
            of the latent dynamics at each parameter value, then solve the latent dynamics using 
            the resulting sample. Otherwise, setting this to true will use the mean of that 
            posterior distribution. If self is not stochastic, this does nothing.

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Z : list[list[numpy.ndarray | torch.Tensor]], len = n_param
            The simulated latent trajectories. Z[i][0] has shape
            (n_t(i), n_initial_conditions, n_z).
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
            # Fetch this parameter's data and switch time.
            ith_coefs  : dict[str, numpy.ndarray | torch.Tensor] = coefs_list[i];
            ith_IC     : list[numpy.ndarray | torch.Tensor]      = IC[i];
            ith_t_Grid : numpy.ndarray | torch.Tensor            = t_Grid[i];
            ith_params : numpy.ndarray                           = params[i, :].reshape(1, -1);

            # Set up the i'th single-parameter solve.
            assert isinstance(ith_coefs, dict) and set(ith_coefs.keys()) == {"A_before", "b_before", "A_after", "b_after"};
            assert isinstance(ith_IC, list) and len(ith_IC) == 1;
            assert len(ith_t_Grid.shape) == 1 or len(ith_t_Grid.shape) == 2;
            if isinstance(ith_t_Grid, torch.Tensor):
                ith_t_Grid = ith_t_Grid.detach().cpu().numpy();
            Same_t_Grid = (len(ith_t_Grid.shape) == 1);
            ith_Z0 = ith_IC[0];
            n_i = ith_Z0.shape[0];
            assert len(ith_Z0.shape) == 2 and ith_Z0.shape[1] == self.n_z;
            if(Same_t_Grid == False):
                assert ith_t_Grid.shape[0] == n_i;
            switch_time_theta = self.switch_time(ith_params);

            # Fetch native coefficients and match them to the IC backend below.
            A_before, b_before, A_after, b_after = ith_coefs["A_before"], ith_coefs["b_before"], ith_coefs["A_after"], ith_coefs["b_after"];

            # Define the right-hand side in either NumPy or PyTorch. The solver backend follows the
            # initial-condition backend; this preserves differentiability for tensor rollouts in
            # training and keeps plotting/sampling paths lightweight with NumPy arrays.
            if isinstance(ith_Z0, numpy.ndarray):
                vals = [];
                for x in [A_before, b_before, A_after, b_after]:
                    vals.append(x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x);
                A_before, b_before, A_after, b_after = vals;
                b_before = b_before.reshape(1, -1); b_after = b_after.reshape(1, -1);
                def f(t : float, z : numpy.ndarray) -> numpy.ndarray:
                    return b_before + numpy.matmul(z, A_before.T) if t < switch_time_theta else b_after + numpy.matmul(z, A_after.T);
            else:
                def to_z(x):
                    return torch.tensor(x, dtype = ith_Z0.dtype, device = ith_Z0.device) if isinstance(x, numpy.ndarray) else x.to(device = ith_Z0.device, dtype = ith_Z0.dtype);
                A_before, b_before, A_after, b_after = to_z(A_before), to_z(b_before), to_z(A_after), to_z(b_after);
                b_before = b_before.reshape(1, -1); b_after = b_after.reshape(1, -1);
                def f(t : float, z : torch.Tensor) -> torch.Tensor:
                    return b_before + torch.matmul(z, A_before.T) if t < switch_time_theta else b_after + torch.matmul(z, A_after.T);

            # Integrate all initial conditions together when they share a time grid; otherwise integrate
            # each row of the IC array with its corresponding row of the time-grid array.
            if(Same_t_Grid == True):
                ith_Z = RK4(f = f, y0 = ith_Z0, t_Grid = ith_t_Grid);
            else:
                Z_list : list[torch.Tensor | numpy.ndarray] = [];
                for j in range(n_i):
                    Z_list.append(RK4(f = f, y0 = ith_Z0[j, :].reshape(1, -1), t_Grid = ith_t_Grid[j, :]));
                ith_Z = numpy.concatenate(Z_list, axis = 1) if isinstance(ith_Z0, numpy.ndarray) else torch.cat(Z_list, dim = 1);

            # Add this parameter's trajectory to the output list.
            Z.append([ith_Z]);

        # All done!
        return Z;
