# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

from    LatentDynamics.Interpolatable   import  InterpolatableLatentDynamics;
from    Schemas                         import  SwitchSINDyLatentDynamicsConfig;
from    Utilities.FiniteDifference      import  Derivative1_Order4, Derivative1_Order2_NonUniform;
from    Utilities.FirstOrderSolvers     import  RK4;

LOGGER  : logging.Logger    = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# SwitchSINDy class
# -------------------------------------------------------------------------------------------------

class SwitchSINDy(InterpolatableLatentDynamics):
    def __init__(   self, 
                    n_z             : int,
                    Uniform_t_Grid  : bool, 
                    switch_time     : callable,
                    config          : SwitchSINDyLatentDynamicsConfig) -> None:
        r"""
        Initializes a SwitchSINDy object.

        This is a SINDy-type latent dynamics model that switches between two affine latent ODEs
        according to a parameter-dependent switch time. For a parameter value theta,

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
            If True, each trajectory has uniform time spacing and an O(h^4) derivative stencil can
            be used. Otherwise, nonuniform-grid finite differences are used.

        switch_time : callable
            A function that takes a numpy.ndarray of parameter values and returns the switch time
            for those parameter values.
        
        config : dict
            The latent-dynamics configuration dictionary. It must three keys: `type`, `trainable`,
            and `switch`. It must have `config["type"] == "switch"` and `config["switch"]` should 
            be a dictionary housing sub-class specific settings. The required `lstsq_reg` entry 
            controls ridge regularization used by `initialize_coefficients(...)` when initializing 
            coefficients from encoded trajectories.

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        assert isinstance(config, SwitchSINDyLatentDynamicsConfig), "config must be a SwitchSINDyLatentDynamicsConfig, got %s" % str(type(config));

        # Run the base class initializer. Note that this sets self.train_coefs.
        InterpolatableLatentDynamics.__init__(   
            self,
            n_z             = n_z, 
            n_coefs         = n_z*(n_z + 1)*2,
            n_IC            = 1,
            Uniform_t_Grid  = Uniform_t_Grid, 
            trainable       = config["trainable"],
            config          = config);

        # Class-specific initialization.
        self.lstsq_reg      : float     = config["switch"]["lstsq_reg"];
        self.switch_time    : callable  = switch_time;
        
        # Setup the loss functions used by compute_losses.
        self.MSE                    = torch.nn.MSELoss(reduction = 'mean');
        self.MAE                    = torch.nn.L1Loss(reduction = 'mean');

        LOGGER.info("Initializing a SwitchSINDY object with n_z = %d, Uniform_t_Grid = %s, lstsq_reg = %s" % (self.n_z, str(self.Uniform_t_Grid), str(self.lstsq_reg)));
        return;



    def _native_from_matrices(self, before : torch.Tensor, after : torch.Tensor) -> dict[str, torch.Tensor]:
        r"""Convert before/after [b; A^T] matrices into native trainable tensors."""

        return {
            "A_before": before[1:, :].T.detach().clone().requires_grad_(True),
            "b_before": before[0, :].detach().clone().requires_grad_(True),
            "A_after":  after[1:, :].T.detach().clone().requires_grad_(True),
            "b_after":  after[0, :].detach().clone().requires_grad_(True),
        };



    def trainable_tensors(self) -> list[torch.Tensor]:
        r"""Return all trainable switching-SINDy tensors."""

        if self.trainable == False:
            return [];

        tensors : list[torch.Tensor] = [];
        for coef_dict in self.train_coefs.values():
            tensors.extend([coef_dict["A_before"], coef_dict["b_before"], coef_dict["A_after"], coef_dict["b_after"]]);
        return tensors;



    def initialize_coefficients(
            self,
            Latent_States   : list[list[torch.Tensor]],
            t_Grid          : list[torch.Tensor],
            device          : torch.device, 
            params          : numpy.ndarray) -> None:
        r"""
        Fit coefficients for the two-regime switching SINDy model.

        This estimates separate affine SINDy coefficient matrices before and after the switch time
        for each parameter combination. The fitted matrices are converted to native dictionaries
        and stored in `self.train_coefs`; no flattened coefficient array is returned.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th list element contains one tensor with shape (n_t(i), n_z), holding the latent
            state trajectory for the i'th parameter combination.

        t_Grid : list[torch.Tensor], len = n_param
            The i'th element is a 1D tensor of shape (n_t(i)) holding the time grid for the i'th
            parameter combination.

        device : torch.device
            The device where we want to store the new coefficients.
        
        params : numpy.ndarray, shape = (n_param, n_p)
            The i'th row holds the parameter values used both to compute the switch time and to key
            `self.train_coefs`.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        None. Coefficients are stored in `self.train_coefs`.
        """

        # Checks.
        assert params is not None, "SwitchSINDy.initialize_coefficients requires params";
        assert isinstance(t_Grid, list) and isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

        for i in range(len(t_Grid)):
            t_Grid0 : torch.Tensor  = t_Grid[i];
            Z       : torch.Tensor  = Latent_States[i][0];
            n_t     : int           = len(t_Grid0);
            if(self.Uniform_t_Grid == True):
                h       : float         = (t_Grid0[1] - t_Grid0[0]).item();
                dZdt    : torch.Tensor  = Derivative1_Order4(Z, h);
            else:
                dZdt                    = Derivative1_Order2_NonUniform(Z, t_Grid = t_Grid0);

            # Build the affine library [1, z] and split it into before/after-switch samples.
            Z_with_ones : torch.Tensor = torch.cat([torch.ones(n_t, 1, device = Z.device, dtype = Z.dtype), Z], dim = 1);
            params_i = params[i, :].reshape(1, -1);
            switch_time_theta : float = self.switch_time(params_i);
            mask_before = t_Grid0 < switch_time_theta;
            mask_after  = ~mask_before;
            n_lib       : int = Z_with_ones.shape[1];

            # Fit one side of the switch. If no time samples fall in a regime, initialize that
            # regime to zero rather than solving an empty least-squares problem.
            def fit_segment(Z_seg : torch.Tensor, dZ_seg : torch.Tensor) -> torch.Tensor:
                if Z_seg.shape[0] == 0:
                    return torch.zeros(self.n_z + 1, self.n_z, device = Z.device, dtype = Z.dtype);
                if self.lstsq_reg > 0.0:
                    gram = Z_seg.T @ Z_seg + self.lstsq_reg * torch.eye(n_lib, device = Z.device, dtype = Z.dtype);
                    return torch.linalg.solve(gram, Z_seg.T @ dZ_seg);
                return torch.linalg.lstsq(Z_seg, dZ_seg).solution;

            coefs_before = fit_segment(Z_with_ones[mask_before], dZdt[mask_before]);
            coefs_after  = fit_segment(Z_with_ones[mask_after],  dZdt[mask_after]);
            self.set_train_coefs(params[i, :], self._native_from_matrices(coefs_before, coefs_after), device);

        # Finally, update the interpolator using the new training coefficients!
        self.update_interpolator();

        # All done :) 
        return None;



    def compute_losses(  
        self,  
        Latent_States   : list[list[torch.Tensor]], 
        loss_type       : str,
        t_Grid          : list[torch.Tensor], 
        params          : numpy.ndarray | None = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        r"""
        Compute switching-SINDy latent-dynamics, coefficient, and stability losses.

        For each parameter combination, this method looks up the native coefficient dictionary in
        `self.train_coefs`, splits the time samples into before/after-switch groups, and evaluates
        the corresponding affine right-hand side on each group.


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
            Per-parameter switching-SINDy residual losses.

        loss_coef_list : list[torch.Tensor], len = n_param
            Per-parameter coefficient regularization values.

        loss_stab_list : list[torch.Tensor], len = n_param
            Per-parameter stability penalties from the before and after systems.
        """

        # Checks.
        assert params is not None, "SwitchSINDy.compute_losses requires params";
        assert isinstance(t_Grid, list) and isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];
        assert loss_type in ["MSE", "MAE"];

        # Prepare containers for the three loss components returned to the Trainer. The Trainer
        # applies the user-specified weights and sums these values into the total objective.
        loss_LD_list   : list[torch.Tensor] = [];
        loss_coef_list : list[torch.Tensor] = [];
        loss_stab_list : list[torch.Tensor] = [];

        # -----------------------------------------------------------------------------------------
        # Loop over parameter combinations.
        # -----------------------------------------------------------------------------------------

        for i in range(len(t_Grid)):
            # Fetch the latent trajectory and time grid for this parameter.
            t_Grid0 : torch.Tensor  = t_Grid[i];
            Z       : torch.Tensor  = Latent_States[i][0];
            n_t     : int           = len(t_Grid0);

            # Approximate dZ/dt using the finite-difference stencil appropriate for the time grid.
            if(self.Uniform_t_Grid == True):
                h       : float         = (t_Grid0[1] - t_Grid0[0]).item();
                dZdt    : torch.Tensor  = Derivative1_Order4(Z, h);
            else:
                dZdt                    = Derivative1_Order2_NonUniform(Z, t_Grid = t_Grid0);

            # -------------------------------------------------------------------------------------
            # Fetch native coefficients for this parameter.
            # -------------------------------------------------------------------------------------

            # Fetch native trainable coefficients for this parameter.
            coef_dict = self.get_train_coefs(params[i, :]);
            A_before = coef_dict["A_before"].to(device = Z.device, dtype = Z.dtype);
            b_before = coef_dict["b_before"].to(device = Z.device, dtype = Z.dtype);
            A_after  = coef_dict["A_after"].to(device = Z.device, dtype = Z.dtype);
            b_after  = coef_dict["b_after"].to(device = Z.device, dtype = Z.dtype);

            # -------------------------------------------------------------------------------------
            # Split the trajectory into before/after-switch samples.
            # -------------------------------------------------------------------------------------

            switch_time_theta : float = self.switch_time(params[i, :].reshape(1, -1));
            mask_before = t_Grid0 < switch_time_theta;
            mask_after  = ~mask_before;

            # -------------------------------------------------------------------------------------
            # Compute the residual loss.
            # -------------------------------------------------------------------------------------

            # Each regime uses its own affine model. It is possible (especially for short or
            # truncated trajectories) for one regime to have no samples, so each term is guarded.
            loss_terms : list[torch.Tensor] = [];
            if mask_before.sum() > 0:
                RHS_b = Z[mask_before] @ A_before.T + b_before.reshape(1, -1);
                residual_b = dZdt[mask_before] - RHS_b;
                if(loss_type == "MSE"):
                    loss_terms.append(torch.sum(residual_b**2));
                else:
                    loss_terms.append(torch.sum(torch.abs(residual_b)));

            if mask_after.sum() > 0:
                RHS_a = Z[mask_after] @ A_after.T + b_after.reshape(1, -1);
                residual_a = dZdt[mask_after] - RHS_a;
                if(loss_type == "MSE"):
                    loss_terms.append(torch.sum(residual_a**2));
                else:
                    loss_terms.append(torch.sum(torch.abs(residual_a)));

            # Normalize by the total number of time samples so trajectories with more frames do not
            # automatically dominate the objective.
            loss_LD = sum(loss_terms) / float(n_t);

            # -------------------------------------------------------------------------------------
            # Compute regularization terms.
            # -------------------------------------------------------------------------------------

            # Coefficient regularization: penalize the sizes of both affine systems.
            loss_coef = torch.norm(A_before, 'fro') + torch.norm(b_before) + torch.norm(A_after, 'fro') + torch.norm(b_after);

            # Stability regularization: apply the base-class differentiable stability penalty to
            # each linear part. The constant terms b_before/b_after do not affect linear stability.
            loss_stab = self.stability_penalty(A_before) + self.stability_penalty(A_after);

            # Package this parameter's losses.
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
        Time integrates the switching SINDy latent dynamics.

        Coefficients are fetched from `self.train_coefs` for training parameters and from
        `self.interpolator` for non-training parameters. Unlike plain SINDy, `params` is required
        because the right-hand side depends on the switch time.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        IC : list[list[numpy.ndarray | torch.Tensor]], len = n_param
            Initial latent states for each parameter/coefficient set. SwitchSINDy has one IC
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
