# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

from    HLaSDI.LatentDynamics.Interpolatable   import  InterpolatableLatentDynamics;
from    HLaSDI.LatentDynamics.LatentDynamics   import  LD_Loss_Container;
from    HLaSDI.Schemas                         import  SwitchSINDyLatentDynamicsConfig;
from    HLaSDI.Utilities.FiniteDifference      import  Derivative1_Order4, Derivative1_Order2_NonUniform;
from    HLaSDI.Utilities.FirstOrderSolvers     import  RK4;

LOGGER  : logging.Logger    = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# SwitchSINDy class
# -------------------------------------------------------------------------------------------------

class SwitchSINDy(InterpolatableLatentDynamics):
    def __init__(   self,
                    n_z             : int,
                    Uniform_t_Grid  : bool,
                    n_p             : int,
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

        n_p : int 
            The number of (scalar) parameters in the parameter space.
            
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
            n_p             = n_p,
            Uniform_t_Grid  = Uniform_t_Grid,
            trainable       = config.trainable,
            config          = config);

        # Class-specific initialization.
        self.lstsq_reg      : float     = config.switch.lstsq_reg;
        self.switch_time    : callable  = switch_time;

        # Setup the loss functions used by compute_losses.
        self.MSE                    = torch.nn.MSELoss(reduction = 'mean');
        self.MAE                    = torch.nn.L1Loss(reduction = 'mean');

        LOGGER.info("Initializing a SwitchSINDY object with n_z = %d, Uniform_t_Grid = %s, lstsq_reg = %s" % (self.n_z, str(self.Uniform_t_Grid), str(self.lstsq_reg)));
        return;



    def _native_from_matrices(self, before : torch.Tensor, after : torch.Tensor) -> dict[str, torch.Tensor]:
        r"""Convert before/after [b; A^T] matrices into native parameters."""

        return {
            "A_before": before[1:, :].T.detach().clone().requires_grad_(True),
            "b_before": before[0, :].detach().clone().requires_grad_(True),
            "A_after":  after[1:, :].T.detach().clone().requires_grad_(True),
            "b_after":  after[0, :].detach().clone().requires_grad_(True),
        };



    def parameters(self) -> list[torch.Tensor]:
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

        None.
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
        t_Grid          : list[torch.Tensor],
        step            : int,
        params          : numpy.ndarray | None = None
    ) -> LD_Loss_Container:
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

        t_Grid : list[torch.Tensor], len = n_param
            Time grids corresponding to the latent trajectories.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used to compute switch times and fetch coefficient dictionaries.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        losses : LD_Loss_Container
            Container housing the loss values, matching loss weights, and parameter rows used to
            compute the losses. Its `losses` dictionary has three keys: LD, coef, and stab.

            losses.losses['LD'] : list[torch.Tensor], len = n_param
                The i'th element of this list is a 0-dimensional tensor whose lone element holds the
                switching-SINDy latent-dynamics loss from the i'th combination of parameter values.

            losses.losses['coef'] : list[torch.Tensor], len = n_param
                The i'th element of this list is a 0-dimensional tensor whose lone element holds the
                coefficient loss (Frobenius norm) of the coefficients for the i'th combination
                of parameter values.

            losses.losses['stab'] : list[torch.Tensor], len = n_param
                The i'th element of this list is a 0-dimensional tensor whose lone element holds the
                stability penalty for the i'th combination of parameter values (see
                LatentDynamics.stability_penalty).
        """

        # Checks.
        assert params is not None, "SwitchSINDy.compute_losses requires params";
        assert isinstance(t_Grid, list) and isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

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
                loss_terms.append(torch.sum(residual_b**2));

            if mask_after.sum() > 0:
                RHS_a = Z[mask_after] @ A_after.T + b_after.reshape(1, -1);
                residual_a = dZdt[mask_after] - RHS_a;
                loss_terms.append(torch.sum(residual_a**2));

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

        losses_dict = {'LD' : loss_LD_list, 'coef' : loss_coef_list, 'stab' : loss_stab_list};

        return LD_Loss_Container(losses = losses_dict, weights = self.loss_weights, params = params);



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
            component, so `IC[i][0]` must have shape (n_z).

        t_Grid : list[numpy.ndarray | torch.Tensor], len = n_param
            One-dimensional time grids at which to solve the latent dynamics.

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
            The simulated latent trajectories. Z[i][0] has shape (n_t(i), n_z).
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
            if isinstance(ith_t_Grid, torch.Tensor):
                ith_t_Grid = ith_t_Grid.detach().cpu().numpy();
            assert len(ith_t_Grid.shape) == 1;
            ith_Z0 = ith_IC[0];
            assert len(ith_Z0.shape) == 1 and ith_Z0.shape[0] == self.n_z;
            switch_time_theta = self.switch_time(ith_params);
            if isinstance(switch_time_theta, torch.Tensor):
                switch_time_theta = switch_time_theta.detach().cpu().numpy();
            switch_time_theta = float(numpy.asarray(switch_time_theta).reshape(-1)[0]);

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
                b_before = b_before.reshape(-1); b_after = b_after.reshape(-1);
                def f(t : float, z : numpy.ndarray) -> numpy.ndarray:
                    return b_before + numpy.matmul(z, A_before.T) if t < switch_time_theta else b_after + numpy.matmul(z, A_after.T);
            else:
                def to_z(x):
                    return torch.tensor(x, dtype = ith_Z0.dtype, device = ith_Z0.device) if isinstance(x, numpy.ndarray) else x.to(device = ith_Z0.device, dtype = ith_Z0.dtype);
                A_before, b_before, A_after, b_after = to_z(A_before), to_z(b_before), to_z(A_after), to_z(b_after);
                b_before = b_before.reshape(-1); b_after = b_after.reshape(-1);
                def f(t : float, z : torch.Tensor) -> torch.Tensor:
                    return b_before + torch.matmul(z, A_before.T) if t < switch_time_theta else b_after + torch.matmul(z, A_after.T);

            # Solve the ODE for this single latent initial state.
            ith_Z = RK4(f = f, y0 = ith_Z0, t_Grid = ith_t_Grid);

            # Add this parameter's trajectory to the output list.
            Z.append([ith_Z]);

        # All done!
        return Z;


    def RHS(    self,
                Z       : list[list[torch.Tensor | numpy.ndarray]],
                t_Grid  : list[numpy.ndarray | torch.Tensor],
                params  : numpy.ndarray,
                sample  : bool = False) -> list[torch.Tensor | numpy.ndarray]:
        r"""
        Evaluate the switching affine RHS at a set of latent states, times, and parameters.

        Specifically, we assume that Z, t_Grid, and params have n_param elements. For each
        parameter value, theta, we evaluate the right hand side of the latent dynamics for theta
        at each time in t_Grid[i]. That is, we compute

            A_before(theta) Z[i][0](t) + b_before(theta), for each t in t_Grid[i] with t < switch_time(theta),
            A_after(theta) Z[i][0](t) + b_after(theta),   for each t in t_Grid[i] with t >= switch_time(theta).

        We compute this quantity for each time and parameter value, returning the results in a
        list.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z : list[list[torch.Tensor | numpy.ndarray]], len = n_param
            i'th element is a list of length one whose first entry is a tensor/array of shape
            [n_t(i), n_z] or [n_t(i), n_batch(i), n_z], where n_t(i) = len(t_Grid[i]). The k'th
            time slice of this tensor should represent the latent state corresponding to i'th
            parameter combination at the k'th time step.

        t_Grid : list[numpy.ndarray | torch.Tensor], len = n_param
            i'th element is a numpy.ndarray or torch.Tensor of shape [n_t(i)] whose j'th element
            holds the time corresponding to the j'th latent state in Z[i][0].

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameters corresponding to the latent solutions stored in Z.

        sample : bool
            If True, draw a sample of the latent dynamics at each non-training parameter value to
            compute the right hand sides. Otherwise, use the interpolator mean. Training parameters
            always use exact training coefficients.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        RH_Sides : list[numpy.ndarray | torch.Tensor], len = n_param
            i'th element is a numpy.ndarray or torch.Tensor with the same leading dimensions as
            Z[i][0] and last dimension n_z. It holds the right-hand side of the sampled (or mean)
            latent dynamics at params[i, :] evaluated at Z[i][0], t_Grid[i], and params[i, :].
        """

        # Checks.
        assert isinstance(params, numpy.ndarray),   "params must be a 2d numpy.ndarray, not %s" % str(type(params));
        assert len(params.shape) == 2,              "params must be a 2d numpy.ndarray of shape (n_param, n_p). Got shape %s" % str(params.shape);
        n_param : int = params.shape[0];
        assert isinstance(Z, list) and len(Z) == n_param,           "Z must have length n_params (%d), got %d" % (n_param, len(Z));
        assert isinstance(t_Grid, list) and len(t_Grid) == n_param, "t_Grid must have length n_params (%d), got %d" % (n_param, len(t_Grid));


        # -----------------------------------------------------------------------------------------
        # Fetch coefficient dictionaries for the passed parameters.
        # -----------------------------------------------------------------------------------------

        coefs_list : list[dict[str, torch.Tensor]] = self._coefs_for_params(params = params, sample = sample);


        # -----------------------------------------------------------------------------------------
        # Compute right hand sides.
        # -----------------------------------------------------------------------------------------

        RH_Sides : list[numpy.ndarray | torch.Tensor] = [];
        LOGGER.debug("Computing RHS with %d parameter combinations" % n_param);
        for i in range(n_param):
            # Fetch this parameter's data and switch time.
            ith_coefs  : dict[str, numpy.ndarray | torch.Tensor] = coefs_list[i];
            ith_Z      : list[numpy.ndarray | torch.Tensor]      = Z[i];
            ith_t_Grid : numpy.ndarray | torch.Tensor            = t_Grid[i];
            ith_params : numpy.ndarray                           = params[i, :].reshape(1, -1);

            # Checks.
            assert isinstance(ith_coefs, dict) and set(ith_coefs.keys()) == {"A_before", "b_before", "A_after", "b_after"};
            assert isinstance(ith_Z, list) and len(ith_Z) == 1;
            ith_Z0 : numpy.ndarray | torch.Tensor = ith_Z[0];
            assert isinstance(ith_Z0, (torch.Tensor, numpy.ndarray));
            assert len(ith_Z0.shape) in {2, 3};
            assert ith_Z0.shape[-1] == self.n_z;
            assert len(ith_t_Grid.shape) == 1;
            assert ith_Z0.shape[0] == ith_t_Grid.shape[0];
            if isinstance(ith_t_Grid, torch.Tensor):
                ith_t_Grid_np : numpy.ndarray = ith_t_Grid.detach().cpu().numpy();
            else:
                ith_t_Grid_np = ith_t_Grid;

            # Fetch native coefficients and switch time for this parameter.
            A_before = ith_coefs["A_before"];
            b_before = ith_coefs["b_before"];
            A_after  = ith_coefs["A_after"];
            b_after  = ith_coefs["b_after"];
            switch_time_theta = self.switch_time(ith_params);
            if isinstance(switch_time_theta, torch.Tensor):
                switch_time_theta = switch_time_theta.detach().cpu().numpy();
            switch_time_theta = float(numpy.asarray(switch_time_theta).reshape(-1)[0]);

            # Evaluate the before/after affine systems using the same backend as the latent states.
            if isinstance(ith_Z0, numpy.ndarray):
                # Map coefficient to numpy.ndarrays 
                if isinstance(A_before, torch.Tensor):
                    A_before = A_before.detach().cpu().numpy();
                    b_before = b_before.detach().cpu().numpy();
                    A_after  = A_after.detach().cpu().numpy();
                    b_after  = b_after.detach().cpu().numpy();

                # Reshape b's to be 2d.
                b_before    = b_before.reshape(1, -1);
                b_after     = b_after.reshape(1, -1);

                # Compute RHS; use "before" dynamics for times before the switch, "after" dynamics
                # for times after the switch.
                RHS_before : numpy.ndarray = numpy.matmul(ith_Z0, A_before.T) + b_before;
                RHS_after  : numpy.ndarray = numpy.matmul(ith_Z0, A_after.T)  + b_after;
                mask_before : numpy.ndarray = (ith_t_Grid_np < switch_time_theta).reshape((ith_Z0.shape[0],) + (1,)*(len(ith_Z0.shape) - 1));
                RH_Sides.append(numpy.where(mask_before, RHS_before, RHS_after));
            else:
                # Make sure coefficients are tensors
                if isinstance(A_before, numpy.ndarray):
                    A_before = torch.tensor(A_before, dtype = ith_Z0.dtype, device = ith_Z0.device);
                    b_before = torch.tensor(b_before, dtype = ith_Z0.dtype, device = ith_Z0.device);
                    A_after  = torch.tensor(A_after,  dtype = ith_Z0.dtype, device = ith_Z0.device);
                    b_after  = torch.tensor(b_after,  dtype = ith_Z0.dtype, device = ith_Z0.device);
                else:
                    A_before = A_before.to(device = ith_Z0.device, dtype = ith_Z0.dtype);
                    b_before = b_before.to(device = ith_Z0.device, dtype = ith_Z0.dtype);
                    A_after  = A_after.to(device = ith_Z0.device, dtype = ith_Z0.dtype);
                    b_after  = b_after.to(device = ith_Z0.device, dtype = ith_Z0.dtype);

                # Reshape b's to be 2d.
                b_before    = b_before.reshape(1, -1);
                b_after     = b_after.reshape(1, -1);

                # Compute RHS; use "before" dynamics for times before the switch, "after" dynamics
                # for times after the switch.
                RHS_before : torch.Tensor = torch.matmul(ith_Z0, A_before.T) + b_before;
                RHS_after  : torch.Tensor = torch.matmul(ith_Z0, A_after.T)  + b_after;                
                mask_before : torch.Tensor = torch.tensor(ith_t_Grid_np < switch_time_theta, dtype = torch.bool, device = ith_Z0.device);
                mask_before = mask_before.reshape((ith_Z0.shape[0],) + (1,)*(len(ith_Z0.shape) - 1));
                RH_Sides.append(torch.where(mask_before, RHS_before, RHS_after));

        # All done!
        return RH_Sides;
