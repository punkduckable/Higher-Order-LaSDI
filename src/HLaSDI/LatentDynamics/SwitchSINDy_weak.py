# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

from    HLaSDI.LatentDynamics.Weak             import  WeakLatentDynamics;
from    HLaSDI.LatentDynamics.Interpolatable   import  InterpolatableLatentDynamics;
from    HLaSDI.LatentDynamics.LatentDynamics   import  LD_Loss_Container;
from    HLaSDI.LatentDynamics.SwitchSINDy      import  SwitchSINDy;
from    HLaSDI.Schemas                         import  SwitchSINDyWeakLatentDynamicsConfig;

LOGGER  : logging.Logger    = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# SwitchSINDy_weak class
# -------------------------------------------------------------------------------------------------

class SwitchSINDy_weak(WeakLatentDynamics, SwitchSINDy):
    def __init__(   self,
                    n_z             : int,
                    Uniform_t_Grid  : bool,
                    n_p             : int,
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

        Note: This class inherits `simulate`, `trainable_tensors`, and `RHS` from SwitchSINDy.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        n_z : int
            The number of dimensions in the latent space.

        Uniform_t_Grid : bool
            Whether each trajectory has uniform time spacing. This argument is kept for API
            consistency with other latent-dynamics classes; weak compute_losses uses stored test
            functions rather than finite differences.

        n_p : int 
            The number of (scalar) parameters in the parameter space.

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
            n_p            = n_p,
            Uniform_t_Grid  = Uniform_t_Grid,
            trainable       = config.trainable,
            config          = config);

        WeakLatentDynamics.__init__(   
            self,
            n_z             = n_z,
            n_IC            = 1,
            n_p             = n_p,
            Uniform_t_Grid  = Uniform_t_Grid,
            trainable       = config.trainable,
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
        t_Grid          : list[torch.Tensor],
        step            : int,
        params          : numpy.ndarray | None = None
    ) -> LD_Loss_Container:
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

        t_Grid : list[torch.Tensor], len = n_param
            Time grids corresponding to the latent trajectories.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used to compute switch times and fetch coefficient dictionaries.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        loss_dict : dict[str, list[torch.Tensor] | torch.Tensor]:
            A loss dictionary with three keys: LD, coef, and stab.

            loss_dict['LD'] : list[torch.Tensor], len = n_param
                The i'th element of this list is a 0-dimensional tensor whose lone element holds the
                weak-form latent-dynamics loss from the i'th combination of parameter values.

            loss_dict['coef'] : list[torch.Tensor], len = n_param
                The i'th element of this list is a 0-dimensional tensor whose lone element holds the
                coefficient loss (Frobenius norm) of the coefficients for the i'th combination
                of parameter values.

            loss_dict['stab'] : list[torch.Tensor], len = n_param
                The i'th element of this list is a 0-dimensional tensor whose lone element holds the
                stability penalty for the i'th combination of parameter values (see
                LatentDynamics.stability_penalty).
        """

        # Checks.
        assert params is not None, "SwitchSINDy_weak.compute_losses requires params";
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

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
            loss_LD = self.MSE(weak_LHS / scale, weak_RHS / scale);

            # Compute regularization terms.
            loss_coef = torch.norm(A_before, 'fro') + torch.norm(b_before) + torch.norm(A_after, 'fro') + torch.norm(b_after);
            loss_stab = self.stability_penalty(A_before) + self.stability_penalty(A_after);

            loss_LD_list.append(loss_LD);
            loss_coef_list.append(loss_coef);
            loss_stab_list.append(loss_stab);

        losses_dict = {'LD' : loss_LD_list, 'coef' : loss_coef_list, 'stab' : loss_stab_list};

        return LD_Loss_Container(losses = losses_dict, weights = self.loss_weights, params = params);
