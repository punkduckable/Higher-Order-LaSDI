# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

from    HLaSDI.LatentDynamics.Weak             import  WeakLatentDynamics;
from    HLaSDI.LatentDynamics.LatentDynamics   import  LD_Loss_Container;
from    HLaSDI.LatentDynamics.SOME             import  SOME;
from    HLaSDI.Schemas                         import  WeakSOMELatentDynamicsConfig;
from    HLaSDI.Utilities.Statistics            import  tensor_statistics;

LOGGER  : logging.Logger    = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# SOME_weak class
# -------------------------------------------------------------------------------------------------

class SOME_weak(WeakLatentDynamics, SOME):
    def __init__(   self,
                    n_z             : int,
                    Uniform_t_Grid  : bool,
                    n_p             : int,
                    config          : WeakSOMELatentDynamicsConfig) -> None:
        r"""
        Initialize a weak-form SOME latent-dynamics model.

        SOME_weak uses the same global mixture-of-second-order affine experts as SOME,

            z''(t) =    \sum_{m = 1}^{N_D} \alpha_m(t, \theta)[K_m z(t) + b_m]
                    +   \sum_{m = 1}^{N_V} \beta_m(t, \theta) C_m z'(t),

        when biases are enabled, and the corresponding bias-free form when biases are disabled.
        The trainable state is therefore still the set of expert matrices, optional expert biases,
        and gate-network parameters owned by SOME.

        The difference from SOME is only the latent-dynamics residual used during training. Rather
        than comparing the RHS against finite-difference estimates of z''(t), this class uses the
        compactly supported weak test functions stored by `WeakLatentDynamics` and enforces

            \int phi''(t) z(t) dt = \int phi(t) f(z(t), z'(t), t, \theta) dt.

        Note: This class inherits `parameters`, `initialize_coefficients`, `RHS`, `simulate`,
        `export`, and `load` from SOME.


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

        config : WeakSOMELatentDynamicsConfig
            Weak SOME latent-dynamics configuration. The `some` settings define the expert and
            gate-network configuration, while the `weak` settings define the weak-form test
            functions.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        assert isinstance(config, WeakSOMELatentDynamicsConfig), "config must be a WeakSOMELatentDynamicsConfig, got %s" % str(type(config));

        SOME.__init__(
            self,
            n_z            = n_z,
            Uniform_t_Grid = Uniform_t_Grid,
            n_p            = n_p,
            config         = config,
        );
        WeakLatentDynamics.__init__(
            self,
            n_z            = n_z,
            n_IC           = 2,
            n_p            = n_p,
            Uniform_t_Grid = Uniform_t_Grid,
            trainable      = config.trainable,
            config         = config,
        );

        LOGGER.info("Initializing a SOME_weak object with n_z = %d, Uniform_t_Grid = %s" % (
            self.n_z,
            str(self.Uniform_t_Grid),
        ));
        return;



    def compute_losses(
        self,
        Latent_States   : list[list[torch.Tensor]],
        t_Grid          : list[torch.Tensor],
        step            : int,
        params          : numpy.ndarray | None = None,
    ) -> LD_Loss_Container:
        r"""
        Compute weak-form SOME latent-dynamics and regularization losses.

        For each parameter value, this method evaluates the split-gate SOME right-hand side,

            f(z, z', t, theta)
                = sum_m alpha_m(t, theta) [K_m z(t) + b_m]
                + sum_m beta_m(t, theta) C_m z'(t),

        on the latent trajectory and compares it to the two equivalent weak second-order terms,

            integral phi''(t) z(t) dt = -integral phi'(t) z'(t) dt
                                       =  integral phi(t) f(z(t), z'(t), t, theta) dt.

        The coefficient and mask terms follow the strong-form SOME implementation. Diversity and
        tail-mass penalties are computed independently for the displacement and velocity gates so
        they can be weighted separately. The diversity penalties are squared coefficients of
        variation of the dense expert loads accumulated over all parameter values and times. The
        tail penalties measure dense softmax mass outside each gate's top active-expert set.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            Encoded latent trajectories. The i'th entry contains two tensors of shape
            (n_t(i), n_z): latent displacement followed by latent velocity.

        t_Grid : list[torch.Tensor], len = n_param
            Time grids corresponding to the latent trajectories. These are used by the SOME gate;
            the weak residual itself uses test functions previously stored for each parameter row.

        step : int
            Optimizer step number, used for periodic coefficient-mask updates when masking is
            enabled.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used as gate-network inputs and as keys for weak test-function lookup.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        losses : LD_Loss_Container
            Container holding scalar losses, their weights, parameter rows, and detached scalar
            diagnostics. The loss keys are `LD`, `coef`, `diversity/D`, `diversity/V`, `tail/D`,
            and `tail/V`.
        """

        # Validate the trajectory, time-grid, and parameter collections before looking up test
        # functions or evaluating the gate.
        assert params is not None, "SOME_weak.compute_losses requires params for gate inputs and weak test functions";
        assert isinstance(params, numpy.ndarray) and len(params.shape) == 2;
        assert params.shape[1] == self.n_p;
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];
        assert len(t_Grid) > 0;

        # Map parameter rows to the gate device once. Keep the NumPy rows below for weak test
        # function lookup.
        w_param         : torch.Tensor = next(self.w.parameters());
        gate_device                    = w_param.device;
        gate_dtype                     = w_param.dtype;
        params_tensor   : torch.Tensor = torch.tensor(params, dtype = gate_dtype, device = gate_device);

        # Accumulate scalar loss contributions and diagnostics across all parameter rows. SOME's
        # coefficients and both expert pools are global, so coefficient/diversity losses are
        # computed once after the trajectory loop.
        loss_LD_list           : list[torch.Tensor]      = [];
        loss_tail_D_list       : list[torch.Tensor]      = [];
        loss_tail_V_list       : list[torch.Tensor]      = [];
        weights_D_list         : list[torch.Tensor]      = [];
        weights_V_list         : list[torch.Tensor]      = [];
        n_engaged_D_list       : list[torch.Tensor]      = [];
        n_engaged_V_list       : list[torch.Tensor]      = [];
        tail_mass_D_list       : list[torch.Tensor]      = [];
        tail_mass_V_list       : list[torch.Tensor]      = [];
        weight_fun_residuals_D : list[torch.Tensor]      = [];
        weight_fun_residuals_V : list[torch.Tensor]      = [];
        metrics                : dict[str, torch.Tensor] = {};
        summed_weights_D       : torch.Tensor = torch.zeros((self.n_experts_D), dtype = self.unmasked_K.dtype, device = self.unmasked_K.device);
        summed_weights_V       : torch.Tensor = torch.zeros((self.n_experts_V), dtype = self.unmasked_K.dtype, device = self.unmasked_K.device);
        times_engaged_D        : torch.Tensor = torch.zeros((self.n_experts_D), dtype = torch.int64, device = self.unmasked_K.device);
        times_engaged_V        : torch.Tensor = torch.zeros((self.n_experts_V), dtype = torch.int64, device = self.unmasked_K.device);

        # Periodically update hard coefficient masks. The effective K/C/b properties apply these
        # masks consistently to weak residuals and coefficient regularization.
        if self.use_mask:
            assert self.first_mask_step is not None;
            assert self.mask_update_freq is not None;
            if step >= self.first_mask_step and (step - self.first_mask_step) % self.mask_update_freq == 0:
                self._update_mask();
            metrics["n_active/K"] = self.K_mask.sum().to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype).detach();
            metrics["n_active/C"] = self.C_mask.sum().to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype).detach();
            if self.use_biases:
                assert self.b_mask is not None;
                metrics["n_active/b"] = self.b_mask.sum().to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype).detach();

        for i in range(len(t_Grid)):
            # Fetch this parameter row, its latent displacement/velocity, and its time grid.
            ith_params_np : numpy.ndarray = params[i, :];
            ith_params    : torch.Tensor  = params_tensor[i, :];
            ith_t_Grid    : torch.Tensor  = t_Grid[i];
            ith_Z         : torch.Tensor  = Latent_States[i][0];
            ith_dZ_dt     : torch.Tensor  = Latent_States[i][1];
            n_t_i         : int           = len(ith_t_Grid);

            assert isinstance(ith_Z, torch.Tensor);
            assert isinstance(ith_dZ_dt, torch.Tensor);
            assert len(ith_Z.shape) == 2 and ith_Z.shape == ith_dZ_dt.shape;
            assert ith_Z.shape == (n_t_i, self.n_z);
            assert len(ith_t_Grid.shape) == 1 and ith_t_Grid.shape[0] == n_t_i;

            # WeakLatentDynamics stores test functions on its own backend. Move them to the
            # latent trajectory's device/dtype before forming the weak integrals.
            Phis0, dPhis0, d2Phis0 = self.get_test_functions(ith_params_np);
            Phis   : torch.Tensor = Phis0.to(device = ith_Z.device, dtype = ith_Z.dtype);
            dPhis  : torch.Tensor = dPhis0.to(device = ith_Z.device, dtype = ith_Z.dtype);
            d2Phis : torch.Tensor = d2Phis0.to(device = ith_Z.device, dtype = ith_Z.dtype);

            # Evaluate the dense displacement and velocity gates and the resulting acceleration.
            # The full dense mixture is used in the weak residual; top-k is only a regularizer.
            ith_RHS, ith_weights_D, ith_weights_V = self._evaluate_rhs(
                Z       = ith_Z,
                dZ_dt   = ith_dZ_dt,
                t_Grid  = ith_t_Grid,
                params  = ith_params,
                t0      = ith_t_Grid[0],
                t_span  = ith_t_Grid[-1] - ith_t_Grid[0]);
            assert isinstance(ith_RHS, torch.Tensor);

            # Collect gate weights and engagement counts on the coefficient backend for global
            # diagnostics and differentiable load-diversity penalties.
            ith_weights_D_local = ith_weights_D.to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype);
            ith_weights_V_local = ith_weights_V.to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype);
            weights_D_list.append(ith_weights_D_local);
            weights_V_list.append(ith_weights_V_local);

            ith_engaged_D : torch.Tensor = (ith_weights_D > self.eps_engaged).to(dtype = torch.bool, device = self.unmasked_K.device);
            ith_engaged_V : torch.Tensor = (ith_weights_V > self.eps_engaged).to(dtype = torch.bool, device = self.unmasked_K.device);
            times_engaged_D += torch.sum(ith_engaged_D, dim = 0);
            times_engaged_V += torch.sum(ith_engaged_V, dim = 0);
            n_engaged_D_list.append(torch.sum(ith_engaged_D, dim = 1).to(dtype = self.unmasked_K.dtype));
            n_engaged_V_list.append(torch.sum(ith_engaged_V, dim = 1).to(dtype = self.unmasked_K.dtype));

            # Integration by parts gives two equivalent weak left-hand sides. Comparing the RHS
            # to both uses the supplied displacement and velocity trajectories symmetrically.
            weak_LHS_D : torch.Tensor = torch.matmul(d2Phis, ith_Z);
            weak_LHS_V : torch.Tensor = -torch.matmul(dPhis, ith_dZ_dt);
            weak_RHS   : torch.Tensor = torch.matmul(Phis, ith_RHS);
            # Normalize by the corresponding test-function derivative norms. This prevents test
            # functions with larger raw quadrature weights from dominating the residual.
            scale_D    : torch.Tensor = torch.linalg.norm(d2Phis, dim = 1, keepdim = True).clamp(min = 1.0e-10);
            scale_V    : torch.Tensor = torch.linalg.norm(dPhis, dim = 1, keepdim = True).clamp(min = 1.0e-10);

            loss_D : torch.Tensor = self.MSE(weak_LHS_D/scale_D, weak_RHS/scale_D);
            loss_V : torch.Tensor = self.MSE(weak_LHS_V/scale_V, weak_RHS/scale_V);
            loss_LD_list.append(0.5*(loss_D + loss_V));

            normalized_residual_D : torch.Tensor = (weak_LHS_D - weak_RHS)/scale_D;
            normalized_residual_V : torch.Tensor = (weak_LHS_V - weak_RHS)/scale_V;
            weight_fun_residuals_D.append(torch.sqrt(torch.mean(normalized_residual_D**2, dim = 1)));
            weight_fun_residuals_V.append(torch.sqrt(torch.mean(normalized_residual_V**2, dim = 1)));

            # Accumulate total dense load per expert over all times and parameter rows. Separate
            # pools are essential because N_D and N_V need not be equal.
            summed_weights_D += torch.sum(ith_weights_D_local, dim = 0);
            summed_weights_V += torch.sum(ith_weights_V_local, dim = 0);

            # Penalize probability mass outside each gate's desired active set. The two branches
            # are intentionally independent: one pool may use every expert while the other remains
            # sparse. The RHS above still uses the complete dense mixtures.
            if self.n_active_D >= self.n_experts_D:
                ith_tail_mass_D : torch.Tensor = torch.zeros((n_t_i), dtype = ith_weights_D.dtype, device = ith_weights_D.device);
            else:
                topk_D = torch.topk(ith_weights_D, self.n_active_D, dim = 1, sorted = False).indices;
                ith_tail_mass_D = 1.0 - torch.sum(ith_weights_D.gather(1, topk_D), dim = 1);

            if self.n_active_V >= self.n_experts_V:
                ith_tail_mass_V : torch.Tensor = torch.zeros((n_t_i), dtype = ith_weights_V.dtype, device = ith_weights_V.device);
            else:
                topk_V = torch.topk(ith_weights_V, self.n_active_V, dim = 1, sorted = False).indices;
                ith_tail_mass_V = 1.0 - torch.sum(ith_weights_V.gather(1, topk_V), dim = 1);

            ith_tail_mass_D = ith_tail_mass_D.to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype);
            ith_tail_mass_V = ith_tail_mass_V.to(device = self.unmasked_K.device, dtype = self.unmasked_K.dtype);
            tail_mass_D_list.append(ith_tail_mass_D);
            tail_mass_V_list.append(ith_tail_mass_V);
            loss_tail_D_list.append(torch.mean(ith_tail_mass_D**2));
            loss_tail_V_list.append(torch.mean(ith_tail_mass_V**2));

        # Dense gate, tail-mass, engagement, and weak-residual diagnostics over all trajectories.
        weights_D     : torch.Tensor = torch.cat(weights_D_list, dim = 0);
        weights_V     : torch.Tensor = torch.cat(weights_V_list, dim = 0);
        tail_masses_D : torch.Tensor = torch.cat(tail_mass_D_list, dim = 0);
        tail_masses_V : torch.Tensor = torch.cat(tail_mass_V_list, dim = 0);
        n_engaged_D   : torch.Tensor = torch.cat(n_engaged_D_list, dim = 0);
        n_engaged_V   : torch.Tensor = torch.cat(n_engaged_V_list, dim = 0);
        metrics.update(tensor_statistics(prefix = "expert/weights/D", values = weights_D));
        metrics.update(tensor_statistics(prefix = "expert/weights/V", values = weights_V));
        metrics.update(tensor_statistics(prefix = "mass/tail/D", values = tail_masses_D));
        metrics.update(tensor_statistics(prefix = "mass/tail/V", values = tail_masses_V));
        metrics.update(tensor_statistics(prefix = "expert/num_engaged/D", values = n_engaged_D));
        metrics.update(tensor_statistics(prefix = "expert/num_engaged/V", values = n_engaged_V));
        metrics.update(tensor_statistics(prefix = "expert/times_engaged/D", values = times_engaged_D));
        metrics.update(tensor_statistics(prefix = "expert/times_engaged/V", values = times_engaged_V));
        metrics.update(tensor_statistics(prefix = "weak/weight_fun_residuals_D", values = torch.cat(weight_fun_residuals_D, dim = 0)));
        metrics.update(tensor_statistics(prefix = "weak/weight_fun_residuals_V", values = torch.cat(weight_fun_residuals_V, dim = 0)));
        metrics["expert/num_ever_engaged/D"] = torch.sum(times_engaged_D > 0).to(dtype = self.unmasked_K.dtype).detach();
        metrics["expert/num_ever_engaged/V"] = torch.sum(times_engaged_V > 0).to(dtype = self.unmasked_K.dtype).detach();

        # Coefficient loss is the sum of the selected norms of every displacement matrix,
        # velocity matrix, and optional displacement bias. Masked coefficients do not contribute.
        K_coef : torch.Tensor        = self.K;
        C_coef : torch.Tensor        = self.C;
        b_coef : torch.Tensor | None = self.b;
        ord    : int                 = 1 if self.coef_norm == 'l1' else 2;
        K_norms : torch.Tensor = torch.linalg.vector_norm(K_coef.reshape(self.n_experts_D, -1), ord = ord, dim = 1).sum();
        C_norms : torch.Tensor = torch.linalg.vector_norm(C_coef.reshape(self.n_experts_V, -1), ord = ord, dim = 1).sum();
        if b_coef is None:
            b_norms : torch.Tensor = torch.zeros((), dtype = K_coef.dtype, device = K_coef.device);
        else:
            b_norms = torch.linalg.vector_norm(b_coef.reshape(self.n_experts_D, -1), ord = ord, dim = 1).sum();
        loss_coef : torch.Tensor = K_norms + C_norms + b_norms;

        # Each diversity loss is the squared coefficient of variation of the corresponding
        # expert loads. Population standard deviation makes a one-expert pool produce zero rather
        # than NaN.
        metrics.update(tensor_statistics(prefix = "expert/summed_weight/D", values = summed_weights_D));
        metrics.update(tensor_statistics(prefix = "expert/summed_weight/V", values = summed_weights_V));
        eps              : float        = torch.finfo(summed_weights_D.dtype).eps;
        mean_D_load      : torch.Tensor = torch.mean(summed_weights_D);
        mean_V_load      : torch.Tensor = torch.mean(summed_weights_V);
        std_D_load       : torch.Tensor = torch.std(summed_weights_D, unbiased = False);
        std_V_load       : torch.Tensor = torch.std(summed_weights_V, unbiased = False);
        loss_diversity_D : torch.Tensor = torch.pow(std_D_load/(mean_D_load + eps), 2);
        loss_diversity_V : torch.Tensor = torch.pow(std_V_load/(mean_V_load + eps), 2);

        # Preserve separate per-parameter tail diagnostics for plotting and debugging. The
        # objective below uses the summed scalar for each pool, matching strong-form SOME.
        mean_tail_D : torch.Tensor = torch.mean(torch.stack(loss_tail_D_list));
        mean_tail_V : torch.Tensor = torch.mean(torch.stack(loss_tail_V_list));
        self.last_tail_mass_loss_D      = mean_tail_D.detach();
        self.last_tail_mass_loss_V      = mean_tail_V.detach();
        self.last_tail_mass_loss_D_list = [loss.detach() for loss in loss_tail_D_list];
        self.last_tail_mass_loss_V_list = [loss.detach() for loss in loss_tail_V_list];

        # Package scalar totals. Keys must match the configured loss weights, but neither trainer
        # assumes a fixed set of latent-dynamics loss names.
        loss_LD     : torch.Tensor = torch.sum(torch.stack(loss_LD_list));
        loss_tail_D : torch.Tensor = torch.sum(torch.stack(loss_tail_D_list));
        loss_tail_V : torch.Tensor = torch.sum(torch.stack(loss_tail_V_list));
        metrics["loss/LD/total"]            = loss_LD.detach();
        metrics["loss/coef/total"]          = loss_coef.detach();
        metrics["loss/coef/K"]              = K_norms.detach();
        metrics["loss/coef/C"]              = C_norms.detach();
        metrics["loss/coef/b"]              = b_norms.detach();
        metrics["loss/diversity/D"]     = loss_diversity_D.detach();
        metrics["loss/diversity/V"]     = loss_diversity_V.detach();
        metrics["loss/tail/D"]          = loss_tail_D.detach();
        metrics["loss/tail/V"]          = loss_tail_V.detach();

        losses_dict = {'LD'             : loss_LD,
                       'coef'           : loss_coef,
                       'diversity/D'    : loss_diversity_D,
                       'diversity/V'    : loss_diversity_V,
                       'tail/D'         : loss_tail_D,
                       'tail/V'         : loss_tail_V};
        return LD_Loss_Container(losses = losses_dict, weights = self.loss_weights, params = params, metrics = metrics);
