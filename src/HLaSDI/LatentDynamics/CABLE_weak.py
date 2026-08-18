# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

from    HLaSDI.LatentDynamics.Weak             import  WeakLatentDynamics;
from    HLaSDI.LatentDynamics.LatentDynamics   import  LD_Loss_Container;
from    HLaSDI.LatentDynamics.CABLE            import  CABLE;
from    HLaSDI.Schemas                         import  WeakCABLELatentDynamicsConfig;
from    HLaSDI.Utilities.Statistics            import  tensor_statistics;

LOGGER  : logging.Logger    = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# CABLE_weak class
# -------------------------------------------------------------------------------------------------

class CABLE_weak(WeakLatentDynamics, CABLE):
    def __init__(   self,
                    n_z             : int,
                    Uniform_t_Grid  : bool,
                    n_p             : int,
                    config          : WeakCABLELatentDynamicsConfig) -> None:
        r"""
        Initialize a weak-form CABLE latent-dynamics model.

        CABLE_weak uses the same global mixture-of-linear/affine-experts latent ODE as CABLE,

            z'(t) = \sum_{m = 1}^{N} w_m(t, \theta) [ A_m z(t) + b_m ],

        when biases are enabled, and the corresponding linear-only form when biases are disabled.
        The trainable state is therefore still the set of expert matrices, optional expert biases,
        and gate-network parameters owned by CABLE.

        The difference from CABLE is only the latent-dynamics residual used during training. Rather
        than comparing the RHS against finite-difference estimates of z'(t), this class uses the
        compactly supported weak test functions stored by `WeakLatentDynamics` and enforces

            - \int phi'(t) z(t) dt = \int phi(t) f(z(t), t, \theta) dt.

        Note: This class inherits `parameters`, `initialize_coefficients`, `RHS`, `simulate`,
        `export`, and `load` from CABLE.


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

        config : WeakCABLELatentDynamicsConfig
            Weak CABLE latent-dynamics configuration. The `cable` settings define the expert and
            gate-network configuration, while the `weak` settings define the weak-form test
            functions.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        assert isinstance(config, WeakCABLELatentDynamicsConfig), "config must be a WeakCABLELatentDynamicsConfig, got %s" % str(type(config));

        CABLE.__init__(
            self,
            n_z            = n_z,
            Uniform_t_Grid = Uniform_t_Grid,
            n_p            = n_p,
            config         = config,
        );
        WeakLatentDynamics.__init__(
            self,
            n_z            = n_z,
            n_IC           = 1,
            n_p            = n_p,
            Uniform_t_Grid = Uniform_t_Grid,
            trainable      = config.trainable,
            config         = config,
        );

        LOGGER.info("Initializing a CABLE_weak object with n_z = %d, Uniform_t_Grid = %s" % (
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
        Compute weak-form CABLE latent-dynamics and regularization losses.

        For each parameter value, this method evaluates the CABLE mixture-of-experts RHS on the
        latent trajectory and compares it to the weak first-derivative term,

            - \int phi'(t) z(t) dt = \int phi(t) f(z(t), t, \theta) dt.

        The coefficient, diversity, tail-mass, and mask terms follow the strong-form CABLE
        implementation. In particular, the diversity loss is the squared coefficient of variation
        of the dense expert loads accumulated over all parameter values and time samples, while the
        tail loss penalizes dense softmax mass outside the top `n_active` experts.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            Encoded latent trajectories. The i'th entry contains one tensor of shape
            (n_t(i), n_z).

        t_Grid : list[torch.Tensor], len = n_param
            Time grids corresponding to the latent trajectories. These are used by the CABLE gate;
            the weak residual itself uses test functions previously stored for each parameter row.

        step : int
            The optimizer step number. This is used for periodic coefficient-mask updates when
            masking is enabled.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used as gate-network inputs and as keys for weak-form test-function
            lookup.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        losses : LD_Loss_Container
            Container housing scalar total losses, matching loss weights, parameter rows, and
            scalar diagnostic metrics. Its `losses` dictionary has keys `LD`, `coef`, `diversity`,
            and `tail`; each value is a scalar tensor. Per-parameter residual and tail diagnostics
            are available in `losses.metrics` under keys such as `loss/LD/<param>`.
        """

        # Checks.
        assert params is not None, "CABLE_weak.compute_losses requires params for gate inputs and weak test functions";
        assert isinstance(params, numpy.ndarray) and len(params.shape) == 2;
        assert params.shape[1] == self.n_p;
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];
        assert len(t_Grid) > 0;

        # Accumulate scalar loss contributions and diagnostics across all parameter rows. The
        # trainable CABLE coefficients are global, so coefficient/diversity losses are computed
        # once after the loop rather than separately per parameter.
        loss_LD_list            : list[torch.Tensor]      = [];
        loss_tail_list          : list[torch.Tensor]      = [];
        weights_list            : list[torch.Tensor]      = [];
        n_engaged_list          : list[torch.Tensor]      = [];
        tail_mass_list          : list[torch.Tensor]      = [];
        weight_fun_residuals    : list[torch.Tensor] = [];
        metrics                 : dict[str, torch.Tensor] = {};
        summed_weights          : torch.Tensor            = torch.zeros((self.n_experts), dtype = self.unmasked_A.dtype, device = self.unmasked_A.device);
        times_engaged           : torch.Tensor            = torch.zeros((self.n_experts), dtype = torch.int64, device = self.unmasked_A.device);

        # Periodically update hard coefficient masks, matching CABLE.compute_losses. Masked entries
        # are multiplied out through the CABLE `A`/`b` properties used below.
        if self.use_mask:
            assert self.first_mask_step is not None;
            assert self.mask_update_freq is not None;
            if step >= self.first_mask_step and (step - self.first_mask_step) % self.mask_update_freq == 0:
                self._update_mask();
            metrics["n_active/A"] = self.A_mask.sum().to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype).detach();
            if self.use_biases:
                metrics["n_active/b"] = self.b_mask.sum().to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype).detach();

        # -----------------------------------------------------------------------------------------
        # Loop over parameter combinations.
        # -----------------------------------------------------------------------------------------

        for i in range(len(t_Grid)):
            # Fetch this parameter's latent trajectory and time grid.
            ith_params  : numpy.ndarray = params[i, :];
            ith_t_Grid  : torch.Tensor  = t_Grid[i];
            ith_Z       : torch.Tensor  = Latent_States[i][0];
            n_t_i       : int           = len(ith_t_Grid);
            assert isinstance(ith_Z, torch.Tensor);
            assert len(ith_Z.shape) == 2 and ith_Z.shape[1] == self.n_z;
            assert len(ith_t_Grid.shape) == 1 and ith_t_Grid.shape[0] == n_t_i;
            assert ith_Z.shape[0] == n_t_i;

            # Fetch weak test functions and match their device/dtype to the latent trajectory.
            # get_test_functions returns rows sampled on the same time grid used when
            # add_weight_functions was called for this parameter.
            Phis0, dPhis0 = self.get_test_functions(ith_params);
            Phis   : torch.Tensor = Phis0.to(device = ith_Z.device, dtype = ith_Z.dtype);
            dPhis  : torch.Tensor = dPhis0.to(device = ith_Z.device, dtype = ith_Z.dtype);

            # Evaluate dense expert weights and the CABLE RHS on the latent trajectory. We keep the
            # dense pre-top-k weights here because the top-k sparsity target is enforced only
            # through the tail-mass loss, not by discontinuously truncating the RHS.
            ith_weights : torch.Tensor = self._weights_for_t_grid(ith_t_Grid, ith_params, t0 = ith_t_Grid[0], t_span = ith_t_Grid[-1] - ith_t_Grid[0]);
            weights_list.append(ith_weights.to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype));
            ith_RHS : torch.Tensor = self._evaluate_torch_rhs_from_weights(ith_Z, ith_weights);

            # Record which experts are engaged during each step for this parameter.
            ith_engaged : torch.Tensor = (ith_weights > self.eps_engaged).to(dtype = torch.bool, device = self.unmasked_A.device);
            times_engaged += torch.sum(ith_engaged, dim = 0);
            n_engaged_list.append(torch.sum(ith_engaged, dim = 1).to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype));

            # Weak residual. Following the weak-form convention used by the other latent dynamics
            # classes, multiplication by sampled test-function rows approximates the time
            # integrals:
            #
            #   weak_LHS[h, :] = -sum_t phi_h'(t) z(t)
            #   weak_RHS[h, :] =  sum_t phi_h(t)  f(z(t), t, theta)
            weak_LHS : torch.Tensor = -torch.matmul(dPhis, ith_Z);
            weak_RHS : torch.Tensor = torch.matmul(Phis, ith_RHS);

            # Normalize each test-function residual by ||phi'|| so losses are less sensitive to
            # the number, width, and location of weak supports.
            scale    : torch.Tensor = torch.linalg.norm(dPhis, dim = 1, keepdim = True).clamp(min = 1.0e-10);
            ith_loss_LD = self.MSE(weak_LHS / scale, weak_RHS / scale);
            loss_LD_list.append(ith_loss_LD);
            metrics[f"loss/LD/{str(ith_params)}"] = ith_loss_LD.detach();

            # Approximate the L2 (integral) norm of phi_h'(t) z(t) - phi_h(t)  f(z(t), t, theta)
            normalized_residual : torch.Tensor = (weak_LHS - weak_RHS) / scale;
            weight_fun_residuals.append(torch.sqrt(torch.mean(normalized_residual**2, dim = 1)));

            # Accumulate dense expert loads. The diversity loss below is a squared-CV penalty on
            # these totals, encouraging every expert to be used somewhere without forcing uniform
            # weights at every time sample.
            summed_weights = summed_weights + torch.sum(ith_weights.to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype), dim = 0);

            # Tail-mass penalty. This measures how much dense softmax probability lies outside the
            # top-n_active experts at each time sample. It is differentiable with respect to the
            # selected dense probabilities, but the RHS above still uses the full dense mixture.
            if self.n_active >= self.n_experts:
                ith_tail_mass : torch.Tensor = torch.zeros((n_t_i), dtype = ith_weights.dtype, device = ith_weights.device);
            else:
                ith_topk_idx        : torch.Tensor = torch.topk(ith_weights, self.n_active, dim = 1, sorted = False).indices;
                ith_topk_dense_mass : torch.Tensor = torch.sum(ith_weights.gather(1, ith_topk_idx), dim = 1);
                ith_tail_mass       : torch.Tensor = 1.0 - ith_topk_dense_mass;
            tail_mass_list.append(ith_tail_mass.to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype));
            ith_tail_loss : torch.Tensor = torch.mean(torch.pow(ith_tail_mass.to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype), 2));
            loss_tail_list.append(ith_tail_loss);
            metrics[f"loss/tail/{str(ith_params)}"] = ith_tail_loss.detach();

        # Dense gate/tail diagnostics across all parameters and times.
        weights     : torch.Tensor = torch.cat(weights_list, dim = 0);
        tail_masses : torch.Tensor = torch.cat(tail_mass_list, dim = 0);
        n_engaged   : torch.Tensor = torch.cat(n_engaged_list, dim = 0);
        metrics.update(tensor_statistics(prefix = "expert/weights",             values = weights));
        metrics.update(tensor_statistics(prefix = "mass/tail",                  values = tail_masses));
        metrics.update(tensor_statistics(prefix = "experts/num_engaged",        values = n_engaged));
        metrics.update(tensor_statistics(prefix = "experts/times_engaged",      values = times_engaged));
        metrics.update(tensor_statistics(prefix = "weak/weight_fun_residuals",  values = torch.cat(weight_fun_residuals, dim = 0)));
        metrics["experts/num_ever_engaged"] = torch.sum(times_engaged > 0).to(device = self.unmasked_A.device, dtype = self.unmasked_A.dtype).detach();

        # Coefficient loss is the sum of the selected norms of each expert matrix plus each
        # optional expert bias. The masked `A`/`b` properties ensure removed coefficients do not
        # contribute to this penalty.
        A_coef : torch.Tensor        = self.A;
        b_coef : torch.Tensor | None = self.b;
        ord    : int                 = 1 if self.coef_norm == 'l1' else 2;
        A_norms : torch.Tensor = torch.linalg.vector_norm(A_coef.reshape(self.n_experts, -1), ord = ord, dim = 1).sum();
        if b_coef is None:
            b_norms : torch.Tensor = torch.zeros((), dtype = A_coef.dtype, device = A_coef.device);
        else:
            b_norms : torch.Tensor = torch.linalg.vector_norm(b_coef.reshape(self.n_experts, -1), ord = ord, dim = 1).sum();
        loss_coef : torch.Tensor = A_norms + b_norms;

        # Diversity is the squared coefficient of variation of the total dense load assigned to
        # each expert. Use the population standard deviation so n_experts = 1 gives zero.
        metrics.update(tensor_statistics(prefix = "expert/load", values = summed_weights));
        eps             : float        = torch.finfo(summed_weights.dtype).eps;
        mean_load       : torch.Tensor = torch.mean(summed_weights);
        std_load        : torch.Tensor = torch.std(summed_weights, unbiased = False);
        loss_diversity  : torch.Tensor = torch.pow(std_load/(mean_load + eps), 2);

        # Preserve the same tail-loss diagnostics as CABLE: a per-parameter list and an unweighted
        # average are useful for plotting, while the objective below uses the summed scalar.
        self.last_tail_mass_loss = torch.mean(torch.stack(loss_tail_list)).detach();
        self.last_tail_mass_loss_list = [loss.detach() for loss in loss_tail_list];

        # Package scalar totals. Loss keys must match config.loss_weights.
        loss_LD   : torch.Tensor = torch.sum(torch.stack(loss_LD_list));
        loss_tail : torch.Tensor = torch.sum(torch.stack(loss_tail_list));
        metrics["loss/LD/total"]        = loss_LD.detach();
        metrics["loss/coef/total"]      = loss_coef.detach();
        metrics["loss/diversity/total"] = loss_diversity.detach();
        metrics["loss/tail/total"]      = loss_tail.detach();

        losses_dict = {'LD' : loss_LD, 'coef' : loss_coef, 'diversity' : loss_diversity, 'tail' : loss_tail};
        return LD_Loss_Container(losses = losses_dict, weights = self.loss_weights, params = params, metrics = metrics);
