# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

from    LatentDynamics.LatentDynamics   import  LatentDynamics;
from    Schemas                         import  CABLELatentDynamicsConfig, CABLELatentDynamicsSettings;
from    EncoderDecoder                  import  MultiLayerPerceptron;
from    Utilities.FiniteDifference      import  Derivative1_Order4, Derivative1_Order2_NonUniform;
from    Utilities.FirstOrderSolvers     import  RK4;

LOGGER  : logging.Logger    = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# SINDy class
# -------------------------------------------------------------------------------------------------

class CABLE(LatentDynamics):
    def __init__(   self, 
                    n_z             : int,
                    Uniform_t_Grid  : bool,
                    n_p             : int,
                    config          : CABLELatentDynamicsConfig) -> None:
        r"""
        Initialize a CABLE latent dynamics model.

        This model assumes a mixture-of-affine-experts style latent ODE:

            z'(t) = \sum_{n = 1}^{N} w_n(\theta, t) [ A_n z(t) + b_n ]
        
        Here, K is the number of experts, each A_n is an n_z x n_z matrix, and b_n is an element 
        of \mathbb{R}^{n_z}, and w_n(t, \theta) is the `weight` of the n'th expert at time t given
        parameter \theta. The expert weights are determined by a `gate` neural network, 

            w : \mathbb{R} \times \mathbb{R}^{n_p} \to \mathbb{R}^{N}

        This network has a soft max applied to its final layer so that the weights for any time, 
        parameter value sum to 1.

        The gate network's parameters and the matrix/vector portion of each expert's affine map
        define the set of trainable parameters.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        n_z : int
            The number of latent dimensions.

        Uniform_t_Grid : bool
            If True, each trajectory has uniform time spacing and we can use the higher-order
            finite-difference stencil. Otherwise, nonuniform-grid finite differences are used.

        n_p : int 
            The number of (scalar) parameters in the parameter space.

        config : CABLELatentDynamicsConfig
            The latent-dynamics configuration dictionary. It must three keys: `type`, `trainable`,
            and `sindy`. It must have `config["type"] == "sindy"` and `config["sindy"]` should be a 
            dictionary housing sub-class specific settings. The required `lstsq_reg` entry controls
            ridge regularization used by `initialize_coefficients(...)` when initializing 
            coefficients from encoded trajectories.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        assert isinstance(config, CABLELatentDynamicsConfig), "config must be a CABLELatentDynamicsConfig, got %s" % str(type(config));

        # Run the base class initializer. 
        LatentDynamics.__init__(
            self,
            n_z            = n_z, 
            n_IC           = 1, 
            n_p            = n_p,
            Uniform_t_Grid = Uniform_t_Grid,
            trainable      = config.trainable,
            stochastic     = False,
            config         = config);

        # Extract sub-class specific attributes
        sub : CABLELatentDynamicsSettings = config.cable
        self.n_experts      : int       = sub.n_experts;
        self.top_k          : int       = sub.top_k;
        self.hidden_widths  : list[int] = sub.hidden_widths;
        self.activations    : list[str] = sub.activations;

        # Initialize the gate network.
        widths      : list[int] = [n_p + 1] + self.hidden_widths + [self.n_experts];
        self.w                  = MultiLayerPerceptron(widths = widths, activations = self.activations);

        # Randomly initialize the experts.
        self.A : torch.Tensor = torch.rand((self.n_experts, self.n_z, self.n_z), dtype = torch.float32);
        self.b : torch.Tensor = torch.zeros((self.n_ezperts, 1, self.n_z), dtype = torch.float32);

        # Setup the loss functions used by compute_losses.
        self.MSE = torch.nn.MSELoss(reduction = 'mean');
        self.MAE = torch.nn.L1Loss(reduction = 'mean');
        return;


    def trainable_tensors(self) -> list[torch.Tensor]:
        r"""
        Return the actual coefficient tensors that should be passed to torch optimizers.

        These are not copies. They are the same tensors stored in `self.train_coefs`, so optimizer
        updates modify the LD-owned coefficient dictionaries used by compute_losses/simulate.
        """

        if self.trainable == False:
            return [];

        tensors : list[torch.Tensor] = [self.A, self.b];
        for param in self.w.parameters():
            tensors.append(param);
        return tensors;


    def move_trainable_tensors_to_device(self, device : torch.device | str) -> None:
        r"""
        Move LD-owned trainable tensor state to a device.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        device : torch.device or str
            The destination device for LD-owned trainable tensor state.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Move A, b, and w to specified device.
        self.A = self.A.to(device = device);
        self.b = self.b.to(device = device);
        self.w = self.w.to(device = device);


    def initialize_coefficients(
            self,
            Latent_States   : list[list[torch.Tensor]],
            t_Grid          : list[torch.Tensor],
            device          : torch.device,
            params          : numpy.ndarray) -> None:
        r"""
        Does nothing; CABLE coefficients are initialized during initialization. All we do here is 
        ensure that the MLP and expert coefficients lie on the correct device.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            The i'th entry contains one tensor with shape (n_t(i), n_z), holding the encoded latent
            trajectory for the i'th parameter.

        t_Grid : list[torch.Tensor], len = n_param
            Time grid for each latent trajectory.

        device : torch.device
            The device where we want to store the new coefficients.
            
        params : numpy.ndarray, shape = (n_param, n_p)
            The parameters we are adding to the training set.
            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        None.
        """

        # Checks.
        assert params is not None, "CABLE.initialize_coefficients requires params!";
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];

        # Move A, b, and w to specified device.
        self.move_trainable_tensors_to_device(device);
    

    def compute_losses(  
        self,  
        Latent_States   : list[list[torch.Tensor]], 
        loss_type       : str,
        t_Grid          : list[torch.Tensor], 
        params          : numpy.ndarray | None = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        r"""
        TODO


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_States : list[list[torch.Tensor]], len = n_param
            Encoded latent trajectories. The i'th entry contains one tensor of shape (n_t(i), n_z).

        loss_type : str
            Either "MSE" or "MAE".

        t_Grid : list[torch.Tensor], len = n_param
            Time grids corresponding to the latent trajectories.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows used to fetch native coefficient dictionaries.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        loss_LD_list, loss_coef_list, loss_stab_list
            Three lists of scalar tensors, one scalar per parameter.
        """

        # Checks.
        assert params is not None, "SINDy.compute_losses requires params to look up train_coefs";
        assert isinstance(t_Grid, list);
        assert isinstance(Latent_States, list);
        assert len(Latent_States) == len(t_Grid) == params.shape[0];
        assert loss_type in ["MSE", "MAE"];

        # Prepare lists for per-parameter losses. The Trainer is responsible for applying weights
        # and summing these scalar losses into the total objective.
        loss_LD_list : list[torch.Tensor] = [];
        summed_weights : torch.Tensor = torch.zeros((self.n_experts), dtype = torch.float32, device = self.A.device);

        n_param : int = len(t_Grid);
        for i in range(n_param):
            # Fetch this parameter's latent trajectory and time grid.
            ith_t_Grid  : torch.Tensor  = t_Grid[i];
            ith_Z       : torch.Tensor  = Latent_States[i][0]; # [n_t_i, n_z]
            n_t_i       : int           = len(ith_t_Grid);

            # Compute dZ/dt. Uniform grids use the higher-order stencil; nonuniform grids use the
            # nonuniform finite-difference helper.
            if(self.Uniform_t_Grid == True):
                h       : float         = (ith_t_Grid[1] - ith_t_Grid[0]).item();
                dZdt    : torch.Tensor  = Derivative1_Order4(ith_Z, h);
            else:
                dZdt                    = Derivative1_Order2_NonUniform(ith_Z, t_Grid = ith_t_Grid);

            # Evaluate the gate network.
            ith_w_inputs    : torch.Tensor = torch.cat([ith_t_Grid, torch.broadcast_to(params[i, :], (n_t_i, self.n_p))], dim = 1);
            ith_raw_logits  : torch.Tensor = self.w(ith_w_inputs); # [n_t_i, n_experts]

            # Only keep the top k weights at each time, then softmax to get the weights.
            topk_vals, topk_idx = torch.topk(ith_raw_logits, self.k, dim = 1, sorted = False);
            ith_logits : torch.Tensor = torch.full_like(ith_raw_logits, float('-inf')); # [n_t_i, n_experts]
            ith_logits.scatter_(1, topk_idx, topk_vals);
            ith_weights : torch.Tensor = torch.softmax(ith_logits);                     # [n_t_i, n_experts]

            # A_bar[t] = sum_i w[t,i] * A[i]   -> single GEMM: [n_t_i, n_experts] @ [n_experts, n_z* n_z]
            A_flat = self.A.reshape(self.n_experts, self.n_z * self.n_z);
            A_bar = (ith_weights @ A_flat).reshape(n_t_i, self.n_z, self.n_z);          # [n_t_i, self.n_z, n_z]

            # b_bar[t] = sum_i w[t,i] * b[i]   -> single GEMM: [n_t_i, n_experts] @ [n_experts, n_z]
            b_bar = ith_weights @ self.b.squeeze(1);                                    # [n_t_i, n_z]

            # At this point, 
            #   A_bar[t, :, :] = \sum_i ith_weights[t, i] self.A[i, :, :]
            #   b_bar[t, :]    = \sum_i ith_weights[t, i ]self.b[i, 0, :]
            # We can now compute A_bar[t, :, :] ith_Z[t, :] for each t using a batch mat mul.
            Az = torch.bmm(A_bar, ith_Z.unsqueeze(-1)).squeeze(-1);                     # [n_t_i, n_z]

            # Finally, the RHS!
            ith_RHS : torch.Tensor = Az + b_bar;                                            # [n_t_i, n_z]

            # Compute the LD loss.
            if(loss_type == "MSE"):
                loss_LD = self.MSE(dZdt, ith_RHS);
            else:
                loss_LD = self.MAE(dZdt, ith_RHS);

            # Now, sum the weights.
            summed_weights += torch.sum(ith_weights, dim = 0);

        # Coefficient loss is the sum of the Frobenius norms of the matrix portions of each expert,
        # plus the L2 norm of the bias portion. We divide by one over the number of parameters
        # because the coefficient loss is identical for each parameter value; this scaling avoids
        # applying the loss n_params times.
        loss_coef_list : list[torch.Tensor] = [(1./n_param)*(torch.norm(self.A, 'fro') + torch.fro(self.b))]*n_param;

        # Stability loss is the coefficient of variation of the summed weights (per expert) across
        # all parameter values and times.
        loss_stab_list : list[torch.Tensor] = [(1./n_param)*torch.pow(torch.std(summed_weights) / torch.mean(summed_weights), 2)]*n_param;

        # All done :) 
        return loss_LD_list, loss_coef_list, loss_stab_list;


    def RHS(    self,
                Z       : list[list[torch.Tensor | numpy.ndarray]],
                t_Grid  : list[numpy.ndarray | torch.Tensor],
                params  : numpy.ndarray,
                sample  : bool = False) -> list[torch.Tensor | numpy.ndarray]:
        r"""
        Evaluate the affine SINDy right-hand side at a set of latent states and parameters.

        For each parameter value, theta, we evaluate

            z'(t) = A(theta) z(t) + b(theta)

        at each latent state in `Z[i][0]`. Training parameters use exact coefficients from
        `self.train_coefs`; testing parameters use the interpolator mean or a sample.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z : list[list[torch.Tensor | numpy.ndarray]], len = n_param
            The i'th element is a one-element list whose first entry is a tensor/array of shape
            (n_t(i), n_z) or (n_t(i), n_batch(i), n_z).

        t_Grid : list[numpy.ndarray | torch.Tensor], len = n_param
            The i'th entry is a one-dimensional time grid with length n_t(i). The SINDy RHS is
            autonomous, so these values are checked for consistency but not otherwise used.

        params : numpy.ndarray, shape = (n_param, n_p)
            Parameter rows corresponding to the latent states stored in Z.

        sample : bool
            If True, use one interpolator sample for each non-training parameter. Otherwise, use
            interpolator posterior means. Training parameters always use exact training
            coefficients.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        RH_Sides : list[numpy.ndarray | torch.Tensor], len = n_param
            The i'th entry has the same backend and leading dimensions as `Z[i][0]` and last
            dimension n_z. It stores A(theta) z + b evaluated at the supplied states.
        """

       


    def simulate(   self,
                    IC      : list[list[numpy.ndarray | torch.Tensor]],
                    t_Grid  : list[numpy.ndarray      | torch.Tensor],
                    params  : numpy.ndarray,
                    sample  : bool = False) -> list[list[numpy.ndarray | torch.Tensor]]:
        r"""
        Time-integrate the native SINDy latent dynamics.

        Coefficients are fetched from `self.train_coefs` for training parameters and from
        `self.interpolator` for non-training parameters.

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        IC : list[list[numpy.ndarray | torch.Tensor]], len = n_param
            Initial latent states for each coefficient set. SINDy has one IC component.

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

      