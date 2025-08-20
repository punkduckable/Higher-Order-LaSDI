# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  os, sys;
src_path : str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
sys.path.append(src_path);

import  numpy;
import  torch;

from    Physics             import  Physics;
from    Animate                     import  Animate_2D_Grid_Scalar;




# -------------------------------------------------------------------------------------------------
# Utility: 2D Burgers (scalar) solver on a periodic uniform grid
# -------------------------------------------------------------------------------------------------

def solver_2d_burgers(  u0      : numpy.ndarray,
                        t_Grid  : numpy.ndarray,
                        Dx      : float,
                        Dy      : float,
                        nu      : float) -> numpy.ndarray:
    """
    Advance the scalar 2D viscous Burgers equation on a periodic, uniform grid using
    a simple explicit forward-Euler time integration and centered differences in space.

    Governing equation (scalar form):

        u_t = - u (u_x + u_y) + \nu (u_xx + u_yy)

    We enforce periodic boundary conditions with numpy.roll.


    
    Arguments
    -------------------------------------------------------------------------------------------------

    u0: numpy.ndarray, shape = (n_x, n_y)
        Initial condition evaluated on the 2D spatial grid.

    t_Grid: numpy.ndarray, shape = (n_t)
        Monotone non-decreasing time values at which we seek an approximation.

    Dx: float
        Grid spacing in the x-direction; must be positive.

    Dy: float
        Grid spacing in the y-direction; must be positive.

    nu: float
        Viscosity coefficient used in the diffusion term; must be non-negative.



    Returns
    -------------------------------------------------------------------------------------------------

    u: numpy.ndarray, shape = (n_t, n_x, n_y)
        Time history of the solution. u[i, ...] approximates the solution at t = t_Grid[i].
    """

    # Type and shape checks
    assert isinstance(u0, numpy.ndarray),                           "u0 must be a numpy array, not a %s" % type(u0);
    assert isinstance(t_Grid, numpy.ndarray),                       "t_Grid must be a numpy array, not a %s" % type(t_Grid);
    assert isinstance(Dx, float) or isinstance(Dx, numpy.floating), "Dx must be a float, not a %s" % type(Dx);
    assert isinstance(Dy, float) or isinstance(Dy, numpy.floating), "Dy must be a float, not a %s" % type(Dy);
    assert isinstance(nu, float) or isinstance(nu, numpy.floating), "nu must be a float, not a %s" % type(nu);
    assert u0.ndim == 2,                                            "u0 must be 2D with shape (n_x, n_y), not %s" % str(u0.shape);
    assert t_Grid.ndim == 1,                                        "t_Grid must be 1D with time stamps, not %s" % str(t_Grid.shape);
    assert Dx > 0.0 and Dy > 0.0,                                   "Dx and Dy must be positive. Got Dx = %s and Dy = %s" % (Dx, Dy);
    assert nu >= 0.0,                                               "nu must be non-negative. Got nu = %s" % nu;

    # Extract sizes
    n_x: int = u0.shape[0];
    n_y: int = u0.shape[1];
    n_t: int = t_Grid.shape[0];

    # Allocate solution array and set IC
    u: numpy.ndarray = numpy.zeros((n_t, n_x, n_y), dtype=numpy.float32);
    u[0, :, :] = u0.astype(numpy.float32);

    # Precompute denominators
    inv_2Dx: float = 1.0 / (2.0 * Dx);
    inv_2Dy: float = 1.0 / (2.0 * Dy);
    inv_Dx2: float = 1.0 / (Dx * Dx);
    inv_Dy2: float = 1.0 / (Dy * Dy);

    # Time stepping
    for n in range(n_t - 1):
        dt: float = float(t_Grid[n + 1] - t_Grid[n]);
        u_n: numpy.ndarray = u[n, :, :];

        # First derivatives (periodic BCs via roll)
        u_x = (numpy.roll(u_n, -1, axis=0) - numpy.roll(u_n, 1, axis=0)) * inv_2Dx;
        u_y = (numpy.roll(u_n, -1, axis=1) - numpy.roll(u_n, 1, axis=1)) * inv_2Dy;

        # Second derivatives (Laplacian pieces)
        u_xx = (numpy.roll(u_n, -1, axis=0) - 2.0 * u_n + numpy.roll(u_n, 1, axis=0)) * inv_Dx2;
        u_yy = (numpy.roll(u_n, -1, axis=1) - 2.0 * u_n + numpy.roll(u_n, 1, axis=1)) * inv_Dy2;

        # RHS: -u(u_x + u_y) + nu*(u_xx + u_yy)
        rhs = -u_n * (u_x + u_y) + nu * (u_xx + u_yy);

        # Forward Euler update
        u[n + 1, :, :] = u_n + dt * rhs;

    return u;


# -------------------------------------------------------------------------------------------------
# Burgers2D class
# -------------------------------------------------------------------------------------------------

class Burgers2D(Physics):
    def __init__(self, config: dict, param_names: list[str]) -> None:
        """
        Initialize a 2D scalar Burgers solver that conforms to the Physics interface.

        The scalar 2D viscous Burgers equation solved is

            u_t = -u (u_x + u_y) + \nu (u_xx + u_yy)

        The initial condition is defined as a function of two spatial variables and two parameters:
    
            u(0, (x, y)) = exp(-k (x^2 + y^2)) * sin(pi * w * x) * sin(pi * w * y)

        Here, k is a model parameter (controls Gaussian width) and w is a frequency that can be
        provided via the configuration. The governing equation uses the viscosity parameter nu,
        which should be included in the provided parameter vector.


        Arguments
        -------------------------------------------------------------------------------------------------

        config: dict
            Configuration dictionary that must include a 'Burgers2D' sub-dictionary with keys:
            - 'n_x', 'x_min', 'x_max'
            - 'n_y', 'y_min', 'y_max'
            - 'uniform_t_grid'
            Additionally, time-stepping parameters are expected:
            - 'n_t', 't_max'
            Optionally, a frequency for the IC:
            - 'w' (default: 0.5)

        param_names: list[str]
            List of parameter names. Must include 'k' (IC parameter) and 'nu' (viscosity in PDE).
        """

        # Basic checks
        assert isinstance(config, dict),        "config must be a dictionary, not a %s" % type(config);
        assert isinstance(param_names, list),   "param_names must be a list, not a %s" % type(param_names);
        assert 'Burgers2D' in config,           "Config must contain a 'Burgers2D' section";
        assert 'k' in param_names,              "param_names must include 'k' for the IC";
        assert 'nu' in param_names,             "param_names must include 'nu' for the PDE viscosity";

        cfg = config['Burgers2D'];

        # Required spatial settings
        self.n_x        : int   = int(cfg['n_x']);
        self.x_min      : float = float(cfg['x_min']);
        self.x_max      : float = float(cfg['x_max']);
        self.n_y        : int   = int(cfg['n_y']);
        self.y_min      : float = float(cfg['y_min']);
        self.y_max      : float = float(cfg['y_max']);
        self.dx         : float = (self.x_max - self.x_min) / max(1, (self.n_x - 1));
        self.dy         : float = (self.y_max - self.y_min) / max(1, (self.n_y - 1));
        assert self.dx > 0.0 and self.dy > 0.0;

        # Time grid controls (mirror 1D Burgers expectations)
        self.n_t        : int   = int(cfg['n_t']);
        self.t_max      : float = float(cfg['t_max']);

        # IC frequency; can be user-configured
        self.w          : float = float(cfg.get('w', 0.5));

        # Build spatial coordinates for convenience and plotting
        self.x_values   : numpy.ndarray = numpy.linspace(self.x_min, self.x_max, self.n_x, dtype=numpy.float32);
        self.y_values   : numpy.ndarray = numpy.linspace(self.y_min, self.y_max, self.n_y, dtype=numpy.float32);

        # Flattened coordinate list for Physics.X_Positions (shape: (2, n_x*n_y))
        x_coords, y_coords = numpy.meshgrid(self.x_values, self.y_values, indexing = 'ij');  # shapes: (n_x, n_y)
        coordinates   = numpy.vstack([x_coords.reshape(-1), y_coords.reshape(-1)]).astype(numpy.float32);  # (2, n_x*n_y)

        # Call super class initializer
        super().__init__(   spatial_dim     = 2,
                            Frame_Shape     = [1, self.n_x*self.n_y],
                            X_Positions     = coordinates,
                            config          = config,
                            param_names     = param_names,
                            Uniform_t_Grid  = bool(cfg['uniform_t_grid']),
                            n_IC            = 1);

        # Parameter indices for fast lookup
        self.k_idx  : int = self.param_names.index('k');
        self.nu_idx : int = self.param_names.index('nu');

        return;



    def initial_condition(self, param: numpy.ndarray) -> list[numpy.ndarray]:
        """
        Evaluate the 2D initial condition on the structured grid defined by (x, y).

            u(0, (x, y)) = exp(-k (x^2 + y^2)) * sin(pi * w * x) * sin(pi * w * y)



        Arguments
        -------------------------------------------------------------------------------------------------

        param: numpy.ndarray, shape = (2)
            Parameter vector. Expects entries for 'k' (IC) and 'nu' (PDE).



        Returns
        -------------------------------------------------------------------------------------------------

        u0: list[numpy.ndarray], len = 1
            Single-element list. The array has shape (1, n_x*n_y) and contains the initial state.
        """

        # Checks
        assert isinstance(param, numpy.ndarray), "param must be a numpy array, not a %s" % type(param);
        assert param.ndim == 1 and param.shape[0] == self.n_p, "param must be 1D with shape (2), not %s" % str(param.shape);

        # Parameters
        k  : float = float(param[self.k_idx].item());
        w  : float = self.w;

        # Build grid and IC
        x_coords, y_coords = numpy.meshgrid(self.x_values, self.y_values, indexing = 'ij');   # shape (n_x, n_y)
        u0 = numpy.exp(-k * (x_coords**2 + y_coords**2)) * numpy.sin(numpy.pi * w * x_coords) * numpy.sin(numpy.pi * w * y_coords);
        u0 = u0.astype(numpy.float32).reshape(1, -1);   # shape (1, n_x*n_y)

        return [u0];



    def solve(self, param: numpy.ndarray) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Solve the scalar 2D viscous Burgers equation for the provided parameters.

        One parameter ('nu') enters the governing equation as viscosity. Another ('k') shapes the
        initial condition as described in initial_condition.



        Arguments
        -------------------------------------------------------------------------------------------------

        param: numpy.ndarray, shape = (self.n_p)
            Parameter vector containing values for 'k' and 'nu'.



        Returns
        -------------------------------------------------------------------------------------------------

        U: list[torch.Tensor], len = 1
            Single element with shape (n_t, n_x, n_y), the solution snapshots at the times in t_Grid.

        t_Grid: torch.Tensor, shape = (n_t)
            Time stamps corresponding to the solution sequence.
        """

        # Checks
        assert isinstance(param, numpy.ndarray), "param must be a numpy array, not a %s" % type(param);
        assert param.ndim == 1 and param.shape[0] == self.n_p, "param must be 1D with shape (2), not %s" % str(param.shape);

        # Extract parameters
        nu  : float         = float(param[self.nu_idx].item());   # viscosity parameter (nu)

        # Initial condition
        u0  : numpy.ndarray = self.initial_condition(param)[0];   # shape (1, n_x*n_y)

        # Reshape u0 to (n_x, n_y). This is what the solver expects.
        u0 = u0.reshape(self.n_x, self.n_y);

        # Set up the t_Grid.
        t_Grid  : numpy.ndarray = numpy.linspace(0, self.t_max, self.n_t, dtype = numpy.float32);
        if(self.Uniform_t_Grid == False and self.n_t > 2):
            r               : float = 0.2*(t_Grid[1] - t_Grid[0]).item();
            t_adjustments           = numpy.random.uniform(low = -r, high = r, size = (self.n_t - 2));
            t_Grid[1:-1]            = t_Grid[1:-1] + t_adjustments;

        # Solve PDE
        U_np : numpy.ndarray = solver_2d_burgers(   u0     = u0,  
                                                    t_Grid = t_Grid,
                                                    Dx     = self.dx,
                                                    Dy     = self.dy,
                                                    nu     = nu);

        # U_np currently has shape (n_t, n_x, n_y). We need to reshape it to (n_t, 1, n_x*n_y).
        U_np = U_np.reshape(self.n_t, 1, -1);    

        # Package results
        U: list[torch.Tensor] = [torch.tensor(U_np, dtype = torch.float32)];
        return U, torch.tensor(t_Grid, dtype = torch.float32);




# -------------------------------------------------------------------------------------------------
# Main function (for testing)
# -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # ---------------------------------------------------------------------------------------------
    # Example: generate a solution and make an animation
    # ---------------------------------------------------------------------------------------------

    # Choose parameters (k for IC, nu for PDE)
    k_val   : float = 0.5;
    nu_val  : float = 0.01;

    # Build parameter array
    param   : numpy.ndarray = numpy.array([k_val, nu_val], dtype = numpy.float32);

    # Configuration for a moderate grid and timeline
    config  : dict = { 'Burgers2D': {
                            'n_x'              : 51,
                            'x_min'            : -2.0,
                            'x_max'            :  2.0,
                            'n_y'              : 51,
                            'y_min'            : -2.0,
                            'y_max'            :  2.0,
                            'n_t'              : 501,
                            't_max'            : 2.0,
                            'uniform_t_grid'   : True,
                            'w'                : 0.5 } };

    # Instantiate physics and solve
    physics : Burgers2D = Burgers2D(config = config, param_names = ['k', 'nu']);
    U_list, t_Grid = physics.solve(param = param);

    # Prepare data for animation
    U_np    : numpy.ndarray = U_list[0].detach().numpy();         # shape (n_t, 1, n_x*n_y)
    T_np    : numpy.ndarray = t_Grid.detach().numpy();            # shape (n_t)

    # Make movie of the true solution using contour plots
    _ = Animate_2D_Grid_Scalar( U           = U_np,
                                x_values    = physics.x_values,
                                y_values    = physics.y_values,
                                T           = T_np,
                                save_dir    = os.path.join(os.path.dirname(os.path.abspath(os.path.pardir)), "Figures", "Burgers2D"),
                                fname       = "Burgers2D.mp4",
                                levels      = 300,
                                fps         = 30,
                                dpi         = 150);


