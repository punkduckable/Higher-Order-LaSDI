# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

from    typing      import  Callable;

import  numpy; 
import  torch;

r"""
The functions in this file implement Runge-Kutta solvers for a general second-order ODE of the 
following form:

    y''(t)          = f(t,   y(t),   y'(t)).

Here, y takes values in some vector space, V.

To understand where our methods come from, let us first make a few substitutions. First, let 

    z(t)            = (y(t), y'(t)) \in V x V. 

Then,

    z'(t)           = (y'(t), y''(t))
                    = (y'(t), f(t,   y(t),   y'(t)))
                    = (y'(t), f(t,   z(t))).

Now, let g : \mathbb{R} x V x V -> V x V be defined by

    g(t, z(t))      = ( z[d + 1:2d], f(t, z(t)) ).

Then,

    z'(t)           = g(t, z(t))

In other words, we reduce the 2nd order ODE in V to a first order one in V x V. 


We can now apply the Runge-Kutta method to this equation. A general explicit s-step Runge-Kutta 
method generates a sequence of time steps, { z_n }_{n \in \mathbb{N}} \subseteq V. 
using the following rule:

    z_{n + 1}       = z_n + h \sum_{i = 1}^{s} b_i k_i 
    k_i             = g(t_n + c_i h,   z_n + h \sum_{j = 1}^{i - 1} a_{i,j} k_j)
Substituting in the definition of z and g gives

    y_{n + 1}       = y_n  + h \sum_{i = 1}^{s} b_i k_i[:d]                      
    y'_{n + 1}      = y'_n + h \sum_{i = 1}^{s} b_i k_i[d:]

    k_i[:d]         = y'_n + h \sum_{j = 1}^{i - 1} a_{i,j} k_j[d:]
    k_i[d:]         = f(t_n + c_n h,   y_n + h \sum_{j = 1}^{i - 1} a_{i,j} k_j[:d],   y'_n + h \sum_{j = 1}^{i - 1} a_{i,j} k_j[d:])

    
If we substitute the 3rd equation into the 1st and assume that \sum_{i = 1}^{s} b_i = 1, 
then we find that

    y_{n + 1}       = y_n + h\sum_{i = 1}^{s} b_i [ y'_n + h \sum_{j = 1}^{i - 1} a_{i,j} k_j[d:] ]
                    = y_n + h y'_n [ \sum_{i = 1}^{s} b_i ] + h^2 \sum_{i = 1}^{s} \sum_{j = 1}^{i - 1} b_i a_{i,j} k_j[d:]
                    = y_n + h y'_n + h^2 \sum_{j = 1}^{s} k_j[d:] \sum_{i = j + 1}^{s} b_i a_{i,j} 
                    = y_n + h y'_n + h^2 \sum_{j = 1}^{s} k_j[d:] \bar{b_j},
where

    \bar{b_j}       = \sum_{k = j + 1}^{s} b_k a_{k,j}.

Likewise, if we substitute the 3rd equation into the 4th and assume that 
c_i = \sum_{j = 1}^{i - 1} a_{i,j} then we find that

    k_i[d:]         = f(t_n + c_n h,   y_n + h c_i y'_n + h^2 \sum_{j = 1}^{i - 1} k_j[d:] \bar{a_{i,j}},   y'_n + h \sum_{j = 1}^{i - 1} a_{i,j} k_j[d:]),

where

    \bar{a_{i,j}}   = \sum_{k = j + 1}^{i - 1} a_{i,k} a_{k,j}


Replacing k_i[d:] with the new letter l_i gives

    y_{n + 1}       = y_n  + h y'_n + h^2 \sum_{i = 1}^{s} l_i \bar{b_i}
    y'_{n + 1}      = y'_n + h \sum_{i = 1}^{s} b_i l_i

    l_i             = f(t_n + c_i h,   y_n + h c_i y'_n + h^2 \sum_{j = 1}^{i - 1} l_j \bar{a_{i,j}},   y'_n + h \sum_{j = 1}^{i - 1} a_{i,j} l_j)

    \bar{b_i}       = \sum_{k = i + 1}^{s} b_k a_{k,i}
    \bar{a_{i,j}}   = \sum_{k = j + 1}^{i - 1} a_{i,k} a_{k,j}

Thus, given an s-step Runge-Kutta method with coefficients c_1, ... , c_s, b_1, ... , b_s, 
and { a_{i,j} : i = 1, 2, ... , s, j = 1, 2, ... , i - 1}, we can use the equations above to
transform it into a method for solving 2nd order ODEs. 
"""


# -------------------------------------------------------------------------------------------------
# Runge-Kutta Solvers
# -------------------------------------------------------------------------------------------------

def RK1(f       : Callable, 
        y0      : numpy.ndarray | torch.Tensor, 
        Dy0     : numpy.ndarray | torch.Tensor, 
        t_Grid  : numpy.ndarray) -> tuple[numpy.ndarray | torch.Tensor, numpy.ndarray | torch.Tensor]:
    r"""
    This function implements a RK1 or Forward-Euler ODE solver for a second-order ODE of the 
    following form:
    
        y''(t)          = f(t,   y(t),   y'(t)).
    
    Here, y takes values in some vector space, V.
  
    In this function, we implement the Forward Euler (RK1) scheme with the following coefficients:
    
        c_1 = 0
        b_1 = 1
        
    Substituting these coefficients into the equations above gives \bar{b_i} = \bar{a_{i,j}} = 0
    for each i,j. Thus, 

        y_{n + 1}       = y_n  + h y'_n 
        y'_{n + 1}      = y'_n + h l_1

        l_1             = f(t_n, y_n, y'_n)
    
    This is the method we implement.


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    f : Callable
        The right-hand side of the ODE (see the top of this doc string). This is a function whose 
        domain and co-domain are \mathbb{R} x V x V and V, respectively. Thus, we assume that 
        f(t, y(t), y'(t)) = y''(t).

    y0 : numpy.ndarray or torch.Tensor, shape = Dy0.shape
        A numpy.ndarray or torch.Tensor holding the initial displacement (i.e., y0 = y(t0)), where
        t0 = t_Grid[0]. Must have the same type as Dy0.

    Dy0: numpy.ndarray or torch.Tensor, shape = y0.shape
        A numpy.ndarray or torch.Tensor holding the initial velocity (i.e., Dy0 = y'(t0)). Must 
        have the same type as y0.

    t_Grid : numpy.ndarray, shape = (n_t)
        i'th element holds the i'th time value. We assume the elements of this array form an 
        increasing sequence.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Two numpy.ndarray or torch.Tensor objects: D, V. 
    
    D : numpy.ndarray or torch.Tensor, shape = (n_t,) + y0.shape
        D[i] an approximation to the displacement at time t_Grid[i]. Thus, D[i] \approx 
        y(t_Grid[i). Will have the same type as y0/Dy0.
    
    V : numpy.ndarray or torch.Tensor, shape = (n_t,) + Dy0.shape
        V[i] holds an approximation to the velocity at time t_Grid[i]. Thus, V[i] \approx 
        y'(t_Grid[i). Will have the same type as y0/Dy0.
    """

    # First, run checks.
    assert(isinstance(y0,       numpy.ndarray)  or isinstance(y0,       torch.Tensor));
    assert(isinstance(Dy0,      numpy.ndarray)  or isinstance(Dy0,      torch.Tensor));
    assert(isinstance(t_Grid,   numpy.ndarray));
    assert(type(y0)         == type(Dy0));
    assert(len(t_Grid.shape) == 1);
    assert(y0.shape         == Dy0.shape);

    # Next, fetch N.
    N : int = t_Grid.size;

    # Initialize lists to hold the results.
    D_list : list[torch.Tensor | numpy.ndarray] = [y0];
    V_list : list[torch.Tensor | numpy.ndarray] = [Dy0];

    # Now, run the time stepping!
    for n in range(N - 1):
        # Fetch the current time, displacement, velocity.
        tn  : float                         = t_Grid[n];
        yn  : numpy.ndarray | torch.Tensor  = D_list[n];
        Dyn : numpy.ndarray | torch.Tensor  = V_list[n];
        hn  : float                         = t_Grid[n + 1] - t_Grid[n];

        # Compute l_1.
        l_1 = f(tn, yn, Dyn);

        # Now compute y{n + 1} and Dy{n + 1}.
        yn1     : numpy.ndarray | torch.Tensor  = yn  + hn*Dyn;
        Dyn1    : numpy.ndarray | torch.Tensor  = Dyn + hn*l_1

        # All done with this step!
        D_list.append(yn1);
        V_list.append(Dyn1);

    # Stack the results.
    if(isinstance(y0, numpy.ndarray)):
        D = numpy.stack(D_list, axis = 0);  # shape = (n_t, y0.shape)
        V = numpy.stack(V_list, axis = 0);  # shape = (n_t, Dy0.shape)
    elif(isinstance(y0, torch.Tensor)):
        D = torch.stack(D_list, dim = 0);  # shape = (n_t, y0.shape)
        V = torch.stack(V_list, dim = 0);  # shape = (n_t, Dy0.shape)

    # All done!
    return (D, V);



def RK2(f       : Callable, 
        y0      : numpy.ndarray | torch.Tensor, 
        Dy0     : numpy.ndarray | torch.Tensor, 
        t_Grid  : numpy.ndarray) -> tuple[numpy.ndarray | torch.Tensor, numpy.ndarray | torch.Tensor]:
    r"""
    This function implements a RK2 based ODE solver for a second-order ODE of the following form:

        y''(t)          = f(t,   y(t),   y'(t)).

    Here, y takes values in some vector space, V.
  
    In this function, we implement the classic RK2 scheme with the following coefficients:

        c_1 = 0
        c_2 = 1

        b_1 = 1/2
        b_2 = 1/2
        
        a_{2,1}         = 1
    
    Substituting these coefficients into the equations above gives

        \bar{b_1}       = b_2 a_{2,1}                           = 1/2
        \bar{b_2}                                               = 0
    
    \bar{a_{i,j}} = 0 for all i, j. Thus,

        y_{n + 1}       = y_n  + h y'_n + (h^2/2) l_1
        y'_{n + 1}      = y'_n + (h/2)( l_1 + l_2 )

        l_1             = f(t_n,        y_n,                            y'_n)
        l_2             = f(t_n + h,    y_n + h y'_n,                   y'_n + h l_1)

    This is the method we implement.


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    f : Callable
        The right-hand side of the ODE (see the top of this doc string). This is a function whose 
        domain and co-domain are \mathbb{R} x V x V and V, respectively. Thus, we assume that 
        f(t, y(t), y'(t)) = y''(t).

    y0 : numpy.ndarray or torch.Tensor, shape = Dy0.shape
        A numpy.ndarray or torch.Tensor holding the initial displacement (i.e., y0 = y(t0)), where
        t0 = t_Grid[0]. Must have the same type as Dy0.

    Dy0: numpy.ndarray or torch.Tensor, shape = y0.shape
        A numpy.ndarray or torch.Tensor holding the initial velocity (i.e., Dy0 = y'(t0)). Must 
        have the same type as y0.

    t_Grid : numpy.ndarray, shape = (n_t)
        i'th element holds the i'th time value. We assume the elements of this array form an 
        increasing sequence.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Two numpy.ndarray or torch.Tensor objects: D, V. 
    
    D : numpy.ndarray or torch.Tensor, shape = (n_t,) + y0.shape
        D[i] an approximation to the displacement at time t_Grid[i]. Thus, D[i] \approx 
        y(t_Grid[i). Will have the same type as y0/Dy0.
    
    V : numpy.ndarray or torch.Tensor, shape = (n_t,) + Dy0.shape
        V[i] holds an approximation to the velocity at time t_Grid[i]. Thus, V[i] \approx 
        y'(t_Grid[i). Will have the same type as y0/Dy0.
    """

    # First, run checks.
    assert(isinstance(y0,       numpy.ndarray)  or isinstance(y0,       torch.Tensor));
    assert(isinstance(Dy0,      numpy.ndarray)  or isinstance(Dy0,      torch.Tensor));
    assert(isinstance(t_Grid,    numpy.ndarray));
    assert(type(y0)             == type(Dy0));
    assert(len(t_Grid.shape)    == 1);
    assert(y0.shape             == Dy0.shape);

    # Next, fetch N.
    N : int = t_Grid.size;

    # Initialize lists to hold the results.
    D_list : list[torch.Tensor | numpy.ndarray] = [y0];
    V_list : list[torch.Tensor | numpy.ndarray] = [Dy0];

    # Now, run the time stepping!
    for n in range(N - 1):
        # Fetch the current time, displacement, velocity.
        tn  : float                         = t_Grid[n];
        yn  : numpy.ndarray | torch.Tensor  = D_list[n];
        Dyn : numpy.ndarray | torch.Tensor  = V_list[n];
        hn  : float                         = t_Grid[n + 1] - t_Grid[n];

        # Compute l_1, l_2.
        l_1 = f(tn,         yn,             Dyn);
        l_2 = f(tn + hn,    yn + hn*Dyn ,   Dyn + hn*l_1);

        # Now compute y{n + 1} and Dy{n + 1}.
        yn1     : numpy.ndarray | torch.Tensor  = yn + hn*Dyn + (hn*hn/2)*l_1;
        Dyn1    : numpy.ndarray | torch.Tensor  = Dyn + (hn/2)*(l_1 + l_2);

        # All done with this step!
        D_list.append(yn1);
        V_list.append(Dyn1);

    # Stack the results.
    if(isinstance(y0, numpy.ndarray)):
        D = numpy.stack(D_list, axis = 0);  # shape = (n_t, y0.shape)
        V = numpy.stack(V_list, axis = 0);  # shape = (n_t, Dy0.shape)
    elif(isinstance(y0, torch.Tensor)):
        D = torch.stack(D_list, dim = 0);  # shape = (n_t, y0.shape)
        V = torch.stack(V_list, dim = 0);  # shape = (n_t, Dy0.shape)

    # All done!
    return (D, V);



def RK4(f       : Callable, 
        y0      : numpy.ndarray | torch.Tensor, 
        Dy0     : numpy.ndarray | torch.Tensor, 
        t_Grid  : numpy.ndarray) -> tuple[numpy.ndarray | torch.Tensor, numpy.ndarray | torch.Tensor]:
    r"""
    This function implements a RK4 based ODE solver for a second-order ODE of the following form:

        y''(t)          = f(t,   y(t),   y'(t)).

    Here, y takes values in some vector space, V.
  
    In this function, we implement the classic RK4 scheme with the following coefficients:

        c_1 = 0
        c_2 = 1/2
        c_3 = 1/2
        c_4 = 1

        b_1 = 1/6
        b_2 = 1/3
        b_3 = 1/3
        b_4 = 1/6

        a_{2,1}         = 1/2
        a_{3,2}         = 1/2
        a_{4,3}         = 1
    
    Substituting these coefficients into the equations above gives

        \bar{b_1}       = 1/6
        \bar{b_2}       = 1/6
        \bar{b_3}       = 1/6
        \bar{b_4}       = 0

        \bar{a_{3,1}}   = a_{3,2} a_{2,1}                       = 1/4
        \bar{a_{4,1}}   = a_{4,2} a_{2,1} + a_{4,3} a_{3, 1}    = 0
        \bar{a_{4,2}}   = a_{4,3} a_{3,2}                       = 1/2
    
    and \bar{a_{i,j}} = 0 for all other i, j. Thus,
    
        y_{n + 1}       = y_n  + h y'_n + (h^2/6)[ l_1 + l_2 + l_3 ]
        y'_{n + 1}      = y'_n + h [ l_1/6 + l_2/3 + l_3/3 + l_4/6 ]

        l_1             = f(t_n,        y_n,                            y'_n)
        l_2             = f(t_n + h/2,  y_n + (h/2) y'_n,               y'_n + (h/2) l_1)
        l_3             = f(t_n + h/2,  y_n + (h/2) y'_n + (h^2/4) l_1, y'_n + (h/2) l_2)
        l_4             = f(t_n + h,    y_n + h y'_n + (h^2/2) l_2,     y'_n + h l_3)

    This is the method we implement.


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    f : Callable
        The right-hand side of the ODE (see the top of this doc string). This is a function whose 
        domain and co-domain are \mathbb{R} x V x V and V, respectively. Thus, we assume that 
        f(t, y(t), y'(t)) = y''(t).

    y0 : numpy.ndarray or torch.Tensor, shape = Dy0.shape
        A numpy.ndarray or torch.Tensor holding the initial displacement (i.e., y0 = y(t0)), where
        t0 = t_Grid[0]. Must have the same type as Dy0.

    Dy0: numpy.ndarray or torch.Tensor, shape = y0.shape
        A numpy.ndarray or torch.Tensor holding the initial velocity (i.e., Dy0 = y'(t0)). Must 
        have the same type as y0.

    t_Grid : numpy.ndarray, shape = (n_t)
        i'th element holds the i'th time value. We assume the elements of this array form an 
        increasing sequence.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Two numpy.ndarray or torch.Tensor objects: D, V. 
    
    D : numpy.ndarray or torch.Tensor, shape = (n_t,) + y0.shape
        D[i] an approximation to the displacement at time t_Grid[i]. Thus, D[i] \approx 
        y(t_Grid[i). Will have the same type as y0/Dy0.
    
    V : numpy.ndarray or torch.Tensor, shape = (n_t,) + Dy0.shape
        V[i] holds an approximation to the velocity at time t_Grid[i]. Thus, V[i] \approx 
        y'(t_Grid[i). Will have the same type as y0/Dy0.
    """

    # First, run checks.
    assert(isinstance(y0,       numpy.ndarray)  or isinstance(y0,       torch.Tensor));
    assert(isinstance(Dy0,      numpy.ndarray)  or isinstance(Dy0,      torch.Tensor));
    assert(isinstance(t_Grid,   numpy.ndarray));
    assert(type(y0)         == type(Dy0));
    assert(len(t_Grid.shape) == 1);
    assert(y0.shape         == Dy0.shape);

    # Next, fetch N.
    N : int = t_Grid.size;

    # Initialize lists to hold the results.
    D_list : list[torch.Tensor | numpy.ndarray] = [y0];
    V_list : list[torch.Tensor | numpy.ndarray] = [Dy0];

    # Now, run the time stepping!
    for n in range(N - 1):
        # Fetch the current time, displacement, velocity.
        tn  : float                         = t_Grid[n];
        yn  : numpy.ndarray | torch.Tensor  = D_list[n];
        Dyn : numpy.ndarray | torch.Tensor  = V_list[n];
        hn  : float                         = t_Grid[n + 1] - t_Grid[n];

        # Compute l_1, l_2, l_3, l_4.
        l_1 = f(tn,         yn,                                 Dyn);
        l_2 = f(tn + hn/2,  yn + (hn/2)*Dyn ,                   Dyn + (hn/2)*l_1);
        l_3 = f(tn + hn/2,  yn + (hn/2)*Dyn + (hn*hn/4)*l_1,    Dyn + (hn/2)*l_2);
        l_4 = f(tn + hn,    yn + hn*Dyn + (hn*hn/2)*l_2,        Dyn + hn*l_3);

        # Now compute y{n + 1} and Dy{n + 1}.
        yn1     : numpy.ndarray | torch.Tensor  = yn + hn*Dyn + (hn*hn/6)*(l_1 + l_2 + l_3);
        Dyn1    : numpy.ndarray | torch.Tensor  = Dyn + (hn/6)*(l_1 + 2*l_2 + 2*l_3 + l_4);

        # All done with this step!
        D_list.append(yn1);
        V_list.append(Dyn1);
    
    # Stack the results.
    if(isinstance(y0, numpy.ndarray)):
        D = numpy.stack(D_list, axis = 0);  # shape = (n_t, y0.shape)
        V = numpy.stack(V_list, axis = 0);  # shape = (n_t, Dy0.shape)
    elif(isinstance(y0, torch.Tensor)):
        D = torch.stack(D_list, dim = 0);  # shape = (n_t, y0.shape)
        V = torch.stack(V_list, dim = 0);  # shape = (n_t, Dy0.shape)

    # All done!
    return (D, V);


# -------------------------------------------------------------------------------------------------
# Batched RK4
# -------------------------------------------------------------------------------------------------


def _rk4_affine_step(
        yn                : torch.Tensor,
        vn                : torch.Tensor,
        K_start           : torch.Tensor,
        K_mid             : torch.Tensor,
        K_end             : torch.Tensor,
        C_start           : torch.Tensor,
        C_mid             : torch.Tensor,
        C_end             : torch.Tensor,
        b_start           : torch.Tensor,
        b_mid             : torch.Tensor,
        b_end             : torch.Tensor,
        h                 : torch.Tensor,
        h_half            : torch.Tensor,
        h_squared_quarter : torch.Tensor,
        h_squared_half    : torch.Tensor,
        h_squared_sixth   : torch.Tensor,
        h_sixth           : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Advance one batched affine RK4 interval, preserving the eager equations."""

    # Each batched matrix-vector product handles all windows. For item i,
    # stage 1 is l1_i = K_start,i @ y_n,i + C_start,i @ v_n,i + b_start,i.
    l1 : torch.Tensor = (torch.bmm(K_start, yn.unsqueeze(-1)).squeeze(-1)
                         + torch.bmm(C_start, vn.unsqueeze(-1)).squeeze(-1) + b_start);

    # Stage 2: y2_i = y_n,i + h_i*v_n,i/2, v2_i = v_n,i + h_i*l1_i/2,
    # then l2_i = K_mid,i @ y2_i + C_mid,i @ v2_i + b_mid,i.
    y2 : torch.Tensor = yn + h_half * vn;
    v2 : torch.Tensor = vn + h_half * l1;
    l2 : torch.Tensor = (torch.bmm(K_mid, y2.unsqueeze(-1)).squeeze(-1)
                         + torch.bmm(C_mid, v2.unsqueeze(-1)).squeeze(-1) + b_mid);

    # Stage 3 reuses midpoint coefficients, but evaluates them at a new state.
    y3 : torch.Tensor = y2 + h_squared_quarter * l1;
    v3 : torch.Tensor = vn + h_half * l2;
    l3 : torch.Tensor = (torch.bmm(K_mid, y3.unsqueeze(-1)).squeeze(-1)
                         + torch.bmm(C_mid, v3.unsqueeze(-1)).squeeze(-1) + b_mid);

    # Stage 4 uses the end coefficients.
    y4 : torch.Tensor = yn + h * vn + h_squared_half * l2;
    v4 : torch.Tensor = vn + h * l3;
    l4 : torch.Tensor = (torch.bmm(K_end, y4.unsqueeze(-1)).squeeze(-1)
                         + torch.bmm(C_end, v4.unsqueeze(-1)).squeeze(-1) + b_end);

    # Classical second-order RK4 update.
    yn1 : torch.Tensor = yn + h * vn + h_squared_sixth * (l1 + l2 + l3);
    vn1 : torch.Tensor = vn + h_sixth * (l1 + 2 * l2 + 2 * l3 + l4);
    return yn1, vn1;


# Compile only one interval. Compiling the 250-interval loop would create a very
# large graph; reusing this callable also lets AOTAutograd fuse pointwise backward
# work without changing the reference CPU path.
_compiled_rk4_affine_step = torch.compile(_rk4_affine_step, fullgraph=True);


def RK4_affine_batched(
        y0      : torch.Tensor,
        Dy0     : torch.Tensor,
        t_grid  : torch.Tensor,
        lengths : torch.Tensor,
        K       : torch.Tensor,
        C       : torch.Tensor,
        b       : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Integrate a batch of affine second-order ODEs using classical RK4.

    Each batch item solves ``y''(t) = K(t) @ y(t) + C(t) @ y'(t) + b(t)``. The
    three coefficient sets for each interval correspond to its start, midpoint,
    and end. RK stages 2 and 3 reuse the midpoint coefficients, but evaluate
    them at different intermediate states, so their accelerations differ.

    CUDA compiles one RK4 interval to fuse tiny pointwise operations; CPU uses
    the same equations eagerly. The existing scalar RK4 remains the independent
    CPU/NumPy reference.

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    y0 : torch.Tensor, shape = (B, d)
        Initial displacement for each batch item.

    Dy0 : torch.Tensor, shape = (B, d)
        Initial velocity for each batch item.

    t_grid : torch.Tensor, shape = (B, T)
        Time grid for each item. Valid times increase strictly; after the final
        valid time, padding repeats that time through column T - 1.

    lengths : torch.Tensor, shape = (B,)
        Integer number of valid time points per item, satisfying 1 <= lengths <= T.

    K : torch.Tensor, shape = (B, T - 1, 3, d, d)
        Effective displacement matrices at each interval's start, midpoint, and end. Specifically,
        K[i, j, :, :, :] contains three effective K matrices for the i'th batch item during its
        j'th RK4 step. They represent K at the step's start, midpoint, and end, in that order.

    C : torch.Tensor, shape = (B, T - 1, 3, d, d)
        Effective velocity matrices with the same stage ordering as K.

    b : torch.Tensor, shape = (B, T - 1, 3, d)
        Effective affine biases with the same stage ordering as K.

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    D, V : tuple[torch.Tensor, torch.Tensor], each shape = (B, T, d)
        Displacement and velocity trajectories, including frozen padded frames. Specifically,
        D[i, j, :] gives the solved displacement at the j'th time step for the i'th batch item.
        Both retain gradients through the initial states and coefficients.

    All floating inputs must have the same dtype and device; lengths must be on
    that device. Value checks for lengths, times, and finite coefficients are
    left to the caller to avoid GPU synchronization. Inactive intervals use a
    zero step size, so finite RHS values leave their states unchanged.
    """

    # Checks.
    if y0.ndim != 2 or Dy0.shape != y0.shape:
        raise ValueError("y0 and Dy0 must have matching shape [B, d]");
    batch_size : int = y0.shape[0];
    state_size : int = y0.shape[1];
    if t_grid.ndim != 2 or t_grid.shape[0] != batch_size or t_grid.shape[1] < 1:
        raise ValueError("t_grid must have shape [B, T] with T >= 1");
    if lengths.shape != (batch_size,) or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("lengths must have shape [B] and an integer dtype");
    n_steps : int = t_grid.shape[1] - 1;
    if K.shape != (batch_size, n_steps, 3, state_size, state_size):
        raise ValueError("K must have shape [B, T-1, 3, d, d]");
    if C.shape != K.shape or b.shape != (batch_size, n_steps, 3, state_size):
        raise ValueError("C and b must have shapes [B, T-1, 3, d, d] and [B, T-1, 3, d]");
    floating_tensors : tuple[torch.Tensor, ...] = (y0, Dy0, t_grid, K, C, b);
    if not y0.is_floating_point() or any(t.dtype != y0.dtype or t.device != y0.device for t in floating_tensors):
        raise ValueError("states, time grid, and coefficients must share floating dtype and device");
    if lengths.device != y0.device:
        raise ValueError("lengths must be on the state device");

    # Mask every invalid interval once, outside the sequential RK loop. A zero h
    # makes the RK update exactly retain the last valid state; no per-row Python
    # branch or per-step torch.where kernel is needed.
    active  : torch.Tensor = torch.arange(n_steps, device=y0.device)[None, :] < (lengths[:, None] - 1); # [B, T-1]
    h_all   : torch.Tensor = torch.where(active, t_grid[:, 1:] - t_grid[:, :-1], 0);                    # [B, T-1]

    # Compute all step-size factors across the time grid before the sequential loop.
    # Slicing these tensors below is a view, rather than a GPU operation per factor and step.
    h_half_all            : torch.Tensor = h_all / 2;
    h_squared_all         : torch.Tensor = h_all * h_all;
    h_squared_quarter_all : torch.Tensor = h_squared_all / 4;
    h_squared_half_all    : torch.Tensor = h_squared_all / 2;
    h_squared_sixth_all   : torch.Tensor = h_squared_all / 6;
    h_sixth_all           : torch.Tensor = h_all / 6;

    # Unbind time and stage views once. Repeated K[:, n, stage] indexing creates
    # separate select/slice backward nodes that scatter into full-size zeros.
    K_steps : tuple[tuple[torch.Tensor, ...], ...] = tuple(step.unbind(dim=1) for step in K.unbind(dim=1));
    C_steps : tuple[tuple[torch.Tensor, ...], ...] = tuple(step.unbind(dim=1) for step in C.unbind(dim=1));
    b_steps : tuple[tuple[torch.Tensor, ...], ...] = tuple(step.unbind(dim=1) for step in b.unbind(dim=1));

    D_list : list[torch.Tensor] = [y0];  # Each entry has shape [B, d].
    V_list : list[torch.Tensor] = [Dy0]; # Each entry has shape [B, d].
    step = _compiled_rk4_affine_step if y0.is_cuda else _rk4_affine_step;
    for n in range(n_steps):
        K_start, K_mid, K_end = K_steps[n];
        C_start, C_mid, C_end = C_steps[n];
        b_start, b_mid, b_end = b_steps[n];

        # Fetch y, y' (v) from the previous time step
        yn          : torch.Tensor  = D_list[-1];           # [B, d]
        vn          : torch.Tensor  = V_list[-1];           # [B, d]

        # Fetch this interval's precomputed step-size factors as [B, 1] views.
        h                 : torch.Tensor = h_all[:, n:n + 1];
        h_half            : torch.Tensor = h_half_all[:, n:n + 1];
        h_squared_quarter : torch.Tensor = h_squared_quarter_all[:, n:n + 1];
        h_squared_half    : torch.Tensor = h_squared_half_all[:, n:n + 1];
        h_squared_sixth   : torch.Tensor = h_squared_sixth_all[:, n:n + 1];
        h_sixth           : torch.Tensor = h_sixth_all[:, n:n + 1];

        yn1, vn1 = step(
            yn, vn, K_start, K_mid, K_end, C_start, C_mid, C_end,
            b_start, b_mid, b_end, h, h_half, h_squared_quarter,
            h_squared_half, h_squared_sixth, h_sixth);
        D_list.append(yn1);
        V_list.append(vn1);

    # All done :)
    return torch.stack(D_list, dim=1), torch.stack(V_list, dim=1);
