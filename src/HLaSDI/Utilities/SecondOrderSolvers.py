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


# How many RK4 steps should we batch into a single compiled chunk?
_RK4_AFFINE_CHUNK_SIZE : int = 8;


def _rk4_affine_chunk(
        yn                 : torch.Tensor,  # [B, d, 1]
        vn                 : torch.Tensor,  # [B, d, 1]
        KC_chunk           : torch.Tensor,  # [8, 3, B, d, 2d]
        b_chunk            : torch.Tensor,  # [8, 3, B, d]
        step_factors_chunk : torch.Tensor,  # [8, 6, padded B]
        ) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Advance eight affine RK4 intervals, retaining every intermediate state."""

    # Initialize lists to hold the D, V computed at each step in this chunk.
    D_list : list[torch.Tensor] = [];
    V_list : list[torch.Tensor] = [];

    # Loop through the steps in this chunk.
    for KC_n, b_n, step_factors_n in zip(
            KC_chunk.unbind(dim=0), b_chunk.unbind(dim=0), step_factors_chunk.unbind(dim=0)):
        # KC_n.shape = [3, B, d, 2d], and b_n.shape = [3, B, d]. The leading axis specifies
        # the stage. Unbinding along this axis gives us the KC and b coefficients for the
        # start, mid, and end stages.
        KC_start, KC_mid, KC_end    = KC_n.unbind(dim=0);
        b_start, b_mid, b_end       = b_n.unbind(dim=0);

        # Keep states and accelerations as column vectors for the batched matmuls.
        # Each factor row has a 16-byte-aligned start; select the active batch entries.
        h, h_half, h_squared_quarter, h_squared_half, h_squared_sixth, h_sixth = (
            step_factors_n[:, :yn.shape[0], None, None].unbind(dim=0));

        # Stage 1:
        #   l_1 = K(t_n)y_n + C(t_n)y'_n + b(t_n)
        # The "start" stage coefficients hold K(t_n), C(t_n), and b(t_n).
        # baddbmm adds the bias to the batched matrix-vector product.
        l1 : torch.Tensor = torch.baddbmm(
            b_start.unsqueeze(-1), KC_start, torch.cat((yn, vn), dim=1));

        # Stage 2:
        #   l_2 = K(t_n + h/2)[y_n + (h/2)y'_n] + C(t_n + h/2)[y'_n + (h/2)l_1] + b(t_n + h/2)
        # The "midpoint" stage coefficients hold K(t_n + h/2), C(t_n + h/2), and b(t_n + h/2).
        y2 : torch.Tensor = yn + h_half * vn;
        v2 : torch.Tensor = vn + h_half * l1;
        l2 : torch.Tensor = torch.baddbmm(
            b_mid.unsqueeze(-1), KC_mid, torch.cat((y2, v2), dim=1));

        # Stage 3:
        #   l_3 = K(t_n + h/2)[y_n + (h/2)y'_n + (h^2/4)l_1] + C(t_n + h/2)[y'_n + (h/2)l_2] + b(t_n + h/2)
        # This stage reuses midpoint coefficients at the updated half-step state.
        y3 : torch.Tensor = y2 + h_squared_quarter * l1;
        v3 : torch.Tensor = vn + h_half * l2;
        l3 : torch.Tensor = torch.baddbmm(
            b_mid.unsqueeze(-1), KC_mid, torch.cat((y3, v3), dim=1));

        # Stage 4:
        #   l_4 = K(t_n + h)[y_n + h y'_n + (h^2/2) l_2] + C(t_n + h)[y'_n + h l_3] + b(t_n + h)
        # The "end" stage coefficients hold K(t_n + h), C(t_n + h), and b(t_n + h)
        y4 : torch.Tensor = yn + h * vn + h_squared_half * l2;
        v4 : torch.Tensor = vn + h * l3;
        l4 : torch.Tensor = torch.baddbmm(
            b_end.unsqueeze(-1), KC_end, torch.cat((y4, v4), dim=1));

        # Compute the second-order RK4 update for this step.
        yn = yn + h * vn + h_squared_sixth * (l1 + l2 + l3);
        vn = vn + h_sixth * (l1 + 2 * l2 + 2 * l3 + l4);

        # Append the solved displacement, velocity from the end of this step to the lists.
        D_list.append(yn);
        V_list.append(vn);

    # All done :)
    return torch.stack(D_list, dim=0), torch.stack(V_list, dim=0);


# Compile a fixed-size chunk rather than specializing the full rollout by its
# changing duration. The CPU path calls this same chunk eagerly.
_compiled_rk4_affine_chunk = torch.compile(_rk4_affine_chunk, fullgraph = True);

def RK4_affine_batched(
        y0      : torch.Tensor,
        Dy0     : torch.Tensor,
        t_grid  : torch.Tensor,
        lengths : torch.Tensor,
        K       : torch.Tensor,
        C       : torch.Tensor,
        b       : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Integrate a batch of B \in \mathbb{N} affine second-order ODEs using classical RK4.

    For each batch item, the goal is to solve ``y''(t) = K(t) @ y(t) + C(t) @ y'(t) + b(t)`` for 
    some K(t), C(t), and b(t) specific to that batch. For each batch item, y lives in \mathbb{R}^d.
    The caller supplies K, C, and b at the start, midpoint, and end of each interval. A grid
    with T time points has T - 1 RK4 intervals.

    CUDA compiles eight RK4 intervals together; CPU evaluates the same intervals eagerly.

    Assumptions:
        - all floating point tensor inputs have the same dtype and device


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    y0 : torch.Tensor, shape = (B, d)
        Initial displacement for each batch item. Here, B is the batch size.

    Dy0 : torch.Tensor, shape = (B, d)
        Initial velocity for each batch item.

    t_grid : torch.Tensor, shape = (T, B)
        The (t, b) element is the t'th time point for batch item b. After its final valid time,
        padding repeats that time through row T - 1.

    lengths : torch.Tensor, shape = (B,)
        The b'th element specifies its number of valid time points, with 1 <= lengths[b] <= T.
        Inactive intervals use zero step size, so padded output frames repeat the last valid state.
        Value bounds are not checked here because that would synchronize CUDA.

    K : torch.Tensor, shape = (T - 1, 3, B, d, d)
        Effective displacement matrices at each interval's start, midpoint, and end. Specifically,
        K[j, s, i, :, :] gives the `K` matrix for interval j, stage s, and batch item i. The stage
        index, s, specifies the start, midpoint, or end time, in that order. This tensor must be 
        contiguous.

    C : torch.Tensor, shape = (T - 1, 3, B, d, d)
        Contiguous effective velocity matrices with the same ordering as K.

    b : torch.Tensor, shape = (T - 1, 3, B, d)
        Contiguous effective affine biases with the same ordering as K.

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    D, V : tuple[torch.Tensor, torch.Tensor], each shape = (T, B, d)
        Displacement and velocity trajectories, including frozen padded frames. Specifically,
        D[j, i, :] gives the solved displacement at the j'th time step for the i'th batch item.
        Both retain gradients through the initial states and coefficients.
    """

    # Checks.
    if y0.ndim != 2 or Dy0.shape != y0.shape:
        raise ValueError("y0 and Dy0 must have matching shape [B, d]");
    B : int = y0.shape[0];
    d : int = y0.shape[1];
    if t_grid.ndim != 2 or t_grid.shape[1] != B or t_grid.shape[0] < 1:
        raise ValueError("t_grid must have shape [T, B] with T >= 1");
    n_steps : int = t_grid.shape[0] - 1;
    if lengths.shape != (B,) or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("lengths must have shape [B] and an integer dtype");
    if K.shape != (n_steps, 3, B, d, d):
        raise ValueError("K must have shape [T-1, 3, B, d, d]");
    if C.shape != K.shape or b.shape != (n_steps, 3, B, d):
        raise ValueError("C and b must have shapes [T-1, 3, B, d, d] and [T-1, 3, B, d]");
    if not K.is_contiguous() or not C.is_contiguous() or not b.is_contiguous():
        raise ValueError("K, C, and b must be contiguous time-major tensors");
    floating_tensors : tuple[torch.Tensor, ...] = (y0, Dy0, t_grid, K, C, b);
    if not y0.is_floating_point() or any(t.dtype != y0.dtype or t.device != y0.device for t in floating_tensors):
        raise ValueError("states, time grid, and coefficients must share floating dtype and device");
    if lengths.device != y0.device:
        raise ValueError("lengths must be on the state device");
    if n_steps == 0:
        return y0.unsqueeze(0), Dy0.unsqueeze(0);

    # ---------------------------------------------------------------------------------------------
    # Compute, pad the "step factors" (step size and values derived from them)

    # Item b has lengths[b] valid time points, hence lengths[b] - 1 valid intervals.
    # Zero step sizes after that freeze the final valid state through padded frames.
    active  : torch.Tensor = torch.arange(n_steps, device = y0.device)[:, None] < (lengths[None, :] - 1); # [T-1, B]
    h_all   : torch.Tensor = torch.where(active, t_grid[1:, :] - t_grid[:-1, :], 0);                       # [T-1, B]

    # Pad each time row of h to a multiple of 16 bytes. In the CUDA/Inductor path, an unaligned
    # row stride caused extra device-to-device copies; padding avoids them.
    elements_per_16_bytes : int = max(1, 16 // y0.element_size());
    num_padding_columns   : int = (-B) % elements_per_16_bytes;
    if num_padding_columns:
        h_all = torch.nn.functional.pad(h_all, (0, num_padding_columns)); # [T-1, padded B]

    # Now, pre-compute the "step factors"; all the various modifications of h we need to 
    # evaluate one RK4 step.
    h_half_all            : torch.Tensor = h_all / 2;                                                   # [T-1, padded B]
    h_squared_all         : torch.Tensor = h_all * h_all;                                               # [T-1, padded B]
    h_squared_quarter_all : torch.Tensor = h_squared_all / 4;                                           # [T-1, padded B]
    h_squared_half_all    : torch.Tensor = h_squared_all / 2;                                           # [T-1, padded B]
    h_squared_sixth_all   : torch.Tensor = h_squared_all / 6;                                           # [T-1, padded B]
    h_sixth_all           : torch.Tensor = h_all / 6;                                                   # [T-1, padded B]

    # Stack along the factor axis: [T-1, 6, padded B]. No further permutation is needed.
    step_factors : torch.Tensor = torch.stack(
        (h_all, h_half_all, h_squared_quarter_all, h_squared_half_all,
         h_squared_sixth_all, h_sixth_all), dim=1);
    step_factors = step_factors.contiguous();                           # [T-1, 6, padded B]

    # ---------------------------------------------------------------------------------------------
    # Execute RK4 solve in chunks of steps.

    # Pack K and C so each forward RK stage uses one rather than two batched matvecs.
    KC : torch.Tensor = torch.cat((K, C), dim=-1);  # [T-1, 3, B, d, 2d]

    # Use the compiled chunk method on CUDA and the same method eagerly on CPU.
    solve_chunk : Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor]
    ] = _compiled_rk4_affine_chunk if y0.is_cuda else _rk4_affine_chunk;

    # Initialize lists to hold the displacement, velocity solutions (grouped by chunk).
    # Add a column axis to each initial state for the three-dimensional batched matmuls.
    yn          : torch.Tensor          = y0.unsqueeze(-1);                 # [B, d, 1]
    vn          : torch.Tensor          = Dy0.unsqueeze(-1);                # [B, d, 1]
    D_chunks    : list[torch.Tensor]    = [yn.unsqueeze(0)];
    V_chunks    : list[torch.Tensor]    = [vn.unsqueeze(0)];

    # Loop through chunks of RK4 steps.
    # Every invocation has the same eight-step shape, regardless of rollout duration.
    # Zero factors and coefficients make the final chunk's extra steps exact no-ops.
    for KC_chunk, b_chunk, factors_chunk in zip(
            KC.split(_RK4_AFFINE_CHUNK_SIZE, dim=0),
            b.split(_RK4_AFFINE_CHUNK_SIZE, dim=0),
            step_factors.split(_RK4_AFFINE_CHUNK_SIZE, dim=0)):
        #   KC_chunk.shape      = [<= _RK4_AFFINE_CHUNK_SIZE, 3, B, d, 2d]
        #   b_chunk.shape       = [<= _RK4_AFFINE_CHUNK_SIZE, 3, B, d]
        #   factors_chunk.shape = [<= _RK4_AFFINE_CHUNK_SIZE, 6, padded B]

        # Keep compiled input shapes fixed as rollout duration changes. Otherwise a short final
        # chunk could trigger shape-specialized recompilation. Zero-pad it to eight steps, then
        # discard the extra output frames below.
        n_padding : int = _RK4_AFFINE_CHUNK_SIZE - KC_chunk.shape[0];
        if n_padding:
            KC_chunk        = torch.cat((KC_chunk, KC_chunk.new_zeros((n_padding, 3, B, d, 2 * d))), dim=0);
            b_chunk         = torch.cat((b_chunk, b_chunk.new_zeros((n_padding, 3, B, d))), dim=0);
            factors_chunk   = torch.cat((factors_chunk, factors_chunk.new_zeros((n_padding, 6, step_factors.shape[2]))), dim=0);

        # Run one chunk of RK4 steps.
        #   D_chunk, V_chunk have shape [_RK4_AFFINE_CHUNK_SIZE, B, d, 1]
        D_chunk, V_chunk = solve_chunk(yn, vn, KC_chunk, b_chunk, factors_chunk);

        # Package the results.
        D_chunks.append(D_chunk);
        V_chunks.append(V_chunk);

        # Carry the final displacement and velocity into the next chunk.
        yn = D_chunk[-1];
        vn = V_chunk[-1];

    # Discard the extra frames from final-chunk padding and remove the column axis; this also 
    # discards any extra frames resulting from zero-padding the final chunk. 
    D : torch.Tensor = torch.cat(D_chunks, dim=0)[:n_steps + 1].squeeze(-1);        # [T, B, d]
    V : torch.Tensor = torch.cat(V_chunks, dim=0)[:n_steps + 1].squeeze(-1);        # [T, B, d]

    # All done :)
    return D, V;
