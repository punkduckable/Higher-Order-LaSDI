# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

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

def RK1(f       : callable, 
        y0      : numpy.ndarray | torch.Tensor, 
        Dy0     : numpy.ndarray | torch.Tensor, 
        t_Grid  : numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray] | tuple[torch.Tensor, torch.Tensor]:
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

    f : callable
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

    # Initialize D, V.
    if(isinstance(y0, numpy.ndarray)):
        D : numpy.ndarray = numpy.empty((N,) + y0.shape, dtype = numpy.float32);
        V : numpy.ndarray = numpy.empty((N,) + y0.shape, dtype = numpy.float32);
    elif(isinstance(y0, torch.Tensor)):
        D : torch.Tensor = torch.empty((N,) + y0.shape, dtype = torch.Tensor);
        V : torch.Tensor = torch.empty((N,) + y0.shape, dtype = torch.Tensor);

    D[0, ...] = y0;
    V[0, ...] = Dy0;

    # Now, run the time stepping!
    for n in range(N - 1):
        # Fetch the current time, displacement, velocity.
        tn  : float                         = t_Grid[n];
        yn  : numpy.ndarray | torch.Tensor  = D[n, ...];
        Dyn : numpy.ndarray | torch.Tensor  = V[n, ...];
        hn  : float                         = t_Grid[n + 1] - t_Grid[n];

        # Compute l_1.
        l_1 = f(tn, yn, Dyn);

        # Now compute y{n + 1} and Dy{n + 1}.
        yn1     : numpy.ndarray | torch.Tensor  = yn  + hn*Dyn;
        Dyn1    : numpy.ndarray | torch.Tensor  = Dyn + hn*l_1

        # All done with this step!
        D[n + 1, ...] = yn1;
        V[n + 1, ...] = Dyn1;

    # All done!
    return (D, V);



def RK2(f       : callable, 
        y0      : numpy.ndarray | torch.Tensor, 
        Dy0     : numpy.ndarray | torch.Tensor, 
        t_Grid  : numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray] | tuple[torch.Tensor, torch.Tensor]:
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

    f : callable
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

    # Initialize D, V.
    if(isinstance(y0, numpy.ndarray)):
        D : numpy.ndarray   = numpy.empty((N,) + y0.shape, dtype = numpy.float32);
        V : numpy.ndarray   = numpy.empty((N,) + y0.shape, dtype = numpy.float32);
    elif(isinstance(y0, torch.Tensor)):
        D : torch.Tensor    = torch.empty((N,) + y0.shape, dtype = torch.float32);
        V : torch.Tensor    = torch.empty((N,) + y0.shape, dtype = torch.float32);

    D[0, :] = y0;
    V[0, :] = Dy0;

    # Now, run the time stepping!
    for n in range(N - 1):
        # Fetch the current time, displacement, velocity.
        tn  : float                         = t_Grid[n];
        yn  : numpy.ndarray | torch.Tensor  = D[n, :];
        Dyn : numpy.ndarray | torch.Tensor  = V[n, :];
        hn  : float                         = t_Grid[n + 1] - t_Grid[n];

        # Compute l_1, l_2.
        l_1 = f(tn,         yn,             Dyn);
        l_2 = f(tn + hn,    yn + hn*Dyn ,   Dyn + hn*l_1);

        # Now compute y{n + 1} and Dy{n + 1}.
        yn1     : numpy.ndarray | torch.Tensor  = yn + hn*Dyn + (hn*hn/2)*l_1;
        Dyn1    : numpy.ndarray | torch.Tensor  = Dyn + (hn/2)*(l_1 + l_2);

        # All done with this step!
        D[n + 1, :] = yn1;
        V[n + 1, :] = Dyn1;

    # All done!
    return (D, V);



def RK4(f       : callable, 
        y0      : numpy.ndarray | torch.Tensor, 
        Dy0     : numpy.ndarray | torch.Tensor, 
        t_Grid  : numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray] | tuple[torch.Tensor, torch.Tensor]:
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

    f : callable
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

    # Initialize D, V.
    if(isinstance(y0, numpy.ndarray)):
        D : numpy.ndarray   = numpy.empty((N,) + y0.shape, dtype = numpy.float32);
        V : numpy.ndarray   = numpy.empty((N,) + y0.shape, dtype = numpy.float32);
    elif(isinstance(y0, torch.Tensor)):
        D : torch.Tensor    = torch.empty((N,) + y0.shape, dtype = torch.float32);
        V : torch.Tensor    = torch.empty((N,) + y0.shape, dtype = torch.float32);

    D[0, ...] = y0;
    V[0, ...] = Dy0;

    # Now, run the time stepping!
    for n in range(N - 1):
        # Fetch the current time, displacement, velocity.
        tn  : float                         = t_Grid[n];
        yn  : numpy.ndarray | torch.Tensor  = D[n, ...];
        Dyn : numpy.ndarray | torch.Tensor  = V[n, ...];
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
        D[n + 1, ...] = yn1;
        V[n + 1, ...] = Dyn1;
    
    # All done!
    return (D, V);
    