# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  numpy; 

r"""
The functions in this file implement Runge-Kutta solvers for a general second-order ODE of the 
following form:
    y''(t)          = f(t,   y(t),   y'(t)).
Here, y takes values in \mathbb{R}^d. 

To understand where our methods come from, let us first make a few substitutions. First, let 
    z(t)            = (y(t), y'(t)) \in \mathbb{R}^{2d}. 
Then, 
    z'(t)           = (y'(t), y''(t))
                    = (y'(t), f(t,   y(t),   y'(t)))
                    = (y'(t), f(t,   z(t))).
Now, let g : \mathbb{R} x \mathbb{R}^{2d} -> \mathbb{R}^{2d} be defined by
    g(t, z(t))      = ( z[d + 1:2d], f(t, z(t)) ).
Then,
    z'(t)           = g(t, z(t))
In other words, we reduce the 2nd order ODE in \mathbb{R}^d to a first order one in 
\mathbb{R}^{2d}. 


We can now apply the Runge-Kutta method to this equation. A general explicit s-step Runge-Kutta 
method generates a sequence of time steps, { y_n }_{n \in \mathbb{N}} \subseteq \mathbb{R}^d 
using the following rule:
    z_{n + 1}       = z_n + h \sum_{i = 1}^{s} b_i k_i 
    k_i             = g(t_n + c_n h,   z_n + h \sum_{j = 1}^{i - 1} a_{i,j} k_j)
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

    l_i             = f(t_n + c_n h,   y_n + h c_i y'_n + h^2 \sum_{j = 1}^{i - 1} l_j \bar{a_{i,j}},   y'_n + h \sum_{j = 1}^{i - 1} a_{i,j} l_j)

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
        y0      : numpy.ndarray, 
        Dy0     : numpy.ndarray, 
        times   : numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    This function implements a RK1 or Forward-Euler ODE solver for a second-order ODE of the 
    following form:
        y''(t)          = f(t,   y(t),   y'(t)).
    Here, y takes values in \mathbb{R}^d. 
  
    In this function, we implement the Forward Euler (RK1) scheme:
     with the following coefficients:
        c_1 = 0
        b_1 = 1
        
    Substituting these coefficients into the equations above gives \bar{b_i} = \bar{a_{i,j}} = 0
    for each i,j. Thus, 
        y_{n + 1}       = y_n  + h y'_n 
        y'_{n + 1}      = y'_n + h l_1

        l_1             = f(t_n,        y_n,                            y'_n)
    This is the method we implement.


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    f: The right-hand side of the ODE (see the top of this doc string). This is a function whose 
    domain and co-domain are \mathbb{R} x \mathbb{R}^d x \mathbb{R}^d and \mathbb{R}^d, 
    respectively. Thus, we assume that f(t, y(t), y'(t)) = y''(t). 

    y0: The initial displacement (y0 = y(t0)), where t0 = times[0].

    Dy0: The initial velocity (Dy0 = y'(t0)), where t0 = times[0].

    times: A 1d numpy.ndarray object whose i'th element holds the i'th time value. We assume the
    elements of this array form an increasing sequence.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Two numpy.ndarray objects: D, V. 
    
    Let N denote the length of times. If y takes values in \mathbb{R}^d, then D and V have shape 
    N x d. The i'th row of D, V represent the displacement and the velocity at times[i], 
    respectively. Thus, if we denote the returned arrays by D and V, respectively, then 
        D[i, :] = y_i   \approx y (times[i]) 
        V[i, :] = y'_i  \approx y'(times[i]) 
    """

    # First, run checks.
    assert(len(times.shape) == 1);
    assert(len(y0.shape)    == 1);
    assert(y0.shape         == Dy0.shape);

    # Next, fetch d, N.
    d : int = y0.size;
    N : int = times.size;

    # Initialize D, V.
    D : numpy.ndarray = numpy.empty((N, d), dtype = numpy.float32);
    V : numpy.ndarray = numpy.empty((N, d), dtype = numpy.float32);

    D[0, :] = y0;
    V[0, :] = Dy0;

    # Now, run the time stepping!
    for n in range(N - 1):
        # Fetch the current time, displacement, velocity.
        tn  : float         = times[n];
        yn  : numpy.ndarray = D[n, :];
        Dyn : numpy.ndarray = V[n, :];
        hn  : float         = times[n + 1] - times[n];

        # Compute l_1.
        l_1 = f(tn, yn, Dyn);

        # Now compute y{n + 1} and Dy{n + 1}.
        yn1     : numpy.ndarray = yn  + hn*Dyn;
        Dyn1    : numpy.ndarray = Dyn + hn*l_1

        # All done with this step!
        D[n + 1, :] = yn1;
        V[n + 1, :] = Dyn1;

    # All done!
    return (D, V);



def RK2(f       : callable, 
        y0      : numpy.ndarray, 
        Dy0     : numpy.ndarray, 
        times   : numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    This function implements a RK2 based ODE solver for a second-order ODE of the following form:
        y''(t)          = f(t,   y(t),   y'(t)).
    Here, y takes values in \mathbb{R}^d. 
  
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

    f: The right-hand side of the ODE (see the top of this doc string). This is a function whose 
    domain and co-domain are \mathbb{R} x \mathbb{R}^d x \mathbb{R}^d and \mathbb{R}^d, 
    respectively. Thus, we assume that f(t, y(t), y'(t)) = y''(t). 

    y0: The initial displacement (y0 = y(0)).

    Dy0: The initial velocity (Dy0 = y'(0)).

    times: A 1d numpy.ndarray object whose i'th element holds the i'th time value. We assume the
    elements of this array form an increasing sequence.


    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Two numpy.ndarray objects: D, V. 
    
    Let N denote the length of times. If y takes values in \mathbb{R}^d, then D and V have shape 
    N x d. The i'th row of D, V represent the displacement and the velocity at times[i], 
    respectively. Thus, if we denote the returned arrays by D and V, respectively, then 
        D[i, :] = y_i   \approx y (times[i]) 
        V[i, :] = y'_i  \approx y'(times[i]) 
    """

    # First, run checks.
    assert(len(times.shape) == 1);
    assert(len(y0.shape)    == 1);
    assert(y0.shape         == Dy0.shape);

    # Next, fetch d, N.
    d : int = y0.size;
    N : int = times.size;

    # Initialize D, V.
    D : numpy.ndarray = numpy.empty((N, d), dtype = numpy.float32);
    V : numpy.ndarray = numpy.empty((N, d), dtype = numpy.float32);

    D[0, :] = y0;
    V[0, :] = Dy0;

    # Now, run the time stepping!
    for n in range(N - 1):
        # Fetch the current time, displacement, velocity.
        tn  : float         = times[n];
        yn  : numpy.ndarray = D[n, :];
        Dyn : numpy.ndarray = V[n, :];
        hn  : float         = times[n + 1] - times[n];

        # Compute l_1, l_2.
        l_1 = f(tn,         yn,                         Dyn);
        l_2 = f(tn + hn/2,  yn + hn*Dyn ,               Dyn + (hn/2)*l_1);

        # Now compute y{n + 1} and Dy{n + 1}.
        yn1     : numpy.ndarray = yn + hn*Dyn + (hn*hn/2)*l_1;
        Dyn1    : numpy.ndarray = Dyn + (hn/2)*(l_1 + l_2);

        # All done with this step!
        D[n + 1, :] = yn1;
        V[n + 1, :] = Dyn1;

    # All done!
    return (D, V);



def RK4(f       : callable, 
        y0      : numpy.ndarray, 
        Dy0     : numpy.ndarray, 
        times   : numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    This function implements a RK4 based ODE solver for a second-order ODE of the following form:
        y''(t)          = f(t,   y(t),   y'(t)).
    Here, y takes values in \mathbb{R}^d. 
  
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

    f: The right-hand side of the ODE (see the top of this doc string). This is a function whose 
    domain and co-domain are \mathbb{R} x \mathbb{R}^d x \mathbb{R}^d and \mathbb{R}^d, 
    respectively. Thus, we assume that f(t, y(t), y'(t)) = y''(t). 

    y0: The initial displacement (y0 = y(0)).

    Dy0: The initial velocity (Dy0 = y'(0)).

    times: A 1d numpy.ndarray object whose i'th element holds the i'th time value. We assume the
    elements of this array form an increasing sequence.


    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Three numpy.ndarray objects: D, V, and T. 
    
    If y takes values in \mathbb{R}^d, then D and V have shape N + 1 x d. The i'th row of D, V
    represent the displacement and the velocity at time i*h, respectively. Thus, if we denote the 
    returned arrays by D and V, respectively, then 
        D[i, :] = y_i   \approx y (i h) 
        V[i, :] = y'_i  \approx y'(i h) 
    Finally, T is an array of shape N + 1 whose i'th entry holds the i'th time value.
    """

    # First, run checks.
    assert(len(times.shape) == 1);
    assert(len(y0.shape)    == 1);
    assert(y0.shape         == Dy0.shape);

    # Next, fetch d, N.
    d : int = y0.size;
    N : int = times.size;

    # Initialize D, V.
    D : numpy.ndarray = numpy.empty((N, d), dtype = numpy.float32);
    V : numpy.ndarray = numpy.empty((N, d), dtype = numpy.float32);

    D[0, :] = y0;
    V[0, :] = Dy0;

    # Now, run the time stepping!
    for n in range(N - 1):
        # Fetch the current time, displacement, velocity.
        tn  : float         = times[n];
        yn  : numpy.ndarray = D[n, :];
        Dyn : numpy.ndarray = V[n, :];
        hn  : float         = times[n + 1] - times[n];

        # Compute l_1, l_2, l_3, l_4.
        l_1 = f(tn,         yn,                                 Dyn);
        l_2 = f(tn + hn/2,  yn + (hn/2)*Dyn ,                   Dyn + (hn/2)*l_1);
        l_3 = f(tn + hn/2,  yn + (hn/2)*Dyn + (hn*hn/4)*l_1,    Dyn + (hn/2)*l_2);
        l_4 = f(tn + hn,    yn + hn*Dyn + (hn*hn/2)*l_2,        Dyn + hn*l_3);

        # Now compute y{n + 1} and Dy{n + 1}.
        yn1     : numpy.ndarray = yn + hn*Dyn + (hn*hn/6)*(l_1 + l_2 + l_3);
        Dyn1    : numpy.ndarray = Dyn + (hn/6)*(l_1 + 2*l_2 + 2*l_3 + l_4);

        # All done with this step!
        D[n + 1, :] = yn1;
        V[n + 1, :] = Dyn1;
    
    # All done!
    return (D, V);
    