# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  numpy; 
import  torch;

r"""
The functions in this file implement Runge-Kutta solvers for a general first-order ODE of the form:

    y'(t)          = f(t,   y(t)).

Here, y takes values in some vector space, V. 

We apply the Runge-Kutta method to this equation. A general explicit s-step Runge-Kutta method
generates a sequence of time steps, { y_n }_{n \in \mathbb{N}} \subseteq V using the following
rule:

    y_{n + 1}       = y_n + h \sum_{i = 1}^{s} b_i k_i 
    k_i             = f(t_n + c_i h,   y_n + h \sum_{j = 1}^{i - 1} a_{i,j} k_j)

Thus, c_1, ... , c_s, b_1, ... , b_s, and { a_{i,j} : i = 1, ... , s, j = 1, ... , i - 1 }, define
the Runge-Kutta method.
"""


# -------------------------------------------------------------------------------------------------
# Runge-Kutta Solvers
# -------------------------------------------------------------------------------------------------

def RK1(f       : callable, 
        y0      : numpy.ndarray | torch.Tensor, 
        t_Grid  : numpy.ndarray) -> numpy.ndarray | torch.Tensor:
    r"""
    This function implements a RK1 or Forward-Euler ODE solver for an ODE of the form:
     
        y'(t)          = f(t, y(t)).
    
    Here, y takes values in some vector space and f : \mathbb{R} x V -> V is some function. 
  
    In this function, we implement the Forward Euler (RK1) scheme with the following coefficients:
    
        c_1 = 0
        b_1 = 1
        
    Substituting these coefficients into the equations above gives \bar{b_i} = \bar{a_{i,j}} = 0
    for each i,j. Thus, 

        y_{n + 1}       = y_n  + h k_1 
        k_1             = f(t_n, y_n)

    This is the method we implement.


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    f : callable
        The right-hand side of the ODE (see the top of this doc string). This is a function whose 
        domain and co-domain are \mathbb{R} x V and V, respectively. Thus, we assume that 
        f(t, y(t)) = y'(t). 

    y0 : numpy.ndarray or torch.Tensor, shape = arbitrary 
        A numpy.ndarray or torch.Tensor holding the initial position (y0 = y(t0)), where 
        t0 = t_Grid[0].

    t_Grid : numpy.ndarray, shape = (n_t)
        i'th element holds the i'th time value. We assume the elements of this array form an 
        increasing sequence.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    Y : numpy.ndarray or torch.Tensor, shape = (n_t,) + y0.shape
        The i'th row of Y represent the solution at time t_Grid[i]. That is,
            Y[i, ...] = y_i   \approx y(t_Grid[i]) 
    """

    # First, run checks.
    assert(isinstance(y0,       numpy.ndarray)  or isinstance(y0,       torch.Tensor));
    assert(isinstance(t_Grid,   numpy.ndarray));
    assert(len(t_Grid.shape) == 1);

    # Next, fetch N.
    N : int = t_Grid.size;

    # Initialize Y
    if(isinstance(y0, numpy.ndarray)):
        Y : numpy.ndarray = numpy.empty((N,) + y0.shape, dtype = numpy.float32);
    elif(isinstance(y0, torch.Tensor)):
        Y : torch.Tensor = torch.empty((N,) + y0.shape, dtype = torch.float32);

    Y[0, ...] = y0;

    # Now, run the time stepping!
    for n in range(N - 1):
        # Fetch the current time, displacement, velocity.
        tn  : float                         = t_Grid[n];
        yn  : numpy.ndarray | torch.Tensor  = Y[n, ...];
        hn  : float                         = t_Grid[n + 1] - t_Grid[n];

        # Compute k_1.
        k1 = f(tn, yn);

        # Now compute y{n + 1}.
        yn1     : numpy.ndarray | torch.Tensor  = yn  + hn*k1;

        # All done with this step!
        Y[n + 1, ...] = yn1;

    # All done!
    return Y;



def RK2(f       : callable, 
        y0      : numpy.ndarray | torch.Tensor, 
        t_Grid  : numpy.ndarray) -> numpy.ndarray | torch.Tensor:
    r"""
    This function implements a RK2 based ODE solver for a second-order ODE of the following form:
    
        y'(t)          = f(t, y(t)).
    
    Here, y takes values in some vector space. 
  
    In this function, we implement the classic RK2 scheme with the following coefficients:
    
        c_1 = 0
        c_2 = 1

        b_1 = 1/2
        b_2 = 1/2
        
        a_{2,1}         = 1
    
    Thus,

        y_{n + 1}       = y_n  + (h/2)(k_1 + k_2)
        k_1             = f(t_n,        y_n)
        k_2             = f(t_n + h,    y_n + h k_1 )
    
    This is the method we implement.


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    f : callable
        The right-hand side of the ODE (see the top of this doc string). This is a function whose 
        domain and co-domain are \mathbb{R} x V and V, respectively. Thus, we assume that 
        f(t, y(t)) = y'(t). 

    y0 : numpy.ndarray or torch.Tensor, shape = arbitrary 
        A numpy.ndarray or torch.Tensor holding the initial position (y0 = y(t0)), where 
        t0 = t_Grid[0].

    t_Grid : numpy.ndarray, shape = (n_t)
        i'th element holds the i'th time value. We assume the elements of this array form an 
        increasing sequence.
    
    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------
    
    Y : numpy.ndarray or torch.Tensor, shape = (n_t,) + y0.shape
        The i'th row of Y represent the solution at time t_Grid[i]. That is,
            Y[i, ...] = y_i   \approx y(t_Grid[i]) 
    """

    # First, run checks.
    assert(len(t_Grid.shape) == 1);
    assert(isinstance(y0,       numpy.ndarray)  or isinstance(y0,       torch.Tensor));
    assert(isinstance(t_Grid,   numpy.ndarray));

    # Next, fetch N.
    N : int = t_Grid.size;

    # Initialize Y
    if(isinstance(y0, numpy.ndarray)):
        Y : numpy.ndarray = numpy.empty((N,) + y0.shape, dtype = numpy.float32);
    elif(isinstance(y0, torch.Tensor)):
        Y : torch.Tensor = torch.empty((N,) + y0.shape, dtype = torch.float32);

    Y[0, ...] = y0;

    # Now, run the time stepping!
    for n in range(N - 1):
        # Fetch the current time, displacement, velocity.
        tn  : float                         = t_Grid[n];
        yn  : numpy.ndarray | torch.Tensor  = Y[n, :];
        hn  : float                         = t_Grid[n + 1] - t_Grid[n];

        # Compute k_1, k_2.
        k_1 = f(tn,         yn);
        k_2 = f(tn + hn,    yn + hn*k_1);

        # Now compute y{n + 1} and Dy{n + 1}.
        yn1     : numpy.ndarray | torch.Tensor  = yn + (hn/2)*(k_1 + k_2);

        # All done with this step!
        Y[n + 1, :] = yn1;

    # All done!
    return Y;



def RK4(f       : callable, 
        y0      : numpy.ndarray | torch.Tensor, 
        t_Grid  : numpy.ndarray) -> numpy.ndarray | torch.Tensor:
    r"""
    This function implements a RK4 based ODE solver for a second-order ODE of the following form:

        y'(t)          = f(t,   y(t))

    Here, y takes values in some vector space. 
  
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
    
    Thus,
    
        y_{n + 1}       = y_n  + h [ k_1/6 + k_2/3 + k_3/3 + k_4/6 ]

        k_1             = f(t_n,        y_n)
        k_2             = f(t_n + h/2,  y_n + (h/2) k_1)
        k_3             = f(t_n + h/2,  y_n + (h/2) k_2)
        k_4             = f(t_n + h,    y_n + h k_4)

    This is the method we implement.


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    f : callable
        The right-hand side of the ODE (see the top of this doc string). This is a function whose 
        domain and co-domain are \mathbb{R} x V and V, respectively. Thus, we assume that 
        f(t, y(t)) = y'(t). 

    y0 : numpy.ndarray or torch.Tensor, shape = arbitrary 
        A numpy.ndarray or torch.Tensor holding the initial position (y0 = y(t0)), where 
        t0 = t_Grid[0].

    t_Grid : numpy.ndarray, shape = (n_t)
        i'th element holds the i'th time value. We assume the elements of this array form an 
        increasing sequence.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------
    
    Y : numpy.ndarray or torch.Tensor, shape = (n_t,) + y0.shape
        The i'th row of Y represent the solution at time t_Grid[i]. That is,
            Y[i, ...] = y_i   \approx y(t_Grid[i])  
    """

    # First, run checks.
    assert(len(t_Grid.shape) == 1);
    assert(isinstance(y0,       numpy.ndarray)  or isinstance(y0,       torch.Tensor));
    assert(isinstance(t_Grid,   numpy.ndarray));

    # Next, fetch N.
    N : int = t_Grid.size;

    # Initialize Y
    if(isinstance(y0, numpy.ndarray)):
        Y : numpy.ndarray = numpy.empty((N,) + y0.shape, dtype = numpy.float32);
    elif(isinstance(y0, torch.Tensor)):
        Y : torch.Tensor = torch.empty((N,) + y0.shape, dtype = torch.float32);

    Y[0, ...] = y0;

    # Now, run the time stepping!
    for n in range(N - 1):
        # Fetch the current time, displacement, velocity.
        tn  : float                         = t_Grid[n];
        yn  : numpy.ndarray | torch.Tensor  = Y[n, :];
        hn  : float                         = t_Grid[n + 1] - t_Grid[n];

        # Compute k_1, k_1, k_1, k_1.
        k_1 = f(tn,         yn);
        k_2 = f(tn + hn/2,  yn + (hn/2)*k_1);
        k_3 = f(tn + hn/2,  yn + (hn/2)*k_2);
        k_4 = f(tn + hn,    yn + hn*k_3);

        # Now compute y{n + 1} and Dy{n + 1}.
        yn1     : numpy.ndarray | torch.Tensor  = yn + (hn/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)

        # All done with this step!
        Y[n + 1, ...] = yn1;
    
    # All done!
    return Y;
    