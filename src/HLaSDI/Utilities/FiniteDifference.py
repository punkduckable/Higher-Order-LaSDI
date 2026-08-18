import  torch;

"""
The functions in this file implement various finite difference approximations for first and second
time derivatives of tensor-valued time sequences.
"""



def Derivative1_Order2_NonUniform(U : torch.Tensor, t_Grid : torch.Tensor) -> torch.Tensor:
    """
    This function finds an O(h^2) approximation of the time derivative to the time series stored in
    the rows of U. We assume that there may be non-uniform time step sizes (the step size differs 
    from step to step). This requires us to use non-standard finite difference techniques. 

    We use the following finite difference techniques to compute the derivative (see 
    "DeriveFiniteDifference.ipynb" for a derivation).
    
        f'(x) = (1/h){                                  - [(2a + b)/(a(a + b))]f(x) + [(a + b)/(ab)]f(x + h)   - [a/(b(a + b))]f(x + 2h) } + O(h^2)
        f'(x) = (1/h){ -[(2a + b)/(a(a + b))]f(x - h)   + [(b - a)/(ab)]f(x)        - [a/(b(a + b))]f(x + h)   }

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    U : torch.Tensor, shape = [Nt, ...]
        A torch.Tensor object representing a time sequence. U[i, ...] represents the value of some 
        function at the i'th time step. Specifically, we assume that U[i, ...] is the value of some 
        function, U, at time t_Grid[i]

    t_Grid : torch.Tensor, shape = (Nt)
        The i'th element of this Tensor should hold the time of the i'th time step.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    dU_dt : torch.Tensor, shape = U.shape
        i'th row holds an O(h^2) approximation of the time derivative of U at the i'th time step.   
    """

    # Checks
    assert(isinstance(t_Grid, torch.Tensor));
    assert(isinstance(U, torch.Tensor));
    assert(U.shape[0]           >= 3);
    assert(len(t_Grid.shape)    == 1);
    assert(len(t_Grid)          == U.shape[0]);

    # Initialize a tensor to hold the time derivative.
    dU_dt   : torch.Tensor  = torch.empty_like(U);
    Nt      : int           = len(t_Grid);


    # Compute the derivative for the first time step.
    a  : float     = t_Grid[1] - t_Grid[0]
    b  : float     = t_Grid[2] - t_Grid[1];

    c0  : float     = (-2*a - b)/(a*(a + b));
    c1  : float     = (a + b)/(a*b);
    c2  : float     = -a/(b*(a + b));

    dU_dt[0, ...]   = c0*U[0, ...] + c1*U[1, ...] + c2*U[2, ...];


    # Compute the derivative for all time steps for which we can use something like the difference rule.
    a  : torch.Tensor = t_Grid[1:(Nt - 1)]    - t_Grid[0:(Nt - 2)];
    b  : torch.Tensor = t_Grid[2:(Nt)]        - t_Grid[1:(Nt - 1)];

    c0  : torch.Tensor = torch.divide(-1*b,   torch.multiply(a, a + b));
    c1  : torch.Tensor = torch.divide(b - a,  torch.multiply(a, b));
    c2  : torch.Tensor = torch.divide(a,      torch.multiply(b, a + b));

    c0 = c0.reshape([-1] + [1]*(len(U.shape) - 1));
    c1 = c1.reshape([-1] + [1]*(len(U.shape) - 1));
    c2 = c2.reshape([-1] + [1]*(len(U.shape) - 1));

    dU_dt[1:(Nt - 1), ...] = torch.multiply(c0, U[0:(Nt - 2), ...]) +  torch.multiply(c1, U[1:(Nt - 1), ...]) + torch.multiply(c2, U[2:Nt, ...]);


    # Compute the derivative for the final time step.
    a  : float     = t_Grid[-2] - t_Grid[-3]
    b  : float     = t_Grid[-1] - t_Grid[-2];

    cm3  : float     = a/(b*(a + b));
    cm2  : float     = -(a + b)/(a*b);
    cm1  : float     = (2*a + b)/(a*(a + b));

    dU_dt[-1, ...] = cm3*U[-3, ...] + cm2*U[-2, ...] + cm1*U[-1, ...];
    

    # All done!
    return dU_dt;




def Derivative1_Order2(U : torch.Tensor, h : float) -> torch.Tensor:
    """
    This function finds an O(h^2) approximation of the time derivative to the time series stored 
    in the rows of U. Specifically, we assume the i'th row of U represents a sample of a function, 
    x, at time t_0 + i*h. We return a new tensor whose i'th row holds an O(h^2) approximation of 
    (d/dt)x(t_0 + i*h)

    We use the following finite difference techniques to compute the derivative (see 
    "DeriveFiniteDifference.ipynb" for a derivation).
    
        f'(x)   = (1/h)[                - (3/2)f(x) +   (2)f(x + h) - (1/2)f(x + 2h) ]  + O(h^2)
        f'(x)   = (1/h)[ (1/2)f(x - h)              + (1/2)f(x + h)                  ]  + O(h^2)


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    U : torch.Tensor, shape = [Nt, ...]
        A torch.Tensor object representing a time sequence. U[i, ...] represents the value of some 
        function at the i'th time step. Specifically, we assume that U[i, ...] is the value of some 
        function, U, at time t_0 + i*h.

    h : float
        The time step size.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    dU_dt : torch.Tensor, shape = U.shape
        i'th row holds an O(h^2) approximation of the time derivative of U at the i'th time step.  
    """

    # For this scheme to work, U must contain at least 3 rows.
    assert(isinstance(U, torch.Tensor));
    assert(U.shape[0] >= 3);

    # Initialize a tensor to hold the time derivative.
    dU_dt   : torch.Tensor  = torch.empty_like(U);
    Nt      : int           = U.shape[0];

    # Compute the derivative for the first time step.
    dU_dt[0, ...] = (-3./2.)*U[0, ...] + 2*U[1, ...] - (1/2)*U[2, ...];

    # Compute the derivative for all time steps for which we can use a central difference rule.
    dU_dt[1:(Nt - 1), ...] = (1./2.)*(U[2:(Nt), ...] - U[0:(Nt - 2), ...]);

    # Compute the derivative for the final time step.
    dU_dt[-1, ...] = (3./2.)*U[-1, ...] - 2*U[-2, ...] + (1./2.)*U[-3, ...];

    # All done!
    return (1./h)*dU_dt;



def Derivative1_Order4(U : torch.Tensor, h : float) -> torch.Tensor:
    """
    This function finds an O(h^4) approximation of the time derivative to the time series stored 
    in the rows of U. Specifically, we assume the i'th row of U represents a sample of a function, 
    x, at time t_0 + i*h. We return a new tensor whose i'th row holds an O(h^4) approximation of 
    (d/dt)x(t_0 + i*h)
    
    We use the following finite difference techniques to compute the derivative (see 
    "DeriveFiniteDifference.ipynb" for a derivation).

        f'(x) = (1/h)[                                  - (25/12)f(x)   + (4)f(x + h)   - (3)f(x + 2h)      + (4/3)f(x + 3h)    - (1/4)f(x + 4h)]   + O(h^4)
        f'(x) = (1/h)[                  -(1/4)f(x - h)  - (5/6)f(x)     + (2/3)f(x + h) - (1/12)f(x + 2h)                                       ]   + O(h^4)
        f'(x) = (1/h)[ (1/12)f(x - 2h)  - (2/3)f(x - h)                 + (2/3)f(x + h) - (1/12)f(x + 2h)                                       ]   + O(h^4)
    
    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    U : torch.Tensor, shape = [Nt, ...]
        A torch.Tensor object representing a time sequence. U[i, ...] represents the value of some 
        function at the i'th time step. Specifically, we assume that U[i, ...] is the value of some 
        function, U, at time t_0 + i*h.

    h : float
        The time step size.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    dU_dt : torch.Tensor, shape = U.shape
        i'th row holds an O(h^4) approximation of the time derivative of U at the i'th time step.  
    """

    # For this scheme to work, U must contain at least 5 rows.
    assert(isinstance(U, torch.Tensor));
    assert(U.shape[0] >= 5);

    # Initialize a tensor to hold the time derivative.
    dU_dt   : torch.Tensor  = torch.empty_like(U);
    Nt      : int           = U.shape[0];

    # Compute the derivative for the first two time steps.
    dU_dt[0, ...] = (-25./12.)*U[0, ...]    + (4)*U[1, ...]         + (-3)*U[2, ...]    + (4./3.)*U[3, ...]     + (-1./4.)*U[4, ...];
    dU_dt[1, ...] = (-1./4.)*U[0, ...]      + (-5./6.)*U[1, ...]    + (3./2.)*U[2, ...] + (-1./2.)*U[3, ...]    + (1./12.)*U[4, ...];
    
    # Compute the derivative for all time steps for which we can use a central difference rule.
    dU_dt[2:(Nt - 2), ...] = (1./12.)*U[0:(Nt - 4), ...]  + (-2./3.)*U[1:(Nt - 3), ...]  + (2./3.)*U[3:(Nt - 1), ...]  + (-1./12.)*U[4:Nt, ...];

    # Compute the derivative for the last two time steps.
    dU_dt[-2, ...] = (1./4.)*U[-1, ...]     + (5./6.)*U[-2, ...]    + (-3./2.)*U[-3, ...]   + (1./2.)*U[-4, ...]    + (-1./12.)*U[-5, ...];
    dU_dt[-1, ...] = (25./12.)*U[-1, ...]   + (-4.)*U[-2, ...]      + (3.)*U[-3, ...]       + (-4./3.)*U[-4, ...]   + (1./4.)*U[-5, ...];

    # All done!
    return (1./h)*dU_dt;




def Derivative2_Order2_NonUniform(U : torch.Tensor, t_Grid : torch.Tensor) -> torch.Tensor:
    """
    This function finds an O(h^2) approximation of the second time derivative to the time series 
    stored in the rows of U. Specifically, we assume the i'th row of U represents a sample of a 
    function, x, at time t_0 + i*h. We return a new tensor whose i'th row holds an O(h^2) 
    approximation of (d^2/dt^2)x(t_0 + i*h)
    
    We use the following finite difference techniques to compute the derivative (see 
    "DeriveFiniteDifference.ipynb" for a derivation).

        f''(x) = (1/h^2)[                                                 ([-2(3a + 2b + c)]/[a(a +b)(a + b + c)])                 f(x) - ([-2(2a + 2b + c)]/[ab(b + c)])  f(x + a) + ([2(2a + b + c)]/[ab(b + c)])       f(x + a + b)   - ([2(2a + b)]/[(a + b + c)(b + c)c])f(x + a + b + c)  ] + O(h^2)
        f''(x) = (1/h^2)[ ([2(2b + c)]/[a(a + b)(a + b + c)]) f(x - a)  - ([2(b(a + 2b + 3c) + c^2 - a^2)]/[ba(b + c)(a + b + c)]) f(x) +     ([2(b + c - a)]/[bc(a + b)]) f(x + b) + ([-2(b - a)]/[c(b + c)(a + b + c)]) f(x + b + c)                                                          ] + O(h^2)
    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    U : torch.Tensor, shape = [Nt, ...]
        A torch.Tensor object representing a time sequence. U[i, ...] represents the value of some 
        function at the i'th time step. Specifically, we assume that U[i, ...] is the value of some 
        function, U, at time t_0 + i*h.

    t_Grid : torch.Tensor, shape = (Nt)
        The i'th element of this Tensor should hold the time of the i'th time step.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    d2U_dt2 : torch.Tensor, shape = U.shape
        i'th row holds an O(h^2) approximation of the time derivative of U at the i'th time step.  
    """

    # Checks. 
    assert(isinstance(t_Grid,   torch.Tensor));
    assert(isinstance(U,        torch.Tensor));
    assert(U.shape[0] >= 4);
    assert(len(t_Grid.shape)    == 1);
    assert(len(t_Grid)          == U.shape[0]);


    # Initialize a tensor to hold the time derivative.
    d2U_dt2 : torch.Tensor  = torch.empty_like(U);
    Nt      : int           = U.shape[0];


    # Compute the derivative for the first time step.
    #   f''(x) = c0 f(x) + c1 f(x + a) + c2 f(x + a + b) + c3 f(x + a + b + c)
    #   c0 =  2(3a + 2b + c) / (a(a + b)(a + b + c))
    #   c1 = -2(2a + 2b + c) / (ab(b + c))
    #   c2 =  2(2a +  b + c) / (bc(a + b))
    #   c3 = -2(2a +  b)     / ((a + b + c)(b + c)c)
    a   : float     = t_Grid[1] - t_Grid[0];
    b   : float     = t_Grid[2] - t_Grid[1];
    c   : float     = t_Grid[3] - t_Grid[2];

    c0  : float     =  2*(3*a + 2*b + c)    / (a*(a + b)*(a + b + c));
    c1  : float     = -2*(2*a + 2*b + c)    / (a*b*(b + c));
    c2  : float     =  2*(2*a +   b + c)    / (b*c*(a + b));
    c3  : float     = -2*(2*a +   b)        / ((a + b + c)*(b + c)*c);

    d2U_dt2[0, ...] = c0*U[0, ...] + c1*U[1, ...] + c2*U[2, ...] + c3*U[3, ...];
    

    # Compute the derivative for all but the last two time steps.
    #   f''(x) = c{-1} f(x - a) + c0 f(x) + c1 f(x + b) + c2 f(x + b + c)
    #   c{-1}   =  2(2b + c)                        / (a(a + b)(a + b + c))
    #   c0      = -2(b(a + 2b + 3c) + c^2 - a^2))   / (ba(b + c)(a + b + c))
    #   c1      =  2(b + c - a)                     / (bc(a + b))
    #   c2      = -2(b - a)                         / (c(b + c)(a + b + c))
    a       : torch.Tensor = t_Grid[1:(Nt - 2)]    - t_Grid[0:(Nt - 3)];
    b       : torch.Tensor = t_Grid[2:(Nt - 1)]    - t_Grid[1:(Nt - 2)];
    c       : torch.Tensor = t_Grid[3:Nt]          - t_Grid[2:(Nt - 1)];

    a_b     : torch.Tensor = a + b;
    b_c     : torch.Tensor = b + c;
    a_b_c   : torch.Tensor = a_b + c;
    bb_bc   : torch.Tensor = torch.multiply(b, b_c);

    cm1 : float     = torch.divide( 2*(2*b + c),        torch.multiply(a, torch.multiply(a_b, a_b_c)));
    c0  : float     = torch.divide(-2*(torch.multiply(b, (a + 2*b + 3*c)) + torch.multiply(c, c) - torch.multiply(a, a)), torch.multiply(a, torch.multiply(bb_bc,   a_b_c)));
    c1  : float     = torch.divide( 2*(b_c - a),        torch.multiply(b, torch.multiply(c, a_b)));
    c2  : float     = torch.divide(-2*(b - a),          torch.multiply(c, torch.multiply(b_c, a_b_c)));

    cm1 = cm1.reshape([-1] + [1]*(len(U.shape) - 1));
    c0  = c0.reshape( [-1] + [1]*(len(U.shape) - 1));
    c1  = c1.reshape( [-1] + [1]*(len(U.shape) - 1));
    c2  = c2.reshape( [-1] + [1]*(len(U.shape) - 1));

    d2U_dt2[1:(Nt - 2), ...] = cm1*U[0:(Nt - 3), ...] + c0*U[1:(Nt - 2), ...] + c1*U[2:(Nt - 1), ...] + c2*U[3:(Nt), ...];


    # Compute the derivative for the second to last time step
    #   f''(x) = c{-1} f(x + a) + c{-2} f(x) + c{-3} f(x - b) + c{-4} f(x - b - c)
    #   c{-1}   =  2(2b + c)                        / (a(a + b)(a + b + c))
    #   c{-2}   = -2(b(a + 2b + 3c) + c^2 - a^2))   / (ba(b + c)(a + b + c))
    #   c{-3}   =  2(b + c - a)                     / (bc(a + b))
    #   c{-4}   = -2(b - a)                         / (c(b + c)(a + b + c))
    c       : float = t_Grid[-3] - t_Grid[-4];
    b       : float = t_Grid[-2] - t_Grid[-3];
    a       : float = t_Grid[-1] - t_Grid[-2];

    cm1     : float = ( 2*(2*b + c))                        / (a*(a + b)*(a + b + c));
    cm2     : float = (-2*(b*(a + 2*b + 3*c) + c*c - a*a))  / (a*b*(b + c)*(a + b + c));
    cm3     : float = ( 2*(b + c - a))                      / (b*c*(a + b));
    cm4     : float = (-2*(b - a))                          / (c*(b + c)*(a + b + c));

    d2U_dt2[-2, ...]    = cm4*U[-4, ...] + cm3*U[-3, ...] + cm2*U[-2, ...] + cm1*U[-1, ...];


    # Compute the derivative for the final time step.
    #   f''(x) = c{-1} f(x) + c{-2} f(x - a) + c{-3} f(x - a - b) + c{-4} f(x - a - b - c)
    #   c{-1} =  2(3a + 2b + c) / (a(a + b)(a + b + c))
    #   c{-2} = -2(2a + 2b + c) / (ab(b + c))
    #   c{-3} =  2(2a + b + c)  / (bc(a + b))
    #   c{-4} = -2(2a + b)      / ((a + b + c)(b + c)c)

    cm1     : float =  2*(3*a + 2*b + c)    / ((a*(a + b)*(a + b + c)));
    cm2     : float = -2*(2*a + 2*b + c)    / (a*b*(b + c));
    cm3     : float =  2*(2*a +   b + c)    / (b*c*(a + b));
    cm4     : float = -2*(2*a +   b)        / ((a + b + c)*(b + c)*c);

    d2U_dt2[-1, ...] = cm1*U[-1, ...] + cm2*U[-2, ...] + cm3*U[-3, ...] + cm4*U[-4, ...];

    # All done!
    return d2U_dt2;



def Derivative2_Order2(U : torch.Tensor, h : float) -> torch.Tensor:
    """
    This function finds an O(h^2) approximation of the second time derivative to the time series 
    stored in the rows of U. Specifically, we assume the i'th row of U represents a sample of a 
    function, x, at time t_0 + i*h. We return a new tensor whose i'th row holds an O(h^2) 
    approximation of (d^2/dt^2)x(t_0 + i*h)
    
    We use the following finite difference techniques to compute the derivative (see 
    "DeriveFiniteDifference.ipynb" for a derivation).

        f''(x) = (1/h^2)[             (2)f(x) -  (5)f(x + h)   + (4)f(x + 2h)   - f(x + 3h) ] + O(h^2)
        f''(x) = (1/h^2)[ f(x - h)  - (2)f(x) +     f(x + h)                                ] + O(h^2)
    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    U : torch.Tensor, shape = [Nt, ...]
        A torch.Tensor object representing a time sequence. U[i, ...] represents the value of some 
        function at the i'th time step. Specifically, we assume that U[i, ...] is the value of some 
        function, U, at time t_0 + i*h.

    h : float
        The time step size.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    d2U_dt2 : torch.Tensor, shape = U.shape
        i'th row holds an O(h^2) approximation of the time derivative of U at the i'th time step. 
    """

    # For this scheme to work, U must contain at least 4 rows.
    assert(isinstance(U, torch.Tensor));
    assert(U.shape[0] >= 4);

    # Initialize a tensor to hold the time derivative.
    d2U_dt2 : torch.Tensor  = torch.empty_like(U);
    Nt      : int           = U.shape[0];

    # Compute the derivative for the first time step.
    #   f''(x) = (1/h^2)[2 f(x) - 5 f(x + h) + 4 f(x + 2h) - f(x + 3h)] + O(h^2)
    d2U_dt2[0, ...] = 2*U[0, ...] - 5*U[1, ...] + 4*U[2, ...] - U[3, ...];
    
    # Compute the derivative for all time steps for which we can use a central difference rule.
    # f''(x) = (1/h^2)[ f(x - h) - 2f(x) + f(x + h)] + O(h^2)
    d2U_dt2[1:(Nt - 1), ...] = U[0:(Nt - 2), ...] - 2*U[1:(Nt - 1), ...] + U[2:Nt, ...];

    # Compute the derivative for the final time step.
    #   f''(x) = (1/h^2)[ -f(x - 3h) + 4 f(x - 2h) - 5 f(x + 2h) + 2f(x)] + O(h^2)
    d2U_dt2[-1, ...] = 2*U[-1, ...] - 5*U[-2, ...] + 4*U[-3, ...] - U[-4, ...];

    # All done!
    return (1./(h*h))*d2U_dt2;



def Derivative2_Order4(U : torch.Tensor, h : float) -> torch.Tensor:
    """
    This function finds an O(h^4) approximation of the second time derivative to the time series 
    stored in the rows of U. Specifically, we assume the i'th row of U represents a sample of a 
    function, x, at time t_0 + i*h. We return a new tensor whose i'th row holds an O(h^4) 
    approximation of (d^2/dt^2)x(t_0 + i*h).

    We use the following finite difference techniques to compute the derivative (see 
    "DeriveFiniteDifference.ipynb" for a derivation).

        f''(x) = (1/h^2)[                                     (15/4)f(x)    - (12 + 5/6)f(x + h)    + (17 + 5/6)f(x + 2h)   - (13)f(x + 3h)     + (5 + 1/12)f(x + 4h)   - (5/6)f(x + 5h) ]  + O(h^4)
        f''(x) = (1/h^2)[                     (5/6)f(x - h) - (5/4)f(x)     - (1/3)f(x + h)         + (7/6)f(x + 2h)        - (1/2)f(x + 3h)    + (1/12)f(x + 4h)]                          + O(h^4)
        f''(x) = (1/h^2)[ -(1/12)f(x - 2h)  + (4/3)f(x - h) - (5/2)f(x)     + (4/3)f(x + h)         - (1/12)f(x + 2h)]                                                                      + O(h^4)


    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    U : torch.Tensor, shape = [Nt, ...]
        A torch.Tensor object representing a time sequence. U[i, ...] represents the value of some 
        function at the i'th time step. Specifically, we assume that U[i, ...] is the value of some 
        function, U, at time t_0 + i*h.

    h : float
        The time step size.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    d2U_dt2 : torch.Tensor, shape = U.shape
        i'th row holds an O(h^4) approximation of the time derivative of U at the i'th time step. 
    """

    # For this scheme to work, U must contain at least 6 rows.
    assert(isinstance(U, torch.Tensor));
    assert(U.shape[0] >= 6);

    # Initialize a tensor to hold the time derivative.
    d2U_dt2 : torch.Tensor  = torch.empty_like(U);
    Nt      : int           = U.shape[0];

    # Compute the derivative for the first two time steps.
    d2U_dt2[0, ...] = (15./4.)*U[0, ...]    + (-12. - 5./6.)*U[1, ...]  + (17. + 5./6.)*U[2, ...]   + (-13.)*U[3, ...]  + (5. + 1./12.)*U[4, ...]   + (-5./6.)*U[5, ...];
    d2U_dt2[1, ...] = (5./6.)*U[0, ...]     + (-5./4.)*U[1, ...]        + (-1./3.)*U[2, ...]        + (7./6.)*U[3, ...] + (-1./2.)*U[4, ...]        + (1./12.)*U[5, ...];

    # Compute the derivative for all time steps for which we can use a central difference rule.
    d2U_dt2[2:(Nt - 2), ...] = (-1./12.)*U[0:(Nt - 4), ...] + (4./3.)*U[1:(Nt - 3), ...] + (-5./2.)*U[2:(Nt - 2), ...] + (4./3.)*U[3:(Nt - 1), ...] + (-1./12.)*U[4:Nt, ...];

    # Compute the derivative for the final two time steps.
    d2U_dt2[-2, ...] = (5./6.)*U[-1, ...]   + (-5./4.)*U[-2, ...]       + (-1./3.)*U[-3, ...]       + (7./6.)*U[-4, ...]    + (-1./2.)*U[-5, ...]       + (1./12.)*U[-6, ...];
    d2U_dt2[-1, ...] = (15./4.)*U[-1, ...]  + (-12. - 5./6.)*U[-2, ...] + (17. + 5./6.)*U[-3, ...]  + (-13.)*U[-4, ...]     + (5. + 1./12.)*U[-5, ...]  + (-5./6.)*U[-6, ...];

    # All done!
    return (1./(h*h))*d2U_dt2;
