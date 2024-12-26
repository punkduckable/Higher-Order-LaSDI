import  torch;
import  torchaudio.functional   as  taf;

"""
The functions in this file implement various finite difference approximations for first and second
time derivatives of tensor-valued time sequences.
"""


def Derivative1_Order2(X : torch.Tensor, h : float) -> torch.Tensor:
    """
    This function finds an O(h^2) approximation of the time derivative to the time series stored 
    in the rows of X. Specifically, we assume the i'th row of X represents a sample of a function, 
    x, at time t_0 + i*h. We return a new tensor whose i'th row holds an O(h^2) approximation of 
    (d/dt)x(t_0 + i*h)



    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    X: A torch.Tensor object representing a time sequence. We assume that X has shape [Nt, ...] 
    and that X[i, ...] represents the value of some function at the i'th time step. Specifically, 
    we assume that X[i, ...] is the value of some function, X, at time t_0 + i*h.

    h: The time step size.



    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    A torch.Tensor object with the same shape as X whose i'th row holds a O(h^2) approximation of 
    the time derivative of X at the i'th time step. 
    """

    # For this scheme to work, X must contain at least 3 rows.
    assert(X.shape[0] >= 3);

    # Initialize a tensor to hold the time derivative.
    dX_dt   : torch.Tensor  = torch.empty_like(X);
    Nt      : int           = X.shape[0];

    # Compute the derivative for the first time step.
    dX_dt[0, ...] = (1./h)*((-3./2.)*X[0, ...] + 2*X[1, ...] - (1/2)*X[2, ...]);

    # Compute the derivative for all time steps for which we can use a central difference rule.
    dX_dt[1:(Nt - 1), ...] = (1./2.)*(X[2:(Nt), ...] - X[0:(Nt - 2), ...]);

    # Compute the derivative for the final time step.
    dX_dt[-1, ...] = (3./2.)*X[-1, ...] - 2*X[-2, ...] + (1./2.)*X[-3, ...];

    # All done!
    return (1./h)*dX_dt;



def Derivative1_Order4(X : torch.Tensor, h : float) -> torch.Tensor:
    """
    This function finds an O(h^4) approximation of the time derivative to the time series stored 
    in the rows of X. Specifically, we assume the i'th row of X represents a sample of a function, 
    x, at time t_0 + i*h. We return a new tensor whose i'th row holds an O(h^4) approximation of 
    (d/dt)x(t_0 + i*h)
    
    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    X: A torch.Tensor object representing a time sequence. We assume that X has shape [Nt, ...] 
    and that X[i, ...] represents the value of some function at the i'th time step. Specifically, 
    we assume that X[i, ...] is the value of some function, X, at time t_0 + i*h.

    h: The time step size.


    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    A torch.Tensor object with the same shape as X whose i'th row holds a O(h^4) approximation of 
    the time derivative of X at the i'th time step. 
    """

    # For this scheme to work, X must contain at least 5 rows.
    assert(X.shape[0] >= 5);

    # Initialize a tensor to hold the time derivative.
    dX_dt   : torch.Tensor  = torch.empty_like(X);
    Nt      : int           = X.shape[0];

    # Compute the derivative for the first two time steps.
    dX_dt[0, ...] = (-25./12.)*X[0, ...]    + (4)*X[1, ...]         + (-3)*X[2, ...]    + (4./3.)*X[3, ...]     + (-1./4.)*X[4, ...];
    dX_dt[1, ...] = (-1./4.)*X[0, ...]      + (-5./6.)*X[1, ...]    + (3./2.)*X[2, ...] + (-1./2.)*X[3, ...]    + (1./12.)*X[4, ...];
    
    # Compute the derivative for all time steps for which we can use a central difference rule.
    dX_dt[2:(Nt - 2), ...] = (1./12.)*X[0:(Nt - 4), ...]  + (-2./3.)*X[1:(Nt - 3), ...]  + (2./3.)*X[3:(Nt - 1), ...]  + (-1./12.)*X[4:Nt, ...];

    # Compute the derivative for the last two time steps.
    dX_dt[-2, ...] = (1./4.)*X[-1, ...]     + (5./6.)*X[-2, ...]    + (-3./2.)*X[-3, ...]   + (1./2.)*X[-4, ...]    + (-1./12.)*X[-5, ...];
    dX_dt[-1, ...] = (25./12.)*X[-1, ...]   + (-4.)*X[-2, ...]      + (3.)*X[-3, ...]       + (-4./3.)*X[-4, ...]   + (1./4.)*X[-5, ...];

    # All done!
    return (1./h)*dX_dt;




def Derivative2_Order2(X : torch.Tensor, h : float) -> torch.Tensor:
    """
    This function finds an O(h^2) approximation of the second time derivative to the time series 
    stored in the rows of X. Specifically, we assume the i'th row of X represents a sample of a 
    function, x, at time t_0 + i*h. We return a new tensor whose i'th row holds an O(h^2) 
    approximation of (d^2/dt^2)x(t_0 + i*h)
    
    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    X: A torch.Tensor object representing a time sequence. We assume that X has shape [Nt, ...] 
    and that X[i, ...] represents the value of some function at the i'th time step. Specifically, 
    we assume that X[i, ...] is the value of some function, x, at time t_0 + i*h.

    h: The time step size.


    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    A torch.Tensor object with the same shape as X whose i'th row holds a O(h^2) approximation of 
    the time derivative of x at the i'th time step (that is, it approximates x''(t_0 + i h)).
    """

    # For this scheme to work, X must contain at least 4 rows.
    assert(X.shape[0] >= 4);

    # Initialize a tensor to hold the time derivative.
    d2X_dt2 : torch.Tensor  = torch.empty_like(X);
    Nt      : int           = X.shape[0];

    # Compute the derivative for the first time step.
    d2X_dt2[0, ...] = 2*X[0, ...] - 5*X[1, ...] + 4*X[2, ...] - X[3, ...];
    
    # Compute the derivative for all time steps for which we can use a central difference rule.
    d2X_dt2[1:(Nt - 1), ...] = X[0:(Nt - 2), ...] - 2*X[1:(Nt - 1), ...] + X[2:Nt, ...];

    # Compute the derivative for the final time step.
    d2X_dt2[-1, ...] = 2*X[-1, ...] - 5*X[-2, ...] + 4*X[-3, ...] - X[-4, ...];

    # All done!
    return (1./(h*h))*d2X_dt2;



def Derivative2_Order4(X : torch.Tensor, h : float) -> torch.Tensor:
    """
    This function finds an O(h^4) approximation of the second time derivative to the time series 
    stored in the rows of X. Specifically, we assume the i'th row of X represents a sample of a 
    function, x, at time t_0 + i*h. We return a new tensor whose i'th row holds an O(h^4) 
    approximation of (d^2/dt^2)x(t_0 + i*h)
    
    

    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    X: A torch.Tensor object representing a time sequence. We assume that X has shape [Nt, ...] 
    and that X[i, ...] represents the value of some function at the i'th time step. Specifically, 
    we assume that X[i, ...] is the value of some function, x, at time t_0 + i*h.

    h: The time step size.


    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    A torch.Tensor object with the same shape as X whose i'th row holds a O(h^4) approximation of 
    the time derivative of x at the i'th time step (that is, it approximates x''(t_0 + i h)).
    """

    # For this scheme to work, X must contain at least 6 rows.
    assert(X.shape[0] >= 6);

    # Initialize a tensor to hold the time derivative.
    d2X_dt2 : torch.Tensor  = torch.empty_like(X);
    Nt      : int           = X.shape[0];

    # Compute the derivative for the first two time steps.
    d2X_dt2[0, ...] = (15./4.)*X[0, ...]    + (-12. - 5./6.)*X[1, ...]  + (17. + 5./6.)*X[2, ...]   + (-13.)*X[3, ...]  + (5. + 1./12.)*X[4, ...]   + (-5./6.)*X[5, ...];
    d2X_dt2[1, ...] = (5./6.)*X[0, ...]     + (-5./4.)*X[1, ...]        + (-1./3.)*X[2, ...]        + (7./6.)*X[3, ...] + (-1./2.)*X[4, ...]        + (1./12.)*X[5, ...];

    # Compute the derivative for all time steps for which we can use a central difference rule.
    d2X_dt2[2:(Nt - 2), ...] = (-1./12.)*X[0:(Nt - 4), ...] + (4./3.)*X[1:(Nt - 3), ...] + (-5./2.)*X[2:(Nt - 2), ...] + (4./3.)*X[3:(Nt - 1), ...] + (-1./12.)*X[4:Nt, ...];

    # Compute the derivative for the final two time steps.
    d2X_dt2[-2, ...] = (5./6.)*X[-1, ...]   + (-5./4.)*X[-2, ...]       + (-1./3.)*X[-3, ...]       + (7./6.)*X[-4, ...]    + (-1./2.)*X[-5, ...]       + (1./12.)*X[-6, ...];
    d2X_dt2[-1, ...] = (15./4.)*X[-1, ...]  + (-12. - 5./6.)*X[-2, ...] + (17. + 5./6.)*X[-3, ...]  + (-13.)*X[-4, ...]     + (5. + 1./12.)*X[-5, ...]  + (-5./6.)*X[-6, ...];

    # All done!
    return (1./(h*h))*d2X_dt2;