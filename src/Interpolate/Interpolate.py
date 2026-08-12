# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

import  logging;

import  numpy;
import  torch;

LOGGER : logging.Logger = logging.getLogger(__name__);



# -------------------------------------------------------------------------------------------------
# Interpolate class
# -------------------------------------------------------------------------------------------------

class Interpolate:
    r"""
    Base class for interpolating latent coefficients for `Interpolatable` latent-dynamics objects.

    Interpolate objects define posterior distributions over training coefficients (conditioned on
    a parameter value), and provide mechanics from sampling from those posterior distributions.
    """

    def __init__(self, train_coefs : dict[tuple[float, ...], dict[str, torch.Tensor]]) -> None:
        r"""
        Build one collection of GPs for each named coefficient tensor.

        For a fixed tensor name (for example "A" or "K"), every training parameter must have a
        tensor with the same shape. We flatten that tensor component-wise and fit one independent GP
        per scalar component, using the parameter tuple as GP input.
        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        train_coefs : dict[tuple[float, ...], dict[str, torch.Tensor]]
            LD-owned training coefficient dictionary. The outer key is an exact parameter tuple;
            the inner dictionary maps coefficient tensor names to tensors.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        
        """

        # Set the training coefficients.
        self.update_train_coefs(train_coefs)


    def update_train_coefs(self, train_coefs : dict[tuple[float, ...], dict[str, torch.Tensor]]) -> None:
        """
        This method updates self's train_coefs attribute use the passed dictionary.

        Specifically, this method builds one collection of GPs for each named coefficient tensor.

        For a fixed tensor name (for example "A" or "K"), every training parameter must have a
        tensor with the same shape. We flatten that tensor component-wise and fit one independent GP
        per scalar component, using the parameter tuple as GP input.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        train_coefs : dict[tuple[float, ...], dict[str, torch.Tensor]]
            LD-owned training coefficient dictionary. The outer key is an exact parameter tuple;
            the inner dictionary maps coefficient tensor names to tensors.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        raise RuntimeError("Abstract function Interpolate.update_train_coefs!");



    def sample(self, param : numpy.ndarray | torch.Tensor | list | tuple) -> dict[str, torch.Tensor]:
        r"""
        Draw one sample from the posterior distributions for each coefficient, when these 
        distributions are conditioned on the passed parameter value. 

        The returned dictionary has the same keys and tensor shapes as each item in `train_coefs`,
        so it can be passed directly to `LatentDynamics.simulate(...)`.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray or torch.Tensor or list or tuple
            Parameter values at which to sample the coefficient posterior.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        out : dict[str, torch.Tensor]
            Native coefficient dictionary containing one posterior sample for each coefficient
            tensor.
        """

        raise RuntimeError("Abstract function Interpolate.sample!");



    def mean(self, param : numpy.ndarray | torch.Tensor | list | tuple) -> dict[str, torch.Tensor]:
        r"""
        Return the posterior mean coefficient dictionary at a requested parameter value.

        This is used for deterministic plotting/rollouts where drawing random coefficient samples
        would make figures nondeterministic.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray or torch.Tensor or list or tuple
            Parameter values at which to evaluate the posterior mean.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        out : dict[str, torch.Tensor]
            Native coefficient dictionary containing the posterior mean for each coefficient
            tensor.
        """

        raise RuntimeError("Abstract function Interpolate.mean!");




    def std(self, param : numpy.ndarray | torch.Tensor | list | tuple) -> dict[str, torch.Tensor]:
        r"""
        Return the standard-deviation of the posterior distributions for each coefficient 
        conditioned on requested parameter.

        The returned tensors use the same native keys and shapes as coefficients in the coefficient 
        dictionary. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param : numpy.ndarray or torch.Tensor or list or tuple
            Parameter values at which to evaluate posterior standard deviations.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        out : dict[str, torch.Tensor]
            Native coefficient dictionary containing posterior standard deviations for each
            coefficient tensor.
        """

        raise RuntimeError("Abstract function Interpolate.std!");

