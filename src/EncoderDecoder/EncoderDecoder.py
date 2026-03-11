# -------------------------------------------------------------------------------------------------
# Import and Setup
# -------------------------------------------------------------------------------------------------


# Add the Physics directory to the search path.
import  sys;
import  os;
src_Path        : str   = os.path.abspath(os.path.dirname(os.path.dirname(__file__)));
Physics_Path    : str  = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "Physics"));
sys.path.append(Physics_Path);

import  logging;

import  torch;
import  numpy;

from    typing          import  TYPE_CHECKING;
if TYPE_CHECKING:
    from    Trainer     import  Trainer;
    from    Physics     import  Physics;


# Set up logging.
LOGGER  : logging.Logger    = logging.getLogger(__name__);




# -------------------------------------------------------------------------------------------------
# EncoderDecoder
# -------------------------------------------------------------------------------------------------

class EncoderDecoder(torch.nn.Module):
    def __init__(   self, 
                    n_IC    : int, 
                    n_z     : int) -> None:
        r"""
        Initializes a EncoderDecoder object. A EncoderDecoder object does two things. a) It can 
        encode FOM states (frames) to their latent encodings, and b) it can decode those latent
        encodings back to the FOM state. An EncoderDecoder object is defined by two variables:

            n_IC (the number of initial conditions)
            n_z (the latent space dimension)
        
        The encoder must map n_IC elements of the FOM space to n_IC elements of \\mathbb{R}^{n_z}. 
        The decoder must map n_IC elements of \\mathbb{R}^{n_z} to n_IC elements of the FOM space.

        To implement a EncoderDecoder subclass, you must implement the Encode, Decode, forward, 
        and latent_initial_conditions, and save methods. You should also implement a "load" 
        function to load the encoded state from a save.


        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        n_IC : int
            The number of initial conditions. The encoder must accept this many elements of the 
            FOM space and map them to the same number of elements of the latent space.

        n_z : int 
            The latent space dimension.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Checks
        assert n_IC > 0,    "n_IC = %d; must be positive" % n_IC;
        assert n_z > 0,     "n_z = %d; must be positive" % n_z;

        # Run the superclass initializer.
        super().__init__();
        
        # Store information (for return purposes).
        self.n_IC           : int       = n_IC;
        self.n_z            : int       = n_z;

        # All done!
        return;



    def Encode(self, X1 : torch.Tensor, Xn_IC : torch.Tensor) -> tuple[torch.Tensor]:
        """
        In general, the Encode method should take n_IC positional arguments, each one containing 
        a batch of elements of the FOM space, and map them to n_IC elements of the latent space. 
        The output must be a tuple of tensors. The input should be n_IC tensors (as positional 
        arguments), each with the same shape.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        X1, ... , Xn_IC : torch.Tensor, shape = (n_inputs, ...)
            The inputs to be encoded.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Z : tuple[torch.Tensor], len = self.n_IC
            A List of n_IC elements of \\mathbb{R}^{n_z}. 
        """

        raise RuntimeError("Abstract method EncoderDecoder.Encode!");



    def Decode(self, Z1 : torch.Tensor, Xn_IC : torch.Tensor) -> tuple[torch.Tensor]:
        """
        This function should accept n_IC elements of the latent space and map them to n_IC elements 
        of the FOM space. The output must be a tuple of tensors. The input should be n_IC tensors 
        (as positional arguments), each with the same shape.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z1, .. , Zn_IC : torch.Tensor, shape = (n_inputs, ...)
            The latent states to be decoded.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        X : tuple[torch.Tensor], len = self.n_IC
            A List of n_IC elements of the FOM space 
        """

        raise RuntimeError("Abstract method EncoderDecoder.Decode!");



    def forward(self, X1 : torch.Tensor, Xn_IC : torch.Tensor) -> tuple[torch.Tensor]:
        """
        This function passes the X's through the encoder, producing a latent state, Z. It then 
        passes Z through the decoder; hopefully producing a set of vectors that approximates X.
        

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        X1, ... , Xn_IC : torch.Tensor, shape = (n_inputs, ...)
            The inputs to be encoded.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Y : tuple[torch.Tensor], len = self.n_IC
            A self.n_IC element tuple of torch.Tensors in the FOM space holding the image of X 
            under the encoder and decoder. 
        """

        raise RuntimeError("Abstract method EncoderDecoder.forward!");



    def latent_initial_conditions(  self,
                                    param_grid     : numpy.ndarray, 
                                    physics        : "Physics",
                                    trainer        : "Trainer") -> list[list[numpy.ndarray]]:
        """
        This function maps a set of initial conditions for the FOM to initial conditions for the 
        latent space dynamics. Specifically, we take in a set of possible parameter values. For 
        each set of parameter values, we recover the FOM IC (from physics), then map this FOM IC to 
        a latent space IC (by encoding it). We do this for each parameter combination and then 
        return a list housing the latent space ICs.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param_grid : numpy.ndarray, shape = (n_param, n_p)
            i,j element of this array holds the value of the j'th parameter in the i'th combination of 
            parameters. Here, n_param is the number of combinations of parameter values and n_p is the 
            number of parameters (in each combination).

        physics : "Physics"
            A "Physics" object that, among other things, stores the IC for each combination of 
            parameter values. This physics object should have the same number of initial conditions as 
            self.
        
        trainer : "Trainer"
            The trainer object used to train the EncoderDecoder.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        Z0 : list[list[numpy.ndarray]], len = n_param
            An n_param element list whose i'th element is an n_IC element list holding the encoding
            of the initial conditions for the i'th combination of parameters. 
            
            If we let U0_i denote the FOM IC for the i'th set of parameters, then the i'th element of 
            the returned list is [self.encoder(*U0_i)].
        """

        # Checks.
        assert isinstance(param_grid, numpy.ndarray),   "type(param_grid) = %s, must be numpy.ndarray" % str(type(param_grid));
        assert len(param_grid.shape) == 2,              "param_grid.shape = %s, must have length 2" % str(param_grid.shape);
        assert physics.n_IC == self.n_IC,               "physics.n_IC = %d, self.n_IC = %d; must be equal" % (physics.n_IC, self.n_IC);

        # Determine device for encoding.
        encoder_device : torch.device = next(self.parameters()).device;

        n_param : int = param_grid.shape[0];
        Z0      : list[list[numpy.ndarray]] = [];
        LOGGER.debug("Encoding initial conditions for %d combinations of parameter values" % n_param);

        has_norm : bool = (trainer is not None) and hasattr(trainer, "has_normalization") and trainer.has_normalization();

        with torch.no_grad():
            for i in range(n_param):
                # Get the ICs for the i'th combination of parameter values.
                ICs : list[numpy.ndarray] = physics.initial_condition(param_grid[i]);
                assert isinstance(ICs, list), "type(ICs) = %s, expected list" % str(type(ICs));
                assert len(ICs) == self.n_IC, "len(ICs) = %d, expected %d (=self.n_IC)" % (len(ICs), self.n_IC);

                # Convert ICs to tensors, optionally normalize, then encode.
                X0_list : list[torch.Tensor] = [];
                for k in range(self.n_IC):
                    x0_np : numpy.ndarray = ICs[k];
                    x0_t  : torch.Tensor  = torch.Tensor(x0_np).reshape((1,) + x0_np.shape).to(encoder_device);
                    if has_norm:
                        x0_t = trainer.normalize_tensor(x0_t, k);
                    X0_list.append(x0_t);

                # Encode (positional arguments). Must return a tuple of length self.n_IC.
                Z0_tuple : tuple[torch.Tensor, ...] = self.Encode(*X0_list);
                assert isinstance(Z0_tuple, tuple), "Encode must return a tuple; got %s" % str(type(Z0_tuple));
                assert len(Z0_tuple) == self.n_IC,  "Encode returned %d outputs; expected %d (=self.n_IC)" % (len(Z0_tuple), self.n_IC);

                # Detach to numpy arrays.
                Z0_i : list[numpy.ndarray] = [];
                for k in range(self.n_IC):
                    Z0_i.append(Z0_tuple[k].detach().cpu().numpy());

                Z0.append(Z0_i);

        return Z0;



    def export(self) -> dict:
        """
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        This function extracts everything we need to recreate self from scratch.
        """

        raise RuntimeError("Abstract method EncoderDecoder.export!");


