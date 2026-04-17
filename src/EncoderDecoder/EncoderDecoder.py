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
    # i'th element is True if the i'th decoder is currently active, otherwise False.
    # Defaults to an array whose 0 element is True and whose other elements are False (only the 
    # first decoder is active).
    Decoder_Active : numpy.ndarray;

    # i,j element holds the weight of the j'th Decoder for the i'th IC. Defaults to an array of 
    # 1's (all decoders get equal weight).
    Decoder_Weight : numpy.ndarray;         # shape (n_IC, n_Decoders)


    def __init__(   self, 
                    n_IC        : int, 
                    n_z         : int,
                    n_Decoders  : int,
                    config      : dict) -> None:
        r"""
        Initializes a EncoderDecoder object. A EncoderDecoder object does two things. a) It can 
        encode FOM states (frames) to their latent encodings, and b) it can decode those latent
        encodings back to the FOM state. In general, the encoder accepts n_IC elements of the 
        FOM space, then encodes them into n_IC elements of the latent space (\\mathbb{R}^{n_z}). 
        Likewise, the Decoder(s) accept n_IC elements of the latent space and decodes them to 
        n_IC elements of the FOM space. 
        
        EncoderDecoder objects natively support using multiple decoders, which enables things 
        like multi-stage training (mLaSDI). The actual decode method should return a weighted sum 
        of these outputs. Thus, an EncoderDecoder object is defined by three variables:

            n_IC (the number of initial conditions)
            n_z (the latent space dimension)
            n_decoders (the number of decoders)
        
        The encoder must map n_IC elements of the FOM space to n_IC elements of \mathbb{R}^{n_z}. 
        Each decoder decoder must map n_IC elements of \mathbb{R}^{n_z} to n_IC elements of the 
        FOM space, and the "Decode" method must return a weighted sum of the decoder outputs.

        To implement a EncoderDecoder subclass, you must implement the Encode, Eval_Decoder, and
        save/load methods. 
        
            - Encode should accept a set of n_IC inputs from the FOM space, encode them, and then 
            return a tuple housing the encoded inputs. 
            
            - Eval_Decoder should accept an integer, i, and a set of n_IC inputs, evaluate the i'th 
            Decoder on the specified inputs, then return a tuple housing the encodings of the inputs. 
            The Decode method operates by returning a tuple of tensors, the j'th one of which holds

                \sum_{d'th decoder is active} Decoder_Weight[j, d] * Eval_Decoder(d, *Zs)[j]
            
            Thus, Eval_Decoder is quite important.

        The base EncoderDecoder class defines the following methods:
        
            - latent_initial_conditions: Maps a set of initial conditions for a given set of physics 
             to the latent space.
              
            - Set_Decoder_Active: Modifies the Decoder_Active attribute used by Decoder.

            - Set_Decoder_Weight: Modifies the Decoder_Weight attribute used by Decoder.

            - Decode: Computes and returns a weighted sum of the decoder outputs.
             
            - forward: Encodes, then Decodes a set of inputs.

        You are welcome to override any of these in your sub-class, though they should have the 
        same signatures (inputs and outputs) as the base class (otherwise, something will probably 
        break elsewhere in the code).

        
        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        n_IC : int
            The number of initial conditions. The encoder must accept this many elements of the 
            FOM space and map them to the same number of elements of the latent space.

        n_z : int 
            The latent space dimension.

        n_Decoders : int
            The number of decoders.

        config: dict
            The "EncoderDecoder" sub dictionary of the configuration file.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Checks
        assert isinstance(n_IC, int),       "n_IC must be an int, not %s"       % str(type(n_IC));
        assert isinstance(n_z, int),        "n_z must be an int, not %s"        % str(type(n_z));
        assert isinstance(n_Decoders, int), "n_Decoders must be an int, not %s" % str(type(n_Decoders));
        assert n_IC > 0,                    "n_IC = %d; must be positive"       % n_IC;
        assert n_z > 0,                     "n_z = %d; must be positive"        % n_z;
        assert n_Decoders > 0,              "n_Decoders = %d; must be positive" % n_Decoders;

        # Run the superclass initializer.
        super().__init__();
        
        # Store information (for return purposes).
        self.n_IC           : int       = n_IC;
        self.n_z            : int       = n_z;
        self.n_Decoders     : int       = n_Decoders;
        self.config         : dict      = config;

        # Set up Decoder_Weight and Decoder_Active.
        self.Decoder_Active     = numpy.empty((n_Decoders), dtype = numpy.bool_);
        self.Decoder_Active[0]  = True;
        for i in range(1, n_Decoders):
            self.Decoder_Active[i]  = False;
        
        self.Decoder_Weight     = numpy.ones((n_IC, n_Decoders), dtype = numpy.float32);
    
        # All done!
        return;


    
    # ---------------------------------------------------------------------------------------------
    # Set_Decoder_Active and Set_Decoder_Weight.
    # ---------------------------------------------------------------------------------------------

    def Set_Decoder_Active(self, i_Decoder : int, active : bool) -> None:
        """
        Either actives (if active = True) or deactivates (if active = False) the i_Decoder'th 
        decoder.


        -------------------------------------------------------------------------------------------
        Args:

        i_Decoder : int 
            The index of the decoder we want to active. Must be in {0, 1, ... , self.n_Decoders - 1}
        
        active : bool
            Either activates (if True) or deactivates (if False) the i_Decoder'th Decoder.
        """

        # Checks
        assert isinstance(i_Decoder, int),                              "i_Decoder must be an integer, not %s" % str(type(i_Decoder));
        assert isinstance(active, bool),                                "active must be a boolean, not %s" % str(type(active));
        assert (i_Decoder >= 0) and (i_Decoder < self.n_Decoders),      "i_Decoder must be in {0, ... , %d}; got %d" % (self.n_Decoders - 1, i_Decoder)

        # Do the thing!
        self.Decoder_Active[i_Decoder] = active;
    
        # Make sure at least one decoder is active
        assert numpy.sum(self.Decoder_Active) > 0,                      "No decoders active! Can not function!";



    def Set_Decoder_Weight(self, i_IC : int, i_Decoder : int, weight : float) -> None:
        """
        Specifies the weight of the i_Decoder'th decoder for the i_IC'th component (often time 
        derivative) of the FOM solution. 

        -------------------------------------------------------------------------------------------
        Args:
        
        i_IC : int 
            The index of the decoder we want to active. Must be in {0, 1, ... , self.n_Decoders - 1}
        
        i_Decoder : int 
            The index of the decoder we want to active. Must be in {0, 1, ... , self.n_Decoders - 1}

        weight : float | int
            Specifies the weight of the i_Decoder'th decoder for the i_IC'th component of the FOM 
            solution.
        """

        # Checks
        assert isinstance(i_IC, int),                                   "i_IC must be an integer, not %s" % str(type(i_IC));
        assert isinstance(i_Decoder, int),                              "i_Decoder must be an integer, not %s" % str(type(i_Decoder));
        assert isinstance(weight, float) or isinstance(weight, int),    "weight must be numeric, not %s" % str(type(float));
        assert (i_IC >= 0) and (i_IC < self.n_IC),                      "i_IC must be in {0, ... , %d}; got %d" % (self.n_IC - 1, i_IC)
        assert (i_Decoder >= 0) and (i_Decoder < self.n_Decoders),      "i_Decoder must be in {0, ... , %d}; got %d" % (self.n_Decoders - 1, i_Decoder)

        # Do the thing!
        self.Decoder_Weight[i_IC, i_Decoder] = weight;



    
    # ---------------------------------------------------------------------------------------------
    # Encode, Decode, forward.
    # ---------------------------------------------------------------------------------------------

    def Encode(self, *Xs : tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        """
        In general, the Encode method should take n_IC positional arguments, each one containing 
        a batch of elements of the FOM space, and map them to n_IC elements of the latent space. 
        The output must be a tuple of tensors. The input should be n_IC tensors (as positional 
        arguments), each with the same shape.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Xs : self.n_IC torch.Tensor's, each of shape (n_inputs, ...)
            The inputs to be encoded. The i'th one should hold the i'th component of the FOM 
            solution that we want to encode.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Z : tuple[torch.Tensor], len = self.n_IC
            A List of n_IC elements of \\mathbb{R}^{n_z}. 
        """

        raise RuntimeError("Abstract method EncoderDecoder.Encode!");


    def Eval_Decoder(self, i_Decoder : int, *Zs : tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        """
        Passes the n_IC elements of Zs through the i_Decoder'th decoder, then returns the 
        corresponding collection of n_IC elements of the FOM space. In general, the Eval_Decoder 
        method should replace the *Zs argument with n_IC positional arguments, each one containing 
        a batch of elements of the latent space, and map them to n_IC elements of the FOM space. 
        The output must be a tuple of tensors.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        i_Decoder : int 
            The index of the decoder we want to use to compute the decoding. Must be in {0, ... , 
            self.n_Decoders - 1}

        Zs : self.n_IC torch.Tensor's, each of shape (n_inputs, self.n_Z)
            The encodings to be decoded. The i'th one should hold the i'th component of the latent 
            state (often the i'th time derivative of the latent state) that we want to decoder 
            through the i_Decoder'th decoder. 


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Xs : tuple[torch.Tensor], len = self.n_IC
            A List of n_IC elements of the FOM space.
        """ 

        raise RuntimeError("Abstract method EncoderDecoder.Eval_Decoder!");




    def Decode(self, *Zs) -> tuple[torch.Tensor]:
        r"""
        Passes the n_IC elements of Zs through the active decoders, then sums the components 
        of the resulting tensors according to the decoder weights. Specifically, the j'th 
        component of the returned tensor holds the following sum:

            \sum_{d'th decoder is active} Decoder_Weight[j, d] * Eval_Decoder(d, *Z)[j]
        
        Thus, this function decodes a batch of latent states (each consisting of n_IC components)
        to a batch of FOM states (again, each one with n_IC components). We literally "decode"
        the batch of latent states.


        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Zs : n_IC torch.Tensors, each of shape = (n_inputs, ...)
            The latent states to be decoded.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        X : tuple[torch.Tensor], len = self.n_IC
            A List of n_IC elements of the FOM space 
        """

        # Checks.
        assert len(Zs) == self.n_IC,                    "Decode must receive a tuple of %d Tensors, got a tuple of length %d" % (self.n_IC, len(Zs));
        for i in range(self.n_IC):
            assert isinstance(Zs[i], torch.Tensor),     "Each tensor to be Decoded must be a tensor. Component %d is a %s" % (i, str(type(torch.Tensor)));
            assert len(Zs[i].shape) == 2,               "Each tensor to be Decoded be a tensor of shape (-1, %d), Component %d has shape %s" % (self.n_z, i, str(Zs[i].shape));
            assert Zs[i].shape[1]   == self.n_z,        "Each tensor to be Decoded be a tensor of shape (-1, %d), Component %d has shape %s" % (self.n_z, i, str(Zs[i].shape));

        # Decode!
        Xs : list[torch.Tensor | None] = [None]*self.n_IC;

        for d in range(self.n_Decoders):
            if(self.Decoder_Active[d] == True):
                dth_Decodings : tuple[torch.Tensor] = self.Eval_Decoder(d, *Zs);

                for j in range(self.n_IC):
                    w       = float(self.Decoder_Weight[j, d])
                    term    = w* dth_Decodings[j];
                    
                    if Xs[j] is None:
                        Xs[j] = term;
                    else:
                        Xs[j] = Xs[j] + term;

        # All done!
        return tuple(Xs);
                


    def forward(self, *Xs : tuple[torch.Tensor]) -> tuple[torch.Tensor]:
        """
        This function passes the X's through the encoder, producing a latent state, Z. It then 
        passes Z through the decoders; hopefully producing a set of vectors that approximates X.
        

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Xs : n_IC torch.Tensors, each of shape (n_inputs, ...)
            The inputs to be encoded and decoded.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Ys : tuple[torch.Tensor], len = self.n_IC
            A self.n_IC element tuple of torch.Tensors in the FOM space holding the image of X 
            under the encoder and decoder. 
        """

        # Checks.
        assert len(Xs) == self.n_IC,                    "forward must receive a tuple of %d Tensors, got a tuple of length %d" % (self.n_IC, len(Xs));
        for i in range(self.n_IC):
            assert isinstance(Xs[i], torch.Tensor),     "Each tensor to be Decoded must be a tensor. Component %d is a %s" % (i, str(type(torch.Tensor)));

        # Encode and Decode!
        Zs : tuple[torch.Tensor] = self.Encode(*Xs);
        Ys : tuple[torch.Tensor] = self.Decode(*Zs);

        # All done!
        return Ys;



    # ---------------------------------------------------------------------------------------------
    # latent_initial_conditions
    # ---------------------------------------------------------------------------------------------

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



    # ---------------------------------------------------------------------------------------------
    # Save, Load
    # ---------------------------------------------------------------------------------------------

    def export(self) -> dict:
        """
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        This function extracts everything we need to recreate self from scratch.
        """

        dict_ =     {   "Decoder_Weight"    : self.Decoder_Weight,
                        "Decoder_Active"    : self.Decoder_Active,
                        "n_z"               : self.n_z,
                        "n_IC"              : self.n_IC,
                        "n_Decoders"        : self.n_Decoders };
    
        return dict_;


    
    def load(self, dict_):
        """
        dict_: Should be the dict returned by the export method.
        """
        
        # Load the decoder weights/which ones are active.
        self.Decoder_Weight     = dict_['Decoder_Weight'];
        self.Decoder_Active     = dict_['Decoder_Active'];
        
        # Make sure n_z, n_IC, and n_Decoders match what we just set up.
        assert self.n_z         == dict_['n_z'];
        assert self.n_IC        == dict_['n_IC'];
        assert self.n_Decoders  == dict_['n_Decoders'];


