# -------------------------------------------------------------------------------------------------
# Import and Setup
# -------------------------------------------------------------------------------------------------

import      logging;

import      torch;
import      numpy;

# Set up logging.
LOGGER  : logging.Logger    = logging.getLogger(__name__);




# -------------------------------------------------------------------------------------------------
# EncoderDecoder
# -------------------------------------------------------------------------------------------------

class EncoderDecoder(torch.nn.Module):
    r"""
    This defines the base class interface for the learned encoder/decoder.
    
    In the HLaSDI framework, a ROM consists of an EncoderDecoder model and a LatentDynamics 
    object (acting as the Encoder/Decoder and Latent Dynamics portions of the ROM, respectively). 
    These are jointly trained via a Trainer object using data from a Physics object. The 
    LatentDynamics object holds the learnedLatentDynamics coefficients for the training set,
    while an Interpolate object samples LatentDynamics coefficients for testing parameter 
    combinations. A Sampler object determines how the model picks which testing example to add
    to the training set after each round of training.
     
    An EncoderDecoder object defines the encoder and decoder portion of a ROM. The Encoder maps 
    a snapshot (fixed time) of the FOM state to a low dimensional latent encoding. Likewise, the 
    decoder learns a mapping from latent encodings to FOM states (ideally acting like the inverse
    of the Encoder when restricted to the FOM solution manifold). 

    If the governing FOM dynamics involves n_IC time derivatives (n_IC'th order in time), then 
    a FOM snapshot must consist of the solution and its first n_IC - 1 time derivatives at a 
    fixed time. EncoderDecoder objects are designed to operate in such situations by learning 
    a separate encoder-decoder pair for each time derivative of the solution. In general, however, 
    each encoder-decoder pair will have the same latent space, and their latent encodings are 
    generally joined by a latent dynamics models that involves all of their encodings.

    EncoderDecoder models natively support multi-stage decoding. Specifically, the decoder is a 
    weighted combination of n_Decoders sub-models (each which map latent encodings FOM 
    snapshots). The user can dynamically adjust the weights and change which models are enabled 
    via the `Set_Decoder_Weight` and `Set_Decoder_Active` methods, respectively. This approach 
    allows for multi-stage Training methods (mLaSDI style).

    

    -----------------------------------------------------------------------------------------------
    Class/instance variables
    -----------------------------------------------------------------------------------------------

    n_IC : int
        Number of FOM/latent components handled together, often corresponding to the state and the
        first `n_IC - 1` time derivatives needed by the latent dynamics.
    
    n_z : int
        Latent-space dimension for each component returned by `Encode(...)`.
    
    n_Decoders : int
        Number of decoder stages available to `Decode(...)`.
    
    config : dict
        The `encoder_decoder` configuration dictionary used by the concrete architecture.
    
    Decoder_Active : numpy.ndarray, shape = (n_Decoders,)
        Boolean mask indicating which decoder stages contribute to `Decode(...)`.
    
    Decoder_Weight : numpy.ndarray, shape = (n_IC, n_Decoders)
        Weight applied to each decoder stage for each output component when forming the weighted
        decoder sum.
    
    trainable : bool
        A boolean indicating if the trainer should train the EncoderDecoder object's parameters.
        Technically this is just a boolean, it's up to the trainer to actually respect it.

        
        
    -----------------------------------------------------------------------------------------------
    Subclassing
    -----------------------------------------------------------------------------------------------
    
    To define a new architecture, subclass `EncoderDecoder`, call `super().__init__(...)`, register
    any PyTorch modules as normal `torch.nn.Module` attributes, and implement:

    - `Encode(*Xs)`: accept exactly `n_IC` FOM tensors and return a tuple of `n_IC` latent tensors,
      each typically shaped `(batch, n_z)`.
    
    - `Eval_Decoder(i_Decoder, *Zs)`: evaluate one decoder stage on exactly `n_IC` latent tensors
      and return a tuple of `n_IC` reconstructed FOM tensors.

    The base `Decode(...)` method handles active decoder selection and weighted summation, and
    `forward(...)` implements encode-then-decode.  Subclasses that store additional non-PyTorch
    state should extend `export()` and `load()` while preserving the base metadata and decoder
    active/weight state.
    """
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
                    trainable   : bool, 
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
        of these outputs. Thus, an EncoderDecoder object is defined by four variables:

            n_IC (the number of initial conditions)
            n_z (the latent space dimension)
            n_decoders (the number of decoders)
            trainable (if the trainer should train the EncoderDecoder)
        
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
        
            - latent_initial_conditions: Maps provided FOM initial conditions to the latent space.
              
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

        trainable : bool 
            Indicates if the trainer should train the EncoderDecoder parameters.

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
        assert isinstance(trainable, bool), "trainable must be a bool, not %s"  % str(type(trainable));
        assert n_IC > 0,                    "n_IC = %d; must be positive"       % n_IC;
        assert n_z > 0,                     "n_z = %d; must be positive"        % n_z;
        assert n_Decoders > 0,              "n_Decoders = %d; must be positive" % n_Decoders;

        # Run the superclass initializer.
        super().__init__();
        
        # Store information (for return purposes).
        self.n_IC           : int       = n_IC;
        self.n_z            : int       = n_z;
        self.n_Decoders     : int       = n_Decoders;
        self.trainable      : bool      = trainable;
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
    # set_trainable, Set_Decoder_Active, and Set_Decoder_Weight.
    # ---------------------------------------------------------------------------------------------

    def set_trainable(self, trainable : bool) -> None:
        """
        Enable or disable gradients for all registered EncoderDecoder parameters.

        Concrete subclasses should call this once after constructing all submodules. This keeps
        the public `trainable` flag and PyTorch's `requires_grad` flags consistent, so frozen
        EncoderDecoder objects are omitted from optimizer updates and do not accumulate gradients.


        -------------------------------------------------------------------------------------------
        Args:

        trainable : bool
            If True, all EncoderDecoder parameters require gradients. If False, all parameters are
            frozen.
        """

        assert isinstance(trainable, bool), "trainable must be a bool, not %s" % str(type(trainable));
        self.trainable = trainable;
        for param in self.parameters():
            param.requires_grad_(trainable);
        return;



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
                                    FOM_IC         : list[list[numpy.ndarray | torch.Tensor]]) -> list[list[numpy.ndarray]]:
        """
        This function maps a set of initial conditions for the FOM to initial conditions for the 
        latent space dynamics. The caller owns any physics lookup and normalization; this method
        only encodes the provided FOM initial-condition arrays/tensors. This keeps the
        EncoderDecoder package independent of Physics, Trainer, and Rollout implementations.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        FOM_IC : list[list[numpy.ndarray | torch.Tensor]], len = N 
            A list of FOM initial conditions to encode. The i'th element should be a list of n_IC 
            FOM initial conditions. If you are using a trainer with normalization, these should 
            already be normalized. 

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        ROM_IC : list[list[numpy.ndarray]], len = N
            An N element list whose i'th element is an n_IC element list holding the encoding
            of the i'th element of FOM_IC. Each encoded initial condition has shape (n_z).
        """

        # Checks.
        assert isinstance(FOM_IC, list),    "FOM_IC must be a list of normalized initial conditions, got %s" % str(type(FOM_IC));
        N : int = len(FOM_IC);

        # Determine device for encoding.
        encoder_device : torch.device = next(self.parameters()).device;

        # Setup 
        LOGGER.debug("Encoding %d sets of FOM initial conditions." % N);
        ROM_IC      : list[list[numpy.ndarray]] = [];

        with torch.no_grad():
            for i in range(N):
                # Get the i'th set of IC's
                ith_FOM_IC : list[numpy.ndarray] = FOM_IC[i];
                assert isinstance(ith_FOM_IC, list), "type(FOM_IC[%d]) = %s, expected list" % (i, str(type(ith_FOM_IC)));
                assert len(ith_FOM_IC) == self.n_IC, "len(FOM_IC[%d]) = %d, expected %d (=self.n_IC)" % (i, len(ith_FOM_IC), self.n_IC);

                # Convert ICs to tensors, then encode.
                ith_FOM_IC_list : list[torch.Tensor] = [];
                for k in range(self.n_IC):
                    ith_FOM_IC_k  : numpy.ndarray | torch.Tensor = ith_FOM_IC[k];
                    assert isinstance(ith_FOM_IC_k, (numpy.ndarray, torch.Tensor)), "type(FOM_IC[%d][%d]) = %s, expected numpy.ndarray or torch.Tensor" % (i, k, str(type(ith_FOM_IC_k)));
                    ith_FOM_IC_t  : torch.Tensor  = torch.as_tensor(ith_FOM_IC_k, dtype = torch.float32, device = encoder_device);
                    ith_FOM_IC_t = ith_FOM_IC_t.reshape((1,) + tuple(ith_FOM_IC_t.shape));
                    ith_FOM_IC_list.append(ith_FOM_IC_t);

                # Encode (positional arguments). This returns a tuple of length self.n_IC.
                ith_ROM_IC_tuple : tuple[torch.Tensor, ...] = self.Encode(*ith_FOM_IC_list);
                assert isinstance(ith_ROM_IC_tuple, tuple), "Encode must return a tuple; got %s" % str(type(ith_ROM_IC_tuple));
                assert len(ith_ROM_IC_tuple) == self.n_IC,  "Encode returned %d outputs; expected %d (=self.n_IC)" % (len(ith_ROM_IC_tuple), self.n_IC);

                # Detach to one-dimensional numpy arrays.
                ith_ROM_IC_np : list[numpy.ndarray] = [];
                for k in range(self.n_IC):
                    ith_ROM_IC_np.append(ith_ROM_IC_tuple[k].detach().cpu().numpy().reshape(-1));

                ROM_IC.append(ith_ROM_IC_np);

        return ROM_IC;



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
