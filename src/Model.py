# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the Physics directory to the search path.
import  sys;
import  os;
Physics_Path    : str  = os.path.abspath(os.path.join(os.path.dirname(__file__), "Physics"));
sys.path.append(Physics_Path);

import  logging;

import  torch;
import  numpy;

from    Physics     import  Physics;


# Set up logging.
LOGGER  : logging.Logger    = logging.getLogger(__name__);


# activation dict
act_dict = {'ELU'           : torch.nn.ELU,
            'hardshrink'    : torch.nn.Hardshrink,
            'hardsigmoid'   : torch.nn.Hardsigmoid,
            'hardtanh'      : torch.nn.Hardtanh,
            'hardswish'     : torch.nn.Hardswish,
            'leakyReLU'     : torch.nn.LeakyReLU,
            'logsigmoid'    : torch.nn.LogSigmoid,
            'PReLU'         : torch.nn.PReLU,
            'ReLU'          : torch.nn.ReLU,
            'ReLU6'         : torch.nn.ReLU6,
            'RReLU'         : torch.nn.RReLU,
            'SELU'          : torch.nn.SELU,
            'CELU'          : torch.nn.CELU,
            'GELU'          : torch.nn.GELU,
            'sigmoid'       : torch.nn.Sigmoid,
            'SiLU'          : torch.nn.SiLU,
            'mish'          : torch.nn.Mish,
            'softplus'      : torch.nn.Softplus,
            'softshrink'    : torch.nn.Softshrink,
            'tanh'          : torch.nn.Tanh,
            'tanhshrink'    : torch.nn.Tanhshrink}




# -------------------------------------------------------------------------------------------------
# MLP class
# -------------------------------------------------------------------------------------------------

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(   self, 
                    widths          : list[int],
                    activation      : str           = 'sigmoid',
                    reshape_index   : int           = None, 
                    reshape_shape   : list[int]     = None) -> None:
        r"""
        This class defines a standard multi-layer network network.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        widths : list[int]
            A list of integers specifying the widths of the layers (including the 
            dimensionality of the domain of each layer, as well as the co-domain of the final 
            layer). Suppose this list has N elements. Then the network will have N - 1 layers. 
            The i'th layer maps from \mathbb{R}^{widths[i]} to \mathbb{R}^{widths[i + 1]}. Thus, 
            the i'th element of this list represents the domain of the i'th layer AND the 
            co-domain of the i-1'th layer.

        activation : str
            A string specifying which activation function we want to use at the end of each 
            layer (except the final one). We use the same activation for each layer. 

        reshape_index : int
            This argument specifies if we should reshape the network's input or output 
            (or neither). If the user specifies reshape_index, then it must be either 0 or -1. 
            Further, in this case, they must also specify reshape_shape (you need to specify both 
            together). If it is 0, then reshape_shape specifies how we reshape the input before 
            passing it through the network (the input to the first layer). If reshape_index is -1, 
            then reshape_shape specifies how we reshape the network output before returning it 
            (the output to the last layer). 

        reshape_shape : list[int] 
            This is a list of k integers specifying the final k dimensions of the shape of the 
            input to the first layer (if reshape_index == 0) or the output of the last layer 
            (if reshape_index == -1). You must specify this argument if and only if you specify 
            reshape_index. 



        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        Nothing!
        """

        # Checks
        assert(isinstance(reshape_shape, list));
        for i in range(len(reshape_shape)):
            assert(isinstance(reshape_shape[i], int));
            assert(reshape_shape[i] > 0);
        assert(isinstance(widths, list));
        for i in range(len(widths)):
            assert(isinstance(widths[i], int));
            assert(widths[i] > 0);
        assert((reshape_index is None) or (reshape_index in [0, -1]));
        assert((reshape_shape is None) or (numpy.prod(reshape_shape) == widths[reshape_index]));

        super().__init__();

        # Note that width specifies the dimensionality of the domains and co-domains of each layer.
        # Specifically, the i'th element specifies the input dimension of the i'th layer, while 
        # the final element specifies the dimensionality of the co-domain of the final layer. Thus, 
        # the number of layers is one less than the length of widths.
        self.n_layers       : int                   = len(widths) - 1;
        self.widths         : list[int]             = widths;

        # Set up the affine parts of the layers.
        self.layers            : list[torch.nn.Module] = [];
        for k in range(self.n_layers):
            self.layers += [torch.nn.Linear(widths[k], widths[k + 1])];
        self.layers = torch.nn.ModuleList(self.layers);

        # Now, initialize the weight matrices and bias vectors in the affine portion of each layer.
        self.init_weight();

        # Reshape input to the 1st layer or output of the last layer.
        self.reshape_index : int        = reshape_index;
        self.reshape_shape : list[int]  = reshape_shape;

        # Set up the activation function. 
        self.activation     : str       = activation;
        self.activation_fn  : callable  = act_dict[self.activation]();
        LOGGER.info("Initializing a MultiLayerPerceptron with widths %s, activation %s, reshape_shape = %s (index %d)" \
                    % (str(self.widths), self.activation, str(self.reshape_shape), self.reshape_index));

        # All done!
        return
    


    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """
        This function defines the forward pass through self.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        X : torch.Tensor
            A tensor holding a batch of inputs. We pass this tensor through the network's layers 
            and then return the result. If self.reshape_index == 0 and self.reshape_shape has k
            elements, then the final k elements of X's shape must match self.reshape_shape. 


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        X_Pred : torch.Tensor, shape = X.Shape
            The image of X under the network's layers. If self.reshape_index == -1 and 
            self.reshape_shape has k elements, then we reshape the output so that the final k 
            elements of its shape match those of self.reshape_shape.
        """

        # If the reshape_index is 0, we need to reshape X before passing it through the first 
        # layer.
        if (self.reshape_index == 0):
            # Make sure the input has a proper shape. There is a lot going on in this line; let's
            # break it down. If reshape_index == 0, then we need to reshape the input, X, before
            # passing it through the layers. Let's assume that reshape_shape has k elements. Then,
            # we need to squeeze the final k dimensions of the input, X, so that the resulting 
            # tensor has a final dimension size that matches the input dimension size for the first
            # layer. The check below makes sure that the final k dimensions of the input, X, match
            # the stored reshape_shape.
            assert(list(X.shape[-len(self.reshape_shape):]) == self.reshape_shape);
            
            # Now that we know the final k dimensions of X have the correct shape, let's squeeze 
            # them into 1 dimension (so that we can pass the squeezed tensor through the first 
            # layer). To do this, we reshape X by keeping all but the last k dimensions of X, and 
            # replacing the last k with a single dimension whose size matches the dimensionality of
            # the domain of the first layer. Note that we use torch.Tensor.view instead of 
            # torch.Tensor.reshape in order to avoid data copying.
            X = X.view(list(X.shape[:-len(self.reshape_shape)]) + [self.widths[self.reshape_index]]);

        # Pass X through the network layers (except for the final one, which has no activation 
        # function).
        for i in range(self.n_layers - 1):
            X : torch.Tensor = self.layers[i](X)            # apply linear layer
            X : torch.Tensor = self.activation_fn(X)        # apply activation

        # Apply the final (output) layer.
        X = self.layers[-1](X)

        # If the reshape_index is -1, then we need to reshape the output before returning. 
        if (self.reshape_index == -1):
            # In this case, we need to split the last dimension of X, the output of the final
            # layer, to match the reshape_shape. This is precisely what the line below does. Note
            # that we use torch.Tensor.view instead of torch.Tensor.reshape in order to avoid data 
            # copying. 
            X = X.view(list(X.shape[:-1]) + self.reshape_shape)

        # All done!
        return X
    


    def init_weight(self) -> None:
        """
        This function initializes the weight matrices and bias vectors in self's layers. It takes 
        no arguments and returns nothing!
        """

        # TODO(kevin): support other initializations?
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
        
        # All done!
        return



# -------------------------------------------------------------------------------------------------
# Autoencoder class
# -------------------------------------------------------------------------------------------------

class Autoencoder(torch.nn.Module):
    def __init__(   self,                     
                    reshape_shape   : list[int],
                    widths          : list[int], 
                    activation      : str           = 'tanh') -> None:
        r"""
        Initializes an Autoencoder object. An Autoencoder consists of two networks, an encoder, 
        E : \mathbb{R}^F -> \mathbb{R}^L, and a decoder, D : \mathbb{R}^L -> \marthbb{R}^F. We 
        assume that the dataset consists of samples of a parameterized L-manifold in 
        \mathbb{R}^F. The idea then is that E and D act like the inverse coordinate patch and 
        coordinate patch, respectively. In our case, E and D are trainable neural networks. We 
        try to train E and map data in \mathbb{R}^F to elements of a low dimensional latent 
        space (\mathbb{R}^L) which D can send back to the original data. (thus, E, and D should
        act like inverses of one another).

        The Autoencoder class implements this model as a trainable torch.nn.Module object. 


        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        reshape_shape : list[int]
            This is a list of k integers specifying the final k dimensions of the shape of the 
            input to the first layer (if reshape_index == 0) or the output of the last layer (if 
            reshape_index == -1). 
        
        widths : list[int]
            A list of integers specifying the widths of the layers in the encoder. We use the 
            revere of this list to specify the widths of the layers in the decoder. See the 
            docstring for the MultiLayerPerceptron class for details on how Widths defines a 
            network.

        activation : str
            specifies which activation function we want to use at the end of each layer (except 
            the final one) in the encoder and decoder. We use the same activation for each layer. 



        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Checks
        assert(isinstance(reshape_shape, list));
        for i in range(len(reshape_shape)):
            assert(isinstance(reshape_shape[i], int));
            assert(reshape_shape[i] > 0);
        assert(isinstance(widths, list));
        for i in range(len(widths)):
            assert(isinstance(widths[i], int));
            assert(widths[i] > 0);

        # Run the superclass initializer.
        super().__init__();
        
        # Store information (for return purposes).
        self.n_IC           : int       = 1;
        self.widths         : list[int] = widths;
        self.n_z            : int       = widths[-1];
        self.activation     : str       = activation;
        self.reshape_shape  : list[int] = reshape_shape;
        LOGGER.info("Initializing an Autoencoder with latent space dimension %d" % self.n_z);

        # Build the encoder, decoder.
        LOGGER.info("Initializing the encoder...");
        self.encoder = MultiLayerPerceptron(
                            widths              = widths, 
                            activation          = activation,
                            reshape_index       = 0,                    # We need to flatten the spatial dimensions of each FOM frame.
                            reshape_shape       = reshape_shape);

        LOGGER.info("Initializing the decoder...");
        self.decoder = MultiLayerPerceptron(
                            widths              = widths[::-1],         # Reverses the order of the the list.
                            activation          = activation,
                            reshape_index       = -1,               
                            reshape_shape       = reshape_shape);       # We need to reshape the network output to a FOM frame.


        # All done!
        return;




    def Encode(self, X : torch.Tensor) -> torch.Tensor:
        """
        This function encodes a set of displacement and velocity frames.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        X : torch.Tensor, shape = (n_Frames,) + self.reshape_shape
            X[i, ...] holds the i'th frame we want to encode. 


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Z : torch.Tensor, shape = (n_Frasmes, self.n_z)
            i,j element holds the j'th component of the encoding of the i'th FOM frame.
        """

        # Check that the inputs have the correct shape.
        assert(len(X.shape)         ==  len(self.reshape_shape) + 1);
        assert(list(X.shape[1:])    ==  self.reshape_shape);
    
        # Encode the frames!
        return self.encoder(X);



    def Decode(self, Z : torch.Tensor)-> torch.Tensor:
        """
        This function decodes a set of latent frames.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Z : torch.Tensor, shape = (n_Frames, self.n_z)
           i,j element holds the j'th component of the encoding of the i'th frame.
     

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        R : torch.Tensor, shpe = (n_Frames,) + self.reshape_shape
            R[i ...] represents the reconstruction of the i'th FOM frame.
        """

        # Check that the input has the correct shape. 
        assert(len(Z.shape)   == 2);
    
        # Decode the frames!
        return self.decoder(Z);




    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """
        This function passes X through the encoder, producing a latent state, Z. It then passes 
        Z through the decoder; hopefully producing a vector that approximates X.
        

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        X : torch.Tensor, shape = (n_Frames,) + self.reshape_shape
            A tensor holding a batch of inputs. We pass this tensor through the encoder + decoder 
            and then return the result.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Y : torch.Tensor, shape = X.shape
            The image of X under the encoder and decoder. 
        """

        # Encoder the input
        Z : torch.Tensor    = self.Encode(X);

        # Now decode z.
        Y : torch.Tensor    = self.Decode(Z);

        # All done! Hopefully Y \approx X.
        return Y;



    def latent_initial_conditions(  self,
                                    param_grid     : numpy.ndarray, 
                                    physics        : Physics) -> list[list[numpy.ndarray]]:
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

        physics : Physics
            A "Physics" object that, among other things, stores the IC for each combination of 
            parameter values. This physics object should have the same number of initial conditions as 
            self.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------
        
        Z0 : list[list[numpy.ndarray]], len = n_param
            An n_param element list whose i'th element is an n_IC element list whose j'th element 
            is an numpy.ndarray of shape (1, n_z) whose k'th element holds the k'th component of 
            the encoding of the initial condition for the j'th derivative of the latent dynamics 
            corresponding to the i'th combination of parameter values.
        
            If we let U0_i denote the FOM IC for the i'th set of parameters, then the i'th element of 
            the returned list is [self.encoder(U0_i)].
        """

        # Checks.
        assert(isinstance(param_grid, numpy.ndarray));
        assert(len(param_grid.shape) == 2);
        assert(physics.n_IC     == self.n_IC);

        # Figure out how many combinations of parameter values there are.
        n_param     : int                   = param_grid.shape[0];
        Z0          : list[numpy.ndarray]   = [];
        LOGGER.debug("Encoding initial conditions for %d parameter values" % n_param);

        # Cycle through the parameters.
        for i in range(n_param):
            # Fetch the IC for the i'th set of parameters. Then map it to a tensor.
            u0 : numpy.ndarray  = physics.initial_condition(param_grid[i])[0];
            u0                  = torch.Tensor(u0).reshape((1,) + u0.shape);

            # Encode the IC, then map the encoding to a numpy array.
            z0 : torch.Tensor   = self.Encode(u0).detach().numpy();

            # Append the new IC to the list of latent ICs
            Z0.append([z0]);

        # Return the list of latent ICs.
        return Z0;



    def export(self) -> dict:
        """
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        This function extracts everything we need to recreate self from scratch. Specifically, we 
        extract the encoder/decoder state dictionaries, self's architecture, activation function 
        and reshape_shape. We store and return this information in a dictionary.
         
        You can pass the returned dictionary to the load_Autoencoder method to generate an 
        Autoencoder object that is identical to self.
        """

        # TO DO: deep export which includes all information needed to re-initialize self from 
        # scratch. This would probably require changing the initializer.

        dict_ = {   'encoder state'  : self.encoder.cpu().state_dict(),
                    'decoder state'  : self.decoder.cpu().state_dict(),
                    'widths'         : self.widths, 
                    'activation'     : self.activation, 
                    'reshape_shape'  : self.reshape_shape};
        return dict_;



def load_Autoencoder(dict_ : dict) -> Autoencoder:
    """
    This function builds an Autoencoder object using the information in dict_. dict_ should be 
    the dictionary returned by the export method for some Autoencoder object (or a de-serialized 
    version of one). The Autoencoder that we recreate should be an identical copy of the object 
    that generated dict_.

    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    dict_: dict
        This should be a dictionary returned by an Autoencoder's export method.

    
    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    AE : Autoencoder 
        An Autoencoder object that is identical to the one that created dict_!
    """

    LOGGER.info("De-serializing an Autoencoder..." );

    # First, extract the parameters we need to initialize an Autoencoder object with the same 
    # architecture as the one that created dict_.
    widths          : list[int] = dict_['widths'];
    activation      : list[int] = dict_['activation'];
    reshape_shape   : list[int] = dict_['reshape_shape'];

    # Now... initialize an Autoencoder object.
    AE = Autoencoder(widths = widths, activation = activation, reshape_shape = reshape_shape);

    # Now, update the encoder/decoder parameters.
    AE.encoder.load_state_dict(dict_['encoder state']); 
    AE.decoder.load_state_dict(dict_['decoder state']); 

    # All done, AE is now identical to the Autoencoder object that created dict_.
    return AE;



# -------------------------------------------------------------------------------------------------
# Displacement, Velocity Autoencoder
# -------------------------------------------------------------------------------------------------

class Autoencoder_Pair(torch.nn.Module):
    """"
    This class defines a pair of auto-encoders for displacement, velocity data. Specifically, each 
    object consists of a pair of auto-encoders, one for processing displacement data and another 
    for processing velocity data. 
    """

    def __init__(   self,
                    reshape_shape       : list[int],
                    widths              : list[int],
                    activation          : str       = "tanh") -> None:
        """
        The initializer for the Autoencoder_Pair class. We assume that each input is a tuple 
        of data, (D, V), representing the displacement and velocity of some system at some point 
        in time. We encode D and V separately; each gets its own autoencoder with distinct weights. 


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------
        
        reshape_shape : list[int], len = k
            specifies the final k dimensions of the shape of the input to the first layer (if 
            reshape_index == 0) or the output of the last layer (if reshape_index == -1). 

        widths : list[int]
            specifies the widths of the layers in each encoder. See Autoencoder docstring.

        activation : str
            specifies which activation function we want to use at the end of each 
            layer (except the final one) in each autoencoder. We use the same activation for each 
            layer. 
    
            
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Checks.
        assert(isinstance(reshape_shape, list));
        for i in range(len(reshape_shape)):
            assert(isinstance(reshape_shape[i], int));
            assert(reshape_shape[i] > 0);
        assert(isinstance(widths, list));
        for i in range(len(widths)):
            assert(isinstance(widths[i], int));
            assert(widths[i] > 0);

        # Call the super class initializer.
        super().__init__();
        LOGGER.info("Initializing an Autoencoder_Pair...");

        # In general, the FOM solution may be vector valued and have multiple spatial dimensions. 
        # We need to know the shape of each FOM frame. 
        self.reshape_shape : list[int]     = reshape_shape; 
        
        # Make sure reshape_shape and widths are compatible. The product of the elements of 
        # reshape_shape is the number of dimensions in each FOM solution frame. This number 
        # represents represents the dimensionality of the input to the encoder (since we pass 
        # a flattened FOM frame as input).
        assert(numpy.prod(self.reshape_shape) == widths[0]);

        # Fetch information about the domain/co-domain.
        self.widths     : list[int]     = widths
        self.n_z        : int           = widths[-1];
        self.n_IC       : int           = 2;

        # Use the settings to set up the activation information for the encoder.
        self.activation : str           =  activation;

        # Next, build the velocity and displacement auto-encoders.
        LOGGER.info("Initializing the Displacement Autoencoder...");
        self.Displacement_Autoencoder   = Autoencoder(  widths          = widths, 
                                                        activation      = activation, 
                                                        reshape_shape   = self.reshape_shape);

        LOGGER.info("Initializing the Velocity Autoencoder...");
        self.Velocity_Autoencoder       = Autoencoder(  widths          = widths, 
                                                        activation      = activation,
                                                        reshape_shape   = self.reshape_shape);



    def Encode(self,
               Displacement_Frames  : torch.Tensor, 
               Velocity_Frames      : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function encodes a set of displacement and velocity frames.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Displacement_Frames : torch.Tensor, shape = (N_Frames,) + self.reshape_shape
            Displacement_Frames[i, ...] represents the displacement portion of the i'th FOM frame.
            Here, N_Frames is the number of frames we want to encode and reshape_shape specifies 
            the shape of each frame. 

        Velocity_Frames : torch.Tensor, shape = (N_Frames,) + self.reshape_shape
            Velocity_Frames[i, ...] represents the velocity portion of the i'th FOM frame. Here, 
            N_Frames is the number of frames we want to encode for each parameter combination and 
            reshape_shape specifies the shape of each frame. 
        

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Latent_Displacement, Latent_Velocity
        
        Latent_Displacement : torch.Tensor, shape = (N_Frames, self.n_z))
            Latent_Displacement[i, :] represents the encoding of the displacement portion of the 
            i'th FOM frame

        Latent_Velocity : torch.Tensor, shape = (N_Frames, self.n_z))
            Latent_Velocity[i, :] represents the encoding of the velocity portion of the i'th FOM 
            frame.
        """

        # Check that we have the same number of displacement, velocity frames.
        assert(isinstance(Displacement_Frames, torch.Tensor));
        assert(isinstance(Velocity_Frames, torch.Tensor));
        assert(len(Displacement_Frames.shape)       ==  len(self.reshape_shape) + 1);
        assert(Displacement_Frames.shape            ==  Velocity_Frames.shape);
        assert(list(Displacement_Frames.shape[1:])  ==  self.reshape_shape);
    
        # Encode the displacement frames.
        Latent_Displacement : torch.Tensor = self.Displacement_Autoencoder.Encode( Displacement_Frames);
        Latent_Velocity     : torch.Tensor = self.Velocity_Autoencoder.Encode(     Velocity_Frames);

        # All done!
        return Latent_Displacement, Latent_Velocity;



    def Decode(self,
               Latent_Displacement  : torch.Tensor, 
               Latent_Velocity      : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function decodes a set of displacement and velocity frames.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Latent_Displacement : torch.Tensor, shape = (N_Frames, self.n_z)
            i,j element represents the j'th component of the encoding of the displacement portion 
            of the i'th FOM frame.

        Latent_Velocity : torch.Tensor, shape = (N_Frames, self.n_z)
            i,j element represents the j'th component of the encoding of the velocity portion of 
            the i'th FOM frame.
     

        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Reconstructed_Displacement, Reconstructed_Velocity
         
        Reconstructed_Displacement : torch.Tensor, shape = (N_Frames,) + self.reshape_shape
            Reconstructed_Displacement[i, ...] represents the reconstruction of the displacement 
            portion of the i'th FOM frame. 

        Reconstructed_Velocity : torch.Tensor, shape = (N_Frames,) + self.reshape_shape
            Reconstructed_Velocity[i, ...] represents the reconstruction of the velocity portion 
            of i'th FOM frame.
        """

        # Check that we have the same number of displacement, velocity frames.
        assert(isinstance(Latent_Displacement, torch.Tensor));
        assert(isinstance(Latent_Velocity, torch.Tensor));
        assert(len(Latent_Displacement.shape)   == 2);
        assert(Latent_Velocity.shape            == Latent_Displacement.shape);
    
        # Encode the displacement frames.
        Reconstructed_Displacement  : torch.Tensor  = self.Displacement_Autoencoder.Decode( Latent_Displacement);
        Reconstructed_Velocity      : torch.Tensor  = self.Velocity_Autoencoder.Decode(     Latent_Velocity);

        # All done!
        return Reconstructed_Displacement, Reconstructed_Velocity;



    def forward(self, Displacement_Frames : torch.Tensor, Velocity_Frames : torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The forward method for the Autoencoder_Pair class. It encodes and then decodes 
        Displacement_Frames and Velocity_Frames.



        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Displacement_Frames : torch.Tensor, shape = (N_Frames,) + self.reshape_shape
            Displacement_Frames[i, ...] represents the displacement portion of the i'th FOM frame.
            Here, N_Frames is the number of frames we want to encode and reshape_shape specifies 
            the shape of each frame. 

        Velocity_Frames : torch.Tensor, shape = (N_Frames,) + self.reshape_shape
            Velocity_Frames[i, ...] represents the velocity portion of the i'th FOM frame. Here, 
            N_Frames is the number of frames we want to encode for each parameter combination and 
            reshape_shape specifies the shape of each frame. 


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Reconstructed_Displacement, Reconstructed_Velocity

        Reconstructed_Displacement : torch.Tensor, shape = (N_Frames,) + self.reshape_shape
            Reconstructed_Displacement[i, ...] represents the reconstruction of the displacement 
            portion of the i'th FOM frame. 

        Reconstructed_Velocity : torch.Tensor, shape = (N_Frames,) + self.reshape_shape
            Reconstructed_Velocity[i, ...] represents the reconstruction of the velocity portion 
            of i'th FOM frame.
        """

        # Encode the displacement, velocity frames
        Latent_Displacement, Latent_Velocity = self.Encode(     Displacement_Frames   = Displacement_Frames, 
                                                                Velocity_Frames       = Velocity_Frames);

        # Now reconstruct displacement, velocity.
        Reconstructed_Displacement, Reconstructed_Velocity = self.Decode(
                                                                Latent_Displacement = Latent_Displacement, 
                                                                Latent_Velocity     = Latent_Velocity);

        # All done!
        return Reconstructed_Displacement, Reconstructed_Velocity;



    def latent_initial_conditions(  self,
                                    param_grid     : numpy.ndarray, 
                                    physics        : Physics) -> list[list[numpy.ndarray]]:
        """
        This function maps a set of initial conditions for the FOM to initial conditions for the 
        latent space dynamics. Specifically, we take in a set of possible parameter values. For 
        each set of parameter values, we recover the FOM IC (from physics), then map this FOM IC 
        to a latent space IC (by encoding it). We do this for each parameter combination and then 
        return a list housing the latent space ICs.

        
        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        param_grid : numpy.ndarray, shape = (n_param, n_p)
            i,j element of this array holds the value of the j'th parameter in the i'th combination 
            of parameter values. Here, n_p is the number of parameters and n_param is the number
            of combinations of parameter values.

        physics : Physics
            allows us to calculate the IC for each combination of parameter values. This physics 
            object should have the same number of initial conditions as self.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Z0 : list[list[numpy.ndarray]], len = n_param
            i'th element is an n_IC element list whose j'th element is an numpy.ndarray of shape 
            (1, n_z) whose k'th element holds the k'th component of the encoding of the initial
            condition for the j'th derivative of the latent dynamics corresponding to the i'th 
            combination of parameter values.
                
            If we let (U0_i, V0_i) denote the initial FOM displacement and velocity for the i'th 
            combination of parameter values, then the i'th element of the returned list is the list 
            [self.encoder(U0_i, V0_i)[0], self.encoder(U0_i, V0_i)[1]].
        """

        # Checks
        assert(isinstance(param_grid, numpy.ndarray));
        assert(len(param_grid.shape) == 2);
        assert(self.n_IC        == physics.n_IC);

        # Figure out how many combinations of parameter values there are.
        n_param     : int                   = param_grid.shape[0];
        Z0          : list[numpy.ndarray]   = [];
        LOGGER.debug("Encoding initial conditions for %d combinations of parameter values" % n_param);

        # Cycle through the parameters.
        for i in range(n_param):
            # Get the ICs for the i'th combination of parameter values.
            ICs     : list[numpy.ndarray]   = physics.initial_condition(param_grid[i]);
            u0      : numpy.ndarray         = ICs[0];
            v0      : numpy.ndarray         = ICs[1];
            
            # Map the ICs to a tensor.
            u0      = torch.Tensor(u0).reshape((1,) + u0.shape);
            v0      = torch.Tensor(v0).reshape((1,) + v0.shape);

            # Encode the IC, then map the encoding to a numpy array.
            z0, Dz0 = self.Encode(  Displacement_Frames = u0, 
                                    Velocity_Frames     = v0);
            z0      : numpy.ndarray = z0.detach().numpy();
            Dz0     : numpy.ndarray = Dz0.detach().numpy();

            # Concatenate the IC's and append them to the list.
            Z0.append([z0, Dz0]);

        # Return the list of latent ICs.
        return Z0;



    def export(self) -> dict:
        """
        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        This function extracts everything we need to recreate self from scratch. Specifically, we 
        extract the encoder/decoder state dictionaries, self's architecture, activation function 
        and reshape_shape. We store and return this information in a dictionary.
         
        You can pass the returned dictionary to the load_Autoencoder_Pair method to generate an 
        Autoencoder object that is identical to self.
        """

        dict_ = {   'reshape_shape'     : self.reshape_shape,
                    'widths'            : self.widths,
                    'activation'        : self.activation,
                    'Displacement dict' : self.cpu().Displacement_Autoencoder.export(),
                    'Velocity dict'     : self.cpu().Velocity_Autoencoder.export()};
        return dict_;
    


def load_Autoencoder_Pair(dict_ : dict) -> Autoencoder_Pair:
    """
    This function builds a Autoencoder_Pair object using the information in dict_. dict_ should be 
    the dictionary returned by the export method for some Autoencoder_Pair object (or a 
    de-serialized version of one). The Autoencoder_Pair that we recreate should be an identical 
    copy of the object that generated dict_.


    
    -----------------------------------------------------------------------------------------------
    Arguments
    -----------------------------------------------------------------------------------------------

    dict_ : dict
        This should be a dictionary returned by a Autoencoder_Pair's export method.

    

    -----------------------------------------------------------------------------------------------
    Returns
    -----------------------------------------------------------------------------------------------

    AEP : Autoencoder_Pair
        A Autoencoder_Pair object that is identical to the one that created dict_!
    """

    LOGGER.info("De-serializing an Autoencoder_Pair..." );

    # First, extract the information we need to initialize a Autoencoder_Pair object with the same 
    # architecture as the one that created dict_.
    reshape_shape   : list[int] = dict_['reshape_shape'];
    widths          : list[int] = dict_['widths'];
    activation      : str       = dict_['activation'];

    # Now initialize the Autoencoder_Pair.
    AEP                     = Autoencoder_Pair( widths          = widths, 
                                                activation      = activation,
                                                reshape_shape   = reshape_shape);
    
    # Now replace its auto-encoders.
    AEP.Displacement_Autoencoder    = load_Autoencoder(dict_['Displacement dict']);
    AEP.Velocity_Autoencoder        = load_Autoencoder(dict_['Velocity dict']);

    # All done!
    return AEP;