# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the Physics directory to the search path.
import  sys;
import  os;
Physics_Path    : str  = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "Physics"));
sys.path.append(Physics_Path);

import  logging;
from    typing      import  Callable;

import  torch;
import  numpy;

from    Physics     import  Physics;


# Set up logging.
LOGGER  : logging.Logger    = logging.getLogger(__name__);


# activation dict
act_dict = {'elu'           : torch.nn.functional.elu,
            'hardshrink'    : torch.nn.functional.hardshrink,
            'hardsigmoid'   : torch.nn.functional.hardsigmoid,
            'hardtanh'      : torch.nn.functional.hardtanh,
            'hardswish'     : torch.nn.functional.hardswish,
            'leakyReLU'     : torch.nn.functional.leaky_relu,
            'logsigmoid'    : torch.nn.functional.logsigmoid,
            'relu'          : torch.nn.functional.relu,
            'relu6'         : torch.nn.functional.relu6,
            'rrelu'         : torch.nn.functional.rrelu,
            'selu'          : torch.nn.functional.selu,
            'celu'          : torch.nn.functional.celu,
            'sin'           : torch.sin,
            'cos'           : torch.cos,
            'gelu'          : torch.nn.functional.gelu,
            'sigmoid'       : torch.nn.functional.sigmoid,
            'silu'          : torch.nn.functional.silu,
            'mish'          : torch.nn.functional.mish,
            'softplus'      : torch.nn.functional.softplus,
            'softshrink'    : torch.nn.functional.softshrink,
            'tanh'          : torch.nn.functional.tanh,
            'tanhshrink'    : torch.nn.functional.tanhshrink};




# -------------------------------------------------------------------------------------------------
# MLP class
# -------------------------------------------------------------------------------------------------

class MultiLayerPerceptron(torch.nn.Module):
    def __init__(   self, 
                    widths          : list[int],
                    activations     : list[str],
                    reshape_index   : int           = 1, 
                    reshape_shape   : list[int]     = []) -> None:
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

        activations : list[str], len = len(widths) - 2
            A list of strings whose i'th element specifies the activation function we want to use 
            after the i'th layer's linear transformation. The final layer has no activation 
            function. 

        reshape_index : int, optional
            This argument specifies if we should reshape the network's input or output 
            (or neither). If the user specifies reshape_index, then it must be either 0 or -1. 
            Further, in this case, they must also specify reshape_shape (you need to specify both 
            together). If it is 0, then reshape_shape specifies how we reshape the input before 
            passing it through the network (the input to the first layer). If reshape_index is -1, 
            then reshape_shape specifies how we reshape the network output before returning it 
            (the output to the last layer). 

        reshape_shape : list[int], optional
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
        assert isinstance(reshape_shape, list),             "type(reshape_shape) == %s, expected list" % (str(type(reshape_shape)));
        for i in range(len(reshape_shape)):
            assert isinstance(reshape_shape[i], int),           "type(reshape_shape[%d]) = %s, expected int" % (i, str(type(reshape_shape[i])));
            assert reshape_shape[i] > 0,                        "reshape_shape[%d] = %d, needs to be positive" % (i, reshape_shape[i]);
        assert isinstance(widths, list),                    "type(widths) = %s, expected list" % str(type(widths));
        for i in range(len(widths)):
            assert isinstance(widths[i], int),                  "type(widths[%d]) = %s, must be int" % (i, str(type(widths[i])));
            assert widths[i] > 0,                               "widths[%d] = %d, must be positive" % (i, widths[i]);
        assert isinstance(activations, list),               "type(activations) = %s, expected list" % str(type(activations));
        assert len(activations) == len(widths) - 2,         "len(activations) = %d, len(widths) = %d; but we must have len(activations) = len(widths) - 2" % (len(activations), len(widths));
        for i in range(len(activations)):
            assert isinstance(activations[i], str),             "type(activations[%d]) = %s, must be str" % (i, str(type(activations[i])));
            assert activations[i].lower() in act_dict.keys(),   "activations[%d] = %s; not in act_dict keys" % (i, activations[i].lower());
        assert isinstance(reshape_shape, list),             "type(reshape_shape) = %s, must be list" % str(type(reshape_shape)); 
        assert isinstance(reshape_index, int),              "type(reshape_index) = %s, must be int" % str(type(reshape_index));
        if(len(reshape_shape) != 0):
            assert reshape_index in [0, -1],                    "reshape_index = %d; must be 0 or -1" % reshape_index;
            assert numpy.prod(reshape_shape) == widths[reshape_index],  "numpy.prod(reshape_shape) = %d, reshape_index = %d, widths[%d] = %d; must be equal" % (numpy.prod(reshape_shape), reshape_index, reshape_index, widths[reshape_index]);

        super().__init__();

        # Note that width specifies the dimensionality of the domains and co-domains of each layer.
        # Specifically, the i'th element specifies the input dimension of the i'th layer, while 
        # the final element specifies the dimensionality of the co-domain of the final layer. Thus, 
        # the number of layers is one less than the length of widths.
        self.n_layers       : int                   = len(widths) - 1;
        self.widths         : list[int]             = widths;
        self.activations    : list[str]             = activations;

        # Set up the affine parts of the layers.
        self.layers = [];
        for k in range(self.n_layers):
            self.layers += [torch.nn.Linear(widths[k], widths[k + 1])];
        self.layers = torch.nn.ModuleList(self.layers);

        # Now, initialize the weight matrices and bias vectors in the affine portion of each layer.
        self.init_weight();

        # Reshape input to the 1st layer or output of the last layer.
        self.reshape_index : int        = reshape_index;
        self.reshape_shape : list[int]  = reshape_shape;

        # Set up the activation functions
        self.activation_fns : list[Callable] = [];
        for i in range(self.n_layers - 1):
            self.activation_fns.append(act_dict[self.activations[i].lower()]);

        LOGGER.info("Initializing a MultiLayerPerceptron with widths %s, activations %s, reshape_shape = %s (index %d)" \
                    % (str(self.widths), str(self.activations), str(self.reshape_shape), self.reshape_index));

        # All done!
        return;
    


    def forward(self, U : torch.Tensor) -> torch.Tensor:
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

        U_Pred : torch.Tensor, shape = X.Shape
            The image of X under the network's layers. If self.reshape_index == -1 and 
            self.reshape_shape has k elements, then we reshape the output so that the final k 
            elements of its shape match those of self.reshape_shape.
        """

        # If the reshape_index is 0, we need to reshape X before passing it through the first 
        # layer.
        if ((len(self.reshape_shape) > 0) and (self.reshape_index == 0)):
            # Make sure the input has a proper shape. There is a lot going on in this line; let's
            # break it down. If reshape_index == 0, then we need to reshape the input, X, before
            # passing it through the layers. Let's assume that reshape_shape has k elements. Then,
            # we need to squeeze the final k dimensions of the input, X, so that the resulting 
            # tensor has a final dimension size that matches the input dimension size for the first
            # layer. The check below makes sure that the final k dimensions of the input, X, match
            # the stored reshape_shape.
            assert(list(U.shape[-len(self.reshape_shape):]) == self.reshape_shape);
            
            # Now that we know the final k dimensions of X have the correct shape, let's squeeze 
            # them into 1 dimension (so that we can pass the squeezed tensor through the first 
            # layer). To do this, we reshape X by keeping all but the last k dimensions of X, and 
            # replacing the last k with a single dimension whose size matches the dimensionality of
            # the domain of the first layer. 
            U = U.reshape(list(U.shape[:-len(self.reshape_shape)]) + [self.widths[self.reshape_index]]);

        # Pass X through the network layers; note that the final layer has no activation function, 
        # so we don't apply an activation function to it.
        for i in range(self.n_layers - 1):
            U = self.activation_fns[i](self.layers[i](U));   # apply linear layer
        U = self.layers[-1](U);                              # apply final layer (no activation)

        # If the reshape_index is -1, then we need to reshape the output before returning. 
        if ((len(self.reshape_shape) > 0) and (self.reshape_index == -1)):
            # In this case, we need to split the last dimension of X, the output of the final
            # layer, to match the reshape_shape. This is precisely what the line below does. Note
            # that we use torch.Tensor.view instead of torch.Tensor.reshape in order to avoid data 
            # copying. 
            U = U.view(list(U.shape[:-1]) + self.reshape_shape);

        # All done!
        return U;
    


    def init_weight(self) -> None:
        """
        This function initializes the weight matrices and bias vectors in self's layers. It takes 
        no arguments and returns nothing!
        """

        # TODO(kevin): support other initializations?
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(layer.weight);
            torch.nn.init.zeros_(layer.bias);
        
        # All done!
        return