# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the Physics directory to the search path.
import  sys;
import  os;
Physics_Path    : str  = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "Physics"));
sys.path.append(Physics_Path);

import  logging;
from    typing      import  Callable, Sequence;

import  torch;
import  numpy;

from    MLP         import  MultiLayerPerceptron, act_dict;
from    Physics     import  Physics;

# Set up logging.
LOGGER  : logging.Logger    = logging.getLogger(__name__);




# -------------------------------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------------------------------

def _as_3tuple(x : int | Sequence[int]) -> tuple[int, int, int]:
    """
    Converts x into a 3-tuple.
    """

    if(isinstance(x, int)):
        assert x > 0, "x = %d, must be positive" % x;
        return (x, x, x);

    # Sequence[int]
    assert len(x) == 3, "len(x) = %d, expected 3" % len(x);
    x0, x1, x2 = int(x[0]), int(x[1]), int(x[2]);
    assert x0 > 0 and x1 > 0 and x2 > 0, "x = %s, must be positive" % str(x);
    return (x0, x1, x2);



def _expand_3tuple_param(    x           : int | Sequence[int] | Sequence[Sequence[int]],
                            n_layers    : int,
                            name        : str) -> list[tuple[int, int, int]]:
    """
    Expands x into a length n_layers list of 3-tuples.
    """

    # Scalar case -> use same value for all layers.
    if(isinstance(x, int)):
        return [_as_3tuple(x)] * n_layers;

    # Sequence[int] of length 3 -> use same 3-tuple for all layers.
    if(len(x) == 3 and all(isinstance(x[i], int) for i in range(3))):
        return [_as_3tuple(x)] * n_layers;

    # Sequence[Sequence[int]] -> one entry per layer.
    assert len(x) == n_layers, "len(%s) = %d, n_layers = %d; must match" % (name, len(x), n_layers);
    out : list[tuple[int, int, int]] = [];
    for i in range(n_layers):
        out.append(_as_3tuple(x[i]));
    return out;



def _conv3d_out_shape(   in_shape    : tuple[int, int, int],
                        kernel      : tuple[int, int, int],
                        stride      : tuple[int, int, int],
                        padding     : tuple[int, int, int]) -> tuple[int, int, int]:
    """
    Computes the output spatial shape of a Conv3d layer (dilation=1).
    """

    out = [];
    for d in range(3):
        # PyTorch: floor((in + 2*pad - (kernel-1) - 1)/stride + 1)
        out_d = (in_shape[d] + 2*padding[d] - (kernel[d] - 1) - 1) // stride[d] + 1;
        out.append(int(out_d));
    return (out[0], out[1], out[2]);




# -------------------------------------------------------------------------------------------------
# CNN_3D_Autoencoder class
# -------------------------------------------------------------------------------------------------

class CNN_3D_Autoencoder(torch.nn.Module):
    def __init__(   self,
                    reshape_shape       : list[int],
                    hidden_widths_fc    : list[int],
                    activations_fc      : list[str],
                    latent_dimension    : int,
                    conv_channels       : list[int],
                    conv_activations    : list[str],
                    conv_kernel_sizes   : int | Sequence[int] | Sequence[Sequence[int]],
                    conv_strides        : int | Sequence[int] | Sequence[Sequence[int]]  = 2,
                    conv_paddings       : int | Sequence[int] | Sequence[Sequence[int]]  = 1) -> None:
        r"""
        Initializes a convolutional autoencoder for 3D spatial data. This model applies a stack
        of 3D convolutions to a 3D image, flattens the resulting feature map, and then applies a
        fully-connected encoder to reach a low-dimensional latent space. The decoder mirrors this
        pipeline with fully-connected layers followed by 3D transpose-convolutions.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        reshape_shape : list[int], len = 3
            Specifies the spatial shape (I, J, K) of each input frame. Inputs to Encode/forward
            can either  have shape (n_Frames, C, I, J, K) or (n_Frames, C, I*J*K), where 
            C = conv_channels[0]. In the latter case, we reshape the input to have shape 
            (n_Frames, C, I, J, K)

        hidden_widths_fc : list[int]
            A list of integers specifying the widths of the hidden fully-connected layers. The
            encoder's final fully-connected layer maps to the latent space of dimension
            latent_dimension. The decoder uses the reversed widths.

        activations_fc : list[str], len = len(hidden_widths_fc)
            Activation(s) for the hidden fully-connected layers. The i'th element holds the name 
            of the activation function we apply after the i'th fully-connected layer of the encoder.
            Note that we do not apply an activation function to the output of the final fully c
            connected layer. The decoder uses the reversed list.

        latent_dimension : int
            The dimension of the latent space.

        conv_channels : list[int]
            A list whose i'th element specifies the number of channels after the i-1'th convolution.
            Equivalently, conv_channels[i] is the number of input channels to the i'th convolutional
            layer. Thus, if this list has M entries then the conv encoder has M-1 layers mapping
            conv_channels[i] -> conv_channels[i+1]. The decoder mirrors this structure.

        conv_activations : list[str], len = len(conv_channels) - 1
            i'th element specifies the activation function we apply after the i'th convolutional 
            layer. The decoder uses the reversed list.

        conv_kernel_sizes, conv_strides, conv_paddings:
            Convolution hyperparameters. Each can be:
                - an int (used for all layers in all three dimensions),
                - a length-3 sequence (used for all layers),
                - a list of length len(conv_channels) - 1 of length-3 sequences (one per layer).

            Note: By default, conv_strides=2 (i.e., we downsample at every conv layer). This is
            the "reduction operation" used by default.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Nothing!
        """

        # Checks: reshape_shape.
        assert isinstance(reshape_shape, list),                 "type(reshape_shape) == %s, expected list" % (str(type(reshape_shape)));
        assert len(reshape_shape) == 3,                         "len(reshape_shape) = %d, expected 3" % len(reshape_shape);
        for i in range(len(reshape_shape)):
            assert isinstance(reshape_shape[i], int),               "type(reshape_shape[%d]) = %s, expected int" % (i, str(type(reshape_shape[i])));
            assert reshape_shape[i] > 0,                            "reshape_shape[%d] = %d, needs to be positive" % (i, reshape_shape[i]);

        # Checks: conv params.
        assert isinstance(conv_channels, list),                 "type(conv_channels) = %s, expected list" % str(type(conv_channels));
        assert len(conv_channels) >= 2,                         "len(conv_channels) = %d; must be at least 2 (input + output channels)" % len(conv_channels);
        for i in range(len(conv_channels)):
            assert isinstance(conv_channels[i], int),               "type(conv_channels[%d]) = %s, must be int" % (i, str(type(conv_channels[i])));
            assert conv_channels[i] > 0,                            "conv_channels[%d] = %d, must be positive" % (i, conv_channels[i]);

        # Checks: FC params.
        assert isinstance(hidden_widths_fc, list),              "type(hidden_widths_fc) = %s, expected list" % str(type(hidden_widths_fc));
        for i in range(len(hidden_widths_fc)):
            assert isinstance(hidden_widths_fc[i], int),            "type(hidden_widths_fc[%d]) = %s, must be int" % (i, str(type(hidden_widths_fc[i])));
            assert hidden_widths_fc[i] > 0,                         "hidden_widths_fc[%d] = %d, must be positive" % (i, hidden_widths_fc[i]);
        assert isinstance(latent_dimension, int),               "type(latent_dimension) = %s, must be int" % str(type(latent_dimension));
        assert latent_dimension > 0,                            "latent_dimension = %d, must be positive" % latent_dimension;

        # activations_fc
        assert isinstance(activations_fc, list),                "type(activations_fc) = %s, must be list" % str(type(activations_fc));
        assert len(activations_fc) == len(hidden_widths_fc), \
            "len(activations_fc) = %d, len(hidden_widths_fc) = %d; must match" % (len(activations_fc), len(hidden_widths_fc));
        for i in range(len(activations_fc)):
            assert isinstance(activations_fc[i], str),             "type(activations_fc[%d]) = %s, must be str" % (i, str(type(activations_fc[i])));
            assert activations_fc[i].lower() in act_dict.keys(),   "activations_fc[%d] = %s; not in act_dict keys" % (i, activations_fc[i].lower());

        # conv_activations
        assert isinstance(conv_activations, list),                "type(conv_activations) = %s, must be list" % str(type(conv_activations));
        n_conv_layers : int = len(conv_channels) - 1;
        assert len(conv_activations) == n_conv_layers, \
            "len(conv_activations) = %d, n_conv_layers = %d; must match" % (len(conv_activations), n_conv_layers);
        for i in range(len(conv_activations)):
            assert isinstance(conv_activations[i], str),            "type(conv_activations[%d]) = %s; must be str" % (i, str(type(conv_activations[i])));
            assert conv_activations[i].lower() in act_dict.keys(),  "conv_activations[%d] = %s; not in act_dict keys" % (i, conv_activations[i].lower());


        # Run the superclass initializer.
        super().__init__();

        # Store information (for return purposes).
        self.n_IC               : int           = 1;
        self.reshape_shape      : list[int]     = reshape_shape;

        self.conv_channels      : list[int]     = conv_channels;
        self.conv_activations   : list[str]     = conv_activations;
        self.n_conv_layers      : int           = n_conv_layers;

        self.hidden_widths_fc   : list[int]     = hidden_widths_fc;
        self.activations_fc     : list[str]     = activations_fc
        self.n_z                : int           = latent_dimension;

        # Expand conv hyperparameters.
        self.conv_kernel_sizes  : list[tuple[int, int, int]] = _expand_3tuple_param(conv_kernel_sizes, self.n_conv_layers, "conv_kernel_sizes");
        self.conv_strides       : list[tuple[int, int, int]] = _expand_3tuple_param(conv_strides,      self.n_conv_layers, "conv_strides");
        self.conv_paddings      : list[tuple[int, int, int]] = _expand_3tuple_param(conv_paddings,     self.n_conv_layers, "conv_paddings");

        LOGGER.info("Initializing a CNN_3D_Autoencoder with latent space dimension %d" % self.n_z);
        LOGGER.info("  Reshape shape:       %s" % str(self.reshape_shape));
        LOGGER.info("  Conv channels:       %s" % str(self.conv_channels));
        LOGGER.info("  Conv kernels:        %s" % str(self.conv_kernel_sizes));
        LOGGER.info("  Conv strides:        %s" % str(self.conv_strides));
        LOGGER.info("  Conv paddings:       %s" % str(self.conv_paddings));
        LOGGER.info("  Conv activations:    %s" % str(self.conv_activations));
        LOGGER.info("  Hidden widths (FC):  %s" % str(self.hidden_widths_fc));
        LOGGER.info("  Activations (FC):    %s" % str(self.activations_fc));

        # Build encoder conv stack and record spatial shapes.
        self.encoder_convs                                      = torch.nn.ModuleList([]);
        self._encoder_shapes    : list[tuple[int, int, int]]    = [tuple(self.reshape_shape)];  # spatial shape before each conv.
        for i in range(self.n_conv_layers):
            self.encoder_convs.append(torch.nn.Conv3d(
                                        in_channels     = self.conv_channels[i],
                                        out_channels    = self.conv_channels[i + 1],
                                        kernel_size     = self.conv_kernel_sizes[i],
                                        stride          = self.conv_strides[i],
                                        padding         = self.conv_paddings[i]));

            out_shape = _conv3d_out_shape(  in_shape    = self._encoder_shapes[i],
                                            kernel      = self.conv_kernel_sizes[i],
                                            stride      = self.conv_strides[i],
                                            padding     = self.conv_paddings[i]);
            assert out_shape[0] > 0 and out_shape[1] > 0 and out_shape[2] > 0, \
                "Conv layer %d produced invalid shape %s; check kernel/stride/padding vs reshape_shape" % (i, str(out_shape));
            self._encoder_shapes.append(out_shape);
            LOGGER.info("Conv Layer %d has an output feature map: (C = %3d, D,H,W = %s )" % (i, self.conv_channels[i + 1], str(out_shape)));

        # Feature map shape at the output of the conv encoder.
        self._conv_latent_shape     : tuple[int, int, int] = self._encoder_shapes[-1];
        self._conv_latent_channels  : int = self.conv_channels[-1];
        self._flatten_dim           : int = int(self._conv_latent_channels * numpy.prod(self._conv_latent_shape).item());
        LOGGER.info("Post-convolution flattened dimension is %d" % self._flatten_dim);

        # Build fully-connected encoder/decoder.
        widths_fc_encoder : list[int] = [self._flatten_dim] + self.hidden_widths_fc + [self.n_z];
        widths_fc_decoder : list[int] = [self.n_z] + self.hidden_widths_fc[::-1] + [self._flatten_dim];

        self.encoder_fc = MultiLayerPerceptron(
                            widths              = widths_fc_encoder,
                            activations         = self.activations_fc,
                            reshape_index       = 1,
                            reshape_shape       = []);

        self.decoder_fc = MultiLayerPerceptron(
                            widths              = widths_fc_decoder,
                            activations         = self.activations_fc[::-1],
                            reshape_index       = 1,
                            reshape_shape       = []);

        # Build decoder conv stack (transpose convs), mirroring encoder.
        self.decoder_convs      = torch.nn.ModuleList([]);
        self._output_paddings   : list[tuple[int, int, int]] = [];

        # Encoder shapes: s0 -> s1 -> ... -> sL, where L = n_conv_layers.
        # Decoder should map sL -> ... -> s0.
        for i in range(self.n_conv_layers - 1, -1, -1):
            # For encoder conv i: channels go c_i -> c_{i+1}, spatial goes s_i -> s_{i+1}.
            in_c    : int                   = self.conv_channels[i + 1];
            out_c   : int                   = self.conv_channels[i];
            s_in    : tuple[int, int, int]  = self._encoder_shapes[i + 1];   # current decoder input spatial
            s_tgt   : tuple[int, int, int]  = self._encoder_shapes[i];       # desired output spatial
            k       : tuple[int, int, int]  = self.conv_kernel_sizes[i];
            st      : tuple[int, int, int]  = self.conv_strides[i];
            p       : tuple[int, int, int]  = self.conv_paddings[i];

            # Compute output_padding per dimension to hit the exact target shape.
            op = [];
            for d in range(3):
                base = (s_in[d] - 1) * st[d] - 2 * p[d] + k[d];
                opd  = s_tgt[d] - base;
                assert opd >= 0 and opd < st[d], \
                    "Invalid output_padding[%d]=%d for layer %d (target=%d, base=%d, stride=%d)" % (d, opd, i, s_tgt[d], base, st[d]);
                op.append(int(opd));
            output_padding : tuple[int, int, int] = (op[0], op[1], op[2]);
            self._output_paddings.append(output_padding);

            self.decoder_convs.append(torch.nn.ConvTranspose3d(
                                        in_channels     = in_c,
                                        out_channels    = out_c,
                                        kernel_size     = k,
                                        stride          = st,
                                        padding         = p,
                                        output_padding  = output_padding));

        # Cache activation functions.
        self._encoder_conv_act_fns : list[Callable] = [];
        for i in range(self.n_conv_layers):
            self._encoder_conv_act_fns.append(act_dict[self.conv_activations[i].lower()]);

        self._decoder_conv_act_fns : list[Callable] = [];
        for i in range(self.n_conv_layers):
            self._decoder_conv_act_fns.append(act_dict[self.conv_activations[::-1][i].lower()]);

        # All done!
        return;



    def Encode(self, U : torch.Tensor) -> tuple[torch.Tensor]:
        """
        This function encodes a set of 3D frames.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        U : torch.Tensor, shape = (n_Frames, C, I, J, K) or (n_Frames, C, I*J*K)
            A tensor holding a batch of inputs.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Z : tuple[torch.Tensor], len = 1
            A single element tuple whose lone element is a tensor of shape = (n_Frames, self.n_z)
            holding the latent-space encodings of each frame.
        """

        # Ensure input is 5D.
        assert isinstance(U, torch.Tensor), "type(U) = %s, must be torch.Tensor" % str(type(U));

        # Allowed input shapes:
        #   (N, C, I, J, K)
        #   (N, C, I*J*K)                (flattened)
        if(len(U.shape) == 3):
            expected = int(numpy.prod(self.reshape_shape).item());
            assert U.shape[-1]  == expected,                "U.shape[-1] = %d, expected %d (=prod(reshape_shape))" % (U.shape[-1], expected);
            assert U.shape[1]   == self.conv_channels[0],   "U.shape[1] = %d, expected %d" % (U.shape[1], self.conv_channels[0]);
            U = U.view((U.shape[0], U.shape[1]) + tuple(self.reshape_shape));

        assert len(U.shape)         == 5,                       "U.shape = %s, expected 2D, 4D, or 5D tensor" % str(U.shape);
        assert list(U.shape[-3:])   == self.reshape_shape,      "U.shape[-3:] = %s, self.reshape_shape = %s" % (str(U.shape[-3:]), str(self.reshape_shape));
        assert U.shape[1]           == self.conv_channels[0],   "U.shape[1] = %d, conv_channels[0] = %d; must match" % (U.shape[1], self.conv_channels[0]);

        # Conv encoder.
        for i in range(self.n_conv_layers):
            U = self._encoder_conv_act_fns[i](self.encoder_convs[i](U));

        # Flatten and FC encoder.
        U = U.reshape((U.shape[0], self._flatten_dim));
        Z : torch.Tensor = self.encoder_fc(U);
        return (Z,);



    def Decode(self, Z : torch.Tensor) -> tuple[torch.Tensor]:
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

        R : tuple[torch.Tensor], len = 1
            A single element tuple whose lone element is a Tensor of shape = (n_Frames, C, I, J, K)
            holding the reconstructed 3D frames; C = self.conv_channels[0].
        """

        # Checks.
        assert isinstance(Z, torch.Tensor), "type(Z) = %s, must be torch.Tensor" % str(type(Z));
        assert len(Z.shape) == 2,          "Z.shape = %s, must have length 2" % str(Z.shape);
        assert Z.shape[1] == self.n_z,     "Z.shape[1] = %d, self.n_z = %d; must match" % (Z.shape[1], self.n_z);

        # FC decoder.
        U : torch.Tensor = self.decoder_fc(Z);
        U = U.view((U.shape[0], self._conv_latent_channels) + self._conv_latent_shape);

        # Conv transpose decoder.
        for i in range(self.n_conv_layers):
            U = self.decoder_convs[i](self._decoder_conv_act_fns[i](U));

        assert list(U.shape[-3:]) == self.reshape_shape, "Decoded output shape mismatch: got %s, expected (n_Frames, %s)" % (str(U.shape), str(self.reshape_shape));
        assert U.shape[1] == self.conv_channels[0], "Decoded channel mismatch: got %d, expected %d" % (U.shape[1], self.conv_channels[0]);
        return (U,);



    def forward(self, X : torch.Tensor) -> tuple[torch.Tensor]:
        """
        This function passes X through the encoder, producing a latent state, Z. It then passes 
        Z through the decoder; hopefully producing a vector that approximates X.
        

        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        U : torch.Tensor, shape = (n_Frames,) + self.reshape_shape
            A tensor holding a batch of inputs. We pass this tensor through the encoder + decoder 
            and then return the result.


        -------------------------------------------------------------------------------------------
        Returns
        -------------------------------------------------------------------------------------------

        Y : tuple[torch.Tensor], len = 1
            A single element tuple whose lone element is a torch.Tensor of shape = X.shape holding 
            the image of X under the encoder and decoder. 
        """

        Z : torch.Tensor        = self.Encode(X)[0];
        Y : tuple[torch.Tensor] = self.Decode(Z);
        return Y;



    def latent_initial_conditions(  self,
                                    param_grid     : numpy.ndarray,
                                    physics        : Physics,
                                    trainer        = None) -> list[list[numpy.ndarray]]:
        """
        This function maps a set of initial conditions for the FOM to initial conditions for the
        latent space dynamics. See Autoencoder.latent_initial_conditions.
        """

        # Checks.
        assert isinstance(param_grid, numpy.ndarray),   "type(param_grid) = %s, must be numpy.ndarray" % str(type(param_grid));
        assert len(param_grid.shape) == 2,              "param_grid.shape = %s, must have length 2" % str(param_grid.shape);
        assert physics.n_IC     == self.n_IC,           "physics.n_IC = %d, self.n_IC = %d; must be equal" % (physics.n_IC, self.n_IC);

        # Figure out how many combinations of parameter values there are.
        n_param     : int                       = param_grid.shape[0];
        Z0          : list[list[numpy.ndarray]] = [];
        LOGGER.debug("Encoding initial conditions for %d parameter values" % n_param);

        # Cycle through the parameters.
        for i in range(n_param):
            # Fetch the IC for the i'th set of parameters. Then map it to a tensor.
            u0_np   : numpy.ndarray = physics.initial_condition(param_grid[i])[0];
            u0      : torch.Tensor  = torch.Tensor(u0_np).reshape((1,) + u0_np.shape);

            # If the trainer uses normalization, normalize the IC before encoding.
            has_norm = (trainer is not None) and hasattr(trainer, "has_normalization") and trainer.has_normalization();
            if(has_norm):
                u0 = trainer.normalize_tensor(u0, 0);
            else:
                LOGGER.warning(f"  No normalization applied to IC for param {i}!");

            # Encode the IC, then map the encoding to a numpy array.
            z0      : numpy.ndarray = self.Encode(u0)[0].detach().numpy();

            # Append the new IC to the list of latent ICs
            Z0.append([z0]);

        # Return the list of latent ICs.
        return Z0;



    def export(self) -> dict:
        """
        This function extracts everything we need to recreate self from scratch.
        """

        dict_ = {   'encoder conv state'  : self.encoder_convs.cpu().state_dict(),
                    'decoder conv state'  : self.decoder_convs.cpu().state_dict(),
                    'encoder fc state'    : self.encoder_fc.cpu().state_dict(),
                    'decoder fc state'    : self.decoder_fc.cpu().state_dict(),
                    'reshape_shape'       : self.reshape_shape,
                    'latent_dimension'    : self.n_z,
                    'hidden_widths_fc'    : self.hidden_widths_fc,
                    'activations_fc'      : self.activations_fc,
                    'conv_channels'       : self.conv_channels,
                    'conv_kernel_sizes'   : self.conv_kernel_sizes,
                    'conv_strides'        : self.conv_strides,
                    'conv_paddings'       : self.conv_paddings,
                    'conv_activations'    : self.conv_activations};
        return dict_;




def load_CNN_3D_Autoencoder(dict_ : dict) -> CNN_3D_Autoencoder:
    """
    This function builds a CNN_3D_Autoencoder object using the information in dict_.
    """

    LOGGER.info("De-serializing a CNN_3D_Autoencoder..." );

    reshape_shape       : list[int]     = dict_['reshape_shape'];
    latent_dimension    : int           = dict_['latent_dimension'];
    hidden_widths_fc    : list[int]     = dict_['hidden_widths_fc'];
    activations_fc      : list[str]     = dict_['activations_fc'];

    conv_channels       : list[int]     = dict_['conv_channels'];
    conv_kernel_sizes                   = dict_['conv_kernel_sizes'];
    conv_strides                        = dict_['conv_strides'];
    conv_paddings                       = dict_['conv_paddings'];
    conv_activations    : list[str]     = dict_['conv_activations'];

    model = CNN_3D_Autoencoder( reshape_shape        = reshape_shape,
                                hidden_widths_fc     = hidden_widths_fc,
                                activations_fc       = activations_fc,
                                latent_dimension     = latent_dimension,
                                conv_channels        = conv_channels,
                                conv_kernel_sizes    = conv_kernel_sizes,
                                conv_strides         = conv_strides,
                                conv_paddings        = conv_paddings,
                                conv_activations     = conv_activations);

    model.encoder_convs.load_state_dict(dict_['encoder conv state']);
    model.decoder_convs.load_state_dict(dict_['decoder conv state']);
    model.encoder_fc.load_state_dict(dict_['encoder fc state']);
    model.decoder_fc.load_state_dict(dict_['decoder fc state']);

    return model;
