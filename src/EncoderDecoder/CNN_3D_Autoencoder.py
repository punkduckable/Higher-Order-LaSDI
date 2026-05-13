# -------------------------------------------------------------------------------------------------
# Imports and Setup
# -------------------------------------------------------------------------------------------------

# Add the Physics directory to the search path.
import  sys;
import  os;
src_Path        : str   = os.path.abspath(os.path.dirname(os.path.dirname(__file__)));
Physics_Path    : str   = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, "Physics"));
sys.path.append(Physics_Path);

import  logging;
from    typing          import  Callable, Sequence;

import  torch;
import  numpy;

from    typing          import  TYPE_CHECKING;
if TYPE_CHECKING:
    from    Trainer     import  Trainer;
    from    Physics     import  Physics;


from    EncoderDecoder  import  EncoderDecoder;
from    MLP             import  MultiLayerPerceptron, act_dict;
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

class CNN_3D_Autoencoder(EncoderDecoder):
    def __init__(   self,
                    Frame_Shape         : list[int],
                    config              : dict) -> None:
        r"""
        Initializes a convolutional autoencoder for 3D spatial data. This model applies a stack
        of 3D convolutions to a 3D image, flattens the resulting feature map, and then applies a
        fully-connected encoder to reach a low-dimensional latent space. The decoder mirrors this
        pipeline with fully-connected layers followed by 3D transpose-convolutions.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        Frame_Shape : list[int], len = 3
            The shape of elements of the FOM space. Specifies the spatial shape (C, I, J, K) of 
            each input frame. Inputs to Encode/forward can either  have shape 
            (n_Frames, C, I, J, K) or (n_Frames, C, I*J*K), where C = conv_channels[0]. In the 
            latter case, we reshape the input to have shape (n_Frames, C, I, J, K).

            
        config: dict
            The "EncoderDecoder" sub dictionary of the configuration file. It must contain a "type"
            key whose value is either "cnn_3d", "cnn_3d_ae", or "cnn_3d_autoencoder". There must 
            also be an item whose key matches the value of the "type" key and whose value is a 
            dictionary specifying the settings to define the CNN_3D_Autoencoder object. Namely, 
            it must contain the following keys:


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

        # Input checks.
        assert 'type' in config;
        assert (config['type'] == "cnn_3d") or (config['type'] == "cnn_3d_ae") or (config['type'] == "cnn_3d_autoencoder");
        cnn_key  : str = config['type'];
        assert cnn_key in config;
        cnn_config              : dict              = config[cnn_key];



        # FC configuration (analogous to the AE's hidden_widths/activations).
        hidden_widths_fc        : list[int]         = cnn_config.get('hidden_widths_fc', cnn_config.get('hidden_widths'));
        latent_dimension        : int               = cnn_config['latent_dimension'];

        # Checks: FC params.
        assert isinstance(hidden_widths_fc, list),              "type(hidden_widths_fc) = %s, expected list" % str(type(hidden_widths_fc));
        for i in range(len(hidden_widths_fc)):
            assert isinstance(hidden_widths_fc[i], int),            "type(hidden_widths_fc[%d]) = %s, must be int" % (i, str(type(hidden_widths_fc[i])));
            assert hidden_widths_fc[i] > 0,                         "hidden_widths_fc[%d] = %d, must be positive" % (i, hidden_widths_fc[i]);
        assert isinstance(latent_dimension, int),               "type(latent_dimension) = %s, must be int" % str(type(latent_dimension));
        assert latent_dimension > 0,                            "latent_dimension = %d, must be positive" % latent_dimension;



        # FC activations can either be a string or a list of strings.
        n_hidden_layers         : int               = len(hidden_widths_fc);
        act_cfg = cnn_config.get('activations_fc', cnn_config.get('activations'));
        if(isinstance(act_cfg, str)):
            activations_fc      : list[str]        = [act_cfg] * n_hidden_layers;
        elif(isinstance(act_cfg, list)):
            activations_fc      : list[str]        = act_cfg;
            assert(len(activations_fc) == n_hidden_layers);
        else:
            raise ValueError("activations_fc must be a string or a list of strings.");

        # Checks: activations_fc
        assert len(activations_fc) == len(hidden_widths_fc), \
            "len(activations_fc) = %d, len(hidden_widths_fc) = %d; must match" % (len(activations_fc), len(hidden_widths_fc));
        for i in range(len(activations_fc)):
            assert isinstance(activations_fc[i], str),             "type(activations_fc[%d]) = %s, must be str" % (i, str(type(activations_fc[i])));
            assert activations_fc[i].lower() in act_dict.keys(),   "activations_fc[%d] = %s; not in act_dict keys" % (i, activations_fc[i].lower());



        # Conv configuration.
        conv_channels       : list[int]     = cnn_config['conv_channels'];
        conv_kernel_sizes                   = cnn_config.get('conv_kernel_sizes', 3);
        conv_strides                        = cnn_config.get('conv_strides', 2);
        conv_paddings                       = cnn_config.get('conv_paddings', 1);

        # Checks: conv params.
        assert isinstance(conv_channels, list),                 "type(conv_channels) = %s, expected list" % str(type(conv_channels));
        assert len(conv_channels) >= 2,                         "len(conv_channels) = %d; must be at least 2 (input + output channels)" % len(conv_channels);
        for i in range(len(conv_channels)):
            assert isinstance(conv_channels[i], int),               "type(conv_channels[%d]) = %s, must be int" % (i, str(type(conv_channels[i])));
            assert conv_channels[i] > 0,                            "conv_channels[%d] = %d, must be positive" % (i, conv_channels[i]);



        # Per-layer conv activations. This can be a string (use same activation for all conv layers)
        # or a list of strings of length len(conv_channels) - 1.
        conv_act_cfg = cnn_config.get('conv_activations', 'relu');
        if(isinstance(conv_act_cfg, str)):
            conv_activations : list[str] = [conv_act_cfg] * (len(conv_channels) - 1);
        elif(isinstance(conv_act_cfg, list)):
            conv_activations = conv_act_cfg;
            assert(len(conv_activations) == len(conv_channels) - 1);
        else:
            raise ValueError("conv_activations must be a string or a list of strings.");

        # Checks: conv_activations
        assert isinstance(conv_activations, list),                "type(conv_activations) = %s, must be list" % str(type(conv_activations));
        n_conv_layers : int = len(conv_channels) - 1;
        assert len(conv_activations) == n_conv_layers, \
            "len(conv_activations) = %d, n_conv_layers = %d; must match" % (len(conv_activations), n_conv_layers);
        for i in range(len(conv_activations)):
            assert isinstance(conv_activations[i], str),            "type(conv_activations[%d]) = %s; must be str" % (i, str(type(conv_activations[i])));
            assert conv_activations[i].lower() in act_dict.keys(),  "conv_activations[%d] = %s; not in act_dict keys" % (i, conv_activations[i].lower());



        # Fetch Frame_Shape from physics (must be 3D for Conv3d).
        assert(len(Frame_Shape) == 4), "physics.Frame_Shape = %s; Conv_Autoencoder requires a 3D spatial shape" % str(Frame_Shape);
        C               : int       = int(Frame_Shape[0]);
        reshape_shape   : list[int] = [int(x) for x in Frame_Shape[1:]];
        
        # Checks: Frame_Shape.
        assert conv_channels[0] == C, "conv_chanels[0] = %d, but the data has %d channels. These must match" % (conv_channels[0], C);
        assert len(reshape_shape) == 3,                             "len(reshape_shape) = %d, expected 3" % len(reshape_shape);
        for i in range(len(reshape_shape)):
            assert isinstance(reshape_shape[i], int),               "type(reshape_shape[%d]) = %s, expected int" % (i, str(type(reshape_shape[i])));
            assert reshape_shape[i] > 0,                            "reshape_shape[%d] = %d, needs to be positive" % (i, reshape_shape[i]);
        


        # Extract n_Decoders
        n_Decoders = config[cnn_key]['n_Decoders'];

        # Run the superclass initializer.
        super().__init__(n_IC   = 1, n_z = latent_dimension, n_Decoders = n_Decoders, config = config);

        # Store information (for return purposes).
        self.reshape_shape      : list[int]     = reshape_shape;
        self.Frame_Shape        : list[int]     = Frame_Shape;

        self.conv_channels      : list[int]     = conv_channels;
        self.conv_activations   : list[str]     = conv_activations;
        self.n_conv_layers      : int           = n_conv_layers;

        self.hidden_widths_fc   : list[int]     = hidden_widths_fc;
        self.activations_fc     : list[str]     = activations_fc

        # Expand conv hyperparameters.
        self.conv_kernel_sizes  : list[tuple[int, int, int]] = _expand_3tuple_param(conv_kernel_sizes, self.n_conv_layers, "conv_kernel_sizes");
        self.conv_strides       : list[tuple[int, int, int]] = _expand_3tuple_param(conv_strides,      self.n_conv_layers, "conv_strides");
        self.conv_paddings      : list[tuple[int, int, int]] = _expand_3tuple_param(conv_paddings,     self.n_conv_layers, "conv_paddings");

        LOGGER.info("Initializing a CNN_3D_Autoencoder with latent space dimension %d" % self.n_z);
        LOGGER.info("  Frame shape:         %s" % str(self.Frame_Shape));
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

        self.fc_decoders = torch.nn.ModuleList([]);
        for i in range(self.n_Decoders):
            self.fc_decoders.append(MultiLayerPerceptron(
                                widths              = widths_fc_decoder,
                                activations         = self.activations_fc[::-1],
                                reshape_index       = 1,
                                reshape_shape       = []));

        # Build decoder conv stack (transpose convs), mirroring encoder.
        self.decoder_convs      = torch.nn.ModuleList([]);          # shape (n_conv_layers, n_Decoders)
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

            ith_decoder_convs = torch.nn.ModuleList([]);
            for j in range(self.n_Decoders):
                ith_decoder_convs.append(torch.nn.ConvTranspose3d(
                                            in_channels     = in_c,
                                            out_channels    = out_c,
                                            kernel_size     = k,
                                            stride          = st,
                                            padding         = p,
                                            output_padding  = output_padding));
            self.decoder_convs.append(ith_decoder_convs);

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



    def Eval_Decoder(self, i_Decoder : int, Z : torch.Tensor) -> tuple[torch.Tensor]:
        """
        This function decodes a set of latent frames.


        -------------------------------------------------------------------------------------------
        Arguments
        -------------------------------------------------------------------------------------------

        i_Decoder : int
            the index of the decoder we want to use to decode Z.

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
        assert len(Z.shape) == 2,           "Z.shape = %s, must have length 2" % str(Z.shape);
        assert Z.shape[1] == self.n_z,      "Z.shape[1] = %d, self.n_z = %d; must match" % (Z.shape[1], self.n_z);
        assert (i_Decoder >= 0) and (i_Decoder < self.n_Decoders),  "i_Decoder must be in {0, ... , %d}, got %d" % (self.n_Decoders - 1, i_Decoder);

        # FC decoder.
        U : torch.Tensor = self.fc_decoders[i_Decoder](Z);
        U = U.view((U.shape[0], self._conv_latent_channels) + self._conv_latent_shape);

        # Conv transpose decoder.
        for i in range(self.n_conv_layers):
            U = self.decoder_convs[i][i_Decoder](self._decoder_conv_act_fns[i](U));

        assert list(U.shape[-3:]) == self.reshape_shape, "Decoded output shape mismatch: got %s, expected (n_Frames, %s)" % (str(U.shape), str(self.reshape_shape));
        assert U.shape[1] == self.conv_channels[0], "Decoded channel mismatch: got %d, expected %d" % (U.shape[1], self.conv_channels[0]);
        return (U,);



    def export(self) -> dict:
        """
        This function extracts everything we need to recreate self from scratch.
        """
        decoders_list = [];
        for i in range(self.n_Decoders):
            decoders_list.append(self.fc_decoders[i].cpu().state_dict());
        
        dict_ = {   'EncoderDecoder dict' : super().export(),
                    'encoder conv state'  : self.encoder_convs.cpu().state_dict(),
                    'decoder conv state'  : self.decoder_convs.cpu().state_dict(),
                    'encoder fc state'    : self.encoder_fc.cpu().state_dict(),
                    'fc decoders state'   : decoders_list,
                    'reshape_shape'       : self.reshape_shape,
                    'latent_dimension'    : self.n_z,
                    'hidden_widths_fc'    : self.hidden_widths_fc,
                    'activations_fc'      : self.activations_fc,
                    'conv_channels'       : self.conv_channels,
                    'conv_kernel_sizes'   : self.conv_kernel_sizes,
                    'conv_strides'        : self.conv_strides,
                    'conv_paddings'       : self.conv_paddings,
                    'conv_activations'    : self.conv_activations,
                    'Frame_Shape'         : self.Frame_Shape,
                    'config'              : self.config};
        return dict_;




def load_CNN_3D_Autoencoder(dict_ : dict) -> CNN_3D_Autoencoder:
    """
    This function builds a CNN_3D_Autoencoder object using the information in dict_.
    """

    LOGGER.info("De-serializing a CNN_3D_Autoencoder..." );

    # Build the model.
    CNN = CNN_3D_Autoencoder( Frame_Shape     = dict_['Frame_Shape'],
                                config          = dict_['config']);

    
    # Set the decoder weights/active.
    CNN.load(dict_ = dict_['EncoderDecoder dict']);

    # Load the state dictionaries
    CNN.encoder_convs.load_state_dict(dict_['encoder conv state']);
    CNN.encoder_fc.load_state_dict(dict_['encoder fc state']);
    CNN.decoder_convs.load_state_dict(dict_['decoder conv state'])

    for i in range(CNN.n_Decoders):
        CNN.fc_decoders[i].load_state_dict(dict_['fc decoders state'][i]);

    # All done!
    return CNN;
