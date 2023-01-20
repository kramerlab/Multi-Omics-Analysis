import functools
import torch
from torch import nn


class VaeBasicModel(torch.nn.Module):
    """
    This is the basic VAE model class, called by all other VAE son classes.
    """

    def __init__(
        self,
        omics_dims,
        norm_type,
        leaky_slope,
        dropout_p,
        latent_space_dim,
        dim_1B,
        dim_1A,
        dim_1C,
    ):
        """
        Initialize the VAE basic class.
        """
        # input tensor
        super().__init__()

        # define the network
        self.netEmbed = define_VAE(
            omics_dims,
            norm_type,
            leaky_slope,
            dropout_p,
            latent_space_dim,
            dim_1B,
            dim_1A,
            dim_1C,
        )

    def forward(self, data_e, data_m, data_c):
        # Get the output tensor
        z, recon_omics, mean, log_var = self.netEmbed.forward(data_e, data_m, data_c)
        latent = mean.detach()
        return z, recon_omics, mean, log_var, latent


class FCBlock(nn.Module):
    """
    Linear => Norm1D => LeakyReLU
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        norm_layer=nn.BatchNorm1d,
        leaky_slope=0.2,
        dropout_p=0,
        activation=True,
        normalization=True,
        activation_name="LeakyReLU",
    ):
        """
        Construct a fully-connected block
        Parameters:
            input_dim (int)         -- the dimension of the input tensor
            output_dim (int)        -- the dimension of the output tensor
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            activation (bool)       -- need activation or not
            normalization (bool)    -- need normalization or not
            activation_name (str)   -- name of the activation function used in the FC block
        """
        super(FCBlock, self).__init__()
        # Linear
        self.fc_block = [nn.Linear(input_dim, output_dim)]
        # Norm
        if normalization:
            # FC block doesn't support InstanceNorm1d
            if (
                isinstance(norm_layer, functools.partial)
                and norm_layer.func == nn.InstanceNorm1d
            ):
                norm_layer = nn.BatchNorm1d
            self.fc_block.append(norm_layer(output_dim))
        # Dropout
        if 0 < dropout_p <= 1:
            self.fc_block.append(nn.Dropout(p=dropout_p))
        # LeakyReLU
        if activation:
            if activation_name.lower() == "leakyrelu":
                self.fc_block.append(
                    nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)
                )
            elif activation_name.lower() == "tanh":
                self.fc_block.append(nn.Tanh())
            else:
                raise NotImplementedError(
                    "Activation function [%s] is not implemented" % activation_name
                )

        self.fc_block = nn.Sequential(*self.fc_block)

    def forward(self, x):
        output = self.fc_block(x)
        return output


# FcVae
class FcVaeABC(nn.Module):
    """
    Defines a fully-connected variational autoencoder for multi-omics dataset
    """

    def __init__(
        self,
        omics_dims,
        norm_layer=nn.BatchNorm1d,
        leaky_slope=0.2,
        dropout_p=0,
        dim_1B=384,
        dim_1A=384,
        dim_1C=384,
        latent_dim=256,
    ):
        """
        Construct a fully-connected variational autoencoder
        Parameters:
            omics_dims (list)       -- the list of input omics dimensions
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            latent_dim (int)        -- the dimensionality of the latent space
        """

        super(FcVaeABC, self).__init__()
        self.A_dim = omics_dims[0]
        self.B_dim = omics_dims[1]
        self.C_dim = omics_dims[2]
        self.dim_1B = dim_1B
        self.dim_1A = dim_1A
        self.dim_1C = dim_1C

        # ENCODER
        # Layer 1
        self.encode_fc_1B = FCBlock(
            self.B_dim,
            dim_1B,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=dropout_p,
            activation=True,
        )
        self.encode_fc_1A = FCBlock(
            self.A_dim,
            dim_1A,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=dropout_p,
            activation=True,
        )
        self.encode_fc_1C = FCBlock(
            self.C_dim,
            dim_1C,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=dropout_p,
            activation=True,
        )
        # Layer 4
        self.encode_fc_mean = FCBlock(
            dim_1C + dim_1B + dim_1A,
            latent_dim,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=0,
            activation=False,
            normalization=False,
        )
        self.encode_fc_log_var = FCBlock(
            dim_1C + dim_1B + dim_1A,
            latent_dim,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=0,
            activation=False,
            normalization=False,
        )

        # DECODER
        # Layer 1
        self.decode_fc_z = FCBlock(
            latent_dim,
            dim_1C + dim_1B + dim_1A,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=dropout_p,
            activation=True,
        )

        # Layer 4
        self.decode_fc_4B = FCBlock(
            dim_1B,
            self.B_dim,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=0,
            activation=False,
            normalization=False,
        )
        self.decode_fc_4A = FCBlock(
            dim_1A,
            self.A_dim,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=0,
            activation=False,
            normalization=False,
        )
        self.decode_fc_4C = FCBlock(
            dim_1C,
            self.C_dim,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=0,
            activation=False,
            normalization=False,
        )

    def encode(self, data_e, data_m, data_c):
        level_2_A = self.encode_fc_1A(data_e)
        level_2_B = self.encode_fc_1B(data_m)
        level_2_C = self.encode_fc_1C(data_c)

        level_3 = torch.cat((level_2_B, level_2_A, level_2_C), 1)
        latent_mean = self.encode_fc_mean(level_3)
        latent_log_var = self.encode_fc_log_var(level_3)

        return latent_mean, latent_log_var

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mean)

    def decode(self, z):
        level_1 = self.decode_fc_z(z)

        level_2_B = level_1.narrow(1, 0, self.dim_1B)
        level_2_A = level_1.narrow(1, self.dim_1B, self.dim_1A)
        level_2_C = level_1.narrow(1, self.dim_1B + self.dim_1A, self.dim_1C)

        recon_B = self.decode_fc_4B(level_2_B)
        recon_A = self.decode_fc_4A(level_2_A)
        recon_C = self.decode_fc_4C(level_2_C)

        return [recon_A, recon_B, recon_C]

    def get_last_encode_layer(self):
        return self.encode_fc_mean

    def forward(self, data_e, data_m, data_c):
        mean, log_var = self.encode(data_e, data_m, data_c)
        z = self.reparameterize(mean, log_var)
        recon_x = self.decode(z)
        return z, recon_x, mean, log_var


class VaeClassifierModel(VaeBasicModel):
    """
    This class implements the VAE classifier model, using the VAE framework with the classification downstream task.
    """

    def __init__(
        self,
        omics_dims,
        dropout_p,
        latent_space_dim,
        dim_1B,
        dim_1A,
        dim_1C,
        class_dim_1,
        leaky_slope,
    ):
        """
        Initialize the VAE_classifier class.
        """
        VaeBasicModel.__init__(
            self,
            omics_dims,
            "batch",
            leaky_slope,
            dropout_p,
            latent_space_dim,
            dim_1B,
            dim_1A,
            dim_1C,
        )
        # specify the training losses you want to print out.

        # define the network
        self.netDown = define_down(
            "batch", leaky_slope, dropout_p, latent_space_dim, 1, class_dim_1
        )

    def classify(self, data_e, data_m, data_c):
        _, _, _, _, latent = VaeBasicModel.forward(
            self, data_e, data_m, data_c
        )
        # Get the output tensor
        y_out = self.netDown(latent)
        return y_out

    def encode(self, data_e, data_m, data_c):
        # Get the output tensor
        z, recon_omics, mean, log_var, _ = VaeBasicModel.forward(
            self, data_e, data_m, data_c
        )
        return z, recon_omics, mean, log_var

    def encode_and_classify(self, data_e, data_m, data_c):
        # Get the output tensor
        z, recon_omics, mean, log_var, latent = VaeBasicModel.forward(
            self, data_e, data_m, data_c
        )
        y_out = self.netDown(latent)
        return z, recon_omics, mean, log_var, y_out

    def forward(self, data_e, data_m , data_c):
        _, _, _, _, latent = VaeBasicModel.forward(
            self, data_e, data_m, data_c
        )
        # Get the output tensor
        return self.netDown(latent)


def define_down(
    norm_type="batch",
    leaky_slope=0.2,
    dropout_p=0,
    latent_dim=256,
    class_num=2,
    class_dim_1=128,
):
    """
    Create the downstream task network
    Parameters:
        norm_type (str)         -- the name of normalization layers used in the network, default: batch
        leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
        latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
        class_num (int)         -- the number of class
    Returns a downstream task network
    The default downstream task network is a multi-layer fully-connected classifier.
    The generator has been initialized by <init_net>.
    :param class_dim_2:
    :param class_dim_1:
    """

    net = None

    # get the normalization layer
    norm_layer = get_norm_layer(norm_type=norm_type)

    net = MultiFcClassifier(
        class_num, latent_dim, norm_layer, leaky_slope, dropout_p, class_dim_1
    )

    return net


def get_norm_layer(norm_type="batch"):
    """
    Return a normalization layer
    Parameters:
        norm_type (str) -- the type of normalization applied to the model, default to use batch normalization, options: [batch | instance | none ]
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm1d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm1d, affine=False, track_running_stats=False
        )
    else:
        raise NotImplementedError("normalization method [%s] is not found" % norm_type)
    return norm_layer


# Class for downstream task
class MultiFcClassifier(nn.Module):
    """
    Defines a multi-layer fully-connected classifier
    """

    def __init__(
        self,
        class_num=2,
        latent_dim=256,
        norm_layer=nn.BatchNorm1d,
        leaky_slope=0.2,
        dropout_p=0,
        class_dim_1=128,
    ):
        """
        Construct a multi-layer fully-connected classifier
        Parameters:
            class_num (int)         -- the number of class
            latent_dim (int)        -- the dimensionality of the latent space and the input layer of the classifier
            norm_layer              -- normalization layer
            leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
            dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
            layer_num (int)         -- the layer number of the classifier, >=3
        """
        super(MultiFcClassifier, self).__init__()

        self.input_fc = FCBlock(
            latent_dim,
            class_dim_1,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=dropout_p,
            activation=True,
        )

        # create a list to store fc blocks
        mul_fc_block = []
        self.mul_fc = nn.Sequential(*mul_fc_block)

        # the output fully-connected layer of the classifier
        self.output_fc = FCBlock(
            class_dim_1,
            class_num,
            norm_layer=norm_layer,
            leaky_slope=leaky_slope,
            dropout_p=0,
            activation=False,
            normalization=False,
        )

    def forward(self, x):
        x1 = self.input_fc(x)
        x2 = self.mul_fc(x1)
        y = self.output_fc(x2)
        return y


def define_VAE(
    omics_dims,
    norm_type="batch",
    leaky_slope=0.2,
    dropout_p=0,
    latent_dim=256,
    dim_1B=384,
    dim_1A=384,
    dim_1C=384,
):
    """
    Create the VAE network
    Parameters:
        omics_dims (list)       -- the list of input omics dimensions
        norm_type (str)         -- the name of normalization layers used in the network, default: batch
        leaky_slope (float)     -- the negative slope of the Leaky ReLU activation function
        dropout_p (float)       -- probability of an element to be zeroed in a dropout layer
        latent_dim (int)        -- the dimensionality of the latent space
    Returns a VAE
    The default backbone of the VAE is one dimensional convolutional layer.
    The generator has been initialized by <init_net>.
    """

    net = None
    # get the normalization layer
    norm_layer = get_norm_layer(norm_type=norm_type)
    net = FcVaeABC(
        omics_dims,
        norm_layer,
        leaky_slope,
        dropout_p,
        dim_1B=dim_1B,
        dim_1A=dim_1A,
        dim_1C=dim_1C,
        latent_dim=latent_dim,
    )
    return net
