import torch
import torch.nn as nn

# Todo: can be conditioned  by args
BatchNorm = nn.Identity  # nn.BatchNorm2d

##########################################################


class FullyConnected(nn.Module):
    """Dense or Fully Connected Layer followed by ReLU"""

    def __init__(self, n_outputs, withrelu=True, **kwargs):
        super(FullyConnected, self).__init__()
        self.withrelu = withrelu
        self.linear = nn.LazyLinear(n_outputs, bias=True)
        # xavier init for the weights
        # nn.init.xavier_uniform_(self.linear.weight)
        ##        nn.init.xavier_normal_(self.linear.weight)
        # constant init for the biais with cte=0.1
        # nn.init.constant_(self.linear.bias, 0.1)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        if self.withrelu:
            x = self.activ(x)
        return x


##########################################################
class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()

        self.pool_window = args.pool_window
        self.num_blocks = args.num_blocks
        ########## Encoder ##########
        self.encoder = nn.ModuleDict([])
        for b in range(self.num_blocks):
            self.encoder[str(b)] = self.init_encoder_block(b, args)
        # classifier
        # nb LazyLinear: in_features  inferred
        self.fc0 = FullyConnected(n_outputs=args.n_bins, withrelu=False)

    def num_flat_features(self, x):
        """
        Parameters
        ----------
        x: the input

        Returns
        -------
        the totale number of features = number of elements of the tensor except the batch dimension

        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        pool = nn.AvgPool2d(
            kernel_size=self.pool_window,
            stride=2,
            padding=int((self.pool_window - 1) / 2),
        )
        ########## Encoder ##########
        unpooled = []
        # print("Input:",x.shape,x.numel())
        for b in range(self.num_blocks):
            x_unpooled = self.encoder[str(b)](x)
            x = pool(x_unpooled)
            # print(f"Encoder {b}:",x.shape,x.numel())

        flat = x.view(-1, self.num_flat_features(x))
        # print("flat shape: ", flat.size())

        x = self.fc0(flat)
        # print('fc0 shape: ', x.size())
        return x

    def init_encoder_block(self, b, args):
        enc_layers = nn.ModuleList([])
        if b == 0:
            enc_layers.append(
                nn.Conv2d(
                    args.num_channels,
                    args.num_kernels,
                    args.kernel_size,
                    padding=args.padding,
                    bias=args.bias,
                )
            )
            enc_layers.append(nn.ReLU(inplace=True))
            for l in range(1, args.num_enc_conv):
                enc_layers.append(
                    nn.Conv2d(
                        args.num_kernels,
                        args.num_kernels,
                        args.kernel_size,
                        padding=args.padding,
                        bias=args.bias,
                    )
                )
                enc_layers.append(BatchNorm(args.num_kernels))
                enc_layers.append(nn.ReLU(inplace=True))
        else:
            for l in range(args.num_enc_conv):
                if l == 0:
                    enc_layers.append(
                        nn.Conv2d(
                            args.num_kernels * (2 ** (b - 1)),
                            args.num_kernels * (2**b),
                            args.kernel_size,
                            padding=args.padding,
                            bias=args.bias,
                        )
                    )
                else:
                    enc_layers.append(
                        nn.Conv2d(
                            args.num_kernels * (2**b),
                            args.num_kernels * (2**b),
                            args.kernel_size,
                            padding=args.padding,
                            bias=args.bias,
                        )
                    )
                enc_layers.append(BatchNorm(args.num_kernels * (2**b)))
                enc_layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*enc_layers)


##########################################################


class PzConv2d(nn.Module):
    """Convolution 2D Layer followed by PReLU activation"""

    def __init__(self, n_in_channels, n_out_channels, **kwargs):
        super(PzConv2d, self).__init__()
        self.conv = nn.Conv2d(n_in_channels, n_out_channels, bias=True, **kwargs)
        ## JEC 11/9/19 use default init :
        ##   kaiming_uniform_ for weights
        ##   bias uniform
        # xavier init for the weights
        ## nn.init.xavier_normal_(self.conv.weight)
        nn.init.xavier_uniform_(self.conv.weight)
        ## constant init for the biais with cte=0.1
        nn.init.constant_(self.conv.bias, 0.1)
        self.activ = nn.PReLU(num_parameters=n_out_channels, init=0.25)

    ##        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        return self.activ(x)


class PzPool2d(nn.Module):
    """Average Pooling Layer"""

    def __init__(self, kernel_size, stride, padding=0):
        super(PzPool2d, self).__init__()
        self.pool = nn.AvgPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=True,
            count_include_pad=False,
        )

    def forward(self, x):
        return self.pool(x)


class PzFullyConnected(nn.Module):
    """Dense or Fully Connected Layer followed by ReLU"""

    def __init__(self, n_outputs, withrelu=True, **kwargs):
        super(PzFullyConnected, self).__init__()
        self.withrelu = withrelu
        self.linear = nn.LazyLinear(n_outputs, bias=True)
        # init
        # nn.init.xavier_uniform_(self.linear.weight)
        # nn.init.constant_(self.linear.bias, 0.1)
        # non-lin
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        if self.withrelu:
            x = self.activ(x)
        return x


class PzInception(nn.Module):
    """Inspection module

    The input (x) is dispatched between

    o a cascade of conv layers s1_0 1x1 , s2_0 3x3
    o a cascade of conv layer s1_2 1x1, followed by pooling layer pool0 2x2
    o a cascade of conv layer s2_2 1x1
    o optionally a cascade of conv layers s1_1 1x1, s2_1 5x5

    then the 3 (or 4) intermediate outputs are concatenated
    """

    def __init__(
        self,
        n_in_channels,
        n_out_channels_1,
        n_out_channels_2,
        without_kernel_5=False,
        debug=False,
    ):
        super(PzInception, self).__init__()
        self.debug = debug
        self.s1_0 = PzConv2d(n_in_channels, n_out_channels_1, kernel_size=1, padding=0)
        self.s2_0 = PzConv2d(
            n_out_channels_1, n_out_channels_2, kernel_size=3, padding=1
        )

        self.s1_2 = PzConv2d(n_in_channels, n_out_channels_1, kernel_size=1)
        self.pad0 = nn.ZeroPad2d([0, 1, 0, 1])
        self.pool0 = PzPool2d(kernel_size=2, stride=1, padding=0)

        self.without_kernel_5 = without_kernel_5
        if not (without_kernel_5):
            self.s1_1 = PzConv2d(
                n_in_channels, n_out_channels_1, kernel_size=1, padding=0
            )
            self.s2_1 = PzConv2d(
                n_out_channels_1, n_out_channels_2, kernel_size=5, padding=2
            )

        self.s2_2 = PzConv2d(n_in_channels, n_out_channels_2, kernel_size=1, padding=0)

    def forward(self, x):
        # x:image tenseur N_batch, Channels, Height, Width
        x_s1_0 = self.s1_0(x)
        x_s2_0 = self.s2_0(x_s1_0)

        x_s1_2 = self.s1_2(x)

        x_pool0 = self.pool0(self.pad0(x_s1_2))

        if not (self.without_kernel_5):
            x_s1_1 = self.s1_1(x)
            x_s2_1 = self.s2_1(x_s1_1)

        x_s2_2 = self.s2_2(x)

        if self.debug:
            print("Inception x_s1_0  :", x_s1_0.size())
        if self.debug:
            print("Inception x_s2_0  :", x_s2_0.size())
        if self.debug:
            print("Inception x_s1_2  :", x_s1_2.size())
        if self.debug:
            print("Inception x_pool0 :", x_pool0.size())

        if not (self.without_kernel_5) and self.debug:
            print("Inception x_s1_1  :", x_s1_1.size())
            print("Inception x_s2_1  :", x_s2_1.size())

        if self.debug:
            print("Inception x_s2_2  :", x_s2_2.size())

        # to be check: dim=1=> NCHW (en TensorFlow axis=3 NHWC)
        if not (self.without_kernel_5):
            output = torch.cat((x_s2_2, x_s2_1, x_s2_0, x_pool0), dim=1)
        else:
            output = torch.cat((x_s2_2, x_s2_0, x_pool0), dim=1)

        if self.debug:
            print("Inception output :", output.shape)
        return output


class NetWithInception(nn.Module):
    """The Networks"""

    def __init__(self, args):
        super(NetWithInception, self).__init__()

        self.n_bins = args.n_bins  # output # bins
        n_input_channels = args.num_channels  # input image #chan (eg. 1:gray scale)

        self.conv0 = PzConv2d(
            n_in_channels=n_input_channels, n_out_channels=64, kernel_size=5, padding=2
        )
        self.pool0 = PzPool2d(kernel_size=2, stride=2, padding=0)
        self.i0 = PzInception(
            n_in_channels=64, n_out_channels_1=24, n_out_channels_2=32
        )

        self.i1 = PzInception(
            n_in_channels=120, n_out_channels_1=32, n_out_channels_2=48
        )

        self.i2 = PzInception(
            n_in_channels=176, n_out_channels_1=48, n_out_channels_2=64
        )

        self.i3 = PzInception(
            n_in_channels=240, n_out_channels_1=64, n_out_channels_2=92
        )

        self.i4 = PzInception(
            n_in_channels=340, n_out_channels_1=76, n_out_channels_2=108
        )

        self.i5 = PzInception(
            n_in_channels=400, n_out_channels_1=76, n_out_channels_2=108
        )

        self.i6 = PzInception(
            n_in_channels=400, n_out_channels_1=76, n_out_channels_2=108
        )

        self.i7 = PzInception(
            n_in_channels=400, n_out_channels_1=76, n_out_channels_2=108
        )

        self.iLast = PzInception(
            n_in_channels=400,
            n_out_channels_1=76,
            n_out_channels_2=108,
            without_kernel_5=True,
        )

        # classifier
        # nb LazyLinear: in_features  inferred
        self.fc0 = PzFullyConnected(n_outputs=self.n_bins, withrelu=False)

    def num_flat_features(self, x):
        """

        Parameters
        ----------
        x: the input

        Returns
        -------
        the totale number of features = number of elements of the tensor except the batch dimension

        """
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        # x:image tenseur N_batch, Channels, Height, Width
        #    size N, Channles=5 filtres, H,W = 64 pixels
        # save original image
        x_in = x

        x = self.conv0(x)
        x = self.pool0(x)
        x = self.i0(x)

        x = self.i1(x)
        x = self.pool0(x)

        x = self.i2(x)

        x = self.i3(x)
        x = self.pool0(x)

        x = self.i4(x)

        x = self.i5(x)
        x = self.pool0(x)

        x = self.i6(x)

        x = self.i7(x)
        x = self.pool0(x)

        x = self.iLast(x)

        flat = x.view(-1, self.num_flat_features(x))

        x = self.fc0(flat)

        return x
