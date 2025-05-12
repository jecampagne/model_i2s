import torch
import torch.nn as nn



################################################ network class #################################################

# Bias Free Batch Norm
class BF_batchNorm(nn.Module):
    def __init__(self, num_kernels):
        super(BF_batchNorm, self).__init__()
        self.register_buffer("running_sd", torch.ones(1,num_kernels,1,1))
        g = (torch.randn( (1,num_kernels,1,1) )*(2./9./64.)).clamp_(-0.025,0.025)
        self.gammas = nn.Parameter(g, requires_grad=True)

    def forward(self, x):
        training_mode = self.training       
        sd_x = torch.sqrt(x.var(dim=(0,2,3) ,keepdim = True, unbiased=False)+ 1e-05)
        if training_mode:
            x = x / sd_x.expand_as(x)
            with torch.no_grad():
                self.running_sd.copy_((1-.1) * self.running_sd.data + .1 * sd_x)

            x = x * self.gammas.expand_as(x)

        else:
            x = x / self.running_sd.expand_as(x)
            x = x * self.gammas.expand_as(x)

        return x



BatchNorm =  nn.BatchNorm2d  # nn.Identity BF_batchNorm


######################################################################################

class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()

        self.pool_window = args.pool_window
        self.num_blocks = args.num_blocks
        ########## Encoder ##########
        self.encoder = nn.ModuleDict([])
        for b in range(self.num_blocks):
            self.encoder[str(b)] = self.init_encoder_block(b, args)

        ########## Mid-layers ##########
        mid_block = nn.ModuleList([])
        for l in range(args.num_mid_conv):
            if l == 0:
                mid_block.append(
                    nn.Conv2d(
                        args.num_kernels * (2**b),
                        args.num_kernels * (2 ** (b + 1)),
                        args.kernel_size,
                        padding=args.padding,
                        bias=args.bias,
                    )
                )
            else:
                mid_block.append(
                    nn.Conv2d(
                        args.num_kernels * (2 ** (b + 1)),
                        args.num_kernels * (2 ** (b + 1)),
                        args.kernel_size,
                        padding=args.padding,
                        bias=args.bias,
                    )
                )
            mid_block.append(BatchNorm(args.num_kernels * (2 ** (b + 1))))
            mid_block.append(nn.ReLU(inplace=True))

        self.mid_block = nn.Sequential(*mid_block)

        ########## Decoder ##########
        self.decoder = nn.ModuleDict([])
        self.upsample = nn.ModuleDict([])
        for b in range(self.num_blocks - 1, -1, -1):
            self.upsample[str(b)], self.decoder[str(b)] = self.init_decoder_block(
                b, args
            )

    def forward(self, x):
        pool = nn.AvgPool2d(
            kernel_size=self.pool_window,
            stride=2,
            padding=int((self.pool_window - 1) / 2),
        )
        ########## Encoder ##########
        unpooled = []
        for b in range(self.num_blocks):
            x_unpooled = self.encoder[str(b)](x)
            x = pool(x_unpooled)
            unpooled.append(x_unpooled)

        ########## Mid-layers ##########
        x = self.mid_block(x)

        ########## Decoder ##########
        for b in range(self.num_blocks - 1, -1, -1):
            x = self.upsample[str(b)](x)
            x = torch.cat([x, unpooled[b]], dim=1)
            x = self.decoder[str(b)](x)

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

    def init_decoder_block(self, b, args):
        dec_layers = nn.ModuleList([])

        # initiate the last block:
        if b == 0:
            for l in range(args.num_dec_conv - 1):
                if l == 0:
                    upsample = nn.ConvTranspose2d(
                        args.num_kernels * 2,
                        args.num_kernels,
                        kernel_size=2,
                        stride=2,
                        bias=args.bias,
                    )
                    dec_layers.append(
                        nn.Conv2d(
                            args.num_kernels * 2,
                            args.num_kernels,
                            kernel_size=args.kernel_size,
                            padding=args.padding,
                            bias=args.bias,
                        )
                    )
                else:
                    dec_layers.append(
                        nn.Conv2d(
                            args.num_kernels,
                            args.num_kernels,
                            args.kernel_size,
                            padding=args.padding,
                            bias=args.bias,
                        )
                    )
                dec_layers.append(BatchNorm(args.num_kernels))
                dec_layers.append(nn.ReLU(inplace=True))

            dec_layers.append(
                nn.Conv2d(
                    args.num_kernels,
                    args.num_channels,
                    kernel_size=args.kernel_size,
                    padding=args.padding,
                    bias=args.bias,
                )
            )

        # other blocks
        else:
            for l in range(args.num_dec_conv):
                if l == 0:
                    upsample = nn.ConvTranspose2d(
                        args.num_kernels * (2 ** (b + 1)),
                        args.num_kernels * (2**b),
                        kernel_size=2,
                        stride=2,
                        bias=args.bias,
                    )
                    dec_layers.append(
                        nn.Conv2d(
                            args.num_kernels * (2 ** (b + 1)),
                            args.num_kernels * (2**b),
                            kernel_size=args.kernel_size,
                            padding=args.padding,
                            bias=args.bias,
                        )
                    )
                else:
                    dec_layers.append(
                        nn.Conv2d(
                            args.num_kernels * (2**b),
                            args.num_kernels * (2**b),
                            args.kernel_size,
                            padding=args.padding,
                            bias=args.bias,
                        )
                    )

                dec_layers.append(BatchNorm(args.num_kernels * (2**b)))
                dec_layers.append(nn.ReLU(inplace=True))
        return upsample, nn.Sequential(*dec_layers)


########################## Encoder part of Unet  ################################


class FullyConnected(nn.Module):
    """Dense or Fully Connected Layer followed by ReLU"""

    def __init__(self, n_outputs, withrelu=True, **kwargs):
        super(FullyConnected, self).__init__()
        self.withrelu = withrelu
        # self.linear = nn.Linear(32768, n_outputs, bias=True)
        self.linear = nn.LazyLinear(n_outputs, bias=True)
        # LazyLinear has a fixed way to initialize its params
        # xavier init for the weights
        ##nn.init.xavier_uniform_(self.linear.weight)
        ##        nn.init.xavier_normal_(self.linear.weight)
        # constant init for the biais with cte=0.1
        ##nn.init.constant_(self.linear.bias, 0.1)
        self.activ = nn.ReLU()
        print("verif: self.withrelu=", self.withrelu)

    def forward(self, x):
        x = self.linear(x)
        if self.withrelu:
            x = self.activ(x)
        return x


##########################################################
class UNet_Encoder(nn.Module):
    def __init__(self, args):
        super(UNet_Encoder, self).__init__()

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

        # Classifier
        flat = x.view(-1, self.num_flat_features(x))
        x = self.fc0(flat)
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
                if args.dropout:
                    enc_layers.append(nn.Dropout(p=args.dropout_p))

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


############################### Inception ###########################


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


##############################  ResNet ############################
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlockV1(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlockV1, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetV1(nn.Module):

    def __init__(
        self,
        block,
        layers,
        num_input_channels=1,
        num_classes=1,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNetV1, self).__init__()

        self.num_input_channels = num_input_channels
        self.n_bins = num_classes

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            self.num_input_channels,
            self.inplanes,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlockV1):  ##JEC
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, **kwargs):
    model = ResNetV1(block, layers, **kwargs)
    return model


def resnet18(**kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet("resnet18", BasicBlockV1, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet("resnet34", BasicBlockV1, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], **kwargs)
