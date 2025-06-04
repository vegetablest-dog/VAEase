from pyro.nn import DenseNN
from utils import *
from torch.serialization import add_safe_globals
def variance_scaling(scale, mode, distribution,
                     in_axis=1, out_axis=0,
                     dtype=torch.float32,
                     device='cpu'):
    """Ported from JAX. """

    def _compute_fans(shape, in_axis=1, out_axis=0):
        receptive_field_size = np.prod(shape) / shape[in_axis] / shape[out_axis]
        fan_in = shape[in_axis] * receptive_field_size
        fan_out = shape[out_axis] * receptive_field_size
        return fan_in, fan_out

    def init(shape, dtype=dtype, device=device):
        fan_in, fan_out = _compute_fans(shape, in_axis, out_axis)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(
                "invalid mode for variance scaling initializer: {}".format(mode))
        variance = scale / denominator
        if distribution == "normal":
            return torch.randn(*shape, dtype=dtype, device=device) * np.sqrt(variance)
        elif distribution == "uniform":
            return (torch.rand(*shape, dtype=dtype, device=device) * 2. - 1.) * np.sqrt(3 * variance)
        else:
            raise ValueError("invalid distribution for variance scaling initializer")

    return init

def default_init(scale=1.):
    return variance_scaling(scale if scale != 0 else 1e-10, 'fan_avg', 'uniform')

class NIN(nn.Module):
    def __init__(self, in_dim, num_units, init_scale=0.1):
        super().__init__()
        self.W = nn.Parameter(default_init(scale=init_scale)((in_dim, num_units)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_units), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        y = torch.tensordot(x, self.W, dims=1) + self.b
        return y.permute(0, 3, 1, 2)


class AttnBlock(nn.Module):
    """Channel-wise self-attention block."""
    def __init__(self, channels):
        super().__init__()
        self.GroupNorm_0 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.NIN_0 = NIN(channels, channels)
        self.NIN_1 = NIN(channels, channels)
        self.NIN_2 = NIN(channels, channels)
        self.NIN_3 = NIN(channels, channels, init_scale=0.)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.GroupNorm_0(x)
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        h = self.NIN_3(h)
        return x + h


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SE(nn.Module):
    def __init__(self, Cin, Cout):
        super(SE, self).__init__()
        num_hidden = max(Cout // 16, 4)
        self.se = nn.Sequential(nn.Linear(Cin, num_hidden), nn.ReLU(inplace=True),
                                nn.Linear(num_hidden, Cout), nn.Sigmoid())

    def forward(self, x):
        se = torch.mean(x, dim=[2, 3])
        se = se.view(se.size(0), -1)
        se = self.se(se)
        se = se.view(se.size(0), -1, 1, 1)
        return x * se


class EncoderBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels * EncoderBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * EncoderBlock.expansion)
        )

        if stride != 1 or in_channels != EncoderBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * EncoderBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * EncoderBlock.expansion)
            )
        else:
            self.shortcut = nn.Identity()

        self.activate = nn.PReLU()

    def forward(self, x):
        return self.activate(self.residual_function(x) + self.shortcut(x))

class EncoderCell(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels * expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * expansion),
            Swish(),
            nn.Conv2d(out_channels * expansion, out_channels * expansion, kernel_size=5, stride=stride, padding=2, groups=out_channels * expansion, bias=False),
            nn.Conv2d(out_channels * expansion, out_channels * expansion, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels * expansion),
            Swish(),
            nn.Conv2d(out_channels * expansion, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # SE(out_channels, out_channels),
        )

        if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            self.shortcut = nn.Identity()
        
        self.activate = Swish()
    
    def forward(self, x):
        return self.activate(self.shortcut(x) +self.net(x))

class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU()
        )

        #conv transpose
        self.convt = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.convt(self.residual_function(x) + x)

class DecoderCell(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,bn = True):
        super().__init__()
        if bn:
            self.net = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                Swish(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                Swish(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                # SE(out_channels, out_channels),
            )
        else: 
            self.net = nn.Sequential(
                Swish(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                Swish(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )


        if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        else:
            self.shortcut = nn.Identity()
        
        self.activate = Swish()
    
    def forward(self, x):
        return self.activate(self.shortcut(x) +self.net(x))

class Res_Encoder(nn.Module):
    def __init__(self, latent_dim, c_dim, block, in_channels):
        super(Res_Encoder, self).__init__()
        # Number of features
        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            block(in_channels, in_channels, 2),
            block(in_channels, in_channels*2, 2),
        )

        self.latent_dim = latent_dim
        if c_dim > 0:
            self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        self.z = DenseNN(in_channels*2 + c_dim//2, hidden_dims=[32], param_dims=[latent_dim, latent_dim], nonlinearity=nn.PReLU())

    def forward(self, x, c=None):
        x = self.net(x)
        x = torch.mean(x, dim=[2, 3])
        if c != None:
            c = self.cprocesser(c)
            mean, logvar = self.z(torch.cat([x, c], dim=1))
        else:
            mean, logvar = self.z(x)
        logvar = torch.clip(-logvar,max=0)
        return mean, logvar

class Res_Encoder_color(nn.Module):
    def __init__(self, latent_dim, c_dim, block, in_channels):
        super().__init__()
        # Number of features
        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            block(in_channels, in_channels, 2),
            block(in_channels, in_channels*2, 2),
        )

        self.latent_dim = latent_dim
        if c_dim > 0:
            self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        self.z = DenseNN(in_channels*2 + c_dim//2, hidden_dims=[32], param_dims=[latent_dim, latent_dim], nonlinearity=nn.PReLU())

    def forward(self, x, c=None):
        x = self.net(x)
        x = torch.mean(x, dim=[2, 3])
        if c != None:
            c = self.cprocesser(c)
            mean, logvar = self.z(torch.cat([x, c], dim=1))
        else:
            mean, logvar = self.z(x)
        logvar = torch.clip(-logvar,max=0)
        return mean, logvar

class Res_Encoder_d(nn.Module):
    def __init__(self, latent_dim, c_dim, block, in_channels):
        super().__init__()
        # Number of features
        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            block(in_channels, in_channels, 2),
            block(in_channels, in_channels*2, 2),
        )

        self.latent_dim = latent_dim
        if c_dim > 0:
            self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        self.z = DenseNN(in_channels*2+ c_dim//2, hidden_dims=[latent_dim*4], param_dims=[latent_dim, latent_dim], nonlinearity=nn.PReLU())

    def forward(self, x, c=None):
        x = self.net(x)
        x = torch.mean(x, dim=[2, 3])
        if c != None:
            c = self.cprocesser(c)
            mean, logvar = self.z(torch.cat([x, c], dim=1))
        else:
            mean, logvar = self.z(x)
        logvar = torch.clip(-logvar,max=0)
        return mean, logvar

class Res_Encoder_SAE(nn.Module):
    def __init__(self, latent_dim, c_dim, block, in_channels):
        super().__init__()
        # Number of features
        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            block(in_channels, in_channels, 2),
            block(in_channels, in_channels*2, 2),
        )

        self.latent_dim = latent_dim
        if c_dim > 0:
            self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        self.z = DenseNN(in_channels*2 + c_dim//2, hidden_dims=[32], param_dims=[latent_dim], nonlinearity=nn.PReLU())

    def forward(self, x, c=None):
        x = self.net(x)
        x = torch.mean(x, dim=[2, 3])
        if c != None:
            c = self.cprocesser(c)
            mean= self.z(torch.cat([x, c], dim=1))
        else:
            mean= self.z(x)
        return mean

class Res_Encoder_SAE_color(nn.Module):
    def __init__(self, latent_dim, c_dim, block, in_channels):
        super().__init__()
        # Number of features
        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.Conv2d(3, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            block(in_channels, in_channels, 2),
            block(in_channels, in_channels*2, 2),
        )

        self.latent_dim = latent_dim
        if c_dim > 0:
            self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        self.z = DenseNN(in_channels*2 + c_dim//2, hidden_dims=[32], param_dims=[latent_dim], nonlinearity=nn.PReLU())

    def forward(self, x, c=None):
        x = self.net(x)
        x = torch.mean(x, dim=[2, 3])
        if c != None:
            c = self.cprocesser(c)
            mean= self.z(torch.cat([x, c], dim=1))
        else:
            mean= self.z(x)
        return mean    

class Res_Encoder_SAE_d(nn.Module):
    def __init__(self, latent_dim, c_dim, block, in_channels):
        super().__init__()
        # Number of features
        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            block(in_channels, in_channels, 2),
            block(in_channels, in_channels*2, 2),
        )

        self.latent_dim = latent_dim
        if c_dim > 0:
            self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        self.z = DenseNN(in_channels*2 + c_dim//2, hidden_dims=[latent_dim*4], param_dims=[latent_dim], nonlinearity=nn.PReLU())

    def forward(self, x, c=None):
        x = self.net(x)
        x = torch.mean(x, dim=[2, 3])
        if c != None:
            c = self.cprocesser(c)
            mean= self.z(torch.cat([x, c], dim=1))
        else:
            mean= self.z(x)
        return mean

class Res_Encoder_Full(nn.Module):
    def __init__(self, latent_dim, c_dim, block, in_channels):
        super(Res_Encoder_Full, self).__init__()
        # Number of features
        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            block(in_channels, in_channels, 2),
            block(in_channels, in_channels*2, 2),
        )

        self.latent_dim = latent_dim
        self.tril_indices = torch.tril_indices(row=latent_dim, col=latent_dim, offset=0)
        if c_dim > 0:
            self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        self.z = DenseNN(in_channels*2 + c_dim//2, hidden_dims=[32], param_dims=[latent_dim, (latent_dim*(latent_dim+1))//2], nonlinearity=nn.PReLU())

    def forward(self, x, c=None, device=torch.device("cuda:1")):
        x = self.net(x.view(-1, 1, 28, 28))
        x = torch.mean(x, dim=[2, 3])
        l = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim), device=device)
        if c != None:
            c = self.cprocesser(c)
            mean, logvar = self.z(torch.cat([x, c], dim=1))
        else:
            l = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim), device=device)
            mean, logvar = self.z(x)
            l[:, self.tril_indices[0], self.tril_indices[1]] = logvar
        return mean, l

class Res_Decoder(nn.Module):
    def __init__(self, latent_dim, c_dim, block, in_channels=32, output_size = [28,28], atten=False):
        super(Res_Decoder, self).__init__()
        self.atten = atten
        self.in_channels = in_channels
        self.output_size = output_size

        if c_dim > 0:
            if atten:
                self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2, latent_dim], nonlinearity=nn.PReLU())
            else:
                self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        assert (output_size[0]%4==0) and (output_size[1]%4==0)
        self.upsample = nn.Linear(in_features=latent_dim+c_dim//2, out_features=(output_size[0]//4)*(output_size[1]//4)*in_channels)

        self.net = nn.Sequential(
            block(in_channels, in_channels),
            block(in_channels, in_channels),
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Flatten()
        )

    def forward(self, z, c=None):
        if c != None:
            if self.atten:
                c, weight = self.cprocesser(c)
                self.weight = torch.sigmoid_(weight)
                z = z * self.weight
            else:
                c = self.cprocesser(c)
            z = self.upsample(torch.cat([z, c], dim=1)).relu_().view(-1, self.in_channels, self.output_size[0]//4, self.output_size[1]//4)
        else:
            z = self.upsample(z).relu_().view(-1, self.in_channels, self.output_size[0]//4, self.output_size[1]//4)
        x = self.net(z)
        return torch.sigmoid_(x)

class Res_Decoder_color(nn.Module):
    def __init__(self, latent_dim, c_dim, block, in_channels=32, output_size = [32,32], atten=False):
        super().__init__()
        self.atten = atten
        self.in_channels = in_channels
        self.output_size = output_size

        if c_dim > 0:
            if atten:
                self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2, latent_dim], nonlinearity=nn.PReLU())
            else:
                self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        assert (output_size[0]%4==0) and (output_size[1]%4==0)
        self.upsample = nn.Linear(in_features=latent_dim+c_dim//2, out_features=(output_size[0]//4)*(output_size[1]//4)*in_channels)

        self.net = nn.Sequential(
            block(in_channels, in_channels),
            block(in_channels, in_channels),
            nn.Conv2d(in_channels, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.PReLU(),
            nn.Flatten()
        )

    def forward(self, z, c=None):
        if c != None:
            if self.atten:
                c, weight = self.cprocesser(c)
                self.weight = torch.sigmoid_(weight)
                z = z * self.weight
            else:
                c = self.cprocesser(c)
            z = self.upsample(torch.cat([z, c], dim=1)).relu_().view(-1, self.in_channels, self.output_size[0]//4, self.output_size[1]//4)
        else:
            z = self.upsample(z).relu_().view(-1, self.in_channels, self.output_size[0]//4, self.output_size[1]//4)
        x = self.net(z)
        return torch.sigmoid_(x)

class Wider_Res_Encoder(nn.Module):
    def __init__(self, latent_dim, c_dim, block, in_channels,densenn_hidden_dim = 32):
        super(Wider_Res_Encoder, self).__init__()
        # Number of features
        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            block(in_channels, in_channels, 1),
            block(in_channels, in_channels*2, 2),
        )

        self.latent_dim = latent_dim
        if c_dim > 0:
            self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        self.z = DenseNN(in_channels*2 + c_dim//2, hidden_dims=[densenn_hidden_dim], param_dims=[latent_dim, latent_dim], nonlinearity=nn.PReLU())

    def forward(self, x, c=None):
        x = self.net(x)
        x = torch.mean(x, dim=[2, 3])
        if c != None:
            c = self.cprocesser(c)
            mean, logvar = self.z(torch.cat([x, c], dim=1))
        else:
            mean, logvar = self.z(x)
        return mean, logvar

class Wider_Res_Decoder(nn.Module):
    def __init__(self, latent_dim, c_dim, block, in_channels=32, output_size = [28,28], atten=False):
        super(Wider_Res_Decoder, self).__init__()
        self.atten = atten
        self.in_channels = in_channels
        self.output_size = output_size

        if c_dim > 0:
            if atten:
                self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2, latent_dim], nonlinearity=nn.PReLU())
            else:
                self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        assert (output_size[0]%4==0) and (output_size[1]%4==0)
        self.upsample = nn.Linear(in_features=latent_dim+c_dim//2, out_features=(output_size[0]//2)*(output_size[1]//2)*in_channels*2)

        self.net = nn.Sequential(
            block(in_channels*2, in_channels),
            DecoderCell(in_channels, in_channels),
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Flatten()
        )

    def forward(self, z, c=None):
        if c != None:
            if self.atten:
                c, weight = self.cprocesser(c)
                self.weight = torch.sigmoid_(weight)
                z = z * self.weight
            else:
                c = self.cprocesser(c)
            z = self.upsample(torch.cat([z, c], dim=1)).relu_().view(-1, self.in_channels*2, self.output_size[0]//2, self.output_size[1]//2)
        else:
            z = self.upsample(z).relu_().view(-1, self.in_channels*2, self.output_size[0]//2, self.output_size[1]//2)
        x = self.net(z)
        return torch.sigmoid_(x)

class Deeper_Res_Encoder(nn.Module):
    def __init__(self, latent_dim, c_dim, block, in_channels,densenn_hidden_dim = 32):
        super(Deeper_Res_Encoder, self).__init__()
        # Number of features
        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            *[block(in_channels, in_channels, 1) for i in range(5)],
            block(in_channels, in_channels*2, 2),
            *[block(in_channels*2, in_channels*2, 1) for i in range(11)],
            block(in_channels*2, in_channels*4, 2),
            *[block(in_channels*4, in_channels*4, 1) for i in range(3)],
        )

        self.latent_dim = latent_dim
        if c_dim > 0:
            self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        self.z = DenseNN(in_channels*4 + c_dim//2, hidden_dims=[densenn_hidden_dim], param_dims=[latent_dim, latent_dim], nonlinearity=nn.PReLU())

    def forward(self, x, c=None):
        x = self.net(x)
        x = torch.mean(x, dim=[2, 3])
        if c != None:
            c = self.cprocesser(c)
            mean, logvar = self.z(torch.cat([x, c], dim=1))
        else:
            mean, logvar = self.z(x)
        logvar = torch.clip(-logvar,max=0)
        return mean, logvar

class Deeper_Res_Decoder(nn.Module):
    def __init__(self, latent_dim, c_dim, block, in_channels=32, output_size = [28,28], atten=False):
        super(Deeper_Res_Decoder, self).__init__()
        self.atten = atten
        self.in_channels = in_channels
        self.output_size = output_size

        if c_dim > 0:
            if atten:
                self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2, latent_dim], nonlinearity=nn.PReLU())
            else:
                self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        assert (output_size[0]%4==0) and (output_size[1]%4==0)
        self.upsample = nn.Linear(in_features=latent_dim+c_dim//2, out_features=(output_size[0]//4)*(output_size[1]//4)*in_channels*4)

        self.net = nn.Sequential(
            *[DecoderCell(in_channels*4, in_channels*4) for i in range(3)],
            block(in_channels*4, in_channels*2),
            *[DecoderCell(in_channels*2, in_channels*2) for i in range(11)],
            block(in_channels*2, in_channels),
            *[DecoderCell(in_channels, in_channels,bn = False) for i in range(5)],
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.PReLU(),
            nn.Flatten()
        )

    def forward(self, z, c=None):
        if c != None:
            if self.atten:
                c, weight = self.cprocesser(c)
                self.weight = torch.sigmoid_(weight)
                z = z * self.weight
            else:
                c = self.cprocesser(c)
            z = self.upsample(torch.cat([z, c], dim=1)).relu_().view(-1, self.in_channels*4, self.output_size[0]//4, self.output_size[1]//4)
        else:
            z = self.upsample(z).relu_().view(-1, self.in_channels*4, self.output_size[0]//4, self.output_size[1]//4)
        x = self.net(z)
        return torch.clip(x,0,1)

class Deeper_Res_Encoder_Full(nn.Module):
    def __init__(self, latent_dim, c_dim, block, in_channels,densenn_hidden_dim = 32):
        super(Deeper_Res_Encoder_Full, self).__init__()
        # Number of features
        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            block(in_channels, in_channels, 1),
            block(in_channels, in_channels, 2),
            block(in_channels, in_channels, 1),
            block(in_channels, in_channels*2, 2),
        )

        self.latent_dim = latent_dim
        self.tril_indices = torch.tril_indices(row=latent_dim, col=latent_dim, offset=0)
        if c_dim > 0:
            self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        self.z = DenseNN(in_channels*2 + c_dim//2, hidden_dims=[densenn_hidden_dim], param_dims=[latent_dim, (latent_dim*(latent_dim+1))//2], nonlinearity=nn.PReLU())

    def forward(self, x, device, c=None):
        x = self.net(x)
        x = torch.mean(x, dim=[2, 3])
        l = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim), device=device)
        if c != None:
            c = self.cprocesser(c)
            mean, logvar = self.z(torch.cat([x, c], dim=1))
        else:
            mean, logvar = self.z(x)
            l[:, self.tril_indices[0], self.tril_indices[1]] = logvar
        return mean, l

class EncoderResblock_GN_Dropout(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, dropout = 0.1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6),
            Swish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels , kernel_size=3, padding=1, bias=False),
        )
        #shortcut
        if stride != 1 or in_channels !=  out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels , kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6),
            )
        else:
            self.shortcut = nn.Identity()

        self.activate = Swish()

    def forward(self, x):
        return self.activate(self.residual_function(x) + self.shortcut(x))

class DncoderResblock_GN(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6),
            Swish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6),
            Swish(),
            nn.Conv2d(out_channels, out_channels , kernel_size=3, padding=1, bias=False),
        )
        #shortcut
        if stride != 1 or in_channels !=  out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels , kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(num_groups=32, num_channels=out_channels , eps=1e-6),
            )
        else:
            self.shortcut = nn.Identity()

        self.activate = Swish()

    def forward(self, x):
        return self.activate(self.residual_function(x) + self.shortcut(x))

class Unet_Encoder(nn.Module):
    def __init__(self, in_channels,latent_dim,img_size = 32, hidden_dim=256, c_dim=0):
        super().__init__()
        # Number of features
        self.in_channels = in_channels

        self.net = nn.Sequential(
            nn.Conv2d(1, in_channels, kernel_size=3, padding=1, bias=False),
            Swish(),
            *[EncoderResblock_GN_Dropout(in_channels, in_channels, 1) for i in range(3)],
            EncoderResblock_GN_Dropout(in_channels, in_channels*2, 2),
            *[EncoderResblock_GN_Dropout(in_channels*2, in_channels*2, 1) for i in range(3)],
            EncoderResblock_GN_Dropout(in_channels*2, in_channels*2, 2),
            *[EncoderResblock_GN_Dropout(in_channels*2, in_channels*2, 1) for i in range(3)],
            EncoderResblock_GN_Dropout(in_channels*2, in_channels*2, 2),
            *[EncoderResblock_GN_Dropout(in_channels*2, in_channels*2, 1) for i in range(2)],
        )

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.img_size = img_size
        if c_dim > 0:
            self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        self.z = DenseNN(in_channels*2*(img_size//8)*(img_size//8) + c_dim//2, hidden_dims=[hidden_dim], param_dims=[latent_dim, latent_dim], nonlinearity=nn.PReLU())

    def forward(self, x, c=None):
        x = self.net(x)
        x = x.view(-1, self.in_channels*2*(self.img_size//8)*(self.img_size//8))
        if c != None:
            c = self.cprocesser(c)
            mean, logvar = self.z(torch.cat([x, c], dim=1))
        else:
            mean, logvar = self.z(x)
        return mean, logvar

class Unet_Decoder(nn.Module):
    def __init__(self, in_channels, latent_dim, c_dim=0 , output_size = [28,28], atten=False):
        super().__init__()
        self.atten = atten
        self.in_channels = in_channels
        self.output_size = output_size

        if c_dim > 0:
            if atten:
                self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2, latent_dim], nonlinearity=nn.PReLU())
            else:
                self.cprocesser = DenseNN(c_dim, hidden_dims=[16], param_dims=[c_dim//2], nonlinearity=nn.PReLU())
        assert (output_size[0]%4==0) and (output_size[1]%4==0)
        self.upsample = nn.Linear(in_features=latent_dim+c_dim//2, out_features=(output_size[0]//8)*(output_size[1]//8)*in_channels*2)

        self.net = nn.Sequential(
            *[DncoderResblock_GN(in_channels*2, in_channels*2) for i in range(2)],
            DecoderBlock(in_channels*2, in_channels*2),
            *[DncoderResblock_GN(in_channels*2, in_channels*2) for i in range(3)],
            DecoderBlock(in_channels*2, in_channels*2),
            *[DncoderResblock_GN(in_channels*2, in_channels*2) for i in range(3)],
            DecoderBlock(in_channels*2, in_channels),
            *[DncoderResblock_GN(in_channels, in_channels) for i in range(3)],
            nn.Conv2d(in_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.PReLU(),
            nn.Flatten()
        )

    def forward(self, z, c=None):
        if c != None:
            if self.atten:
                c, weight = self.cprocesser(c)
                self.weight = torch.sigmoid_(weight)
                z = z * self.weight
            else:
                c = self.cprocesser(c)
            z = self.upsample(torch.cat([z, c], dim=1)).relu_().view(-1, self.in_channels*2, self.output_size[0]//8, self.output_size[1]//8)
        else:
            z = self.upsample(z).relu_().view(-1, self.in_channels*2, self.output_size[0]//8, self.output_size[1]//8)
        x = self.net(z)
        return torch.sigmoid(x)


class Linear_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Linear_Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim*2),
            Swish(),
            nn.Linear(latent_dim*2, latent_dim*2),
            Swish(),
            nn.Linear(latent_dim*2, latent_dim*2),
            Swish(),
            nn.Linear(latent_dim*2, latent_dim*2),
        )

    def forward(self, x):
        x = self.net(x)
        mean,logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clip(logvar,max=0)
        return mean,logvar

class Simple_Linear_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim*2),
            Swish(),
        )
        self.sigma_map = nn.Linear(latent_dim,latent_dim)

    def forward(self, x):
        x = self.net(x)
        mean,logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.log(torch.clip(F.sigmoid(self.sigma_map(logvar)),min=1e-7))
        return mean,logvar

class Simple_Linear_Sparse_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

    
class Sparse_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim*2),
            Swish(),
            nn.Linear(latent_dim*2, latent_dim*2),
            Swish(),
            nn.Linear(latent_dim*2, latent_dim*2),
            Swish(),
            nn.Linear(latent_dim*2, latent_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return x

class Linear_Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Linear_Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, output_dim,),
        )

    def forward(self, z):
        return self.net(z)

class Linear_Res(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Linear_Res, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, output_dim*2,bias=False),
            Swish(),
            nn.Linear(output_dim*2, output_dim,bias=False),
            Swish(),
        )
        if latent_dim==output_dim:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Linear(latent_dim, output_dim,bias=False)
       

    def forward(self, z):
        return self.net(z)+ self.shortcut(z)
    
class Linear_resDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            Linear_Res(latent_dim,latent_dim*2),
            nn.Linear(latent_dim*2, output_dim,bias=False),
        )

    def forward(self, z):
        return self.net(z)


class ResMLP(nn.Module):
    def __init__(self,in_dim,out_dim, extension = 4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim*extension),
            Swish(),
            nn.Linear(out_dim*extension, out_dim*extension),
            nn.BatchNorm1d(out_dim*extension),
            Swish(),
            nn.Linear(out_dim*extension, out_dim),
            nn.BatchNorm1d(out_dim),
            Swish(),
        )
        if in_dim==out_dim:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm(out_dim),
                Swish(),
            )

    def forward(self, x):
        return self.skip_connection(x)+self.net(x)

class MLP_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, clip_flag = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.clip_flag = clip_flag
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim*2),
            nn.BatchNorm1d(latent_dim*2),
            Swish(),
            ResMLP(latent_dim*2, latent_dim*2),
            ResMLP(latent_dim*2, latent_dim*2),
            ResMLP(latent_dim*2, latent_dim*2),
            nn.Linear(latent_dim*2, latent_dim*2),
        )

    def forward(self, x):
        x = self.net(x)
        mean,logvar = torch.chunk(x, 2, dim=1)
        if self.clip_flag:
            logvar = torch.clip(logvar,max=0)
        return mean,logvar

class MLP_Simple_Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, clip_flag = True):
        super().__init__()
        self.latent_dim = latent_dim
        self.clip_flag = clip_flag
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim*2),
            nn.BatchNorm1d(latent_dim*2),
            Swish(),
            ResMLP(latent_dim*2, latent_dim*2),
            nn.Linear(latent_dim*2, latent_dim*2),
        )

    def forward(self, x):
        x = self.net(x)
        mean,logvar = torch.chunk(x, 2, dim=1)
        if self.clip_flag:
            logvar = torch.clip(logvar,max=0)
        return mean,logvar

class MLP_Encoder_SAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim*2),
            nn.BatchNorm1d(latent_dim*2),
            Swish(),
            ResMLP(latent_dim*2, latent_dim*2),
            ResMLP(latent_dim*2, latent_dim*2),
            ResMLP(latent_dim*2, latent_dim*2),
            nn.Linear(latent_dim*2, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.net(x)
        return x

class MLP_Simple_Encoder_SAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim*2),
            nn.BatchNorm1d(latent_dim*2),
            Swish(),
            ResMLP(latent_dim*2, latent_dim*2),
            nn.Linear(latent_dim*2, latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.net(x)
        return x

class MLP_Decoder(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            Swish(),
            ResMLP(latent_dim*2, latent_dim*2),
            ResMLP(latent_dim*2, latent_dim*2),
            ResMLP(latent_dim*2, latent_dim*2),
            nn.Linear(latent_dim*2,out_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return x

class MLP_Simple_Decoder(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            Swish(),
            ResMLP(latent_dim*2, latent_dim*2),
            nn.Linear(latent_dim*2,out_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return x

class FCNDecoder(nn.Module):
    def __init__(self, latent_dim, out_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim, out_dim*2),
            nn.LeakyReLU(negative_slope = 0.2),
            # nn.Linear(out_dim*2, out_dim*2),
            # nn.LeakyReLU(),
            nn.Linear(out_dim*2,out_dim),
        )
    def forward(self, x):
        x = self.net(x)
        return x

class C_Processor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=[16]):
        super(C_Processor, self).__init__()
        self.net = DenseNN(input_dim, hidden_dims=hidden_dim,
                  param_dims=[output_dim], nonlinearity=nn.ReLU())

    def forward(self, x):
        return self.net(x)


class ParamPrior(nn.Module):
    def __init__(self, c_dim, latent_dim, hidden_dim=[16]):
        super(ParamPrior, self).__init__()

        self.c_processor = C_Processor(c_dim, latent_dim, hidden_dim)
        self.mu_net = nn.Linear(in_features=latent_dim, out_features=latent_dim)
        self.logvar_net = nn.Linear(in_features=latent_dim, out_features=latent_dim)

    def forward(self, c):
        c = self.c_processor(c)
        return self.mu_net(c), self.logvar_net(c)


class Syn_Encoder(nn.Module):
    def __init__(self, data_dim, c_dim, latent_dim, hidden=[16]):
        super(Syn_Encoder, self).__init__()


        self.c_processor = C_Processor(c_dim, latent_dim, hidden)
        self.net = DenseNN(latent_dim + data_dim, hidden_dims=hidden, param_dims=[latent_dim], nonlinearity=nn.ReLU())
        self.nc_net = DenseNN(data_dim, hidden_dims=hidden, param_dims=[latent_dim], nonlinearity=nn.ReLU())

        self.mu_c = nn.Linear(latent_dim, latent_dim)
        self.logvar_c = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, c):

        c = self.c_processor(c)
        z = self.net(torch.cat([x, c], dim=1))
        return self.mu_c(z), self.logvar_c(z)




class Syn_Generator(nn.Module):
    def __init__(self, data_dim, c_dim, latent_dim, hidden=[16]):
        super(Syn_Generator, self).__init__()

        self.c_processor = C_Processor(c_dim, latent_dim, hidden)
        self.net = DenseNN(input_dim=2 * latent_dim, hidden_dims=hidden, param_dims=[data_dim], nonlinearity=nn.ReLU())
        self.nc_net = DenseNN(input_dim=latent_dim, hidden_dims=hidden, param_dims=[data_dim], nonlinearity=nn.ReLU())

    def forward(self, z, c):

        c = self.c_processor(c)
        return self.net(torch.cat([z, c], dim=1))

from torchvision import transforms
IMAGE_SIZE = 150
celeb_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, antialias=True),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor()]) 

celeb_transform1 = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, antialias=True),
    transforms.CenterCrop(IMAGE_SIZE)])


class VAE_Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[32, 64, 128, 256, 512], latent_dim=128, image_size=IMAGE_SIZE):
        super(VAE_Encoder, self).__init__()
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.image_size = image_size

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        out = self.encoder(torch.rand(1, 3, image_size, image_size))
        self.size = out.shape[2]
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.size * self.size, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.size * self.size, latent_dim)

    def forward(self, x):
        result = self.encoder(x)
        # import pdb;pdb.set_trace()
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = -F.relu(self.fc_var(result))
        return mu, log_var

class SAE_Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[32, 64, 128, 256, 512], latent_dim=128, image_size=IMAGE_SIZE):
        super(SAE_Encoder, self).__init__()
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.image_size = image_size

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        out = self.encoder(torch.rand(1, 3, image_size, image_size))
        self.size = out.shape[2]
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.size * self.size, latent_dim)   

    def forward(self, x):
        result = self.encoder(x)
        # import pdb;pdb.set_trace()
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        return mu

class VAE_Decoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dims=[32, 64, 128, 256, 512], image_size=5):
        super(VAE_Decoder, self).__init__()
        self.hidden_dims = hidden_dims[:]
        self.latent_dim = latent_dim
        self.image_size = image_size

        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * image_size * image_size)
        hidden_dims.reverse()
        # import pdb; pdb.set_trace()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid())

    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.image_size, self.image_size)
        # import pdb; pdb.set_trace()
        result = self.decoder(result)
        result = self.final_layer(result)
        # result = celeb_transform1(result)
        result = torch.flatten(result, start_dim=1)
        result = torch.nan_to_num(result)
        return result


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, transpose=False):
        super(ResidualBlock, self).__init__()
        self.transpose = transpose

        if transpose:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if transpose:
                self.shortcut = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=2, output_padding=1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class VAE_Encoder_Res(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[32, 64, 128, 256], latent_dim=128, image_size=IMAGE_SIZE):
        super(VAE_Encoder_Res, self).__init__()
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.image_size = image_size

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                ResidualBlock(in_channels, h_dim)  # 使用残差块
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        out = self.encoder(torch.rand(1, 3, image_size, image_size))
        self.size = out.shape[2]
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.size * self.size, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.size * self.size, latent_dim)

    def forward(self, x):
        result = self.encoder(x)
        # import pdb; pdb.set_trace()
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = -F.relu(self.fc_var(result))
        return mu, log_var


class VAE_Decoder_Res(nn.Module):
    def __init__(self, latent_dim=128, hidden_dims=[32, 64, 128, 256], image_size=5):
        super(VAE_Decoder_Res, self).__init__()
        self.hidden_dims = hidden_dims[:]
        self.latent_dim = latent_dim
        self.image_size = image_size

        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * image_size * image_size)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i + 1], transpose=True)  # 使用残差块
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid())

    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[-1], self.image_size, self.image_size)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result


add_safe_globals([
    NIN,
    AttnBlock,
    SE,
    EncoderBlock,
    EncoderCell,
    DecoderBlock,
    DecoderCell,
    Res_Encoder,
    Res_Encoder_color,
    Res_Encoder_d,
    Res_Encoder_SAE,
    Res_Encoder_SAE_color,
    Res_Encoder_SAE_d,
    Res_Encoder_Full,
    Res_Decoder,
    Res_Decoder_color,
    Wider_Res_Encoder,
    Wider_Res_Decoder,
    Deeper_Res_Encoder,
    Deeper_Res_Decoder,
    Deeper_Res_Encoder_Full,
    EncoderResblock_GN_Dropout,
    DncoderResblock_GN,
    Unet_Encoder,
    Unet_Decoder,
    Linear_Encoder,
    Simple_Linear_Encoder,
    Simple_Linear_Sparse_Encoder,
    Sparse_Encoder,
    Linear_Decoder,
    Linear_Res,
    Linear_resDecoder,
    ResMLP,
    MLP_Encoder,
    MLP_Simple_Encoder,
    MLP_Encoder_SAE,
    MLP_Simple_Encoder_SAE,
    MLP_Decoder,
    MLP_Simple_Decoder,
    FCNDecoder,
    C_Processor,
    ParamPrior,
    Syn_Encoder,
    Syn_Generator,
    VAE_Encoder,
    SAE_Encoder,
    VAE_Decoder,

])
