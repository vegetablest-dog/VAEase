from models import *

# Hyper parameters
case = "VAEp"
date = datetime.date.today().strftime("%m%d")
os.makedirs(date, exist_ok=True)
ds = "fmnist"
bs = 2048
latent_dim = 32
lr = 0.05
base_channels = 16
nepoch = 300
loggamma_coef = -7
t0 = 20
save_path = f"{case}/{ds}/{latent_dim}"
os.makedirs(save_path, exist_ok=True)
train = False
seed = 114514
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda:0")
dat, test_dat, loader, test_label = load_data(bs, device, ds)
input_dim = dat.size(1) * dat.size(2)
c_dim = 0
if train:
    encoder = Res_Encoder(latent_dim, c_dim, EncoderBlock, base_channels).to(device)
    generator = Res_Decoder(latent_dim, c_dim, DecoderBlock, base_channels).to(device)
    loggamma = nn.Parameter(loggamma_coef*torch.ones(1, device=device))
    opt = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters())+[loggamma], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=t0,T_mult=2,eta_min=1e-6)

    t1 = time.time()
    for epoch in range(nepoch):
        for batch, data in enumerate(loader):
            x = data[0].to(device)
            mean, logvar = encoder(x)
            postz = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean, device=device)
            kl = torch.sum(torch.exp(logvar) + torch.square(mean) - logvar, dim=1) - latent_dim

            std = torch.exp(0.5 *logvar)
            xhat = generator((1-std)*postz)
            recon1 = torch.sum(torch.square(x.view(-1,784) - xhat), dim=1) / torch.exp(loggamma)
            recon2 = input_dim * loggamma + math.log(2 * math.pi) * input_dim
            loss = torch.mean(recon1 + recon2 + kl, dim=0)
            gamma = torch.exp(loggamma)

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
        print("Current Time:",time.ctime())
        print(f"epoch = {epoch}, loss = {loss.item()}, kl = {torch.mean(kl)}, recon1 = {torch.mean(recon1*gamma)}, recon2 = {torch.mean(recon2)}, gamma= {gamma.data}")
    print(f"Training {nepoch} epochs uses {time.time() - t1} seconds.")


    ## Save models
    torch.save(encoder, f"{save_path}/encoder.pth")
    torch.save(generator, f"{save_path}/generator.pth")
    torch.save(loggamma, f"{save_path}/loggamma.pth")


encoder = torch.load(f"{save_path}/encoder.pth")
generator = torch.load(f"{save_path}/generator.pth")
loggamma = torch.load(f"{save_path}/loggamma.pth")

# View the latent dimensions pattern via encoder
test_samplesize = 500

x = test_dat[:test_samplesize].view(test_samplesize,1,28,28)


mean, logvar = encoder(x)
postz = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean, device=device)
kl = torch.sum(torch.exp(logvar) + torch.square(mean) - logvar, dim=1) - latent_dim

std = torch.exp(0.5 *logvar)
xhat = generator((1-std)*postz)
recon1 = torch.sum(torch.square(x.view(-1,784) - xhat), dim=1) / torch.exp(loggamma)
recon2 = input_dim * loggamma + math.log(2 * math.pi) * input_dim
loss = torch.mean(recon1 + recon2 + kl, dim=0)
gamma = torch.exp(loggamma)

print(f"nll={loss}, recon={recon1.mean()*gamma}, KL={kl.mean()}")
var = torch.exp(logvar)
print(var.mean(dim=0))
labels = test_label[:test_samplesize]
res = [var.mean(dim=0)]
ADs = []
for element in set(labels.tolist()):
    label_var = var[labels == element].mean(dim=0)
    res.append(label_var)
    print(f'manifold {element}:', label_var)
    ad = torch.sum(label_var<0.05)
    ADs.append(ad)
    print(f'manifold dim:', ad)
    print(f'manifold AD index:',torch.where(label_var<0.05)[0])
res = torch.stack(res, dim=0)
ADs = torch.stack(ADs, dim=0).double()
print(f'recon={recon1.mean()*gamma}')
print('AD:',ADs)
print('AAD:',torch.mean(ADs))
print('std AD:',torch.std(ADs))
ADs = torch.sum(var<0.05, dim=1).double()
print('AAD:',torch.mean(ADs))
print('std AD:',torch.std(ADs))

# use minima variance to devide
AD = []
for b in range(logvar.shape[0]):
    logvar_slice = logvar[b]
    # import pdb;pdb.set_trace()
    thr = find_optimal_threshold_torch(logvar_slice)
    ad = torch.sum(logvar_slice <= thr)
    AD.append(ad)
AD_list = torch.tensor(AD).double()
print('AAD:',torch.mean(AD_list))
print('std AD:',torch.std(AD_list))


