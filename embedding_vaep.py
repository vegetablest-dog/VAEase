import torch
from models import *
from sync_dataset import *
from torch.serialization import add_safe_globals

device = torch.device("cuda:0")
nepoch = 150
t0 = 10
seed = 114514
sample_size = 200000
bs = 512
lr = 0.001
coef = -15
print(f"coef = {coef}")

# coefs = [-20, -10, -5, 0, 5, 10, 20]
torch.manual_seed(seed)
np.random.seed(seed)
mix_param = torch.tensor([0.2, 0.3, 0.5, 0.3, 0.2], device=device)
latent_dim = 512
c_dim = 0
d = 768

train = True
load_checkpoint = False



print("Current Time:",time.ctime())
print('config setting:')
print({
'nepoch' :nepoch,
'sample_size':sample_size,
'bs':bs,
'coef':coef,
'seed':seed,
'latent_dim':latent_dim,
'd':d,
'lr':lr,
})


data_path = 'embedding_data/train-embeddings.pt'
dataset = CustomDataset(data_path)

save_path = f"activation/embedding_vaep/{coef}"
os.makedirs(save_path, exist_ok=True)


train_loader = DataLoader(dataset, batch_size = bs, shuffle = True)
print("Current Time:",time.ctime())
print(f'Dataset preparation is completed.')

add_safe_globals([Simple_Linear_Encoder, Linear_Decoder])

if train:
    encoder = Simple_Linear_Encoder(d, latent_dim).to(device)
    generator = Linear_Decoder(latent_dim, d).to(device)
    loggamma = nn.Parameter(coef*torch.ones(1, device=device))
    opt = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters())+[loggamma], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=t0,T_mult=2,eta_min=1e-6)

    t1 = time.time()
    start_epoch = 0
    

    for epoch in range(start_epoch,nepoch):
        encoder.train()
        generator.train()
        for batch, data in enumerate(train_loader):
            x = data.to(device).float()
            mean, logvar = encoder(x)
            postz = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean, device=device)
            kl = torch.sum(torch.exp(logvar) + torch.square(mean) - logvar, dim=1) - latent_dim

            var = torch.exp(0.5 * logvar)
            xhat = generator((1-var)*postz)
            recon1 = torch.sum(torch.square(x - xhat), dim=1) / torch.exp(loggamma)
            recon2 = d* loggamma + math.log(2 * math.pi) *d
            loss = torch.mean(recon1 + recon2 + kl, dim=0)
            gamma = torch.exp(loggamma)

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

        

        # print("Current Time:",time.ctime())
        print(f"epoch = {epoch}, loss = {loss.item()}, kl = {torch.mean(kl)}, recon1 = {torch.mean(recon1*gamma)}, recon2 = {torch.mean(recon2)}, gamma= {gamma.data}")
    print(f"Training {nepoch} epochs uses {time.time() - t1} seconds.")


    
    ## Save models
    torch.save(encoder, f"{save_path}/encoder.pth")
    torch.save(generator, f"{save_path}/generator.pth")
    torch.save(loggamma, f"{save_path}/loggamma.pth")


encoder = torch.load(f"{save_path}/encoder.pth",weights_only = False).to(device)
generator = torch.load(f"{save_path}/generator.pth",weights_only = False).to(device)
loggamma = torch.load(f"{save_path}/loggamma.pth",weights_only = False).to(device)

# View the latent dimensions pattern via encoder
test_data_path = 'embedding_data/val-embeddings.pt'
test_dataset = CustomDataset(test_data_path)
test_loader = DataLoader(test_dataset, batch_size = bs, shuffle = False)
for data in test_loader:
    break


x = data.to(device).float()


mean, logvar = encoder(x)
postz = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean, device=device)
kl = torch.sum(torch.exp(logvar) + torch.square(mean) - logvar, dim=1) - latent_dim

var = torch.exp(0.5 * logvar)
xhat = generator((1-var)*postz)
recon1 = torch.sum(torch.square(x - xhat), dim=1) / torch.exp(loggamma)
recon2 = d * loggamma + math.log(2 * math.pi) * d
loss = torch.mean(recon1 + recon2 + kl, dim=0)
gamma = torch.exp(loggamma)

print(f"nll={loss}, recon={(recon1.mean()*gamma).item()}, KL={kl.mean()}, gamma={gamma.item()}")
var = torch.exp(logvar)
# print(var[0])
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
# sorted_var, indices = torch.sort(var, descending=True)
# print(sorted_var)
# print((var > 0.9 ).sum().item())
# print((var < 0.5 ).sum().item())
# print((var < 0.05 ).sum().item())
