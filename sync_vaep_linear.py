import torch
from models import *
from sync_dataset import *


device = torch.device("cuda:0")
nepoch = 150
t0 = 10
seed = 114514
sample_size = 500000
bs = 1024
lr = 0.01
coef = -0.5
# coefs = [-20, -10, -5, 0, 5, 10, 20]
torch.manual_seed(seed)
mix_param = torch.tensor([0.2, 0.3, 0.5, 0.3, 0.2], device=device)
latent_dim = 20
c_dim = 0
d = 40
num_manifolds = 3
dim_per_manifold = 4
train = False
load_checkpoint = False
load_checkpoint_path = None
setting_dim = None


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
   'num_manifolds':num_manifolds,
   'dim_per_manifold':dim_per_manifold,
})

train_dataset_config = ml_collections.ConfigDict()
train_dataset_config.seed = seed
train_dataset_config.data = ml_collections.ConfigDict()
train_dataset_config.data.data_samples = sample_size
train_dataset_config.data.num_manifolds = num_manifolds
train_dataset_config.data.dim_per_manifold = dim_per_manifold
train_dataset_config.data.d = d
train_dataset_config.data.setting_dim = setting_dim

save_path = f"synthetic/composedlinear_vaep/{d}_{latent_dim}_{num_manifolds}_{dim_per_manifold}"
os.makedirs(save_path, exist_ok=True)

dataset_save_path = f"sync_dataset/composedlinear"
os.makedirs(dataset_save_path, exist_ok=True)
dataset_filepath = f"{dataset_save_path}/{d}_{num_manifolds}_{dim_per_manifold}_{sample_size}_dataset.pkl"

if os.path.exists(dataset_filepath):
    with open(dataset_filepath, 'rb') as f:
        dataset = pickle.load(f)
else:
    dataset = ComposedLinearDataset(train_dataset_config)
    with open(dataset_filepath, 'wb') as f:
        pickle.dump(dataset, f)

train_dataset,test_dataset = random_split(dataset,[int(len(dataset)*0.9),len(dataset)-int(len(dataset)*0.9)])
train_loader = DataLoader(train_dataset,batch_size = bs, shuffle = True)
print("Current Time:",time.ctime())
print(f'{d}_{num_manifolds}_{dim_per_manifold}_{sample_size} dimension Dataset preparation is completed.')

if train:
    encoder = Linear_Encoder(d, latent_dim).to(device)
    generator = Linear_Decoder(latent_dim, d).to(device)
    loggamma = nn.Parameter(coef*torch.ones(1, device=device))
    opt = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters())+[loggamma], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=t0,T_mult=2,eta_min=1e-6)

    t1 = time.time()
    start_epoch = 0
    
    if load_checkpoint:
        # load chackpoint to train
        checkpoint = torch.load(load_checkpoint_path)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        generator.load_state_dict(checkpoint['generator_state_dict'])
        loggamma = checkpoint['loggamma']
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch,nepoch):
        encoder.train()
        generator.train()
        for batch, data in enumerate(train_loader):
            x = data[0].to(device).view(-1,d)
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

        

        print("Current Time:",time.ctime())
        print(f"epoch = {epoch}, loss = {loss.item()}, kl = {torch.mean(kl)}, recon1 = {torch.mean(recon1*gamma)}, recon2 = {torch.mean(recon2)}, gamma= {gamma.data}")
    print(f"Training {nepoch} epochs uses {time.time() - t1} seconds.")


    
    ## Save models
    torch.save(encoder, f"{save_path}/encoder.pth")
    torch.save(generator, f"{save_path}/generator.pth")
    torch.save(loggamma, f"{save_path}/loggamma.pth")


encoder = torch.load(f"{save_path}/encoder.pth").to(device)
generator = torch.load(f"{save_path}/generator.pth").to(device)
loggamma = torch.load(f"{save_path}/loggamma.pth").to(device)

# View the latent dimensions pattern via encoder
test_loader = DataLoader(test_dataset, batch_size = bs, shuffle = False)
for data in test_loader:
    break


x = data[0].to(device).view(-1,d)
labels = data[1]

mean, logvar = encoder(x)
postz = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean, device=device)
kl = torch.sum(torch.exp(logvar) + torch.square(mean) - logvar, dim=1) - latent_dim

var = torch.exp(0.5 * logvar)
xhat = generator((1-var)*postz)
recon1 = torch.sum(torch.square(x - xhat), dim=1) / torch.exp(loggamma)
recon2 = d * loggamma + math.log(2 * math.pi) * d
loss = torch.mean(recon1 + recon2 + kl, dim=0)
gamma = torch.exp(loggamma)

print(f"nll={loss}, recon={(recon1.mean()*gamma).item()}, KL={kl.mean()}")
var = torch.exp(logvar)
print(var.mean(dim=0))
print(dataset.dim_index)
for element in set(labels.tolist()):
    print(f'manifold {element}:', var[labels == element].mean(dim=0))
    print(torch.sum(var[labels == element].mean(dim=0)<0.05))
    thr = find_optimal_threshold_torch(logvar[labels == element].mean(dim=0))
    ad_o = torch.sum(logvar[labels == element].mean(dim=0)<=thr)
    print(ad_o)
# sorted_var, indices = torch.sort(var, descending=True)
# print(sorted_var)
# print((var > 0.9 ).sum().item())
# print((var < 0.5 ).sum().item())
# print((var < 0.05 ).sum().item())
