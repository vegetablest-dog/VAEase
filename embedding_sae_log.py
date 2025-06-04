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



    # coefs = [-20, -10, -5, 0, 5, 10, 20]
torch.manual_seed(seed)
np.random.seed(seed)
mix_param = torch.tensor([0.2, 0.3, 0.5, 0.3, 0.2], device=device)
latent_dim = 512
c_dim = 0
d = 768
lambda1 = 0.0002
lambda2 = 0.0005
eps = 1e-3

train = True
load_checkpoint = False



print("Current Time:",time.ctime())
print('config setting:')
print({
'nepoch' :nepoch,
'sample_size':sample_size,
'bs':bs,
'seed':seed,
'latent_dim':latent_dim,
'd':d,
'lr':lr,
})


data_path = 'embedding_data/train-embeddings.pt'
dataset = CustomDataset(data_path)

save_path = f"activation/embedding_sae_log"
os.makedirs(save_path, exist_ok=True)


train_loader = DataLoader(dataset, batch_size = bs, shuffle = True)
print("Current Time:",time.ctime())
print(f'Dataset preparation is completed.')

add_safe_globals([Simple_Linear_Sparse_Encoder, Linear_Decoder])

if train:
    encoder = Simple_Linear_Sparse_Encoder(d, latent_dim).to(device)
    generator = Linear_Decoder(latent_dim, d).to(device)
    opt = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=t0,T_mult=2,eta_min=1e-6)

    t1 = time.time()
    start_epoch = 0
    

    for epoch in range(start_epoch,nepoch):
        encoder.train()
        generator.train()
        for batch, data in enumerate(train_loader):
            x = data.to(device).float()
            z = encoder(x)
            x_hat = generator(z)

            recon = torch.sum(torch.square(x - x_hat), dim=1)
            l_1 = lambda1 * torch.sum(torch.log(torch.abs(z)+eps), dim=1)
            l_2 = lambda2 *  sum(torch.norm(param, p=2) ** 2 for param in generator.parameters())
            loss = torch.mean(recon + l_1 + l_2, dim=0)

            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

        

        print("Current Time:",time.ctime())
        print(f"epoch = {epoch}, loss = {loss.item()},recon ={torch.mean(recon)}, l1 = {torch.mean(l_1)}, l2 = {l_2}")
    print(f"Training {nepoch} epochs uses {time.time() - t1} seconds.")

    
    ## Save models
    torch.save(encoder, f"{save_path}/encoder.pth")
    torch.save(generator, f"{save_path}/generator.pth")



encoder = torch.load(f"{save_path}/encoder.pth",weights_only = False).to(device)
generator = torch.load(f"{save_path}/generator.pth",weights_only = False).to(device)

# View the latent dimensions pattern via encoder
test_data_path = 'embedding_data/val-embeddings.pt'
test_dataset = CustomDataset(test_data_path)
test_loader = DataLoader(test_dataset, batch_size = bs, shuffle = False)
for data in test_loader:
    break


x = data.to(device).float()
z = encoder(x)
x_hat = generator(z)

recon = torch.sum(torch.square(x - x_hat), dim=1)
l_1 = lambda1 * torch.sum(torch.log(torch.abs(z)+eps), dim=1)
l_2 = lambda2 *  sum(torch.norm(param, p=2) ** 2 for param in generator.parameters())
loss = torch.mean(recon + l_1 + l_2, dim=0)

print(f"epoch = {epoch}, loss = {loss.item()},recon ={torch.mean(recon)}, l1 = {torch.mean(l_1)}, l2 = {l_2}")

# print(var.mean(dim=0))
AD = []
z = torch.abs(z)
for b in range(z.shape[0]):
    z_slice = z[b]
    # import pdb;pdb.set_trace()
    thr = find_optimal_threshold_torch(torch.log(z_slice))
    ad = torch.sum(torch.log(z_slice) >thr)
    AD.append(ad)
AD_list = torch.tensor(AD).double()
print('AAD:',torch.mean(AD_list))
print('std AD:',torch.std(AD_list))


# sorted_var, indices = torch.sort(var, descending=True)
# print(sorted_var)
# print((var > 0.9 ).sum().item())
# print((var < 0.5 ).sum().item())
# print((var < 0.05 ).sum().item())
