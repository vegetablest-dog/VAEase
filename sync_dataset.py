import torch
from models import *

class SampleBox(nn.Module):
    def __init__(self, d, r, kappa, mix_param, comp_param, device, seed=123):
        super(SampleBox, self).__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.d = d
        self.r = r
        self.kappa = kappa
        self.device = device

        self.XTransformer = nn.Linear(self.r, self.d)

        mix = D.Categorical(probs=mix_param)
        comp = D.Independent(D.Normal(loc=comp_param[0], scale=comp_param[1]), 1)
        self.gmm = D.MixtureSameFamily(mix, comp)


    def generate_xnc(self, sample_size, x_tran):
        # Mixed Gaussian
        with torch.no_grad():
            x = self.gmm.sample((sample_size,)) + torch.randn((sample_size, self.r), device=self.device)
            x = x_tran(self.XTransformer(x))
        return x
    
class KSphere(nn.Module):

    def __init__(self, d, r, device, seed=114514):
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.d = d
        self.r = r
        self.device = device

        embedding_matrix = torch.randn(size=(d, r+1))
        q, r = np.linalg.qr(embedding_matrix)
        self.q = torch.from_numpy(q)
        
    def generate_xnc(self, sample_size):
        with torch.no_grad():
            new_data = torch.randn((sample_size, self.r+1))
            norms = torch.linalg.norm(new_data, dim=1)
            new_data = new_data / norms[:,None]
            new_data = (self.q @ new_data.T).T
        return new_data.to(self.device)
    

from torch.utils.data import random_split, Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, config):
        super(SyntheticDataset, self).__init__()
        self.data, self.labels = self.create_dataset(config)
        self.return_labels = False
   
    def create_dataset(self, config):
        raise NotImplemented
        # return data, labels

    # def log_prob(self, xs, ts):
    #     raise NotImplemented

    # def ground_truth_score(self, xs, ts):
    #     log_prob_x = lambda x: self.log_prob(x,ts)
    #     return compute_grad(log_prob_x, xs)

    def __getitem__(self, index):
        if self.return_labels:
            item = self.data[index], self.labels[index]
        else:
            item = self.data[index]
        return item 

    def __len__(self):
        return len(self.data)
    
import random

class FixedSquaresManifold(SyntheticDataset):
    def __init__(self, config):
        super().__init__(config)
    
    def get_the_squares(self, seed, num_squares, square_range, img_size):
        random.seed(seed)
        squares_info = []
        for _ in range(num_squares):
            side = random.choice(square_range)
            start = (side+1)//2
            finish = img_size - (side+1)//2
            x = random.choice(np.arange(start, finish))
            y = random.choice(np.arange(start, finish))
            squares_info.append([x, y, side])

        return squares_info

    def create_dataset(self, config):
        num_samples = config.data.data_samples
        num_squares = config.data.num_squares #10
        square_range = config.data.square_range #[3, 5]
        img_size = config.data.image_size #32
        seed = config.seed 

        squares_info = self.get_the_squares(seed, num_squares, square_range, img_size)
        data = []
        for num in tqdm(range(num_samples)):
            img = torch.zeros(size=(img_size, img_size))
            for i in range(num_squares):
                x, y, side = squares_info[i]
                img = self.paint_the_square(img, x, y, side)   
            data.append(img.to(torch.float32).unsqueeze(0))
        
        data = torch.stack(data)
        return data, []
    
    def paint_the_square(self, img, center_x, center_y, side):
        c = random.random()
        for i in range(side):
            for j in range(side):
                img[center_x - ((side+1)//2 - 1) + i, center_y - ((side+1)//2 - 1) + j]+=c
        return img
    

class ComposedLinearDataset(SyntheticDataset):
    def __init__(self, config):
        super().__init__(config)
        self.return_labels = True

    def create_dataset(self, config):
        num_samples = config.data.data_samples
        num_manifolds = config.data.num_manifolds
        dim_per_manifold = config.data.dim_per_manifold
        d = config.data.d 
        setting_dim = config.data.setting_dim
        seed = config.seed 
        torch.manual_seed(seed)
        random.seed(seed)
        dim_index = []
        dim_index_matrix = []
        for i in range(num_manifolds):
            I = torch.zeros((d,d))
            indexs = []
            for j in range(dim_per_manifold):
                if setting_dim is not None:
                    index = setting_dim[i][j]
                else:
                    index = random.choice(list(range(d)))
                indexs.append(index)
                I[index,index]=1
            dim_index_matrix.append(I)
            dim_index.append(indexs)
        self.dim_index = [dim_index]
        #warning: we did not guarentee the manifolds are different from each other.
        randn_matrix = torch.randn(d,d)
        Q, R = torch.linalg.qr(randn_matrix)
        map_matrix = Q
        map_matrix = randn_matrix
        data = []
        labels = []
        for num in tqdm(range(num_samples)):
            selected_manifold = random.choice(list(range(num_manifolds)))
            I = dim_index_matrix[selected_manifold]
            z = torch.randn(d)
            x = map_matrix@I@z
            data.append(x.to(torch.float32).unsqueeze(0))
            labels.append(selected_manifold)
        data = torch.stack(data)
        labels = torch.tensor(labels)

        return data, labels


class ComposedSphereDataset(SyntheticDataset):
    def __init__(self, config):
        super().__init__(config)
        self.return_labels = True

    def create_dataset(self, config):
        self.manifolds_sample_size = config.data.manifolds_sample_size
        self.manifolds_dims = config.data.manifolds_dims
        self.d = config.data.d 
        seed = config.seed 
        torch.manual_seed(seed)
        self.manifold_nums = min(len(self.manifolds_sample_size),len(self.manifolds_dims))
        data = []
        labels = []
        qs = []
        for i in range(self.manifold_nums):
            da,q = self.generate_manifold_samples(self.manifolds_sample_size[i],self.manifolds_dims[i])
            data.append(da)
            qs.append(q)
            labels.append(torch.tensor([i]*self.manifolds_sample_size[i]))
        
        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)
        q_t = torch.cat(qs, dim=0)
        
        return data, labels
    
    def generate_manifold_samples(self, sample_size, r):
        embedding_matrix = torch.randn(size=(self.d, r+1))
        q, _ = np.linalg.qr(embedding_matrix)
        q = torch.from_numpy(q)
        new_data = torch.randn((sample_size, r+1))
        norms = torch.linalg.norm(new_data, dim=1)
        new_data = new_data / norms[:,None]
        new_data = (q @ new_data.T).T
        return new_data,q


class ComposedSphereDataset_Orth(SyntheticDataset):

    def __init__(self, config):
        super().__init__(config)
        self.return_labels = True

    def create_dataset(self, config):
        self.manifolds_sample_size = config.data.manifolds_sample_size
        self.manifolds_dims = config.data.manifolds_dims
        self.d = config.data.d 
        seed = config.seed 
        torch.manual_seed(seed)
        self.manifold_nums = min(len(self.manifolds_sample_size),len(self.manifolds_dims))
        data = []
        labels = []
        embedding_matrix = torch.randn(size=(self.d, self.d))
        q, _ = np.linalg.qr(embedding_matrix)
        q = torch.from_numpy(q)
        index = 0
        data_scale = 1
        for i in range(self.manifold_nums):
            mani_dim = self.manifolds_dims[i]
            da = self.generate_manifold_samples(self.manifolds_sample_size[i],mani_dim)
            da = da @ q[index:index+mani_dim+1] * data_scale
            v = torch.randn(self.d)
            da = da + v
            index += mani_dim+1
            data.append(da)
            labels.append(torch.tensor([i]*self.manifolds_sample_size[i]))
        
        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)
        
        return data, labels
    
    def generate_manifold_samples(self, sample_size, r):
        new_data = torch.randn((sample_size, r+1)) + torch.uniform(-0.5, 0.5, size=(sample_size, r+1))
        norms = torch.linalg.norm(new_data, dim=1)
        new_data = new_data / norms[:,None]
        v = torch.randn(r+1)
        new_data = new_data + v
        return new_data

class MLPDataset(SyntheticDataset):

    def __init__(self, config):
        super().__init__(config)
        self.return_labels = True

    def create_dataset(self, config):
        self.manifolds_sample_size = config.data.manifolds_sample_size
        self.manifolds_dims = config.data.manifolds_dims
        self.d = config.data.d 
        seed = config.seed 
        torch.manual_seed(seed)
        self.manifold_nums = min(len(self.manifolds_sample_size),len(self.manifolds_dims))
        data = []
        labels = []
        data_scale = 1
        for i in range(self.manifold_nums):
            mani_dim = self.manifolds_dims[i]
            da = self.generate_manifold_samples(self.manifolds_sample_size[i],mani_dim)
            da = da * data_scale
            data.append(da)
            labels.append(torch.tensor([i]*self.manifolds_sample_size[i]))
        
        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)
        
        return data, labels
    
    def generate_manifold_samples(self, sample_size, r):
        new_data = torch.randn((sample_size, r)) 
        matrix_A = torch.randn((r, r))
        bias_A = torch.randn((r))
        matrix_B = torch.randn((r, self.d))
        bias_B = torch.randn((self.d))
        # new_data = new_data @ matrix_A + bias_A
        # new_data = new_data @ matrix_A + bias_A
        # new_data = new_data * new_data
        # new_data = new_data @ matrix_B + bias_B
        new_data = torch.nn.functional.leaky_relu(new_data @ matrix_A + bias_A, negative_slope = 0.2)@matrix_B + bias_B
        return new_data

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.transforms.base import CompositeTransform
import nflows.utils.torchutils as torchutils
import nflows.transforms as transforms


class SplineFlowDataset(SyntheticDataset):

    def __init__(self, config):
        super().__init__(config)
        self.return_labels = True

    def create_dataset(self, config):
        self.manifolds_sample_size = config.data.manifolds_sample_size
        self.manifolds_dims = config.data.manifolds_dims
        self.d = config.data.d 
        seed = config.seed 
        torch.manual_seed(seed)
        self.manifold_nums = min(len(self.manifolds_sample_size),len(self.manifolds_dims))
        data = []
        labels = []
        data_scale = 1
        for i in range(self.manifold_nums):
            mani_dim = self.manifolds_dims[i]
            da = self.generate_manifold_samples(self.manifolds_sample_size[i],mani_dim)
            da = da * data_scale
            data.append(da)
            labels.append(torch.tensor([i]*self.manifolds_sample_size[i]))
        
        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)
        
        return data, labels
    
    def generate_manifold_samples(self, sample_size, r):
        new_data = torch.randn((sample_size, r)) 
        matrix_A = torch.randn((r, self.d))
        bias_A = torch.randn((self.d))
        new_data = new_data @ matrix_A + bias_A
        
        class CustomTransformNet(torch.nn.Module):
            def __init__(self, input_dim, output_dim):
                super(CustomTransformNet, self).__init__()
                self.layer = torch.nn.Linear(input_dim, output_dim)

            def forward(self, x, context=None):
                if context is not None:
                    pass
                return self.layer(x)

        def create_transform_net(input_dim, output_dim):
            return CustomTransformNet(input_dim, output_dim)
        
        rq_spline_transform = transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=[1]*self.d,
                transform_net_create_fn=lambda input_dim, output_dim: create_transform_net(input_dim, output_dim), 
                num_bins=10,
                tails='linear',
                tail_bound=5.0
            )
        new_data = rq_spline_transform(new_data)
        
        return new_data[0].detach()


class TestDataset(SyntheticDataset):

    def __init__(self, config):
        super().__init__(config)
        self.return_labels = True

    def create_dataset(self, config):
        self.manifolds_sample_size = config.data.manifolds_sample_size
        self.manifolds_dims = config.data.manifolds_dims
        self.d = config.data.d 
        seed = config.seed 
        torch.manual_seed(seed)
        self.manifold_nums = min(len(self.manifolds_sample_size),len(self.manifolds_dims))
        data = []
        labels = []
        new_data = torch.randn(self.manifolds_sample_size[0],self.manifolds_dims[0])
        A = torch.randn(self.manifolds_dims[0],self.d)
        data.append(new_data @ A)
        labels.append(torch.tensor([0]*self.manifolds_sample_size[0]))
        new_data = torch.randn(self.manifolds_sample_size[1],self.manifolds_dims[1])
        B = torch.rand(self.manifolds_dims[1],self.d)*10-5
        data.append(torch.square(new_data @ B)*0.2)
        labels.append(torch.tensor([1]*self.manifolds_sample_size[1]))
        
        data = torch.cat(data, dim=0)
        labels = torch.cat(labels, dim=0)
        
        return data, labels
    

class TwoPointsDataset(SyntheticDataset):
    def __init__(self, config):
        super().__init__(config)
        self.return_labels=True

    def create_dataset(self, config):
        num_samples = config.data.data_samples
        d = config.data.d
        scale = config.data.scale
        data = []
        labels = []
        data.append(torch.ones((num_samples,d))*scale)
        data.append(torch.zeros((num_samples,d)))
        labels.append(torch.tensor([1]*num_samples))
        labels.append(torch.tensor([0]*num_samples))
        data = torch.cat(data,dim=0)
        labels = torch.cat(labels, dim=0)

        return data,labels


class CustomDataset(Dataset):
    def __init__(self, file_path):
        self.data = torch.load(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class CustomLabeledDataset(Dataset):
    def __init__(self, file_path):
        self.data = torch.load(file_path)
        self.features = self.data[0]
        self.labels = self.data[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx],self.labels[idx]