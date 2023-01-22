"""
@author: Muhammad Suleman
This code is simplified implementation of
 "A. Iscen, G. Tolias, Y. Avrithis, O. Chum. "Label Propagation for Deep Semi-supervised Learning", CVPR 2019
Original Code: https://github.com/ahmetius/LP-DeepSSL
Coogle Colab Notebook of this code is available at
    https://colab.research.google.com/drive/1rruP9-2k6vLRAYGoqSh9bhdz1oaZcagO#scrollTo=7SZLFptiHa8g
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import Sampler
import itertools
import numpy as np
import random
import faiss # pip install faiss-gpu
from faiss import normalize_L2
import scipy
import scipy.stats

# Batch sampling code ------------------------------------------------------------------------------------------------------------------------

batch_size = 64
labeled_batch_size=32

class TwoStreamBatchSampler(Sampler):
    """Iterate over two sets of indices
    An 'epoch' is one iteration through all the unlabeled indices(primary).
    During the epoch, the labeled indices(secondary) are iterated through
    as many times as needed. For each batch of 64, 32 are labeled samples and 32 unlabeled.
    It will be used only in label propagation training
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)



# Prepare Label Data-------------------------------------------------------------------------------------------------------------
"""Prepare indexes and get Labeled Data subset.Fraction represents amount of data selected out of total samples.
   For example if we have 10,000 images and we select frac=0.01 this means 1% data or 100 images will selected randomly"""


def label_prop_prepare(dataset,frac): 
  label_index_size=int(len(dataset.data)*frac)
  divs=int(len(dataset.data)/label_index_size)
  r=random.randint(1, divs)
  label_index=np.arange((r-1)*label_index_size,r*label_index_size)
  unlabel_index = np.array(sorted(set(range(len(dataset.data))) - set(label_index)))
  

  label_index_list=label_index.tolist()
  unlabel_index_list=unlabel_index.tolist()
  # dataset.targets[unlabel_index]=-1
  

  batch_sampler = TwoStreamBatchSampler(
            unlabel_index_list, label_index_list, batch_size, labeled_batch_size)
  
  return label_index,unlabel_index,label_index_list,unlabel_index_list,batch_sampler

# ---------------------------------------------------------------------------------------------------------------------------------------


# Get training data from open datasets.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Get test data from open datasets.
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# call Label Prepare function
label_index,unlabel_index,label_index_list,unlabel_index_list,batch_sampler = label_prop_prepare(training_data,frac=0.003)

# Obtain labeled and unlabeled data subsets
trainset_labeled = torch.utils.data.Subset(training_data, label_index_list)
trainset_unlabeled = torch.utils.data.Subset(training_data, unlabel_index_list)


# Create data loaders.
train_label_dataloader = DataLoader(trainset_labeled, batch_size=batch_size) #  Shape of X [No_samples, Channel, Height, Width]
train_feat_dataloader = DataLoader(training_data, batch_size=batch_size)  # Used for feature extraction
test_dataloader = DataLoader(test_data, batch_size=batch_size)
train_final_dataloader = DataLoader(training_data, batch_sampler=batch_sampler) # Used for training with label propagation

# Run program on GPU if available
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

# Declare model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.pool = nn.MaxPool2d(2,2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(12*12*64,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout1(x)
        x = x.view(-1,12*12*64)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

model = NeuralNetwork().to(device) #run model on GPU e.g.(device=cuda)


# initialize weights 
num_classes=10
p_weights = np.ones((len(training_data.targets),))
class_weights = np.ones(((num_classes)),dtype = np.float32)
all_labels=training_data.targets
all_labels = np.array(all_labels)


# declare custom weighted loss function
def element_weighted_loss(y_hat, y, weights,class_weights):
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    loss = criterion(y_hat, y)
    loss = loss * weights
    loss=  loss * class_weights
    return loss.mean()

loss_fn = nn.CrossEntropyLoss() #standard loss function for test data
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# define train function 
def train(dataloader, model,optimizer,p_weights,class_weights):

    model.train()

    for batch, ((X,y), p_weights_b,c_weights_b) in enumerate(zip(dataloader, p_weights,class_weights)):
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        # p_weights_b=np.asarray(p_weights_b)
        p_weights_t=torch.from_numpy(np.asarray(p_weights_b))
        c_weights_t=torch.from_numpy(np.asarray(c_weights_b))

        # weighted loss calculation
        target_var = torch.autograd.Variable(y.cuda())
        weight_var = torch.autograd.Variable(p_weights_t.cuda())
        c_weight_var = torch.autograd.Variable(c_weights_t.cuda())
        loss = element_weighted_loss(pred, target_var,weight_var,c_weight_var)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"training loss: {loss:>7f} ")


def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Data: \n Accuracy: {(100*correct):>0.1f}%,  loss: {test_loss:>8f} \n")

# Initial training without label propagation
epochs = 30
for t in range(epochs):
    print(f"Epoch {t+1}\n---------Labeled Data Only-----------")
    train(train_label_dataloader, model, optimizer,p_weights,class_weights)
    test(test_dataloader, model)

# Hook feature used to extract features
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# Function for extracting features
def extract_features(train_loader,model):
    model.eval()
    embeddings_all= []
    for i, (batch_input, target) in enumerate(train_loader):
    
        X = batch_input
        y = batch_input[1]
        X = torch.autograd.Variable(X.cuda())
        y = torch.autograd.Variable(y.cuda())
        model.fc1.register_forward_hook(get_activation('fc1'))
        feature=activation['fc1']
        embeddings_all.append(feature.data.cpu())

    embeddings_all = np.asarray(torch.cat(embeddings_all).numpy())
    return embeddings_all

# Label Propagation Function
def update_plabels(feat, k = 50, max_iter = 20):
      
        alpha = 0.99
        labels = all_labels
        labeled_idx = label_index

        
        # kNN search for the graph
        d = feat.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(torch.cuda.device_count()) - 1
        index = faiss.GpuIndexFlatIP(res,d,flat_config)   # build the index

        normalize_L2(feat)
        index.add(feat) 
        N = feat.shape[0]



        D, I = index.search(feat, k + 1)


        # Create the graph
        D = D[:,1:] ** 3
        I = I[:,1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx,(k,1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N)) #eq 9
        W = W + W.T  #make matrix symmetric

        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis = 1)
        S[S==0] = 1
        D = np.array(1./ np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D

        # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
        Z = np.zeros((N,num_classes))
        A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
        for i in range(num_classes):
            cur_idx = labeled_idx[np.where(labels[labeled_idx] ==i)]
            y = np.zeros((N,))
            y[cur_idx] = 1.0 / cur_idx.shape[0]
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
            Z[:,i] = f   #eq 10

        # Handle numberical errors
        Z[Z < 0] = 0 

        # Compute the weight for each instance based on the entropy (eq 11 from the paper)
        probs_l1 = F.normalize(torch.tensor(Z),1).numpy()
        probs_l1[probs_l1 <0] = 0
        entropy = scipy.stats.entropy(probs_l1.T)
        weights = 1 - entropy / np.log(num_classes)
        weights = weights / np.max(weights)   #eq 11
        p_labels = np.argmax(probs_l1,1)

        # Compute the accuracy of pseudolabels for statistical purposes
        correct_idx = (p_labels == labels)
        acc = correct_idx.mean()
        print(f'Label Propagation accuracy= {acc*100:>0.2f}%')

        p_labels[labeled_idx] = labels[labeled_idx]
        weights[labeled_idx] = 1.0

        p_weights = weights.tolist()
        p_labels = p_labels

        # Compute the weight for each class
        for i in range(num_classes):
            cur_idx = np.where(np.asarray(p_labels) == i)[0]
            class_weights[i] = (float(labels.shape[0]) / (num_classes)) / cur_idx.size

        return p_labels,p_weights,class_weights


global_epochs = 10
for t in range(global_epochs):
    feat=extract_features(train_feat_dataloader,model)
    p_labels,p_weights,class_weights=update_plabels(feat,k=50,max_iter=20)
    training_data.targets=p_labels
    print(f"Epoch {t+1}")
    train(train_final_dataloader, model, optimizer,p_weights,class_weights)
    test(test_dataloader, model)
