# https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py
import torch
from torch._C import device
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torch import nn


def KNN(features, train_features, labels, train_labels, K=1):
    B, _ = features.shape
    num_correct = 0
    #######################################################################################
    # Question 2 Implementing the K-NN Evaluation (10 points)                             #
    # K: number of neighbors to vote                                                      #
    # Implement the K-NN evaluation by                                                    #
    # 1) find the K nearest train features for each feature                               #
    # 2) predict the labels as the most frequent train-labels from the nearest neighbors  #
    #    (If the number of neighbors are identical, return the smallest index)            #
    #    (e.g. [0(label_idx):32(number_of_neighbors), 1:6, 2:32] --> class:0)             #
    # 3) return the number of correct labels (labels == predicted_labels)                 #
    #######################################################################################
    ### YOUR CODE HERE (~ 6 lines)
    sim_matrix = torch.mm(features, train_features)
    # find k-nearest train features for each feature
    _, sim_indices = sim_matrix.topk(k=K, dim=-1)
    sim_labels = torch.gather(train_labels.expand(B, -1), dim=-1, index=sim_indices)
    # counts for each class
    one_hot_label = torch.zeros(B*K, labels.unique().size(0), device=sim_labels.device)
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    pred_scores = torch.sum(one_hot_label.view(B, -1, labels.unique().size(0)), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    num_correct += (pred_labels[:, 0] == labels).sum().item()
    ### END OF YOUR CODE
    return num_correct


class KNNValidation(object):
    def __init__(self, args, model, K=1):
        self.model = model
        self.device = torch.device('cuda' if next(model.parameters()).is_cuda else 'cpu')
        self.args = args
        self.K = K

        base_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = datasets.CIFAR10(root=args.data_root,
                                         train=True,
                                         download=True,
                                         transform=base_transforms)

        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           num_workers=args.num_workers,
                                           pin_memory=True,
                                           drop_last=True)

        val_dataset = datasets.CIFAR10(root=args.data_root,
                                       train=False,
                                       download=True,
                                       transform=base_transforms)

        self.val_dataloader = DataLoader(val_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         num_workers=args.num_workers,
                                         pin_memory=True,
                                         drop_last=True)

    def _topk_retrieval(self):
        """Extract features from validation split and search on train split features."""
        n_data = self.train_dataloader.dataset.data.shape[0]
        feat_dim = self.args.feat_dim

        self.model.eval()
        if str(self.device) == 'cuda':
            torch.cuda.empty_cache()

        train_features = torch.zeros([feat_dim, n_data], device=self.device)
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.train_dataloader):
                inputs = inputs.to(self.device)
                batch_size = inputs.size(0)

                # forward
                features = self.model(inputs)
                features = nn.functional.normalize(features)
                train_features[:, batch_idx * batch_size:batch_idx * batch_size + batch_size] = features.data.t()

            train_labels = torch.LongTensor(self.train_dataloader.dataset.targets).cuda()

        total = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.val_dataloader):
                labels = labels.cuda(non_blocking=True)
                features = self.model(inputs.to(self.device))
                num_correct = KNN(features, train_features, labels, train_labels)
                total += labels.size(0)
                correct += num_correct
        top1 = correct / total

        return top1

    def eval(self):
        return self._topk_retrieval()


if __name__=='__main__':
    print("=====Test K-Nearest Neighbor======")
    features = torch.FloatTensor([[0.2, -0.1,  0.4],
                                  [0.1,  0.3,  0.1],
                                  [0.4, -0.2,  0.2],
                                  [0.4, -0.1,  0.3],
                                  [0.2, -0.1,  0.1]])
    labels = torch.LongTensor([0, 1, 2, 2, 0])


    train_features = torch.FloatTensor([[0.2, -0.2,  0.4],
                                        [0.2,  0.4,  0.2],
                                        [0.2, -0.1,  0.2],
                                        [0.4, -0.1,  0.3],
                                        [0.2, -0.1,  0.2]]).T
    train_labels = torch.LongTensor([0, 1, 1, 2, 0])

    k_1_correct = KNN(features, train_features, labels, train_labels, K=1)
    k_2_correct = KNN(features, train_features, labels, train_labels, K=2)
    k_3_correct = KNN(features, train_features, labels, train_labels, K=3)
    k_4_correct = KNN(features, train_features, labels, train_labels, K=4)
    k_5_correct = KNN(features, train_features, labels, train_labels, K=5)

    assert k_1_correct == 4, f"Neareset neighbor at K=1 does not match expected result."
    assert k_2_correct == 3, f"Neareset neighbor at K=2 does not match expected result."
    assert k_3_correct == 2, f"Neareset neighbor at K=3 does not match expected result."
    assert k_4_correct == 3, f"Neareset neighbor at K=4 does not match expected result."
    assert k_5_correct == 2, f"Neareset neighbor at K=5 does not match expected result."
    print('[------passed------]')