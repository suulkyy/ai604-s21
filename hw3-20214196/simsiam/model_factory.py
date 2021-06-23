from torch import nn
import copy
from .resnet_cifar import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        hidden_dim = out_dim
        self.num_layers = num_layers
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)  # Page:5, Paragraph:2
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        out_dim = in_dim
        hidden_dim = int(out_dim / 4)

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.backbone = self.get_backbone(args.arch)
        out_dim = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()
        self.projector = projection_MLP(out_dim, args.feat_dim,
                                        args.num_proj_layers)
        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )
        self.use_momentum_encoder = args.use_momentum_encoder
        if args.use_momentum_encoder:
            self.momentum_encoder = copy.deepcopy(self.encoder)
        self.use_predictor = args.use_predictor
        if args.use_predictor:
            self.predictor = prediction_MLP(args.feat_dim)

    @staticmethod
    def get_backbone(backbone_name):
        return {'resnet18': ResNet18(),
                'resnet34': ResNet34(),
                'resnet50': ResNet50(),
                'resnet101': ResNet101(),
                'resnet152': ResNet152()}[backbone_name]

    def forward(self, im_aug1, im_aug2):
        z1, z2, p1, p2 = None, None, None, None
        ###################################################################################
        # Question 1.1 Implementing the Model (10 points)                                 #
        # Implement the model to forward the two agumented images im_aug1 and im_aug2     #
        # 1) if using the momentum_encoder, z2 is the output of the momentum_encoder      #
        # 2) If not using the predictor layer, just set p1 and p2 to None                 #
        ###################################################################################
        ### YOUR CODE HERE (~ 8 lines)
        # # f: backbone + projection mlp
        # f = self.encoder
        # # m: prediction mlp
        # h = self.predictor
        # Projections
        z1, z2 = self.encoder(im_aug1), self.encoder(im_aug2) 
        # Momentum encoder
        if self.use_momentum_encoder:
            z2 = self.momentum_encoder(im_aug2)
        # Predictons
        if self.use_predictor:
            p1, p2 = self.predictor(z1), self.predictor(z2)
        ### END OF YOUR CODE

        return {'z1': z1, 'z2': z2, 'p1': p1, 'p2': p2}
