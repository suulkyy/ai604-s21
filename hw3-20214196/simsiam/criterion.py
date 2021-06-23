from torch import nn
import torch


class SSLLoss(nn.Module):
    def __init__(self, loss_type='contrastive', stop_gradient=True, use_predictor=False, temperature=0.5):
        super().__init__()
        self.stop_gradient = stop_gradient
        self.use_predictor = use_predictor

        # contrastive loss
        self.temperature = temperature
        if loss_type == 'contrastive':
            self.loss_func = self.contrastive_loss
        else:
            self.loss_func = self.align_loss

    def contrastive_loss(self, z1, z2):
        B, D = z1.shape
        loss = None
        ###################################################################################
        # Question 1.2 Implementing the Contrastive Loss (20 points)                      #
        # Implement the contrastive loss of representation z1 and z2                      #
        # B: batch_size                                                                   #
        # D: representation_dimension                                                     #
        # 1) representation of each instance should be normalized
        # 2) logits should be divided by temperature to yield a smoother distribution     #
        ###################################################################################
        ### YOUR CODE HERE (~ 9 lines)
        z_hat = torch.cat([torch.nn.functional.normalize(z1, dim=1), torch.nn.functional.normalize(z2, dim=1)], dim=0)
        sim_matrix = torch.nn.functional.cosine_similarity(z_hat.unsqueeze(1), z_hat.unsqueeze(0), dim=-1)
        # Create 'positive' labels
        pos = torch.cat([torch.diag(sim_matrix, B), torch.diag(sim_matrix, B)]).view(2*B, 1)
        # Create 'negative' labels
        diag = torch.eye(2*B, dtype=torch.bool, device=z1.device)
        diag[B:, :B] = diag[:B, B:] = diag[:B, :B]
        neg = sim_matrix[~diag].view(2*B, -1)
        # calculate logits and divide logits by temperature
        logits = torch.cat([pos, neg], dim=1)/self.temperature
        # calculate loss
        loss = torch.nn.functional.cross_entropy(logits, torch.zeros(2*B, dtype=torch.int64, device=z1.device), reduction='sum')/(2*B)
        ### END OF YOUR CODE
        return loss

    def align_loss(self, z1, z2):
        B, D = z1.shape
        loss = None
        #############################################################################
        # Question 1.3 Implementing the Align Loss (10 points)                       #
        # Implement the align loss of representation v1 and v2                      #
        # B: batch_size                                                             #
        # D: representation_dimension                                               #
        # 1) representation of each instance should be normalized                   #
        # 2) what is the relationship between l2-distance and cosine similarity     #
        #############################################################################
        ### YOUR CODE HERE (~ 1 line)
        loss = -torch.nn.functional.cosine_similarity(torch.nn.functional.normalize(z1, dim=1), torch.nn.functional.normalize(z2, dim=1), dim=-1).mean()
        ### END OF YOUR CODE
        return loss

    def forward(self, z1, z2, p1, p2):
        if self.use_predictor:
            if self.stop_gradient:
                z1 = z1.detach()
                z2 = z2.detach()

            loss1 = self.loss_func(p1, z2)
            loss2 = self.loss_func(p2, z1)

            return 0.5 * loss1 + 0.5 * loss2
        else:
            if self.stop_gradient:
                z2 = z2.detach()

            return self.loss_func(z1, z2)


if __name__=='__main__':
    print("====Test Self-Supervised Loss Function====")
    x1 = torch.FloatTensor([[0.2, -0.1,  0.4],
                            [0.1,  0.3,  0.1]])
    x2 = torch.FloatTensor([[0.3, -0.2,  0.3],
                            [0.2,  0.4,  0.2]])
    Wz = torch.nn.Parameter(torch.FloatTensor([[0.42,  0.13, -0.30],
                                               [0.21,  0.23, -0.13],
                                               [-0.12, 0.31,  0.05]]))
    Wp = torch.nn.Parameter(torch.FloatTensor([[0.30,  0.17, -0.35],
                                               [0.19,  0.41, -0.03],
                                               [-0.22, 0.21,  0.12]]))

    print('[Test 1.1: Contrastive Loss w/o predictor]')
    z1 = torch.mm(x1, Wz)
    z2 = torch.mm(x2, Wz)
    p1 = torch.mm(x1, Wp)
    p2 = torch.mm(x2, Wp)

    criterion = SSLLoss(loss_type='contrastive', use_predictor=False)
    loss = criterion(z1, z2, p1, p2)
    assert loss.detach().allclose(torch.tensor(1.0549), atol=1e-3), f"Contrastive Loss does not match expected result."
    print('[-----------------passed-----------------]')

    print('[Test 1.2: Contrastive Loss w/_ predictor]')
    criterion = SSLLoss(loss_type='contrastive', use_predictor=True)
    loss = criterion(z1, z2, p1, p2)
    assert loss.detach().allclose(torch.tensor(1.1306), atol=1e-3), f"Contrastive Loss does not match expected result."
    print('[-----------------passed-----------------]')

    print('[Test 2.1: Align Loss w/o predictor]')
    criterion = SSLLoss(loss_type='align', use_predictor=False)
    loss = criterion(z1, z2, p1, p2)
    assert loss.detach().allclose(torch.tensor(-0.9508), atol=1e-3), f"Align Loss does not match expected result."
    print('[--------------passed--------------]')

    print('[Test 2.2: Align Loss w/_ predictor]')
    criterion = SSLLoss(loss_type='align', use_predictor=True)
    loss = criterion(z1, z2, p1, p2)
    assert loss.detach().allclose(torch.tensor(-0.7458), atol=1e-3), f"Align Loss does not match expected result."
    print('[--------------passed--------------]')
