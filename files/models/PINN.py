import torch
import torch.nn as nn
import torch.nn.functional as F
class NM_PINN(nn.Module):
    def __init__(self, num_channels=22, F1=8, D=2, num_classes=4, fs=250, pool_factor=8):
        super(NM_PINN, self).__init__()
        self.F1 = F1
        self.D = D  
        self.fs = fs
        self.pool_factor = pool_factor  
        self.dt_phys = pool_factor / fs 
        
        assert D == 2, "D must be 2 to split spatial features into Excitatory and Inhibitory pairs."

        kernel_length = fs // 2
        self.temp_conv = nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, kernel_size=(num_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.activation = nn.ELU()

        # --- Macroscopic Physics Parameters ---
        self.tau_E = nn.Parameter(torch.full((1, F1, 1), 0.01))
        self.tau_I = nn.Parameter(torch.full((1, F1, 1), 0.01))
        self.w_EE = nn.Parameter(torch.full((1, F1, 1), 1.2))
        self.w_EI = nn.Parameter(torch.full((1, F1, 1), 1.0))
        self.w_IE = nn.Parameter(torch.full((1, F1, 1), 1.0))
        self.w_II = nn.Parameter(torch.full((1, F1, 1), 0.5))
        self.P = nn.Parameter(torch.zeros(1, F1, 1))
        self.Q = nn.Parameter(torch.zeros(1, F1, 1))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        self.dropout = nn.Dropout(p=0.5)
        
        # --- FUSED CLASSIFIER ---
        # x_pool (16) + E_feat (8) + I_feat (8) = 32 dimensions
        fused_feature_dim = (F1 * D) + F1 + F1 
        self.classifier = nn.Linear(fused_feature_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def compute_wc_loss(self, E, I):
        tE = F.softplus(self.tau_E); tI = F.softplus(self.tau_I)
        wEE = F.softplus(self.w_EE); wEI = F.softplus(self.w_EI)
        wIE = F.softplus(self.w_IE); wII = F.softplus(self.w_II)

        dE_dt = (E[:, :, 1:] - E[:, :, :-1]) / self.dt_phys
        dI_dt = (I[:, :, 1:] - I[:, :, :-1]) / self.dt_phys
        E_t = E[:, :, :-1]; I_t = I[:, :, :-1]

        drv_E = wEE * E_t - wEI * I_t + self.P
        drv_I = wIE * E_t - wII * I_t + self.Q

        res_E = tE * dE_dt + E_t - torch.sigmoid(drv_E)
        res_I = tI * dI_dt + I_t - torch.sigmoid(drv_I)
        return torch.mean(res_E ** 2) + torch.mean(res_I ** 2)

    def compute_orthogonality_loss(self):
        """ Penalize spatial correlation to force distinct biological representations. """
        weights = self.depthwise.weight.squeeze() # Shape: (16, 22)
        
        W_E = weights[:self.F1, :] # Shape: (8, 22)
        W_I = weights[self.F1:, :] # Shape: (8, 22)
        
        W_E_norm = F.normalize(W_E, p=2, dim=1)
        W_I_norm = F.normalize(W_I, p=2, dim=1)
        
        similarity = torch.sum(W_E_norm * W_I_norm, dim=1)
        return torch.mean(similarity ** 2)

    def forward(self, x):
        if len(x.shape) == 3: x = x.unsqueeze(1)
        
        # 1. Base Feature Extraction
        x = self.bn1(self.temp_conv(x))
        x_main = self.activation(self.bn2(self.depthwise(x)))

        # 2. Physics Branch Processing
        x_phys = F.avg_pool1d(x_main.squeeze(2), self.pool_factor)
        E_phys = torch.sigmoid(x_phys[:, :self.F1, :])
        I_phys = torch.sigmoid(x_phys[:, self.F1:, :])
        
        # 3. Calculate Regularization Losses
        loss_wc = self.compute_wc_loss(E_phys, I_phys)
        loss_ortho = self.compute_orthogonality_loss()

        # 4. Feature Fusion
        E_feat = E_phys.mean(dim=-1)
        I_feat = I_phys.mean(dim=-1)
        x_pool = self.avg_pool(x_main).squeeze(-1).squeeze(-1) 
        
        # Concatenate: (Batch, 16) + (Batch, 8) + (Batch, 8) -> (Batch, 32)
        fused_features = self.dropout(torch.cat([x_pool, E_feat, I_feat], dim=-1))
        
        # 5. Final Classification
        logits = self.classifier(fused_features)
        
        return logits, loss_wc, loss_ortho
'''
====================================================================================================
Layer (type:depth-idx)                   Input Shape          Output Shape         Param #
====================================================================================================
NM_PINN                                  [64, 22, 500]        [64, 4]              64
├─Conv2d: 1-1                            [64, 1, 22, 500]     [64, 8, 22, 500]     1,000
├─BatchNorm2d: 1-2                       [64, 8, 22, 500]     [64, 8, 22, 500]     16
├─Conv2d: 1-3                            [64, 8, 22, 500]     [64, 16, 1, 500]     352
├─BatchNorm2d: 1-4                       [64, 16, 1, 500]     [64, 16, 1, 500]     32
├─ELU: 1-5                               [64, 16, 1, 500]     [64, 16, 1, 500]     --
├─AdaptiveAvgPool2d: 1-6                 [64, 16, 1, 500]     [64, 16, 1, 1]       --
├─Dropout: 1-7                           [64, 32]             [64, 32]             --
├─Linear: 1-8                            [64, 32]             [64, 4]              132
====================================================================================================
Total params: 1,596
Trainable params: 1,596
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 715.28
====================================================================================================
Input size (MB): 2.82
Forward/backward pass size (MB): 98.31
Params size (MB): 0.01
Estimated Total Size (MB): 101.13
====================================================================================================
'''