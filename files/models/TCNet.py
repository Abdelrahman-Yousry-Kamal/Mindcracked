import torch
import torch.nn as nn
import torch.nn.functional as F

"""
used EEGTCNet combination for comparisons:

EEGTCNet(nb_classes=4,
        Chans=22,
        dropoutRate=0.5,
        D=4)

should have at least 750 epoch!
"""

class TCNBlock(nn.Module):
    """Single residual block of the TCN module (dilated causal conv x2 + residual)."""
    def __init__(
        self,
        Chans,
        FK,
        kernLength,
        dilation,
        dropoutRate,
    ):
        super().__init__()
        pad = (kernLength - 1) * dilation  # causal padding

        self.conv1 = nn.Conv1d(Chans, FK, kernLength, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(FK)
        self.drop1 = nn.Dropout(dropoutRate)

        self.conv2 = nn.Conv1d(FK, FK, kernLength, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(FK)
        self.drop2 = nn.Dropout(dropoutRate)

        self.pad = pad
        self.downsample = nn.Conv1d(Chans, FK, 1) if Chans != FK else None


    def forward(self, x):
        out = F.elu(self.bn1(self.conv1(x))[:, :, :-self.pad if self.pad else None])
        out = self.drop1(out)
        out = F.elu(self.bn2(self.conv2(out))[:, :, :-self.pad if self.pad else None])
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return F.elu(out + res)


class EEGTCNet(nn.Module):
    def __init__(
        self,
        Chans,                     # C
        nb_classes: int = 4,
        kernLength: int = 32,              # kernel size of first conv (phi^1)
        dropoutRate = 0.2,
        F1: int = 8,               # temporal filters
        F2: int = 16,              # spatial/separable filters
        FT: int = 12,              # number of filters in TCN
        KT: int = 4,               # kernel size in TCN
        tcn_layers: int = 2,       # number of dilated residual blocks
        pt: float = 0.3,           # dropout for TCN module
        D = 2,                     # depth multiplier for depthwise conv (fixed: F1*2 in table)
    ):
        super().__init__()
        # phi^1: temporal convolution
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kernLength),
                padding='same',
                bias=False,
            ),

            nn.BatchNorm2d(F1),
        )

        # phi^2: depthwise spatial convolution + pool
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=F1,
                out_channels=F1*D,
                kernel_size=(Chans, 1),
                groups=F1,
                bias=False,
            ),

            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropoutRate),
        )

        # phi^3: separable convolution + pool
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F1 * D,
                kernel_size=(1, 16),
                padding='same',
                groups=F1 * D,      # depthwise part
                bias=False,
            ),

            nn.Conv2d(
                in_channels=F1 * D,
                out_channels=F2,
                kernel_size=(1, 1),
                bias=False,
            ),

            nn.ELU(),
            nn.BatchNorm2d(F2),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropoutRate),
        )

        # phi^4: TCN module (stack of dilated residual blocks)
        tcn_blocks = []
        in_ch = F2
        for i in range(tcn_layers):
            dilation = 2 ** i
            tcn_blocks.append(TCNBlock(in_ch, FT, KT, dilation, pt))
            in_ch = FT
        self.tcn = nn.Sequential(*tcn_blocks)

        # phi^5: dense classification head (uses last TCN timestep)
        self.fc = nn.Linear(FT, nb_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        # x: (batch, 1, C, T)
        x = self.layer1(x)                # (B, F1, C, T)
        x = self.layer2(x)                       # (B, F1*2, 1, T//8)

        x = self.layer3(x)                         # (B, F2, 1, T//64)

        x = x.squeeze(2)                                      # (B, F2, T//64)
        x = self.tcn(x)                                       # (B, FT, T//64)

        x = x[:, :, -1]                                       # last time step -> (B, FT)
        return self.fc(x)                                     # (B, nb_classes) raw logits



    @staticmethod
    def training(model, criterion, optimizer, epochs, lr, train_loader, device="cuda" if torch.cuda.is_available() else "cpu"):
        model = model.to(device)
        for epoch in range(1, epochs+1):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
            
                optimizer.zero_grad()
                logits = model(inputs)
            
                ce_loss = criterion(logits, labels)
                total_loss += ce_loss.item()
            
                ce_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
        
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | "
                    f"CE Loss: {total_loss/len(train_loader):.4f} | "
                    f"Train Acc: {train_acc:.2f}%")
