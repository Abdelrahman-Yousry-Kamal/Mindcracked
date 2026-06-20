import torch
import torch.nn as nn

from Mindcracked.files.layers.layers import MaxNormLinear, Conv2dWithConstraint

"""
used EEGNet combination for comparisons:

EEGNet(nb_classes=4,
        Chans=22,
        Samples=500,
        dropoutRate=0.5,
        kernLength=22,
        F1=8,
        D=4,
        F2=32,
        norm_rate=0.25,
        dropoutType="Dropout",
        pk1=4,
        pk2=8,)
"""


class EEGNet(nn.Module):
    """
    PyTorch implementation of EEGNet.

    Original paper:
        Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for
        EEG-based Brain-Computer Interfaces", J. Neural Eng. 2018.

    Args:
        nb_classes  : number of output classes
        Chans       : number of EEG channels (electrodes)
        Samples     : number of time samples per trial
        dropoutRate : dropout probability
        kernLength  : length of the temporal convolution kernel (Block 1)
        F1          : number of temporal filters
        D           : depth multiplier for the depthwise conv
        F2          : number of pointwise filters (= F1 * D by convention)
        norm_rate   : max-norm constraint on the final Dense layer
        dropoutType : 'Dropout' or 'SpatialDropout2D'
    """
    def __init__(
        self,
        nb_classes,
        Chans=64,
        Samples=128,
        dropoutRate=0.5,
        kernLength=64,
        F1=8,
        D=2,
        F2=16,
        norm_rate=0.25,
        dropoutType="Dropout",
        pk1=4,
        pk2=8,
    ):
        super().__init__()

        # dropout type
        if dropoutType == "SpatialDropout2D":
            drop = nn.Dropout2d
        elif dropoutType == "Dropout":
            drop = nn.Dropout
        else:
            raise ValueError(
                "dropoutType must be 'Dropout' or 'SpatialDropout2D'."
            )

        # Legend:
        #   C  = number of channels
        #   T  = number of time samples
        #   F1 = number of temporal filters
        #   D  = depth multiplier
        #   F2 = number of pointwise filters
        #   N  = number of output classes

        # ---------------------------------------------------------------------------
        # Block | Layer          | # Filters | Size   | # Params           | Output 
        # ---------------------------------------------------------------------------
        # 1     | Input          |           |        |                    | (C, T)
        #       | Reshape        |           |        |                    | (1, C, T)
        #       | Conv2D         | F1        | (1,64) | 64 * F1            | (F1, C, T)
        #       | BatchNorm      |           |        | 2 * F1             | (F1, C, T)
        #       | DepthwiseConv2D| D * F1    | (C,1)  | C * D * F1         | (D*F1, 1, T)
        #       | BatchNorm      |           |        | 2 * D * F1         | (D*F1, 1, T)
        #       | Activation**   |           |        |                    | (D*F1, 1, T)
        #       | AveragePool2D  |           | (1,4)  |                    | (D*F1, 1, T//4)
        #       | Dropout*       |           |        |                    | (D*F1, 1, T//4)

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=F1,
                kernel_size=(1, kernLength),
                padding='same',
                bias=False,
            ),

            nn.BatchNorm2d(F1),

            Conv2dWithConstraint(
                in_channels=F1,
                out_channels=F1 * D,
                kernel_size=(Chans, 1),
                groups=F1,
                bias=False,
            ),

            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pk1)),
            drop(p=dropoutRate)
        )

        # ---------------------------------------------------------------------------
        # Block | Layer          | # Filters | Size   | # Params           | Output 
        # ---------------------------------------------------------------------------
        # 2     | SeparableConv2D| F2        | (1,16) | 16*D*F1 + F2*(D*F1)| (F2, 1, T//4)
        #       | BatchNorm      |           |        | 2 * F2             | (F2, 1, T//4)
        #       | Activation**   |           |        |                    | (F2, 1, T//4)
        #       | AveragePool2D  |           | (1,8)  |                    | (F2, 1, T//32)
        #       | Dropout*       |           |        |                    | (F2, 1, T//32)
        #       | Flatten        |           |        |                    | (F2 * (T//32))

        self.block2 = nn.Sequential(
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

            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pk2)),
            drop(p=dropoutRate)
        )

        self.flatten_size = F2 * (Samples // (pk1*pk2))

        # ---------------------------------------------------------------------------
        # Block | Layer          | # Filters | Size   | # Params           | Output 
        # ---------------------------------------------------------------------------
        #       | Dense          |           |        | N * (F2 * (T//32)) | N
        
        self.dense = MaxNormLinear(self.flatten_size, nb_classes, max_norm_val=0.25)
        self.norm_rate = norm_rate
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)


    def _apply_max_norm(self, weight, max_val):
        """Clip weight rows to have L2-norm ≤ max_val (mirrors Keras max_norm)."""
        with torch.no_grad():
            norms = weight.norm(dim=1, keepdim=True).clamp(min=1e-8)
            weight.copy_(weight * (norms.clamp(max=max_val) / norms))


    def forward(self, x):
        """
        Args:
            x : Tensor of shape (B, 1, Chans, Samples)
                i.e. batch * 1 channel * electrodes * time

        Returns:
            Tensor of shape (B, nb_classes) with softmax probabilities.
        """
        x = x.unsqueeze(1)
        x = self.block1(x)           # (B, F1, Chans, Samples)
        x = self.block2(x)
        x = x.flatten(start_dim=1) # (B, flatten_size)
        self._apply_max_norm(self.dense.weight, self.norm_rate)
        x = self.dense(x)

        return x


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
