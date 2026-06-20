import torch
import torch.nn as nn

from Mindcracked.files.layers.layers import Conv2dWithConstraint, LinearWithConstraint
"""
used SCNN combination for comparisons:

SCNN(nb_classes=4,
        Chans=22,
        Samples=500,
        dropoutRate=0.5,
        )

overfits (100%)! 
"""


class SCNN(nn.Module):
    def __init__(
        self,
        Chans,
        Samples,
        nb_classes=2,
        dropoutRate=0.25,
        kernLength=13,
        pk1=35,
        pk2=7,
    ):
        super().__init__()

        self.layer1 = nn.Sequential(
            Conv2dWithConstraint(
                in_channels=1,
                out_channels=40,
                kernel_size=(1, kernLength),
                bias=True,
                padding='same',
                max_norm=2,
            ),

            Conv2dWithConstraint(
                in_channels=40,
                out_channels=40,
                kernel_size=(Chans, 1),
                bias=False,
                max_norm=2,
            ),

            nn.BatchNorm2d(40, eps=1e-05, momentum=0.1),
        )
        
        self.layer2 = nn.Sequential(
            nn.AvgPool2d((1,pk1), stride = (1,pk2)),
        )

        self.layer3 = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropoutRate),
            LinearWithConstraint(40 * (((Samples - pk1) // pk2) + 1), nb_classes, max_norm=0.5)
        )


    def forward(self, x):

        x = x.unsqueeze(1)
        x = self.layer1(x)
        x = torch.square(x)
        x = self.layer2(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.layer3(x)

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
