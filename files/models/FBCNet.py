import torch
import torch.nn as nn

import numpy as np
from scipy.signal import butter, sosfilt

"""
used FBCNet combination for comparisons:

Not working

FBCNet(nb_classes=4,
        Chans=22,
        Samples=500,)
"""


class FBCNet(nn.Module):
    """
    FBCNet: Filter Bank Convolutional Network for EEG classification.

    Args:
        Chans       : number of EEG channels
        Samples     : number of time points
        freq_bands  : list of (low_hz, high_hz) tuples defining the filter bank
        sfreq       : sampling frequency of the EEG signal (Hz)
        F1          : number of spatial convolution filters per frequency band
        w           : temporal variance window size (number of windows to split T into)
        nb_classes  : number of output classes (N_c)
    """

    def __init__(
        self,
        Chans,
        Samples,
        freq_bands= [(i, i+4) for i in range(4, 37, 4)],
        sfreq = 250.0,
        F1 = 32,
        w = 8,
        nb_classes = 4,
        max_norm_spatial = 2.0,
        max_norm_fc = 0.5,
    ):
        super().__init__()

        assert Samples % w == 0, f"Samples ({Samples}) must be divisible by w ({w})"

        self.Chans = Chans
        self.Samples = Samples
        self.sfreq = sfreq
        self.w = w
        self.nb_classes = nb_classes
        self.max_norm_spatial = max_norm_spatial
        self.max_norm_fc = max_norm_fc

        # Pre-compute and store SOS filter coefficients for each band (not nn.Parameters)
        self.freq_bands = freq_bands
        self.N_b = len(freq_bands)
        self._build_filters()

        self.spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=self.N_b,
                out_channels=self.N_b * F1,
                kernel_size=(Chans, 1),
                groups=self.N_b,
                bias=False,
            ),
            nn.BatchNorm2d(self.N_b * F1),
            nn.ELU(),
        )
        
        self.fc = nn.Linear(self.N_b * F1 * w, nb_classes)


    def _build_filters(self):
        """Build 4th-order Butterworth bandpass SOS filters."""
        sos_list = []
        for (low, high) in self.freq_bands:
            sos = butter(4, [low, high], btype="bandpass", fs=self.sfreq, output="sos")
            sos_list.append(sos)
        # Store as plain Python list (not a tensor); used in forward via scipy
        self._sos_filters = sos_list


    def _bandpass_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the filter bank to the raw EEG signal.

        Args:
            x: Raw EEG tensor of shape (C, T)  [single sample, called per-sample]
               OR (batch, C, T) for batch processing.

        Returns:
            Filtered tensor of shape (batch, N_b, C, T)
        """
        # Accept (C, T) or (batch, C, T)
        if x.dim() == 2:
            x = x.unsqueeze(0)               # → (1, C, T)

        batch, Chans, Samples = x.shape
        device = x.device
        x_np = x.cpu().numpy()               # scipy works on numpy

        out = np.zeros((batch, self.N_b, Chans, Samples), dtype=np.float32)
        for b_idx, sos in enumerate(self._sos_filters):
            filtered = sosfilt(sos, x_np, axis=-1).astype(np.float32)  # (batch, C, T)
            out[:, b_idx, :, :] = filtered

        return torch.from_numpy(out).to(device)  # (batch, N_b, C, T)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG input of shape (C, T) or (batch, C, T)

        Returns:
            Class logits of shape (batch, N_c)
        """
        # ── 1. Bandpass filtering & concatenation ──────────────────────
        # x_filt: (batch, N_b, C, T)
        x = self._bandpass_filter(x)

        # ── 2. Spatial Convolution ─────────────────────────────────────
        # Conv2d expects (batch, in_channels, H, W)
        # Here: in_channels = N_b, H = C, W = T
        x = self.spatial(x)

        # ── 3. Temporal Variance Pooling ───────────────────────────────
        # Reshape T → (w, window_len) then take variance over window_len
        batch, NbM, one, T = x.shape
        x = x.reshape(batch, NbM, self.w, int(T/self.w))
        x = x.var(dim=-1, keepdim=True)                                 # (batch, N_b*m, w)

        # ── 4. Flatten & FC ───────────────────────────────────────────
        x = x.flatten(start_dim=1)        # (batch, N_b*m*w)
        x = self.fc(x)                    # (batch, N_c)

        return x


    @staticmethod
    def training(
        model,
        criterion,
        optimizer,
        lr,
        train_loader,
        max_epochs_stage1 = 1500,
        max_epochs_stage2 = 600,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        model = model.to(device)
        for epoch in range(1, max_epochs_stage1+1):
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
                model.apply_max_norm()
            
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_acc = 100 * correct / total
            train_loss = total_loss / len(train_loader)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{max_epochs_stage1}] | "
                    f"CE Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.2f}%")



import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

import numpy as np
from scipy.signal import cheby2, sosfilt

"""
used FBCNet combination for comparisons:

FBCNet(nb_classes=4,
        Chans=22,
        Samples=500,)
"""


class FBCNet(nn.Module):
    """
    FBCNet: Filter Bank Convolutional Network for EEG classification.

    Args:
        Chans            : number of EEG channels
        Samples          : number of time points
        freq_bands       : list of (low_hz, high_hz) tuples defining the filter bank
        sfreq            : sampling frequency of the EEG signal (Hz)
        F1               : number of spatial convolution filters per frequency band (m in the paper)
        w                : temporal variance window size (number of windows to split T into)
        nb_classes       : number of output classes (N_c)
        max_norm_spatial : max L2 weightnorm constraint on the spatial conv kernel (paper: 2)
        max_norm_fc      : max L2 weightnorm constraint on the FC layer weights (paper: 0.5)
    """

    def __init__(
        self,
        Chans,
        Samples,
        freq_bands= [(i, i+4) for i in range(4, 37, 4)],
        sfreq = 250.0,
        F1 = 32,
        w = 8,
        nb_classes = 4,
        max_norm_spatial = 2.0,
        max_norm_fc = 0.5,
    ):
        super().__init__()

        assert Samples % w == 0, f"Samples ({Samples}) must be divisible by w ({w})"

        self.Chans = Chans
        self.Samples = Samples
        self.sfreq = sfreq
        self.w = w
        self.nb_classes = nb_classes
        self.max_norm_spatial = max_norm_spatial
        self.max_norm_fc = max_norm_fc

        # Pre-compute and store SOS filter coefficients for each band (not nn.Parameters)
        self.freq_bands = freq_bands
        self.N_b = len(freq_bands)
        self._build_filters()

        self.spatial = nn.Sequential(
            nn.Conv2d(
                in_channels=self.N_b,
                out_channels=self.N_b * F1,
                kernel_size=(Chans, 1),
                groups=self.N_b,
                bias=False,
            ),
            nn.BatchNorm2d(self.N_b * F1),
            nn.ELU(),
        )

        self.fc = nn.Linear(self.N_b * F1 * w, nb_classes)


    def _build_filters(self):
        """Build 4th-order Chebyshev Type II bandpass SOS filters (2Hz transition band, -30dB stopband ripple)."""
        sos_list = []
        for (low, high) in self.freq_bands:
            sos = cheby2(4, 30, [low, high], btype="bandpass", fs=self.sfreq, output="sos")
            sos_list.append(sos)
        # Store as plain Python list (not a tensor); used in forward via scipy
        self._sos_filters = sos_list


    def _bandpass_filter(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the filter bank to the raw EEG signal.

        Args:
            x: Raw EEG tensor of shape (C, T)  [single sample, called per-sample]
               OR (batch, C, T) for batch processing.

        Returns:
            Filtered tensor of shape (batch, N_b, C, T)
        """
        # Accept (C, T) or (batch, C, T)
        if x.dim() == 2:
            x = x.unsqueeze(0)               # → (1, C, T)

        batch, Chans, Samples = x.shape
        device = x.device
        x_np = x.cpu().numpy()               # scipy works on numpy

        out = np.zeros((batch, self.N_b, Chans, Samples), dtype=np.float32)
        for b_idx, sos in enumerate(self._sos_filters):
            filtered = sosfilt(sos, x_np, axis=-1).astype(np.float32)  # (batch, C, T)
            out[:, b_idx, :, :] = filtered

        return torch.from_numpy(out).to(device)  # (batch, N_b, C, T)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG input of shape (C, T) or (batch, C, T)

        Returns:
            Class logits of shape (batch, N_c)
        """
        # ── 1. Bandpass filtering & concatenation ──────────────────────
        # x_filt: (batch, N_b, C, T)
        x = self._bandpass_filter(x)

        # ── 2. Spatial Convolution ─────────────────────────────────────
        # Conv2d expects (batch, in_channels, H, W)
        # Here: in_channels = N_b, H = C, W = T
        x = self.spatial(x)

        # ── 3. Temporal Variance Pooling ───────────────────────────────
        # Reshape T → (w, window_len) then take variance over window_len
        batch, NbM, one, T = x.shape
        x = x.reshape(batch, NbM, self.w, T // self.w)
        x = x.var(dim=-1, keepdim=True)                                 # (batch, N_b*m, w)

        # ── 4. Log activation, flatten & FC ─────────────────────────────
        # Paper: variance features pass through a log activation before the FC layer
        x = torch.log(x + 1e-6)
        x = x.flatten(start_dim=1)        # (batch, N_b*m*w)
        x = self.fc(x)                    # (batch, N_c)

        return x


    def apply_max_norm(self):
        """
        Renormalize the spatial conv and FC weights to their max L2 norm constraint.

        Paper specifies ||w||2 < 2 on the depthwise spatial conv kernels and
        ||w||2 < 0.5 on the FC layer weights (Table S1). PyTorch has no
        built-in max-norm constraint (unlike Keras' kernel_constraint), so
        this has to be called manually after every optimizer.step().
        """
        self._renorm(self.spatial[0].weight, self.max_norm_spatial)
        self._renorm(self.fc.weight, self.max_norm_fc)


    @staticmethod
    def _renorm(weight, max_norm):
        with torch.no_grad():
            norm = weight.norm(2, dim=tuple(range(1, weight.dim())), keepdim=True)
            desired = torch.clamp(norm, max=max_norm)
            weight.mul_(desired / (norm + 1e-8))


    @staticmethod
    def _evaluate(model, criterion, loader, device):
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)

                loss = criterion(logits, labels)
                total_loss += loss.item()

                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total_loss / len(loader), 100 * correct / total


    @staticmethod
    def training(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        max_epochs_stage1 = 1500,
        max_epochs_stage2 = 600,
        patience = 200,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Two-stage training procedure with early stopping, as described in the paper (Sec. III.D).

        Stage 1: train on train_loader only. Validation accuracy is monitored every
                 epoch; if there is no improvement for `patience` consecutive epochs,
                 training stops and the weights with the best validation accuracy
                 are restored.
        Stage 2: continue training on train_loader + val_loader combined. Training
                 stops once the validation loss drops below the stage 1 training
                 loss recorded at the best-val-accuracy checkpoint.

        Args:
            train_loader      : DataLoader for the training-only split
            val_loader         : DataLoader for the held-out validation split
            max_epochs_stage1 : hard cap on stage 1 epochs (paper: 1500)
            max_epochs_stage2 : hard cap on stage 2 epochs (paper: 600)
            patience           : early stopping patience in epochs (paper: 200)
        """
        model = model.to(device)

        # ── Stage 1: train-only, early stopping on validation accuracy ─────
        best_val_acc = 0.0
        best_state = None
        epochs_since_improvement = 0
        stage1_train_loss = None

        for epoch in range(1, max_epochs_stage1 + 1):
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
                model.apply_max_norm()

                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = total_loss / len(train_loader)
            train_acc = 100 * correct / total
            val_loss, val_acc = FBCNet._evaluate(model, criterion, val_loader, device)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                stage1_train_loss = train_loss
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            if epoch % 10 == 0 or epoch == 1:
                print(f"[Stage 1] Epoch [{epoch}/{max_epochs_stage1}] | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                    f"Val Acc: {val_acc:.2f}% (best: {best_val_acc:.2f}%)")

            if epochs_since_improvement >= patience:
                print(f"[Stage 1] Early stopping at epoch {epoch} "
                    f"(no val improvement for {patience} epochs)")
                break

        # Restore the best validation-accuracy weights found in stage 1
        if best_state is not None:
            model.load_state_dict(best_state)

        # ── Stage 2: train + val combined, stop once val loss drops ────────
        #            below the stage 1 training loss ──────────────────────
        combined_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
        combined_loader = DataLoader(
            combined_dataset, batch_size=train_loader.batch_size, shuffle=True
        )

        for epoch in range(1, max_epochs_stage2 + 1):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for inputs, labels in combined_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                logits = model(inputs)

                ce_loss = criterion(logits, labels)
                total_loss += ce_loss.item()

                ce_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                model.apply_max_norm()

                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = 100 * correct / total
            val_loss, val_acc = FBCNet._evaluate(model, criterion, val_loader, device)

            if epoch % 10 == 0 or epoch == 1:
                print(f"[Stage 2] Epoch [{epoch}/{max_epochs_stage2}] | "
                    f"Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.2f}% (stage 1 train loss: {stage1_train_loss:.4f})")

            if stage1_train_loss is not None and val_loss < stage1_train_loss:
                print(f"[Stage 2] Stopping at epoch {epoch} "
                    f"(val loss {val_loss:.4f} < stage 1 train loss {stage1_train_loss:.4f})")
                break

        return model
