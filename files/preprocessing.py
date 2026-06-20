import mne
import torch
import numpy as np
import seaborn as sns
import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

class BCICausalPreprocessor:
    def __init__(self, lowcut=8.0, highcut=30.0, fs=250):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.eog_weights = None  
        
        # Design the causal IIR filter (Butterworth)
        self.b, self.a = signal.butter(4, [self.lowcut, self.highcut], btype='bandpass', fs=self.fs)


    def fit_eog_regression(self, eeg_data, eog_data):
        """Calculates the regression weights to subtract EOG from EEG."""
        eog_with_bias = np.vstack([eog_data, np.ones(eog_data.shape[1])])
        eog_cov = np.linalg.pinv(eog_with_bias @ eog_with_bias.T)
        self.eog_weights = eog_cov @ eog_with_bias @ eeg_data.T
        return self.eog_weights


    def apply_eog_regression(self, eeg_data, eog_data):
        """Applies the learned weights to remove EOG artifacts."""
        if self.eog_weights is None:
            raise ValueError("Fit the preprocessor on training data first!")
            
        eog_with_bias = np.vstack([eog_data, np.ones(eog_data.shape[1])])
        eeg_clean = eeg_data - (self.eog_weights.T @ eog_with_bias)
        return eeg_clean


    def causal_filter(self, data):
        """Applies a strictly causal filter using scipy.signal.lfilter."""
        filtered_data = signal.lfilter(self.b, self.a, data, axis=-1)
        return filtered_data


    def process_file(self, filepath, is_training=True):
        """Loads a GDF file, handles NaNs, removes EOG, and causally filters."""
        raw = mne.io.read_raw_gdf(filepath, preload=True)
        data = raw.get_data()
        
        # Handle NaNs
        data = np.nan_to_num(data, nan=0.0)
        
        # Split EEG (first 22) and EOG (last 3)
        eeg_data = data[:22, :]
        eog_data = data[22:25, :]
        
        # EOG Artifact Removal
        if is_training:
            self.fit_eog_regression(eeg_data, eog_data)
        eeg_clean = self.apply_eog_regression(eeg_data, eog_data)
        
        # Causal Filtering
        eeg_filtered = self.causal_filter(eeg_clean)
        
        # Extract Events AND the event dictionary
        events, event_dict = mne.events_from_annotations(raw)
        
        return eeg_filtered, events, event_dict


    @staticmethod
    def generate_causal_windows(eeg_filtered, events, event_dict, window_size_sec=2.0, fs=250, is_training=True):
        window_samples = int(window_size_sec * fs)
        
        # Motor imagery typically happens 0.5s to 3.5s AFTER the cue
        # BCI Comp IV 2a cues are at t=2s of the trial structure, MI lasts until t=6s.
        offset_start = int(0.5 * fs) 
        offset_end = int(3.5 * fs)   
        stride = int(0.2 * fs) # 200ms stride (50 samples) drastically reduces redundancy
        
        total_samples = eeg_filtered.shape[1]
        mne_id_to_gdf_str = {v: k for k, v in event_dict.items()}
        
        if is_training:
            target_events = {'769': 0, '770': 1, '771': 2, '772': 3}
        else:
            target_events = {'783': -1} 
        
        X_windows, y_labels, time_indices = [], [], []
        
        for event in events:
            sample_idx = event[0]
            mne_event_id = event[2] 
            gdf_event_str = mne_id_to_gdf_str.get(mne_event_id, "")
            
            if gdf_event_str in target_events:
                label = target_events[gdf_event_str]
                
                # Start and end relative to the cue
                start_idx = sample_idx + offset_start
                end_idx = sample_idx + offset_end
                
                for t in range(start_idx + window_samples, end_idx, stride):
                    if t <= total_samples:
                        window = eeg_filtered[:, t - window_samples : t]
                        
                        if window.shape[1] == window_samples:
                            X_windows.append(window)
                            y_labels.append(label)
                            time_indices.append((t - start_idx) / fs)
                        
        return np.array(X_windows, dtype=np.float32), np.array(y_labels, dtype=np.int64), np.array(time_indices, dtype=np.float32)


    @staticmethod
    def evaluate(model, device, eval_loader, true_trial_labels, trial_mappings):
        model.eval()

        # Dictionary to accumulate logits per trial
        trial_logits_accumulator = {i: [] for i in range(len(true_trial_labels))}

        with torch.no_grad():
            window_idx = 0 
            for inputs, labels in eval_loader:
                inputs = inputs.to(device)
                logits = model(inputs)
                
                logits_cpu = logits.cpu().numpy()
                
                for batch_i in range(logits.size(0)):
                    current_trial = trial_mappings[window_idx]
                    trial_logits_accumulator[current_trial].append(logits_cpu[batch_i])
                    window_idx += 1

        # Aggregate and score
        eval_correct = 0
        all_true_labels = []
        all_pred_labels = []

        for trial_id, logit_list in trial_logits_accumulator.items():
            if len(logit_list) == 0:
                continue
                
            # Sum/Mean all window logits within the trial
            mean_logits = np.mean(logit_list, axis=0)
            predicted_class = np.argmax(mean_logits)
            true_class = true_trial_labels[trial_id]
            
            # Store for the confusion matrix
            all_true_labels.append(true_class)
            all_pred_labels.append(predicted_class)
            
            if predicted_class == true_class:
                eval_correct += 1

        final_holdout_acc = 100 * eval_correct / len(true_trial_labels)

        print(f"\n{'='*50}")
        print(f"🏆 OFFICIAL TRIAL-LEVEL EVALUATION RESULTS (T -> E)")
        print(f"{'='*50}")
        print(f"Target Subject File:  A01E.gdf")
        print(f"Total True Trials:    {len(true_trial_labels)}")
        print(f"Final True Test Acc:  {final_holdout_acc:.2f}%")
        print(f"{'='*50}\n")

        return final_holdout_acc, all_true_labels, all_pred_labels


    @staticmethod
    def visualize(final_holdout_acc, all_true_labels, all_pred_labels):
        class_names = ['Left Hand', 'Right Hand', 'Both Feet', 'Tongue']

        cm = confusion_matrix(all_true_labels, all_pred_labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, 
                    yticklabels=class_names,
                    annot_kws={"size": 14})
        
        plt.title(f'Holdout Confusion Matrix (Acc: {final_holdout_acc:.1f}%)', fontsize=14, pad=15)
        plt.xlabel('Predicted MI Class', fontsize=12, labelpad=10)
        plt.ylabel('True MI Class', fontsize=12, labelpad=10)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()