import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() #sets the matplotlib style to seaborn style

from scipy.io import loadmat 
from scipy.ndimage import convolve1d
from scipy.signal import butter
from scipy.signal import sosfiltfilt
from scipy.signal import welch
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import euclidean
import numpy as np

np.random.seed(42)

import plotly.express as px


def emg_envelope(emg_rectified, stimulus, repetition, window_size = 25, n_channels = 10, n_stimuli = 12, n_repetitions = 10):

    #defining the length of the moving average window
    window_size = 25
    mov_mean_weights = np.ones(window_size) / window_size
    
    #initializing the data structure
    emg_windows = [[None for repetition_idx in range(n_repetitions)] for stimuli_idx in range(n_stimuli)]
    emg_envelopes = [[None for repetition_idx in range(n_repetitions)] for stimuli_idx in range(n_stimuli)]

    for stimuli_idx in range(n_stimuli):
        for repetition_idx in range(n_repetitions):
            idx = np.logical_and(stimulus == stimuli_idx + 1, repetition == repetition_idx + 1).flatten() #generate a boolean mask that identifies the time points in the dataset
                                                                                                        #where both the stimulus and repetition conditions are satisfied
            emg_windows[stimuli_idx][repetition_idx] = emg_rectified[idx, :] #for all channels 
            emg_envelopes[stimuli_idx][repetition_idx] = convolve1d(emg_windows[stimuli_idx][repetition_idx], mov_mean_weights, axis=0) 

    return emg_envelopes

#emg_average_activations has data for emg_average_activations[:, stimuli_idx, repetition_idx] 
#create a function that analysis each stimuli individually and returns repetition index of an trial where the signal is significantly diffrent from other trials 
#first approach see how differnet the mean signal is 

def compute_emg_average_activations(emg_envelopes, n_channels = 10, n_stimuli = 12, n_repetitions = 10):
    emg_average_activations = np.zeros((n_channels, n_stimuli, n_repetitions))
    for stimuli_idx in range(n_stimuli):
        for repetition_idx in range(n_repetitions):
            #mean across time for each channel
            emg_average_activations[:, stimuli_idx, repetition_idx] = np.mean(emg_envelopes[stimuli_idx][repetition_idx], axis=0) 
    return emg_average_activations



def trial_to_exclude(emg_average_activations, stimuli_idx, threshold_factor=1.5, use_iqr=True):
    """
    Identify trials where the channel distribution deviates significantly from others,
    using either IQR or MAD for outlier detection.

    Args:
        emg_average_activations (np.ndarray): 3D array (n_channels, n_stimuli, n_repetitions).
        stimuli_idx (int): Index of the stimulus to analyze (0-based).
        threshold_factor (float): Multiplier for the threshold (default: 1.5).
        use_iqr (bool): Whether to use IQR (True) or MAD (False) for outlier detection.

    Returns:
        list: Indices of trials to exclude for the specified stimulus.
    """
    # Extract data for the given stimulus (shape: n_channels x n_repetitions)
    data_from_stimuli = emg_average_activations[:, stimuli_idx, :]
    
    # Compute the median pattern across repetitions
    median_pattern = np.median(data_from_stimuli, axis=1)  # Shape: (n_channels,)

    # Compute distances of each trial's pattern from the median pattern
    distances = np.array([euclidean(data_from_stimuli[:, r], median_pattern)
                          for r in range(data_from_stimuli.shape[1])])

    if use_iqr:
        # IQR-based outlier detection
        Q1 = np.percentile(distances, 25)  # First quartile
        Q3 = np.percentile(distances, 75)  # Third quartile
        IQR = Q3 - Q1

        # Define the threshold for outliers
        lower_bound = Q1 - threshold_factor * IQR
        upper_bound = Q3 + threshold_factor * IQR

        # Identify trials that fall outside the bounds
        trials_to_exclude = np.where((distances < lower_bound) | (distances > upper_bound))[0]
    else:
        # MAD-based outlier detection
        mad_distance = np.median(np.abs(distances - np.median(distances)))
        threshold = np.median(distances) + threshold_factor * mad_distance

        # Identify trials exceeding the threshold
        trials_to_exclude = np.where(distances > threshold)[0]

    return trials_to_exclude


def trial_to_exclude_all(emg_average_activations, threshold_factor=1.5, use_iqr=True, print_list = False):
    """
    Apply the exclusion method (IQR or MAD) to all stimuli.

    Args:
        emg_average_activations (np.ndarray): 3D array (n_channels, n_stimuli, n_repetitions).
        threshold_factor (float): Multiplier for the threshold (default: 1.5).
        use_iqr (bool): Whether to use IQR (True) or MAD (False) for outlier detection.

    Returns:
        list: A list where each element contains the indices of excluded trials for each stimulus.
    """
    n_stimuli = emg_average_activations.shape[1]
    exclude_list = []

    for stimuli_idx in range(n_stimuli):
        trials_to_exclude = trial_to_exclude(emg_average_activations, stimuli_idx, threshold_factor, use_iqr)
        exclude_list.append(trials_to_exclude)
        if print_list:
            print(f"Trials to exclude for Stimulus {stimuli_idx + 1}: {trials_to_exclude + 1}")  # 1-based indexing

    return exclude_list

def plot_heatmap(emg_average_activations, n_stimuli = 12):
    fig, ax = plt.subplots(6,2, figsize=(30, 25), constrained_layout=True, sharex=True, sharey=True)
    ax = ax.ravel()

    for stimuli_idx in range(n_stimuli):
        sns.heatmap(np.squeeze(emg_average_activations[:, stimuli_idx, :]), ax=ax[stimuli_idx] ,xticklabels=False, yticklabels=False, cbar = True)
        ax[stimuli_idx].title.set_text("Stimulus " + str(stimuli_idx + 1))
        ax[stimuli_idx].set_xlabel("Repetition")
        ax[stimuli_idx].set_ylabel("EMG channel")


def mean_absolute_value(trial):
    """Mean absolute value (MAV)."""
    return np.mean(np.abs(trial), axis=0)

def root_mean_square(trial):
    """Root mean square (RMS)."""
    return np.sqrt(np.mean(trial**2, axis=0))

def waveform_length(trial):
    """Waveform length (WL)."""
    return np.sum(np.abs(np.diff(trial, axis=0)), axis=0)


def slope_sign_changes(trial, threshold=0.01):
    """Slope sign changes (SSC)."""
    diff1 = np.diff(trial, axis=0)[:-1]
    diff2 = np.diff(trial, axis=0)[1:]
    return np.sum((diff1 * diff2 < 0) & (np.abs(diff1) > threshold), axis=0)

def variance(trial):
    """Variance of the signal."""
    return np.var(trial, axis=0)

def mean_frequency(trial):
    """Mean frequency (MF)."""
    freq = np.fft.rfft(trial, axis=0)  # Perform FFT per channel
    power = np.abs(freq) ** 2
    freqs = np.fft.rfftfreq(trial.shape[0])
    return np.sum(freqs[:, None] * power, axis=0) / np.sum(power, axis=0)

def maximum_amplitude(trial):
    """Maximum amplitude (MAX)."""
    return np.max(trial, axis=0)

def build_dataset_from_ninapro(processed_emg, features=None, exclude_list=None):
    """
    Builds a dataset for machine learning by processing the pre-processed EMG envelopes.

    Parameters:
    - processed_emg: List of lists containing the pre-processed EMG envelopes for each stimulus and repetition.
    - stimulus: 1D array with stimulus labels corresponding to the EMG data.
    - repetition: 1D array with repetition labels corresponding to the EMG data.
    - features: List of feature functions to extract from the EMG data.
    - exclude_list: List of lists of trials to exclude for each stimulus.

    Returns:
    - dataset: 2D array of extracted features.
    - labels: 1D array of labels for each sample.
    """
    #emg_envelopes[1][5][:, channel_idx]


    n_stimuli = len(processed_emg)
    n_repetitions = len(processed_emg[0]) if n_stimuli > 0 else 0
    n_channels = processed_emg[0][0].shape[1] if n_stimuli > 0 and n_repetitions > 0 else 0
    n_features = len(features) * n_channels  # Total number of features (features * channels)

    dataset = []
    labels = []
    
    for stimuli_idx in range(n_stimuli):
        for repetition_idx in range(n_repetitions):
            # Skip excluded trials
            if exclude_list and repetition_idx in exclude_list[stimuli_idx]:
                continue

            # Retrieve the pre-processed EMG envelope for the current stimulus and repetition
            emg_data = processed_emg[stimuli_idx][repetition_idx]

            # Compute features for this trial
            trial_features = []
            for feature_func in features:
                trial_features.extend(feature_func(emg_data))  # Compute feature for all channels

            # Append features and corresponding label
            dataset.append(trial_features)
            labels.append(stimuli_idx + 1)  # Labels are 1-based (stimuli index + 1)

    # Convert lists to numpy arrays for consistency
    dataset = np.array(dataset)
    labels = np.array(labels)

    return dataset, labels

def plot_features_by_stimulus_and_metric(dataset, labels, n_stimuli, n_repetitions, n_channels):
    """
    Visualize all repetitions of a specific metric for each stimulus in a compact layout.

    Parameters:
        dataset: 2D numpy array (n_trials, n_features * n_channels).
        labels: 1D numpy array of labels corresponding to each trial.
        n_stimuli: Number of stimuli.
        n_repetitions: Number of repetitions per stimulus.
        n_channels: Number of EMG channels.
    """
    # Number of metrics (features) per channel
    n_features = dataset.shape[1] // n_channels

    # Feature names
    feature_names = [
        "Mean Absolute Value", "Root Mean Square", "Waveform Length",
        "Slope Sign Changes", "Variance", "Mean Frequency", "Maximum Amplitude"
    ]
    if n_features > len(feature_names):
        feature_names = [f"Feature {i+1}" for i in range(n_features)]

    # Reshape dataset to (n_trials, n_features, n_channels)
    reshaped_dataset = dataset.reshape(-1, n_features, n_channels)

    # Iterate through stimuli
    for stimuli_idx in range(n_stimuli):
        n_rows = (n_features + 1) // 2  # Calculate the number of rows for 2 columns
        fig, axes = plt.subplots(
            n_rows, 2, figsize=(14, n_rows * 3), sharex=True, constrained_layout=True
        )
        axes = axes.flatten()

        # Select trials corresponding to the current stimulus
        stimulus_trials = np.where(labels == stimuli_idx + 1)[0]
        stimulus_data = reshaped_dataset[stimulus_trials]

        # Iterate through each metric (feature)
        for feature_idx in range(n_features):
            ax = axes[feature_idx]
            feature_data = stimulus_data[:, feature_idx, :]  # Extract data for the current feature

            # Plot all repetitions for the current feature
            for repetition_idx in range(n_repetitions):
                repetition_data = feature_data[repetition_idx, :]
                ax.plot(
                    range(1, n_channels + 1),  # Channel indices
                    repetition_data,
                    label=f"Repetition {repetition_idx + 1}",
                    alpha=0.7
                )

            ax.set_title(f"{feature_names[feature_idx]} (Stimulus {stimuli_idx + 1})")
            ax.set_ylabel("Feature Value")
            ax.set_xlabel("Channel")
            ax.legend(loc="upper right", fontsize=6)

        # Hide unused axes if n_features is odd
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(f"Features for Stimulus {stimuli_idx + 1}", fontsize=16)
        plt.show()

def plot_feature_subjects(dataset, labels, subject_ids, n_stimuli, n_channels, feature_names):
    """
    Plot the mean feature value per channel for each stimulus, comparing different subjects.
    The plots will be arranged in columns of 3.
    
    Args:
        dataset: 2D numpy array (n_samples, n_features * n_channels).
        labels: 1D numpy array of stimulus labels corresponding to each trial.
        subject_ids: 1D numpy array of subject IDs corresponding to each trial.
        n_stimuli: Number of unique stimuli.
        n_channels: Number of EMG channels.
        feature_names: List of feature names corresponding to the features in the dataset.
    """
    n_features = dataset.shape[1] // n_channels  # Number of features per channel

    # Reshape dataset to (n_samples, n_features, n_channels)
    reshaped_dataset = dataset.reshape(-1, n_features, n_channels)

    # Iterate through each stimulus
    for stimuli_idx in range(n_stimuli):
        # Select data for the current stimulus
        stimulus_trials = np.where(labels == stimuli_idx + 1)[0]
        stimulus_data = reshaped_dataset[stimulus_trials]
        stimulus_subjects = subject_ids[stimulus_trials]

        # Create a subplot grid (with 3 columns)
        n_rows = (n_features + 2) // 3  # Calculate number of rows needed for 3 columns
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 5), sharex=True, constrained_layout=True)
        axes = axes.flatten()  # Flatten the axes array to make indexing easier

        # Iterate through each feature
        for feature_idx, feature_name in enumerate(feature_names):
            ax = axes[feature_idx]

            # Calculate the mean per channel for each subject
            for subject_id in np.unique(subject_ids):
                # Select data for the current subject
                subject_trials = np.where(stimulus_subjects == subject_id)[0]
                if len(subject_trials) > 0:
                    subject_data = stimulus_data[subject_trials, feature_idx, :]
                    mean_per_channel = np.mean(subject_data, axis=0)

                    # Plot the subject's mean feature value per channel
                    ax.plot(
                        range(1, n_channels + 1),  # Channel indices
                        mean_per_channel,
                        label=f"Subject {subject_id}",
                        alpha=0.7,
                    )

            # Configure the plot
            ax.set_title(f"{feature_name} (Stimulus {stimuli_idx + 1})")
            ax.set_xlabel("Channel")
            ax.set_ylabel("Mean Feature Value")
            ax.set_xticks(range(1, n_channels + 1))
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(alpha=0.3)

        # Hide unused axes if the number of features isn't a multiple of 3
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])

        # Show the plot
        plt.tight_layout()
        plt.show()

