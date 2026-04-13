import numpy as np
from scipy.signal import firwin, filtfilt

try:
    import mne
except ImportError:
    mne = None


class EEGPreprocessor:
    """
    Pipeline:
    1. Load raw EEG -> MNE Raw object
    2. Bandpass filter 1-60 Hz
    3. ICA artifact removal (EOG, EMG)
    4. Segment into fixed-length windows (e.g., 10s = 2560 samples)
    5. Extract sub-band signals via FIR filters:
       - delta: 1-4 Hz
       - theta: 4-8 Hz
       - alpha: 8-13 Hz
       - beta: 13-30 Hz
       - gamma: 30-60 Hz
    6. Output shape: (n_samples, n_channels, n_subbands, n_timepoints)
       i.e., (N, 7, 5, T)
    """

    ELECTRODES = ['FP1', 'FP2', 'F7', 'F3', 'Fz', 'F4', 'F8']
    SUBBANDS = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 60),
    }

    def __init__(self, sfreq=256, segment_sec=10):
        self.sfreq = sfreq
        self.segment_sec = segment_sec
        self.segment_len = sfreq * segment_sec

    def load_and_filter(self, filepath: str):
        """Load raw EEG and apply 1-60 Hz bandpass."""
        if mne is None:
            raise ImportError("mne is required for raw EEG loading. Install with: pip install mne")
        raw = mne.io.read_raw(filepath, preload=True)
        raw.filter(l_freq=1.0, h_freq=60.0, method='fir')
        return raw

    def run_ica(self, raw, n_components=15):
        """ICA artifact removal."""
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=42)
        ica.fit(raw)
        eog_indices, _ = ica.find_bads_eog(raw, threshold=3.0)
        ica.exclude = eog_indices
        raw_clean = ica.apply(raw.copy())
        return raw_clean

    def extract_subbands(self, data: np.ndarray) -> np.ndarray:
        """
        data: (n_channels, n_timepoints)
        returns: (n_channels, n_subbands, n_timepoints)
        """
        result = []
        for low, high in self.SUBBANDS.values():
            coeffs = firwin(101, [low, high], pass_zero=False, fs=self.sfreq)
            filtered = filtfilt(coeffs, 1.0, data, axis=-1)
            result.append(filtered)
        return np.stack(result, axis=1)  # (ch, 5, T)

    def segment(self, data: np.ndarray) -> np.ndarray:
        """Segment continuous data into fixed-length windows."""
        n_segments = data.shape[-1] // self.segment_len
        segments = []
        for i in range(n_segments):
            start = i * self.segment_len
            end = start + self.segment_len
            segments.append(data[..., start:end])
        return np.stack(segments, axis=0)  # (N, ch, subbands, T)

    def process(self, filepath: str) -> np.ndarray:
        """Full pipeline: load -> filter -> ICA -> subband -> segment."""
        raw = self.load_and_filter(filepath)
        raw_clean = self.run_ica(raw)
        picks = mne.pick_channels(raw_clean.ch_names, self.ELECTRODES)
        data = raw_clean.get_data(picks=picks)  # (7, total_T)
        subbands = self.extract_subbands(data)   # (7, 5, total_T)
        segments = self.segment(subbands)         # (N, 7, 5, T)
        return segments

    def process_numpy(self, data: np.ndarray) -> np.ndarray:
        """Process pre-loaded numpy array (7, total_T) -> (N, 7, 5, T)."""
        subbands = self.extract_subbands(data)
        segments = self.segment(subbands)
        return segments
