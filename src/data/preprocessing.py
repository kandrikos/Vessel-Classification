import tensorflow as tf

# Constants for spectrograms
FREQ_BINS = 95  # Number of frequency bins for all spectrogram types
TIME_STEPS = 126  # Fixed number of time steps we want
HOP_LENGTH = 256  # Used to generate an output of 128 on x axis
N_FFT = 2048
FMAX = 4186  # Maximum frequency
FMIN = 18.0  # Minimum frequency


def create_mel_spectrogram(waveform, sample_rate):
    """Creates a mel spectrogram with fixed dimensions.

    Args:
        waveform (tf.Tensor): The input waveform
        sample_rate (int): The sample rate of the audio

    Returns:
        tf.Tensor: The mel spectrogram of shape (FREQ_BINS, TIME_STEPS)
    """
    waveform = tf.cast(waveform, dtype=tf.float32)

    # Remove extra dimensions if present
    waveform = tf.squeeze(waveform)

    stft = tf.signal.stft(
        waveform,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
    )

    # Convert to magnitude spectrogram
    magnitude_spectogram = tf.abs(stft)

    # Create mel filterbank matrix
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=FREQ_BINS,
        num_spectrogram_bins=N_FFT // 2 + 1,
        sample_rate=sample_rate,
        lower_edge_hertz=FMIN,
        upper_edge_hertz=FMAX,
    )

    mel_spectogram = tf.matmul(tf.square(magnitude_spectogram), mel_matrix)
    mel_spectogram = tf.math.log(mel_spectogram + 1e-6)

    # Convert to (time_steps, freq_bins)
    mel_spectogram = tf.transpose(mel_spectogram)

    # Resize to desired time steps
    current_time_steps = tf.shape(mel_spectogram)[0]
    scale = TIME_STEPS / current_time_steps
    mel_spectogram = tf.image.resize(
        tf.expand_dims(mel_spectogram, -1), [TIME_STEPS, FREQ_BINS]
    )
    mel_spectogram = tf.squeeze(mel_spectogram)
    mel_spectogram = tf.transpose(mel_spectogram)

    return mel_spectogram


def create_stft_spectrogram(waveform):
    """Creates a STFT spectrogram with fixed dimensions.

    Args:
        waveform (tf.Tensor): The input waveform

    Returns:
        tf.Tensor: The STFT spectrogram of shape (FREQ_BINS, TIME_STEPS)
    """
    waveform = tf.cast(waveform, dtype=tf.float32)
    waveform = tf.squeeze(waveform)

    stft = tf.signal.stft(
        waveform,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
    )

    # Get magnitude spectrogram
    spectrogram = tf.abs(stft)
    spectrogram = tf.square(spectrogram)

    # Convert to (time_steps, freq_bins)
    spectrogram = tf.transpose(spectrogram)

    # Resize to desired dimensions
    spectrogram = tf.image.resize(
        tf.expand_dims(spectrogram, -1), [TIME_STEPS, FREQ_BINS]
    )
    spectrogram = tf.squeeze(spectrogram)
    spectrogram = tf.transpose(spectrogram)

    return spectrogram


def create_log_spectrogram(waveform):
    """Creates a log-scaled spectrogram with fixed dimensions.

    Args:
        waveform (tf.Tensor): The input waveform

    Returns:
        tf.Tensor: The log-scaled spectrogram of shape (FREQ_BINS, TIME_STEPS)
    """
    waveform = tf.cast(waveform, dtype=tf.float32)
    waveform = tf.squeeze(waveform)

    stft = tf.signal.stft(
        waveform,
        frame_length=N_FFT,
        frame_step=HOP_LENGTH,
        fft_length=N_FFT,
        window_fn=tf.signal.hann_window,
    )

    spectrogram = tf.abs(stft)
    spectrogram = tf.square(spectrogram)
    spectrogram = tf.math.log(spectrogram + 1e-6)

    # Convert to (time_steps, freq_bins)
    spectrogram = tf.transpose(spectrogram)

    # Resize to desired dimensions
    spectrogram = tf.image.resize(
        tf.expand_dims(spectrogram, -1), [TIME_STEPS, FREQ_BINS]
    )
    spectrogram = tf.squeeze(spectrogram)
    spectrogram = tf.transpose(spectrogram)

    return spectrogram


def get_preprocessing_layer(pre_processing_type):
    """Returns the appropriate preprocessing function based on the type.

    Args:
        pre_processing_type (str): The type of preprocessing to apply

    Returns:
        function: The preprocessing function
    """
    _pre_processing_layers = {
        "tfmel": create_mel_spectrogram,
        "stft": create_stft_spectrogram,
        "logscale": create_log_spectrogram,
    }

    return _pre_processing_layers[pre_processing_type.lower()]
