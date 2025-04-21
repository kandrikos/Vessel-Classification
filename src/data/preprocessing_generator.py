import glob
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from preprocessing import get_preprocessing_layer
from scipy.io import wavfile
from tqdm import tqdm


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def load_and_preprocess_audio(file_path, target_sample_rate):
    # Load the audio file using scipy
    sample_rate, waveform = wavfile.read(file_path)

    # Convert to float32 and normalize if integer data
    if waveform.dtype.kind in "iu":
        waveform = waveform.astype(np.float32)
        waveform /= np.iinfo(waveform.dtype).max

    # Convert to mono if stereo
    if len(waveform.shape) > 1 and waveform.shape[1] > 1:
        waveform = np.mean(waveform, axis=1)

    # Convert to tensorflow tensor
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)

    # Resample if necessary
    if sample_rate != target_sample_rate:
        waveform = tf.sparse.to_dense(
            tf.sparse.reorder(
                tf.audio.resample(waveform, sample_rate, target_sample_rate)
            )
        )

    # Add channel dimension if needed
    if len(waveform.shape) == 1:
        waveform = tf.expand_dims(waveform, axis=0)

    return waveform


def generate_dataset_artifacts(root_path, target_sample_rate=32000):
    audio_dir = root_path / "audio"
    ship_types = [os.path.basename(d) for d in glob.glob(str(audio_dir / "*"))]

    # Store shapes for analysis
    shapes = {"tfmel": set(), "stft": set(), "logscale": set()}

    # Process first file of first ship type to get shapes
    first_ship = ship_types[0]
    first_audio = glob.glob(str(audio_dir / first_ship / "*.wav"))[0]
    print(f"\nAnalyzing shapes using file: {first_audio}")

    waveform = load_and_preprocess_audio(first_audio, target_sample_rate)

    for preprocessing in ["tfmel", "stft", "logscale"]:
        transform_fn = get_preprocessing_layer(preprocessing)
        if preprocessing == "tfmel":
            spectrogram = transform_fn(waveform, target_sample_rate)
        else:
            spectrogram = transform_fn(waveform)

        spec_shape = spectrogram.numpy().shape
        print(f"{preprocessing} spectrogram shape: {spec_shape}")

    print("\nProceeding with dataset generation...")

    # Create directories for each preprocessing type
    for preprocessing in ["tfmel", "stft", "logscale"]:
        print(f"\nStarting with {preprocessing} spectrograms...")

        # Create the output directory
        preprocessing_dir = root_path / preprocessing
        create_dir(str(preprocessing_dir))

        # Get the appropriate preprocessing function
        transform_fn = get_preprocessing_layer(preprocessing)

        for ship in ship_types:
            print(f"Generating data from {ship}")
            ship_dir = create_dir(str(preprocessing_dir / ship))
            audio_files = [Path(d) for d in glob.glob(str(audio_dir / ship / "*.wav"))]

            # Count files that already exist
            existing_files = sum(
                1
                for audio in audio_files
                if os.path.exists(Path(ship_dir) / f"{audio.stem}.npy")
            )
            total_files = len(audio_files)

            if existing_files > 0:
                print(
                    f"Found {existing_files}/{total_files} already processed files. Skipping these files."
                )

            for audio in tqdm(audio_files):
                # Create the output file path
                file_path = Path(ship_dir) / f"{audio.stem}.npy"

                # Skip if the file already exists
                if os.path.exists(file_path):
                    continue

                try:
                    # Load and preprocess audio
                    waveform = load_and_preprocess_audio(str(audio), target_sample_rate)

                    # Generate spectrogram based on type
                    if preprocessing == "tfmel":
                        spectrogram = transform_fn(waveform, target_sample_rate)
                    else:
                        spectrogram = transform_fn(waveform)

                    # Convert to numpy
                    spectrogram_np = spectrogram.numpy()

                    # Keep track of shapes
                    shapes[preprocessing].add(spectrogram_np.shape)

                    # Save the array
                    np.save(file_path, spectrogram_np)

                except Exception as e:
                    print(f"Error processing {audio}: {str(e)}")
                    continue

    # Print summary of all shapes encountered
    print("\nShape summary:")
    for prep_type, shape_set in shapes.items():
        print(f"{prep_type} shapes encountered: {shape_set}")


def main():
    print("Generating preprocessed files for underwater acoustic dataset\n")

    # Hardcoded path to the dataset
    root_dir = (
        Path(__file__).resolve().parents[2]
        / "datasets"
        / "VTUAD"
        / "inclusion_2000_exclusion_4000"
    )

    # Generate dataset for each split
    for split in ["test", "validation", "train"]:
        print(f"\nGenerating the {split} dataset")
        split_path = root_dir / split
        if os.path.exists(split_path):
            generate_dataset_artifacts(split_path)
        else:
            print(f"Split path {split_path} does not exist, skipping...")


if __name__ == "__main__":
    main()
