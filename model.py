import librosa
import soundfile as sf
import os


#############################################################################
# YOUR ANONYMIZATION MODEL
# ---------------------
# Should be implemented in the 'anonymize' function
# !! DO NOT MODIFY THE NAME OF THE FUNCTION !!
#
# If you trained a machine learning model you can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> SAVE YOUR PARAMETERS IN THE parameters/ DIRECTORY <!>
############################################################################

def anonymize(input_audio_path):  # <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
    """
    Anonymization algorithm

    Parameters
    ----------
    input_audio_path : str
        Path to the source audio file in one ".wav" format.

    Returns
    -------
    audio : numpy.ndarray, shape (samples,), dtype=np.float32
        The anonymized audio signal as a 1D NumPy array of type np.float32,
        which ensures compatibility with soundfile.write().
    sr : int
        The sample rate of the processed audio.
    """
    # Load the audio
    audio, sr = librosa.load(input_audio_path, sr=None)

    # Apply pitch shifting (+5 semitones)
    n_steps = 5  # Modify if needed
    audio_anonymized = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

    return audio_anonymized, sr


# Define dataset path
dataset_path = "evaluation_data/Enrollment"

# Iterate through all speakers
for speaker in os.listdir(dataset_path):
    speaker_path = os.path.join(dataset_path, speaker)

    if os.path.isdir(speaker_path):
        anonymized_dir = os.path.join(speaker_path, "anonymized")
        os.makedirs(anonymized_dir, exist_ok=True)

        # Process each .wav file in the speaker's directory
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                input_audio_path = os.path.join(speaker_path, file)

                # Apply anonymization
                anonymized_audio, sample_rate = anonymize(input_audio_path)

                # Save the anonymized file in the 'anonymized' subfolder
                output_audio_path = os.path.join(anonymized_dir, f"anon_{file}")
                sf.write(output_audio_path, anonymized_audio, sample_rate)

                print(f"Anonymized audio saved as: {output_audio_path}")