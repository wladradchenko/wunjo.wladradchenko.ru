"""Module for separating audio sources using MDX architecture models."""

import os
import gc
import torch
import onnxruntime as ort
import numpy as np
import librosa
from tqdm import tqdm
from pydub import AudioSegment


def normalize(wave, max_peak=1.0):
    """Normalize audio waveform to a specified peak value.

    Args:
        wave (array-like): Audio waveform.
        max_peak (float): Maximum peak value for normalization.

    Returns:
        array-like: Normalized or original waveform.
    """
    maxv = np.abs(wave).max()
    if maxv > max_peak:
        wave *= max_peak / maxv

    return wave


def wave_to_spectrogram_no_mp(wave):
    spec = librosa.stft(wave, n_fft=2048, hop_length=1024)
    if spec.ndim == 1:
        spec = np.asfortranarray([spec, spec])
    return spec


def spectrogram_to_wave_no_mp(spec, n_fft=2048, hop_length=1024):
    wave = librosa.istft(spec, n_fft=n_fft, hop_length=hop_length)
    if wave.ndim == 1:
        wave = np.asfortranarray([wave, wave])
    return wave


def reduce_vocal_aggressively(X, y, softmask):
    v = X - y
    y_mag_tmp = np.abs(y)
    v_mag_tmp = np.abs(v)
    v_mask = v_mag_tmp > y_mag_tmp
    y_mag = np.clip(y_mag_tmp - v_mag_tmp * v_mask * softmask, 0, np.inf)
    return y_mag * np.exp(1.0j * np.angle(y))


def invert_audio(specs, invert_p=True):
    ln = min([specs[0].shape[2], specs[1].shape[2]])
    specs[0] = specs[0][:, :, :ln]
    specs[1] = specs[1][:, :, :ln]
    if invert_p:
        X_mag = np.abs(specs[0])
        y_mag = np.abs(specs[1])
        max_mag = np.where(X_mag >= y_mag, X_mag, y_mag)
        v_spec = specs[1] - max_mag * np.exp(1.0j * np.angle(specs[0]))
    else:
        specs[1] = reduce_vocal_aggressively(specs[0], specs[1], 0.2)
        v_spec = specs[0] - specs[1]
    return v_spec


def invert_stem(mixture, stem):
    mixture = wave_to_spectrogram_no_mp(mixture)
    stem = wave_to_spectrogram_no_mp(stem)
    output = spectrogram_to_wave_no_mp(invert_audio([mixture, stem]))
    return -output.T


class CommonSeparator:
    """
    This class contains the common methods and attributes common to all architecture-specific Separator classes.
    """

    ALL_STEMS = "All Stems"
    VOCAL_STEM = "Vocals"
    INST_STEM = "Instrumental"
    OTHER_STEM = "Other"
    BASS_STEM = "Bass"
    DRUM_STEM = "Drums"
    GUITAR_STEM = "Guitar"
    PIANO_STEM = "Piano"
    SYNTH_STEM = "Synthesizer"
    STRINGS_STEM = "Strings"
    WOODWINDS_STEM = "Woodwinds"
    BRASS_STEM = "Brass"
    WIND_INST_STEM = "Wind Inst"
    NO_OTHER_STEM = "No Other"
    NO_BASS_STEM = "No Bass"
    NO_DRUM_STEM = "No Drums"
    NO_GUITAR_STEM = "No Guitar"
    NO_PIANO_STEM = "No Piano"
    NO_SYNTH_STEM = "No Synthesizer"
    NO_STRINGS_STEM = "No Strings"
    NO_WOODWINDS_STEM = "No Woodwinds"
    NO_WIND_INST_STEM = "No Wind Inst"
    NO_BRASS_STEM = "No Brass"
    PRIMARY_STEM = "Primary Stem"
    SECONDARY_STEM = "Secondary Stem"
    LEAD_VOCAL_STEM = "lead_only"
    BV_VOCAL_STEM = "backing_only"
    LEAD_VOCAL_STEM_I = "with_lead_vocals"
    BV_VOCAL_STEM_I = "with_backing_vocals"
    LEAD_VOCAL_STEM_LABEL = "Lead Vocals"
    BV_VOCAL_STEM_LABEL = "Backing Vocals"

    NON_ACCOM_STEMS = (VOCAL_STEM, OTHER_STEM, BASS_STEM, DRUM_STEM, GUITAR_STEM, PIANO_STEM, SYNTH_STEM, STRINGS_STEM, WOODWINDS_STEM, BRASS_STEM, WIND_INST_STEM)

    def __init__(self, config):
        # Inferencing device / acceleration config
        self.torch_device = config.get("torch_device")
        self.torch_device_cpu = config.get("torch_device_cpu")
        self.torch_device_mps = config.get("torch_device_mps")
        self.onnx_execution_provider = config.get("onnx_execution_provider")

        # Model data
        self.model_name = config.get("model_name")
        self.model_path = config.get("model_path")
        self.model_data = config.get("model_data")

        # Output directory and format
        self.output_dir = config.get("output_dir")
        self.output_format = config.get("output_format")

        # Functional options which are applicable to all architectures and the user may tweak to affect the output
        self.normalization_threshold = config.get("normalization_threshold")
        self.enable_denoise = config.get("enable_denoise")
        self.output_single_stem = config.get("output_single_stem")
        self.invert_using_spec = config.get("invert_using_spec")
        self.sample_rate = config.get("sample_rate")

        # Model specific properties
        self.primary_stem_name = self.model_data.get("primary_stem", "Vocals")
        self.secondary_stem_name = "Vocals" if self.primary_stem_name == "Instrumental" else "Instrumental"
        self.is_karaoke = self.model_data.get("is_karaoke", False)
        self.is_bv_model = self.model_data.get("is_bv_model", False)
        self.bv_model_rebalance = self.model_data.get("is_bv_model_rebalanced", 0)

        # print(f"Common params: model_name={self.model_name}, model_path={self.model_path}")
        # print(f"Common params: output_dir={self.output_dir}, output_format={self.output_format}")
        # print(f"Common params: normalization_threshold={self.normalization_threshold}")
        # print(f"Common params: enable_denoise={self.enable_denoise}, output_single_stem={self.output_single_stem}")
        # print(f"Common params: invert_using_spec={self.invert_using_spec}, sample_rate={self.sample_rate}")

        # print(f"Common params: primary_stem_name={self.primary_stem_name}, secondary_stem_name={self.secondary_stem_name}")
        # print(f"Common params: is_karaoke={self.is_karaoke}, is_bv_model={self.is_bv_model}, bv_model_rebalance={self.bv_model_rebalance}")

        # File-specific variables which need to be cleared between processing different audio inputs
        self.audio_file_path = None
        self.audio_file_base = None

        self.primary_source = None
        self.secondary_source = None

        self.primary_stem_output_path = None
        self.secondary_stem_output_path = None

        self.cached_sources_map = {}

    def separate(self, audio_file_path):
        """
        Placeholder method for separating audio sources. Should be overridden by subclasses.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def final_process(self, stem_path, source, stem_name):
        """
        Finalizes the processing of a stem by writing the audio to a file and returning the processed source.
        """
        print(f"Finalizing {stem_name} stem processing and writing audio...")
        self.write_audio(stem_path, source)

        return {stem_name: source}

    def cached_sources_clear(self):
        """
        Clears the cache dictionaries for VR, MDX, and Demucs models.

        This function is essential for ensuring that the cache does not hold outdated or irrelevant data
        between different processing sessions or when a new batch of audio files is processed.
        It helps in managing memory efficiently and prevents potential errors due to stale data.
        """
        self.cached_sources_map = {}

    def cached_source_callback(self, model_architecture, model_name=None):
        """
        Retrieves the model and sources from the cache based on the processing method and model name.

        Args:
            model_architecture: The architecture type (VR, MDX, or Demucs) being used for processing.
            model_name: The specific model name within the architecture type, if applicable.

        Returns:
            A tuple containing the model and its sources if found in the cache; otherwise, None.

        This function is crucial for optimizing performance by avoiding redundant processing.
        If the requested model and its sources are already in the cache, they can be reused directly,
        saving time and computational resources.
        """
        model, sources = None, None

        mapper = self.cached_sources_map[model_architecture]

        for key, value in mapper.items():
            if model_name in key:
                model = key
                sources = value

        return model, sources

    def cached_model_source_holder(self, model_architecture, sources, model_name=None):
        """
        Update the dictionary for the given model_architecture with the new model name and its sources.
        Use the model_architecture as a key to access the corresponding cache source mapper dictionary.
        """
        self.cached_sources_map[model_architecture] = {**self.cached_sources_map.get(model_architecture, {}), **{model_name: sources}}

    def prepare_mix(self, mix):
        """
        Prepares the mix for processing. This includes loading the audio from a file if necessary,
        ensuring the mix is in the correct format, and converting mono to stereo if needed.
        """
        # Store the original path or the mix itself for later checks
        audio_path = mix

        # Check if the input is a file path (string) and needs to be loaded
        if not isinstance(mix, np.ndarray):
            print(f"Loading audio from file: {mix}")
            mix, sr = librosa.load(mix, mono=False, sr=self.sample_rate)
            print(f"Audio loaded. Sample rate: {sr}, Audio shape: {mix.shape}")
        else:
            # Transpose the mix if it's already an ndarray (expected shape: [channels, samples])
            print("Transposing the provided mix array.")
            mix = mix.T
            print(f"Transposed mix shape: {mix.shape}")

        # If the original input was a filepath, check if the loaded mix is empty
        if isinstance(audio_path, str):
            if not np.any(mix):
                error_msg = f"Audio file {audio_path} is empty or not valid"
                print(error_msg)
                raise ValueError(error_msg)
            else:
                print("Audio file is valid and contains data.")

        # Ensure the mix is in stereo format
        if mix.ndim == 1:
            print("Mix is mono. Converting to stereo.")
            mix = np.asfortranarray([mix, mix])
            print("Converted to stereo mix.")

        # Final log indicating successful preparation of the mix
        print("Mix preparation completed.")
        return mix

    def write_audio(self, stem_path: str, stem_source):
        """
        Writes the separated audio source to a file.
        """
        print(f"Entering write_audio with stem_path: {stem_path}")

        stem_source = normalize(wave=stem_source, max_peak=self.normalization_threshold)

        # Check if the numpy array is empty or contains very low values
        if np.max(np.abs(stem_source)) < 1e-6:
            print("Warning: stem_source array is near-silent or empty.")
            return

        # If output_dir is specified, create it and join it with stem_path
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            stem_path = os.path.join(self.output_dir, stem_path)

        print(f"Audio data shape before processing: {stem_source.shape}")
        print(f"Data type before conversion: {stem_source.dtype}")

        # Ensure the audio data is in the correct format (e.g., int16)
        if stem_source.dtype != np.int16:
            stem_source = (stem_source * 32767).astype(np.int16)
            print("Converted stem_source to int16.")

        # Correctly interleave stereo channels
        stem_source_interleaved = np.empty((2 * stem_source.shape[0],), dtype=np.int16)
        stem_source_interleaved[0::2] = stem_source[:, 0]  # Left channel
        stem_source_interleaved[1::2] = stem_source[:, 1]  # Right channel

        print(f"Interleaved audio data shape: {stem_source_interleaved.shape}")

        # Create a pydub AudioSegment
        try:
            audio_segment = AudioSegment(stem_source_interleaved.tobytes(), frame_rate=self.sample_rate, sample_width=stem_source.dtype.itemsize, channels=2)
            print("Created AudioSegment successfully.")
        except (IOError, ValueError) as e:
            print(f"Specific error creating AudioSegment: {e}")
            return

        # Determine file format based on the file extension
        file_format = stem_path.lower().split(".")[-1]

        # For m4a files, specify mp4 as the container format as the extension doesn't match the format name
        if file_format == "m4a":
            file_format = "mp4"
        elif file_format == "mka":
            file_format = "matroska"

        # Export using the determined format
        try:
            with open(stem_path, 'wb') as f:
                audio_segment.export(f, format=file_format)
                print(f"Exported audio file successfully to {stem_path}")
        except (IOError, ValueError) as e:
            print(f"Error exporting audio file: {e}")

    def clear_gpu_cache(self):
        """
        This method clears the GPU cache to free up memory.
        """
        print("Running garbage collection...")
        gc.collect()
        if self.torch_device == torch.device("mps"):
            print("Clearing MPS cache...")
            torch.mps.empty_cache()
        if self.torch_device == torch.device("cuda"):
            print("Clearing CUDA cache...")
            torch.cuda.empty_cache()

    def clear_file_specific_paths(self):
        """
        Clears the file-specific variables which need to be cleared between processing different audio inputs.
        """
        print("Clearing input audio file paths, sources and stems...")

        self.audio_file_path = None
        self.audio_file_base = None

        self.primary_source = None
        self.secondary_source = None

        self.primary_stem_output_path = None
        self.secondary_stem_output_path = None


class MDXSeparator(CommonSeparator):
    """
    MDXSeparator is responsible for separating audio sources using MDX models.
    It initializes with configuration parameters and prepares the model for separation tasks.
    """

    def __init__(self, common_config):
        # Any configuration values which can be shared between architectures should be set already in CommonSeparator,
        # e.g. user-specified functionality choices (self.output_single_stem) or common model parameters (self.primary_stem_name)
        super().__init__(config=common_config)

        # Initializing user-configurable parameters, passed through with an mdx_from the CLI or Separator instance

        # Pick a segment size to balance speed, resource use, and quality:
        # - Smaller sizes consume less resources.
        # - Bigger sizes consume more resources, but may provide better results.
        # - Default size is 256. Quality can change based on your pick.
        self.segment_size = 256

        # This option controls the amount of overlap between prediction windows.
        #  - Higher values can provide better results, but will lead to longer processing times.
        #  - For Non-MDX23C models: You can choose between 0.001-0.999
        self.overlap = 0.25

        # Number of batches to be processed at a time.
        # - Higher values mean more RAM usage but slightly faster processing times.
        # - Lower values mean less RAM usage but slightly longer processing times.
        # - Batch size value has no effect on output quality.
        # BATCH_SIZE = ('1', ''2', '3', '4', '5', '6', '7', '8', '9', '10')
        self.batch_size = 1

        # hop_length is equivalent to the more commonly used term "stride" in convolutional neural networks
        # In machine learning, particularly in the context of convolutional neural networks (CNNs),
        # the term "stride" refers to the number of pixels by which we move the filter across the input image.
        # Strides are a crucial component in the convolution operation, a fundamental building block of CNNs used primarily in the field of computer vision.
        # Stride is a parameter that dictates the movement of the kernel, or filter, across the input data, such as an image.
        # When performing a convolution operation, the stride determines how many units the filter shifts at each step.
        # The choice of stride affects the model in several ways:
        # Output Size: A larger stride will result in a smaller output spatial dimension.
        # Computational Efficiency: Increasing the stride can decrease the computational load.
        # Field of View: A higher stride means that each step of the filter takes into account a wider area of the input image.
        #   This can be beneficial when the model needs to capture more global features rather than focusing on finer details.
        self.hop_length = 1024

        # If enabled, model will be run twice to reduce noise in output audio.
        self.enable_denoise = False

        print(f"MDX arch params: batch_size={self.batch_size}, segment_size={self.segment_size}")
        print(f"MDX arch params: overlap={self.overlap}, hop_length={self.hop_length}, enable_denoise={self.enable_denoise}")

        # Initializing model-specific parameters from model_data JSON
        self.compensate = self.model_data["compensate"]
        self.dim_f = self.model_data["mdx_dim_f_set"]
        self.dim_t = 2 ** self.model_data["mdx_dim_t_set"]
        self.n_fft = self.model_data["mdx_n_fft_scale_set"]
        self.config_yaml = self.model_data.get("config_yaml", None)

        print(f"MDX arch params: compensate={self.compensate}, dim_f={self.dim_f}, dim_t={self.dim_t}, n_fft={self.n_fft}")
        # print(f"MDX arch params: config_yaml={self.config_yaml}")

        # In UVR, these variables are set but either aren't useful or are better handled in audio-separator.
        # Leaving these comments explaining to help myself or future developers understand why these aren't in audio-separator.

        # "chunks" is not actually used for anything in UVR...
        # self.chunks = 0

        # "adjust" is hard-coded to 1 in UVR, and only used as a multiplier in run_model, so it does nothing.
        # self.adjust = 1

        # "hop" is hard-coded to 1024 in UVR. We have a "hop_length" parameter instead
        # self.hop = 1024

        # "margin" maps to sample rate and is set from the GUI in UVR (default: 44100). We have a "sample_rate" parameter instead.
        # self.margin = 44100

        # "dim_c" is hard-coded to 4 in UVR, seems to be a parameter for the number of channels, and is only used for checkpoint models.
        # We haven't implemented support for the checkpoint models here, so we're not using it.
        # self.dim_c = 4

        self.load_model()

        self.n_bins = 0
        self.trim = 0
        self.chunk_size = 0
        self.gen_size = 0
        self.stft = None

        self.primary_source = None
        self.secondary_source = None
        self.audio_file_path = None
        self.audio_file_base = None

        self.progress_callback = None

    def load_model(self):
        """
        Load the model into memory from file on disk, initialize it with config from the model data,
        and prepare for inferencing using hardware accelerated Torch device.
        """
        print("Loading ONNX model for inference...")

        if self.segment_size == self.dim_t:
            ort_session_options = ort.SessionOptions()
            ort_session_options.log_severity_level = 3
            ort_inference_session = ort.InferenceSession(self.model_path, providers=self.onnx_execution_provider, sess_options=ort_session_options)
            self.model_run = lambda spek: ort_inference_session.run(None, {"input": spek.cpu().numpy()})[0]
            print("Model loaded successfully using ONNXruntime inferencing session.")
        else:
            raise "Segment size is not equal dim_t, need to install onnx2torch and uncommitted 458-467 lines"

    def separate(self, audio_file_path, progress_callback=None):
        """
        Separates the audio file into primary and secondary sources based on the model's configuration.
        It processes the mix, demixes it into sources, normalizes the sources, and saves the output files.

        Args:
            audio_file_path (str): The path to the audio file to be processed.

        Returns:
            list: A list of paths to the output files generated by the separation process.
        """
        self.progress_callback = progress_callback

        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        # Prepare the mix for processing
        print(f"Preparing mix for input audio file {self.audio_file_path}...")
        mix = self.prepare_mix(self.audio_file_path)

        print("Normalizing mix before demixing...")
        mix = normalize(wave=mix, max_peak=self.normalization_threshold)

        # Start the demixing process
        source = self.demix(mix)
        print("Demixing completed.")

        # In UVR, the source is cached here if it's a vocal split model, but we're not supporting that yet

        # Initialize the list for output files
        output_files = []
        print("Processing output files...")

        # Normalize and transpose the primary source if it's not already an array
        if not isinstance(self.primary_source, np.ndarray):
            print("Normalizing primary source...")
            self.primary_source = normalize(wave=source, max_peak=self.normalization_threshold).T

        # Process the secondary source if not already an array
        if not isinstance(self.secondary_source, np.ndarray):
            print("Producing secondary source: demixing in match_mix mode")
            raw_mix = self.demix(mix, is_match_mix=True)

            if self.invert_using_spec:
                print("Inverting secondary stem using spectogram as invert_using_spec is set to True")
                self.secondary_source = invert_stem(raw_mix, source)
            else:
                print("Inverting secondary stem by subtracting of transposed demixed stem from transposed original mix")
                self.secondary_source = mix.T - source.T

        # Save and process the secondary stem if needed
        if not self.output_single_stem or self.output_single_stem.lower() == self.secondary_stem_name.lower():
            self.secondary_stem_output_path = os.path.join(
                f"{self.audio_file_base}_({self.secondary_stem_name})_{self.model_name}.{self.output_format.lower()}")

            print(f"Saving {self.secondary_stem_name} stem to {self.secondary_stem_output_path}...")
            self.final_process(self.secondary_stem_output_path, self.secondary_source, self.secondary_stem_name)
            output_files.append(self.secondary_stem_output_path)

        # Save and process the primary stem if needed
        if not self.output_single_stem or self.output_single_stem.lower() == self.primary_stem_name.lower():
            self.primary_stem_output_path = os.path.join(
                f"{self.audio_file_base}_({self.primary_stem_name})_{self.model_name}.{self.output_format.lower()}")
            if not isinstance(self.primary_source, np.ndarray):
                self.primary_source = source.T

            print(f"Saving {self.primary_stem_name} stem to {self.primary_stem_output_path}...")
            self.final_process(self.primary_stem_output_path, self.primary_source, self.primary_stem_name)
            output_files.append(self.primary_stem_output_path)

        # Not yet implemented from UVR features:
        # self.process_vocal_split_chain(secondary_sources)
        # print("Vocal split chain processed.")

        return output_files

    def initialize_model_settings(self):
        """
        This function sets up the necessary parameters for the model, like the number of frequency bins (n_bins), the trimming size (trim),
        the size of each audio chunk (chunk_size), and the window function for spectral transformations (window).
        It ensures that the model is configured with the correct settings for processing the audio data.
        """
        print("Initializing model settings...")

        # n_bins is half the FFT size plus one (self.n_fft // 2 + 1).
        self.n_bins = self.n_fft // 2 + 1

        # trim is half the FFT size (self.n_fft // 2).
        self.trim = self.n_fft // 2

        # chunk_size is the hop_length size times the segment size minus one
        self.chunk_size = self.hop_length * (self.segment_size - 1)

        # gen_size is the chunk size minus twice the trim size
        self.gen_size = self.chunk_size - 2 * self.trim

        self.stft = STFT(self.n_fft, self.hop_length, self.dim_f, self.torch_device)

        print(f"Model input params: n_fft={self.n_fft} hop_length={self.hop_length} dim_f={self.dim_f}")
        print(f"Model settings: n_bins={self.n_bins}, trim={self.trim}, chunk_size={self.chunk_size}, gen_size={self.gen_size}")

    def initialize_mix(self, mix, is_ckpt=False):
        """
        After prepare_mix segments the audio, initialize_mix further processes each segment.
        It ensures each audio segment is in the correct format for the model, applies necessary padding,
        and converts the segments into tensors for processing with the model.
        This step is essential for preparing the audio data in a format that the neural network can process.
        """
        # Log the initialization of the mix and whether checkpoint mode is used
        print(f"Initializing mix with is_ckpt={is_ckpt}. Initial mix shape: {mix.shape}")

        # Ensure the mix is a 2-channel (stereo) audio signal
        if mix.shape[0] != 2:
            error_message = f"Expected a 2-channel audio signal, but got {mix.shape[0]} channels"
            print(error_message)
            raise ValueError(error_message)

        # If in checkpoint mode, process the mix differently
        if is_ckpt:
            print("Processing in checkpoint mode...")
            # Calculate padding based on the generation size and trim
            pad = self.gen_size + self.trim - (mix.shape[-1] % self.gen_size)
            print(f"Padding calculated: {pad}")
            # Add padding at the beginning and the end of the mix
            mixture = np.concatenate(
                (np.zeros((2, self.trim), dtype="float32"), mix, np.zeros((2, pad), dtype="float32")), 1)
            # Determine the number of chunks based on the mixture's length
            num_chunks = mixture.shape[-1] // self.gen_size
            print(f"Mixture shape after padding: {mixture.shape}, Number of chunks: {num_chunks}")
            # Split the mixture into chunks
            mix_waves = [mixture[:, i * self.gen_size: i * self.gen_size + self.chunk_size] for i in range(num_chunks)]
        else:
            # If not in checkpoint mode, process normally
            print("Processing in non-checkpoint mode...")
            mix_waves = []
            n_sample = mix.shape[1]
            # Calculate necessary padding to make the total length divisible by the generation size
            pad = self.gen_size - n_sample % self.gen_size
            print(f"Number of samples: {n_sample}, Padding calculated: {pad}")
            # Apply padding to the mix
            mix_p = np.concatenate((np.zeros((2, self.trim)), mix, np.zeros((2, pad)), np.zeros((2, self.trim))), 1)
            print(f"Shape of mix after padding: {mix_p.shape}")

            # Process the mix in chunks
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i: i + self.chunk_size])
                mix_waves.append(waves)
                print(f"Processed chunk {len(mix_waves)}: Start {i}, End {i + self.chunk_size}")
                i += self.gen_size

        # Convert the list of wave chunks into a tensor for processing on the specified device
        mix_waves_tensor = torch.tensor(mix_waves, dtype=torch.float32).to(self.torch_device)
        print(f"Converted mix_waves to tensor. Tensor shape: {mix_waves_tensor.shape}")

        return mix_waves_tensor, pad

    def demix(self, mix, is_match_mix=False):
        """
        Demixes the input mix into its constituent sources. If is_match_mix is True, the function adjusts the processing
        to better match the mix, affecting chunk sizes and overlaps. The demixing process involves padding the mix,
        processing it in chunks, applying windowing for overlaps, and accumulating the results to separate the sources.
        """
        print(f"Starting demixing process with is_match_mix: {is_match_mix}...")
        self.initialize_model_settings()

        # Preserves the original mix for later use.
        # In UVR, this is used for the pitch fix and VR denoise processes, which aren't yet implemented here.
        org_mix = mix
        print(f"Original mix stored. Shape: {org_mix.shape}")

        # Initializes a list to store the separated waveforms.
        tar_waves_ = []

        # Handling different chunk sizes and overlaps based on the matching requirement.
        if is_match_mix:
            # Sets a smaller chunk size specifically for matching the mix.
            chunk_size = self.hop_length * (self.segment_size - 1)
            # Sets a small overlap for the chunks.
            overlap = 0.02
            print(f"Chunk size for matching mix: {chunk_size}, Overlap: {overlap}")
        else:
            # Uses the regular chunk size defined in model settings.
            chunk_size = self.chunk_size
            # Uses the overlap specified in the model settings.
            overlap = self.overlap
            print(f"Standard chunk size: {chunk_size}, Overlap: {overlap}")

        # Calculates the generated size after subtracting the trim from both ends of the chunk.
        gen_size = chunk_size - 2 * self.trim
        print(f"Generated size calculated: {gen_size}")

        # Calculates padding to make the mix length a multiple of the generated size.
        pad = gen_size + self.trim - ((mix.shape[-1]) % gen_size)
        # Prepares the mixture with padding at the beginning and the end.
        mixture = np.concatenate((np.zeros((2, self.trim), dtype="float32"), mix, np.zeros((2, pad), dtype="float32")),
                                 1)
        print(f"Mixture prepared with padding. Mixture shape: {mixture.shape}")

        # Calculates the step size for processing chunks based on the overlap.
        step = int((1 - overlap) * chunk_size)
        print(f"Step size for processing chunks: {step} as overlap is set to {overlap}.")

        # Initializes arrays to store the results and to account for overlap.
        result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
        divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)

        # Initializes counters for processing chunks.
        total = 0
        total_chunks = (mixture.shape[-1] + step - 1) // step
        print(f"Total chunks to process: {total_chunks}")

        # Processes each chunk of the mixture.
        for i in tqdm(range(0, mixture.shape[-1], step)):
            total += 1
            start = i
            end = min(i + chunk_size, mixture.shape[-1])
            # print(f"Processing chunk {total}/{total_chunks}: Start {start}, End {end}")

            # Handles windowing for overlapping chunks.
            chunk_size_actual = end - start
            window = None
            if overlap != 0:
                window = np.hanning(chunk_size_actual)
                window = np.tile(window[None, None, :], (1, 2, 1))
                # print("Window applied to the chunk.")

            # Zero-pad the chunk to prepare it for processing.
            mix_part_ = mixture[:, start:end]
            if end != i + chunk_size:
                pad_size = (i + chunk_size) - end
                mix_part_ = np.concatenate((mix_part_, np.zeros((2, pad_size), dtype="float32")), axis=-1)

            # Converts the chunk to a tensor for processing.
            # mix_part = torch.tensor([mix_part_], dtype=torch.float32).to(self.torch_device)
            mix_part_ = np.array([mix_part_], dtype=np.float32)  # Efficient conversion to numpy array
            mix_part = torch.tensor(mix_part_).to(self.torch_device)  # Then convert to tensor
            # Splits the chunk into smaller batches if necessary.
            mix_waves = mix_part.split(self.batch_size)
            total_batches = len(mix_waves)
            # print(f"Mix part split into batches. Number of batches: {total_batches}")

            with torch.no_grad():
                # Processes each batch in the chunk.
                batches_processed = 0
                for mix_wave in mix_waves:
                    batches_processed += 1
                    # print(f"Processing mix_wave batch {batches_processed}/{total_batches}")

                    # Runs the model to separate the sources.
                    tar_waves = self.run_model(mix_wave, is_match_mix=is_match_mix)

                    # Applies windowing if needed and accumulates the results.
                    if window is not None:
                        tar_waves[..., :chunk_size_actual] *= window
                        divider[..., start:end] += window
                    else:
                        divider[..., start:end] += 1

                    result[..., start:end] += tar_waves[..., : end - start]

            # Updating progress
            if self.progress_callback:
                self.progress_callback(round(total / total_chunks * 100, 0), "Demix...")

        # Normalizes the results by the divider to account for overlap.
        print("Normalizing result by dividing result by divider.")
        epsilon = 1e-8  # A small constant to avoid division by zero
        tar_waves = result / (divider + epsilon)
        tar_waves_.append(tar_waves)

        # Reshapes the results to match the original dimensions.
        tar_waves_ = np.vstack(tar_waves_)[:, :, self.trim: -self.trim]
        tar_waves = np.concatenate(tar_waves_, axis=-1)[:, : mix.shape[-1]]

        # Extracts the source from the results.
        source = tar_waves[:, 0:None]
        print(f"Concatenated tar_waves. Shape: {tar_waves.shape}")

        # TODO: In UVR, pitch changing happens here. Consider implementing this as a feature.

        # Compensates the source if not matching the mix.
        if not is_match_mix:
            source *= self.compensate
            print("Match mix mode; compensate multiplier applied.")

        # TODO: In UVR, VR denoise model gets applied here. Consider implementing this as a feature.

        print("Demixing process completed.")
        return source

    def run_model(self, mix, is_match_mix=False):
        """
        Processes the input mix through the model to separate the sources.
        Applies STFT, handles spectrum modifications, and runs the model for source separation.
        """
        # Applying the STFT to the mix. The mix is moved to the specified device (e.g., GPU) before processing.
        spek = self.stft(mix.to(self.torch_device))
        # print(f"STFT applied on mix. Spectrum shape: {spek.shape}")

        # Zeroing out the first 3 bins of the spectrum. This is often done to reduce low-frequency noise.
        spek[:, :, :3, :] *= 0

        # Handling the case where the mix needs to be matched (is_match_mix = True)
        if is_match_mix:
            spec_pred = spek.cpu().numpy()
            # print("is_match_mix: spectrum prediction obtained directly from STFT output.")
        else:
            # If denoising is enabled, the model is run on both the negative and positive spectrums.
            if self.enable_denoise:
                # Assuming spek is a tensor and self.model_run can process it directly
                spec_pred_neg = self.model_run(-spek)  # Ensure this line correctly negates spek and runs the model
                spec_pred_pos = self.model_run(spek)
                # Ensure both spec_pred_neg and spec_pred_pos are tensors before applying operations
                spec_pred = (spec_pred_neg * -0.5) + (spec_pred_pos * 0.5)  # [invalid-unary-operand-type]
                # print("Model run on both negative and positive spectrums for denoising.")
            else:
                spec_pred = self.model_run(spek)
                # print("Model run on the spectrum without denoising.")

        # Applying the inverse STFT to convert the spectrum back to the time domain.
        result = self.stft.inverse(torch.tensor(spec_pred).to(self.torch_device)).cpu().detach().numpy()
        # print(f"Inverse STFT applied. Returning result with shape: {result.shape}")

        return result


class STFT:
    """
    This class performs the Short-Time Fourier Transform (STFT) and its inverse (ISTFT).
    These functions are essential for converting the audio between the time domain and the frequency domain,
    which is a crucial aspect of audio processing in neural networks.
    """

    def __init__(self, n_fft, hop_length, dim_f, device):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dim_f = dim_f
        self.device = device
        # Create a Hann window tensor for use in the STFT.
        self.hann_window = torch.hann_window(window_length=self.n_fft, periodic=True)

    def __call__(self, input_tensor):
        # Determine if the input tensor's device is not a standard computing device (i.e., not CPU or CUDA).
        is_non_standard_device = not input_tensor.device.type in ["cuda", "cpu"]

        # If on a non-standard device, temporarily move the tensor to CPU for processing.
        if is_non_standard_device:
            input_tensor = input_tensor.cpu()

        # Transfer the pre-defined window tensor to the same device as the input tensor.
        stft_window = self.hann_window.to(input_tensor.device)

        # Extract batch dimensions (all dimensions except the last two which are channel and time).
        batch_dimensions = input_tensor.shape[:-2]

        # Extract channel and time dimensions (last two dimensions of the tensor).
        channel_dim, time_dim = input_tensor.shape[-2:]

        # Reshape the tensor to merge batch and channel dimensions for STFT processing.
        reshaped_tensor = input_tensor.reshape([-1, time_dim])

        # Perform the Short-Time Fourier Transform (STFT) on the reshaped tensor.
        stft_output = torch.stft(reshaped_tensor, n_fft=self.n_fft, hop_length=self.hop_length, window=stft_window, center=True, return_complex=True)
        stft_output_real = torch.view_as_real(stft_output)

        # Rearrange the dimensions of the STFT output to bring the frequency dimension forward.
        permuted_stft_output = stft_output_real.permute([0, 3, 1, 2])

        # Reshape the output to restore the original batch and channel dimensions, while keeping the newly formed frequency and time dimensions.
        final_output = permuted_stft_output.reshape([*batch_dimensions, channel_dim, 2, -1, permuted_stft_output.shape[-1]]).reshape(
            [*batch_dimensions, channel_dim * 2, -1, permuted_stft_output.shape[-1]]
        )

        # If the original tensor was on a non-standard device, move the processed tensor back to that device.
        if is_non_standard_device:
            final_output = final_output.to(self.device)

        # Return the transformed tensor, sliced to retain only the required frequency dimension (`dim_f`).
        return final_output[..., : self.dim_f, :]

    def pad_frequency_dimension(self, input_tensor, batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins):
        """
        Adds zero padding to the frequency dimension of the input tensor.
        """
        # Create a padding tensor for the frequency dimension
        freq_padding = torch.zeros([*batch_dimensions, channel_dim, num_freq_bins - freq_dim, time_dim]).to(input_tensor.device)

        # Concatenate the padding to the input tensor along the frequency dimension.
        padded_tensor = torch.cat([input_tensor, freq_padding], -2)

        return padded_tensor

    def calculate_inverse_dimensions(self, input_tensor):
        # Extract batch dimensions and frequency-time dimensions.
        batch_dimensions = input_tensor.shape[:-3]
        channel_dim, freq_dim, time_dim = input_tensor.shape[-3:]

        # Calculate the number of frequency bins for the inverse STFT.
        num_freq_bins = self.n_fft // 2 + 1

        return batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins

    def prepare_for_istft(self, padded_tensor, batch_dimensions, channel_dim, num_freq_bins, time_dim):
        """
        Prepares the tensor for Inverse Short-Time Fourier Transform (ISTFT) by reshaping
        and creating a complex tensor from the real and imaginary parts.
        """
        # Reshape the tensor to separate real and imaginary parts and prepare for ISTFT.
        reshaped_tensor = padded_tensor.reshape([*batch_dimensions, channel_dim // 2, 2, num_freq_bins, time_dim])

        # Flatten batch dimensions and rearrange for ISTFT.
        flattened_tensor = reshaped_tensor.reshape([-1, 2, num_freq_bins, time_dim])

        # Rearrange the dimensions of the tensor to bring the frequency dimension forward.
        permuted_tensor = flattened_tensor.permute([0, 2, 3, 1])

        # Combine real and imaginary parts into a complex tensor.
        complex_tensor = permuted_tensor[..., 0] + permuted_tensor[..., 1] * 1.0j

        return complex_tensor

    def inverse(self, input_tensor):
        # Determine if the input tensor's device is not a standard computing device (i.e., not CPU or CUDA).
        is_non_standard_device = not input_tensor.device.type in ["cuda", "cpu"]

        # If on a non-standard device, temporarily move the tensor to CPU for processing.
        if is_non_standard_device:
            input_tensor = input_tensor.cpu()

        # Transfer the pre-defined Hann window tensor to the same device as the input tensor.
        stft_window = self.hann_window.to(input_tensor.device)

        batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins = self.calculate_inverse_dimensions(input_tensor)

        padded_tensor = self.pad_frequency_dimension(input_tensor, batch_dimensions, channel_dim, freq_dim, time_dim, num_freq_bins)

        complex_tensor = self.prepare_for_istft(padded_tensor, batch_dimensions, channel_dim, num_freq_bins, time_dim)

        # Perform the Inverse Short-Time Fourier Transform (ISTFT).
        istft_result = torch.istft(complex_tensor, n_fft=self.n_fft, hop_length=self.hop_length, window=stft_window, center=True)

        # Reshape ISTFT result to restore original batch and channel dimensions.
        final_output = istft_result.reshape([*batch_dimensions, 2, -1])

        # If the original tensor was on a non-standard device, move the processed tensor back to that device.
        if is_non_standard_device:
            final_output = final_output.to(self.device)

        return final_output


def get_model_hash(model_path):
    """
    This method returns the MD5 hash of a given model file.
    """
    import hashlib
    print(f"Calculating hash of model file {model_path}")
    try:
        # Open the model file in binary read mode
        with open(model_path, "rb") as f:
            # Move the file pointer 10MB before the end of the file
            f.seek(-10000 * 1024, 2)
            # Read the file from the current pointer to the end and calculate its MD5 hash
            return hashlib.md5(f.read()).hexdigest()
    except IOError as e:
        # If an IOError occurs (e.g., if the file is less than 10MB large), log the error
        print(f"IOError seeking -10MB or reading model file for hash calculation: {e}")
        # Attempt to open the file again, read its entire content, and calculate the MD5 hash
        return hashlib.md5(open(model_path, "rb").read()).hexdigest()
