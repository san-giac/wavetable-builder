import streamlit as st
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from streamlit_sortables import sort_items

st.set_page_config(layout="wide")

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def normalize_to_minus_6dbfs(signal):
    peak = np.max(np.abs(signal))
    if peak < 1e-12:
        return signal
    target_linear = 10 ** (-0.5 / 20.0)  # -6 dB
    scale = target_linear / peak
    return signal * scale

def generate_playback_wave(single_cycle, duration=5.0, freq=261.63, sr=44100):
    wave_len = len(single_cycle)
    phase_inc = (wave_len * freq) / sr
    total_samples = int(sr * duration)

    out = np.zeros(total_samples, dtype=np.float32)
    phase = 0.0
    for i in range(total_samples):
        idx = int(phase) % wave_len
        out[i] = single_cycle[idx]
        phase += phase_inc

    return out

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------

st.title("Wavetable Builder")

st.write("""
Upload up to 16 single-cycle audio files.
You can reorder the waves via drag-and-drop, visualize the combined waveform,
and listen to a 5-second morphing playback.
""")

if "wave_chunks" not in st.session_state:
    st.session_state.wave_chunks = [None] * 16
if "file_order" not in st.session_state:
    st.session_state.file_order = list(range(16))
if "sample_rate" not in st.session_state:
    st.session_state.sample_rate = None

wave_chunks = st.session_state.wave_chunks
file_order = st.session_state.file_order

st.subheader("1) Upload up to 16 Audio Files")
uploaded = st.file_uploader(
    "Select audio files (max 16).",
    type=["wav", "flac", "aiff", "aif", "ogg", "mp3"],
    accept_multiple_files=True
)

if uploaded:
    used_files = uploaded[:16]
    for i, upfile in enumerate(used_files):
        try:
            y, sr = librosa.load(upfile, sr=None, mono=True)
            if st.session_state.sample_rate is None:
                st.session_state.sample_rate = sr
            elif sr != st.session_state.sample_rate:
                st.warning(f"Sample rate mismatch in {upfile.name} ({sr} vs {st.session_state.sample_rate}). No resampling applied.")

            length = len(y)
            if length < 2048:
                padded = np.zeros(2048, dtype=y.dtype)
                padded[:length] = y
                y = padded
            else:
                y = y[:2048]

            y = y.astype(np.float32)

            wave_chunks[i] = y
            st.session_state.wave_chunks = wave_chunks

        except Exception as e:
            st.warning(f"Error loading file {upfile.name}: {e}")
            wave_chunks[i] = None

st.subheader("2) Reorder Waves (Drag and Drop)")

files_list = []
for idx in range(16):
    if wave_chunks[idx] is not None and idx < len(uploaded):
        files_list.append(uploaded[idx].name)
    else:
        files_list.append(f"[Empty Slot {idx + 1}]")

sorted_files = sort_items(files_list)

new_order = []
for filename in sorted_files:
    if "[Empty Slot" not in filename:
        idx = next((i for i, f in enumerate(uploaded) if f.name == filename), None)
        if idx is not None and wave_chunks[idx] is not None:
            new_order.append(idx)
    else:
        new_order.append(None)

while len(new_order) < 16:
    new_order.append(None)

st.session_state.file_order = new_order
file_order = st.session_state.file_order

st.subheader("3) Combined Wavetable Preview + Morphing Playback")

used_chunks = []
for idx in file_order:
    if idx is not None and wave_chunks[idx] is not None:
        used_chunks.append(wave_chunks[idx])

if used_chunks:
    combined_wave = np.concatenate(used_chunks)
    fig, ax = plt.subplots(figsize=(15, 3))
    ax.plot(combined_wave, color="blue", linewidth=0.8)
    pos = 0
    for chunk in used_chunks:
        pos += len(chunk)
        ax.axvline(x=pos, color='red', linestyle='--', linewidth=1)
    ax.set_title("Wavetable Preview (Combined Waves)", fontsize=14)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle=':')
    st.pyplot(fig)

    sr = 44100
    playback_duration = 5.0
    num_steps = 500
    max_morph = len(used_chunks) - 1
    morph_values = np.linspace(0, max_morph, num_steps)
    playback_buffer = np.array([], dtype=np.float32)

    def get_morphed_wave(morph_pos, chunks):
        pos = morph_pos
        i = int(np.floor(pos))
        alpha = pos - i
        i = max(0, min(i, len(chunks) - 1))
        if i == len(chunks) - 1 or alpha < 1e-6:
            return chunks[i]
        else:
            waveA = chunks[i]
            waveB = chunks[i + 1]
            return (1.0 - alpha) * waveA + alpha * waveB

    for m in morph_values:
        single_cycle = get_morphed_wave(m, used_chunks)
        segment = generate_playback_wave(single_cycle, duration=playback_duration / num_steps, freq=261.63, sr=sr)
        playback_buffer = np.concatenate((playback_buffer, segment))

    playback_buffer = normalize_to_minus_6dbfs(playback_buffer)

    playback_io = io.BytesIO()
    sf.write(playback_io, playback_buffer, sr, format="WAV")
    playback_io.seek(0)

    st.audio(playback_io.read(), format="audio/wav", start_time=0)
else:
    st.write("No waves loaded yet.")

# ---------------------------------------------------------------------
# Combine & Export Wavetable
# ---------------------------------------------------------------------
st.subheader("4) Download Combined Wavetable")

if used_chunks:
    normalized_wave = normalize_to_minus_6dbfs(np.concatenate(used_chunks))
    out_sr = st.session_state.sample_rate or 44100
    combined_io = io.BytesIO()
    sf.write(combined_io, normalized_wave, out_sr, format="WAV")
    combined_io.seek(0)

    st.download_button(
        label="Download Combined Wavetable (Normalized)",
        data=combined_io.getvalue(),
        file_name="combined_wavetable.wav",
        mime="audio/wav"
    )
else:
    st.write("No waves loaded to combine.")
