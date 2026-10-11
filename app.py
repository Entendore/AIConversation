import streamlit as st
import torch
import os
import json
import io
import time
import wave
import pyaudio
import torchaudio as ta
import ollama
import numpy as np

# Try to import Whisper with error handling
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    st.error("Failed to import Whisper. Please install it with: pip install openai-whisper")
    WHISPER_AVAILABLE = False

# Try to import TTS (Coqui)
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    st.error("Failed to import Coqui TTS. Please install it with: pip install TTS")
    TTS_AVAILABLE = False

# Try to import speechbrain for VAD
try:
    from speechbrain.pretrained import VAD
    VAD_AVAILABLE = True
except ImportError:
    st.warning("Failed to import SpeechBrain VAD. Using energy-based VAD instead.")
    VAD_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="Live AI Voice Assistant",
    page_icon="🤖",
    layout="centered"
)

# --- Configuration Constants ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30  # 30ms chunks
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
SILENCE_THRESHOLD = 500  # Energy threshold for detecting silence
SILENCE_DURATION = 1.0  # Seconds of silence to stop recording

# --- Model Loading with Caching ---
@st.cache_resource
def load_whisper_model():
    if not WHISPER_AVAILABLE:
        return None
    try:
        model = whisper.load_model("base") # Use "base" for multilingual
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        return None

@st.cache_resource
def load_tts_model():
    if not TTS_AVAILABLE:
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        # Using a multi-lingual model from Coqui TTS
        # You can change this to other models like "tts_models/multilingual/multi-dataset/xtts_v2"
        model = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False).to(device)
        return model
    except Exception as e:
        st.error(f"Error loading Coqui TTS model: {e}")
        return None

@st.cache_resource
def load_vad_model():
    if not VAD_AVAILABLE:
        return None
    try:
        # Load the VAD model
        vad = VAD.from_hparams(
            source="speechbrain/vad-crdnn-libriparty",
            savedir="pretrained_models/vad-crdnn-libriparty"
        )
        return vad
    except Exception as e:
        st.error(f"Error loading VAD model: {e}")
        return None

# --- Session State Initialization ---
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": "You are a helpful and witty AI assistant. Be concise and friendly."}]
    if "conversation_active" not in st.session_state:
        st.session_state.conversation_active = False
    if "voice_prompt_path" not in st.session_state:
        st.session_state.voice_prompt_path = None

initialize_session_state()

# --- Simple Energy-based VAD ---
def is_speech_energy(audio_chunk, threshold=SILENCE_THRESHOLD):
    """Simple energy-based VAD"""
    # Convert bytes to numpy array
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
    # Calculate RMS energy
    energy = np.sqrt(np.mean(audio_data**2))
    return energy > threshold

# --- Live Audio Recording with Voice Activity Detection ---
def record_audio_until_silence():
    """Records audio from the microphone until a period of silence is detected."""
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK_SIZE)

    st.session_state.current_status = "🎙️ Listening... Speak now."
    st.rerun() # Update UI immediately

    frames = []
    num_silent_chunks = 0
    speech_detected = False
    silence_threshold_chunks = int(SILENCE_DURATION * 1000 / CHUNK_DURATION_MS)

    while True:
        chunk = stream.read(CHUNK_SIZE)
        frames.append(chunk)
        
        # Use energy-based VAD to check if the chunk contains speech
        is_speech = is_speech_energy(chunk)

        if is_speech:
            num_silent_chunks = 0
            if not speech_detected:
                speech_detected = True
                st.session_state.current_status = "🔴 Recording..."
                st.rerun()
        else:
            if speech_detected:
                num_silent_chunks += 1
                if num_silent_chunks > silence_threshold_chunks:
                    break # Stop recording after a period of silence

    stream.stop_stream()
    stream.close()
    audio.terminate()

    if not speech_detected:
        return None # No speech was detected

    st.session_state.current_status = "⏳ Processing..."
    st.rerun()

    # Save the recorded frames to a WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    wav_buffer.seek(0)
    return wav_buffer

# --- Main Application Logic ---
def main():
    whisper_model = load_whisper_model()
    tts_model = load_tts_model()

    st.title("🤖 Live AI Voice Assistant")
    
    # Display warnings if models are not available
    if not WHISPER_AVAILABLE:
        st.warning("Whisper is not available. Speech-to-text functionality will be disabled.")
    if not TTS_AVAILABLE:
        st.warning("TTS is not available. Text-to-speech functionality will be disabled.")
    
    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        available_models = [model["name"] for model in ollama.list().get("models", [])]
        if not available_models:
            st.warning("No Ollama models found. Please run `ollama pull llama3`.")
            selected_model = "llama3"
        else:
            selected_model = st.selectbox("Select Ollama Model", available_models)

        # Voice cloning
        st.header("🎙️ Voice Cloning (Optional)")
        voice_prompt_file = st.file_uploader("Upload voice prompt (WAV, 3-10s)", type=['wav'])
        if voice_prompt_file:
            temp_prompt_path = os.path.join("temp_voice_prompt.wav")
            with open(temp_prompt_path, "wb") as f:
                f.write(voice_prompt_file.getbuffer())
            st.session_state.voice_prompt_path = temp_prompt_path
            st.success("Custom voice loaded!")
        else:
            st.session_state.voice_prompt_path = None
            st.info("Using default voice.")

        st.header("💾 Session Management")
        if st.button("🆕 New Conversation"):
            st.session_state.messages = [{"role": "system", "content": "You are a helpful and witty AI assistant."}]
            st.session_state.conversation_active = False
            st.rerun()

        # Download chat history
        if len(st.session_state.messages) > 1:
            st.download_button(
                label="💾 Save Conversation",
                data=json.dumps(st.session_state.messages, indent=2),
                file_name="conversation.json",
                mime="application/json"
            )

        # Upload chat history
        uploaded_file = st.file_uploader("Load Conversation", type="json")
        if uploaded_file is not None:
            try:
                st.session_state.messages = json.load(uploaded_file)
                st.success("Conversation loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to load conversation: {e}")

    # --- Conversation Control Buttons ---
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎙️ Start Live Conversation", disabled=st.session_state.conversation_active, type="primary"):
            st.session_state.conversation_active = True
            st.rerun()
    with col2:
        if st.button("⏹️ Stop Conversation", disabled=not st.session_state.conversation_active):
            st.session_state.conversation_active = False
            st.session_state.current_status = "Conversation stopped."
            st.rerun()

    # --- Status Display ---
    if st.session_state.conversation_active:
        st.info(st.session_state.get("current_status", "Ready..."))
    
    # --- Chat History Display ---
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "system":
                continue
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # --- The Live Conversation Loop ---
    if st.session_state.conversation_active and whisper_model and tts_model:
        audio_buffer = record_audio_until_silence()

        if audio_buffer:
            try:
                # 1. Transcribe
                result = whisper_model.transcribe(audio_buffer)
                user_text = result['text'].strip()
                
                if user_text:
                    st.session_state.messages.append({"role": "user", "content": user_text})
                    with chat_container:
                        with st.chat_message("user"):
                            st.markdown(user_text)

                    # 2. Generate Response
                    response = ollama.chat(model=selected_model, messages=st.session_state.messages)
                    ai_text = response['message']['content']
                    st.session_state.messages.append({"role": "assistant", "content": ai_text})
                    
                    # 3. Generate TTS using Coqui TTS
                    output_path = "ai_response.wav"
                    
                    # If voice cloning is enabled, use the voice prompt
                    if st.session_state.voice_prompt_path:
                        # For voice cloning with Coqui TTS, you might need to use a different model
                        # like "tts_models/multilingual/multi-dataset/xtts_v2"
                        tts_model.tts_to_file(
                            text=ai_text,
                            speaker_wav=st.session_state.voice_prompt_path,
                            language="en",
                            file_path=output_path
                        )
                    else:
                        # Regular TTS without voice cloning
                        tts_model.tts_to_file(
                            text=ai_text,
                            file_path=output_path
                        )

                    # 4. Update UI and play audio
                    with chat_container:
                        with st.chat_message("assistant"):
                            st.markdown(ai_text)
                    
                    st.session_state.current_status = "🔊 Speaking..."
                    st.rerun()

                    # Auto-play the response
                    st.audio(output_path, format="audio/wav", autoplay=True)
                    os.remove(output_path)

                # Continue the loop
                time.sleep(1) # Brief pause before listening again
                st.rerun()

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.conversation_active = False
                st.rerun()
        else:
            # No speech detected, continue listening
            st.rerun()

if __name__ == "__main__":
    main()