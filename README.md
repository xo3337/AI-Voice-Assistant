
# 📣 AI Voice Assistant – Python (Whisper + Cohere + gTTS)

This project is a voice assistant built with Python that:

- 🎙️ Records your voice from the microphone  
- 🧠 Transcribes speech to text using [Whisper](https://github.com/openai/whisper)  
- 🤖 Sends the text to [Cohere](https://cohere.com/) to generate a response  
- 🔊 Converts the response back to speech using [gTTS](https://pypi.org/project/gTTS/)  
- ✅ Plays the generated voice response out loud  

---

## 🛠️ Features

- Real-time microphone input
- Intelligent response generation
- Natural-sounding text-to-speech
- Retry and fallback logic for robustness
- Designed for **Windows** (can work on Linux/macOS with small changes)

---

## 📦 Requirements

Install Python packages:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```txt
sounddevice
scipy
openai-whisper
cohere
gTTS
pygame
numpy
imageio-ffmpeg
```

---

## 🔧 Setup Instructions

### 1. Install FFmpeg (Required for Whisper + audio encoding)

Download and install FFmpeg from:  
👉 [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

✅ Make sure `ffmpeg.exe` is added to your Windows system PATH.

---

### 2. Set Your Cohere API Key

You can:

- Set the environment variable:

```bash
set COHERE_API_KEY=your_api_key_here
```

OR

- Edit this line in the script:

```python
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "your_api_key_here")
```

---

### 3. Run the Assistant

```bash
python app.py
```

You’ll hear a prompt to speak. The assistant will:

1. Record audio  
2. Transcribe your speech  
3. Generate a smart reply  
4. Read it back to you  

---

## 🗣️ Language Support

By default, this version is tuned for **English**. To change to **Arabic**:

### 🔁 How to Switch to Arabic:

1. **Use `whisper.load_model("base")` instead of `"base.en"`**  
2. Set `language='ar'` in the `model.transcribe()` call:

```python
result = model.transcribe(
    audio_path,
    language='ar',  # Change from 'en' to 'ar'
    ...
)
```

3. Set TTS to Arabic:

```python
gTTS(text=text, lang='ar', slow=False)
```
## 👨‍💻 the code
```python
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import cohere
from gtts import gTTS
import os
import sys
import numpy as np
import imageio_ffmpeg
import time
import logging
import pygame  # Added for reliable audio playback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize pygame mixer for audio playback
pygame.mixer.init()

def record_audio(output_path="input_audio.wav", duration=5, sample_rate=16000):
    """Reliable audio recording with comprehensive error handling"""
    try:
        logger.info("Initializing audio recording...")
        
        # Reset audio system
        sd._terminate()
        sd._initialize()
        
        # List and select input device
        devices = sd.query_devices()
        input_device = None
        
        logger.debug("Available audio devices:")
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                logger.debug(f"{i}: {dev['name']}")
                if input_device is None and ('Microphone' in dev['name'] or 'Input' in dev['name']):
                    input_device = i
        
        if input_device is None:
            raise RuntimeError("No suitable input device found")
        
        logger.info(f"Selected input device: {devices[input_device]['name']}")
        
        # Configure recording parameters
        sd.default.device = input_device
        sd.default.samplerate = sample_rate
        sd.default.channels = 1
        sd.default.dtype = 'int16'
        
        logger.info(f"Recording {duration} seconds... Speak now!")
        recording = sd.rec(int(duration * sample_rate), blocking=False)
        
        # Visual countdown with early termination check
        start_time = time.time()
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            print(f"\rRecording... {duration - elapsed:.1f}s remaining", end='')
            time.sleep(0.1)
            
            # Check if stream is still active
            if not any(recording):
                raise RuntimeError("Audio stream terminated unexpectedly")
        
        # Finalize recording
        sd.stop()
        write(output_path, sample_rate, recording)
        
        # Verify recording quality
        audio_data = np.frombuffer(recording, dtype=np.int16)
        if np.max(np.abs(audio_data)) < 1000:  # Adjust threshold as needed
            logger.warning("Recording may be too quiet - check microphone levels")
        
        logger.info(f"Successfully saved recording to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Recording failed: {str(e)}")
        logger.info("Troubleshooting steps:")
        logger.info("1. Check microphone connection")
        logger.info("2. Verify microphone permissions")
        logger.info("3. Try a different USB port")
        return False

def transcribe_audio(audio_path):
    """Robust audio transcription with Whisper"""
    try:
        logger.info("Starting audio transcription...")
        
        # Ensure ffmpeg is available
        ffmpeg_path = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
        if ffmpeg_path not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + ffmpeg_path
        
        # Load Whisper model with error handling
        try:
            model = whisper.load_model("base.en")
        except Exception as e:
            logger.warning(f"Failed to load base.en model: {str(e)}")
            model = whisper.load_model("base")
        
        # Transcribe with conservative settings
        result = model.transcribe(
            audio_path,
            language='en',
            fp16=False,
            temperature=0.2,
            initial_prompt="Transcribe this audio clearly and accurately."
        )
        
        text = result['text'].strip()
        if not text:
            logger.warning("No speech detected in audio")
            return "Hello"  # Fallback text
        
        logger.info(f"Transcription: {text}")
        return text
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        return "Could not understand the audio"

def generate_response(prompt, api_key):
    """Generate AI response with Cohere API"""
    try:
        logger.info("Generating AI response...")
        
        # Initialize Cohere client with timeout
        co = cohere.Client(api_key, timeout=30)
        
        # Generate response with compatible parameters
        response = co.generate(
            model="command",  # Stable model name
            prompt=prompt,
            temperature=0.7,
            max_tokens=150
        )
        
        # Validate response
        if not response.generations:
            raise ValueError("Empty response from API")
        
        response_text = response.generations[0].text.strip()
        logger.info(f"AI Response: {response_text}")
        return response_text
        
    except cohere.CohereAPIError as e:
        logger.error(f"Cohere API Error: {str(e)}")
        return "I'm having trouble connecting to the AI service right now."
    except Exception as e:
        logger.error(f"Response generation failed: {str(e)}")
        return "Let me think differently about that. " + prompt[:100]  # Fallback

def text_to_speech(text, output_path="response.mp3"):
    """Convert text to speech with robust error handling"""
    try:
        if not text or len(text.strip()) < 3:
            raise ValueError("Text too short for TTS")
            
        logger.info("Converting text to speech...")
        
        # Generate speech with timeout
        tts = gTTS(
            text=text,
            lang='en',
            slow=False,
            timeout=15
        )
        
        # Save to temporary file first
        temp_path = f"temp_{output_path}"
        tts.save(temp_path)
        
        # Verify file was created
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 1024:
            raise IOError("Generated audio file is invalid")
        
        # Atomic file replacement
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(temp_path, output_path)
        
        # Play audio using pygame for reliable playback
        play_audio(output_path)
        logger.info(f"Audio response saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Text-to-speech failed: {str(e)}")

def play_audio(filepath):
    """Reliable audio playback using pygame"""
    try:
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
            
    except Exception as e:
        logger.error(f"Audio playback failed: {str(e)}")
        # Fallback to system playback
        if sys.platform == "win32":
            os.system(f'start "{filepath}"')
        elif sys.platform == "darwin":
            os.system(f'afplay "{filepath}"')
        else:
            os.system(f'xdg-open "{filepath}"')

def run_pipeline(api_key, duration=5, max_retries=2):
    """Main execution pipeline with retry logic"""
    # Clean previous files
    for f in ["input_audio.wav", "response.mp3", "temp_response.mp3"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass
    
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"\n=== Attempt {attempt + 1} ===")
            
            # 1. Record audio
            if not record_audio(duration=duration):
                if attempt == max_retries:
                    text_to_speech("Failed to record audio. Please check your microphone.")
                continue
                
            # 2. Transcribe audio
            user_input = transcribe_audio("input_audio.wav")
            if not user_input or user_input.lower() == "could not understand audio":
                if attempt == max_retries:
                    text_to_speech("Sorry, I couldn't understand you. Please try again.")
                continue
                
            # 3. Generate response
            ai_response = generate_response(user_input, api_key)
            if not ai_response:
                if attempt == max_retries:
                    text_to_speech("I couldn't generate a response. Please try again later.")
                continue
                
            # 4. Speak response
            text_to_speech(ai_response)
            return  # Success
            
        except KeyboardInterrupt:
            logger.info("\nOperation cancelled by user")
            return
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            if attempt == max_retries:
                text_to_speech("Sorry, I encountered a technical problem. Please try again later.")
            time.sleep(1)  # Brief pause before retry

if __name__ == "__main__":
    # Get API key (prefer environment variable)
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "your key")
    
    if not COHERE_API_KEY:
        logger.error("COHERE_API_KEY environment variable not set")
        sys.exit(1)
        
    print("\n=== AI Voice Assistant ===")
    print("Press Ctrl+C to exit\n")
    
    try:
        run_pipeline(COHERE_API_KEY, duration=5)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        pygame.mixer.quit()
        print("\nAssistant session ended")
```
---

## 🔍 Troubleshooting

- ❌ **No audio detected**  
  Make sure your microphone is working and allowed in system settings.

- ❌ **Whisper not transcribing**  
  Match input sample rate and channels:  
  - Use **mono (1 channel)**  
  - Use **16000 Hz** sample rate  
  - Use `int16` format

- ❌ **Cohere key error**  
  Double-check your API key and internet connection.

- ❌ **gTTS too short error**  
  Ensure the generated response text is at least a few characters.

---

## 🧠 Example Output

```
🎙️ You: "What's the capital of Egypt?"
🤖 AI: "The capital of Egypt is Cairo."
🔊 *Audio plays back response*
```

---

## 🙋 Credits

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Cohere API](https://cohere.com/)
- [gTTS](https://pypi.org/project/gTTS/)
- [pygame](https://www.pygame.org/)
