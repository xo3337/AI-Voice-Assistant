
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
