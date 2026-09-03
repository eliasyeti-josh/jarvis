"""
================================================================================
 J.A.R.V.I.S. — Your Personal Voice Assistant
 Voice In (SpeechRecognition) -> Think (Gemini 2.5 Flash) -> Voice Out (pyttsx3)
================================================================================

SETUP INSTRUCTIONS (Windows/Mac/Linux):

1. Install Python 3.10+ from python.org (ensure "Add to PATH" is checked).

2. Install dependencies (open Command Prompt / PowerShell / Terminal):

    pip install SpeechRecognition google-genai pyttsx3 pyaudio python-dotenv

   NOTE: If "pip install pyaudio" fails on Windows with a build error, run:
    pip install pipwin
    pipwin install pyaudio

3. Create a .env file in the same directory as jarvis.py with your API key:

    GEMINI_API_KEY=your-actual-api-key-here
    ASSISTANT_NAME=JARVIS
    USER_NAME=Sir

   OR set GEMINI_API_KEY as a permanent environment variable.

4. Run the assistant:

    python jarvis.py

5. Start speaking! Say your exit phrase to close the program.

================================================================================
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import Optional

import speech_recognition as sr
import pyttsx3
from google import genai
from google.genai import types

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ==============================================================================
# CONFIGURATION & PERSONALIZATION
# ==============================================================================

class Config:
    """Personalization settings for your JARVIS."""
    
    # AI Model
    GEMINI_MODEL = "gemini-2.5-flash"
    
    # Your assistant's personality
    ASSISTANT_NAME = os.environ.get("ASSISTANT_NAME", "JARVIS").strip()
    USER_NAME = os.environ.get("USER_NAME", "Sir").strip()
    
    # Wake word and exit phrases
    WAKE_WORD = os.environ.get("WAKE_WORD", "").strip().lower()  # Empty = always listening
    EXIT_PHRASES = tuple(
        phrase.strip().lower() 
        for phrase in os.environ.get(
            "EXIT_PHRASES", 
            "shut down,goodbye,exit,power down,see you,goodbye jarvis"
        ).split(",")
    )
    
    # Audio settings
    LISTEN_TIMEOUT = 8  # Seconds to wait for speech to start
    PHRASE_TIME_LIMIT = 15  # Max seconds per phrase
    AMBIENT_NOISE_DURATION = 1.5  # Calibration duration
    
    # Voice settings
    SPEECH_RATE = int(os.environ.get("SPEECH_RATE", "178"))
    VOICE_GENDER = os.environ.get("VOICE_GENDER", "male").lower()
    
    # Temperature: 0.0 = deterministic, 1.0 = creative
    TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.8"))
    MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "400"))
    
    @staticmethod
    def get_system_instruction() -> str:
        """Build the system prompt from environment or use default."""
        custom_instruction = os.environ.get("SYSTEM_INSTRUCTION", "").strip()
        
        if custom_instruction:
            return custom_instruction
        
        # Default sophisticated assistant personality
        return (
            f"You are {Config.ASSISTANT_NAME}, a sophisticated, witty AI assistant. "
            f"Address the user as '{Config.USER_NAME}' occasionally. Keep responses "
            "conversational, concise (2-4 sentences unless asked for detail), and "
            "engaging. You are speaking your responses aloud, so avoid markdown, "
            "bullet points, asterisks, or any text formatting that cannot be spoken "
            "naturally. Be helpful, intelligent, and maintain a refined tone."
        )


# ==============================================================================
# VOICE OUTPUT ENGINE (pyttsx3)
# ==============================================================================

class VoiceOutput:
    """Wraps pyttsx3 to speak text aloud."""

    def __init__(self):
        try:
            self.engine = pyttsx3.init(driverName="sapi5" if sys.platform == "win32" else None)
            self._configure_voice()
            self._configure_rate_and_volume()
            self.is_available = True
        except Exception as e:
            print(f"WARNING: Text-to-speech initialization failed: {e}")
            self.is_available = False

    def _configure_voice(self):
        """Selects a voice matching the preferred gender."""
        if not self.is_available:
            return
            
        try:
            voices = self.engine.getProperty("voices")
            selected_voice_id = None
            gender = Config.VOICE_GENDER

            for voice in voices:
                name = (voice.name or "").lower()
                voice_id = (voice.id or "").lower()
                is_flagged_male = getattr(voice, "gender", None) == "VoiceGenderMale"
                is_flagged_female = getattr(voice, "gender", None) == "VoiceGenderFemale"
                
                has_gender_keyword = False
                if gender == "male":
                    has_gender_keyword = any(
                        keyword in name or keyword in voice_id
                        for keyword in ("david", "mark", "george", "male", "guy", "ryan", "james")
                    )
                    if is_flagged_male or has_gender_keyword:
                        selected_voice_id = voice.id
                        break
                elif gender == "female":
                    has_gender_keyword = any(
                        keyword in name or keyword in voice_id
                        for keyword in ("zira", "cortana", "susan", "female", "lady", "victoria")
                    )
                    if is_flagged_female or has_gender_keyword:
                        selected_voice_id = voice.id
                        break

            if selected_voice_id is None and voices:
                selected_voice_id = voices[0].id

            if selected_voice_id:
                self.engine.setProperty("voice", selected_voice_id)
        except Exception as e:
            print(f"WARNING: Could not configure voice: {e}")

    def _configure_rate_and_volume(self):
        """Sets speaking pace and volume."""
        if not self.is_available:
            return
            
        try:
            self.engine.setProperty("rate", Config.SPEECH_RATE)
            self.engine.setProperty("volume", 1.0)
        except Exception as e:
            print(f"WARNING: Could not set voice rate/volume: {e}")

    def speak(self, text: str):
        """Speaks the given text aloud, blocking until finished."""
        if not text:
            return
        
        print(f"{Config.ASSISTANT_NAME}: {text}\n")
        
        if self.is_available:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"ERROR: Could not speak: {e}")
        else:
            print("(Text-to-speech unavailable; would have spoken above)")


# ==============================================================================
# AUDIO INPUT / SPEECH RECOGNITION
# ==============================================================================

class VoiceInput:
    """Wraps SpeechRecognition to continuously listen and transcribe speech."""

    def __init__(self):
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            self._calibrate_for_ambient_noise()
            self.is_available = True
        except Exception as e:
            print(f"ERROR: Microphone/recognizer initialization failed: {e}")
            self.is_available = False

    def _calibrate_for_ambient_noise(self):
        """Samples ambient room noise for better filtering."""
        if not self.is_available:
            return
            
        print("🎤 Calibrating microphone for ambient noise... please stay quiet.")
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(
                    source, duration=Config.AMBIENT_NOISE_DURATION
                )
            print(f"✓ Calibration complete. Energy threshold: "
                  f"{self.recognizer.energy_threshold:.1f}\n")
        except Exception as e:
            print(f"WARNING: Calibration failed: {e}\n")

    def listen_and_transcribe(self) -> Optional[str]:
        """
        Listens on the microphone for one spoken phrase and returns the
        transcribed text, or None if nothing usable was captured.
        """
        if not self.is_available:
            return None
            
        try:
            with self.microphone as source:
                try:
                    audio = self.recognizer.listen(
                        source,
                        timeout=Config.LISTEN_TIMEOUT,
                        phrase_time_limit=Config.PHRASE_TIME_LIMIT,
                    )
                except sr.WaitTimeoutError:
                    return None

            text = self.recognizer.recognize_google(audio)
            print(f"👤 You said: {text}")
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"[Speech recognition service error: {e}]")
            return None
        except Exception as e:
            print(f"[Unexpected audio error: {e}]")
            return None


# ==============================================================================
# THINKING BACKEND (Gemini)
# ==============================================================================

class ThinkingCore:
    """Wraps the google-genai client to generate intelligent responses."""

    def __init__(self):
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print(
                "\n❌ ERROR: GEMINI_API_KEY is not set.\n"
                "\nSet it one of these ways:\n"
                "  1. Create .env file with: GEMINI_API_KEY=your-key-here\n"
                "  2. Set environment variable (Windows):\n"
                "     setx GEMINI_API_KEY \"your-key-here\"\n"
                "  3. Set environment variable (Mac/Linux):\n"
                "     export GEMINI_API_KEY=\"your-key-here\"\n"
            )
            sys.exit(1)

        try:
            self.client = genai.Client(api_key=api_key)
            self.chat = self.client.chats.create(
                model=Config.GEMINI_MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=Config.get_system_instruction(),
                    max_output_tokens=Config.MAX_TOKENS,
                    temperature=Config.TEMPERATURE,
                ),
            )
        except Exception as e:
            print(f"ERROR: Failed to initialize Gemini: {e}")
            sys.exit(1)

    def generate_reply(self, user_text: str) -> str:
        """Sends user_text to Gemini and returns the model's text reply."""
        try:
            response = self.chat.send_message(user_text)
            reply = (response.text or "").strip()
            if not reply:
                reply = f"I didn't catch that, {Config.USER_NAME}. Could you repeat?"
            return reply
        except Exception as e:
            raise e


# ==============================================================================
# MAIN APPLICATION LOOP
# ==============================================================================

BOOT_LINES = [
    "All systems online. Good to be back.",
    "Systems nominal. Standing by and listening.",
    "Online. Shall we get started?",
    "Ready to assist you.",
]

IDLE_TIMEOUT_ACKS = [
    "Still here whenever you need me.",
    "I'm listening.",
    "Standing by.",
]


def run_jarvis():
    print("\n" + "=" * 70)
    print(f" 🤖 {Config.ASSISTANT_NAME.upper()} VOICE CORE — BOOTING")
    print("=" * 70)
    print(f" Assistant: {Config.ASSISTANT_NAME}")
    print(f" User: {Config.USER_NAME}")
    print(f" Model: {Config.GEMINI_MODEL}")
    print(f" Temperature: {Config.TEMPERATURE} | Max tokens: {Config.MAX_TOKENS}")
    if Config.WAKE_WORD:
        print(f" Wake word: '{Config.WAKE_WORD}'")
    print(f" Exit phrases: {', '.join(Config.EXIT_PHRASES[:3])}...")
    print("=" * 70 + "\n")

    try:
        voice_out = VoiceOutput()
        voice_in = VoiceInput()
        brain = ThinkingCore()
    except Exception as startup_error:
        print(f"FATAL: {Config.ASSISTANT_NAME} failed to initialize: {startup_error}")
        sys.exit(1)

    voice_out.speak(random.choice(BOOT_LINES))

    consecutive_silence_count = 0
    conversation_turn = 0

    while True:
        try:
            user_text = voice_in.listen_and_transcribe()

            if user_text is None:
                consecutive_silence_count += 1
                if consecutive_silence_count % 20 == 0:
                    voice_out.speak(random.choice(IDLE_TIMEOUT_ACKS))
                continue

            consecutive_silence_count = 0
            conversation_turn += 1
            lowered = user_text.lower().strip()

            # Check for wake word (if configured)
            if Config.WAKE_WORD and Config.WAKE_WORD not in lowered:
                continue

            # Check for exit phrases
            if any(phrase in lowered for phrase in Config.EXIT_PHRASES):
                voice_out.speak(f"Shutting down. It's been a pleasure. {Config.ASSISTANT_NAME} out.")
                print(f"\n✓ {Config.ASSISTANT_NAME} closed gracefully.")
                break

            try:
                print(f"[Turn {conversation_turn}] Processing...")
                reply_text = brain.generate_reply(user_text)
            except Exception as api_error:
                print(f"[Gemini API error: {api_error}]")
                voice_out.speak(
                    f"My apologies — I'm experiencing trouble with my thinking core. "
                    f"Do try again, {Config.USER_NAME}."
                )
                continue

            voice_out.speak(reply_text)

        except KeyboardInterrupt:
            voice_out.speak(f"Powering down. Goodbye, {Config.USER_NAME}.")
            print(f"\n✓ {Config.ASSISTANT_NAME} terminated by user.")
            break

        except sr.RequestError as network_error:
            print(f"[Network/microphone error: {network_error}]")
            voice_out.speak("I appear to have lost my connection momentarily.")
            time.sleep(2)
            continue

        except Exception as unexpected_error:
            print(f"[Unexpected error, recovering: {unexpected_error}]")
            time.sleep(1)
            continue


def create_sample_env():
    """Creates a sample .env file if one doesn't exist."""
    env_path = Path(".env")
    if not env_path.exists():
        sample_content = """# J.A.R.V.I.S. Configuration

# REQUIRED: Your Gemini API key
GEMINI_API_KEY=your-actual-api-key-here

# PERSONALIZATION
ASSISTANT_NAME=JARVIS
USER_NAME=Sir

# Optional: Custom wake word (leave empty to always listen)
WAKE_WORD=

# Optional: Exit phrases (comma-separated)
EXIT_PHRASES=shut down,goodbye,exit,power down,goodbye jarvis

# VOICE SETTINGS
VOICE_GENDER=male  # or "female"
SPEECH_RATE=178    # Speed (100-300, higher = faster)

# AI PERSONALITY
TEMPERATURE=0.8    # 0.0 = precise, 1.0 = creative
MAX_TOKENS=400     # Max response length

# OPTIONAL: Custom system instruction to fully customize personality
# SYSTEM_INSTRUCTION=You are a helpful AI assistant...
"""
        env_path.write_text(sample_content)
        print(f"✓ Created sample .env file. Edit it to personalize your assistant!\n")


if __name__ == "__main__":
    create_sample_env()
    run_jarvis()
