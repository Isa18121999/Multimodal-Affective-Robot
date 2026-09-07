import asyncio
import base64
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from enum import Enum

import cv2
import edge_tts
import httpx
import speech_recognition as sr
from groq import Groq
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import async_session
from app.models import Interaccion, SesionPaciente, SesionRobot
from app.notifications import crear_notificacion
from app.support_protocol import (
    RISK_MESSAGE,
    is_risk_disclosure,
    personalize_message,
    plan_for_emotion,
)
from app.websocket_manager import ws_manager
from src.emotion.chuwi_emotional_manager import ChuwiEmotionalManager
from src.integration.voice_emotion_analyzer import VoiceEmotionAnalyzer


class RobotState(str, Enum):
    STOPPED = "stopped"
    WAITING = "waiting"
    INTERACTING = "interacting"


class RobotController:
    GREETING = "Hola, me llamo Chuwi. ¿Cómo te llamas?"

    def __init__(self, emotional_manager=None):
        self.state = RobotState.STOPPED
        self.current_session_id: int | None = None
        self.historial: list[str] = []
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self.cap = None
        self.active_profile: dict | None = None
        self.current_child_name: str | None = None
        self.active_patient_id: int | None = None
        self.active_patient_code: str | None = None
        self.distance_cm: float | None = None
        self.current_emotion: str | None = None
        self.current_face_signal: dict | None = None
        self.current_voice_signal: dict | None = None
        self.last_speech: str | None = None
        self.listening = False
        self.emotional_manager = emotional_manager or ChuwiEmotionalManager()
        self.voice_emotion_analyzer = VoiceEmotionAnalyzer()

    def set_active_profile(self, profile: dict | None):
        self.active_profile = profile

    def set_active_patient(self, patient_id: int, public_code: str, profile: dict | None):
        self.active_patient_id = patient_id
        self.active_patient_code = public_code
        self.active_profile = profile

    @property
    def is_running(self) -> bool:
        return self.state != RobotState.STOPPED

    async def start(self):
        if self.is_running:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            self._task = None
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        self.state = RobotState.STOPPED
        self.current_session_id = None
        self.current_child_name = None
        self.current_emotion = None
        self.current_face_signal = None
        self.current_voice_signal = None
        self.last_speech = None
        self.listening = False
        await ws_manager.broadcast("robot_state", {"state": self.state.value})

    def update_distance(self, distance_cm: float):
        self.distance_cm = distance_cm
        return self.distance_cm

    def _init_camera(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)

    def _detect_face(self) -> bool:
        self._init_camera()
        ret, frame = self.cap.read()
        if not ret:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        return len(faces) > 0

    def _take_photo(self, path: str):
        self._init_camera()
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(path, frame)

    async def _detect_emotion(self, image_path: str) -> dict:
        """Obtiene la señal facial y la normaliza al vocabulario de Chuwi."""
        try:
            with open(image_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            model = getattr(settings, "GEMINI_MODEL", "gemini-2.5-flash")
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={settings.GEMINI_API_KEY}"
            )
            payload = {
                "contents": [{
                    "parts": [
                        {
                            "text": (
                                "Analiza únicamente la expresión facial visible. "
                                "Responde con una sola palabra en español de esta lista: "
                                "feliz, triste, enojado, asustado, sorprendido, neutral."
                            )
                        },
                        {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
                    ]
                }]
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, timeout=20)
                response.raise_for_status()
                data = response.json()
            raw_emotion = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            emotion = self.emotional_manager.normalize_emotion(raw_emotion)
            if emotion not in {"HAPPY", "SAD", "FRUSTRATED", "FEAR", "ANXIOUS", "CALM"}:
                emotion = "CALM"
            return {"emotion": emotion, "confidence": 0.75, "source": "face"}
        except Exception as exc:
            print(f"Emotion error: {exc}")
            return {"emotion": "CALM", "confidence": 0.0, "source": "face"}

    def _build_modalities(self) -> dict:
        modalities = {}
        if self.current_face_signal:
            modalities["face"] = self.current_face_signal
        if self.current_voice_signal:
            modalities["voice"] = self.current_voice_signal
        return modalities

    def _analyze_emotion(self, text: str) -> dict:
        """Único punto de entrada del controlador al motor emocional multimodal."""
        return self.emotional_manager.analyze_interaction(
            text,
            modalities=self._build_modalities(),
        )

    async def _generate_response(self, emotional_result: dict, text: str | None = None) -> str:
        if text:
            self.historial.append(f"Usuario: {text}")
        context = "\n".join(self.historial[-6:])
        profile_note = ""
        if self.active_profile:
            interests = ", ".join(self.active_profile.get("intereses", []))
            topics = ", ".join(self.active_profile.get("temas_a_evitar", []))
            profile_note = (
                f"Usa lenguaje apropiado para {self.active_profile.get('rango_edad', '6-12')} años. "
                f"Intereses: {interests or 'ninguno'}. Temas a evitar: {topics or 'ninguno'}.\n"
            )
        emotion = emotional_result["emotion"]
        strategy = emotional_result.get("strategy") or "acompañamiento"
        prompt = (
            "Eres Chuwi, un robot social de apoyo emocional para niños hospitalizados.\n"
            "Responde de forma corta, cálida, amigable, natural y apropiada para un niño.\n"
            "No diagnostiques, no prometas curas, no des terapia clínica ni sugieras medicamentos.\n"
            f"{profile_note}"
            f"Emoción multimodal detectada: {emotion}. Estrategia: {strategy}.\n"
            f"Contexto:\n{context}"
        )
        try:
            client = Groq(api_key=settings.GROQ_API_KEY)
            response = await asyncio.to_thread(
                client.chat.completions.create,
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=100,
            )
            text_response = response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"Groq error: {exc}")
            fallback = {
                "FEAR": "Estoy contigo. Podemos ir paso a paso.",
                "SAD": "Estoy aquí para escucharte. ¿Quieres contarme cómo te sientes?",
                "ANXIOUS": "Podemos respirar juntos y conversar tranquilamente.",
                "FRUSTRATED": "Entiendo que algo puede haberte molestado. Podemos ir paso a paso.",
                "HAPPY": "¡Qué bueno! Me alegra compartir este momento contigo.",
            }
            text_response = fallback.get(emotion, "Te estoy escuchando y quiero acompañarte.")
        self.historial.append(f"Chuwi: {text_response}")
        return text_response

    async def _speak(self, text: str, db: AsyncSession):
        self.last_speech = text
        await crear_notificacion(
            db,
            tipo="hablar",
            mensaje=f"Chuwibot dice: {text[:100]}",
            sesion_id=self.current_session_id,
        )
        await ws_manager.broadcast(
            "robot_speak",
            {"text": text, "sesion_id": self.current_session_id},
        )
        try:
            communicate = edge_tts.Communicate(text=f"{text}...", voice="es-MX-DaliaNeural")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as file:
                path = file.name
            try:
                await communicate.save(path)
                await asyncio.to_thread(
                    subprocess.run,
                    ["mpg123", "-a", "alsa", path],
                    check=False,
                )
            finally:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
        except Exception as exc:
            print(f"TTS error: {exc}")

    async def _listen(self) -> str:
        self.listening = True
        self.current_voice_signal = None
        await ws_manager.broadcast(
            "robot_listening",
            {"listening": True, "sesion_id": self.current_session_id},
        )

        def listen_blocking():
            recognizer = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=1)
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=6)
                return audio, recognizer.recognize_google(audio, language="es-PE")
            except Exception as exc:
                print(f"Listen error: {exc}")
                return None, ""

        audio, text = await asyncio.to_thread(listen_blocking)
        if audio is not None:
            self.current_voice_signal = self.voice_emotion_analyzer.analyze(audio)
        self.listening = False
        await ws_manager.broadcast(
            "robot_listening",
            {"listening": False, "sesion_id": self.current_session_id},
        )
        if text:
            await ws_manager.broadcast(
                "robot_heard",
                {"text": text, "sesion_id": self.current_session_id},
            )
        return text.strip()

    @staticmethod
    def _extract_name(text: str) -> str | None:
        match = re.search(
            r"(?:me llamo|mi nombre es|soy)\s+([A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,30})",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).capitalize()
        words = text.strip().split()
        if len(words) == 1 and words[0].isalpha() and len(words[0]) <= 30:
            return words[0].capitalize()
        return None

    async def _save_interaction(self, db: AsyncSession, rol: str, texto: str, emocion: str | None = None):
        if rol == "usuario":
            texto = re.sub(
                r"\b(me llamo|mi nombre es)\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{2,30}\b",
                "[identidad omitida]",
                texto,
                flags=re.IGNORECASE,
            )
        db.add(
            Interaccion(
                sesion_id=self.current_session_id,
                rol=rol,
                texto=texto,
                emocion=emocion,
            )
        )
        await db.commit()
        await ws_manager.broadcast(
            "robot_interaction",
            {
                "rol": rol,
                "texto": texto,
                "emocion": emocion,
                "sesion_id": self.current_session_id,
            },
        )

    async def _run_loop(self):
        self.state = RobotState.WAITING
        await ws_manager.broadcast("robot_state", {"state": self.state.value})
        while not self._stop_event.is_set():
            try:
                has_face = await asyncio.to_thread(self._detect_face)
                if has_face:
                    await self._interact()
                await asyncio.sleep(0.3)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"Robot loop error: {exc}")
                await asyncio.sleep(1)
        self.state = RobotState.STOPPED

    async def _interact(self):
        self.state = RobotState.INTERACTING
        self.current_child_name = None
        self.current_emotion = None
        self.current_face_signal = None
        self.current_voice_signal = None
        await ws_manager.broadcast("robot_state", {"state": self.state.value})

        async with async_session() as db:
            session = SesionRobot()
            db.add(session)
            await db.commit()
            await db.refresh(session)
            self.current_session_id = session.id

            if self.active_patient_id:
                db.add(SesionPaciente(session_id=session.id, paciente_id=self.active_patient_id))
                await db.commit()

            await crear_notificacion(
                db,
                tipo="activacion",
                mensaje="Persona detectada - interacción iniciada",
                sesion_id=session.id,
            )

            self.historial = []
            await self._save_interaction(db, "robot", self.GREETING)
            await self._speak(self.GREETING, db)

            name_reply = await self._listen()
            self.current_child_name = self._extract_name(name_reply) if name_reply else None
            if name_reply:
                await self._save_interaction(db, "usuario", "[nombre compartido]")

            welcome = (
                f"Mucho gusto, {self.current_child_name}. Voy a acompañarte un momento."
                if self.current_child_name
                else "Mucho gusto. Voy a acompañarte un momento."
            )
            await self._save_interaction(db, "robot", "Inicio de acompañamiento después de preguntar el nombre")
            await self._speak(welcome, db)

            descriptor, photo_path = tempfile.mkstemp(suffix=".jpg")
            os.close(descriptor)
            try:
                await asyncio.to_thread(self._take_photo, photo_path)
                self.current_face_signal = await self._detect_emotion(photo_path)
            finally:
                try:
                    os.remove(photo_path)
                except FileNotFoundError:
                    pass

            initial_result = self._analyze_emotion(name_reply or "")
            self.current_emotion = initial_result["emotion"]
            await ws_manager.broadcast(
                "robot_emotion",
                {
                    "emotion": self.current_emotion,
                    "confidence": initial_result["confidence"],
                    "scores": initial_result["scores"],
                    "sesion_id": session.id,
                },
            )
            session.emocion_inicial = self.current_emotion
            await db.commit()

            allowed = self.active_profile.get("tecnicas_autorizadas") if self.active_profile else None
            support = plan_for_emotion(self.current_emotion, allowed)
            support_message = personalize_message(support, self.active_profile)
            spoken_support = (
                f"{self.current_child_name}, {support_message}"
                if self.current_child_name and not support_message.startswith(f"{self.current_child_name},")
                else support_message
            )
            await self._save_interaction(db, "robot", support_message, self.current_emotion)
            await self._speak(spoken_support, db)

            retries = 0
            while not self._stop_event.is_set():
                text = await self._listen()
                if not text:
                    retries += 1
                    if retries >= 3:
                        farewell = "No te escuché bien, volveré a esperar."
                        await self._save_interaction(db, "robot", farewell)
                        await self._speak(farewell, db)
                        break
                    continue

                retries = 0
                await self._save_interaction(db, "usuario", text)

                if is_risk_disclosure(text):
                    await crear_notificacion(
                        db,
                        tipo="alerta",
                        mensaje="Posible situación de riesgo: se requiere acompañamiento inmediato de un adulto.",
                        sesion_id=session.id,
                    )
                    await self._save_interaction(db, "robot", RISK_MESSAGE)
                    await self._speak(RISK_MESSAGE, db)
                    break

                if any(word in text.lower() for word in ("chau", "adiós", "adios")):
                    goodbye = "Fue lindo hablar contigo. Nos vemos pronto."
                    await self._save_interaction(db, "robot", goodbye)
                    await self._speak(goodbye, db)
                    break

                emotional_result = self._analyze_emotion(text)
                self.current_emotion = emotional_result["emotion"]
                await ws_manager.broadcast(
                    "robot_emotion",
                    {
                        "emotion": self.current_emotion,
                        "confidence": emotional_result["confidence"],
                        "scores": emotional_result["scores"],
                        "sesion_id": session.id,
                    },
                )
                response = await self._generate_response(emotional_result, text)
                await self._save_interaction(db, "robot", response, self.current_emotion)
                spoken_response = (
                    f"{self.current_child_name}, {response}"
                    if self.current_child_name and not response.startswith(f"{self.current_child_name},")
                    else response
                )
                await self._speak(spoken_response, db)

            session.fin = datetime.now(timezone.utc)
            await db.commit()

        self.current_session_id = None
        self.current_child_name = None
        self.current_emotion = None
        self.current_face_signal = None
        self.current_voice_signal = None
        self.listening = False
        self.state = RobotState.WAITING
        await ws_manager.broadcast("robot_state", {"state": self.state.value})
        await asyncio.sleep(5)


robot = RobotController()
