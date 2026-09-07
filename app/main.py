from datetime import datetime, timezone
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from sqlalchemy import desc, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.auth import create_access_token, get_current_user, get_user_from_token, hash_password, verify_password
from app.config import settings, validate_deployment_settings
from app.database import async_session, engine, get_db, init_db
from app.auth import create_access_token, get_current_user, hash_password, verify_password
from app.config import settings, validate_deployment_settings
from app.database import async_session, get_db, init_db
from app.models import (
    Interaccion,
    Notificacion,
    PacienteRegistro,
    PerfilAcompanamiento,
    RegistroAccesoPaciente,
    SesionPaciente,
    SesionRobot,
    Usuario,
)
from app.robot_controller import robot
from app.websocket_manager import ws_manager

app = FastAPI(title="Chuwibot", version="1.0.0")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def on_startup():
    validate_deployment_settings()
    await init_db()
    async with async_session() as db:
        result = await db.execute(select(Usuario).where(Usuario.username == "admin"))
        if result.scalar_one_or_none() is None:
            password = settings.INITIAL_ADMIN_PASSWORD or "admin123"
            db.add(Usuario(username="admin", password_hash=hash_password(password), rol="admin"))
            await db.commit()


@app.get("/health", include_in_schema=False)
async def healthcheck():
    try:
        async with engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
    except SQLAlchemyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Base de datos no disponible") from exc
    return {"status": "ok"}


class RegisterRequest(BaseModel):
    username: str
    password: str
    rol: str = "operador"


class SupportProfileRequest(BaseModel):
    alias: str
    rango_edad: str = "6-12"
    intereses: list[str] = []
    temas_a_evitar: list[str] = []
    tecnicas_autorizadas: list[str] = ["respiracion", "visualizacion", "anclaje"]
    autorizado_por_adulto: bool = False
    revisado_por_equipo: bool = False


class PatientRegistryRequest(BaseModel):
    nombre_completo: str
    identificador_clinico: str | None = None
    perfil_id: int | None = None


class PatientAccessRequest(BaseModel):
    motivo: str


@app.post("/api/login")
async def login(form: OAuth2PasswordRequestForm = Depends(), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Usuario).where(Usuario.username == form.username))
    user = result.scalar_one_or_none()
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Credenciales incorrectas")
    return {"access_token": create_access_token({"sub": user.username}), "token_type": "bearer", "rol": user.rol}


@app.post("/api/register")
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db), current_user: Usuario = Depends(get_current_user)):
    if current_user.rol != "admin":
        raise HTTPException(status_code=403, detail="Solo admin puede registrar usuarios")
    existing = await db.execute(select(Usuario).where(Usuario.username == req.username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Usuario ya existe")
    if req.rol not in {"admin", "doctor", "operador"}:
        raise HTTPException(status_code=400, detail="Rol no válido")
    db.add(Usuario(username=req.username, password_hash=hash_password(req.password), rol=req.rol))
    await db.commit()
    return {"message": "Usuario creado"}


@app.post("/api/robot/start")
async def robot_start(current_user: Usuario = Depends(get_current_user)):
    if robot.is_running:
        raise HTTPException(status_code=400, detail="Robot ya está activo")
    await robot.start()
    return {"message": "Robot iniciado", "state": robot.state.value}


@app.post("/api/robot/stop")
async def robot_stop(current_user: Usuario = Depends(get_current_user)):
    if not robot.is_running:
        raise HTTPException(status_code=400, detail="Robot ya está detenido")
    await robot.stop()
    return {"message": "Robot detenido", "state": robot.state.value}


@app.get("/api/robot/status")
async def robot_status():
    return {
        "state": robot.state.value,
        "session_id": robot.current_session_id,
        "active_patient": robot.active_patient_code,
        "distance_cm": getattr(robot, "distance_cm", None),
        "emotion": getattr(robot, "current_emotion", None),
        "last_speech": getattr(robot, "last_speech", None),
        "listening": getattr(robot, "listening", False),
    }


def serialize_profile(profile: PerfilAcompanamiento) -> dict:
    return {
        "id": profile.id,
        "alias": profile.alias,
        "rango_edad": profile.rango_edad,
        "intereses": profile.intereses,
        "temas_a_evitar": profile.temas_a_evitar,
        "tecnicas_autorizadas": profile.tecnicas_autorizadas,
        "autorizado_por_adulto": profile.autorizado_por_adulto,
        "revisado_por_equipo": profile.revisado_por_equipo,
    }


@app.get("/api/support-profiles")
async def list_support_profiles(db: AsyncSession = Depends(get_db), current_user: Usuario = Depends(get_current_user)):
    result = await db.execute(select(PerfilAcompanamiento).order_by(PerfilAcompanamiento.alias))
    return [serialize_profile(profile) for profile in result.scalars().all()]


@app.post("/api/support-profiles")
async def create_support_profile(req: SupportProfileRequest, db: AsyncSession = Depends(get_db), current_user: Usuario = Depends(get_current_user)):
    if current_user.rol != "admin":
        raise HTTPException(status_code=403, detail="Solo administración crea perfiles")
    profile = PerfilAcompanamiento(**req.model_dump())
    db.add(profile)
    await db.commit()
    await db.refresh(profile)
    return serialize_profile(profile)


@app.post("/api/support-profiles/{profile_id}/activate")
async def activate_support_profile(profile_id: int, db: AsyncSession = Depends(get_db), current_user: Usuario = Depends(get_current_user)):
    if robot.is_running:
        raise HTTPException(status_code=400, detail="Detén el robot antes de cambiar de perfil")
    profile = await db.get(PerfilAcompanamiento, profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Perfil no encontrado")
    if not profile.autorizado_por_adulto or not profile.revisado_por_equipo:
        raise HTTPException(status_code=400, detail="El perfil necesita autorización del adulto y revisión del equipo de salud")
    robot.set_active_profile(serialize_profile(profile))
    return {"message": f"Perfil de {profile.alias} activo", "profile": serialize_profile(profile)}


def require_clinical_access(user: Usuario):
    if user.rol not in {"doctor", "admin"}:
        raise HTTPException(status_code=403, detail="Acceso reservado para personal clínico autorizado")


def serialize_patient_public(patient: PacienteRegistro) -> dict:
    return {"id": patient.id, "codigo_publico": patient.codigo_publico, "perfil_id": patient.perfil_id}


@app.post("/api/clinical/patients")
async def create_patient_registry(req: PatientRegistryRequest, db: AsyncSession = Depends(get_db), current_user: Usuario = Depends(get_current_user)):
    require_clinical_access(current_user)
    if req.perfil_id and not await db.get(PerfilAcompanamiento, req.perfil_id):
        raise HTTPException(status_code=404, detail="Perfil de acompañamiento no encontrado")
    patient = PacienteRegistro(codigo_publico=f"temporal-{uuid4()}", nombre_completo=req.nombre_completo, identificador_clinico=req.identificador_clinico, perfil_id=req.perfil_id)
    db.add(patient)
    await db.flush()
    patient.codigo_publico = f"Paciente {patient.id}"
    await db.commit()
    await db.refresh(patient)
    return serialize_patient_public(patient)


@app.get("/api/clinical/patients")
async def list_patient_registry(db: AsyncSession = Depends(get_db), current_user: Usuario = Depends(get_current_user)):
    require_clinical_access(current_user)
    result = await db.execute(select(PacienteRegistro).order_by(PacienteRegistro.id))
    return [serialize_patient_public(patient) for patient in result.scalars().all()]


@app.post("/api/clinical/patients/{patient_id}/open")
async def open_patient_registry(patient_id: int, req: PatientAccessRequest, db: AsyncSession = Depends(get_db), current_user: Usuario = Depends(get_current_user)):
    require_clinical_access(current_user)
    patient = await db.get(PacienteRegistro, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    db.add(RegistroAccesoPaciente(paciente_id=patient.id, usuario_id=current_user.id, motivo=req.motivo))
    await db.commit()
    return {"id": patient.id, "codigo_publico": patient.codigo_publico, "nombre_completo": patient.nombre_completo, "identificador_clinico": patient.identificador_clinico, "perfil_id": patient.perfil_id}


@app.post("/api/clinical/patients/{patient_id}/activate")
async def activate_patient(patient_id: int, db: AsyncSession = Depends(get_db), current_user: Usuario = Depends(get_current_user)):
    require_clinical_access(current_user)
    if robot.is_running:
        raise HTTPException(status_code=400, detail="Detén el robot antes de cambiar de paciente")
    patient = await db.get(PacienteRegistro, patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    profile = await db.get(PerfilAcompanamiento, patient.perfil_id) if patient.perfil_id else None
    if profile and (not profile.autorizado_por_adulto or not profile.revisado_por_equipo):
        raise HTTPException(status_code=400, detail="El perfil asignado aún no está autorizado")
    robot.set_active_patient(patient.id, patient.codigo_publico, serialize_profile(profile) if profile else None)
    return {"message": f"{patient.codigo_publico} activo", "patient": serialize_patient_public(patient)}


@app.get("/api/sessions")
async def get_sessions(limit: int = 20, db: AsyncSession = Depends(get_db), current_user: Usuario = Depends(get_current_user)):
    result = await db.execute(select(SesionRobot).order_by(desc(SesionRobot.inicio)).limit(limit))
    sessions = result.scalars().all()
    session_ids = [session.id for session in sessions]
    patient_codes: dict[int, str] = {}
    if session_ids:
        links = await db.execute(select(SesionPaciente.sesion_id, PacienteRegistro.codigo_publico).join(PacienteRegistro, PacienteRegistro.id == SesionPaciente.paciente_id).where(SesionPaciente.sesion_id.in_(session_ids)))
        patient_codes = {session_id: code for session_id, code in links.all()}
    return [{"id": s.id, "paciente": patient_codes.get(s.id, "Paciente no asignado"), "inicio": s.inicio.isoformat(), "fin": s.fin.isoformat() if s.fin else None, "emocion_inicial": s.emocion_inicial} for s in sessions]


@app.get("/api/sessions/{session_id}/interactions")
async def get_interactions(session_id: int, db: AsyncSession = Depends(get_db), current_user: Usuario = Depends(get_current_user)):
    result = await db.execute(select(Interaccion).where(Interaccion.sesion_id == session_id).order_by(Interaccion.timestamp))
    return [{"id": i.id, "rol": i.rol, "texto": i.texto, "emocion": i.emocion, "timestamp": i.timestamp.isoformat()} for i in result.scalars().all()]


@app.get("/api/notifications")
async def get_notifications(limit: int = 50, db: AsyncSession = Depends(get_db), current_user: Usuario = Depends(get_current_user)):
    result = await db.execute(select(Notificacion).order_by(desc(Notificacion.timestamp)).limit(limit))
    return [{"id": n.id, "tipo": n.tipo, "mensaje": n.mensaje, "timestamp": n.timestamp.isoformat(), "leida": n.leida, "sesion_id": n.sesion_id} for n in result.scalars().all()]


@app.post("/api/notifications/{notif_id}/read")
async def mark_notification_read(notif_id: int, db: AsyncSession = Depends(get_db), current_user: Usuario = Depends(get_current_user)):
    notif = await db.get(Notificacion, notif_id)
    if not notif:
        raise HTTPException(status_code=404, detail="Notificación no encontrada")
    notif.leida = True
    await db.commit()
    return {"message": "Marcada como leída"}


@app.get("/api/dashboard/stats")
async def dashboard_stats(db: AsyncSession = Depends(get_db), current_user: Usuario = Depends(get_current_user)):
    total_sessions = await db.scalar(select(func.count(SesionRobot.id)))
    total_interactions = await db.scalar(select(func.count(Interaccion.id)))
    unread_notifs = await db.scalar(select(func.count(Notificacion.id)).where(Notificacion.leida == False))
    emotions_result = await db.execute(select(SesionRobot.emocion_inicial, func.count(SesionRobot.id)).where(SesionRobot.emocion_inicial.isnot(None)).group_by(SesionRobot.emocion_inicial))
    emotions = {row[0]: row[1] for row in emotions_result.all()}
    return {"total_sessions": total_sessions or 0, "total_interactions": total_interactions or 0, "unread_notifications": unread_notifs or 0, "emotions": emotions, "robot_state": robot.state.value}


@app.websocket("/ws/robot")
async def websocket_endpoint(websocket: WebSocket, token: str | None = Query(default=None)):
    if not token:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    try:
        async with async_session() as db:
            await get_user_from_token(token, db)
    except HTTPException:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@app.get("/", response_class=HTMLResponse)
async def page_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def page_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/interaction", response_class=HTMLResponse)
async def page_interaction(request: Request):
    return templates.TemplateResponse("interaction.html", {"request": request})


@app.get("/history", response_class=HTMLResponse)
async def page_history(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/clinical", response_class=HTMLResponse)
async def page_clinical(request: Request):
    return templates.TemplateResponse("clinical.html", {"request": request})
