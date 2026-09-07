from __future__ import annotations

import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Usuario(Base):
    __tablename__ = "usuarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    rol: Mapped[str] = mapped_column(String(20), default="operador")
    creado_en: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class PerfilAcompanamiento(Base):
    """Perfil mínimo, no clínico, aprobado para personalizar a Chuwi."""

    __tablename__ = "perfiles_acompanamiento"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    alias: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    rango_edad: Mapped[str] = mapped_column(String(20), default="6-12")
    intereses: Mapped[list[str]] = mapped_column(JSON, default=list)
    temas_a_evitar: Mapped[list[str]] = mapped_column(JSON, default=list)
    tecnicas_autorizadas: Mapped[list[str]] = mapped_column(JSON, default=lambda: ["respiracion", "visualizacion", "anclaje"])
    autorizado_por_adulto: Mapped[bool] = mapped_column(Boolean, default=False)
    revisado_por_equipo: Mapped[bool] = mapped_column(Boolean, default=False)


class PacienteRegistro(Base):
    """Registro identificable. Nunca debe enviarse a las APIs del dashboard."""

    __tablename__ = "pacientes_registro"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    codigo_publico: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    nombre_completo: Mapped[str] = mapped_column(String(150), nullable=False)
    identificador_clinico: Mapped[str | None] = mapped_column(String(80), unique=True, nullable=True)
    perfil_id: Mapped[int | None] = mapped_column(ForeignKey("perfiles_acompanamiento.id"), nullable=True)
    creado_en: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SesionRobot(Base):
    __tablename__ = "sesiones_robot"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    inicio: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    fin: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    emocion_inicial: Mapped[str | None] = mapped_column(String(50), nullable=True)

    interacciones: Mapped[list[Interaccion]] = relationship(back_populates="sesion")
    notificaciones: Mapped[list[Notificacion]] = relationship(back_populates="sesion")


class SesionPaciente(Base):
    """Enlace privado entre una sesión anónima y un registro de paciente."""

    __tablename__ = "sesiones_paciente"

    sesion_id: Mapped[int] = mapped_column(ForeignKey("sesiones_robot.id"), primary_key=True)
    paciente_id: Mapped[int] = mapped_column(ForeignKey("pacientes_registro.id"))


class RegistroAccesoPaciente(Base):
    """Auditoría de consultas de identidad clínica por profesionales."""

    __tablename__ = "registro_accesos_paciente"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    paciente_id: Mapped[int] = mapped_column(ForeignKey("pacientes_registro.id"))
    usuario_id: Mapped[int] = mapped_column(ForeignKey("usuarios.id"))
    motivo: Mapped[str] = mapped_column(String(250))
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Interaccion(Base):
    __tablename__ = "interacciones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sesion_id: Mapped[int] = mapped_column(ForeignKey("sesiones_robot.id"))
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    rol: Mapped[str] = mapped_column(String(20))
    texto: Mapped[str] = mapped_column(Text)
    emocion: Mapped[str | None] = mapped_column(String(50), nullable=True)

    sesion: Mapped[SesionRobot] = relationship(back_populates="interacciones")


class Notificacion(Base):
    __tablename__ = "notificaciones"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sesion_id: Mapped[int | None] = mapped_column(ForeignKey("sesiones_robot.id"), nullable=True)
    tipo: Mapped[str] = mapped_column(String(30))
    mensaje: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    leida: Mapped[bool] = mapped_column(Boolean, default=False)

    sesion: Mapped[SesionRobot | None] = relationship(back_populates="notificaciones")
