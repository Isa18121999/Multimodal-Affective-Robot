# CHUWI – Robot Social para Regulación Emocional
Chuwi es un robot social diseñado para brindar apoyo emocional inmediato a personas que experimentan episodios de ansiedad o estrés.
El sistema reconoce emociones mediante visión computacional y una red neuronal (CNN) y responde con mensajes calmantes generados por voz (TTS).

El objetivo principal es ofrecer una interacción empática, accesible y en tiempo real, sin reemplazar ayuda profesional, sino acompañando al usuario mientras recibe atención adecuada
Al detectar una persona, Chuwi inicia la conversación con: "Hola, me llamo
Chuwi. Me alegra conocerte." La aplicación web registra sesiones e
interacciones en PostgreSQL. Esos registros sirven para auditoría y métricas; aun no constituyen un conjunto de datos validado para entrenar o evaluar un detector de emociones en niños.

## Despliegue con Docker Compose

1. Copia la configuración y sustituye **todas** las credenciales de ejemplo:

   ```bash
   cp .env.example .env
   ```

2. Inicia la aplicación y PostgreSQL:

   ```bash
   docker compose up --build -d
   ```

3. Comprueba que ambos servicios estén sanos y abre `http://localhost:8000`:

   ```bash
   docker compose ps
   curl --fail http://localhost:8000/health
   ```

El contenedor web ejecuta la aplicación sin recarga automática y sin montar el
código fuente. En producción, `INITIAL_ADMIN_PASSWORD` y un `SECRET_KEY`
aleatorio son obligatorios; la cuenta `admin` se crea únicamente si todavía no
existe. PostgreSQL conserva los datos en el volumen nombrado `pgdata`.
