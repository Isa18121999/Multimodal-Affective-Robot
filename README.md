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

## Benchmark reproducible de reconocimiento facial

El repositorio incluye un benchmark reproducible para comparar modelos ligeros de estimación de expresiones faciales:

- MobileNetV3-Large
- EfficientNet-B0
- MobileViT-XXS
- DeiT-Tiny

Todos utilizan las mismas particiones auditadas, entrada 224x224, semilla 42, class weighting y selección del checkpoint por Macro-F1 de validation. El benchmark reporta métricas globales y una evaluación separada del origen Children_Emotion.

Las imágenes auxiliary_age_faces no participan en el entrenamiento porque no tienen etiquetas emocionales validadas. La evaluación infantil no se presenta como independiente por sujeto porque el manifest no contiene un subject-ID confiable.

Instalación de dependencias de entrenamiento:

    pip install -r requirements-training.txt

Ejecución:

    python scripts/benchmark_facial_models.py --dataset-root /ruta/al/dataset_unificado_emociones

Los artefactos se generan en artifacts/facial_benchmark/. La elección del modelo final se realizará después de comparar Macro-F1, F1 por clase, desempeño infantil, número de parámetros y velocidad de inferencia.

