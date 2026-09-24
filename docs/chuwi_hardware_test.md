# Prueba autónoma de Chuwi en Raspberry Pi 5

La validación física debe ejecutarse directamente en la Raspberry Pi 5, sin
depender de una computadora externa.

## Ejecución

Desde el repositorio:

    chmod +x scripts/test_chuwi_hardware.py
    python3 scripts/test_chuwi_hardware.py

Si se usa el entorno virtual del proyecto:

    source robot_env/bin/activate
    python scripts/test_chuwi_hardware.py

## Qué verifica

- modelo del dispositivo;
- cámara USB mediante /dev/video*;
- micrófono mediante ALSA/arecord;
- salida de audio;
- PyTorch, torchvision, Pillow y OpenCV;
- disponibilidad de timm;
- recursos de CPU/RAM.

## Qué no verifica

Esta prueba no determina la precisión del reconocimiento emocional.
La precisión se debe evaluar por separado con el conjunto infantil y una prueba
en vivo controlada.

## Prueba final del robot

Después de pasar este diagnóstico se debe ejecutar el servicio de Chuwi y
comprobar la cadena completa:

cámara -> reconocimiento facial -> fusión multimodal -> gestor emocional ->
respuesta -> TTS -> parlante.

El objetivo es que el sistema pueda iniciar y funcionar sin estar conectado a
una computadora.
