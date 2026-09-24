# Benchmark reproducible de modelos faciales de Chuwi

## Objetivo

Comparar arquitecturas ligeras de CNN y Transformer para la estimación de expresiones faciales de Chuwi bajo el mismo protocolo experimental.

## Modelos

- MobileNetV3-Large: baseline móvil.
- EfficientNet-B0: baseline CNN eficiente.
- MobileViT-XXS: arquitectura híbrida CNN/Transformer ligera.
- DeiT-Tiny: Transformer ligero de referencia.

La selección final no se decide por antigüedad ni por una métrica aislada.

## Protocolo fijo

- Entrada: 224x224 RGB.
- Semilla: 42.
- Train: 8,573 imágenes.
- Validation: 1,074 imágenes.
- Test: 1,074 imágenes.
- Clases canónicas: ANGER, DISGUST, FEAR, HAPPY, CALM, SAD, SURPRISE.
- Normalización ImageNet.
- Augmentation únicamente en train.
- AdamW.
- Class weighting.
- Early stopping por Macro-F1 de validation.
- El mejor checkpoint se selecciona exclusivamente con validation.

## Evaluación

Se registran:

- accuracy;
- Macro-F1;
- weighted-F1;
- precision, recall y F1 por clase;
- matriz de confusión;
- número de parámetros;
- tiempo de entrenamiento;
- tiempo de inferencia;
- muestras/segundo;
- desempeño separado sobre Children_Emotion.

## Limitación

El manifest no contiene un identificador de sujeto confiable. Por ello, el resultado de Children_Emotion se describe como evaluación por imágenes y no como evaluación estrictamente independiente por sujeto.

Las imágenes auxiliary_age_faces no participan en el benchmark porque no tienen etiquetas emocionales validadas.

## Ejecución

Instalar las dependencias de entrenamiento desde requirements-training.txt y ejecutar:

    python scripts/benchmark_facial_models.py --dataset-root /ruta/al/dataset_unificado_emociones

Los artefactos quedan en artifacts/facial_benchmark/.

El benchmark no modifica el dataset y conserva run_metadata.json con la configuración del experimento.
