# Protocolo de selección del modelo facial infantil

## 1. Problema que orienta la selección

Chuwi es un robot social afectivo orientado al acompañamiento socioemocional de niños hospitalizados. El reconocimiento facial es una señal auxiliar dentro de una arquitectura multimodal; no constituye por sí solo una medida clínica del estado emocional.

Por tanto, la selección del clasificador facial debe considerar simultáneamente:
1. desempeño equilibrado entre emociones;
2. desempeño sobre imágenes cuya población infantil está verificada;
3. errores por clase;
4. costo computacional y tiempo de inferencia;
5. viabilidad para Raspberry Pi 5.

## 2. Dataset

El benchmark utiliza exclusivamente la fuente Children_Emotion.

Se excluye Micro_Expressions porque la edad de sus participantes no está verificada como infantil en el conjunto auditado. Las imágenes auxiliares de edad tampoco se usan para entrenamiento porque no tienen etiquetas emocionales.

El conjunto infantil auditado contiene 1,380 imágenes:
- anger: 239
- disgust: 0
- fear: 205
- happiness: 235
- neutral: 240
- sadness: 227
- surprise: 234

La ausencia de imágenes infantiles etiquetadas como disgust debe quedar explícita en la tesis: no es una clase evaluable con este conjunto y no debe inventarse desempeño para ella.

## 3. Modelos comparados

- MobileNetV3-Large
- EfficientNet-B0
- MobileViT-XXS
- DeiT-Tiny

Todos se ejecutan bajo las mismas condiciones: semilla 42, entrada 224x224, batch 32, AdamW, learning rate 1e-4, weight decay 1e-4, máximo 20 épocas y early stopping de 4 épocas basado en Macro-F1 de validación.

## 4. Métrica principal

La métrica principal es Macro-F1. Se usa porque promedia el F1 de las clases y permite observar el comportamiento por emoción sin que una clase con más imágenes domine el indicador.

También se registran:
- Accuracy;
- Weighted-F1;
- F1 por emoción;
- matriz de confusión;
- número de parámetros;
- tiempo de entrenamiento;
- tiempo y rendimiento de inferencia.

## 5. Regla de interpretación

No se declara un modelo ganador antes de ejecutar el experimento.

Después de obtener los resultados se debe analizar:

Macro-F1 → F1 por emoción → matriz de confusión → costo computacional → prueba en Raspberry Pi 5.

El modelo seleccionado debe quedar justificado por los resultados y por los requisitos del sistema, no por una preferencia previa.

## 6. Limitaciones

Este benchmark no demuestra eficacia clínica ni reducción de ansiedad. Evalúa únicamente el desempeño técnico del componente de reconocimiento facial.

Además, la fuente infantil disponible es relativamente pequeña y no contiene ejemplos etiquetados de disgust, por lo que la cobertura de las siete clases no es homogénea.

## 7. Antecedentes que justifican el enfoque

La literatura sobre expresiones faciales infantiles señala que los conjuntos de datos para niños son más limitados que los conjuntos de adultos y que la edad puede influir en la expresión facial. Por ello, utilizar una población infantil identificada es metodológicamente más coherente con el problema de Chuwi que mezclar indiscriminadamente imágenes de población no verificada.

Los antecedentes de robots sociales en contextos pediátricos también muestran que las intervenciones se estudian en términos de interacción, ansiedad, estrés y bienestar, pero no justifican afirmar que un clasificador facial aislado produzca un efecto clínico.

## 8. Reproducibilidad

El benchmark se ejecuta mediante:

    python scripts/benchmark_facial_models.py \
      --dataset-root <DATASET> \
      --manifest <DATASET>/emotion_manifest.csv \
      --output artifacts/facial_benchmark \
      --models mobilenet_v3_large efficientnet_b0 mobilevit_xxs deit_tiny \
      --epochs 20 --batch-size 32 --lr 1e-4 \
      --weight-decay 1e-4 --patience 4 --seed 42

El notebook chuwi_benchmark_colab.ipynb permite ejecutar el experimento en Google Colab cuando la Raspberry Pi no sea el entorno adecuado para entrenar.