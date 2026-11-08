# Generación de Acelerogramas Sísmicos para Lima Metropolitana mediante cWGAN-GP

<p align="center">
  <strong>Deep Learning · Generative AI · Earthquake Engineering · Seismic Ground Motion</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch" />
  <img src="https://img.shields.io/badge/Architecture-cWGAN--GP-purple" />
  <img src="https://img.shields.io/badge/Domain-Seismic%20Engineering-orange" />
  <img src="https://img.shields.io/badge/Region-Lima%20%7C%20Peru-red" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
</p>

---

## 1. Descripción del proyecto

Este repositorio presenta la implementación de una arquitectura de aprendizaje profundo basada en **Conditional Wasserstein Generative Adversarial Networks with Gradient Penalty (cWGAN-GP)** para la generación de acelerogramas sísmicos triaxiales sintéticos, representativos de las condiciones sismotectónicas y geotécnicas de Lima Metropolitana y el Callao, Perú.

La metodología integra modelos generativos profundos con parámetros sismológicos y geotécnicos, permitiendo sintetizar historias temporales de aceleración condicionadas por tres variables físicas fundamentales:

- **Magnitud de Momento (\(M_w\)):** representa el tamaño del evento sísmico.
- **Distancia Hipocentral (\(R_{\mathrm{hypo}}\)):** representa la separación entre el hipocentro y la estación de registro.
- **Velocidad promedio de onda de corte (\(V_{s30}\)):** caracteriza las condiciones dinámicas del perfil superficial del suelo.

El modelo genera tres componentes ortogonales del movimiento del terreno: **Este-Oeste (EW), Norte-Sur (NS) y Vertical (UD)**.

El objetivo principal es desarrollar una herramienta computacional que contribuya a la generación de movimientos sísmicos sintéticos para estudios de ingeniería sísmica, análisis dinámico no lineal y evaluación de la respuesta estructural mediante análisis tiempo-historia.

## 2. Contexto y motivación

La costa central del Perú presenta una elevada exposición al peligro sísmico debido al proceso de subducción de la placa de Nazca bajo la placa Sudamericana.

Lima Metropolitana y el Callao concentran una importante infraestructura urbana, industrial y estratégica, cuya vulnerabilidad frente a eventos sísmicos de gran magnitud constituye un desafío para la ingeniería estructural.

Una de las principales dificultades para los análisis dinámicos es la disponibilidad limitada de registros acelerográficos instrumentales que representen adecuadamente determinados escenarios de demanda sísmica.

En este contexto, los modelos generativos profundos ofrecen una alternativa computacional para aprender patrones estadísticos y temporales presentes en catálogos de movimientos sísmicos registrados.

La propuesta implementa una arquitectura cWGAN-GP condicionada por variables físicas, orientada a sintetizar acelerogramas triaxiales y estudiar su comportamiento temporal y espectral.

## 3. Arquitectura del modelo generativo

La arquitectura se fundamenta en las redes generativas antagónicas Wasserstein condicionales con penalización de gradiente, siguiendo los fundamentos de WGAN-GP de Gulrajani et al. (2017) y el enfoque de generación de movimientos sísmicos de Florez et al. (2020).

### 3.1. Variables de condicionamiento

El vector de condiciones físicas está definido por:

\[
\mathbf{c} =
\begin{bmatrix}
M_w & R_{\mathrm{hypo}} & V_{s30}
\end{bmatrix}
\]

Estas variables se procesan mediante redes densas de embedding independientes para construir representaciones latentes que se incorporan al generador y al crítico.

| Componente | Dimensión del embedding |
|---|---:|
| Embedding del generador | 150 |
| Embedding del crítico | 4000 |
| Vector de ruido latente | 100 |

### 3.2. Generador (Generator)

El generador \(G\) recibe un vector de ruido aleatorio gaussiano:

\[
\mathbf{z} \sim \mathcal{N}(0,I_{100})
\]

junto con las condiciones sismológicas y geotécnicas.

Su arquitectura utiliza bloques convolucionales y operaciones de remuestreo mediante *Nearest-Neighbor Upsampling*, con el propósito de evitar artefactos asociados a determinadas configuraciones de convolución transpuesta.

La salida contempla dos ramas:

- **Rama de aceleración:** genera las tres componentes de movimiento del terreno, con 4,000 muestras por componente.
- **Rama de intensidad:** estima el factor de escala energético mediante la predicción de \(\log_{10}(PGA)\).

La representación de salida corresponde a:

\[
\mathbf{a}_{syn} \in \mathbb{R}^{3 \times 4000}
\]

### 3.3. Crítico (Discriminator)

El crítico \(D\) evalúa las muestras reales y sintéticas mediante una función de puntuación continua, característica de la formulación Wasserstein.

La arquitectura integra las señales triaxiales, la representación de intensidad y los embeddings de las condiciones físicas.

Se incorpora la penalización de gradiente (*Gradient Penalty*) para regularizar el crítico y favorecer el cumplimiento de la condición 1-Lipschitz:

\[
L_{GP} =
\lambda
\mathbb{E}_{\hat{x}}
\left[
\left(
\|\nabla_{\hat{x}}D(\hat{x})\|_2-1
\right)^2
\right]
\]

Donde:

\[
\lambda = 10
\]

Esta estrategia contribuye a estabilizar el entrenamiento adversarial y mejorar la calidad de las distribuciones generadas.

## 4. Catálogo sísmico y procesamiento de datos

El entrenamiento se desarrolla a partir de un catálogo de registros acelerográficos triaxiales procesados y seleccionados para representar escenarios sísmicos relevantes para Lima Metropolitana.

### 4.1. Características del catálogo

| Característica | Descripción |
|---|---|
| Registros procesados | 5,214 |
| Componentes | EW, NS y UD |
| Frecuencia de muestreo | 100 Hz |
| Intervalo temporal | 0.01 s |
| Número de muestras | 4,000 |
| Duración nominal | 40 s |
| Dimensión del conjunto | (5214, 3, 4000) |
| Entrenamiento | 4,692 registros (90 %) |
| Validación | 522 registros (10 %) |

### 4.2. Rango de variables físicas

| Parámetro | Rango considerado |
|---|---:|
| Magnitud de momento (\(M_w\)) | 4.5 – 6.9 |
| Distancia hipocentral (\(R_{\mathrm{hypo}}\)) | 10 – 250 km |
| Velocidad de onda de corte (\(V_{s30}\)) | 350 – 800 m/s |
| Clasificación de suelo de referencia | S1 y S2 |

### 4.3. Estructura de los datos

El conjunto de datos procesado se organiza en dos archivos principales:

- `downsamp_1x_sel.npy`: matriz de acelerogramas triaxiales.
- `wforms_table_sel.csv`: tabla de metadatos sismológicos y geotécnicos asociados a cada registro.

Los datos se normalizan y las variables de condicionamiento se escalan para su incorporación al proceso de entrenamiento.

## 5. Estructura del repositorio

```text
cwgan_lima_peru/
│
├── app.py                      # Aplicación web de generación (Streamlit)
├── gan.py                      # Arquitectura Generator y Discriminator
├── data_utils.py               # Carga y procesamiento de datos
├── train_wgan.py               # Entrenamiento cWGAN-GP
├── requirements.txt            # Dependencias de Python
├── README.md                   # Documentación del proyecto
├── LICENSE                     # Licencia MIT
├── .gitignore                  # Exclusiones de Git
│
├── config/
│   └── metadata.json           # Metadatos y parámetros del modelo
│
├── data/
│   ├── wforms_table_sel.csv     # Metadatos sísmicos
│   └── downsamp_1x_sel.npy      # Acelerogramas procesados
│
├── exp_c12/
│   └── models/                  # Checkpoints del entrenamiento
│
└── images/
    └── resultados.png           # Figuras de investigación
```

*La estructura representa la organización prevista del proyecto. Los archivos de datos y checkpoints pueden distribuirse por separado debido a sus dimensiones y condiciones de disponibilidad.*

## 6. Instalación y configuración

### 6.1. Clonar el repositorio

```bash
git clone https://github.com/ccueva-uni/cwgan_lima_peru.git

cd cwgan_lima_peru
```

### 6.2. Crear un entorno virtual

**Windows:**

```bash
python -m venv venv

venv\Scripts\activate
```

**Linux / macOS:**

```bash
python3 -m venv venv

source venv/bin/activate
```

### 6.3. Instalar dependencias

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### 6.4. Requisitos de hardware

Se recomienda utilizar una GPU NVIDIA compatible con CUDA para acelerar el entrenamiento y la generación de acelerogramas.

**Hardware utilizado como referencia experimental:**

- GPU: NVIDIA GeForce RTX 3080 Ti Laptop.
- Memoria de GPU: 16 GB.
- Framework: PyTorch.
- Lenguaje: Python.

La ejecución en CPU puede ser posible, aunque con mayores tiempos de procesamiento.

## 7. Entrenamiento del modelo

Para iniciar el entrenamiento de la arquitectura cWGAN-GP:

```bash
python train_wgan.py
```

Los principales parámetros de entrenamiento considerados son:

| Parámetro | Valor |
|---|---:|
| Batch size | 64 – 128 |
| Iteraciones del crítico | 12 |
| Número de épocas | 80 – 100 |
| Learning rate | 0.0001 |
| Optimizador | Adam |
| \(\beta_1\) | 0.0 |
| \(\beta_2\) | 0.9 |
| Gradient Penalty | 10.0 |

Los checkpoints generados durante el entrenamiento permiten recuperar los pesos aprendidos y utilizarlos posteriormente para la generación de nuevas señales sintéticas.

## 8. Generación de acelerogramas y validación espectral

Una vez entrenado el modelo, es posible generar acelerogramas sintéticos condicionados por un escenario sísmico determinado.

### 8.1. Generación de señales sintéticas

La generación utiliza los parámetros:

\[
\mathbf{c} =
(M_w,R_{\mathrm{hypo}},V_{s30})
\]

El generador produce las componentes EW, NS y UD correspondientes al escenario seleccionado.

## 9. Resultados y evaluación del modelo

Los resultados experimentales se analizan desde dos perspectivas complementarias:

### 9.1. Dominio del tiempo

Se estudia la capacidad del modelo para reproducir características temporales del movimiento sísmico:

- Evolución temporal de las aceleraciones.
- Diferencias entre componentes horizontales y vertical.
- Distribución de amplitudes.
- Desarrollo y duración de la fase de movimiento fuerte.
- Decaimiento de la señal y comportamiento de la coda.

Se consideran los checkpoints correspondientes a las épocas finales del entrenamiento, particularmente las épocas 79 y 80.

### 9.2. Dominio de la frecuencia y respuesta espectral

La evaluación espectral permite estudiar:

- Distribución de energía en función del período.
- Variabilidad entre realizaciones sintéticas.
- Mediana espectral.
- Dispersión mediante percentiles 16–84.
- Comparación con el espectro de diseño de la Norma E.030 para Lima Metropolitana.

La finalidad es evaluar la capacidad del modelo para generar movimientos sísmicos con características temporales y espectrales representativas de los registros empleados en el entrenamiento.

## 10. Aplicación web de generación

El proyecto contempla una interfaz desarrollada con **Streamlit**, destinada a facilitar la generación de acelerogramas sintéticos sin necesidad de ejecutar manualmente el proceso de inferencia desde la consola.

La interfaz permite configurar los parámetros físicos del escenario sísmico:

- Magnitud de momento (\(M_w\)).
- Distancia hipocentral (\(R_{\mathrm{hypo}}\)).
- Velocidad de onda de corte (\(V_{s30}\)).

Asimismo, permite visualizar las señales triaxiales generadas y facilitar su análisis y exportación.

Para iniciar la aplicación, cuando `app.py` y los archivos del modelo estén disponibles:

```bash
python -m streamlit run app.py
```

## 11. Referencias bibliográficas

1. Florez, M. A., et al. (2020). *Generating synthetic earthquake ground motions with deep generative neural networks*.

2. Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). Improved Training of Wasserstein GANs. *Advances in Neural Information Processing Systems (NeurIPS)*.

3. Ministerio de Vivienda, Construcción y Saneamiento del Perú. (2018). *Norma Técnica E.030: Diseño Sismorresistente*. Reglamento Nacional de Edificaciones.

## 12. Alcances y limitaciones

Los acelerogramas generados constituyen realizaciones sintéticas aprendidas a partir del catálogo utilizado para entrenar el modelo.

Su aplicación en ingeniería requiere una evaluación independiente de las características estadísticas, sismológicas, temporales y espectrales de las señales.

La generación sintética no sustituye la selección de registros compatibles con el escenario sísmico, la evaluación de peligro sísmico ni los procedimientos de validación exigidos para un análisis estructural de diseño.

La reproducibilidad de los resultados depende de la disponibilidad del catálogo original, los metadatos, la configuración experimental y los pesos del modelo entrenado.

## 13. Autor

**Carlos Eduardo Cueva Caller**

Investigación y desarrollo en ingeniería sísmica, inteligencia artificial y aprendizaje profundo aplicado a la ingeniería.

GitHub: [@ccueva-uni](https://github.com/ccueva-uni)

Repositorio: [cwgan_lima_peru](https://github.com/ccueva-uni/cwgan_lima_peru)

---

<p align="center">
  <strong>Ingeniería Sísmica + Inteligencia Artificial + Deep Generative Models</strong>
  <br>
  Lima Metropolitana y Callao — Perú 🇵🇪
</p>
