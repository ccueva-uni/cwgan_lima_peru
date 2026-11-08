
import json
from pathlib import Path
import io
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

from gan import Generator


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Generador de acelerogramas sintéticos | cWGAN",
    page_icon="🌎",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent

CONFIG_PATH = ROOT / "exp_c12" / "config_d.json"
MODELS_DIR = ROOT / "exp_c12" / "models"
METADATA_PATH = ROOT / "wforms_table_sel.csv"
WAVEFORMS_PATH = ROOT / "downsamp_1x_sel.npy"


# ============================================================
# 2. CARGA DEL MODELO Y DATOS
# ============================================================

@st.cache_resource(show_spinner="Cargando modelo cWGAN...")

def load_model_and_data(
    model_path_string,
    model_mtime_ns,
    model_size
):

    model_path = Path(model_path_string)

    required = [
        CONFIG_PATH,
        model_path,
        METADATA_PATH,
        WAVEFORMS_PATH
    ]

    missing = [
        str(p.relative_to(ROOT))
        for p in required
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "No se encontraron estos archivos:\n- "
            + "\n- ".join(missing)
        )

    # --------------------------------------------------------
    # Dispositivo
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --------------------------------------------------------
    # Configuración experimental
    # --------------------------------------------------------

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    z_size = int(config.get("noise_dim", 100))
    sample_rate = float(config.get("sample_rate", 100.0))

    # --------------------------------------------------------
    # Metadatos del catálogo
    # --------------------------------------------------------

    df_meta = pd.read_csv(METADATA_PATH)

    needed_columns = ["dist", "mag", "vs30"]

    absent = [
        c for c in needed_columns
        if c not in df_meta.columns
    ]

    if absent:
        raise ValueError(
            f"Faltan columnas en wforms_table_sel.csv: {absent}"
        )

    ranges = {
        "dist": (
            float(df_meta["dist"].min()),
            float(df_meta["dist"].max())
        ),
        "mag": (
            float(df_meta["mag"].min()),
            float(df_meta["mag"].max())
        ),
        "vs30": (
            float(df_meta["vs30"].min()),
            float(df_meta["vs30"].max())
        ),
    }

    # --------------------------------------------------------
    # Cargar catálogo de acelerogramas
    # --------------------------------------------------------

    wfs_raw = np.load(
        WAVEFORMS_PATH,
        mmap_mode="r"
    )

    if wfs_raw.ndim != 3 or wfs_raw.shape[1] != 3:

        raise ValueError(
            "Se esperaba downsamp_1x_sel.npy con dimensiones "
            "(N, 3, muestras). "
            f"Dimensiones encontradas: {wfs_raw.shape}"
        )

    cnorms_all = np.max(
        np.abs(wfs_raw),
        axis=2
    )

    pga_max = float(np.max(cnorms_all))
    pga_min = float(np.min(cnorms_all))

    if (
        pga_min <= 0
        or pga_max <= 0
        or pga_max <= pga_min
    ):
        raise ValueError(
            "No se pudieron calcular límites válidos de PGA."
        )

    log_pga_min = float(np.log10(pga_min))
    log_pga_max = float(np.log10(pga_max))

    # --------------------------------------------------------
    # CARGAR CHECKPOINT SELECCIONADO
    # --------------------------------------------------------

    generator = Generator(
        z_size=z_size
    ).to(device)

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    else:

        state_dict = checkpoint

    generator.load_state_dict(state_dict)

    generator.eval()

    # --------------------------------------------------------
    # Identificar época del checkpoint
    # --------------------------------------------------------

    match = re.search(
        r"epoch_(\d+)",
        model_path.stem
    )

    checkpoint_epoch = (
        int(match.group(1))
        if match
        else None
    )

    return {
        "device": device,
        "config": config,
        "z_size": z_size,
        "sample_rate": sample_rate,
        "ranges": ranges,
        "log_pga_min": log_pga_min,
        "log_pga_max": log_pga_max,
        "generator": generator,
        "checkpoint_path": str(model_path),
        "checkpoint_epoch": checkpoint_epoch,
    }


# ============================================================
# 3. GENERACIÓN DE ACELEROGRAMAS SINTÉTICOS
# ============================================================

def synthesize(
    model_data,
    dist_km,
    magnitude,
    vs30,
    seed=None
):

    device = model_data["device"]
    ranges = model_data["ranges"]

    # --------------------------------------------------------
    # Semilla aleatoria
    # --------------------------------------------------------

    if seed is not None:

        torch.manual_seed(int(seed))
        np.random.seed(int(seed))

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    # --------------------------------------------------------
    # Normalización de condiciones
    # --------------------------------------------------------

    d_min, d_max = ranges["dist"]
    m_min, m_max = ranges["mag"]
    v_min, v_max = ranges["vs30"]

    if (
        d_max == d_min
        or m_max == m_min
        or v_max == v_min
    ):
        raise ValueError("Rango de entrenamiento nulo.")

    dist_norm = (
        dist_km - d_min
    ) / (
        d_max - d_min
    )

    mag_norm = (
        magnitude - m_min
    ) / (
        m_max - m_min
    )

    vs30_norm = (
        vs30 - v_min
    ) / (
        v_max - v_min
    )

    t_dist = torch.tensor(
        [[dist_norm]],
        dtype=torch.float32,
        device=device
    )

    t_mag = torch.tensor(
        [[mag_norm]],
        dtype=torch.float32,
        device=device
    )

    t_vs30 = torch.tensor(
        [[vs30_norm]],
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # Ruido latente
    # --------------------------------------------------------

    z = torch.randn(
        1,
        3,
        model_data["z_size"],
        device=device
    )

    # --------------------------------------------------------
    # Generación con el checkpoint seleccionado
    # --------------------------------------------------------

    with torch.inference_mode():

        onda_norm, fake_lcn = model_data["generator"](
            z,
            t_dist,
            t_mag,
            t_vs30
        )

    onda = onda_norm.detach().cpu().numpy()
    lcn = fake_lcn.detach().cpu().numpy()

    # --------------------------------------------------------
    # Eliminar dimensiones singleton
    # --------------------------------------------------------

    onda = np.squeeze(onda)
    lcn = np.squeeze(lcn).reshape(-1)

    # --------------------------------------------------------
    # Verificar dimensiones
    # --------------------------------------------------------

    if onda.ndim != 2:

        raise ValueError(
            f"Forma inesperada de onda: {onda.shape}"
        )

    if onda.shape[0] != 3 and onda.shape[1] == 3:
        onda = onda.T

    if onda.shape[0] != 3:

        raise ValueError(
            f"Se esperaban 3 componentes: {onda.shape}"
        )

    if lcn.size != 3:

        raise ValueError(
            f"Se esperaban 3 valores LCN: {lcn.shape}"
        )

    # --------------------------------------------------------
    # Desnormalización PGA
    # --------------------------------------------------------

    log_pga = (
        ((lcn + 1.0) / 2.0)
        * (
            model_data["log_pga_max"]
            - model_data["log_pga_min"]
        )
        + model_data["log_pga_min"]
    )

    pga_scale = np.power(
        10.0,
        log_pga
    ).reshape(3)

    # --------------------------------------------------------
    # Acelerograma físico
    # --------------------------------------------------------

    waveform_real = (
        onda * pga_scale[:, None]
    )

    if not np.all(np.isfinite(waveform_real)):

        raise ValueError(
            "Se detectaron valores NaN o infinitos."
        )

    return waveform_real, pga_scale


# ============================================================
# 4. EXPORTACIÓN CSV
# ============================================================

def make_csv(waveform, sample_rate):

    time = (
        np.arange(waveform.shape[1])
        / sample_rate
    )

    df = pd.DataFrame({

        "Tiempo_s": time,
        "EW_cm_s2": waveform[0],
        "NS_cm_s2": waveform[1],
        "UD_cm_s2": waveform[2],

    })

    buffer = io.StringIO()

    df.to_csv(
        buffer,
        index=False,
        float_format="%.8f"
    )

    return buffer.getvalue().encode("utf-8")


# ============================================================
# 5. INTERFAZ PRINCIPAL
# ============================================================

st.title(
    "Generador de acelerogramas sintéticos "
    "(Proyecto Final Maestría en Inteligencia Artificial)"
)

st.subheader(
    "Modelo generativo condicionado cWGAN-GP "
    "(Cueva et al., 2026)"
)

st.markdown(
    "Síntesis de movimientos sísmicos triaxiales condicionados por "
    "distancia hipocentral, magnitud Mw y Vs30."
)


# ============================================================
# 6. DETECCIÓN AUTOMÁTICA DE CHECKPOINTS
# ============================================================

model_files = sorted(
    MODELS_DIR.glob("model_G_epoch_*.pth"),
    key=lambda p: (
        int(re.search(r"epoch_(\d+)", p.stem).group(1))
        if re.search(r"epoch_(\d+)", p.stem)
        else -1
    )
)

if not model_files:

    st.error(
        "No se encontraron modelos "
        "model_G_epoch_*.pth en exp_c12/models/."
    )

    st.info(
        "Copia allí tus pesos entrenados, por ejemplo "
        "model_G_epoch_1.pth hasta model_G_epoch_79.pth."
    )

    st.stop()


# ============================================================
# 7. SELECCIÓN DE LA ÉPOCA
# ============================================================

st.markdown("### Selección del modelo entrenado")

model_labels = {}

for p in model_files:

    match = re.search(
        r"epoch_(\d+)",
        p.stem
    )

    if match:

        model_labels[p.name] = (
            f"Época {match.group(1)}"
        )

    else:

        model_labels[p.name] = p.name


# Seleccionar época 79 por defecto si existe

default_index = next(

    (
        i
        for i, p in enumerate(model_files)

        if re.search(r"epoch_(\d+)", p.stem)
        and int(
            re.search(r"epoch_(\d+)", p.stem).group(1)
        ) == 79

    ),

    len(model_files) - 1

)


selected_model_name = st.selectbox(

    "Checkpoint del generador",

    options=[
        p.name
        for p in model_files
    ],

    index=default_index,

    format_func=lambda name: model_labels[name],

    help=(
        "Selecciona la época cuyos pesos deseas utilizar "
        "para generar los acelerogramas."
    )

)


selected_model_path = (
    MODELS_DIR / selected_model_name
)


# ============================================================
# 8. CARGA DEL CHECKPOINT SELECCIONADO
# ============================================================

try:

    checkpoint_stat = selected_model_path.stat()

    data = load_model_and_data(

        str(selected_model_path),

        checkpoint_stat.st_mtime_ns,

        checkpoint_stat.st_size

    )

except Exception as exc:

    st.error(
        "No fue posible inicializar el modelo seleccionado."
    )

    st.exception(exc)

    st.info(
        "Verifica que gan.py, wforms_table_sel.csv, "
        "downsamp_1x_sel.npy, config_d.json y el checkpoint "
        "seleccionado estén en las rutas indicadas."
    )

    st.stop()


# ============================================================
# 9. INFORMACIÓN DEL MODELO ACTIVO
# ============================================================

st.success(

    f"Generador cargado correctamente | "
    f"Época: {data['checkpoint_epoch']} | "
    f"Archivo: {Path(data['checkpoint_path']).name}"

)

ranges = data["ranges"]

sample_rate = data["sample_rate"]

n_samples = 4000

duration = n_samples / sample_rate


c1, c2, c3 = st.columns(3)

c1.metric(
    "Dispositivo",
    str(data["device"]).upper()
)

c2.metric(
    "Frecuencia de muestreo",
    f"{sample_rate:g} Hz"
)

c3.metric(
    "Duración del registro",
    f"{duration:.2f} s"
)


# ============================================================
# 10. PARÁMETROS DE ENTRADA
# ============================================================

st.divider()

st.markdown("### Parámetros de entrada")


with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    dist = col1.number_input(

        "Distancia hipocentral (km)",

        min_value=float(ranges["dist"][0]),

        max_value=float(ranges["dist"][1]),

        value=float(
            np.clip(45.0, *ranges["dist"])
        ),

        step=0.1

    )


    mag = col2.number_input(

        "Magnitud momento Mw",

        min_value=float(ranges["mag"][0]),

        max_value=float(ranges["mag"][1]),

        value=float(
            np.clip(7.0, *ranges["mag"])
        ),

        step=0.1

    )


    vs30 = col3.number_input(

        "Vs30 (m/s)",

        min_value=float(ranges["vs30"][0]),

        max_value=float(ranges["vs30"][1]),

        value=float(
            np.clip(400.0, *ranges["vs30"])
        ),

        step=1.0

    )


    st.caption(

        f"Rangos del catálogo: "
        f"distancia {ranges['dist'][0]:.2f}–"
        f"{ranges['dist'][1]:.2f} km · "
        f"Mw {ranges['mag'][0]:.2f}–"
        f"{ranges['mag'][1]:.2f} · "
        f"Vs30 {ranges['vs30'][0]:.1f}–"
        f"{ranges['vs30'][1]:.1f} m/s"

    )


    seed_col, button_col = st.columns([1, 2])


    seed = seed_col.number_input(

        "Semilla (opcional; -1 = aleatoria)",

        min_value=-1,

        max_value=2_147_483_647,

        value=-1,

        step=1

    )


    generate = button_col.button(

        "⚡ Generar acelerograma",

        type="primary",

        use_container_width=True

    )


# ============================================================
# 11. EJECUTAR GENERACIÓN
# ============================================================

if generate:

    try:

        with st.spinner(
            "Generando acelerograma triaxial..."
        ):

            waveform, pga_scales = synthesize(

                data,

                dist,

                mag,

                vs30,

                None if seed == -1 else int(seed)

            )


        st.session_state["waveform"] = waveform

        st.session_state["pga_scales"] = pga_scales

        st.session_state["input_values"] = (
            dist,
            mag,
            vs30
        )

        st.session_state["seed"] = seed

        st.session_state["generated_checkpoint"] = (
            data["checkpoint_path"]
        )

        st.session_state["generated_epoch"] = (
            data["checkpoint_epoch"]
        )

        st.session_state["generated_sample_rate"] = (
            sample_rate
        )

    except Exception as exc:

        st.error(
            f"Error durante la síntesis: {exc}"
        )


# ============================================================
# 12. RESULTADOS
# ============================================================

if "waveform" in st.session_state:

    waveform = st.session_state["waveform"]

    pga_values = np.max(
        np.abs(waveform),
        axis=1
    )

    input_values = st.session_state["input_values"]

    result_sample_rate = st.session_state.get(
        "generated_sample_rate",
        sample_rate
    )

    st.divider()

    st.markdown(
        "### Resultados de la síntesis"
    )

    st.caption(

        f"Condiciones utilizadas: "
        f"R = {input_values[0]:.2f} km · "
        f"Mw = {input_values[1]:.2f} · "
        f"Vs30 = {input_values[2]:.1f} m/s"

    )

    st.caption(

        f"Checkpoint utilizado: "
        f"Época {st.session_state['generated_epoch']} · "
        f"{Path(st.session_state['generated_checkpoint']).name}"

    )


    # --------------------------------------------------------
    # PGA
    # --------------------------------------------------------

    p1, p2, p3 = st.columns(3)

    for col, label, value in zip(

        [p1, p2, p3],

        ["PGA EW", "PGA NS", "PGA UD"],

        pga_values

    ):

        col.metric(

            label,

            f"{value:.2f} cm/s²",

            f"{value / 980.665:.4f} g"

        )


    # --------------------------------------------------------
    # Gráficas triaxiales
    # --------------------------------------------------------

    time = (
        np.arange(waveform.shape[1])
        / result_sample_rate
    )

    fig, axes = plt.subplots(

        3,

        1,

        figsize=(13, 8),

        sharex=True

    )

    labels = ["EW", "NS", "UD"]

    for i, ax in enumerate(axes):

        ax.plot(

            time,

            waveform[i],

            linewidth=0.65

        )

        ax.set_ylabel(
            f"{labels[i]}\n(cm/s²)"
        )

        ax.grid(
            True,
            alpha=0.3
        )


    axes[-1].set_xlabel(
        "Tiempo (s)"
    )

    fig.suptitle(
        "Acelerogramas sintéticos triaxiales"
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # --------------------------------------------------------
    # Tabla de valores
    # --------------------------------------------------------

    with st.expander(
        "Ver tabla de valores"
    ):

        preview = pd.DataFrame({

            "Tiempo (s)": time,

            "EW (cm/s²)": waveform[0],

            "NS (cm/s²)": waveform[1],

            "UD (cm/s²)": waveform[2],

        })

        st.dataframe(

            preview,

            use_container_width=True,

            height=300

        )


    # --------------------------------------------------------
    # Descargar CSV
    # --------------------------------------------------------

    st.download_button(

        "⬇️ Descargar acelerograma (CSV)",

        data=make_csv(
            waveform,
            result_sample_rate
        ),

        file_name="acelerograma_sintetico_cwgan.csv",

        mime="text/csv",

        use_container_width=True

    )


else:

    st.info(
        "Introduce los parámetros y pulsa «Generar acelerograma»."
    )


# ============================================================
# 13. NOTA FINAL
# ============================================================

st.divider()

st.caption(

    "Herramienta de investigación. Los acelerogramas sintéticos "
    "deben someterse a validación física, estadística y sismológica "
    "antes de utilizarse en análisis o diseño de ingeniería."

)
