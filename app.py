import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
import pygame

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, sosfiltfilt, sosfilt

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# ---------------------------------------------------------
# FUNCIONES MATEMÁTICAS
# ---------------------------------------------------------
def normalizar_audio(senal):
    # Normalizar significa ajustar la amplitud para que el audio no sature al reproducirse.
    senal = np.asarray(senal, dtype=np.float32)

    max_val = np.max(np.abs(senal))

    if max_val == 0:
        return senal

    senal_normalizada = senal / max_val

    # Se deja un pequeño margen para evitar distorsión.
    senal_normalizada = senal_normalizada * 0.95

    return senal_normalizada.astype(np.float32)


def calcular_fft(senal, sr):
    # La FFT transforma la señal del dominio del tiempo al dominio de la frecuencia.
    # rfft es la FFT para señales reales, como el audio, y devuelve solo frecuencias positivas.
    n = len(senal)
    frecuencias = rfftfreq(n, d=1 / sr)
    magnitud = np.abs(rfft(senal)) / n
    return frecuencias, magnitud


def calcular_frecuencia_dominante(senal, sr):
    frecuencias, magnitud = calcular_fft(senal, sr)

    if len(magnitud) <= 1:
        return 0

    # Se ignora 0 Hz porque representa el componente constante de la señal.
    indice_mayor = np.argmax(magnitud[1:]) + 1
    return frecuencias[indice_mayor]


def aplicar_filtro_pasabanda(senal, sr, fmin, fmax, orden):
    # El filtro pasa banda conserva principalmente las frecuencias entre fmin y fmax.
    sos = butter(
        N=orden,
        Wn=[fmin, fmax],
        btype="bandpass",
        fs=sr,
        output="sos"
    )

    try:
        return sosfiltfilt(sos, senal)
    except ValueError:
        # Para audios muy cortos se usa un filtrado más simple.
        return sosfilt(sos, senal)


def preparar_senal_para_tiempo(senal, sr, max_puntos=50000):
    n = len(senal)

    if n > max_puntos:
        paso = n // max_puntos
        senal_reducida = senal[::paso]
        tiempo = np.arange(0, len(senal_reducida)) * paso / sr
    else:
        senal_reducida = senal
        tiempo = np.arange(0, n) / sr

    return tiempo, senal_reducida


class FFTAudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Análisis y filtrado de señales de audio mediante FFT")
        self.root.minsize(1100, 680)
        self.cargar_icono_app()
        self.root.state("zoomed")

        # Variables de audio
        self.ruta_audio = None
        self.nombre_audio = None
        self.senal_original = None
        self.senal_filtrada = None
        self.sr = None
        self.duracion = None
        self.num_muestras = None
        self.canales = None
        self.ultimo_fmin = None
        self.ultimo_fmax = None
        self.ultimo_orden = None

        # Archivos temporales para reproducir
        self.temp_dir = tempfile.mkdtemp()
        self.temp_original_wav = os.path.join(self.temp_dir, "audio_original_temp.wav")
        self.temp_filtrado_wav = os.path.join(self.temp_dir, "audio_filtrado_temp.wav")

        # Inicializar pygame
        pygame.mixer.init()

        # Crear interfaz
        self.crear_interfaz()

    def cargar_icono_app(self):
        ruta_icono = os.path.join(os.path.dirname(__file__), "app_icon.ico")

        if not os.path.exists(ruta_icono):
            return

        try:
            self.root.iconbitmap(ruta_icono)
        except tk.TclError:
            pass

    # ---------------------------------------------------------
    # INTERFAZ PRINCIPAL
    # ---------------------------------------------------------
    def crear_interfaz(self):
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Panel izquierdo
        self.panel_izquierdo = ttk.Frame(self.root, padding=10)
        self.panel_izquierdo.grid(row=0, column=0, sticky="ns")

        # Panel derecho
        self.panel_derecho = ttk.Frame(self.root, padding=10)
        self.panel_derecho.grid(row=0, column=1, sticky="nsew")
        self.panel_derecho.rowconfigure(0, weight=1)
        self.panel_derecho.columnconfigure(0, weight=1)

        self.crear_panel_controles()
        self.crear_panel_graficas()

    def crear_panel_controles(self):
        titulo = ttk.Label(
            self.panel_izquierdo,
            text="Proyecto FFT Audio",
            font=("Arial", 17, "bold")
        )
        titulo.pack(pady=(0, 10))

        subtitulo = ttk.Label(
            self.panel_izquierdo,
            text="Análisis y filtrado de señales de audio",
            font=("Arial", 10),
            wraplength=310
        )
        subtitulo.pack(pady=(0, 15))

        # Sección 1: cargar audio
        frame_carga = ttk.LabelFrame(self.panel_izquierdo, text="1. Cargar audio", padding=10)
        frame_carga.pack(fill="x", pady=5)

        btn_cargar = ttk.Button(
            frame_carga,
            text="Cargar archivo de audio",
            command=self.cargar_audio
        )
        btn_cargar.pack(fill="x", pady=4)

        formatos = ttk.Label(
            frame_carga,
            text="Formatos aceptados: .wav, .mp3, .flac, .ogg, .m4a",
            wraplength=300
        )
        formatos.pack(pady=4)

        # Sección 2: información del audio
        frame_info = ttk.LabelFrame(self.panel_izquierdo, text="2. Información del audio", padding=10)
        frame_info.pack(fill="x", pady=5)

        self.txt_info = tk.Text(frame_info, height=8, width=38, state="disabled")
        self.txt_info.pack(fill="x")

        # Sección 3: resumen matemático
        frame_resumen = ttk.LabelFrame(self.panel_izquierdo, text="3. Resumen matemático", padding=10)
        frame_resumen.pack(fill="x", pady=5)

        self.txt_resumen = tk.Text(frame_resumen, height=8, width=38, state="disabled")
        self.txt_resumen.pack(fill="x")

        # Sección 4: reproducción
        frame_reproduccion = ttk.LabelFrame(self.panel_izquierdo, text="4. Reproducción", padding=10)
        frame_reproduccion.pack(fill="x", pady=5)

        btn_original = ttk.Button(
            frame_reproduccion,
            text="Reproducir audio original",
            command=self.reproducir_original
        )
        btn_original.pack(fill="x", pady=3)

        btn_filtrado = ttk.Button(
            frame_reproduccion,
            text="Reproducir audio filtrado",
            command=self.reproducir_filtrado
        )
        btn_filtrado.pack(fill="x", pady=3)

        btn_detener = ttk.Button(
            frame_reproduccion,
            text="Detener reproducción",
            command=self.detener_audio
        )
        btn_detener.pack(fill="x", pady=3)

        # Sección 5: filtro pasa banda
        frame_filtro = ttk.LabelFrame(self.panel_izquierdo, text="5. Filtro pasa banda", padding=10)
        frame_filtro.pack(fill="x", pady=5)

        ttk.Label(frame_filtro, text="Frecuencia mínima (Hz):").pack(anchor="w")
        self.entry_fmin = ttk.Entry(frame_filtro)
        self.entry_fmin.pack(fill="x", pady=3)
        self.entry_fmin.insert(0, "300")

        ttk.Label(frame_filtro, text="Frecuencia máxima (Hz):").pack(anchor="w")
        self.entry_fmax = ttk.Entry(frame_filtro)
        self.entry_fmax.pack(fill="x", pady=3)
        self.entry_fmax.insert(0, "3000")

        ttk.Label(frame_filtro, text="Orden del filtro:").pack(anchor="w")
        self.spin_orden = ttk.Spinbox(frame_filtro, from_=1, to=10)
        self.spin_orden.pack(fill="x", pady=3)
        self.spin_orden.set(4)

        btn_filtrar = ttk.Button(
            frame_filtro,
            text="Aplicar filtro pasa banda",
            command=self.aplicar_filtro
        )
        btn_filtrar.pack(fill="x", pady=(8, 3))

        # Sección 6: exportar
        frame_exportar = ttk.LabelFrame(self.panel_izquierdo, text="6. Exportar resultado", padding=10)
        frame_exportar.pack(fill="x", pady=5)

        btn_guardar = ttk.Button(
            frame_exportar,
            text="Descargar audio filtrado .wav",
            command=self.guardar_audio_filtrado
        )
        btn_guardar.pack(fill="x", pady=3)

        # Explicaciones breves
        frame_ayuda = ttk.LabelFrame(self.panel_izquierdo, text="Ayuda rápida", padding=10)
        frame_ayuda.pack(fill="both", expand=True, pady=5)

        ayuda = (
            "Frecuencia mínima: límite inferior del rango que se desea conservar.\n\n"
            "Frecuencia máxima: límite superior del rango que se desea conservar.\n\n"
            "Orden del filtro: controla la intensidad del filtrado. Un orden mayor filtra con más fuerza, "
            "pero puede modificar más la señal.\n\n"
            "Gráfica en el tiempo: muestra cómo cambia la amplitud del audio durante la duración del archivo.\n\n"
            "Espectro de frecuencia: muestra qué frecuencias tienen mayor presencia en el audio."
        )

        lbl_ayuda = ttk.Label(frame_ayuda, text=ayuda, wraplength=310, justify="left")
        lbl_ayuda.pack(anchor="w")

        # Estado
        self.lbl_estado = ttk.Label(
            self.panel_izquierdo,
            text="Estado: esperando audio...",
            foreground="blue",
            wraplength=310
        )
        self.lbl_estado.pack(fill="x", pady=8)

    def crear_panel_graficas(self):
        self.notebook = ttk.Notebook(self.panel_derecho)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.tab_original = ttk.Frame(self.notebook)
        self.tab_filtrado = ttk.Frame(self.notebook)
        self.tab_comparacion = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_original, text="Análisis original")
        self.notebook.add(self.tab_filtrado, text="Análisis filtrado")
        self.notebook.add(self.tab_comparacion, text="Comparación")

        self.crear_figura_original()
        self.crear_figura_filtrado()
        self.crear_figura_comparacion()

    def crear_figura_original(self):
        self.fig_original = Figure(figsize=(8, 6), dpi=100)

        self.ax_tiempo_original = self.fig_original.add_subplot(211)
        self.ax_fft_original = self.fig_original.add_subplot(212)

        self.fig_original.tight_layout(pad=3)

        self.canvas_original = FigureCanvasTkAgg(self.fig_original, master=self.tab_original)
        self.canvas_original.get_tk_widget().pack(fill="both", expand=True)

    def crear_figura_filtrado(self):
        self.fig_filtrado = Figure(figsize=(8, 6), dpi=100)

        self.ax_tiempo_filtrado = self.fig_filtrado.add_subplot(211)
        self.ax_fft_filtrado = self.fig_filtrado.add_subplot(212)

        self.fig_filtrado.tight_layout(pad=3)

        self.canvas_filtrado = FigureCanvasTkAgg(self.fig_filtrado, master=self.tab_filtrado)
        self.canvas_filtrado.get_tk_widget().pack(fill="both", expand=True)

    def crear_figura_comparacion(self):
        self.fig_comparacion = Figure(figsize=(8, 6), dpi=100)

        self.ax_comp_tiempo = self.fig_comparacion.add_subplot(211)
        self.ax_comp_fft = self.fig_comparacion.add_subplot(212)

        self.fig_comparacion.tight_layout(pad=3)

        self.canvas_comparacion = FigureCanvasTkAgg(self.fig_comparacion, master=self.tab_comparacion)
        self.canvas_comparacion.get_tk_widget().pack(fill="both", expand=True)

    # ---------------------------------------------------------
    # CARGA DE AUDIO
    # ---------------------------------------------------------
    def cargar_audio(self):
        ruta = self.seleccionar_archivo_audio()
        if not ruta:
            return

        if not self.archivo_audio_valido(ruta):
            messagebox.showerror(
                "Formato no válido",
                "El formato seleccionado no es válido. Use .wav, .mp3, .flac, .ogg o .m4a."
            )
            return

        try:
            self.actualizar_estado("Cargando audio, espere...")

            self.ruta_audio = ruta
            self.nombre_audio = os.path.basename(ruta)

            audio, sr = librosa.load(ruta, sr=None, mono=False)
            self.sr = sr
            self.senal_original, self.canales = self.convertir_a_mono(audio)

            self.num_muestras = len(self.senal_original)
            self.duracion = self.num_muestras / self.sr

            self.guardar_wav_temporal(self.temp_original_wav, self.senal_original)
            self.senal_filtrada = None
            self.ultimo_fmin = None
            self.ultimo_fmax = None
            self.ultimo_orden = None

            self.mostrar_info_audio()
            self.mostrar_resumen_matematico()
            self.graficar_original()
            self.limpiar_graficas_filtrado()

            self.actualizar_estado("Audio cargado correctamente.")
            messagebox.showinfo("Carga completa", "El audio se cargó correctamente.")

        except Exception as e:
            messagebox.showerror(
                "Error al cargar audio",
                f"No se pudo cargar el archivo.\n\nDetalle del error:\n{e}"
            )
            self.actualizar_estado("Error al cargar audio.")

    def seleccionar_archivo_audio(self):
        formatos_validos = [
            ("Archivos de audio", "*.wav *.mp3 *.flac *.ogg *.m4a"),
            ("WAV", "*.wav"),
            ("MP3", "*.mp3"),
            ("FLAC", "*.flac"),
            ("OGG", "*.ogg"),
            ("M4A", "*.m4a")
        ]

        return filedialog.askopenfilename(
            title="Seleccionar archivo de audio",
            filetypes=formatos_validos
        )

    def archivo_audio_valido(self, ruta):
        extension = os.path.splitext(ruta)[1].lower()
        return extension in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]

    def convertir_a_mono(self, audio):
        # Si el audio tiene varios canales, se promedian para analizar una sola señal.
        if audio.ndim == 1:
            return audio.astype(np.float32), 1

        canales = audio.shape[0]
        senal_mono = np.mean(audio, axis=0).astype(np.float32)
        return senal_mono, canales

    def guardar_wav_temporal(self, ruta_wav, senal):
        self.liberar_audio_cargado()
        audio_temp = normalizar_audio(senal)
        sf.write(ruta_wav, audio_temp, self.sr, subtype="PCM_16")

    def mostrar_info_audio(self):
        if self.senal_original is None:
            return

        if self.canales == 1:
            texto_canales = "1 canal, tratado como mono"
        else:
            texto_canales = f"{self.canales} canales, tratado como mono para el análisis"

        info = (
            f"Nombre: {self.nombre_audio}\n"
            f"Duración: {self.duracion:.2f} segundos\n"
            f"Frecuencia de muestreo: {self.sr} Hz\n"
            f"Número de muestras: {self.num_muestras}\n"
            f"Canales: {texto_canales}\n"
            f"Frecuencia máxima analizable: {self.sr / 2:.2f} Hz"
        )

        self.txt_info.config(state="normal")
        self.txt_info.delete("1.0", tk.END)
        self.txt_info.insert(tk.END, info)
        self.txt_info.config(state="disabled")

    def mostrar_resumen_matematico(self):
        if self.senal_original is None:
            return

        nyquist = self.sr / 2
        resolucion = self.sr / self.num_muestras
        dominante_original = calcular_frecuencia_dominante(self.senal_original, self.sr)

        resumen = (
            f"Muestras analizadas: {self.num_muestras}\n"
            f"Frecuencia de muestreo: {self.sr} Hz\n"
            f"Frecuencia de Nyquist: {nyquist:.2f} Hz\n"
            f"Resolución en frecuencia: {resolucion:.4f} Hz\n"
            f"Frecuencia dominante original: {dominante_original:.2f} Hz"
        )

        if self.senal_filtrada is not None:
            dominante_filtrada = calcular_frecuencia_dominante(self.senal_filtrada, self.sr)
            resumen += (
                f"\nFrecuencia dominante filtrada: {dominante_filtrada:.2f} Hz\n"
                f"Rango conservado: {self.ultimo_fmin:.1f} Hz a {self.ultimo_fmax:.1f} Hz\n"
                f"Orden del filtro: {self.ultimo_orden}"
            )

        self.txt_resumen.config(state="normal")
        self.txt_resumen.delete("1.0", tk.END)
        self.txt_resumen.insert(tk.END, resumen)
        self.txt_resumen.config(state="disabled")

    # ---------------------------------------------------------
    # REPRODUCCIÓN
    # ---------------------------------------------------------
    def reproducir_original(self):
        if self.senal_original is None:
            messagebox.showwarning("Sin audio", "Primero debe cargar un archivo de audio.")
            return

        self.reproducir_wav(self.temp_original_wav)

    def reproducir_filtrado(self):
        if self.senal_filtrada is None:
            messagebox.showwarning("Sin audio filtrado", "Primero debe aplicar el filtro pasa banda.")
            return

        self.reproducir_wav(self.temp_filtrado_wav)

    def reproducir_wav(self, ruta_wav):
        try:
            self.liberar_audio_cargado()
            pygame.mixer.music.load(ruta_wav)
            pygame.mixer.music.play()
            self.actualizar_estado("Reproduciendo audio...")
        except Exception as e:
            messagebox.showerror(
                "Error de reproducción",
                f"No se pudo reproducir el audio.\n\nDetalle:\n{e}"
            )

    def detener_audio(self):
        self.liberar_audio_cargado()
        self.actualizar_estado("Reproducción detenida.")

    def liberar_audio_cargado(self):
        pygame.mixer.music.stop()

        try:
            pygame.mixer.music.unload()
        except pygame.error:
            pass

    # ---------------------------------------------------------
    # FILTRADO PASA BANDA
    # ---------------------------------------------------------
    def aplicar_filtro(self):
        if self.senal_original is None:
            messagebox.showwarning("Sin audio", "Primero debe cargar un archivo de audio.")
            return

        parametros = self.leer_parametros_filtro()
        if parametros is None:
            return

        fmin, fmax, orden = parametros

        if not self.validar_parametros_filtro(fmin, fmax, orden):
            return

        try:
            self.actualizar_estado("Aplicando filtro pasa banda...")

            filtrada = aplicar_filtro_pasabanda(
                self.senal_original,
                self.sr,
                fmin,
                fmax,
                orden
            )

            self.senal_filtrada = normalizar_audio(filtrada)
            self.ultimo_fmin = fmin
            self.ultimo_fmax = fmax
            self.ultimo_orden = orden

            self.guardar_wav_temporal(self.temp_filtrado_wav, self.senal_filtrada)
            self.mostrar_resumen_matematico()
            self.graficar_filtrado(fmin, fmax, orden)
            self.graficar_comparacion()

            self.notebook.select(self.tab_comparacion)

            self.actualizar_estado("Procesamiento terminado correctamente.")
            messagebox.showinfo(
                "Filtro aplicado",
                "El filtrado pasa banda terminó correctamente."
            )

        except Exception as e:
            messagebox.showerror(
                "Error al filtrar",
                f"Ocurrió un error durante el filtrado.\n\nDetalle:\n{e}"
            )
            self.actualizar_estado("Error durante el filtrado.")

    def leer_parametros_filtro(self):
        try:
            fmin = float(self.entry_fmin.get())
            fmax = float(self.entry_fmax.get())
            orden = int(self.spin_orden.get())
        except ValueError:
            messagebox.showerror(
                "Parámetros incorrectos",
                "Ingrese valores numéricos válidos para las frecuencias y el orden."
            )
            return None

        return fmin, fmax, orden

    def validar_parametros_filtro(self, fmin, fmax, orden):
        # Nyquist es la frecuencia máxima que se puede analizar correctamente.
        # En una señal digital equivale a la mitad de la frecuencia de muestreo.
        nyquist = self.sr / 2

        if fmin <= 0:
            messagebox.showerror(
                "Parámetro incorrecto",
                "La frecuencia mínima debe ser mayor que 0 Hz."
            )
            return False

        if fmax <= 0:
            messagebox.showerror(
                "Parámetro incorrecto",
                "La frecuencia máxima debe ser mayor que 0 Hz."
            )
            return False

        if fmin >= fmax:
            messagebox.showerror(
                "Parámetro incorrecto",
                "La frecuencia mínima debe ser menor que la frecuencia máxima."
            )
            return False

        if fmax >= nyquist:
            messagebox.showerror(
                "Parámetro incorrecto",
                f"La frecuencia máxima debe ser menor que la frecuencia de Nyquist.\n\n"
                f"Para este audio, la frecuencia máxima permitida es menor que {nyquist:.2f} Hz."
            )
            return False

        if orden < 1 or orden > 10:
            messagebox.showerror(
                "Parámetro incorrecto",
                "El orden del filtro debe estar entre 1 y 10."
            )
            return False

        return True

    # ---------------------------------------------------------
    # GRAFICACIÓN
    # ---------------------------------------------------------
    def graficar_senal_tiempo(self, eje, senal, titulo, etiqueta=None, alpha=1.0):
        tiempo, senal_reducida = preparar_senal_para_tiempo(senal, self.sr)

        if etiqueta:
            eje.plot(tiempo, senal_reducida, label=etiqueta, alpha=alpha)
        else:
            eje.plot(tiempo, senal_reducida, alpha=alpha)

        eje.set_title(titulo)
        eje.set_xlabel("Tiempo (s)")
        eje.set_ylabel("Amplitud")
        eje.grid(True)

    def graficar_espectro(self, eje, senal, titulo, etiqueta=None, alpha=1.0):
        frecuencias, magnitud = calcular_fft(senal, self.sr)

        if etiqueta:
            eje.plot(frecuencias, magnitud, label=etiqueta, alpha=alpha)
        else:
            eje.plot(frecuencias, magnitud, alpha=alpha)

        eje.set_title(titulo)
        eje.set_xlabel("Frecuencia (Hz)")
        eje.set_ylabel("Magnitud")
        eje.grid(True)

    def graficar_original(self):
        self.ax_tiempo_original.clear()
        self.ax_fft_original.clear()

        self.graficar_senal_tiempo(
            self.ax_tiempo_original,
            self.senal_original,
            "Señal original en el dominio del tiempo"
        )
        self.graficar_espectro(
            self.ax_fft_original,
            self.senal_original,
            "Espectro de frecuencia original usando FFT"
        )

        self.fig_original.tight_layout(pad=3)
        self.canvas_original.draw()

        self.notebook.select(self.tab_original)

    def graficar_filtrado(self, fmin, fmax, orden):
        self.ax_tiempo_filtrado.clear()
        self.ax_fft_filtrado.clear()

        self.graficar_senal_tiempo(
            self.ax_tiempo_filtrado,
            self.senal_filtrada,
            "Señal filtrada en el dominio del tiempo"
        )
        self.graficar_espectro(
            self.ax_fft_filtrado,
            self.senal_filtrada,
            f"Espectro filtrado - Pasa banda {fmin:.1f} Hz a {fmax:.1f} Hz | Orden {orden}"
        )

        self.fig_filtrado.tight_layout(pad=3)
        self.canvas_filtrado.draw()

    def graficar_comparacion(self):
        self.ax_comp_tiempo.clear()
        self.ax_comp_fft.clear()

        self.graficar_senal_tiempo(
            self.ax_comp_tiempo,
            self.senal_original,
            "Comparación en el dominio del tiempo",
            etiqueta="Original"
        )
        self.graficar_senal_tiempo(
            self.ax_comp_tiempo,
            self.senal_filtrada,
            "Comparación en el dominio del tiempo",
            etiqueta="Filtrada",
            alpha=0.8
        )
        self.ax_comp_tiempo.legend()

        self.graficar_espectro(
            self.ax_comp_fft,
            self.senal_original,
            "Comparación del espectro de frecuencia",
            etiqueta="Original"
        )
        self.graficar_espectro(
            self.ax_comp_fft,
            self.senal_filtrada,
            "Comparación del espectro de frecuencia",
            etiqueta="Filtrada",
            alpha=0.8
        )
        self.ax_comp_fft.legend()

        self.fig_comparacion.tight_layout(pad=3)
        self.canvas_comparacion.draw()

    def limpiar_graficas_filtrado(self):
        self.ax_tiempo_filtrado.clear()
        self.ax_fft_filtrado.clear()
        self.ax_comp_tiempo.clear()
        self.ax_comp_fft.clear()

        self.ax_tiempo_filtrado.set_title("Señal filtrada")
        self.ax_fft_filtrado.set_title("Espectro filtrado")
        self.ax_comp_tiempo.set_title("Comparación en el tiempo")
        self.ax_comp_fft.set_title("Comparación en frecuencia")

        self.canvas_filtrado.draw()
        self.canvas_comparacion.draw()

    # ---------------------------------------------------------
    # GUARDAR AUDIO
    # ---------------------------------------------------------
    def guardar_audio_filtrado(self):
        if self.senal_filtrada is None:
            messagebox.showwarning(
                "Sin audio filtrado",
                "Primero debe aplicar el filtro para generar una señal filtrada."
            )
            return

        ruta_guardado = filedialog.asksaveasfilename(
            title="Guardar audio filtrado",
            defaultextension=".wav",
            filetypes=[("Archivo WAV", "*.wav")]
        )

        if not ruta_guardado:
            return

        try:
            sf.write(
                ruta_guardado,
                self.senal_filtrada,
                self.sr,
                subtype="PCM_16"
            )

            messagebox.showinfo(
                "Archivo guardado",
                f"El audio filtrado se guardó correctamente en:\n{ruta_guardado}"
            )

            self.actualizar_estado("Audio filtrado guardado correctamente.")

        except Exception as e:
            messagebox.showerror(
                "Error al guardar",
                f"No se pudo guardar el archivo.\n\nDetalle:\n{e}"
            )

    # ---------------------------------------------------------
    # ESTADO
    # ---------------------------------------------------------
    def actualizar_estado(self, texto):
        self.lbl_estado.config(text=f"Estado: {texto}")
        self.root.update_idletasks()

    # ---------------------------------------------------------
    # CIERRE
    # ---------------------------------------------------------
    def cerrar_app(self):
        try:
            pygame.mixer.music.stop()
            pygame.mixer.quit()
        except Exception:
            pass

        self.root.destroy()


if __name__ == "__main__":
    ventana = tk.Tk()
    app = FFTAudioApp(ventana)
    ventana.protocol("WM_DELETE_WINDOW", app.cerrar_app)
    ventana.mainloop()
