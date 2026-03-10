import warnings
warnings.simplefilter('ignore')  # Hide unnecessary warnings
import os
# STRICT GAG ORDER FOR TENSORFLOW (Must be set before any AI imports)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Hides oneDNN custom operations warnings
import logging
import sys
import time
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import platform
import subprocess
import threading
import json
# The separators and transcription tools are lazy loaded to save RAM

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Change this from DEBUG to INFO!
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Explicitly silence noisy third-party libraries just in case
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR) # TensorFlow's internal abseil logger

def get_base_path():
    if hasattr(sys, '_MEIPASS'):
        # Running as a PyInstaller .exe
        return sys._MEIPASS
    # Running as a normal script
    return os.path.abspath(".")

# Get your base path (works in both Python and compiled PyInstaller .exe)
base_path = get_base_path()

# 1. Reroute HuggingFace (Wav2Vec2)
os.environ["HF_HOME"] = os.path.join(base_path, "pretrained_models", "huggingface")

# 2. Reroute PyTorch (Demucs & OpenUnmix)
os.environ["TORCH_HOME"] = os.path.join(base_path, "pretrained_models", "torch")

def open_file(path):
    """Open a file using the system's default application."""
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.call(("open", path))
    else:
        subprocess.call(("xdg-open", path))

class SeparationApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Audio Separation Tool")
        self.geometry("1200x700")
        self.minsize(900, 600) # Prevents window from shrinking too much and squishing UI

        # Load settings from file
        self.settings_file = "settings.json"
        self.load_settings()
        
        # Ensure folders exist
        os.makedirs(self.input_folder, exist_ok=True)
        for folder in self.output_folders.values():
            os.makedirs(folder, exist_ok=True)
        
        # --- SMART MEMORY: Initialize as None for Lazy Loading ---
        self.spleeter_sep = None
        self.demucs_sep = None
        self.openunmix_sep = None
        self.whisper_trans = None
        self.wav2vec2_trans = None
        self.vosk_trans = None
        # --------------------------------------------------------

        # Data lists
        self.songs = []
        self.folders = []
        self.all_items = []  # Tracks all listbox items (folders and songs)
        self.vocals = []
        self.instrumentals = []
        self.transcriptions = []

        # Main container
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Content frame (Defined FIRST so we can attach tab frames to it)
        self.content_frame = ctk.CTkFrame(main_frame)
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

        # Tab frames (FIXED: All three are now attached to self.content_frame)
        self.input_frame = ctk.CTkFrame(self.content_frame)
        self.output_frame = ctk.CTkFrame(self.content_frame)
        self.settings_frame = ctk.CTkFrame(self.content_frame)

        # Sidebar
        self.sidebar = ctk.CTkFrame(main_frame, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="ns", rowspan=2)
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(3, weight=1)  # Spacer row

        # --- NAVIGATION BUTTONS (Wired directly to _switch_tab) ---
        self.input_button = ctk.CTkButton(
            self.sidebar, 
            text="Input", 
            width=180,
            command=lambda: self._switch_tab(self.input_frame, self.input_button, "input")
        )
        self.input_button.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        self.output_button = ctk.CTkButton(
            self.sidebar, 
            text="Output", 
            width=180,
            command=lambda: self._switch_tab(self.output_frame, self.output_button, "output")
        )
        self.output_button.grid(row=1, column=0, padx=20, pady=(20, 10), sticky="ew")

        self.settings_button = ctk.CTkButton(
            self.sidebar, 
            text="Settings", 
            width=180,
            command=lambda: self._switch_tab(self.settings_frame, self.settings_button, "settings")
        )
        self.settings_button.grid(row=4, column=0, padx=20, pady=(20, 10), sticky="ew")
        # ----------------------------------------------------------

        # Appearance mode selection (bottom-aligned)
        appearance_mode_label = ctk.CTkLabel(self.sidebar, text="Appearance Mode:", anchor="w")
        appearance_mode_label.grid(row=5, column=0, padx=20, pady=(20, 0), sticky="w")

        appearance_mode_optionemenu = ctk.CTkOptionMenu(
            self.sidebar, 
            values=["Light", "Dark", "System"],
            command=self.change_appearance_mode_event,
            width=160
        )
        appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10), sticky="ew")
        appearance_mode_optionemenu.set(self.appearance_mode)
        ctk.set_appearance_mode(self.appearance_mode)

        # UI Scaling (Zoom) selection (bottom-aligned)
        scaling_values = [f"{i}%" for i in range(50, 201, 10)]
        self.scaling_optionemenu = ctk.CTkOptionMenu(
            self.sidebar, 
            values=scaling_values,
            command=self.change_scaling_event,
            width=160
        )
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.scaling_optionemenu.set(self.scaling)

        # Create tab contents
        self.create_input_tab()
        self.create_output_tab()
        self.create_settings_tab() 
        
        # Apply themes
        self.change_scaling_event(self.scaling)
        self.update_listbox_themes()

        # Progress bar and text area
        progress_frame = ctk.CTkFrame(main_frame)
        progress_frame.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(10, 0))
        progress_frame.grid_columnconfigure(1, weight=1)

        self.progress_bar = ctk.CTkProgressBar(progress_frame, mode="determinate", width=600, height=20)
        self.progress_bar.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.progress_bar.set(0)
        self.progress_bar.grid_remove()
        self.progress_text = ctk.CTkLabel(progress_frame, text="Ready", font=ctk.CTkFont(size=12))
        self.progress_text.grid(row=0, column=1, sticky="w")

        self.abort_separation = False
        self.abort_button = ctk.CTkButton(progress_frame, text="Abort", command=self.abort_separation_process, width=80)
        self.abort_button.grid(row=0, column=2, sticky="e", padx=(10, 0))
        self.abort_button.grid_remove()  # Hide initially

        # Load initial songs and outputs
        self.load_input()
        self.load_outputs()
        
        # Initially show input tab
        self._switch_tab(self.input_frame, self.input_button, "input")
        
    def _switch_tab(self, active_frame, active_button, tab_name):
        """Helper function to cleanly switch UI tabs and manage state."""
        
        # 1. Define all available tabs
        frames = [self.input_frame, self.output_frame, self.settings_frame]
        buttons = [self.input_button, self.output_button, self.settings_button]
        
        # 2. Cache default theme colors (Faster than looking it up for every button)
        theme_fg = ctk.ThemeManager.theme["CTkButton"]["fg_color"]
        theme_text = ctk.ThemeManager.theme["CTkButton"]["text_color"]
        theme_hover = ctk.ThemeManager.theme["CTkButton"]["hover_color"]

        # 3. Reset everything (Hide all frames, reset all buttons)
        for frame in frames:
            frame.grid_forget()
            
        for btn in buttons:
            btn.configure(fg_color=theme_fg, text_color=theme_text, hover_color=theme_hover)

        # 4. Show the requested frame
        active_frame.grid(row=0, column=0, sticky="nsew")

        # 5. Highlight the active button dynamically
        is_dark = ctk.get_appearance_mode() == "Dark"
        active_button.configure(
            fg_color="#FFFFFF" if is_dark else "#000000",
            text_color="#000000" if is_dark else "#FFFFFF",
            hover_color="#CCCCCC" if is_dark else "#333333"
        )
        
        # 6. Trigger our Smart Memory Unloader! (If you implemented it)
        self._free_inactive_models(tab_name)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        # Change buttons background and frontground color depending on active tab (Not )
        if self.input_frame.winfo_ismapped():  # If input tab is active
            if new_appearance_mode == "Dark":
                self.input_button.configure(fg_color="#FFFFFF", text_color="#000000", hover_color="#CCCCCC")  # White bg, black text
            else:
                self.input_button.configure(fg_color="#000000", text_color="#FFFFFF", hover_color="#333333")  # Black bg, white text
        elif self.output_frame.winfo_ismapped():  # If output tab is active
            if new_appearance_mode == "Dark":
                self.output_button.configure(fg_color="#FFFFFF", text_color="#000000", hover_color="#CCCCCC")
            else:
                self.output_button.configure(fg_color="#000000", text_color="#FFFFFF", hover_color="#333333")
        elif self.settings_frame.winfo_ismapped():  # If settings tab is active
            if new_appearance_mode == "Dark":
                self.settings_button.configure(fg_color="#FFFFFF", text_color="#000000", hover_color="#CCCCCC")  # White bg, black text
            else:
                self.settings_button.configure(fg_color="#000000", text_color="#FFFFFF", hover_color="#333333")  # Black bg, white text 
        # Change appearance mode and save it
        ctk.set_appearance_mode(new_appearance_mode)
        self.appearance_mode = new_appearance_mode
        self.save_settings()
        self.update_listbox_themes()  # Update listbox backgrounds

    def change_scaling_event(self, new_scaling: str):
        self.scaling = new_scaling
        scaling_float = int(new_scaling.replace("%", "")) / 100
        
        # 1. Scale the CustomTkinter UI
        ctk.set_widget_scaling(scaling_float)
        
        # 2. Scale the standard Tkinter listboxes (Base size 12)
        new_size = int(12 * scaling_float)
        font_config = ("Arial", new_size)
        
        if hasattr(self, 'songs_listbox'): self.songs_listbox.configure(font=font_config)
        if hasattr(self, 'trans_listbox'): self.trans_listbox.configure(font=font_config)
        if hasattr(self, 'vocals_listbox'): self.vocals_listbox.configure(font=font_config)
        if hasattr(self, 'instr_listbox'): self.instr_listbox.configure(font=font_config)

    def update_listbox_themes(self):
        """Update listbox backgrounds based on appearance mode."""
        if ctk.get_appearance_mode() == "Dark":
            bg_color = "#000000"
            fg_color = "#FFFFFF"  # White text
        else:
            bg_color = "#FFFFFF"
            fg_color = "#000000"  # Black text
        self.songs_listbox.configure(bg=bg_color, fg=fg_color)
        self.vocals_listbox.configure(bg=bg_color, fg=fg_color)
        self.instr_listbox.configure(bg=bg_color, fg=fg_color)
        self.trans_listbox.configure(bg=bg_color, fg=fg_color)

    def create_input_tab(self):
        for widget in self.input_frame.winfo_children():
            widget.destroy()
            
        frame = self.input_frame
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=0, minsize=350) # Prevents right menu squish
        frame.grid_rowconfigure(0, weight=0)
        frame.grid_rowconfigure(1, weight=1)

        # Path bar frame
        path_frame = ctk.CTkFrame(frame)
        path_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        path_frame.grid_columnconfigure(1, weight=1)

        path_label = ctk.CTkLabel(path_frame, text="Current Folder:", anchor="w")
        path_label.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        self.path_var = tk.StringVar(value=self.input_folder)
        self.path_entry = ctk.CTkEntry(path_frame, textvariable=self.path_var)
        self.path_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(10, 5))
        self.path_entry.bind("<Return>", self.on_path_enter)

        btn_frame = ctk.CTkFrame(path_frame, fg_color="transparent")
        btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 10))
        
        self.back_button = ctk.CTkButton(btn_frame, text="Back", command=self.go_back, width=80)
        self.back_button.pack(side="left", padx=5)

        self.change_folder_button = ctk.CTkButton(btn_frame, text="Change Folder/New Folder", command=self.change_input_folder)
        self.change_folder_button.pack(side="left", padx=5)

        self.add_song_button = ctk.CTkButton(btn_frame, text="Add Song", command=self.add_song)
        self.add_song_button.pack(side="right", padx=5)

        # Songs/Folders list
        self.songs_listbox = tk.Listbox(frame, bg="#000000", fg="#FFFFFF")
        self.songs_listbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0,10))
        self.songs_listbox.bind("<Double-Button-1>", self.on_listbox_double_click)

        # Separation menu frame 
        sep_scrollable = ctk.CTkScrollableFrame(frame, width=350) 
        sep_scrollable.grid(row=0, column=1, rowspan=4, sticky="nsew", padx=10, pady=10)
        sep_scrollable.grid_columnconfigure(0, weight=1)

        sep_label = ctk.CTkLabel(sep_scrollable, text="Separation Menu", font=ctk.CTkFont(size=20, weight="bold"))
        sep_label.grid(row=0, column=0, pady=(10,20))

        # AI Tool selection
        self.ai_tool_var = tk.StringVar(value="Spleeter")
        self.radio_spleeter = ctk.CTkRadioButton(sep_scrollable, text="Spleeter", variable=self.ai_tool_var, value="Spleeter", command=self.on_tool_change)
        self.radio_demucs = ctk.CTkRadioButton(sep_scrollable, text="Demucs", variable=self.ai_tool_var, value="Demucs", command=self.on_tool_change)
        self.radio_openunmix = ctk.CTkRadioButton(sep_scrollable, text="OpenUnmix", variable=self.ai_tool_var, value="OpenUnmix", command=self.on_tool_change)

        self.radio_spleeter.grid(row=1, column=0, sticky="w", padx=20, pady=5)
        self.radio_demucs.grid(row=2, column=0, sticky="w", padx=20, pady=5)
        self.radio_openunmix.grid(row=3, column=0, sticky="w", padx=20, pady=5)

        # Model selection
        self.model_label = ctk.CTkLabel(sep_scrollable, text="Model:", anchor="w")
        self.model_label.grid(row=4, column=0, sticky="w", padx=20, pady=(10,0))
        self.model_var = tk.StringVar(value="umxl")
        self.model_menu = ctk.CTkOptionMenu(
            sep_scrollable, variable=self.model_var, values=["umxl", "umxhq", "umx"], width=200
        )
        self.model_menu.grid(row=5, column=0, sticky="ew", padx=20, pady=5)

        # Output format
        self.format_label = ctk.CTkLabel(sep_scrollable, text="Output Format:", anchor="w")
        self.format_label.grid(row=6, column=0, sticky="w", padx=20, pady=(10,0))
        self.format_var = tk.StringVar(value="wav")
        self.format_menu = ctk.CTkOptionMenu(
            sep_scrollable, variable=self.format_var, values=["wav", "mp3", "flac"], command=self.on_format_change, width=200
        )
        self.format_menu.grid(row=7, column=0, sticky="ew", padx=20, pady=5)

        # WAV/FLAC options
        self.wav_flac_frame = ctk.CTkFrame(sep_scrollable)
        self.wav_flac_frame.grid(row=8, column=0, sticky="ew", padx=20, pady=5)
        self.wav_flac_frame.grid_remove() 

        self.channel_label = ctk.CTkLabel(self.wav_flac_frame, text="Channels:", anchor="w")
        self.channel_label.grid(row=0, column=0, sticky="w", padx=20, pady=(10,0))
        self.channel_var = tk.StringVar(value="Stereo")
        self.channel_menu = ctk.CTkOptionMenu(self.wav_flac_frame, variable=self.channel_var, values=["Mono", "Stereo"])
        self.channel_menu.grid(row=1, column=0, sticky="ew", padx=20, pady=5)

        self.sr_label = ctk.CTkLabel(self.wav_flac_frame, text="Sample Rate (Hz):", anchor="w")
        self.sr_label.grid(row=2, column=0, sticky="w", padx=20, pady=(10,0))
        self.sr_var = tk.StringVar(value="44100")
        self.sr_entry = ctk.CTkEntry(self.wav_flac_frame, textvariable=self.sr_var, width=150, placeholder_text="44100")
        self.sr_entry.grid(row=3, column=0, sticky="ew", padx=20, pady=5)

        self.bit_depth_frame = ctk.CTkFrame(self.wav_flac_frame)
        self.bit_depth_frame.grid(row=4, column=0, sticky="ew", padx=20, pady=5)
        self.bit_depth_frame.grid_remove()

        self.bit_depth_var = tk.BooleanVar(value=True)
        self.int24_radiobutton = ctk.CTkRadioButton(self.bit_depth_frame, text="24-bit", variable=self.bit_depth_var, value=True)
        self.int24_radiobutton.grid(row=0, column=0, sticky="w", padx=20, pady=5)
        self.float32_radiobutton = ctk.CTkRadioButton(self.bit_depth_frame, text="Float32 (bigger)", variable=self.bit_depth_var, value=False)
        self.float32_radiobutton.grid(row=1, column=0, sticky="w", padx=20, pady=5)

        # MP3 options
        self.mp3_frame = ctk.CTkFrame(sep_scrollable)
        self.mp3_frame.grid(row=9, column=0, sticky="ew", padx=20, pady=5)
        self.mp3_frame.grid_remove()

        self.bitrate_label = ctk.CTkLabel(self.mp3_frame, text="Bitrate (kbps):", anchor="w")
        self.bitrate_label.grid(row=0, column=0, sticky="w", padx=20, pady=(10,0))
        self.bitrate_var = tk.StringVar(value="192")
        self.bitrate_entry = ctk.CTkEntry(self.mp3_frame, textvariable=self.bitrate_var, width=150, placeholder_text="192")
        self.bitrate_entry.grid(row=1, column=0, sticky="ew", padx=20, pady=5)

        self.mp3_preset_label = ctk.CTkLabel(self.mp3_frame, text="MP3 Preset (2=Best Quality, 7=Fastest):", anchor="w")
        self.mp3_preset_label.grid(row=2, column=0, sticky="w", padx=20, pady=(10,0))
        self.mp3_preset_slider = ctk.CTkSlider(self.mp3_frame, from_=2, to=7, number_of_steps=5, command=self.update_mp3_preset_label)
        self.mp3_preset_slider.set(2)
        self.mp3_preset_slider.grid(row=3, column=0, sticky="ew", padx=20, pady=5)
        self.mp3_preset_value_label = ctk.CTkLabel(self.mp3_frame, text="Current: 2", anchor="w")
        self.mp3_preset_value_label.grid(row=4, column=0, sticky="w", padx=20, pady=5)

        # Shifts (for Demucs)
        self.shifts_frame = ctk.CTkFrame(sep_scrollable)
        self.shifts_frame.grid(row=10, column=0, sticky="ew", padx=20, pady=5)
        self.shifts_frame.grid_remove() 
        self.shifts_label = ctk.CTkLabel(self.shifts_frame, text="Shifts (increases quality but slows process):", anchor="w")
        self.shifts_label.grid(row=0, column=0, sticky="w", padx=20, pady=5)
        self.shifts_var = tk.StringVar(value="1")
        self.shifts_entry = ctk.CTkEntry(self.shifts_frame, textvariable=self.shifts_var, width=150, placeholder_text="1")
        self.shifts_entry.grid(row=1, column=0, sticky="ew", padx=20, pady=5)

        # Separate button
        self.separate_button = ctk.CTkButton(sep_scrollable, text="Separate", command=self.separate_audio)
        self.separate_button.grid(row=17, column=0, sticky="ew", padx=20, pady=(20,10))

        self.on_tool_change()

    def create_output_tab(self):
        for widget in self.output_frame.winfo_children():
            widget.destroy()

        frame = self.output_frame
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=0, minsize=350) # Prevents right menu squish
        frame.grid_rowconfigure((1, 3, 5), weight=1)

        # Transcriptions
        ctk.CTkLabel(frame, text="Transcriptions", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 5))
        ctk.CTkButton(frame, text="Change Folder", command=lambda: self.change_output_folder("transcriptions")).grid(row=0, column=0, sticky="e", padx=10, pady=(10, 5))
        self.trans_listbox = tk.Listbox(frame, bg="#000000", fg="#FFFFFF")
        self.trans_listbox.grid(row=1, column=0, sticky="nsew", padx=10)
        self.trans_listbox.bind("<Double-Button-1>", self.open_selected_transcription)

        # Vocals
        ctk.CTkLabel(frame, text="Vocals", font=ctk.CTkFont(size=18, weight="bold")).grid(row=2, column=0, sticky="w", padx=10, pady=(20, 5))
        ctk.CTkButton(frame, text="Change Folder", command=lambda: self.change_output_folder("vocals")).grid(row=2, column=0, sticky="e", padx=10, pady=(20, 5))
        self.vocals_listbox = tk.Listbox(frame, bg="#000000", fg="#FFFFFF")
        self.vocals_listbox.grid(row=3, column=0, sticky="nsew", padx=10)
        self.vocals_listbox.bind("<Double-Button-1>", self.open_selected_vocal)

        # Instrumentals
        ctk.CTkLabel(frame, text="Instrumentals", font=ctk.CTkFont(size=18, weight="bold")).grid(row=4, column=0, sticky="w", padx=10, pady=(20, 5))
        ctk.CTkButton(frame, text="Change Folder", command=lambda: self.change_output_folder("instrumentals")).grid(row=4, column=0, sticky="e", padx=10, pady=(20, 5))
        self.instr_listbox = tk.Listbox(frame, bg="#000000", fg="#FFFFFF")
        self.instr_listbox.grid(row=5, column=0, sticky="nsew", padx=10)

        # TRANSCRIPTION MENU 
        trans_menu = ctk.CTkScrollableFrame(frame, width=350)
        trans_menu.grid(row=0, column=1, rowspan=6, sticky="nsew", padx=10, pady=10)
        trans_menu.grid_columnconfigure(0, weight=1)

        self.trans_model_label = ctk.CTkLabel(trans_menu, text="Transcription Menu", font=ctk.CTkFont(size=20, weight="bold"))
        self.trans_model_label.grid(row=0, column=0, pady=(10, 20))
        
        ctk.CTkLabel(trans_menu, text="Tool:", anchor="w").grid(row=1, column=0, sticky="w", padx=20)
        self.trans_tool_var = tk.StringVar(value="whisper")
        
        ctk.CTkRadioButton(trans_menu, text="Whisper", variable=self.trans_tool_var, value="whisper", command=self.on_trans_tool_change).grid(row=2, column=0, sticky="w", padx=20, pady=5)
        ctk.CTkRadioButton(trans_menu, text="Wav2Vec2", variable=self.trans_tool_var, value="wav2vec2", command=self.on_trans_tool_change).grid(row=3, column=0, sticky="w", padx=20, pady=5)
        ctk.CTkRadioButton(trans_menu, text="vosk", variable=self.trans_tool_var, value="vosk", command=self.on_trans_tool_change).grid(row=4, column=0, sticky="w", padx=20, pady=5)

        ctk.CTkLabel(trans_menu, text="Model:", anchor="w").grid(row=5, column=0, sticky="w", padx=20, pady=(10, 0))
        self.trans_model_var = tk.StringVar()
        self.trans_model_menu = ctk.CTkOptionMenu(trans_menu, variable=self.trans_model_var, values=[], width=200)
        self.trans_model_menu.grid(row=6, column=0, sticky="ew", padx=20, pady=5)

        self.trans_lang_label = ctk.CTkLabel(trans_menu, text="Language:", anchor="w")
        self.trans_lang_label.grid(row=7, column=0, sticky="w", padx=20, pady=(10, 0))
        self.trans_lang_var = tk.StringVar(value="auto")
        self.trans_lang_menu = ctk.CTkOptionMenu(trans_menu, variable=self.trans_lang_var, values=["auto", "cs", "en", "fr", "de", "es"], width=200)
        self.trans_lang_menu.grid(row=8, column=0, sticky="ew", padx=20, pady=5)

        self.use_spk_id_var = tk.BooleanVar(value=False)
        self.spk_toggle = ctk.CTkSwitch(trans_menu, text="Identify Speakers", variable=self.use_spk_id_var, progress_color="#1f538d")
        self.spk_toggle.grid(row=8, column=0, sticky="w", padx=20, pady=10)
        self.spk_toggle.grid_remove() 
        
        self.trans_button = ctk.CTkButton(trans_menu, text="Transcribe", command=self.run_standalone_transcription)
        self.trans_button.grid(row=9, column=0, sticky="ew", padx=20, pady=(30, 10))
        
        ctk.CTkLabel(trans_menu, text="Note: Select a file in 'Vocals' list first.", font=ctk.CTkFont(size=10, slant="italic")).grid(row=10, column=0, padx=20)

        self.on_trans_tool_change()

    def create_settings_tab(self):
        for widget in self.settings_frame.winfo_children():
            widget.destroy()

        self.settings_frame.grid_columnconfigure(0, weight=1)
        self.settings_frame.grid_rowconfigure(0, weight=1)

        scroll_frame = ctk.CTkScrollableFrame(self.settings_frame)
        scroll_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # DIRECTORY SETTINGS
        dir_frame = ctk.CTkFrame(scroll_frame)
        dir_frame.pack(fill="x", padx=10, pady=(0, 20))
        ctk.CTkLabel(dir_frame, text="Folder Directories", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=15, pady=(15, 5))

        self.settings_input_var = tk.StringVar(value=self.input_folder)
        self.settings_vocals_var = tk.StringVar(value=self.output_folders["vocals"])
        self.settings_instr_var = tk.StringVar(value=self.output_folders["instrumentals"])
        self.settings_trans_var = tk.StringVar(value=self.output_folders["transcriptions"])

        folders = [
            ("Input Folder:", self.settings_input_var),
            ("Vocals Folder:", self.settings_vocals_var),
            ("Instrumentals:", self.settings_instr_var),
            ("Transcriptions:", self.settings_trans_var)
        ]

        for text, var in folders:
            row_frame = ctk.CTkFrame(dir_frame, fg_color="transparent")
            row_frame.pack(fill="x", padx=15, pady=5)
            ctk.CTkLabel(row_frame, text=text, width=120, anchor="w").pack(side="left")
            ctk.CTkEntry(row_frame, textvariable=var).pack(side="left", fill="x", expand=True, padx=(5, 5))
            ctk.CTkButton(row_frame, text="Browse", width=60, 
                          command=lambda v=var: self._browse_folder(v)).pack(side="right")

        # PREFERENCES
        ui_frame = ctk.CTkFrame(scroll_frame)
        ui_frame.pack(fill="x", padx=10, pady=(0, 20))
        ctk.CTkLabel(ui_frame, text="Preferences", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=15, pady=(15, 5))

        row_frame1 = ctk.CTkFrame(ui_frame, fg_color="transparent")
        row_frame1.pack(fill="x", padx=15, pady=(5, 15))
        ctk.CTkLabel(row_frame1, text="Listbox Font Size:", width=120, anchor="w").pack(side="left")
        self.font_size_var = tk.StringVar(value=str(self.font_size))
        ctk.CTkOptionMenu(row_frame1, variable=self.font_size_var, values=[str(i) for i in range(10, 24)]).pack(side="left", fill="x", expand=True, padx=(5, 0))

        # MODEL SETTINGS
        mod_frame = ctk.CTkFrame(scroll_frame)
        mod_frame.pack(fill="x", padx=10, pady=(0, 20))
        ctk.CTkLabel(mod_frame, text="AI Models (JSON Format)", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=15, pady=(15, 5))

        def create_model_box(parent, title, data):
            ctk.CTkLabel(parent, text=title, anchor="w").pack(anchor="w", padx=15, pady=(10, 0))
            box = ctk.CTkTextbox(parent, height=60, font=("Consolas", 11))
            box.pack(fill="x", padx=15, pady=(0, 5))
            box.insert("0.0", json.dumps(data))
            return box

        self.demucs_models_text = create_model_box(mod_frame, "Demucs:", self.separator_models.get("Demucs", []))
        self.openunmix_models_text = create_model_box(mod_frame, "OpenUnmix:", self.separator_models.get("OpenUnmix", []))
        self.whisper_models_text = create_model_box(mod_frame, "Whisper:", self.transcription_models.get("whisper", []))
        self.wav2vec2_models_text = create_model_box(mod_frame, "Wav2Vec2:", self.transcription_models.get("wav2vec2", []))
        self.vosk_models_text = create_model_box(mod_frame, "Vosk:", self.transcription_models.get("vosk", []))
        ctk.CTkLabel(mod_frame, text="").pack(pady=2) 

        # ACTION BUTTONS & MEMORY
        action_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        action_frame.pack(fill="x", padx=10, pady=(10, 10))
        
        self.auto_flush_memory = getattr(self, 'auto_flush_memory', True)
        self.flush_switch = ctk.CTkSwitch(action_frame, text="Auto-Flush RAM", command=self._toggle_auto_flush)
        if self.auto_flush_memory: self.flush_switch.select()
        self.flush_switch.pack(side="left", padx=(5, 15))
        
        ctk.CTkButton(action_frame, text="Force Flush Now", fg_color="#8B0000", hover_color="#5C0000", command=self._manual_flush).pack(side="left", padx=5)

        ctk.CTkButton(action_frame, text="Save Settings", height=40, font=ctk.CTkFont(weight="bold"), command=self.save_settings_changes).pack(side="right", padx=5)
        ctk.CTkButton(action_frame, text="Restore Defaults", height=40, fg_color="transparent", border_width=1, text_color=("gray10", "gray90"), command=self.restore_defaults).pack(side="right", padx=5)

    def update_mp3_preset_label(self, value):
        self.mp3_preset_value_label.configure(text=f"Current: {int(value)}")

    def on_tool_change(self):
        tool = self.ai_tool_var.get()
        try:
            if tool == "Spleeter":
                self.model_label.grid_remove()
                self.model_menu.grid_remove()  # Hide model selection for Spleeter
            elif tool == "Demucs":
                values = self.separator_models.get("Demucs", ["mdx", "mdx_extra", "htdemucs"])
                self.model_menu.configure(values=values)
                self.model_var.set(values[0] if values else "mdx")
                self.model_label.grid()
                self.model_menu.grid()
            elif tool == "OpenUnmix":
                values = self.separator_models.get("OpenUnmix", ["umxl", "umxhq", "umx"])
                self.model_menu.configure(values=values)
                self.model_var.set(values[0] if values else "umxl")
                self.model_label.grid()
                self.model_menu.grid()
        except Exception as e:
            logging.error(f"Error updating separator models: {e}", exc_info=True)
            # Fallback to defaults
            if tool == "Demucs":
                self.model_menu.configure(values=["mdx", "mdx_extra", "htdemucs"])
                self.model_var.set("mdx")
            elif tool == "OpenUnmix":
                self.model_menu.configure(values=["umxl", "umxhq", "umx"])
                self.model_var.set("umxl")
            self.model_label.grid()
            self.model_menu.grid()
        self.on_format_change()  # Refresh format options

    def on_format_change(self, *args):
        fmt = self.format_var.get()
        tool = self.ai_tool_var.get()
        if fmt in ["wav", "flac"]:
            self.wav_flac_frame.grid()  # Show WAV/FLAC options
            self.mp3_frame.grid_remove()  # Hide MP3 options
            if tool == "Demucs" and fmt == "wav":
                self.bit_depth_frame.grid()  # Show bit depth for Demucs WAV
            else:
                self.bit_depth_frame.grid_remove()  # Hide bit depth
        elif fmt == "mp3":
            self.wav_flac_frame.grid_remove()  # Hide WAV/FLAC options
            self.mp3_frame.grid()  # Show MP3 options
            self.bit_depth_frame.grid_remove()  # Ensure bit depth is hidden
        # Shifts is handled separately based on tool
        if tool == "Demucs":
            self.mp3_preset_label.grid()
            self.mp3_preset_slider.grid()
            self.mp3_preset_value_label.grid()
            self.shifts_frame.grid()  # Show shifts for Demucs
        else:
            self.mp3_preset_label.grid_remove()
            self.mp3_preset_slider.grid_remove()
            self.mp3_preset_value_label.grid_remove()
            self.shifts_frame.grid_remove()  # Hide shifts for other tools

    def on_trans_tool_change(self, *args):
        """
        Complete logic for UI visibility and model population.
        Handles: model lists, language dropdown visibility, and speaker ID toggle.
        """
        tool = self.trans_tool_var.get()
            
        try:
            # 1. RETRIEVE DATA FROM SETTINGS
            # Get models from settings.json or use hardcoded fallbacks
            all_models = self.transcription_models.get(tool, [])
            
            # 2. MODEL SELECTION LOGIC
            if tool in ["whisper", "wav2vec2", "vosk"]:
                # Filter logic for Vosk (don't show the spk model in the dropdown)
                if tool == "vosk":
                    values = [m for m in all_models if "spk" not in m]
                    if not values: values = ["vosk-model-small-cs-0.4-rhassspy"]
                else:
                    values = all_models if all_models else ["base"]

                # Update the dropdown menu values
                if hasattr(self, 'trans_model_menu'):
                    self.trans_model_menu.configure(values=values)
                    self.trans_model_var.set(values[0] if values else "")

                # Show the widgets
                self.trans_model_label.grid()
                self.trans_model_menu.grid()
            else:
                self.trans_model_label.grid_remove()
                self.trans_model_menu.grid_remove()

            # 3. LANGUAGE SELECTION VISIBILITY
            # Only Whisper is multilingual in a single model; others are per-model.
            if tool == "whisper":
                if hasattr(self, 'trans_lang_label'): self.trans_lang_label.grid()
                if hasattr(self, 'trans_lang_menu'): self.trans_lang_menu.grid()
            else:
                if hasattr(self, 'trans_lang_label'): self.trans_lang_label.grid_remove()
                if hasattr(self, 'trans_lang_menu'): self.trans_lang_menu.grid_remove()

            # 4. SPEAKER ID (DIARIZATION) VISIBILITY
            # Feature exclusive to our Vosk implementation
            if tool == "vosk":
                if hasattr(self, 'spk_toggle'): self.spk_toggle.grid()
            else:
                if hasattr(self, 'spk_toggle'): self.spk_toggle.grid_remove()
                    
        except Exception as e:
            logging.error(f"Error in on_trans_tool_change: {e}")
            
    def load_input(self):
        self.songs_listbox.delete(0, tk.END)
        self.folders.clear()
        self.songs.clear()
        self.all_items.clear()  # Clear all items
        if not os.path.isdir(self.input_folder):
            return

        items = sorted(os.listdir(self.input_folder))
        for item in items:
            full_path = os.path.join(self.input_folder, item)
            if os.path.isdir(full_path):
                self.all_items.append(('folder', full_path))
                self.folders.append(full_path)
                self.songs_listbox.insert(tk.END, f"[Folder] {item}")
            elif item.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                song_data = {'path': full_path, 'name': item}
                self.all_items.append(('song', song_data))
                self.songs.append(song_data)
                self.songs_listbox.insert(tk.END, item)

        # Update path bar
        self.path_var.set(self.input_folder)
    
    def change_input_folder(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_folder = folder
            self.load_input()
            # Prompt to save as default
            if messagebox.askyesno("Save as Default", "Save this folder as the new default input folder?"):
                # Refresh Settings tab variable than save
                self.settings_input_var.set(self.input_folder)
                self.save_settings()

    def go_back(self):
        parent = os.path.dirname(self.input_folder)
        if parent and os.path.isdir(parent):
            self.input_folder = parent
            self.load_input()

    def on_path_enter(self, event=None):
        new_path = self.path_var.get().strip()
        if os.path.isdir(new_path):
            self.input_folder = new_path
            self.load_input()
        else:
            messagebox.showwarning("Invalid Path", "The entered path is not a valid directory.")

    def on_listbox_double_click(self, event=None):
            sel = self.songs_listbox.curselection()
            if not sel:
                return
            idx = sel[0]
            item_type, item_data = self.all_items[idx]

            if item_type == 'folder':
                folder_path = item_data
                if os.path.isdir(folder_path):
                    self.input_folder = folder_path
                    self.load_input()
            elif item_type == 'song':
                open_file(item_data['path'])

    def add_song(self):
        filetypes = [("Audio files", "*.mp3 *.wav *.flac *.m4a"), ("All files", "*.*")]
        paths = filedialog.askopenfilenames(title="Select audio files to add", filetypes=filetypes)
        if not paths:
            return
        for path in paths:
            try:
                dest = os.path.join(self.input_folder, os.path.basename(path))
                if not os.path.exists(dest):
                    shutil.copy2(path, dest)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to copy {path}:\n{e}")
        self.load_input()

    def load_outputs(self):
        # Clear and load vocals
        self.vocals_listbox.delete(0, tk.END)
        self.vocals.clear()
        vocals_dir = self.output_folders["vocals"]
        if os.path.isdir(vocals_dir):
            for f in sorted(os.listdir(vocals_dir)):
                if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                    full_path = os.path.join(vocals_dir, f)
                    self.vocals.append({'path': full_path, 'name': f})
                    self.vocals_listbox.insert(tk.END, f)

        # Clear and load instrumentals
        self.instr_listbox.delete(0, tk.END)
        self.instrumentals.clear()
        instr_dir = self.output_folders["instrumentals"]
        if os.path.isdir(instr_dir):
            for f in sorted(os.listdir(instr_dir)):
                if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                    full_path = os.path.join(instr_dir, f)
                    self.instrumentals.append({'path': full_path, 'name': f})
                    self.instr_listbox.insert(tk.END, f)

        # Clear and load transcriptions
        self.trans_listbox.delete(0, tk.END)
        self.transcriptions.clear()
        trans_dir = self.output_folders["transcriptions"]
        if os.path.isdir(trans_dir):
            for f in sorted(os.listdir(trans_dir)):
                if f.lower().endswith(('.txt', '.lrc')):
                    full_path = os.path.join(trans_dir, f)
                    self.transcriptions.append({'path': full_path, 'name': f})
                    self.trans_listbox.insert(tk.END, f)

    def change_output_folder(self, filetype):
        folder = filedialog.askdirectory(title=f"Select {filetype.capitalize()} Output Folder")
        if folder:
            self.output_folders[filetype] = folder
            self.load_outputs()
            # Prompt to save as default
            if messagebox.askyesno("Save as Default", f"Save this folder as the new default {filetype} folder?"):
                # Refresh Settings tab variables than save
                if filetype == "vocals":
                    self.settings_vocals_var.set(self.output_folders["vocals"])
                elif filetype == "instrumentals":
                    self.settings_instr_var.set(self.output_folders["instrumentals"])
                elif filetype == "transcriptions":
                    self.settings_trans_var.set(self.output_folders["transcriptions"])
                self.save_settings()
                
    def _browse_folder(self, string_var):
        """Helper to let users click a button instead of typing a path."""
        folder = filedialog.askdirectory(initialdir=string_var.get())
        if folder:
            string_var.set(folder)

    def load_settings(self):
        defaults = self.DEFAULT_SETTINGS
        if not os.path.exists(self.settings_file):
            self._apply_dict_to_state(defaults)
            self.save_settings()
            return

        try:
            with open(self.settings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Merge loaded data with defaults to ensure missing keys don't break the app
            for key in defaults:
                if key not in data:
                    data[key] = defaults[key]
                    
            self._apply_dict_to_state(data)
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Settings file corrupted: {e}. Restoring defaults.")
            self._apply_dict_to_state(defaults)
            self.save_settings()

    def restore_defaults(self):
        """Wipes current settings back to factory state."""
        if messagebox.askyesno("Restore Defaults", "Are you sure you want to reset all settings and paths?"):
            self._apply_dict_to_state(self.DEFAULT_SETTINGS)
            self.save_settings()
            # Rebuild the tab to visually show the reset
            self.create_settings_tab()
            messagebox.showinfo("Success", "Settings restored to defaults.")

    def _apply_dict_to_state(self, data):
        """Helper to apply a dictionary to the app's variables cleanly."""
        self.input_folder = data["input_folder"]
        self.output_folders = {
            "vocals": data["vocals_folder"],
            "instrumentals": data["instrumentals_folder"],
            "transcriptions": data["transcriptions_folder"]
        }
        self.appearance_mode = data["appearance_mode"]
        self.scaling = data["scaling"]
        self.font_size = data["font_size"]
        self.separator_models = data["separator_models"]
        self.transcription_models = data["transcription_models"]

    def save_settings_changes(self):
        """Fired when the user clicks 'Save Settings' in the UI."""
        
        # Helper function to convert "model1, model2" string back to ["model1", "model2"] list
        def parse_csv(var):
            return [x.strip() for x in var.get().split(",") if x.strip()]

        # Apply directory variables from UI inputs
        self.input_folder = self.settings_input_var.get()
        self.output_folders["vocals"] = self.settings_vocals_var.get()
        self.output_folders["instrumentals"] = self.settings_instr_var.get()
        self.output_folders["transcriptions"] = self.settings_trans_var.get()
        
        # Save the current state of the Auto-Flush switch
        self.auto_flush_memory = bool(self.flush_switch.get())
        
        # Apply model variables by parsing the CSV strings
        self.separator_models["Demucs"] = parse_csv(self.model_vars["demucs"])
        self.separator_models["OpenUnmix"] = parse_csv(self.model_vars["openunmix"])
        self.transcription_models["whisper"] = parse_csv(self.model_vars["whisper"])
        self.transcription_models["wav2vec2"] = parse_csv(self.model_vars["wav2vec2"])
        self.transcription_models["vosk"] = parse_csv(self.model_vars["vosk"])

        # Ensure folders physically exist
        os.makedirs(self.input_folder, exist_ok=True)
        for folder in self.output_folders.values():
            os.makedirs(folder, exist_ok=True)

        # Write the updated variables to the settings.json file
        self.save_settings()
        
        # Reload relevant UI lists to show any folder changes
        if hasattr(self, 'load_input'): self.load_input()
        if hasattr(self, 'load_outputs'): self.load_outputs()
        
        messagebox.showinfo("Saved", "Settings successfully updated!")

    def save_settings(self):
        """Writes current state to disk."""
        data = {
            "input_folder": self.input_folder,
            "vocals_folder": self.output_folders["vocals"],
            "instrumentals_folder": self.output_folders["instrumentals"],
            "transcriptions_folder": self.output_folders["transcriptions"],
            "appearance_mode": self.appearance_mode,
            "scaling": self.scaling,
            # Note: "font_size" has been permanently removed since it scales automatically
            "auto_flush_memory": self.auto_flush_memory, 
            "separator_models": self.separator_models,
            "transcription_models": self.transcription_models
        }
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving settings: {e}")

    @property
    def DEFAULT_SETTINGS(self):
        """Centralized defaults so we only ever write this once."""
        return {
            "input_folder": "input",
            "vocals_folder": "output/vocals",
            "instrumentals_folder": "output/instrumentals",
            "transcriptions_folder": "output/text",
            "appearance_mode": "Dark",
            "scaling": "100%",
            "font_size": 12,
            "auto_flush_memory": True, # <-- Default to True for safety
            "separator_models": {
                "Spleeter": [],
                "Demucs": ["mdx", "mdx_extra", "htdemucs"],
                "OpenUnmix": ["umxl", "umxhq", "umx"]
            },
            "transcription_models": {
                "whisper": ["large", "medium", "small", "tiny", "base", "turbo"],
                "wav2vec2": [
                    "facebook/wav2vec2-base-960h", 
                    "facebook/wav2vec2-large-960h",
                    "facebook/wav2vec2-large-xlsr-53-czech",
                    "facebook/wav2vec2-large-xlsr-53-french"
                ],
                "vosk": [
                    "vosk-model-spk-0.4",
                    "vosk-model-small-cs-0.4-rhassspy",
                    "vosk-model-small-fr-0.22",
                    "vosk-model-small-fr-pguyot-0.3",
                    "vosk-model-small-en-us-0.15",
                    "vosk-model-en-us-0.22-lgraph"
                ]
            }
        }

    def open_selected_song(self, event=None):
        sel = self.songs_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        open_file(self.songs[idx]['path'])

    def open_selected_vocal(self, event=None):
        sel = self.vocals_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        open_file(self.vocals[idx]['path'])

    def open_selected_instrumental(self, event=None):
        sel = self.instr_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        open_file(self.instrumentals[idx]['path'])

    def open_selected_transcription(self, event=None):
        sel = self.trans_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        open_file(self.transcriptions[idx]['path'])

    def _toggle_auto_flush(self):
        """Updates the boolean variable when the user clicks the switch."""
        self.auto_flush_memory = bool(self.flush_switch.get())
        
    def _manual_flush(self):
        """Forcefully wipes everything when the user clicks the manual button."""
        # Temporarily fake being in settings with auto-flush on, just to run the wipe
        temp_state = getattr(self, 'auto_flush_memory', True)
        self.auto_flush_memory = True
        self._free_inactive_models("settings")
        self.auto_flush_memory = temp_state
        
        # Visually reset the progress bar to 0% and update the text to show readiness
        self.progress_bar.set(0)
        self.progress_text.configure(text="Ready. RAM was cleared and optimized for the next task.")

    def _free_inactive_models(self, active_tab: str):
        """Wipes models based on the active tab and user settings."""
        
        # 1. Check if user turned off Auto-Flush in settings
        if not getattr(self, 'auto_flush_memory', True):
            return  # Do nothing if set to manual
            
        # 2. Safety Check: Don't mess with memory or UI text if a task is actively running!
        if self.abort_button.winfo_ismapped():
            return 

        import gc
        import torch
        freed_something = False

        # --- CLEAR SEPARATION MODELS (If going to Output or Settings) ---
        if active_tab in ["output", "settings"]:
            if getattr(self, 'spleeter_sep', None) is not None:
                del self.spleeter_sep; self.spleeter_sep = None; freed_something = True
            if getattr(self, 'demucs_sep', None) is not None:
                del self.demucs_sep; self.demucs_sep = None; freed_something = True
            if getattr(self, 'openunmix_sep', None) is not None:
                del self.openunmix_sep; self.openunmix_sep = None; freed_something = True

        # --- CLEAR TRANSCRIPTION MODELS (If going to Input or Settings) ---
        if active_tab in ["input", "settings"]:
            if getattr(self, 'whisper_trans', None) is not None:
                del self.whisper_trans; self.whisper_trans = None; freed_something = True
            if getattr(self, 'wav2vec2_trans', None) is not None:
                del self.wav2vec2_trans; self.wav2vec2_trans = None; freed_something = True
            if getattr(self, 'vosk_trans', None) is not None:
                del self.vosk_trans; self.vosk_trans = None; freed_something = True

        # 3. Perform the flush and update the UI
        if freed_something:
            print(f"[INFO] Switched to '{active_tab}'. Cleared inactive models.")
            # ... (gc.collect() and empty_cache() logic) ...
            
            # Update the progress text to inform the user!
            self.progress_bar.set(0)
            self.progress_text.configure(text=f"Ready. RAM cleared for {active_tab.capitalize()} workspace.")

    def abort_separation_process(self):
        self.abort_separation = True
        self.progress_text.configure(text="Aborting separation...")
        self.abort_button.grid_remove()

    def separate_audio(self):
        """
        Overview: Function to prepare for separation in thread for non-blocking effect.
        Handles UI prep (validation, selections), separation logic and progress updates.
        Runs in background thread for non-blocking UI.
        Captures console progress from tools and updates bottom progress bar/text.
        """
        sel = self.songs_listbox.curselection()
        if not sel:
            self.after(0, lambda: messagebox.showwarning("No selection", "Please select a song to separate."))
            return
        idx = sel[0]
        item_type, item_data = self.all_items[idx]
        if item_type != 'song':
            self.after(0, lambda: messagebox.showwarning("Invalid selection", "Please select a song to separate, not a folder."))
            return

        # Gather all params
        song = item_data
        input_path = song['path']
        song_name = os.path.splitext(os.path.basename(song['name']))[0]
        ai_tool = self.ai_tool_var.get()
        model = self.model_var.get()
        fmt = self.format_var.get()
        sr = int(self.sr_var.get()) if fmt in ["wav", "flac"] else 44100
        bitrate = self.bitrate_var.get() if fmt == "mp3" else "192k"
        bit_depth = self.bit_depth_var.get() if fmt == "wav" and ai_tool == "Demucs" else None
        mp3_preset = int(self.mp3_preset_slider.get()) if fmt == "mp3" and ai_tool == "Demucs" else None
        shifts = int(self.shifts_var.get()) if ai_tool == "Demucs" else None
        vocals_folder = self.output_folders["vocals"]
        instr_folder = self.output_folders["instrumentals"]
        
        # Start the thread
        thread = threading.Thread(target=self._run_separation, args=(input_path, song_name, vocals_folder, instr_folder,
                                                                     ai_tool, model, fmt, sr, bitrate, song, bit_depth, mp3_preset, shifts))
        thread.daemon = True
        thread.start()

    def _run_separation(self, input_path, song_name, vocals_folder, instr_folder,
                        ai_tool, model, fmt, sr, bitrate, song, bit_depth, mp3_preset, shifts):
        """
        Overview: Internal method for the threaded separation logic.
        """
        # Define progress callback
        def update_progress(percent, message):
                # 1. Thread-safe UI updates and Subprocess-safe Abort
                if self.abort_separation:
                    self.after(0, lambda: self.progress_text.configure(text="Separation aborted."))
                    self.after(0, lambda: self.progress_bar.configure(mode="indeterminate"))
                    self.after(0, lambda: self.progress_bar.set(0))
                    self.after(0, lambda: self.abort_button.grid_remove())
                    self.after(0, lambda: self.progress_bar.grid_remove())
                    
                    # Raise a standard exception so the separator modules can catch it and kill subprocesses
                    raise RuntimeError("ABORT_REQUESTED") 
                
                # 2. Thread-safe GUI manipulation
                if (percent == 0 or percent == 100) and ("Ready" in message or "completed" in message):  
                    self.abort_separation = False
                    self.after(0, lambda: self.abort_button.grid_remove()) 
                    self.after(0, lambda: self.progress_bar.configure(mode="indeterminate"))
                    self.after(0, lambda: self.progress_bar.set(0))
                    self.after(0, lambda: self.progress_bar.grid_remove())
                else:
                    self.after(0, lambda: self.abort_button.grid())
                    self.after(0, lambda: self.progress_bar.configure(mode="determinate"))
                    self.after(0, lambda: self.progress_bar.set(percent / 100.0))
                    self.after(0, lambda: self.progress_bar.grid())
                    
                self.after(0, lambda: self.progress_text.configure(text=message))

        try:
            update_progress(5, f"Starting separation... Loading {ai_tool} model.")

            # --- LAZY LOADING BLOCK ---
            # We import and initialize the models ONLY when needed to save RAM
            if ai_tool == "Spleeter" and self.spleeter_sep is None:
                import separators.spleeter_separator as spleeter_sep
                self.spleeter_sep = spleeter_sep.SpleeterSeparator()
                
            elif ai_tool == "Demucs" and getattr(self, "demucs_sep", None) is None:
                import separators.demucs_separator as demucs_sep
                self.demucs_sep = demucs_sep.DemucsSeparator()
                
            elif ai_tool == "OpenUnmix" and getattr(self, "openunmix_sep", None) is None:
                import separators.openunmix_separator as openunmix_sep
                self.openunmix_sep = openunmix_sep.OpenUnmixSeparator()
            # --------------------------

            update_progress(10, f"{ai_tool} loaded. Processing audio...")

            success = False
            vocals_path = None
            instr_path = None
            result = None
            
            # Now the separation logic executes safely
            if ai_tool == "Spleeter":
                result = self.spleeter_sep.separate(
                    input_path, song_name, vocals_folder, instr_folder, fmt, sr, bitrate,
                    progress_callback=update_progress
                )
            elif ai_tool == "Demucs":
                result = self.demucs_sep.separate(
                    input_path, song_name, vocals_folder, instr_folder,
                    model, fmt, sr, bitrate, bit_depth, mp3_preset, shifts,
                    progress_callback=update_progress
                )
            elif ai_tool == "OpenUnmix":
                result = self.openunmix_sep.separate(
                    input_path, song_name, vocals_folder, instr_folder, model, fmt, sr, bitrate,
                    progress_callback=update_progress
                )
                
            if isinstance(result, tuple) and len(result) >= 3:
                success, vocals_path, instr_path = result
            else:
                success = False

            if not success:
                update_progress(0, f"Separation failed for {ai_tool} on {song_name}. Check terminal.")
                logging.info(f"Separation failed for {ai_tool} on {song_name}.")
                return
            else:
                self.after(0, lambda: self.progress_text.configure(text=f"Separation completed! Files saved as {vocals_path}, {instr_path}."))   
                self.after(0, self.load_outputs)

        except Exception as e:
            # 3. Handle the clean abort silently, log real errors
            if str(e) == "ABORT_REQUESTED":
                logging.info("Separation cleanly aborted by user.")
            else:
                update_progress(0, f"Error: {str(e)}")
                logging.error(f"Thread error: {e}", exc_info=True)
                
        finally:
            # Hide Abort button and progress bar after completion or crash
            self.after(0, lambda: self.abort_button.grid_remove())  
            self.after(0, lambda: self.progress_bar.grid_remove())

    def run_standalone_transcription(self):
        """Runs transcription on a file already in the Vocals listbox."""
        sel = self.vocals_listbox.curselection()
        if not sel:
            messagebox.showwarning("Selection Required", "Please select a file from the Vocals list first.")
            return
        
        filename = self.vocals_listbox.get(sel[0])
        vocal_path = os.path.join(self.output_folders["vocals"], filename)
        
        tool = self.trans_tool_var.get()
        model = self.trans_model_var.get()
        lang = self.trans_lang_var.get()
        # Get the state of the Speaker ID switch
        use_spk = self.use_spk_id_var.get() if tool == "vosk" else False
        
        # Create output name based on tool and model
        base_name = os.path.splitext(filename)[0]
        out_name = f"{base_name}_{tool}_{model.replace('/', '_')}.txt"
        out_path = os.path.join(self.output_folders["transcriptions"], out_name)

        # Add 'use_spk' to the thread arguments
        threading.Thread(target=self._exec_standalone_trans, 
                         args=(vocal_path, out_path, tool, model, lang, use_spk), 
                         daemon=True).start()

    def _exec_standalone_trans(self, input_p, output_p, tool, model, lang, use_spk):
        """
        Executes transcription in a background thread to prevent UI freezing.
        Handles lazy-loading, real-time progress tracking, and file saving.
        """
        def update_trans_progress(percent, message):
            self.after(0, lambda: self.progress_bar.set(percent / 100.0))
            self.after(0, lambda: self.progress_text.configure(text=message))

        # --- STAGE 1: UI Prep ---
        self.after(0, lambda: self.progress_bar.grid())
        update_trans_progress(10, f"Initializing {tool} ({model})...")
        
        # --- STAGE 2: Lazy Loading ---
        if tool == "whisper" and getattr(self, "whisper_trans", None) is None:
            import separators.whisper_transcription as whisper_trans
            self.whisper_trans = whisper_trans.WhisperTranscription()
            
        elif tool == "vosk" and getattr(self, "vosk_trans", None) is None:
            import separators.vosk_transcription as vosk_trans
            self.vosk_trans = vosk_trans.VoskTranscription()
            
        elif tool == "wav2vec2" and getattr(self, "wav2vec2_trans", None) is None:
            import separators.wav2vec2_transcription as w2v2_trans 
            self.wav2vec2_trans = w2v2_trans.Wav2Vec2Transcription()

        success = False
        saved_filename = "Transcription file" 

        # --- STAGE 3: Execution (Real Progress!) ---
        try:
            # Run the selected tool and capture the (success, filename) tuple
            if tool == "whisper":
                result = self.whisper_trans.transcribe(
                    audio_path=input_p, 
                    output_path=output_p, 
                    model_name=model, 
                    language=lang,
                    progress_callback=update_trans_progress
                )
            elif tool == "wav2vec2":
                result = self.wav2vec2_trans.transcribe(
                    audio_path=input_p, 
                    output_path=output_p, 
                    model_name=model,
                    progress_callback=update_trans_progress
                )
            elif tool == "vosk":
                result = self.vosk_trans.transcribe(
                    audio_path=input_p, 
                    output_path=output_p, 
                    model_name=model, 
                    use_diarization=use_spk,
                    progress_callback=update_trans_progress
                )
            
            # Unpack the result properly for ALL models
            if isinstance(result, tuple) and len(result) == 2:
                success, saved_filename = result
            else:
                success = result # Fallback just in case
                
            # --- STAGE 4: UI Cleanup ---
            if success:
                update_trans_progress(100, f"Success! Saved as: {saved_filename}")
                self.after(3000, lambda: self.progress_bar.grid_remove()) # Let user read the success message
                self.after(0, self.load_outputs)
            else:
                update_trans_progress(0, f"{tool} transcription failed. Check terminal.")
                
        except Exception as e:
            logging.error(f"Standalone trans error: {e}", exc_info=True)
            update_trans_progress(0, f"Error: {str(e)}")

if __name__ == "__main__":
    app = SeparationApp()
    app.mainloop()
