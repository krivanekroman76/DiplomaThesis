import warnings
warnings.simplefilter('ignore')  # Hide unnecessary warnings
import logging
import sys
import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import platform
import subprocess
import threading
import json
# Separation tools
import separators.spleeter_separator as spleeter
import separators.demucs_separator as demucs
import separators.openunmix_separator as openunmix
# Transcription tools
import separators.whisper_transcription as whisper 
import separators.wav2vec2_transcription as wav2vec2 
import separators.vosk_transcription as vosk

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def get_base_path():
    if hasattr(sys, '_MEIPASS'):
        # Running as a PyInstaller .exe
        return sys._MEIPASS
    # Running as a normal script
    return os.path.abspath(".")

# Use this to prefix your local folder paths
base_path = get_base_path()

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

        # Load settings from file
        self.settings_file = "settings.json"
        self.load_settings()
        
        # Ensure folders exist
        os.makedirs(self.input_folder, exist_ok=True)
        for folder in self.output_folders.values():
            os.makedirs(folder, exist_ok=True)
        
        # Instantiate separator objects
        self.spleeter_sep = spleeter.SpleeterSeparator()
        self.demucs_sep = demucs.DemucsSeparator()
        self.openunmix_sep = openunmix.OpenUnmixSeparator()
        self.whisper_trans = whisper.WhisperTranscription()
        self.wav2vec2_trans = wav2vec2.Wav2Vec2Transcription()
        self.vosk_trans = vosk.VoskTranscription()

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

        # Sidebar
        self.sidebar = ctk.CTkFrame(main_frame, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="ns", rowspan=2)
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)
        # Configure rows for top-aligned navigation and bottom-aligned settings
        self.sidebar.grid_rowconfigure(3, weight=1)  # Spacer row to push settings down

        # Navigation buttons in sidebar (top-aligned)
        input_button = ctk.CTkButton(
            self.sidebar, 
            text="Input", 
            command=self.show_input,
            width=180
        )
        input_button.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        output_button = ctk.CTkButton(
            self.sidebar, 
            text="Output", 
            command=self.show_output,
            width=180
        )
        output_button.grid(row=1, column=0, padx=20, pady=(20, 10), sticky="ew")

        settings_button = ctk.CTkButton(
            self.sidebar, 
            text="Settings", 
            command=self.show_settings,
            width=180
        )
        settings_button.grid(row=4, column=0, padx=20, pady=(20, 10), sticky="ew")

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
        scaling_values = [f"{i}%" for i in range(50, 201, 10)]  # 50%, 60%, ..., 200%
        self.scaling_optionemenu = ctk.CTkOptionMenu(
            self.sidebar, 
            values=scaling_values,
            command=self.change_scaling_event,
            width=160
        )
        self.scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.scaling_optionemenu.set(self.scaling)

        # Content frame
        self.content_frame = ctk.CTkFrame(main_frame)
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

        # Input, output, and settings frames
        self.input_frame = ctk.CTkFrame(self.content_frame)
        self.output_frame = ctk.CTkFrame(self.content_frame)
        self.settings_frame = ctk.CTkScrollableFrame(self.content_frame)

        # Create tab contents
        self.create_input_tab()
        self.create_output_tab()
        self.create_settings_tab() 
        #Change listbox font size from settings.json
        self.apply_font_size()
        self.update_listbox_themes()

        # Buttons in input tab
        self.input_button = input_button
        self.output_button = output_button
        self.settings_button = settings_button

        # Progress bar and text area at the bottom (non-modal, accessible)
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
        
        # Initially show input
        self.show_input()

    def load_settings(self):
        """Loads settings from a JSON file or initializes them with default values."""
        defaults = {
            "input_folder": "input",
            "vocals_folder": "output/vocals",
            "instrumentals_folder": "output/instrumentals",
            "transcriptions_folder": "output/text",
            "appearance_mode": "Dark",
            "scaling": "100%",
            "font_size": 12,
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

        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Load paths and UI preferences
                self.input_folder = data.get("input_folder", defaults["input_folder"])
                self.output_folders = {
                    "vocals": data.get("vocals_folder", defaults["vocals_folder"]),
                    "instrumentals": data.get("instrumentals_folder", defaults["instrumentals_folder"]),
                    "transcriptions": data.get("transcriptions_folder", defaults["transcriptions_folder"])
                }
                self.appearance_mode = data.get("appearance_mode", defaults["appearance_mode"])
                self.scaling = data.get("scaling", defaults["scaling"])
                self.font_size = data.get("font_size", 12)
                
                # Load model configurations
                self.separator_models = data.get("separator_models", defaults["separator_models"])
                self.transcription_models = data.get("transcription_models", defaults["transcription_models"])
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Settings error: {e}. Restoring defaults.")
                self.set_defaults(defaults)
        else:
            # Create settings with default values if file does not exist
            self.set_defaults(defaults)
    
    def set_defaults(self, defaults):
        self.input_folder = defaults["input_folder"]
        self.output_folders = {
            "vocals": defaults["vocals_folder"],
            "instrumentals": defaults["instrumentals_folder"],
            "transcriptions": defaults["transcriptions_folder"]
        }
        self.appearance_mode = defaults["appearance_mode"]
        self.scaling = defaults["scaling"]
        self.font_size = 12
        self.separator_models = defaults["separator_models"]
        self.transcription_models = defaults["transcription_models"]
        self.save_settings()
        self.load_settings()
        self.show_settings()
        
    def save_settings(self):
        """Save current folders, models, and settings to settings.json."""
        data = {
            "input_folder": self.input_folder,
            "vocals_folder": self.output_folders["vocals"],
            "instrumentals_folder": self.output_folders["instrumentals"],
            "transcriptions_folder": self.output_folders["transcriptions"],
            "appearance_mode": self.appearance_mode,
            "scaling": self.scaling,
            "font_size": self.font_size,
            "separator_models": self.separator_models,
            "transcription_models": self.transcription_models
        }
        try:
            with open(self.settings_file, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Settings saved to {self.settings_file}")
        except Exception as e:
            logging.info(f"Error saving settings: {e}")
    
    def show_input(self):
        self.input_frame.grid(row=0, column=0, sticky="nsew")
        self.output_frame.grid_forget()
        self.settings_frame.grid_forget()
        # Highlight active button (black/white bg, darker hover, contrasting text)
        if ctk.get_appearance_mode() == "Dark":
            self.input_button.configure(fg_color="#FFFFFF", text_color="#000000", hover_color="#CCCCCC")  # White bg, black text
        else:
            self.input_button.configure(fg_color="#000000", text_color="#FFFFFF", hover_color="#333333")  # Black bg, white text
        # Reset others with default colors
        self.output_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"], text_color=ctk.ThemeManager.theme["CTkButton"]["text_color"], hover_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"])
        self.settings_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"], text_color=ctk.ThemeManager.theme["CTkButton"]["text_color"], hover_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"])

    def show_output(self):
        self.output_frame.grid(row=0, column=0, sticky="nsew")
        self.input_frame.grid_forget()
        self.settings_frame.grid_forget()
        # Highlight active button
        if ctk.get_appearance_mode() == "Dark":
            self.output_button.configure(fg_color="#FFFFFF", text_color="#000000", hover_color="#CCCCCC")
        else:
            self.output_button.configure(fg_color="#000000", text_color="#FFFFFF", hover_color="#333333")
        # Reset others
        self.input_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"], text_color=ctk.ThemeManager.theme["CTkButton"]["text_color"], hover_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"])
        self.settings_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"], text_color=ctk.ThemeManager.theme["CTkButton"]["text_color"], hover_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"])

    def show_settings(self):
        self.settings_frame.grid(row=0, column=0, sticky="nsew")
        self.input_frame.grid_forget()
        self.output_frame.grid_forget()
        # Highlight active button
        if ctk.get_appearance_mode() == "Dark":
            self.settings_button.configure(fg_color="#FFFFFF", text_color="#000000", hover_color="#CCCCCC")
        else:
            self.settings_button.configure(fg_color="#000000", text_color="#FFFFFF", hover_color="#333333")
        # Reset others
        self.input_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"], text_color=ctk.ThemeManager.theme["CTkButton"]["text_color"], hover_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"])
        self.output_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"], text_color=ctk.ThemeManager.theme["CTkButton"]["text_color"], hover_color=ctk.ThemeManager.theme["CTkButton"]["hover_color"])

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
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)
        self.scaling = new_scaling
        self.save_settings()
        # Reinitialize all tabs, because changing scale loads all items so it broke GUI tabs
        self.progress_bar.grid_remove()
        self.abort_button.grid_remove()         
        if self.input_frame.winfo_ismapped():  # If input tab is active
            print("Input frame detected")
            self.create_input_tab()
        elif self.output_frame.winfo_ismapped():  # If output tab is active
            print("Output frame detected")
            self.create_output_tab()
        elif self.settings_frame.winfo_ismapped():  # If settings tab is active
            print("Settings frame detected")
            self.create_settings_tab()

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
        frame = self.input_frame
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=0)
        frame.grid_rowconfigure(0, weight=0)
        frame.grid_rowconfigure(1, weight=1)

        # Path bar frame
        path_frame = ctk.CTkFrame(frame)
        path_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        path_frame.grid_columnconfigure(1, weight=1)

        path_label = ctk.CTkLabel(path_frame, text="Current Folder:", anchor="w")
        path_label.grid(row=0, column=0, sticky="ew", padx=10)

        self.path_var = tk.StringVar(value=self.input_folder)
        self.path_entry = ctk.CTkEntry(path_frame, textvariable=self.path_var)
        self.path_entry.grid(row=0, column=1, columnspan=6, sticky="ew", padx=10)
        self.path_entry.bind("<Return>", self.on_path_enter)

        self.back_button = ctk.CTkButton(path_frame, text="Back", command=self.go_back, width=80)
        self.back_button.grid(row=2, column=0, sticky="ew", padx=5)

        self.change_folder_button = ctk.CTkButton(path_frame, text="Change Folder/New Folder", command=self.change_input_folder)
        self.change_folder_button.grid(row=2, column=1, sticky="ew", padx=5)

        self.add_song_button = ctk.CTkButton(path_frame, text="Add Song", command=self.add_song)
        self.add_song_button.grid(row=2, column=3, sticky="ew", padx=5)

        # Songs/Folders list
        self.songs_listbox = tk.Listbox(frame, bg="#000000", fg="#FFFFFF")
        self.songs_listbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0,10))
        self.songs_listbox.bind("<Double-Button-1>", self.on_listbox_double_click)

        # Separation menu frame 
        sep_scrollable = ctk.CTkScrollableFrame(frame, width=350, height=600) 
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
        self.model_var = tk.StringVar(value="umxl")  # Default for OpenUnmix
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

        # Conditional frames for format-specific options
        # WAV/FLAC options
        self.wav_flac_frame = ctk.CTkFrame(sep_scrollable)
        self.wav_flac_frame.grid(row=8, column=0, sticky="ew", padx=20, pady=5)
        self.wav_flac_frame.grid_remove()  # Hide initially

        # Channel selection (inside wav_flac_frame)
        self.channel_label = ctk.CTkLabel(self.wav_flac_frame, text="Channels:", anchor="w")
        self.channel_label.grid(row=0, column=0, sticky="w", padx=20, pady=(10,0))
        self.channel_var = tk.StringVar(value="Stereo")
        self.channel_menu = ctk.CTkOptionMenu(self.wav_flac_frame, variable=self.channel_var, values=["Mono", "Stereo"])
        self.channel_menu.grid(row=1, column=0, sticky="ew", padx=20, pady=5)

        # Sample Rate
        self.sr_label = ctk.CTkLabel(self.wav_flac_frame, text="Sample Rate (Hz):", anchor="w")
        self.sr_label.grid(row=2, column=0, sticky="w", padx=20, pady=(10,0))
        self.sr_var = tk.StringVar(value="44100")
        self.sr_entry = ctk.CTkEntry(self.wav_flac_frame, textvariable=self.sr_var, width=150, placeholder_text="44100")
        self.sr_entry.grid(row=3, column=0, sticky="ew", padx=20, pady=5)

        # Bit depth radiobuttons (for Demucs WAV)
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
        self.mp3_frame.grid_remove()  # Hide initially

        # Bitrate
        self.bitrate_label = ctk.CTkLabel(self.mp3_frame, text="Bitrate (kbps):", anchor="w")
        self.bitrate_label.grid(row=0, column=0, sticky="w", padx=20, pady=(10,0))
        self.bitrate_var = tk.StringVar(value="192")
        self.bitrate_entry = ctk.CTkEntry(self.mp3_frame, textvariable=self.bitrate_var, width=150, placeholder_text="192")
        self.bitrate_entry.grid(row=1, column=0, sticky="ew", padx=20, pady=5)

        # MP3 Preset Slider
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
        self.shifts_frame.grid_remove()  # Hide initially
        self.shifts_label = ctk.CTkLabel(self.shifts_frame, text="Shifts (increases quality but slows process):", anchor="w")
        self.shifts_label.grid(row=0, column=0, sticky="w", padx=20, pady=5)
        self.shifts_var = tk.StringVar(value="1")
        self.shifts_entry = ctk.CTkEntry(self.shifts_frame, textvariable=self.shifts_var, width=150, placeholder_text="1")
        self.shifts_entry.grid(row=1, column=0, sticky="ew", padx=20, pady=5)

        # Separate button
        self.separate_button = ctk.CTkButton(sep_scrollable, text="Separate", command=self.separate_audio)
        self.separate_button.grid(row=17, column=0, sticky="ew", padx=20, pady=(20,10))

        # Initial tool change to set defaults
        self.on_tool_change()

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
            
    def apply_font_size(self):
        font = ("TkDefaultFont", self.font_size)
        # Apply to listboxes
        self.songs_listbox.configure(font=font)
        self.vocals_listbox.configure(font=font)
        self.instr_listbox.configure(font=font)
        self.trans_listbox.configure(font=font)
        # Apply to textboxes in settings tab
        self.demucs_models_text.configure(font=font)
        self.openunmix_models_text.configure(font=font)
        self.whisper_models_text.configure(font=font)
        self.wav2vec2_models_text.configure(font=font)
        self.vosk_models_text.configure(font=font)
     
    def create_output_tab(self):
        frame = self.output_frame
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=0) # Right sidebar for tools
        frame.grid_rowconfigure((1, 3, 5), weight=1)

        # --- LEFT SIDE: LISTBOXES (Your existing layout) ---
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

        # --- RIGHT SIDE: TRANSCRIPTION MENU (Matches Input Tab style) ---
        trans_menu = ctk.CTkScrollableFrame(frame, width=350, height=600)
        trans_menu.grid(row=0, column=1, rowspan=6, sticky="nsew", padx=10, pady=10)
        trans_menu.grid_columnconfigure(0, weight=1)

        self.trans_model_label = ctk.CTkLabel(trans_menu, text="Transcription Menu", font=ctk.CTkFont(size=20, weight="bold"))
        self.trans_model_label.grid(row=0, column=0, pady=(10, 20))
        
        # Tool Selection (Radio Buttons)
        ctk.CTkLabel(trans_menu, text="Tool:", anchor="w").grid(row=1, column=0, sticky="w", padx=20)
        self.trans_tool_var = tk.StringVar(value="whisper")
        
        ctk.CTkRadioButton(trans_menu, text="Whisper", variable=self.trans_tool_var, value="whisper", 
                           command=self.on_trans_tool_change).grid(row=2, column=0, sticky="w", padx=20, pady=5)
        ctk.CTkRadioButton(trans_menu, text="Wav2Vec2", variable=self.trans_tool_var, value="wav2vec2", 
                           command=self.on_trans_tool_change).grid(row=3, column=0, sticky="w", padx=20, pady=5)
        ctk.CTkRadioButton(trans_menu, text="vosk", variable=self.trans_tool_var, value="vosk", 
                           command=self.on_trans_tool_change).grid(row=4, column=0, sticky="w", padx=20, pady=5)

        ctk.CTkLabel(trans_menu, text="Model:", anchor="w").grid(row=5, column=0, sticky="w", padx=20, pady=(10, 0))
        self.trans_model_var = tk.StringVar()
        self.trans_model_menu = ctk.CTkOptionMenu(trans_menu, variable=self.trans_model_var, values=[], width=200)
        self.trans_model_menu.grid(row=6, column=0, sticky="ew", padx=20, pady=5)

        # Language Selection
        self.trans_lang_label = ctk.CTkLabel(trans_menu, text="Language:", anchor="w")
        self.trans_lang_label.grid(row=7, column=0, sticky="w", padx=20, pady=(10, 0))
        self.trans_lang_var = tk.StringVar(value="auto")
        self.trans_lang_menu = ctk.CTkOptionMenu(trans_menu, variable=self.trans_lang_var, values=["auto", "cs", "en", "fr", "de", "es"], width=200)
        self.trans_lang_menu.grid(row=8, column=0, sticky="ew", padx=20, pady=5)

        # Speaker Identification Toggle (Only for Vosk)
        self.use_spk_id_var = tk.BooleanVar(value=False)
        self.spk_toggle = ctk.CTkSwitch(trans_menu, text="Identify Speakers", 
                                       variable=self.use_spk_id_var,
                                       progress_color="#1f538d")
        self.spk_toggle.grid(row=8, column=0, sticky="w", padx=20, pady=10)
        self.spk_toggle.grid_remove() # Hidden by default
        
        # Run Button
        self.trans_button = ctk.CTkButton(trans_menu, text="Transcribe", command=self.run_standalone_transcription)
        self.trans_button.grid(row=9, column=0, sticky="ew", padx=20, pady=(30, 10))
        
        ctk.CTkLabel(trans_menu, text="Note: Select a file in 'Vocals' list first.", font=ctk.CTkFont(size=10, slant="italic")).grid(row=10, column=0, padx=20)

        self.on_trans_tool_change()
        
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
                
    def create_settings_tab(self):
        frame = self.settings_frame
        frame.grid_columnconfigure((1, 4), weight=1)

        # Folder label
        settings_label = ctk.CTkLabel(frame, text="Default Folders Settings", font=ctk.CTkFont(size=20, weight="bold"))
        settings_label.grid(row=0, column=0, columnspan=2, pady=(20, 20))
        # Input folder
        input_label = ctk.CTkLabel(frame, text="Input Folder:", anchor="w")
        input_label.grid(row=1, column=0, sticky="w", padx=20, pady=(10, 0))
        self.settings_input_var = tk.StringVar(value=self.input_folder)
        input_entry = ctk.CTkEntry(frame, textvariable=self.settings_input_var, width=400)
        input_entry.grid(row=1, column=1, sticky="ew", padx=20, pady=5)
        # Vocals folder
        vocals_label = ctk.CTkLabel(frame, text="Vocals Folder:", anchor="w")
        vocals_label.grid(row=2, column=0, sticky="w", padx=20, pady=(10, 0))
        self.settings_vocals_var = tk.StringVar(value=self.output_folders["vocals"])
        vocals_entry = ctk.CTkEntry(frame, textvariable=self.settings_vocals_var, width=400)
        vocals_entry.grid(row=2, column=1, sticky="ew", padx=20, pady=5)
        # Instrumentals folder
        instr_label = ctk.CTkLabel(frame, text="Instrumentals Folder:", anchor="w")
        instr_label.grid(row=3, column=0, sticky="w", padx=20, pady=(10, 0))
        self.settings_instr_var = tk.StringVar(value=self.output_folders["instrumentals"])
        instr_entry = ctk.CTkEntry(frame, textvariable=self.settings_instr_var, width=400)
        instr_entry.grid(row=3, column=1, sticky="ew", padx=20, pady=5)
        # Transcriptions folder
        trans_label = ctk.CTkLabel(frame, text="Transcriptions Folder:", anchor="w")
        trans_label.grid(row=4, column=0, sticky="w", padx=20, pady=(10, 0))
        self.settings_trans_var = tk.StringVar(value=self.output_folders["transcriptions"])
        trans_entry = ctk.CTkEntry(frame, textvariable=self.settings_trans_var, width=400)
        trans_entry.grid(row=5, column=1, sticky="ew", padx=20, pady=5)
        
        font_size_label = ctk.CTkLabel(frame, text="Listbox Font Size:", anchor="w")
        font_size_label.grid(row=6, column=0, sticky="w", padx=20, pady=(20, 0))
        self.font_size_var = tk.StringVar(value=str(self.font_size))
        font_size_menu = ctk.CTkOptionMenu(frame, variable=self.font_size_var, values=[str(i) for i in range(10, 32)], width=200)
        font_size_menu.grid(row=6, column=1, sticky="ew", padx=20, pady=5)

        # Buttons
        save_btn = ctk.CTkButton(frame, text="Save Changes", command=self.save_settings_changes)
        save_btn.grid(row=7, column=1, columnspan=4, padx=10, pady=5, sticky="w")
        restore_btn = ctk.CTkButton(frame, text="Restore Defaults", command=self.restore_defaults)
        restore_btn.grid(row=7, column=1, columnspan=4, padx=10, pady=5, sticky="s")

        # Models label in second column
        model_label = ctk.CTkLabel(frame, text="Model Dropdown Menu Settings", font=ctk.CTkFont(size=20, weight="bold"))
        model_label.grid(row=0, column=3, columnspan=2, pady=(20, 20))
        # Separator Models - Demucs
        demucs_models_label = ctk.CTkLabel(frame, text="Demucs Models (Edit/Reorder):", anchor="w")
        demucs_models_label.grid(row=1, column=3, sticky="w", padx=20, pady=(20, 0))
        self.demucs_models_text = ctk.CTkTextbox(frame, width=400, height=50)
        self.demucs_models_text.grid(row=1, column=4, sticky="ew", padx=20, pady=5)
        self.demucs_models_text.insert("0.0", json.dumps(self.separator_models.get("Demucs", [])))
        # Separator Models - OpenUnmix
        openunmix_models_label = ctk.CTkLabel(frame, text="OpenUnmix Models (Edit/Reorder):", anchor="w")
        openunmix_models_label.grid(row=2, column=3, sticky="w", padx=20, pady=(20, 0))
        self.openunmix_models_text = ctk.CTkTextbox(frame, width=400, height=50)
        self.openunmix_models_text.grid(row=3, column=4, sticky="ew", padx=20, pady=5)
        self.openunmix_models_text.insert("0.0", json.dumps(self.separator_models.get("OpenUnmix", [])))
        # Transcription Models - Whisper
        whisper_models_label = ctk.CTkLabel(frame, text="Whisper Models (Edit/Reorder):", anchor="w")
        whisper_models_label.grid(row=4, column=3, sticky="w", padx=20, pady=(20, 0))
        self.whisper_models_text = ctk.CTkTextbox(frame, width=400, height=50)
        self.whisper_models_text.grid(row=4, column=4, sticky="ew", padx=20, pady=5)
        self.whisper_models_text.insert("0.0", json.dumps(self.transcription_models.get("whisper", [])))
        # Transcription Models - Wav2Vec2
        wav2vec2_models_label = ctk.CTkLabel(frame, text="Wav2Vec2 Models (Edit/Reorder):", anchor="w")
        wav2vec2_models_label.grid(row=5, column=3, sticky="w", padx=20, pady=(20, 0))
        self.wav2vec2_models_text = ctk.CTkTextbox(frame, width=400, height=50)
        self.wav2vec2_models_text.grid(row=5, column=4, sticky="ew", padx=20, pady=5)
        self.wav2vec2_models_text.insert("0.0", json.dumps(self.transcription_models.get("wav2vec2", [])))
        # Transcription Models - vosk
        vosk_models_label = ctk.CTkLabel(frame, text="vosk Models (Edit/Reorder):", anchor="w")
        vosk_models_label.grid(row=6, column=3, sticky="w", padx=20, pady=(20, 0))
        self.vosk_models_text = ctk.CTkTextbox(frame, width=400, height=50)
        self.vosk_models_text.grid(row=6, column=4, sticky="ew", padx=20, pady=5)
        self.vosk_models_text.insert("0.0", json.dumps(self.transcription_models.get("vosk", [])))

    def save_settings_changes(self):
        self.input_folder = self.settings_input_var.get()
        self.output_folders["vocals"] = self.settings_vocals_var.get()
        self.output_folders["instrumentals"] = self.settings_instr_var.get()
        self.output_folders["transcriptions"] = self.settings_trans_var.get()
        self.font_size = int(self.font_size_var.get())
        try:
            self.separator_models["Demucs"] = json.loads(self.demucs_models_text.get("0.0", "end"))
            self.separator_models["OpenUnmix"] = json.loads(self.openunmix_models_text.get("0.0", "end"))
            self.transcription_models["whisper"] = json.loads(self.whisper_models_text.get("0.0", "end"))
            self.transcription_models["wav2vec2"] = json.loads(self.wav2vec2_models_text.get("0.0", "end"))
            self.transcription_models["vosk"] = json.loads(self.vosk_models_text.get("0.0", "end"))
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Invalid JSON in model fields.")
            return
        self.save_settings()
        os.makedirs(self.input_folder, exist_ok=True)
        for folder in self.output_folders.values():
            os.makedirs(folder, exist_ok=True)
        self.load_input()
        self.load_outputs()
        self.on_tool_change()
        self.on_trans_tool_change()
        self.apply_font_size()
        messagebox.showinfo("Settings Saved", "Default folders and models updated and saved.")

    def restore_defaults(self):
        defaults = {
            "input_folder": "input",
            "vocals_folder": "output/vocals",
            "instrumentals_folder": "output/instrumentals",
            "transcriptions_folder": "output/text",
            "separator_models": {
                "Spleeter": [],
                "Demucs": ["mdx", "mdx_extra", "htdemucs"],
                "OpenUnmix": ["umxl", "umxhq", "umx"]
            },
            "transcription_models": {
                "whisper": ["large", "medium", "small", "tiny", "base", "turbo"],
                "wav2vec2": ["facebook/wav2vec2-base-960h", "facebook/wav2vec2-large-960h"],
                "vosk": [
                    "vosk-model-small-cs-0.4-rhassspy",
                    "vosk-model-small-en-us-0.15",
                    "vosk-model-en-us-0.22-lgraph",
                    "vosk-model-spk-0.4"
                ]
            }
        }
        # Restore all attributes from defaults dictionary
        self.input_folder = defaults["input_folder"]
        self.output_folders = {
            "vocals": defaults["vocals_folder"],
            "instrumentals": defaults["instrumentals_folder"],
            "transcriptions": defaults["transcriptions_folder"]
        }
        self.separator_models = defaults["separator_models"]
        self.transcription_models = defaults["transcription_models"]
        
        # Save updated settings to the configuration file
        self.save_settings()
        
        # Update UI control variables
        self.settings_input_var.set(self.input_folder)
        self.settings_vocals_var.set(self.output_folders["vocals"])
        self.settings_instr_var.set(self.output_folders["instrumentals"])
        self.settings_trans_var.set(self.output_folders["transcriptions"])
        
        # Update model textboxes with JSON formatted strings
        self.demucs_models_text.delete("0.0", "end")
        self.demucs_models_text.insert("0.0", json.dumps(self.separator_models["Demucs"], indent=4))
        
        self.openunmix_models_text.delete("0.0", "end")
        self.openunmix_models_text.insert("0.0", json.dumps(self.separator_models["OpenUnmix"], indent=4))
        
        self.whisper_models_text.delete("0.0", "end")
        self.whisper_models_text.insert("0.0", json.dumps(self.transcription_models["whisper"], indent=4))
        
        self.wav2vec2_models_text.delete("0.0", "end")
        self.wav2vec2_models_text.insert("0.0", json.dumps(self.transcription_models["wav2vec2"], indent=4))
        
        self.vosk_models_text.delete("0.0", "end")
        self.vosk_models_text.insert("0.0", json.dumps(self.transcription_models["vosk"], indent=4))
        
        # Ensure that input and output directories exist
        os.makedirs(self.input_folder, exist_ok=True)
        for folder in self.output_folders.values():
            os.makedirs(folder, exist_ok=True)
        
        # Reload file lists and UI data
        self.load_input()
        self.load_outputs()
        
        # Notify user and switch to the settings tab
        messagebox.showinfo("Defaults Restored", "All settings, including Vosk models, have been reset to defaults.")
        self.show_settings()
        
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
        This is the 'execution' part, called by the threaded function.
        Updates progress bar/text with console reports.
        Handles separation and UI updates.
        """
        # Define progress callback
        def update_progress(percent, message):
                # Check for abort on every progress update (responsive even during long separations)
                if self.abort_separation:
                    self.after(0, lambda: self.progress_text.configure(text="Separation aborted."))
                    self.after(0, lambda: self.progress_bar.configure(mode="indeterminate"))
                    self.after(0, lambda: self.progress_bar.set(0))
                    self.abort_button.grid_remove()
                    self.progress_bar.grid_remove()
                    raise SystemExit("Aborted")  # Exit the thread immediately
                
                # Hide bar for idle states, show bar when separating
                if (percent == 0 or percent == 100) and ("Ready" in message or "completed" in message):  
                    self.abort_separation = False
                    self.abort_button.grid_remove() 
                    self.progress_bar.configure(mode="indeterminate")
                    self.progress_bar.set(0)
                    self.progress_bar.grid_remove()
                else:
                    self.abort_button.grid()
                    self.progress_bar.configure(mode="determinate")
                    self.progress_bar.set(percent / 100.0)
                    self.progress_bar.grid()
                self.after(0, lambda: self.progress_text.configure(text=message))

        try:
            # Update initial progress
            update_progress(0, "Starting separation...")

            success = False
            vocals_path = None
            instr_path = None
            result = None
            if ai_tool == "Spleeter":
                result = self.spleeter_sep.separate(
                    input_path, song_name, vocals_folder, instr_folder, fmt, sr, bitrate,
                    progress_callback=update_progress  # Pass the callback
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
            if isinstance(result, tuple) and len(result) >= 4:
                success, vocals_path, instr_path = result
            else:
                success = False

            if not success:
                update_progress(0, "Separation failed for {ai_tool} on {song_name}. Check terminal for errors.")
                logging.info(f"Separation failed for {ai_tool} on {song_name}.")
                return
            else:
                # Final updates Print names of new files and update output tab
                self.after(0, lambda: self.progress_text.configure(text=f"Separation completed! Files saved as {vocals_path}, {instr_path}. Check output tab."))   
                self.after(0, self.load_outputs)

        except Exception as e:
            update_progress(0, f"Error: {str(e)}")
            logging.error(f"Thread error: {e}", exc_info=True)
        # Hide Abort button and progress bar after completion
        self.abort_button.grid_remove()  
        self.progress_bar.grid_remove()
            
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
            self.after(0, lambda: self.progress_text.configure(text=f"Transcribing with {tool}..."))
            
            success = False
            try:
                if tool == "whisper":
                    success = self.whisper_trans.transcribe(input_p, output_p, model, language=lang)
                elif tool == "wav2vec2":
                    success = self.wav2vec2_trans.transcribe(input_p, output_p, model)
                elif tool == "vosk":
                    success = self.vosk_trans.transcribe(input_p, output_p, model, use_diarization=use_spk)
                    
                if success:
                    self.after(0, lambda: self.progress_text.configure(text="Transcription finished!"))
                    self.after(0, self.load_outputs)
                else:
                    self.after(0, lambda: self.progress_text.configure(text="Transcription failed."))
            except Exception as e:
                logging.error(f"Standalone trans error: {e}")
                
if __name__ == "__main__":
    app = SeparationApp()
    app.mainloop()
