import warnings
warnings.simplefilter('ignore')  # Hide unnecessary warnings
import os
# STRICT GAG ORDER FOR TENSORFLOW (Must be set before any AI imports)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Hides oneDNN custom operations warnings
import logging
import sys
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from customtkinter import filedialog
import platform
import subprocess
import threading
import json
import urllib.request
import zipfile
import tempfile
import multiprocessing
# The separators and transcription tools are lazy loaded to save RAM

ctk.set_appearance_mode("System")
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

def get_app_dir():
    """Always returns the directory containing the .exe or the main .py script."""
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        return os.path.dirname(sys.executable)
    else:
        # Running as a normal Python script
        return os.path.dirname(os.path.abspath(__file__))

def get_app_dir():
    """ Get path to the permanent location of the EXE (Write-Writeable) """
    if hasattr(sys, 'frozen'): # 'frozen' means it's a PyInstaller EXE
        return os.path.dirname(sys.executable)
    return os.path.abspath(".")

# Use the permanent directory for models
app_dir = get_app_dir()

# 1. Reroute HuggingFace (Wav2Vec2)
os.environ["HF_HOME"] = os.path.join(app_dir, "pretrained_models", "huggingface")

# 2. Reroute PyTorch (Demucs & OpenUnmix)
os.environ["TORCH_HOME"] = os.path.join(app_dir, "pretrained_models", "torch")

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
        #self.minsize(900, 600)

        # 1. Load Settings and Setup Data
        self.settings_file = os.path.join(get_app_dir(), "settings.json")
        self.load_settings()
        
        os.makedirs(self.input_folder, exist_ok=True)
        for folder in self.output_folders.values():
            os.makedirs(folder, exist_ok=True)
        
        # Initialize model directory check on startup
        self.check_models_directory()

        # Initialize AI models and Data Lists
        self.spleeter_sep = self.demucs_sep = self.openunmix_sep = None
        self.whisper_trans = self.wav2vec2_trans = self.vosk_trans = None
        self.songs = []
        self.folders = []
        self.all_items = []
        self.vocals = []
        self.instrumentals = []
        self.transcriptions = []
        self.abort_separation = False
        self.input_tab_loaded = False
        self.output_tab_loaded = False

        # 2. Set Theme and Scaling (Global)
        ctk.set_appearance_mode(self.appearance_mode)
        ctk.set_default_color_theme(self.color_theme)
        ctk.set_widget_scaling(int(self.scaling.replace("%", "")) / 100)

        # 3. BUILD UI (The Body)
        self._setup_main_containers()

        # 4. Initial tab loading
        # Tells the app to draw the window immediately, then run the tab switch 100 milliseconds later
        self.after(100, lambda: self._switch_tab(self.input_frame, self.input_button, "input"))

    def _switch_tab(self, active_frame, active_button, tab_name):
        frames = [self.input_frame, self.output_frame, self.settings_frame]
        buttons = [self.input_button, self.output_button, self.settings_button]
        
        # Hide all frames
        for frame in frames:
            frame.grid_forget()

        # --- NEW: ACTIVE MEMORY MANAGEMENT (Forget inactive tabs) ---
        if tab_name != "input" and self.input_tab_loaded:
            for widget in self.input_frame.winfo_children():
                widget.destroy()
            self.input_tab_loaded = False
            
        if tab_name != "output" and self.output_tab_loaded:
            for widget in self.output_frame.winfo_children():
                widget.destroy()
            self.output_tab_loaded = False

        # Show active frame
        active_frame.grid(row=0, column=0, sticky="nsew")

        # Highlight active button
        active_button.configure(
            fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"],
            text_color="white"
        )

        # Force Tkinter to drop focus from any entry boxes before we destroy them!
        self.focus_set() 

        if tab_name != "input" and self.input_tab_loaded:
            for widget in self.input_frame.winfo_children():
                widget.destroy()
            self.input_tab_loaded = False
            
        if tab_name != "output" and self.output_tab_loaded:
            for widget in self.output_frame.winfo_children():
                widget.destroy()
            self.output_tab_loaded = False

        # --- LAZY LOADING LOGIC ---
        if tab_name == "input" and not self.input_tab_loaded:
            self.create_input_tab() # Make sure to create the base tab UI!
            self.load_input()
            self.input_tab_loaded = True
            
        elif tab_name == "output" and not self.output_tab_loaded:
            self.create_output_tab() # Make sure to create the base tab UI!
            self.load_outputs()
            self.output_tab_loaded = True
        
        elif tab_name == "settings":
            self.create_settings_tab() # Rebuild settings if needed

        # Free memory if needed
        if hasattr(self, '_free_inactive_models'):
            self._free_inactive_models(tab_name)
        
    def setup_progress_bar(self, parent):
        progress_frame = ctk.CTkFrame(parent, height=40, corner_radius=0)
        progress_frame.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(0, 10))
        progress_frame.grid_propagate(False)
        progress_frame.grid_columnconfigure(1, weight=1)

        self.progress_bar = ctk.CTkProgressBar(progress_frame, mode="determinate", height=12)
        self.progress_bar.grid(row=0, column=0, sticky="w", padx=10)
        self.progress_bar.set(0)
        self.progress_bar.grid_remove()

        self.progress_text = ctk.CTkLabel(progress_frame, text="Ready", font=ctk.CTkFont(size=12))
        self.progress_text.grid(row=0, column=1, sticky="w")

        self.abort_button = ctk.CTkButton(progress_frame, text="Abort", command=self.abort_separation_process, width=80, height=24, corner_radius=0)
        self.abort_button.grid(row=0, column=2, sticky="e", padx=10)
        self.abort_button.grid_remove()

    def change_scaling_event(self, new_scaling: str):
        # GUARD: If it's already the current scale, do absolutely nothing!
        if new_scaling == self.scaling:
            return 
            
        self.scaling = new_scaling
        self.after(100, self._master_reload_pipeline)

    def update_theme_settings(self, new_value: str, setting_type: str):
        if setting_type == "mode":
            if new_value == self.appearance_mode:
                return
            self.appearance_mode = new_value
            ctk.set_appearance_mode(self.appearance_mode)
            self.save_settings()
            
        elif setting_type == "theme":
            # GUARD: If it's already the current theme, do absolutely nothing!
            if new_value == self.color_theme:
                return
            self.color_theme = new_value
            self.after(100, self._master_reload_pipeline)

    def _setup_main_containers(self):
        # --- ROOT CONTAINER ---
        # Transparent and sharp
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True)
        
        self.main_frame.grid_columnconfigure(0, weight=0)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=0)

        # --- SMART SIDEBAR (Replaces the old "NORMAL SIDEBAR") ---
        base_scale = int(self.scaling.replace("%", ""))
        
        if base_scale >= 150:
            self.sidebar = ctk.CTkScrollableFrame(self.main_frame, width=160, corner_radius=0, fg_color="transparent")
        else:
            self.sidebar = ctk.CTkFrame(self.main_frame, width=160, corner_radius=0, fg_color="transparent")
            
        self.sidebar.grid(row=0, column=0, sticky="nsew", rowspan=2)
        
        # THE MAGIC SPACER: Row 3 acts as an invisible spring pushing row 4+ to the bottom
        self.sidebar.grid_rowconfigure(3, weight=1)
        
        # --- NAVIGATION BUTTONS ---
        btn_width = 120

        self.input_button = ctk.CTkButton(
            self.sidebar, text="Input", width=btn_width, corner_radius=0,
            command=lambda: self._switch_tab(self.input_frame, self.input_button, "input")
        )
        self.input_button.grid(row=0, column=0, padx=10, pady=(20, 10), sticky="ew")
        
        self.output_button = ctk.CTkButton(
            self.sidebar, text="Output", width=btn_width, corner_radius=0,
            command=lambda: self._switch_tab(self.output_frame, self.output_button, "output")
        )
        self.output_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.settings_button = ctk.CTkButton(
            self.sidebar, text="Settings", width=btn_width, corner_radius=0,
            command=lambda: self._switch_tab(self.settings_frame, self.settings_button, "settings")
        )
        self.settings_button.grid(row=2, column=0, padx=10, pady=(10, 20), sticky="ew")

        # --- Appearance Mode (Light/Dark) ---
        appearance_mode_label = ctk.CTkLabel(self.sidebar, text="Appearance Mode:", anchor="w")
        appearance_mode_label.grid(row=4, column=0, padx=10, pady=(10, 0), sticky="w")
        
        self.appearance_var = ctk.StringVar(value=self.appearance_mode)
        self.appearance_menu = ctk.CTkOptionMenu(
            self.sidebar, values=["Light", "Dark", "System"],
            variable=self.appearance_var,
            corner_radius=0, # Removed corners
            command=lambda val: self.update_theme_settings(val, "mode")
        )
        self.appearance_menu.grid(row=5, column=0, padx=10, pady=(5, 10), sticky="ew")

        # --- Color Theme (Blue, Green, etc.) ---
        color_theme_label = ctk.CTkLabel(self.sidebar, text="Color Theme:", anchor="w")
        color_theme_label.grid(row=6, column=0, padx=10, pady=(10, 0), sticky="w")

        self.color_var = ctk.StringVar(value=self.color_theme)
        self.color_menu = ctk.CTkOptionMenu(
            self.sidebar, values=["blue", "green", "dark-blue"],
            variable=self.color_var,
            corner_radius=0, # Removed corners
            command=lambda val: self.update_theme_settings(val, "theme")
        )
        self.color_menu.grid(row=7, column=0, padx=10, pady=(5, 10), sticky="ew")

        # --- UI SCALING (Dynamic 5-Step Carousel) ---
        scaling_label = ctk.CTkLabel(self.sidebar, text="UI Scaling:", anchor="w")
        scaling_label.grid(row=8, column=0, padx=10, pady=(10, 0), sticky="w")

        base = int(self.scaling.replace("%", ""))
        base = max(70, min(180, base)) 
        new_values = [f"{base - 20}%", f"{base - 10}%", f"{base}%", f"{base + 10}%", f"{base + 20}%"]

        self.scaling_menu = ctk.CTkSegmentedButton(
            self.sidebar, 
            values=new_values,
            corner_radius=0 # Removed corners
        )
        self.scaling_menu.grid(row=9, column=0, padx=10, pady=(5, 20), sticky="ew")
        
        self.scaling_menu.set(f"{base}%")
        self.scaling_menu.configure(command=self.change_scaling_event)

        # --- CONTENT AREA ---
        self.content_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent", corner_radius=0)
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

        # --- THE 3 MAIN TABS (These get the background colors!) ---
        # "gray85" is for light mode, "gray17" is for dark mode.
        tab_bg = ("gray85", "gray17")
        self.input_frame = ctk.CTkFrame(self.content_frame, fg_color=tab_bg, corner_radius=0)
        self.output_frame = ctk.CTkFrame(self.content_frame, fg_color=tab_bg, corner_radius=0)
        self.settings_frame = ctk.CTkFrame(self.content_frame, fg_color=tab_bg, corner_radius=0)

        # Initialize empty tab contents
        self.create_input_tab()
        self.create_output_tab()
        self.create_settings_tab()

        # --- PROGRESS BAR (Bottom) ---
        self.setup_progress_bar(self.main_frame)

    def _master_reload_pipeline(self):
        # 1. Save settings and remember current tab
        self.save_settings()
        current_tab = "input"
        if hasattr(self, "output_frame") and self.output_frame.winfo_ismapped():
            current_tab = "output"
        elif hasattr(self, "settings_frame") and self.settings_frame.winfo_ismapped():
            current_tab = "settings"

        # 2. Destroy the UI cleanly in one shot
        if hasattr(self, "main_frame"):
            self.main_frame.destroy()

        # 3. Apply new scale and theme globally
        scaling_float = int(self.scaling.replace("%", "")) / 100
        try: ctk.set_widget_scaling(scaling_float)
        except Exception: pass
        try: ctk.set_appearance_mode(self.appearance_mode)
        except Exception: pass
        try: ctk.set_default_color_theme(self.color_theme)
        except Exception: pass

        # 4. Reset lazy loading flags (forces them to rebuild fresh)
        self.input_tab_loaded = False
        self.output_tab_loaded = False

        # 5. Rebuild the main containers
        self._setup_main_containers()

        # 6. Restore active tab (This automatically triggers lazy-loading!)
        tab_map = {
            "input": (self.input_frame, self.input_button),
            "output": (self.output_frame, self.output_button),
            "settings": (self.settings_frame, self.settings_button)
        }
        f, b = tab_map.get(current_tab, (self.input_frame, self.input_button))
        self._switch_tab(f, b, current_tab)

    def _safe_rebuild_ui(self):
        # 1. Remember the current tab
        current_tab = "input"
        if hasattr(self, "output_frame") and self.output_frame.winfo_ismapped():
            current_tab = "output"
        elif hasattr(self, "settings_frame") and self.settings_frame.winfo_ismapped():
            current_tab = "settings"

        # 2. Destroy the visible UI instantly (Stops visual jumps/tearing!)
        if hasattr(self, "main_frame"):
            self.main_frame.destroy()
        
        self.update() # Force Tkinter to process the deletion

        # 3. Apply the new scale and themes globally WHILE the screen is clear
        scaling_float = int(self.scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(scaling_float)
        ctk.set_appearance_mode(self.appearance_mode)
        ctk.set_default_color_theme(self.color_theme)

        # 4. Rebuild everything with the new scale/theme perfectly applied
        self._setup_main_containers()
        self.load_input()
        self.load_outputs()

        # 5. Restore the active tab
        tab_map = {
            "input": (self.input_frame, self.input_button),
            "output": (self.output_frame, self.output_button),
            "settings": (self.settings_frame, self.settings_button)
        }
        f, b = tab_map.get(current_tab, (self.input_frame, self.input_button))
        self._switch_tab(f, b, current_tab)
        
    def create_input_tab(self):
        for widget in self.input_frame.winfo_children():
            widget.destroy()
            
        frame = self.input_frame
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=0) 
        frame.grid_rowconfigure(0, weight=0)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_rowconfigure(2, weight=0) # NEW: Row for pagination controls

        self.input_selection_dict = {}
        
        # Make sure we have a tracker for input files and pages
        if not hasattr(self, 'input_files'):
            self.input_files = []
            
        # Safely create the dictionary if it doesn't exist yet
        if not hasattr(self, 'current_pages'):
            self.current_pages = {}
            
        if "input" not in self.current_pages:
            self.current_pages["input"] = 0
            
        # Make sure ITEMS_PER_PAGE is set early too, just in case!
        if not hasattr(self, 'ITEMS_PER_PAGE'):
            self.ITEMS_PER_PAGE = 10

        # ==========================================
        # LEFT COLUMN: BROWSER AND LIST
        # ==========================================
        path_frame = ctk.CTkFrame(frame)
        path_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        path_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(path_frame, text="Current Folder:", anchor="w").grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))

        self.path_var = tk.StringVar(value=self.input_folder)
        self.path_entry = ctk.CTkEntry(path_frame, textvariable=self.path_var)
        self.path_entry.grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(10, 5))
        self.path_entry.bind("<Return>", getattr(self, "on_path_enter", lambda e: None))

        btn_frame = ctk.CTkFrame(path_frame, fg_color="transparent")
        btn_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=(0, 10))
        
        ctk.CTkButton(btn_frame, text="Back", command=getattr(self, "go_back", None), width=80, corner_radius=0).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Change / New Folder", command=getattr(self, "change_input_folder", None), corner_radius=0).pack(side="left", padx=5)
        ctk.CTkButton(btn_frame, text="Add Song", command=getattr(self, "add_song", None), corner_radius=0).pack(side="right", padx=5)
        
        self.songs_list_frame = ctk.CTkScrollableFrame(frame, corner_radius=0)
        self.songs_list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(25, 10))

        # --- NEW: INPUT PAGINATION CONTROLS ---
        self.input_page_frame = ctk.CTkFrame(frame, fg_color="transparent")
        self.input_page_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.input_btn_prev = ctk.CTkButton(self.input_page_frame, text="<", width=30, command=lambda: self.change_page("input", -1))
        self.input_btn_prev.pack(side="left", padx=5)

        self.input_lbl_page = ctk.CTkLabel(self.input_page_frame, text="Page 1 of 1")
        self.input_lbl_page.pack(side="left", padx=10, expand=True)

        self.input_btn_next = ctk.CTkButton(self.input_page_frame, text=">", width=30, command=lambda: self.change_page("input", 1))
        self.input_btn_next.pack(side="right", padx=5)

        # ==========================================
        # RIGHT COLUMN: SEPARATION MENU
        # ==========================================
        self.frame_width = 200

        sep_scrollable = ctk.CTkScrollableFrame(
            frame, corner_radius=0,
            width=self.frame_width, 
            fg_color=("gray90", "gray16") 
        )
        sep_scrollable.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=10, pady=10)
        sep_scrollable.propagate(False)
        
        # NEW: Force the inner column to stretch, allowing sticky="ew" to work perfectly
        sep_scrollable.grid_columnconfigure(0, weight=1)

        # Slightly reduced font size to ensure it fits
        ctk.CTkLabel(sep_scrollable, text="Separation Menu", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, pady=(10,20))

        # AI Tool selection (Reduced padx to 10)
        self.ai_tool_var = tk.StringVar(value="Spleeter")
        ctk.CTkRadioButton(sep_scrollable, text="Spleeter", variable=self.ai_tool_var, value="Spleeter", command=self.on_tool_change).grid(row=1, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkRadioButton(sep_scrollable, text="Demucs", variable=self.ai_tool_var, value="Demucs", command=self.on_tool_change).grid(row=2, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkRadioButton(sep_scrollable, text="OpenUnmix", variable=self.ai_tool_var, value="OpenUnmix", command=self.on_tool_change).grid(row=3, column=0, sticky="w", padx=10, pady=5)

        self.model_label = ctk.CTkLabel(sep_scrollable, text="Model:", anchor="w")
        self.model_label.grid(row=4, column=0, sticky="w", padx=10, pady=(10,0))
        
        # Removed hardcoded width, relying on sticky="ew"
        self.model_var = tk.StringVar(value="umxl")
        self.model_menu = ctk.CTkOptionMenu(sep_scrollable, variable=self.model_var, corner_radius=0, values=["umxl", "umxhq", "umx"])
        self.model_menu.grid(row=5, column=0, sticky="ew", padx=10, pady=5)

        ctk.CTkLabel(sep_scrollable, text="Output Format:", anchor="w").grid(row=6, column=0, sticky="w", padx=10, pady=(10,0))
        self.format_var = tk.StringVar(value="wav")
        self.format_menu = ctk.CTkOptionMenu(sep_scrollable, variable=self.format_var, corner_radius=0, values=["wav", "mp3", "flac"], command=self.on_format_change)
        self.format_menu.grid(row=7, column=0, sticky="ew", padx=10, pady=5)

        self.wav_flac_frame = ctk.CTkFrame(sep_scrollable)
        self.wav_flac_frame.grid(row=8, column=0, sticky="ew", padx=10, pady=5)
        self.wav_flac_frame.grid_remove() 

        self.channel_frame = ctk.CTkFrame(self.wav_flac_frame, fg_color="transparent")
        self.channel_frame.pack(fill="x", padx=5, pady=10)

        # Tightened the spacing on the Mono/Stereo toggle
        self.channel_var = tk.StringVar(value="Stereo")
        self.mono_label = ctk.CTkLabel(self.channel_frame, text="Mono", text_color="gray")
        self.mono_label.pack(side="left", padx=(0, 5))

        def on_channel_toggle():
            if self.channel_switch_var.get() == "Stereo":
                self.channel_var.set("Stereo")
                self.mono_label.configure(text_color="gray")
                self.stereo_label.configure(text_color=("black", "white"))
            else:
                self.channel_var.set("Mono")
                self.mono_label.configure(text_color=("black", "white"))
                self.stereo_label.configure(text_color="gray")

        self.channel_switch_var = ctk.StringVar(value="Stereo")
        self.channel_switch = ctk.CTkSwitch(self.channel_frame, text="", variable=self.channel_switch_var, onvalue="Stereo", offvalue="Mono", command=on_channel_toggle, width=35)
        self.channel_switch.pack(side="left")

        self.stereo_label = ctk.CTkLabel(self.channel_frame, text="Stereo", text_color=("black", "white"))
        self.stereo_label.pack(side="left", padx=(5, 0))

        ctk.CTkLabel(self.wav_flac_frame, text="Sample Rate (Hz):", anchor="w").pack(anchor="w", padx=10, pady=(10,0))
        self.sr_var = tk.StringVar(value="44100")
        self.sr_entry = ctk.CTkEntry(self.wav_flac_frame, textvariable=self.sr_var, placeholder_text="44100")
        self.sr_entry.pack(fill="x", padx=10, pady=5)

        self.bit_depth_frame = ctk.CTkFrame(self.wav_flac_frame)
        self.bit_depth_frame.pack(fill="x", padx=10, pady=5)
        self.bit_depth_frame.pack_forget()

        self.bit_depth_var = tk.BooleanVar(value=True)
        ctk.CTkRadioButton(self.bit_depth_frame, text="24-bit", variable=self.bit_depth_var, value=True).pack(anchor="w", padx=10, pady=5)
        ctk.CTkRadioButton(self.bit_depth_frame, text="Float32", variable=self.bit_depth_var, value=False).pack(anchor="w", padx=10, pady=5)

        self.mp3_frame = ctk.CTkFrame(sep_scrollable)
        self.mp3_frame.grid(row=9, column=0, sticky="ew", padx=10, pady=5)
        self.mp3_frame.grid_remove()

        self.bitrate_frame = ctk.CTkFrame(self.mp3_frame, fg_color="transparent")
        self.bitrate_frame.pack(fill="x", padx=10, pady=(5, 10))

        ctk.CTkLabel(self.bitrate_frame, text="Bitrate:").pack(side="left")
        self.bitrate_var = tk.StringVar(value="192")
        self.bitrate_entry = ctk.CTkEntry(self.bitrate_frame, textvariable=self.bitrate_var, width=60, placeholder_text="192")
        self.bitrate_entry.pack(side="right")

        self.mp3_preset_label = ctk.CTkLabel(self.mp3_frame, text="MP3 Preset (2=Best):", anchor="w")
        self.mp3_preset_label.pack(anchor="w", padx=10, pady=(10,0))
        
        self.mp3_preset_slider = ctk.CTkSlider(self.mp3_frame, from_=2, to=7, number_of_steps=5, command=getattr(self, "update_mp3_preset_label", None))
        self.mp3_preset_slider.set(2)
        self.mp3_preset_slider.pack(fill="x", padx=10, pady=5)
        
        self.mp3_preset_value_label = ctk.CTkLabel(self.mp3_frame, text="Current: 2", anchor="w")
        self.mp3_preset_value_label.pack(anchor="w", padx=10, pady=5)

        self.shifts_frame = ctk.CTkFrame(sep_scrollable)
        self.shifts_frame.grid(row=10, column=0, sticky="ew", padx=10, pady=5)
        self.shifts_frame.grid_remove()

        ctk.CTkLabel(self.shifts_frame, text="Shifts (qual/speed):", anchor="w").pack(anchor="w", padx=10, pady=5)
        self.shifts_var = tk.StringVar(value="1")
        ctk.CTkEntry(self.shifts_frame, textvariable=self.shifts_var, placeholder_text="1").pack(fill="x", padx=10, pady=5)

        # Removed hardcoded width here too
        self.separate_button = ctk.CTkButton(sep_scrollable, text="Start Batch Separation", height=40, corner_radius=0, font=ctk.CTkFont(weight="bold"), command=self.separate_audio)
        self.separate_button.grid(row=11, column=0, sticky="ew", padx=10, pady=(30,10))

        if hasattr(self, 'on_tool_change'): self.on_tool_change()

    def create_output_tab(self):
        for widget in self.output_frame.winfo_children():
            widget.destroy()

        frame = self.output_frame
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=0) 
        # We need more rows now: Header(0), List(1), Page(2), Header(3), List(4), Page(5)...
        frame.grid_rowconfigure((1, 4, 7), weight=1) 

        # --- NEW: PAGINATION TRACKERS ---
        self.ITEMS_PER_PAGE = 10
        self.current_pages = {"trans": 0, "vocals": 0, "instr": 0}

        # Dictionaries to track selection states for batch processing
        self.trans_selection_dict = {}
        self.vocals_selection_dict = {}
        self.instr_selection_dict = {}

        # ==========================================
        # LEFT COLUMN: OUTPUT LISTS & PAGINATION
        # ==========================================
        
        # --- 1. Transcriptions ---
        header_frame1 = ctk.CTkFrame(frame, fg_color="transparent")
        header_frame1.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        ctk.CTkLabel(header_frame1, text="Transcriptions", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
        ctk.CTkButton(header_frame1, text="Change Folder", width=100, corner_radius=0, command=lambda: self.change_output_folder("transcriptions")).pack(side="right")
        
        self.trans_list_frame = ctk.CTkScrollableFrame(frame, corner_radius=0, height=100)
        self.trans_list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 5))

        self.trans_page_frame = ctk.CTkFrame(frame, fg_color="transparent", height=30)
        self.trans_page_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 15))
        self.trans_btn_prev = ctk.CTkButton(self.trans_page_frame, text="<", width=30, corner_radius=0, command=lambda: self.change_page("trans", -1))
        self.trans_btn_prev.pack(side="left")
        self.trans_lbl_page = ctk.CTkLabel(self.trans_page_frame, text="Page 1 of 1", font=ctk.CTkFont(weight="bold"))
        self.trans_lbl_page.pack(side="left", expand=True)
        self.trans_btn_next = ctk.CTkButton(self.trans_page_frame, text=">", width=30, corner_radius=0, command=lambda: self.change_page("trans", 1))
        self.trans_btn_next.pack(side="right")

        # --- 2. Vocals ---
        header_frame2 = ctk.CTkFrame(frame, fg_color="transparent")
        header_frame2.grid(row=3, column=0, sticky="ew", padx=10)
        ctk.CTkLabel(header_frame2, text="Vocals", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
        ctk.CTkButton(header_frame2, text="Change Folder", width=100, corner_radius=0, command=lambda: self.change_output_folder("vocals")).pack(side="right")
        
        self.vocals_list_frame = ctk.CTkScrollableFrame(frame, corner_radius=0, height=100)        
        self.vocals_list_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=(5, 5))

        self.vocals_page_frame = ctk.CTkFrame(frame, fg_color="transparent", height=30)
        self.vocals_page_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 15))
        self.vocals_btn_prev = ctk.CTkButton(self.vocals_page_frame, text="<", width=30, corner_radius=0, command=lambda: self.change_page("vocals", -1))
        self.vocals_btn_prev.pack(side="left")
        self.vocals_lbl_page = ctk.CTkLabel(self.vocals_page_frame, text="Page 1 of 1", font=ctk.CTkFont(weight="bold"))
        self.vocals_lbl_page.pack(side="left", expand=True)
        self.vocals_btn_next = ctk.CTkButton(self.vocals_page_frame, text=">", width=30, corner_radius=0, command=lambda: self.change_page("vocals", 1))
        self.vocals_btn_next.pack(side="right")

        # --- 3. Instrumentals ---
        header_frame3 = ctk.CTkFrame(frame, fg_color="transparent")
        header_frame3.grid(row=6, column=0, sticky="ew", padx=10)
        ctk.CTkLabel(header_frame3, text="Instrumentals", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
        ctk.CTkButton(header_frame3, text="Change Folder", width=100, corner_radius=0, command=lambda: self.change_output_folder("instrumentals")).pack(side="right")
        
        self.instr_list_frame = ctk.CTkScrollableFrame(frame, corner_radius=0, height=100)
        self.instr_list_frame.grid(row=7, column=0, sticky="nsew", padx=10, pady=(5, 5))

        self.instr_page_frame = ctk.CTkFrame(frame, fg_color="transparent", height=30)
        self.instr_page_frame.grid(row=8, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.instr_btn_prev = ctk.CTkButton(self.instr_page_frame, text="<", width=30, corner_radius=0, command=lambda: self.change_page("instr", -1))
        self.instr_btn_prev.pack(side="left")
        self.instr_lbl_page = ctk.CTkLabel(self.instr_page_frame, text="Page 1 of 1", font=ctk.CTkFont(weight="bold"))
        self.instr_lbl_page.pack(side="left", expand=True)
        self.instr_btn_next = ctk.CTkButton(self.instr_page_frame, text=">", width=30, corner_radius=0, command=lambda: self.change_page("instr", 1))
        self.instr_btn_next.pack(side="right")

        # ==========================================
        # RIGHT COLUMN: TRANSCRIPTION MENU
        # ==========================================
        # Adjusted rowspan to 9 to match the new left column layout
        trans_menu = ctk.CTkScrollableFrame(frame, width=self.frame_width, corner_radius=0, fg_color=("gray90", "gray16"))
        trans_menu.grid(row=0, column=1, rowspan=9, sticky="nsew", padx=10, pady=10) 
        trans_menu.grid_columnconfigure(0, weight=1)
        trans_menu.grid_columnconfigure(1, weight=0)
        trans_menu.propagate(False)

        # Reduced font size slightly to match the Separation Menu
        ctk.CTkLabel(trans_menu, text="Transcription Menu", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, pady=(10, 20))
        
        ctk.CTkLabel(trans_menu, text="Tool:", anchor="w").grid(row=1, column=0, sticky="w", padx=10)
        self.trans_tool_var = tk.StringVar(value="whisper")
        
        ctk.CTkRadioButton(trans_menu, text="Whisper", variable=self.trans_tool_var, value="whisper", command=getattr(self, "on_trans_tool_change", None)).grid(row=2, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkRadioButton(trans_menu, text="Wav2Vec2", variable=self.trans_tool_var, value="wav2vec2", command=getattr(self, "on_trans_tool_change", None)).grid(row=3, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkRadioButton(trans_menu, text="Vosk", variable=self.trans_tool_var, value="vosk", command=getattr(self, "on_trans_tool_change", None)).grid(row=4, column=0, sticky="w", padx=10, pady=5)

        self.trans_model_label = ctk.CTkLabel(trans_menu, text="Model:", anchor="w")
        self.trans_model_label.grid(row=5, column=0, sticky="w", padx=10, pady=(10, 0))
        
        self.trans_model_var = tk.StringVar()
        # Removed hardcoded width, relying on sticky="ew"
        self.trans_model_menu = ctk.CTkOptionMenu(trans_menu, variable=self.trans_model_var, corner_radius=0, values=[])
        self.trans_model_menu.grid(row=6, column=0, sticky="ew", padx=10, pady=5)

        self.trans_lang_label = ctk.CTkLabel(trans_menu, text="Language:", anchor="w")
        self.trans_lang_label.grid(row=7, column=0, sticky="w", padx=10, pady=(10, 0))
        
        self.trans_lang_var = tk.StringVar(value="auto")
        # Removed hardcoded width, relying on sticky="ew"
        self.trans_lang_menu = ctk.CTkOptionMenu(trans_menu, variable=self.trans_lang_var, corner_radius=0, values=["auto", "cs", "en", "fr", "de", "es"])
        self.trans_lang_menu.grid(row=8, column=0, sticky="ew", padx=10, pady=5)

        self.use_spk_id_var = tk.BooleanVar(value=False)
        self.spk_toggle = ctk.CTkSwitch(trans_menu, text="Identify Speakers", variable=self.use_spk_id_var, progress_color="#1f538d")
        self.spk_toggle.grid(row=9, column=0, sticky="w", padx=10, pady=10)
        self.spk_toggle.grid_remove() 
        
        # Removed hardcoded width, relying on sticky="ew"
        self.trans_button = ctk.CTkButton(trans_menu, text="Transcribe", height=40, font=ctk.CTkFont(weight="bold"), command=getattr(self, "run_standalone_transcription", None), corner_radius=0)
        self.trans_button.grid(row=10, column=0, sticky="ew", padx=10, pady=(30, 10))
        
        ctk.CTkLabel(trans_menu, text="Turn ON switches in 'Vocals' to process.", font=ctk.CTkFont(size=11, slant="italic")).grid(row=11, column=0, padx=10)

        if hasattr(self, 'on_trans_tool_change'): self.on_trans_tool_change()

    def create_file_row(self, parent_frame, file_name, file_path, item_idx, selection_dict, is_folder=False):
        # --- NEW: Determine if the file is a text document ---
        is_txt = file_name.lower().endswith('.txt')

        # 1. Flat base for maximum performance
        row_frame = ctk.CTkFrame(parent_frame, corner_radius=0, fg_color=("gray85", "gray20"))
        row_frame.pack(fill="x", padx=2, pady=2)
        
        # 2. Selection checkbox (Far left)
        # --- NEW: We only create and show the checkbox if it is NOT a .txt file ---
        if not is_txt:
            if item_idx not in selection_dict:
                var = tk.BooleanVar(value=False)
                item_type = 'folder' if is_folder else 'song'
                item_data = {'name': file_name, 'path': file_path} 
                
                selection_dict[item_idx] = {
                    "var": var,
                    "type": item_type,
                    "data": item_data
                }
                
            chk = ctk.CTkCheckBox(row_frame, text="", variable=selection_dict[item_idx]["var"], width=24, corner_radius=0)
            chk.pack(side="left", padx=(5, 5))
        else:
            # --- NEW: Invisible spacer ---
            # We pack an empty label of the exact same width (24) so the Book buttons 
            # stay perfectly aligned with the Play buttons in mixed folders.
            spacer = ctk.CTkLabel(row_frame, text="", width=24)
            spacer.pack(side="left", padx=(5, 5))
            
        # 3. Play / Open buttons (Right next to the checkbox)
        if is_folder:
            open_btn = ctk.CTkButton(row_frame, text="Open", width=50, corner_radius=0, 
                                     command=lambda p=file_path: self.enter_folder(p))
            open_btn.pack(side="left", padx=(0, 5))
        elif is_txt:
            # --- NEW: Book button for text files ---
            read_btn = ctk.CTkButton(row_frame, text="📖", width=30, corner_radius=0,
                                     command=lambda p=file_path: self.play_audio(p))
            read_btn.pack(side="left", padx=(0, 5))
        else:
            play_btn = ctk.CTkButton(row_frame, text="▶", width=30, corner_radius=0,
                                     command=lambda p=file_path: self.play_audio(p))
            play_btn.pack(side="left", padx=(0, 5))

        # 4. File icon and name (Middle, expands and pushes the rest to the right)
        # --- NEW: Different icons for different file types ---
        if is_folder:
            display_text = f"📁 {file_name}"
        elif is_txt:
            display_text = f"📝 {file_name}"
        else:
            display_text = f"🎵 {file_name}"
            
        lbl = ctk.CTkLabel(row_frame, text=display_text, anchor="w")
        lbl.pack(side="left", fill="x", expand=True, padx=5) 
        
        # 5. Delete button (Stays on the far right)
        if not is_folder:
            del_btn = ctk.CTkButton(row_frame, text="🗑", width=30, corner_radius=0, fg_color="#a83232", hover_color="#8a2929",
                                    command=lambda p=file_path: self.delete_file(p))
            del_btn.pack(side="right", padx=5)

    def enter_folder(self, folder_path):
        """Enters the selected subfolder and redraws the UI."""
        # 1. Change the currently searched folder to the new one
        self.input_folder = folder_path
        
        # 2. Clear old data so the app doesn't crash with an IndexError (as we fixed previously)
        self.all_items.clear()
        self.input_selection_dict.clear()
        
        # 3. Call the function that renders the input page again.
        # It will now read the new value of self.input_folder and show its contents.
        self.load_input()

    def play_audio(self, path):
        """Plays the song in the system's default media player (Groove Music, VLC, etc.)."""
        try:
            abs_path = os.path.abspath(path)
            
            if platform.system() == "Windows":
                os.startfile(abs_path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", abs_path])
            else:
                subprocess.Popen(["xdg-open", abs_path])
        except Exception as e:
            # Translated the error message as well for consistency
            print(f"Error playing file: {e}")

    def delete_file(self, file_path, tab_to_reload="input"):
        """Deletes a file directly from the app and refreshes the UI."""
        file_name = os.path.basename(file_path)
        if messagebox.askyesno("Delete File", f"Are you sure you want to permanently delete:\n{file_name}?"):
            try:
                os.remove(file_path)
                # Refresh the correct tab so the deleted file disappears from the list
                if tab_to_reload == "input":
                    self.load_input()
                else:
                    self.load_outputs()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete file: {e}")

    def create_settings_tab(self):
        """!
        @brief Populates the Settings tab with directory configurations, a Model Manager, 
               and system resource controls.
        
        This redesign removes manual CSV typing in favor of one-click downloads 
        and clear status indications, providing a premium user experience.
        """
        # 1. Clear existing widgets
        for widget in self.settings_frame.winfo_children():
            widget.destroy()

        # 2. Main Scrollable Container
        scroll_frame = ctk.CTkScrollableFrame(self.settings_frame, corner_radius=0)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # ==========================================
        # SECTION 1: DIRECTORY SETTINGS
        # ==========================================
        dir_frame = ctk.CTkFrame(scroll_frame)
        dir_frame.pack(fill="x", padx=10, pady=(0, 20))
        
        ctk.CTkLabel(dir_frame, text="Folder Directories", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=15, pady=(15, 10))

        # Setup StringVars (PŘIDÁNA PROMĚNNÁ PRO MODELY)
        self.settings_models_var = tk.StringVar(value=self.models_dir)
        self.settings_input_var = tk.StringVar(value=self.input_folder)
        self.settings_vocals_var = tk.StringVar(value=self.output_folders["vocals"])
        self.settings_instr_var = tk.StringVar(value=self.output_folders["instrumentals"])
        self.settings_trans_var = tk.StringVar(value=self.output_folders["transcriptions"])

        # PŘIDÁNO DO SEZNAMU (hned na první místo)
        folders = [
            ("AI Models Folder:", self.settings_models_var),
            ("Input Folder:", self.settings_input_var),
            ("Vocals Folder:", self.settings_vocals_var),
            ("Instrumentals:", self.settings_instr_var),
            ("Transcriptions:", self.settings_trans_var)
        ]

        # Uniform layout loop for folders
        for text, var in folders:
            row_frame = ctk.CTkFrame(dir_frame, fg_color="transparent")
            row_frame.pack(fill="x", padx=15, pady=2)
            
            ctk.CTkLabel(row_frame, text=text, width=130, anchor="w").pack(side="left")
            ctk.CTkEntry(row_frame, textvariable=var).pack(side="left", fill="x", expand=True)
            ctk.CTkButton(row_frame, text="Browse", width=70, corner_radius=0,
                          command=lambda v=var: self._browse_folder(v)).pack(side="right", padx=(10, 0))

        # ==========================================
        # SECTION 2: AI MODELS (CSV Format)
        # ==========================================
        mod_frame = ctk.CTkFrame(scroll_frame)
        mod_frame.pack(fill="x", padx=10, pady=(0, 20))
        
        ctk.CTkLabel(mod_frame, text="AI Models List", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=15, pady=(15, 5))
        ctk.CTkLabel(mod_frame, text="Edit the lists below. The app will attempt to use these models when selected.", 
                     text_color="gray", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=15, pady=(0, 10))

        self.model_vars = {}
        
        def list_to_csv(model_list):
            return ", ".join(model_list)

        # Updated descriptions for better UX
        models_setup = [
            ("Demucs:", "demucs", self.separator_models.get("Demucs", []), "High quality separation (Downloads automatically on first use)"),
            ("OpenUnmix:", "openunmix", self.separator_models.get("OpenUnmix", []), "Alternative separation models"),
            ("Whisper:", "whisper", self.transcription_models.get("whisper", []), "Accurate transcription (Downloads automatically on first use)"),
            ("Wav2Vec2:", "wav2vec2", self.transcription_models.get("wav2vec2", []), "Fast transcription"),
            ("Vosk:", "vosk", self.transcription_models.get("vosk", []), "Fast offline transcription (Requires manual download or scanning)")
        ]

        for text, dict_key, model_list, desc in models_setup:
            # Row container
            row_frame = ctk.CTkFrame(mod_frame, fg_color="transparent")
            row_frame.pack(fill="x", padx=15, pady=5)
            
            self.model_vars[dict_key] = tk.StringVar(value=list_to_csv(model_list))
            
            # Label and Entry
            input_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
            input_frame.pack(side="top", fill="x")
            
            ctk.CTkLabel(input_frame, text=text, width=100, anchor="w").pack(side="left")
            ctk.CTkEntry(input_frame, textvariable=self.model_vars[dict_key]).pack(side="left", fill="x", expand=True, padx=(5, 5))
            
            # Description text below the entry
            ctk.CTkLabel(row_frame, text=desc, text_color="gray", font=ctk.CTkFont(size=10)).pack(side="top", anchor="w", padx=(105, 0))

        # Action Buttons for Models
        action_mod_frame = ctk.CTkFrame(mod_frame, fg_color="transparent")
        action_mod_frame.pack(fill="x", padx=15, pady=15)

        ctk.CTkButton(action_mod_frame, text="Download Default Models", width=180,
                      command=self.download_default_models).pack(side="left", padx=(0, 10))
                      
        ctk.CTkButton(action_mod_frame, text="Scan Directory", width=120, fg_color="gray40", hover_color="gray30",
                      command=self.scan_models_directory).pack(side="left", padx=(0, 10))

        # ==========================================
        # SECTION 3: SYSTEM ACTIONS & MEMORY
        # ==========================================
        action_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        action_frame.pack(fill="x", padx=10, pady=(10, 20))
        
        # Top Row of Action Frame: Memory Management
        memory_row = ctk.CTkFrame(action_frame, fg_color="transparent")
        memory_row.pack(fill="x", pady=(0, 15))
        
        self.auto_flush_memory = getattr(self, 'auto_flush_memory', True)
        self.flush_switch = ctk.CTkSwitch(memory_row, text="Auto-Flush RAM after processing", command=self._toggle_auto_flush)
        if self.auto_flush_memory: self.flush_switch.select()
        self.flush_switch.pack(side="left", padx=(5, 15))
        
        ctk.CTkButton(memory_row, text="Force Flush Now", corner_radius=0, fg_color="#8B0000", hover_color="#5C0000", 
                      command=self._manual_flush).pack(side="left", padx=5)

        ctk.CTkLabel(action_frame, text="Compute Device:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)

        self.device_var = ctk.StringVar(value=getattr(self, 'device_var', ctk.StringVar(value="Auto")).get())
        self.device_dropdown = ctk.CTkOptionMenu(
            action_frame, values=["Auto", "GPU", "CPU"], variable=self.device_var
        )
        self.device_dropdown.pack(side="left", padx=5)

        # Bottom Row of Action Frame: Save/Restore Buttons
        button_row = ctk.CTkFrame(action_frame, fg_color="transparent")
        button_row.pack(fill="x", pady=(15, 0))
        
        ctk.CTkButton(button_row, text="Save Settings", height=40, font=ctk.CTkFont(weight="bold"), corner_radius=0, 
                      command=self.save_settings_changes).pack(side="right", padx=(10, 0))
        
        ctk.CTkButton(button_row, text="Restore Defaults", height=40, fg_color="transparent", corner_radius=0, 
                      border_width=1, text_color=("gray10", "gray90"), 
                      command=self.restore_defaults).pack(side="right", padx=5)
        
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

    def _apply_dict_to_state(self, data):
        """Maps dictionary values from settings to the app's internal variables."""
        
        # 1. Directories
        self.input_folder = data.get("input_folder", "input")
        
        # Combine the output paths into the dictionary your app expects
        self.output_folders = {
            "vocals": data.get("vocals_folder", "output/vocals"),
            "instrumentals": data.get("instrumentals_folder", "output/instrumentals"),
            "transcriptions": data.get("transcriptions_folder", "output/text")
        }
        
        # 2. UI Preferences
        self.appearance_mode = data.get("appearance_mode", "Dark")
        self.color_theme = data.get("color_theme", "blue")
        self.scaling = data.get("scaling", "100%")
        
        # 3. System Preferences
        self.auto_flush_memory = data.get("auto_flush_memory", True)
        
        # 4. AI Models
        self.separator_models = data.get("separator_models", {})
        self.transcription_models = data.get("transcription_models", {})

    def restore_defaults(self):
        """Wipes current settings back to factory state."""
        if messagebox.askyesno("Restore Defaults", "Are you sure you want to reset all settings and paths?"):
            self._apply_dict_to_state(self.DEFAULT_SETTINGS)
            self.save_settings()
            # Rebuild the tab to visually show the reset
            self.create_settings_tab()
            messagebox.showinfo("Success", "Settings restored to defaults. Please restart to apply theme changes.")

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
        
        messagebox.showinfo("Saved", "Settings successfully updated!")

    def save_settings(self):
        """Writes current state to disk."""
        data = {
            "input_folder": self.input_folder,
            "vocals_folder": self.output_folders["vocals"],
            "instrumentals_folder": self.output_folders["instrumentals"],
            "transcriptions_folder": self.output_folders["transcriptions"],
            "appearance_mode": getattr(self, "appearance_mode", "Dark"),
            "color_theme": getattr(self, "color_theme", "blue"), # <-- NEW: Saves the color theme
            "scaling": getattr(self, "scaling", "100%"),
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
            "color_theme": "blue", # <-- NEW: Default color theme fallback
            "scaling": "100%",
            "auto_flush_memory": True,
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
                    "vosk-model-small-cs-0.4-rhasspy",
                    "vosk-model-small-fr-0.22",
                    "vosk-model-small-fr-pguyot-0.3",
                    "vosk-model-small-en-us-0.15",
                    "vosk-model-en-us-0.22-lgraph"
                ]
            }
        }
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

    def update_mp3_preset_label(self, value):
        self.mp3_preset_value_label.configure(text=f"Current: {int(value)}")

    def on_tool_change(self, *args):
        # 1. SAFETY CHECK: Skip visual updates if the UI tab hasn't been drawn yet
        if not (hasattr(self, 'model_label') and self.model_label.winfo_exists()):
            return

        tool = self.ai_tool_var.get()
        try:
            if tool == "Spleeter":
                self.model_label.grid_remove()
                self.model_menu.grid_remove()  # Hide model selection for Spleeter
            else:
                # Handle both Demucs and OpenUnmix dynamically
                default_models = ["mdx", "mdx_extra", "htdemucs"] if tool == "Demucs" else ["umxl", "umxhq", "umx"]
                values = self.separator_models.get(tool, default_models)
                if not values: values = default_models # Failsafe if JSON list is empty
                
                self.model_menu.configure(values=values)
                self.model_var.set(values[0])
                self.model_label.grid()
                self.model_menu.grid()
        except Exception as e:
            import logging
            logging.error(f"Error updating separator models: {e}", exc_info=True)

        self.on_format_change()  # Refresh format options depending on the new tool

    def on_format_change(self, *args):
        # 1. SAFETY CHECK: Skip visual updates if the UI tab hasn't been drawn yet
        if not (hasattr(self, 'wav_flac_frame') and self.wav_flac_frame.winfo_exists()):
            return

        fmt = self.format_var.get()
        tool = self.ai_tool_var.get()

        # 1. Main Format Frames (Using Grid)
        if fmt in ["wav", "flac"]:
            self.wav_flac_frame.grid() 
            if hasattr(self, 'mp3_frame'): self.mp3_frame.grid_remove() 
        elif fmt == "mp3":
            self.wav_flac_frame.grid_remove() 
            if hasattr(self, 'mp3_frame'): self.mp3_frame.grid() 

        # 2. Bit Depth (Inside wav_flac_frame, using Pack)
        if tool == "Demucs" and fmt == "wav":
            if hasattr(self, 'bit_depth_frame'): self.bit_depth_frame.pack(fill="x", padx=20, pady=5)
        else:
            if hasattr(self, 'bit_depth_frame'): self.bit_depth_frame.pack_forget()

        # 3. MP3 Preset Sliders (Inside mp3_frame, using Pack)
        if tool == "Demucs" and fmt == "mp3":
            if hasattr(self, 'mp3_preset_label'): self.mp3_preset_label.pack(fill="x", padx=20, pady=5)
            if hasattr(self, 'mp3_preset_slider'): self.mp3_preset_slider.pack(fill="x", padx=20, pady=5)
            if hasattr(self, 'mp3_preset_value_label'): self.mp3_preset_value_label.pack(anchor="w", padx=20, pady=5)
        else:
            if hasattr(self, 'mp3_preset_label'): self.mp3_preset_label.pack_forget()
            if hasattr(self, 'mp3_preset_slider'): self.mp3_preset_slider.pack_forget()
            if hasattr(self, 'mp3_preset_value_label'): self.mp3_preset_value_label.pack_forget()

        # 4. Shifts (Main scrollable frame, using Grid)
        if tool == "Demucs":
            if hasattr(self, 'shifts_frame'): self.shifts_frame.grid() 
        else:
            if hasattr(self, 'shifts_frame'): self.shifts_frame.grid_remove()
    
    def on_trans_tool_change(self, *args):
        """
        Complete logic for UI visibility and model population.
        Handles: model lists, language dropdown visibility, and speaker ID toggle.
        """
        # 1. SAFETY CHECK: Skip visual updates if the UI tab hasn't been drawn yet
        if not (hasattr(self, 'trans_model_label') and self.trans_model_label.winfo_exists()):
            return

        tool = self.trans_tool_var.get()
            
        try:
            # 1. RETRIEVE DATA FROM SETTINGS
            all_models = self.transcription_models.get(tool, [])
            
            # 2. MODEL SELECTION LOGIC
            if tool in ["whisper", "wav2vec2", "vosk"]:
                # Filter logic for Vosk (don't show the spk model in the dropdown)
                if tool == "vosk":
                    values = [m for m in all_models if "spk" not in m]
                    if not values: values = ["vosk-model-small-cs-0.4-rhasspy"]
                else:
                    values = all_models if all_models else ["base"]

                # Update the dropdown menu values
                if hasattr(self, 'trans_model_menu'):
                    self.trans_model_menu.configure(values=values)
                    self.trans_model_var.set(values[0] if values else "")

                self.trans_model_label.grid()
                self.trans_model_menu.grid()
            else:
                self.trans_model_label.grid_remove()
                self.trans_model_menu.grid_remove()

            # 3. LANGUAGE SELECTION VISIBILITY
            if tool == "whisper":
                if hasattr(self, 'trans_lang_label'): self.trans_lang_label.grid()
                if hasattr(self, 'trans_lang_menu'): self.trans_lang_menu.grid()
            else:   
                if hasattr(self, 'trans_lang_label'): self.trans_lang_label.grid_remove()
                if hasattr(self, 'trans_lang_menu'): self.trans_lang_menu.grid_remove()

            # 4. SPEAKER ID (DIARIZATION) VISIBILITY
            if tool == "vosk":
                if hasattr(self, 'spk_toggle'): self.spk_toggle.grid(row=9, column=0, sticky="w", padx=20, pady=10)
            else:
                if hasattr(self, 'spk_toggle'): self.spk_toggle.grid_remove()
                    
        except Exception as e:
            import logging
            logging.error(f"Error in on_trans_tool_change: {e}")
            
    def load_input(self):
        # 1. Clear old data
        self.folders.clear()
        self.songs.clear()
        self.input_selection_dict.clear() # Clear the batch selection tracker
        
        # Create a unified list for the pagination renderer
        self.input_files = [] 

        if not os.path.isdir(self.input_folder):
            return

        # 2. Scan directory
        items = sorted(os.listdir(self.input_folder))
        
        # 3. Populate unified lists
        for item in items:
            full_path = os.path.join(self.input_folder, item)
            
            if os.path.isdir(full_path):
                self.folders.append(full_path)
                self.input_files.append({
                    'name': item, 
                    'path': full_path, 
                    'is_folder': True
                })
            elif item.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                self.songs.append({'path': full_path, 'name': item})
                self.input_files.append({
                    'name': item, 
                    'path': full_path, 
                    'is_folder': False
                })

        # Update the path bar at the top of the tab
        self.path_var.set(self.input_folder)

        # 4. Reset page trackers
        if not hasattr(self, 'current_pages'):
            self.current_pages = {}
        self.current_pages["input"] = 0
        
        # 5. Let the unified render_page function handle all the UI drawing
        self.render_page("input")

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
        # 1. Clear previous UI selection states
        self.vocals_selection_dict.clear()
        self.instr_selection_dict.clear()
        self.trans_selection_dict.clear()

        # 2. Clear old data and read the hard drive
        self.vocals.clear()
        self.instrumentals.clear()
        self.transcriptions.clear()

        vocals_dir = self.output_folders.get("vocals", "")
        if os.path.isdir(vocals_dir):
            for f in sorted(os.listdir(vocals_dir)):
                if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                    self.vocals.append({'path': os.path.join(vocals_dir, f), 'name': f})

        instr_dir = self.output_folders.get("instrumentals", "")
        if os.path.isdir(instr_dir):
            for f in sorted(os.listdir(instr_dir)):
                if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                    self.instrumentals.append({'path': os.path.join(instr_dir, f), 'name': f})

        trans_dir = self.output_folders.get("transcriptions", "")
        if os.path.isdir(trans_dir):
            for f in sorted(os.listdir(trans_dir)):
                if f.lower().endswith(('.txt', '.lrc')):
                    self.transcriptions.append({'path': os.path.join(trans_dir, f), 'name': f})

        # 3. Reset page trackers
        self.current_pages = {"trans": 0, "vocals": 0, "instr": 0}
        
        # 4. Stagger the UI drawing
        self.after(10, lambda: self.render_page("trans"))
        self.after(50, lambda: self.render_page("vocals"))
        self.after(100, lambda: self.render_page("instr"))

    def change_page(self, category: str, delta: int):
        """Triggered by the < and > buttons to change the page number."""
        self.current_pages[category] += delta
        self.render_page(category)

    def render_page(self, category: str):
        if category == "vocals":
            data_list = self.vocals
            list_frame = self.vocals_list_frame
            sel_dict = self.vocals_selection_dict
            lbl_page = self.vocals_lbl_page
            btn_prev = self.vocals_btn_prev
            btn_next = self.vocals_btn_next
        elif category == "instr":
            data_list = self.instrumentals
            list_frame = self.instr_list_frame
            sel_dict = self.instr_selection_dict
            lbl_page = self.instr_lbl_page
            btn_prev = self.instr_btn_prev
            btn_next = self.instr_btn_next
        elif category == "trans":
            data_list = self.transcriptions
            list_frame = self.trans_list_frame
            sel_dict = self.trans_selection_dict
            lbl_page = self.trans_lbl_page
            btn_prev = self.trans_btn_prev
            btn_next = self.trans_btn_next
        elif category == "input": 
            data_list = self.input_files 
            list_frame = self.songs_list_frame
            sel_dict = self.input_selection_dict
            lbl_page = self.input_lbl_page
            btn_prev = self.input_btn_prev
            btn_next = self.input_btn_next
        else:
            return

        # Destroy old widgets
        for widget in list_frame.winfo_children():
            widget.destroy()

        # Math to determine pages
        total_items = len(data_list)
        total_pages = max(1, (total_items + getattr(self, "ITEMS_PER_PAGE", 10) - 1) // getattr(self, "ITEMS_PER_PAGE", 10))
        
        # Safely get current page
        if not hasattr(self, 'current_pages'): self.current_pages = {}
        curr_page = max(0, min(self.current_pages.get(category, 0), total_pages - 1))
        self.current_pages[category] = curr_page

        start_idx = curr_page * getattr(self, "ITEMS_PER_PAGE", 10)
        end_idx = min(start_idx + getattr(self, "ITEMS_PER_PAGE", 10), total_items)

        # Draw the specific rows
        for i in range(start_idx, end_idx):
            item = data_list[i]
            self.create_file_row(
                parent_frame=list_frame, 
                file_name=item['name'], 
                file_path=item['path'], 
                item_idx=i, 
                selection_dict=sel_dict,
                is_folder=item.get('is_folder', False) # Safely pass is_folder for inputs
            )

        # Update button states
        lbl_page.configure(text=f"Page {curr_page + 1} of {total_pages}")
        btn_prev.configure(state="normal" if curr_page > 0 else "disabled")
        btn_next.configure(state="normal" if curr_page < total_pages - 1 else "disabled")

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

    def check_models_directory(self):
        """!
        @brief Checks for the AI models directory on startup.
        
        Reads 'settings.json'. If missing, prompts the user to select or 
        auto-create a directory. Offers to download default models.
        """
        self.models_dir = ""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                    self.models_dir = settings.get("models_dir", "")
            except json.JSONDecodeError:
                pass # Corrupted or empty file
                
        if not self.models_dir or not os.path.exists(self.models_dir):
            dialog = ctk.CTkToplevel(self)
            dialog.title("Initial Setup")
            dialog.geometry("550x350") # Made slightly taller to fit 3 buttons
            dialog.attributes("-topmost", True)
            dialog.grab_set()

            ctk.CTkLabel(dialog, text="Welcome! Where would you like to store AI models?", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(20, 5))
            ctk.CTkLabel(dialog, text="Models can take up several gigabytes of space. Choose a suitable location.", text_color="gray").pack(pady=(0, 20))

            download_var = ctk.BooleanVar(value=True)

            def set_folder_and_close(selected_path):
                self.models_dir = selected_path
                os.makedirs(self.models_dir, exist_ok=True)
                
                # Save safely
                curr_settings = {}
                if os.path.exists(self.settings_file):
                    with open(self.settings_file, "r") as f:
                        try: curr_settings = json.load(f)
                        except json.JSONDecodeError: pass
                            
                curr_settings["models_dir"] = self.models_dir
                with open(self.settings_file, "w") as f:
                    json.dump(curr_settings, f, indent=4)
                
                os.environ["TORCH_HOME"] = self.models_dir
                os.environ["HF_HOME"] = self.models_dir
                os.environ["MODEL_PATH"] = self.models_dir

                dialog.destroy()
                
                # wait for the progress bar to be initialized
                if download_var.get():
                    self.after(500, self.download_default_models)

            # --- NEW: App Folder Logic ---
            def use_app_folder():
                # Now it perfectly targets the .exe folder every time
                local_path = os.path.join(get_app_dir(), "Models")
                set_folder_and_close(local_path)

            def use_auto_folder():
                auto_path = os.path.join(os.path.expanduser("~"), "Documents", "AudioSeparatorModels")
                set_folder_and_close(auto_path)

            def select_folder():
                folder = filedialog.askdirectory(title="Select AI Models Directory")
                if folder:
                    set_folder_and_close(folder)

            btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            btn_frame.pack(fill="x", padx=30, pady=10)

            # --- UPDATED: 3 Stacked Buttons ---
            ctk.CTkButton(btn_frame, text="Current App Folder (Next to .exe)", height=40, command=use_app_folder).pack(fill="x", pady=(0, 10))
            ctk.CTkButton(btn_frame, text="Automatic (Documents)", height=40, command=use_auto_folder).pack(fill="x", pady=(0, 10))
            ctk.CTkButton(btn_frame, text="Select Custom Folder", height=40, fg_color="gray40", hover_color="gray30", command=select_folder).pack(fill="x")

            ctk.CTkCheckBox(dialog, text="Download default models after setup", variable=download_var).pack(pady=20)

            self.wait_window(dialog)
        else:
            os.environ["TORCH_HOME"] = self.models_dir
            os.environ["HF_HOME"] = self.models_dir
            os.environ["MODEL_PATH"] = self.models_dir

    def download_default_models(self):
        """!
        @brief Unified function to download essential default models in the background.
        Includes a status tracker and spawns a summary popup when finished.
        """
        self.progress_bar.set(0)
        self.progress_bar.grid()
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start()

        # ==========================================
        # SAFE ZONE (Main Thread)
        # ==========================================
        demucs_csv = ""
        if hasattr(self, 'model_vars') and "demucs" in self.model_vars:
            demucs_csv = self.model_vars["demucs"].get()
        elif hasattr(self, 'separator_models'):
            demucs_csv = ", ".join(self.separator_models.get("Demucs", ["htdemucs"]))
            
        demucs_models = [m.strip() for m in demucs_csv.split(",") if m.strip()]

        default_vosk = "vosk-model-small-en-us-0.15"
        vosk_dir = os.path.join(self.models_dir, "vosk")
        vosk_path = os.path.join(vosk_dir, default_vosk)
        vosk_needs_download = not os.path.exists(vosk_path)

        def download_thread():
            # Dictionary to track the status of EVERY tool
            status_report = {
                "Spleeter": "Pending ⏭️",
                "Demucs": "Skipped ⏳",
                "OpenUnmix": "Pending ⏳",
                "Whisper": "Pending ⏳",
                "Wav2Vec2": "Pending ⏳",
                "Vosk": "Pending ⏳"
            }

            # ==========================================
            # DANGEROUS ZONE (Background Thread)
            # ==========================================
            
            # 1. Spleeter
            try:
                self.after(0, lambda: self.progress_text.configure(text="Verifying Spleeter model..."))
                
                # FORCE Spleeter to use your specific Models folder
                os.environ["MODEL_PATH"] = self.models_dir
                
                # When MODEL_PATH is set, Spleeter saves it as Models/2stems 
                # (instead of the default pretrained_models/2stems)
                spleeter_check = os.path.join(self.models_dir, "2stems")
                
                if os.path.exists(spleeter_check):
                    status_report["Spleeter"] = "Found 🔍"
                else:
                    status_report["Spleeter"] = "Downloaded ✅"
                    
                from spleeter.separator import Separator
                # This will now download into your Models folder if missing, or skip if found
                Separator('spleeter:2stems')
                
            except ImportError:
                status_report["Spleeter"] = "Not Installed ❌"
            except Exception as e:
                status_report["Spleeter"] = "Error ❌"
                print(f"Spleeter Error: {e}")

            # 2. Demucs
            if demucs_models:
                try:
                    import demucs.pretrained
                    # Demucs saves .th files with hashes in the checkpoints folder
                    checkpoints_dir = os.path.join(self.models_dir, "hub", "checkpoints")
                    if os.path.exists(checkpoints_dir) and any(f.endswith('.th') for f in os.listdir(checkpoints_dir)):
                        status_report["Demucs"] = "Found 🔍"
                    else:
                        status_report["Demucs"] = "Downloaded ✅"
                        
                    for m in demucs_models:
                        self.after(0, lambda mod=m: self.progress_text.configure(text=f"Verifying Demucs: {mod}..."))
                        demucs.pretrained.get_model(m)
                except Exception as e:
                    status_report["Demucs"] = "Error ❌"
                    print(f"Demucs Error: {e}")

            # 3. OpenUnmix
            try:
                self.after(0, lambda: self.progress_text.configure(text="Verifying OpenUnmix model..."))
                hub_dir = os.path.join(self.models_dir, "hub")
                # PyTorch hub creates a folder with "open-unmix" in the name
                if os.path.exists(hub_dir) and any("open-unmix" in d.lower() for d in os.listdir(hub_dir)):
                    status_report["OpenUnmix"] = "Found 🔍"
                else:
                    status_report["OpenUnmix"] = "Downloaded ✅"
                    
                import torch
                torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq', trust_repo=True)
            except Exception as e:
                status_report["OpenUnmix"] = "Error ❌"
                print(f"OpenUnmix Error: {e}")

            # 4. Whisper
            try:
                self.after(0, lambda: self.progress_text.configure(text="Verifying Whisper model..."))
                import whisper
                whisper_dir = os.path.join(self.models_dir, "whisper")
                os.makedirs(whisper_dir, exist_ok=True)
                
                # Whisper usually names the base model file "base.pt"
                expected_model = os.path.join(whisper_dir, "base.pt")
                if not os.path.exists(expected_model):
                    status_report["Whisper"] = "Downloaded ✅"
                else:
                    status_report["Whisper"] = "Found 🔍"
                
                # Force whisper to download directly into your Models folder
                whisper.load_model("base", download_root=whisper_dir)
            except Exception as e:
                status_report["Whisper"] = "Error ❌"
                print(f"Whisper Error: {e}")

            # 5. Wav2Vec2
            try:
                self.after(0, lambda: self.progress_text.configure(text="Verifying Wav2Vec2 model..."))
                
                # We MUST set HuggingFace environments BEFORE importing transformers
                hf_dir = os.path.join(self.models_dir, "huggingface")
                os.environ["HF_HOME"] = hf_dir
                os.environ["HF_HUB_CACHE"] = os.path.join(hf_dir, "hub")
                
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
                
                hf_hub_dir = os.path.join(hf_dir, "hub")
                # HF uses folders formatted like "models--facebook--wav2vec2..."
                if os.path.exists(hf_hub_dir) and any("models--facebook--wav2vec2" in d for d in os.listdir(hf_hub_dir)):
                    status_report["Wav2Vec2"] = "Found 🔍"
                else:
                    status_report["Wav2Vec2"] = "Downloaded ✅"
                
                model_name = "facebook/wav2vec2-base-960h"
                Wav2Vec2Processor.from_pretrained(model_name)
                Wav2Vec2ForCTC.from_pretrained(model_name)
            except Exception as e:
                status_report["Wav2Vec2"] = "Error ❌"
                print(f"Wav2Vec2 Error: {e}")

            # 6. Vosk
            try:
                if vosk_needs_download:
                    self.after(0, lambda: self.progress_text.configure(text="Downloading Vosk model..."))
                    url = f"https://alphacephei.com/vosk/models/{default_vosk}.zip"
                    os.makedirs(vosk_dir, exist_ok=True)
                    self._download_and_extract_sync(url, vosk_dir)
                    status_report["Vosk"] = "Downloaded ✅"
                else:
                    status_report["Vosk"] = "Found 🔍"
            except Exception as e:
                status_report["Vosk"] = "Error ❌"
                print(f"Vosk Error: {e}")

            # ==========================================
            # CLEANUP & UI UPDATE
            # ==========================================
            self.after(0, lambda: self.progress_text.configure(text="✅ Background setup complete!"))
            
            if hasattr(self, 'scan_models_directory'):
                self.after(0, self.scan_models_directory)

            # Stop Progress Bar
            self.after(0, self.progress_bar.stop)
            self.after(0, lambda: self.progress_bar.configure(mode="determinate"))
            self.after(0, lambda: self.progress_bar.set(1.0)) # <--- Fixed line
            self.after(3000, getattr(self.progress_bar, 'grid_remove', lambda: None))

            # SHOW THE SUMMARY POPUP (Wait a tiny bit so the UI has time to catch up)
            self.after(500, lambda: self.show_model_summary(status_report))

        # Start the background thread
        threading.Thread(target=download_thread, daemon=True).start()

    def _download_and_extract_sync(self, download_url, extract_path):
        """!
        @brief Helper for synchronous download & extract (should be called inside a thread).
        """
        temp_dir = tempfile.gettempdir()
        zip_path = os.path.join(temp_dir, "temp_model_download.zip")
        
        import urllib.request
        import zipfile
        
        urllib.request.urlretrieve(download_url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # FIX: Extracts exactly to the path we provide
            zip_ref.extractall(extract_path) 
            
        if os.path.exists(zip_path):
            os.remove(zip_path)

    def show_model_summary(self, status_report):
        """!
        @brief Displays a popup window summarizing the results of the model download thread.
        """
        dialog = ctk.CTkToplevel(self)
        dialog.title("Setup Summary")
        dialog.geometry("350x300")
        dialog.attributes("-topmost", True)
        dialog.grab_set()

        # Header
        ctk.CTkLabel(dialog, text="Model Setup Results", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20, 15))

        # Dynamically create labels for each tool's status
        for tool, status in status_report.items():
            # Pick a color based on the status emoji
            text_color = "white" # Default
            if "❌" in status:
                text_color = "#ff6666" # Light Red
            elif "✅" in status:
                text_color = "#66ff66" # Light Green
            elif "🔍" in status:
                text_color = "#66ccff" # Light Blue

            row_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            row_frame.pack(fill="x", padx=40, pady=5)
            
            ctk.CTkLabel(row_frame, text=f"{tool}:", font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")
            ctk.CTkLabel(row_frame, text=status, text_color=text_color, font=ctk.CTkFont(size=14)).pack(side="right")

        # Close button
        ctk.CTkButton(dialog, text="Awesome!", command=dialog.destroy, width=120).pack(pady=(25, 10))

    def scan_models_directory(self):
        """!
        @brief Scans the models_dir and smartly finds downloaded models for vosk only.
        Updates UI entries for Vosk, but ignores PyTorch hashed files for Demucs/OpenUnmix.
        """
        if not self.models_dir or not os.path.exists(self.models_dir):
            if hasattr(self, 'progress_text'):
                self.progress_text.configure(text="❌ Models directory not found.")
            return

        # 1. Define where Vosk stores its files
        vosk_path = os.path.join(self.models_dir, "vosk")

        # 2. Scan for Vosk (Look for subfolders)
        found_vosk = []
        if os.path.exists(vosk_path):
            found_vosk = [d for d in os.listdir(vosk_path) if os.path.isdir(os.path.join(vosk_path, d))]

        # 3. Safely update ONLY the Vosk UI text entry
        if hasattr(self, 'model_vars'):
            if found_vosk and "vosk" in self.model_vars:
                self.model_vars["vosk"].set(", ".join(sorted(list(set(found_vosk)))))
                self.transcription_models["vosk"] = sorted(list(set(found_vosk)))

        # 4. Refresh the dropdown menus in the Input/Output tabs (Safely)
        try:
            if hasattr(self, 'on_tool_change'): 
                self.on_tool_change()
            if hasattr(self, 'on_trans_tool_change'): 
                self.on_trans_tool_change()
        except Exception as e:
            # If the separation tabs aren't loaded yet, ignore the visual update
            pass

        if hasattr(self, 'progress_text'):
            self.progress_text.configure(text="🔍 Available models scanned and updated.")

    def import_custom_model(self):
        """!
        @brief Safely copies a user-selected custom model folder into the app's models directory.
        """
        source_dir = filedialog.askdirectory(title="Select Custom Model Folder (e.g., OpenUnmix)")
        if not source_dir: return 
            
        folder_name = os.path.basename(source_dir)
        destination_dir = os.path.join(self.models_dir, folder_name)
        
        if os.path.exists(destination_dir):
            messagebox.showwarning("Model Exists", f"A model named '{folder_name}' is already in the models folder.")
            return
            
        has_model_files = any(f.endswith(('.pth', '.pt', '.bin', '.onnx', '.json', '.yaml')) for f in os.listdir(source_dir))
        
        if not has_model_files:
            if not messagebox.askyesno("Suspicious Folder", "This folder doesn't seem to contain standard AI model files.\n\nAre you sure you want to import it?"):
                return

        try:
            self.progress_text.configure(text=f"Importing {folder_name}...")
            shutil.copytree(source_dir, destination_dir)
            self.scan_models_directory() # Auto-scan after import
            messagebox.showinfo("Success", f"Model '{folder_name}' imported successfully!")
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to copy model directory.\n\nDetails: {str(e)}")

    def _browse_folder(self, string_var):
        """Helper to let users click a button instead of typing a path."""
        folder = filedialog.askdirectory(initialdir=string_var.get())
        if folder:
            string_var.set(folder)

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

    def show_batch_summary_window(self, title, details):
        """Creates a non-blocking, floating window to display batch results."""
        summary_win = ctk.CTkToplevel(self)
        summary_win.title(title)
        summary_win.geometry("550x400")
        
        # Optional: Make it stay on top of the main window so it doesn't get lost
        summary_win.attributes("-topmost", True)
        
        # Add a scrollable textbox for the details
        textbox = ctk.CTkTextbox(summary_win, wrap="word", font=("Roboto", 13))
        textbox.pack(fill="both", expand=True, padx=20, pady=20)
        textbox.insert("0.0", details)
        textbox.configure(state="disabled")  # Make it read-only so the user can't type in it
        
        # Add a close button
        close_btn = ctk.CTkButton(summary_win, text="Close", corner_radius=0, command=summary_win.destroy)
        close_btn.pack(pady=(0, 20))

    def abort_separation_process(self):
        self.abort_separation = True
        self.progress_text.configure(text="Aborting separation...")
        self.abort_button.grid_remove()

    def separate_audio(self):
        """
        @brief Prepares a batch of selected files and folders for audio separation.
        
        This method iterates through the user's selection in the GUI. If a folder 
        is selected, it automatically scrapes the directory for valid audio files 
        (.wav, .mp3, .flac, .ogg, .m4a, .aac) while using a set to prevent duplicate 
        file processing. It then retrieves all active UI parameters (AI tool, format, 
        bitrates, device, etc.) and spawns a daemon thread to execute the batch 
        separation without freezing the main application window.
        
        @note Displays a warning messagebox if no valid files/folders are selected, 
              or if numerical inputs (like sample rate or shifts) contain invalid characters.
              
        @return None
        """
        import os # Ensure os is imported if not already at the top
        
        # 1. Gather all selected files
        selected_files = []
        selected_paths = set() # To prevent duplicates if user checks folder AND song
        
        # Define what audio files we want to grab from folders
        valid_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

        for idx, info in self.input_selection_dict.items():
            if info["var"].get():  # If checked
                
                # Handle individual songs
                if info["type"] == 'song':
                    file_path = info["data"]["path"]
                    if file_path not in selected_paths:
                        selected_files.append(info["data"])
                        selected_paths.add(file_path)
                        
                # Handle folders
                elif info["type"] == 'folder':
                    folder_path = info["data"]["path"]
                    try:
                        # Look directly inside the folder (no subfolders)
                        for filename in os.listdir(folder_path):
                            file_path = os.path.join(folder_path, filename)
                            
                            # If it's a file and has a valid audio extension
                            if os.path.isfile(file_path):
                                ext = os.path.splitext(filename)[1].lower()
                                if ext in valid_extensions and file_path not in selected_paths:
                                    
                                    # Create a mock data dictionary so it matches the song format
                                    song_data = {"name": filename, "path": file_path}
                                    selected_files.append(song_data)
                                    selected_paths.add(file_path)
                                    
                    except Exception as e:
                        import logging
                        logging.error(f"Error reading folder {folder_path}: {e}")

        if not selected_files:
            from tkinter import messagebox
            self.after(0, lambda: messagebox.showwarning("No selection", "Please select at least one song or folder to separate."))
            return

        # 2. Gather UI params
        ai_tool = self.ai_tool_var.get()
        model = self.model_var.get()
        channels = self.channel_var.get() 
        fmt = self.format_var.get()
        vocals_folder = self.output_folders["vocals"]
        instr_folder = self.output_folders["instrumentals"]
        device = self.device_var.get()

        try:
            sr = int(self.sr_var.get()) if fmt in ["wav", "flac"] else 44100
            shifts = int(self.shifts_var.get()) if ai_tool == "Demucs" else 1
            mp3_preset = int(self.mp3_preset_slider.get()) if fmt == "mp3" and ai_tool == "Demucs" else 2
            bitrate = f"{int(self.bitrate_var.get())}k" if fmt == "mp3" else "192k"
        except ValueError:
            from tkinter import messagebox
            self.after(0, lambda: messagebox.showwarning("Invalid Input", "Please ensure parameters contain only numbers. Using defaults."))
            sr, shifts, mp3_preset, bitrate = 44100, 1, 2, "192k"

        bit_depth = self.bit_depth_var.get() if fmt == "wav" and ai_tool == "Demucs" else None
        
        # 3. Start the thread, passing the LIST of selected files PLUS 'device'
        import threading
        thread = threading.Thread(target=self._run_separation, args=(
            selected_files, vocals_folder, instr_folder, ai_tool, model, channels, 
            fmt, sr, bitrate, bit_depth, mp3_preset, shifts, device 
        ))
        thread.daemon = True
        thread.start()

    def _run_separation(self, selected_files, vocals_folder, instr_folder,
                        ai_tool, model, channels, fmt, sr, bitrate, bit_depth, mp3_preset, shifts, device): 
        """
        @brief Executes audio separation on a batch of files iteratively in a background thread.
        
        This method handles the heavy lifting of the separation process. It safely manages 
        lazy loading of the requested AI model, iterates through the provided batch of files, 
        tracks success/failure states, and securely updates the GUI main thread using Tkinter's 
        `after()` method.
        
        @param selected_files (list) A list of dictionaries, each containing 'name' and 'path' of the audio file.
        @param vocals_folder (str) Directory path to save the extracted vocal tracks.
        @param instr_folder (str) Directory path to save the extracted instrumental tracks.
        @param ai_tool (str) The AI framework being used (e.g., "Demucs", "Spleeter").
        @param model (str) The specific model variant to load.
        @param channels (int) The number of separation stems (e.g., 2, 4, 6).
        @param fmt (str) The desired output audio format (e.g., "mp3", "wav").
        @param sr (int) Sample rate for the output audio.
        @param bitrate (str) Target bitrate for compressed formats (e.g., "192k").
        @param bit_depth (int) Bit depth for lossless formats like WAV.
        @param mp3_preset (int) Encoding preset specifically for Demucs MP3 output.
        @param shifts (int) Number of random shifts for Demucs to improve quality.
        @param device (str) Hardware acceleration choice ("Auto", "CPU", "GPU").
        
        @note Monitors `self.abort_separation` to gracefully halt the batch loop if the user cancels.
        
        @return None
        """
        total_files = len(selected_files)

        def update_progress(percent, message, file_index):
            if self.abort_separation:
                self.after(0, lambda: self.progress_text.configure(text="Separation aborted by user."))
                self.after(0, lambda: self.progress_bar.configure(mode="indeterminate"))
                self.after(0, lambda: self.progress_bar.grid_remove())
                self.after(0, lambda: self.abort_button.grid_remove())
                raise RuntimeError("ABORT_REQUESTED") 
            
            # Format message with batch progress (e.g., "[1/3] Demucs separating...")
            batch_msg = f"[{file_index}/{total_files}] {message}"
            
            self.after(0, lambda: self.abort_button.grid())
            self.after(0, lambda: self.progress_bar.configure(mode="determinate"))
            self.after(0, lambda: self.progress_bar.set(percent / 100.0))
            self.after(0, lambda: self.progress_bar.grid())
            self.after(0, lambda: self.progress_text.configure(text=batch_msg))

        try:
            # --- LAZY LOADING BLOCK ---
            update_progress(5, f"Loading {ai_tool} model...", 1)
            # Force CPU environment variable BEFORE loading TensorFlow API if needed
            if device == "CPU":
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            elif device == "GPU" and "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] == "-1":
                del os.environ["CUDA_VISIBLE_DEVICES"]

            if ai_tool == "Spleeter" and getattr(self, "spleeter_sep", None) is None:
                import separators.spleeter_separator as spleeter_sep
                self.spleeter_sep = spleeter_sep.SpleeterSeparator()
            elif ai_tool == "Demucs" and getattr(self, "demucs_sep", None) is None:
                import separators.demucs_separator as demucs_sep
                self.demucs_sep = demucs_sep.DemucsSeparator()
            elif ai_tool == "OpenUnmix" and getattr(self, "openunmix_sep", None) is None:
                import separators.openunmix_separator as openunmix_sep
                self.openunmix_sep = openunmix_sep.OpenUnmixSeparator()

            # --- BATCH TRACKING LISTS ---
            successful_files = []
            failed_files = []

            # --- BATCH LOOP ---
            for i, song in enumerate(selected_files, 1):
                if self.abort_separation: break

                input_path = song['path']
                song_name = os.path.splitext(song['name'])[0]
                
                # Callback specifically tied to the current file's index
                cb = lambda p, m: update_progress(p, m, i)
                
                if ai_tool == "Spleeter":
                    result = self.spleeter_sep.separate(input_path, song_name, vocals_folder, instr_folder, channels, fmt, sr, bitrate, device, progress_callback=cb)
                elif ai_tool == "Demucs":
                    result = self.demucs_sep.separate(input_path, song_name, vocals_folder, instr_folder, model, channels, fmt, sr, bitrate, bit_depth, mp3_preset, shifts, device, progress_callback=cb)
                elif ai_tool == "OpenUnmix":
                    result = self.openunmix_sep.separate(input_path, song_name, vocals_folder, instr_folder, model, channels, fmt, sr, bitrate, device, progress_callback=cb)
                
                # Check if the tuple returned success (True) and has the output names
                if isinstance(result, tuple) and len(result) >= 3 and result[0]:
                    vocals_name = result[1]
                    instr_name = result[2]
                    successful_files.extend([vocals_name, instr_name]) # Add the saved file names to our list
                else:
                    logging.error(f"Failed to separate: {song_name}")
                    failed_files.append(song_name) # Add the original song name to the failed list

            # --- COMPLETION ---
            if not self.abort_separation:
                success_count = len(successful_files) // 2 
                self.after(0, lambda: self.progress_text.configure(text=f"Batch complete! {success_count}/{total_files} processed successfully."))
                self.after(0, lambda: self.progress_bar.grid_remove())
                self.after(0, self.load_outputs) 

                # Build the detailed message
                details = ""
                if successful_files:
                    details += "✅ Successfully generated:\n" + "\n".join(successful_files) + "\n\n"
                if failed_files:
                    details += "❌ Failed to process:\n" + "\n".join(failed_files) + "\n\nCheck the terminal for detailed error logs."

                title = "Batch Finished with Errors" if failed_files else "Batch Finished Successfully"
                
                # Show our custom, non-blocking window!
                self.after(0, lambda: self.show_batch_summary_window(title, details))

        except Exception as e:
            if str(e) == "ABORT_REQUESTED":
                logging.info("Separation aborted by user.")
            else:
                self.after(0, lambda: self.progress_text.configure(text=f"Error: {str(e)}"))
                logging.error(f"Thread error: {e}", exc_info=True)
        finally:
            self.abort_separation = False
            self.after(0, lambda: self.abort_button.grid_remove())

    def run_standalone_transcription(self):
        """
        @brief Prepares and initiates a batch transcription process.
        
        Gathers selected audio files from both the Vocals and Instrumentals tracking 
        dictionaries in the GUI. Collects user-defined parameters for the transcription 
        tool and spawns a daemon thread (`_exec_standalone_trans`) to prevent freezing 
        the main application window.
        
        @note Displays a warning messagebox if no valid files are currently selected.
        @return None
        """
        # 1. Gather selected vocals and instrumentals directly from the dictionaries
        selected_audio_files = []
        
        # Grab selected vocals
        for idx, info in self.vocals_selection_dict.items():
            if info["var"].get(): # If the switch is ON
                selected_audio_files.append(info["data"])

        # Grab selected instrumentals
        for idx, info in self.instr_selection_dict.items():
            if info["var"].get(): # If the switch is ON
                selected_audio_files.append(info["data"])

        # Check if anything was selected at all
        if not selected_audio_files:
            self.after(0, lambda: messagebox.showwarning("Selection Required", "Please turn on the switch for at least one Vocal or Instrumental file."))
            return
        
        # 2. Gather UI params
        tool = self.trans_tool_var.get()
        model = self.trans_model_var.get()
        lang = self.trans_lang_var.get()
        use_spk = self.use_spk_id_var.get() if tool == "vosk" else False
        out_folder = self.output_folders["transcriptions"]
        
        # --- NEW: Get the device from the UI ---
        device = self.device_var.get()

        # 3. Start thread
        threading.Thread(
            target=self._exec_standalone_trans, 
            # --- NEW: Pass device to args ---
            args=(selected_audio_files, out_folder, tool, model, lang, use_spk, device), 
            daemon=True
        ).start()

    def _exec_standalone_trans(self, selected_vocals, output_folder, tool, model, lang, use_spk, device):
        """
        @brief Executes batch transcription on selected audio files in a background thread.
        
        Manages the lazy loading of transcription models (Whisper, Vosk, Wav2Vec2) and 
        iterates over the selected batch. It tracks successful and failed operations, 
        formats output file names, and safely pushes UI updates to the main thread.
        
        @param selected_vocals (list) List of dictionaries containing file data to transcribe.
        @param output_folder (str) Directory path to save the generated text/subtitle files.
        @param tool (str) The specific transcription tool to use (e.g., "whisper", "vosk").
        @param model (str) The specific model tier/variant selected.
        @param lang (str) Target language for transcription models that support it.
        @param use_spk (bool) Flag indicating whether to use Speaker Diarization (Vosk only).
        @param device (str) Hardware acceleration choice ("Auto", "CPU", "GPU").
        
        @note Updates a summary UI window at the completion of the batch process.
        @return None
        """
        total_files = len(selected_vocals)

        def update_trans_progress(percent, message, file_index):
            batch_msg = f"[{file_index}/{total_files}] {message}"
            self.after(0, lambda: self.progress_bar.grid())
            self.after(0, lambda: self.progress_bar.set(percent / 100.0))
            self.after(0, lambda: self.progress_text.configure(text=batch_msg))

        try:
            # --- NEW: Apply environment variables for strict CPU fallback ---
            if device == "CPU":
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            elif device == "GPU" and "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] == "-1":
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # --- LAZY LOADING ---
            update_trans_progress(10, f"Initializing {tool}...", 1)
            if tool == "whisper" and getattr(self, "whisper_trans", None) is None:
                import separators.whisper_transcription as whisper_trans
                self.whisper_trans = whisper_trans.WhisperTranscription()
            elif tool == "vosk" and getattr(self, "vosk_trans", None) is None:
                import separators.vosk_transcription as vosk_trans
                self.vosk_trans = vosk_trans.VoskTranscription()
            elif tool == "wav2vec2" and getattr(self, "wav2vec2_trans", None) is None:
                import separators.wav2vec2_transcription as w2v2_trans 
                self.wav2vec2_trans = w2v2_trans.Wav2Vec2Transcription()

            # --- BATCH TRACKING LISTS ---
            successful_files = []  
            failed_files = []      

            # --- BATCH LOOP ---
            for i, vocal in enumerate(selected_vocals, 1):
                vocal_path = vocal['path']
                filename = vocal['name']
                
                # Create output name
                base_name = os.path.splitext(filename)[0]
                out_name = f"{base_name}_{tool}_{model.replace('/', '_')}.txt"
                out_path = os.path.join(output_folder, out_name)

                cb = lambda p, m: update_trans_progress(p, m, i)

                # --- NEW: Pass device=device to your transcribe methods ---
                if tool == "whisper":
                    result = self.whisper_trans.transcribe(vocal_path, out_path, model, lang, device_choice=device, progress_callback=cb)
                elif tool == "wav2vec2":
                    result = self.wav2vec2_trans.transcribe(vocal_path, out_path, model, device_choice=device, progress_callback=cb)
                elif tool == "vosk":
                    result = self.vosk_trans.transcribe(vocal_path, out_path, model, use_spk, device_choice=device, progress_callback=cb)
                
                # Check results and append to our lists instead of just counting!
                if isinstance(result, tuple) and len(result) == 2 and result[0]:
                    successful_files.append(result[1]) # Add the saved filename
                elif result is True: # Fallback
                    successful_files.append(out_name)
                else:
                    failed_files.append(filename) # Add the original vocal name that failed
                    logging.error(f"Failed to transcribe: {filename}")

            # --- COMPLETION ---
            self.after(0, lambda: self.progress_text.configure(text=f"Batch complete! {len(successful_files)}/{total_files} saved."))
            self.after(3000, lambda: self.progress_bar.grid_remove()) 
            self.after(0, self.load_outputs)

            # Build the detailed message
            details = ""
            if successful_files:
                details += "✅ Successfully transcribed:\n" + "\n".join(successful_files) + "\n\n"
            if failed_files:
                details += "❌ Failed to transcribe:\n" + "\n".join(failed_files) + "\n\nCheck the terminal for detailed error logs."

            title = "Transcription Finished with Errors" if failed_files else "Transcription Finished Successfully"
            
            self.after(0, lambda: self.show_batch_summary_window(title, details))

        except Exception as e:
            logging.error(f"Standalone trans error: {e}", exc_info=True)
            # Capture the error message as a string right now
            err_msg = str(e) 
            # Pass the captured string into the lambda as a default argument
            self.after(0, lambda msg=err_msg: self.progress_text.configure(text=f"Error: {msg}"))

if __name__ == "__main__":
    # MUST BE THE VERY FIRST THING UNDER __main__
    multiprocessing.freeze_support()

    app = SeparationApp()
    app.mainloop()
