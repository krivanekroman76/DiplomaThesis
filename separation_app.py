import warnings
warnings.simplefilter('ignore')  # Hide unnecessary warnings
import os
# STRICT GAG ORDER FOR TENSORFLOW (Must be set before any AI imports)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Hides oneDNN custom operations warnings
# Force Numba to use a stable threading layer before librosa is even imported
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'
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
import math
import queue
from dataclasses import dataclass
from typing import Optional
import torch
import gc
from typing import Dict, Any
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
SpleeterSeparator = None
DemucsSeparator = None
OpenUnmixSeparator = None
def get_app_dir():
    """Always returns the directory containing the .exe or the main .py script.
       Use this for saving settings.json or user output files!"""
    if getattr(sys, 'frozen', False):
        os.environ['NUMBA_CACHE_DIR'] = os.path.join(os.environ.get('TEMP', os.getcwd()), 'numba_cache')
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

def get_binaries_dir():
    """Returns the directory containing bundled files like ffmpeg.exe.
       When compiled, this magically points to the temporary folder."""
    if getattr(sys, 'frozen', False):
        # We use getattr to avoid Pylance "unknown attribute" warnings
        # and provide a fallback to the current directory just in case.
        return getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    else:
        # When running uncompiled, everything is just in the same folder
        return os.path.dirname(os.path.abspath(__file__))

def setup_ffmpeg_environment():
    """Injects the local ffmpeg path into the system's PATH variable temporarily."""
    # Use the BINARIES directory to find ffmpeg, NOT the app directory!
    bin_dir = get_binaries_dir()
    
    # Check if we've already injected it during this session
    if os.environ.get("FFMPEG_INJECTED") != "TRUE":
        
        # Now it correctly looks inside '_internal' when compiled
        if os.path.exists(os.path.join(bin_dir, "ffmpeg.exe")):
            # Prepend our internal folder to the system PATH
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            logging.info(f"FFmpeg path injected from: {bin_dir}")
        else:
            logging.warning(f"ffmpeg.exe not found in {bin_dir}! Audio processing may fail if not installed system-wide.")
            
        # Set the lock so it never runs again
        os.environ["FFMPEG_INJECTED"] = "TRUE"

# --- CALL IT IMMEDIATELY ---
setup_ffmpeg_environment()

def open_file(path):
    """Open a file using the system's default application."""
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.call(("open", path))
    else:
        subprocess.call(("xdg-open", path))

@dataclass
class SeparationSettings:
    ai_tool: str
    model: str
    channels: str
    fmt: str
    sr: int
    bitrate: str
    bit_depth: Optional[str]  # Using Optional since it can be None
    mp3_preset: int
    shifts: int
    overlap: float
    flac_compression: int
    device: str
    vocals_folder: str
    instr_folder: str

@dataclass
class TranscriptionSettings:
    tool: str
    model: str
    lang: str
    use_spk: bool
    device: str
    output_folder: str

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
        
        # Set up the Threading Mailbox EARLY
        self.folder_size_queue = queue.Queue()
        self._check_size_queue()

        # Pagination Trackers
        self.current_pages = {
            "input": 0,
            "vocals": 0, 
            "instr": 0, 
            "trans": 0
        }

        # 2. Set Theme and Scaling (Global)
        ctk.set_appearance_mode(self.appearance_mode)
        ctk.set_default_color_theme(self.color_theme)
        # Calculate scaling safely
        try:
            scaling_float = int(self.scaling.replace("%", "")) / 100
            ctk.set_widget_scaling(scaling_float)
        except (ValueError, AttributeError):
            ctk.set_widget_scaling(1.0)

        self.update() # Forces Windows to create the window handle and initialize fonts

        # 3. BUILD UI (The Body)
        self._setup_main_containers()

        # 4. Initial tab loading is deferred until the main window is fully rendered to avoid heavy processing during startup
        self.after(200, self._initial_startup_sequence)

        self._last_width = self.winfo_width()
        self._last_height = self.winfo_height()
        
        # Bind the configure event
        self.bind("<Configure>", self._handle_resize)
        self._resize_timer = None

        # Force Tkinter to finish drawing the initial window geometry (e.g., 1200x700)
        self.update_idletasks() 

        # Intercept the "X" close button
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
            """Standard cleanup before exiting."""
            # 1. Stop any active AI separation
            self.abort_separation = True 

            # 2. Clear AI models from RAM/VRAM
            self.spleeter_sep = None
            self.demucs_sep = None
            self.openunmix_sep = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            gc.collect()

            # 3. Destroy the UI and exit
            self.destroy()
    
    def _initial_startup_sequence(self):
        """Handles heavy UI math after the main window is actually visible."""
        # 1. Decide which tab to show
        if getattr(self, "is_first_run", False):
            self._switch_tab(self.settings_frame, self.settings_button, "settings")
            self.after(500, self.show_welcome_tutorial)
            self.is_first_run = False
            self.save_settings()
        else:
            self._switch_tab(self.input_frame, self.input_button, "input")

        # 2. Now that the window is rendered, calculate the layout
        self._recalculate_pagination(is_initialization=True)

    def _handle_resize(self, event):
        # Only trigger if the main window was resized, not a sub-widget
        if event.widget == self:
            if self._resize_timer:
                self.after_cancel(self._resize_timer)
            # Wait 250ms after the last move before recalculating
            self._resize_timer = self.after(250, self._recalculate_pagination)

    def show_welcome_tutorial(self):
        """Displays a paginated interactive tutorial for first-time users."""
        dialog = ctk.CTkToplevel(self)
        dialog.title("Audio Separator - Quick Start Guide")
        dialog.geometry("600x500") 
        dialog.attributes("-topmost", True)
        dialog.resizable(False, False)
    
        self.tutorial_page = 0
        
        # --- REFINED CONTENT BASED ON YOUR UI ---
        pages = [
            {
                "title": "Welcome to Audio Separator! 🚀",
                "text": "This tool uses state-of-the-art AI to pull vocals and instruments apart from any song.\n\n"
                        "Let's get you set up in 60 seconds so you can start processing your first batch."
            },
            {
                "title": "⚙️ Step 1: Initial Setup",
                "text": "**Location: Settings Tab**\n\n"
                        "Before your first run, tell the app where to save your AI models and where to look for music.\n\n"
                        "• Use **'Download Default Models'** to get the AI brains ready.\n"
                        "• Set your **Default Input/Output folders** to save time later."
            },
            {
                "title": "🎵 Step 2: Load Your Music",
                "text": "**Location: Input Tab**\n\n"
                        "Click **'Add Song'** to bring files into the list. \n\n"
                        "**Pro Tip:** Click the **'i' (Info)** button next to a song to open the **Audio Inspector**. This lets you check bitrates and sync your project settings automatically!"
            },
            {
                "title": "✂️ Step 3: Choose Your AI Tool",
                "text": "**Location: Separation Menu (Right Side)**\n\n"
                        "Choose your weapon:\n"
                        "• **Spleeter:** Fast and efficient.\n"
                        "• **Demucs:** High quality, great for complex tracks.\n"
                        "• **OpenUnmix:** Excellent for research-grade separation.\n\n"
                        "Hit **'Start Batch Separation'** to begin the magic."
            },
            {
                "title": "🎧 Step 4: Review Your Tracks",
                "text": "**Location: Separated Output Tab**\n\n"
                        "Your files are now split into **Vocals** and **Instrumentals** lists.\n\n"
                        "• Press ▶ to preview your results.\n"
                        "• If you don't like a result, use the 🗑 button to clean up your workspace."
            },
            {
                "title": "🗣️ Step 5: High-Accuracy Transcription",
                "text": "**Location: Separated Output -> Transcription Menu**\n\n"
                        "Want lyrics or scripts?\n"
                        "1. **Check the box** next to any Vocal track.\n"
                        "2. Select a tool like **Whisper** or **Vosk** on the right.\n"
                        "3. Click **'Transcribe'**."
            },
            {
                "title": "📝 Step 6: Final Results",
                "text": "**Location: Transcribed Output Tab**\n\n"
                        "Your text files live here. Click the **📖 (Reader)** icon to view the transcription instantly.\n\n"
                        "Everything is saved to your output folder automatically!"
            }
        ]

        # --- UI LAYOUT ---
        content_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        content_frame.pack(fill="both", expand=True, padx=40, pady=(20, 10))

        title_label = ctk.CTkLabel(content_frame, text="", font=ctk.CTkFont(size=24, weight="bold"), text_color=("#3B8ED0", "#1f6aa5"))
        title_label.pack(pady=(10, 10))

        # Progress indicator (e.g., Step 1 of 7)
        progress_label = ctk.CTkLabel(content_frame, text="", font=ctk.CTkFont(size=12))
        progress_label.pack(pady=(0, 20))

        text_label = ctk.CTkLabel(content_frame, text="", font=ctk.CTkFont(size=16), wraplength=500, justify="left")
        text_label.pack(pady=10, fill="both", expand=True)

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack(fill="x", padx=30, pady=20)

        def update_ui():
            page = pages[self.tutorial_page]
            title_label.configure(text=page["title"])
            text_label.configure(text=page["text"])
            progress_label.configure(text=f"Step {self.tutorial_page + 1} of {len(pages)}")
            
            # --- INTELLIGENT HIGHLIGHTING ---
            try:
                if self.tutorial_page == 1: # Settings
                    self._switch_tab(self.settings_frame, self.settings_button, "settings")
                elif self.tutorial_page in [2, 3]: # Input
                    self._switch_tab(self.input_frame, self.input_button, "input")
                elif self.tutorial_page in [4, 5]: # Sep Output
                    self._switch_tab(self.sep_out_frame, self.sep_out_button, "sep_out")
                elif self.tutorial_page == 6: # Trans Output
                    self._switch_tab(self.trans_out_frame, self.trans_out_button, "trans_out")
                
                # Flash the main button for the current tab
                current_tab_btn = [self.settings_button, self.input_button, self.sep_out_button, self.trans_out_button]
                # Logic to trigger flash_highlight on relevant buttons could go here
            except Exception as e:
                logging.debug(f"Tutorial highlight skip: {e}")

            # Navigation logic
            if self.tutorial_page == 0:
                prev_btn.configure(state="disabled", fg_color="transparent")
            else:
                prev_btn.configure(state="normal", fg_color=("#3B8ED0", "#1f6aa5"))
                
            if self.tutorial_page == len(pages) - 1:
                next_btn.configure(text="Finish & Explore! ✨")
                skip_btn.pack_forget() 
            else:
                skip_btn.pack(expand=True)
                next_btn.configure(text="Next ➔")

        def go_next():
            if self.tutorial_page < len(pages) - 1:
                self.tutorial_page += 1
                update_ui()
            else:
                dialog.destroy()

        def go_prev():
            if self.tutorial_page > 0:
                self.tutorial_page -= 1
                update_ui()

        # --- CONTROLS ---
        prev_btn = ctk.CTkButton(btn_frame, text="⬅ Previous", width=110, command=go_prev)
        prev_btn.pack(side="left")

        next_btn = ctk.CTkButton(btn_frame, text="Next ➔", width=140, command=go_next, font=ctk.CTkFont(weight="bold"))
        next_btn.pack(side="right")
        
        skip_btn = ctk.CTkButton(btn_frame, text="Skip Tour", width=80, fg_color="transparent", border_width=1, command=dialog.destroy)
        skip_btn.pack(expand=True)

        update_ui()
        
    def flash_highlight(self, widget, flashes=3):
        """Flashes a widget to draw attention. Smartly handles borders, text, and backgrounds."""
        try:
            if not widget.winfo_exists():
                return

            # Smart attribute targeting based on widget type
            if isinstance(widget, ctk.CTkEntry):
                color_attr = "border_color"
            elif isinstance(widget, ctk.CTkLabel):
                color_attr = "text_color" # Flash the text color for labels!
            else:
                color_attr = "fg_color"

            original_color = widget.cget(color_attr)
            highlight_color = "#D4AF37" # Gold/yellow highlight
            
            def toggle(count):
                if count <= 0 or not widget.winfo_exists():
                    try: widget.configure(**{color_attr: original_color})
                    except: pass
                    return
                
                current_color = widget.cget(color_attr)
                new_color = highlight_color if current_color == original_color else original_color
                update_kwargs: Dict[str, Any] = {color_attr: new_color}
                widget.configure(**update_kwargs)
                
                self.after(350, toggle, count - 1)
                
            toggle(flashes * 2)
        except Exception as e:
            logging.info(f"Highlight warning: {e}")

    def _switch_tab(self, active_frame, active_button, tab_name):
        frames = [self.input_frame, self.sep_out_frame, self.trans_out_frame, self.settings_frame]
        buttons = [self.input_button, self.sep_out_button, self.trans_out_button, self.settings_button]
        
        # Hide all frames
        for frame in frames:
            frame.grid_forget()

        # --- ACTIVE MEMORY MANAGEMENT (Forget inactive tabs) ---
        if tab_name != "input" and getattr(self, 'input_tab_loaded', False):
            for widget in self.input_frame.winfo_children(): widget.destroy()
            self.input_tab_loaded = False
            
        if tab_name != "sep_out" and getattr(self, 'sep_out_tab_loaded', False):
            for widget in self.sep_out_frame.winfo_children(): widget.destroy()
            self.sep_out_tab_loaded = False

        if tab_name != "trans_out" and getattr(self, 'trans_out_tab_loaded', False):
            for widget in self.trans_out_frame.winfo_children(): widget.destroy()
            self.trans_out_tab_loaded = False

        # --- FIXED: Added Memory Management for Settings ---
        if tab_name != "settings" and getattr(self, 'settings_tab_loaded', False):
            for widget in self.settings_frame.winfo_children(): widget.destroy()
            self.settings_tab_loaded = False

        # Show active frame
        active_frame.grid(row=0, column=0, sticky="nsew")

        # Grab the standard and active colors directly from the current UI theme
        default_bg = ctk.ThemeManager.theme["CTkButton"]["fg_color"]
        active_bg = ctk.ThemeManager.theme["CTkButton"]["hover_color"]
        default_text = ctk.ThemeManager.theme["CTkButton"]["text_color"]

        # Reset all buttons to the standard button look
        for button in buttons:
            button.configure(
                fg_color=default_bg, 
                text_color=default_text
            )

        # Make the active button darker to show it is currently selected
        active_button.configure(
            fg_color=active_bg,
            text_color=default_text
        )

        self.focus_set() 

        # --- LAZY LOADING LOGIC ---
        if tab_name == "input" and not getattr(self, 'input_tab_loaded', False):
            self.create_input_tab() 
            if hasattr(self, 'load_input'): self.load_input()
            self.input_tab_loaded = True
            
        elif tab_name == "sep_out" and not getattr(self, 'sep_out_tab_loaded', False):
            self.create_sep_out_tab() 
            if hasattr(self, 'load_separation_outputs'): self.load_separation_outputs() 
            self.sep_out_tab_loaded = True

        elif tab_name == "trans_out" and not getattr(self, 'trans_out_tab_loaded', False):
            self.create_trans_out_tab() 
            if hasattr(self, 'load_transcription_outputs'): self.load_transcription_outputs() 
            self.trans_out_tab_loaded = True
        
        # --- FIXED: Added Lazy Loading Check for Settings ---
        elif tab_name == "settings" and not getattr(self, 'settings_tab_loaded', False):
            if hasattr(self, 'create_settings_tab'):
                self.after(100, self.create_settings_tab)
            self.settings_tab_loaded = True

        if hasattr(self, '_free_inactive_models'):
            self._free_inactive_models(tab_name)

    def update_theme_settings(self, new_value: str, setting_type: str):
        """Updates the appearance mode or color theme safely."""
        
        if setting_type == "mode":
            # GUARD: If it's already the current mode, do nothing
            if new_value == self.appearance_mode:
                return
            
            self.appearance_mode = new_value
            ctk.set_appearance_mode(self.appearance_mode)
            
            # Save the new mode to settings.json
            if hasattr(self, 'save_settings'):
                self.save_settings()
                
        elif setting_type == "theme":
            # GUARD: If it's already the current theme, do absolutely nothing!
            if new_value == self.color_theme:
                return
            
            self.color_theme = new_value
            
            # Save the new theme to settings
            if hasattr(self, 'save_settings'):
                self.save_settings()
                
            # DEFER the massive reload by 100ms so the dropdown menu can close safely!
            self.after(100, self._master_reload_pipeline)

    def change_scaling_event(self, new_scaling: str):
        """Saves the scaling, enforces limits, and triggers a full UI rebuild."""
        
        raw_val = int(new_scaling.replace("%", ""))
        clamped_val = max(50, min(200, raw_val))
        final_scaling_str = f"{clamped_val}%"
        
        self.scaling = final_scaling_str
        if hasattr(self, 'save_settings'):
            self.save_settings()

        self.after(100, self._master_reload_pipeline)

    def _setup_main_containers(self):
        # --- ROOT CONTAINER ---
        # Transparent and sharp
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.pack(fill="both", expand=True)
        
        self.main_frame.grid_columnconfigure(0, weight=0)
        self.main_frame.grid_columnconfigure(1, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=0)

        base_scale = int(self.scaling.replace("%", ""))
        
        if base_scale >= 120:
            self.sidebar = ctk.CTkScrollableFrame(self.main_frame, corner_radius=0, fg_color="transparent", width=170)
        else:
            self.sidebar = ctk.CTkFrame(self.main_frame, corner_radius=0, fg_color="transparent", width=170)
            
        self.sidebar.grid(row=0, column=0, sticky="nsew", rowspan=2)
        
        # --- NAVIGATION BUTTONS ---
        btn_width = 135

        self.input_button = ctk.CTkButton(
            self.sidebar, text="Input", width=btn_width, corner_radius=0,
            command=lambda: self._switch_tab(self.input_frame, self.input_button, "input")
        )
        self.input_button.grid(row=0, column=0, padx=10, pady=(20, 10), sticky="ew")
        
        self.sep_out_button = ctk.CTkButton(
            self.sidebar, text="Separated Output", width=btn_width, corner_radius=0,
            command=lambda: self._switch_tab(self.sep_out_frame, self.sep_out_button, "sep_out")
        )
        self.sep_out_button.grid(row=1, column=0, padx=10, pady=(20, 10), sticky="ew")

        self.trans_out_button = ctk.CTkButton(
            self.sidebar, text="Transcribed Output", width=btn_width, corner_radius=0,
            command=lambda: self._switch_tab(self.trans_out_frame, self.trans_out_button, "trans_out")
        )
        self.trans_out_button.grid(row=2, column=0, padx=10, pady=(20, 10), sticky="ew")

        # 2. THE SPACER (This pushes everything below row 3 to the bottom, only when using a non-scrollable sidebar)
        self.sidebar.grid_rowconfigure(3, weight=1)

        self.settings_button = ctk.CTkButton(
            self.sidebar, text="Settings", width=btn_width, corner_radius=0,
            command=lambda: self._switch_tab(self.settings_frame, self.settings_button, "settings")
        )
        self.settings_button.grid(row=4, column=0, padx=10, pady=(10, 20), sticky="ew")

        # --- Appearance Mode (Light/Dark) ---
        appearance_mode_label = ctk.CTkLabel(self.sidebar, text="Appearance Mode:", anchor="w")
        appearance_mode_label.grid(row=6, column=0, padx=10, pady=(10, 20), sticky="w")
        
        self.appearance_var = ctk.StringVar(value=self.appearance_mode)
        self.appearance_menu = ctk.CTkOptionMenu(
            self.sidebar, values=["Light", "Dark", "System"],
            variable=self.appearance_var,
            corner_radius=0, # Removed corners
            command=lambda val: self.update_theme_settings(val, "mode")
        )
        self.appearance_menu.grid(row=7, column=0, padx=10, pady=(10, 20), sticky="ew")

        # --- Color Theme (Blue, Green, etc.) ---
        color_theme_label = ctk.CTkLabel(self.sidebar, text="Color Theme:", anchor="w")
        color_theme_label.grid(row=8, column=0, padx=10, pady=(10, 20), sticky="w")

        self.color_var = ctk.StringVar(value=self.color_theme)
        self.color_menu = ctk.CTkOptionMenu(
            self.sidebar, values=["blue", "green", "dark-blue"],
            variable=self.color_var,
            corner_radius=0, # Removed corners
            command=lambda val: self.update_theme_settings(val, "theme")
        )
        self.color_menu.grid(row=9, column=0, padx=10, pady=(10, 20), sticky="ew")

        # --- UI SCALING (Dynamic 5-Step Carousel) ---
        scaling_label = ctk.CTkLabel(self.sidebar, text="UI Scaling:", anchor="w")
        scaling_label.grid(row=10, column=0, padx=10, pady=(10, 20), sticky="w")

        # 1. Get the current scaling value and ensure it's within reasonable limits
        base = int(self.scaling.replace("%", ""))
        base = max(50, min(200, base)) 

        # 2. Calculate the value of the FIRST button (scrolling window)
        start_val = max(50, min(160, base - 20))

        # 3. Generate 5 buttons from start_val upwards
        new_values = [
            f"{start_val}%", 
            f"{start_val + 10}%", 
            f"{start_val + 20}%", 
            f"{start_val + 30}%", 
            f"{start_val + 40}%"
        ]

        self.scaling_menu = ctk.CTkSegmentedButton(
            self.sidebar, 
            values=new_values,
            corner_radius=0
        )
        self.scaling_menu.grid(row=11, column=0, padx=10, pady=(10, 20), sticky="ew")
        
        # Select the correct value (even if it's not in the middle)
        self.scaling_menu.set(f"{base}%")
        self.scaling_menu.configure(command=self.change_scaling_event)

        # --- CONTENT AREA ---
        self.content_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent", corner_radius=0)
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

        # --- THE 4 MAIN TABS (These get the background colors!) ---
        tab_bg = ("gray85", "gray17")
        self.input_frame = ctk.CTkFrame(self.content_frame, fg_color=tab_bg, corner_radius=0)
        self.sep_out_frame = ctk.CTkFrame(self.content_frame, fg_color=tab_bg, corner_radius=0)
        self.trans_out_frame = ctk.CTkFrame(self.content_frame, fg_color=tab_bg, corner_radius=0)
        self.settings_frame = ctk.CTkFrame(self.content_frame, fg_color=tab_bg, corner_radius=0)

        # Initialize empty tab contents
        self.create_input_tab()
        self.create_sep_out_tab()
        self.create_trans_out_tab()
        if hasattr(self, 'create_settings_tab'):
            self.after(100, self.create_settings_tab)

        # --- PROGRESS BAR (Bottom) ---
        self.setup_progress_bar(self.main_frame)

    def _master_reload_pipeline(self):
       # 0. PREVENT SHRINKING: Hard-lock the window size BEFORE destroying anything
        current_width = self.winfo_width()
        current_height = self.winfo_height()
        
        # Force the OS to lock this geometry right now
        self.geometry(f"{current_width}x{current_height}")
        self.update_idletasks() 

        # 1. Save settings and remember current tab
        current_tab = "input"
        if hasattr(self, "sep_out_frame") and self.sep_out_frame.winfo_ismapped():
            current_tab = "sep_out"
        elif hasattr(self, "trans_out_frame") and self.trans_out_frame.winfo_ismapped():
            current_tab = "trans_out"

        # 1.5. KILL THE SIDEBAR GHOSTS (Prevents CustomTkinter OptionMenu crashes)
        # We MUST destroy the floating dropdowns before destroying the main frame!
        if hasattr(self, 'appearance_menu') and hasattr(self.appearance_menu, '_dropdown_menu'):
            try:
                if self.appearance_menu._dropdown_menu: self.appearance_menu._dropdown_menu.destroy()
            except Exception: pass
            
        if hasattr(self, 'color_menu') and hasattr(self.color_menu, '_dropdown_menu'):
            try:
                if self.color_menu._dropdown_menu: self.color_menu._dropdown_menu.destroy()
            except Exception: pass

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
            "sep_out": (self.sep_out_frame, self.sep_out_button),
            "trans_out": (self.trans_out_frame, self.trans_out_button),
            "settings": (self.settings_frame, self.settings_button)
        }
        f, b = tab_map.get(current_tab, (self.input_frame, self.input_button))
        self._switch_tab(f, b, current_tab)

        # 7. LOCK THE WINDOW SIZE (Restores the size we captured in Step 0)
        self.geometry(f"{current_width}x{current_height}")

    def create_input_tab(self):
        for widget in self.input_frame.winfo_children():
            widget.destroy()
            
        frame = self.input_frame
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=0) 
        frame.grid_rowconfigure(0, weight=0)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_rowconfigure(2, weight=0)

        # Basic Tracking Init
        if not hasattr(self, 'input_selection_dict'): self.input_selection_dict = {} # <--- ADD THIS LINE BACK
        if not hasattr(self, 'input_files'): self.input_files = []
        if not hasattr(self, 'current_pages'): self.current_pages = {}
        if "input" not in self.current_pages: self.current_pages["input"] = 0
        self.ITEMS_PER_PAGE = getattr(self, 'ITEMS_PER_PAGE_TRANS', 10)

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
        self.input_browse_btn = ctk.CTkButton(btn_frame, text="Change / New Folder", command=getattr(self, "change_input_folder", None), corner_radius=0)
        self.input_browse_btn.pack(side="left", padx=5)
        self.add_file_btn = ctk.CTkButton(btn_frame, text="Add Song", command=getattr(self, "add_song", None), corner_radius=0)
        self.add_file_btn.pack(side="right", padx=5)

        self.songs_list_frame = ctk.CTkScrollableFrame(frame, corner_radius=0)
        self.songs_list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(25, 10))

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
        sep_scrollable = ctk.CTkScrollableFrame(frame, corner_radius=0, width=200, fg_color=("gray90", "gray16"))
        sep_scrollable.grid(row=0, column=1, rowspan=3, sticky="nsew", padx=10, pady=10)
        sep_scrollable.propagate(False)
        sep_scrollable.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(sep_scrollable, text="Separation Menu", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, pady=(10,20))

        # --- Base Settings ---
        self.ai_tool_var = tk.StringVar(value="Spleeter")
        for i, tool in enumerate(["Spleeter", "Demucs", "OpenUnmix"], start=1):
            ctk.CTkRadioButton(sep_scrollable, text=tool, variable=self.ai_tool_var, value=tool, command=self.update_ui_state).grid(row=i, column=0, sticky="w", padx=10, pady=5)

        # Model Frame (Dynamic)
        self.model_frame = ctk.CTkFrame(sep_scrollable, fg_color="transparent")
        self.model_frame.grid(row=4, column=0, sticky="ew")
        ctk.CTkLabel(self.model_frame, text="Model:", anchor="w").pack(fill="x", padx=10, pady=(10,0))
        self.model_var = tk.StringVar(value="umxl")
        self.model_menu = ctk.CTkOptionMenu(self.model_frame, variable=self.model_var, corner_radius=0)
        self.model_menu.pack(fill="x", padx=10, pady=5)

        # Format 
        ctk.CTkLabel(sep_scrollable, text="Output Format:", anchor="w").grid(row=5, column=0, sticky="w", padx=10, pady=(10,0))
        self.format_var = tk.StringVar(value="wav")
        ctk.CTkOptionMenu(sep_scrollable, variable=self.format_var, corner_radius=0, values=["wav", "mp3", "flac"], command=self.update_ui_state).grid(row=6, column=0, sticky="ew", padx=10, pady=5)

        # --- Common Audio Settings ---
        common_frame = ctk.CTkFrame(sep_scrollable, fg_color="transparent")
        common_frame.grid(row=7, column=0, sticky="ew", padx=5, pady=0)
        
        channel_frame = ctk.CTkFrame(common_frame, fg_color="transparent")
        channel_frame.pack(fill="x", padx=5, pady=10)
        self.channel_var = tk.StringVar(value="Stereo")
        self.mono_label = ctk.CTkLabel(channel_frame, text="Mono", text_color="gray")
        self.mono_label.pack(side="left", padx=(0, 5))
        
        def on_channel_toggle():
            is_stereo = self.channel_switch_var.get() == "Stereo"
            self.channel_var.set("Stereo" if is_stereo else "Mono")
            self.mono_label.configure(text_color="gray" if is_stereo else ("black", "white"))
            self.stereo_label.configure(text_color=("black", "white") if is_stereo else "gray")

        self.channel_switch_var = ctk.StringVar(value="Stereo")
        ctk.CTkSwitch(channel_frame, text="", variable=self.channel_switch_var, onvalue="Stereo", offvalue="Mono", command=on_channel_toggle, width=35).pack(side="left")
        self.stereo_label = ctk.CTkLabel(channel_frame, text="Stereo", text_color=("black", "white"))
        self.stereo_label.pack(side="left", padx=(5, 0))

        ctk.CTkLabel(common_frame, text="Sample Rate (Hz):", anchor="w").pack(anchor="w", padx=5, pady=(5,0))
        self.sr_var = tk.StringVar(value="44100")
        ctk.CTkEntry(common_frame, textvariable=self.sr_var, placeholder_text="44100").pack(fill="x", padx=5, pady=5)

        # ==========================================
        # DYNAMIC SETTINGS BLOCKS
        # ==========================================

        # Bit Depth Block
        self.bit_depth_frame = ctk.CTkFrame(sep_scrollable, fg_color="transparent")
        self.bit_depth_frame.grid(row=8, column=0, sticky="ew")
        self.bit_depth_var = tk.StringVar(value="16-bit")
        ctk.CTkRadioButton(self.bit_depth_frame, text="16-bit", 
                        variable=self.bit_depth_var, value="16-bit").pack(anchor="w", padx=10, pady=2)
        ctk.CTkRadioButton(self.bit_depth_frame, text="24-bit", 
                        variable=self.bit_depth_var, value="24-bit").pack(anchor="w", padx=10, pady=2)
        ctk.CTkRadioButton(self.bit_depth_frame, text="Float32", 
                        variable=self.bit_depth_var, value="32-bit").pack(anchor="w", padx=10, pady=2)
        # FLAC Block
        self.flac_frame = ctk.CTkFrame(sep_scrollable, fg_color="transparent")
        self.flac_frame.grid(row=9, column=0, sticky="ew")
        ctk.CTkLabel(self.flac_frame, text="FLAC Compression (0-8):", anchor="w").pack(anchor="w", padx=10, pady=(5,0))
        self.flac_slider = ctk.CTkSlider(self.flac_frame, from_=0, to=8, number_of_steps=8, command=getattr(self, "update_flac_label", None))
        self.flac_slider.set(5)
        self.flac_slider.pack(fill="x", padx=10, pady=5)
        self.flac_value_label = ctk.CTkLabel(self.flac_frame, text="Current: 5", anchor="w")
        self.flac_value_label.pack(anchor="w", padx=10, pady=(0,5))

        # MP3 Bitrate Block
        self.mp3_bitrate_frame = ctk.CTkFrame(sep_scrollable, fg_color="transparent")
        self.mp3_bitrate_frame.grid(row=10, column=0, sticky="ew")
        ctk.CTkLabel(self.mp3_bitrate_frame, text="Bitrate:").pack(side="left", padx=10, pady=10)
        self.bitrate_var = tk.StringVar(value="192")
        ctk.CTkEntry(self.mp3_bitrate_frame, textvariable=self.bitrate_var, width=60).pack(side="right", padx=10, pady=10)

        # MP3 Preset Block
        self.mp3_preset_frame = ctk.CTkFrame(sep_scrollable, fg_color="transparent")
        self.mp3_preset_frame.grid(row=11, column=0, sticky="ew")
        ctk.CTkLabel(self.mp3_preset_frame, text="MP3 Preset (2=Best):", anchor="w").pack(anchor="w", padx=10, pady=(5,0))
        self.mp3_preset_slider = ctk.CTkSlider(self.mp3_preset_frame, from_=2, to=7, number_of_steps=5, command=getattr(self, "update_mp3_preset_label", None))
        self.mp3_preset_slider.set(2)
        self.mp3_preset_slider.pack(fill="x", padx=10, pady=5)
        self.mp3_preset_value_label = ctk.CTkLabel(self.mp3_preset_frame, text="Current: 2", anchor="w")
        self.mp3_preset_value_label.pack(anchor="w", padx=10, pady=(0,5))

        # Demucs Specific Block
        self.demucs_frame = ctk.CTkFrame(sep_scrollable, fg_color="transparent")
        self.demucs_frame.grid(row=12, column=0, sticky="ew")
        ctk.CTkLabel(self.demucs_frame, text="Shifts (qual/speed):", anchor="w").pack(anchor="w", padx=10, pady=(5,0))
        self.shifts_var = tk.StringVar(value="1")
        ctk.CTkEntry(self.demucs_frame, textvariable=self.shifts_var).pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(self.demucs_frame, text="Overlap (0.1 - 0.99):", anchor="w").pack(anchor="w", padx=10, pady=(10,0))
        self.overlap_slider = ctk.CTkSlider(self.demucs_frame, 
                                            from_=float(0.1), # type: ignore
                                            to=float(0.99),   # type: ignore
                                            command=getattr(self, "update_overlap_label", None))
        self.overlap_slider.set(0.25)
        self.overlap_slider.pack(fill="x", padx=10, pady=5)
        self.overlap_value_label = ctk.CTkLabel(self.demucs_frame, text="Current: 0.25", anchor="w")
        self.overlap_value_label.pack(anchor="w", padx=10, pady=(0,5))

        # Execute visibility logic right away to hide what shouldn't be seen on launch
        self.update_ui_state()

        # Separate Button
        self.separate_button = ctk.CTkButton(sep_scrollable, text="Start Batch Separation", height=40, corner_radius=0, font=ctk.CTkFont(weight="bold"), command=self.separate_audio)
        self.separate_button.grid(row=13, column=0, sticky="ew", padx=10, pady=(30,10))
        ctk.CTkLabel(sep_scrollable, text="Turn ON switches \nnext to input songs\nto separate audio.", font=ctk.CTkFont(size=11, slant="italic")).grid(row=14, column=0, padx=10)

    def create_sep_out_tab(self):
        for widget in getattr(self, "sep_out_frame", self.main_frame).winfo_children():
            widget.destroy()

        frame = self.sep_out_frame
        
        # Configure Columns: Left side expands (Lists), Right side stays fixed (Menu)
        frame.grid_columnconfigure(0, weight=1) 
        frame.grid_columnconfigure(1, weight=0) 
        
        # Configure Rows: Row 1 (Vocals) and Row 4 (Instrumentals) will expand equally
        frame.grid_rowconfigure(1, weight=1)  
        frame.grid_rowconfigure(4, weight=1)  

        # --- PAGINATION TRACKERS ---
        if not hasattr(self, 'current_pages'):
            self.current_pages = {"trans": 0, "vocals": 0, "instr": 0}
        
        self.ITEMS_PER_PAGE = getattr(self, 'ITEMS_PER_PAGE_SEP', 10)
        self.vocals_selection_dict = {}
        self.instr_selection_dict = {}

        # ==========================================
        # LEFT COLUMN (TOP): VOCALS
        # ==========================================
        header_frame2 = ctk.CTkFrame(frame, fg_color="transparent")
        header_frame2.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        ctk.CTkLabel(header_frame2, text="Vocals", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
        ctk.CTkButton(header_frame2, text="Change Folder", width=100, corner_radius=0, command=lambda: self.change_output_folder("vocals")).pack(side="right")
        
        self.vocals_list_frame = ctk.CTkScrollableFrame(frame, corner_radius=0)        
        self.vocals_list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 5))

        self.vocals_page_frame = ctk.CTkFrame(frame, fg_color="transparent", height=30)
        self.vocals_page_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 15))
        self.vocals_btn_prev = ctk.CTkButton(self.vocals_page_frame, text="<", width=30, corner_radius=0, command=lambda: self.change_page("vocals", -1))
        self.vocals_btn_prev.pack(side="left")
        self.vocals_lbl_page = ctk.CTkLabel(self.vocals_page_frame, text="Page 1 of 1", font=ctk.CTkFont(weight="bold"))
        self.vocals_lbl_page.pack(side="left", expand=True)
        self.vocals_btn_next = ctk.CTkButton(self.vocals_page_frame, text=">", width=30, corner_radius=0, command=lambda: self.change_page("vocals", 1))
        self.vocals_btn_next.pack(side="right")

        # ==========================================
        # LEFT COLUMN (BOTTOM): INSTRUMENTALS
        # ==========================================
        header_frame3 = ctk.CTkFrame(frame, fg_color="transparent")
        header_frame3.grid(row=3, column=0, sticky="ew", padx=10, pady=(10, 0))
        ctk.CTkLabel(header_frame3, text="Instrumentals", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
        ctk.CTkButton(header_frame3, text="Change Folder", width=100, corner_radius=0, command=lambda: self.change_output_folder("instrumentals")).pack(side="right")
        
        self.instr_list_frame = ctk.CTkScrollableFrame(frame, corner_radius=0)
        self.instr_list_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=(5, 5))

        self.instr_page_frame = ctk.CTkFrame(frame, fg_color="transparent", height=30)
        self.instr_page_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=(0, 15))
        self.instr_btn_prev = ctk.CTkButton(self.instr_page_frame, text="<", width=30, corner_radius=0, command=lambda: self.change_page("instr", -1))
        self.instr_btn_prev.pack(side="left")
        self.instr_lbl_page = ctk.CTkLabel(self.instr_page_frame, text="Page 1 of 1", font=ctk.CTkFont(weight="bold"))
        self.instr_lbl_page.pack(side="left", expand=True)
        self.instr_btn_next = ctk.CTkButton(self.instr_page_frame, text=">", width=30, corner_radius=0, command=lambda: self.change_page("instr", 1))
        self.instr_btn_next.pack(side="right")

        # ==========================================
        # RIGHT COLUMN: TRANSCRIPTION MENU
        # ==========================================
        # We set rowspan=6 so it stretches alongside both lists beautifully
        trans_menu = ctk.CTkScrollableFrame(frame, width=250, corner_radius=0, fg_color=("gray90", "gray16"))
        trans_menu.grid(row=0, column=1, rowspan=6, sticky="nsew", padx=10, pady=10) 
        trans_menu.grid_columnconfigure(0, weight=1)
        trans_menu.propagate(False)

        ctk.CTkLabel(trans_menu, text="Transcription Menu", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, pady=(10, 20))
        
        ctk.CTkLabel(trans_menu, text="Tool:", anchor="w").grid(row=1, column=0, sticky="w", padx=10)
        self.trans_tool_var = tk.StringVar(value="whisper")
        
        ctk.CTkRadioButton(trans_menu, text="Whisper", variable=self.trans_tool_var, value="whisper", command=getattr(self, "on_trans_tool_change", None)).grid(row=2, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkRadioButton(trans_menu, text="Wav2Vec2", variable=self.trans_tool_var, value="wav2vec2", command=getattr(self, "on_trans_tool_change", None)).grid(row=3, column=0, sticky="w", padx=10, pady=5)
        ctk.CTkRadioButton(trans_menu, text="Vosk", variable=self.trans_tool_var, value="vosk", command=getattr(self, "on_trans_tool_change", None)).grid(row=4, column=0, sticky="w", padx=10, pady=5)

        self.trans_model_label = ctk.CTkLabel(trans_menu, text="Model:", anchor="w")
        self.trans_model_label.grid(row=5, column=0, sticky="w", padx=10, pady=(10, 0))
        
        self.trans_model_var = tk.StringVar()
        self.trans_model_menu = ctk.CTkOptionMenu(trans_menu, variable=self.trans_model_var, corner_radius=0, values=[])
        self.trans_model_menu.grid(row=6, column=0, sticky="ew", padx=10, pady=5)

        self.trans_lang_label = ctk.CTkLabel(trans_menu, text="Language:", anchor="w")
        self.trans_lang_label.grid(row=7, column=0, sticky="w", padx=10, pady=(10, 0))
        
        self.trans_lang_var = tk.StringVar(value="auto")
        self.trans_lang_menu = ctk.CTkOptionMenu(trans_menu, variable=self.trans_lang_var, corner_radius=0, values=["auto", "cs", "en", "fr", "de", "es"])
        self.trans_lang_menu.grid(row=8, column=0, sticky="ew", padx=10, pady=5)

        self.use_spk_id_var = tk.BooleanVar(value=False)
        self.spk_toggle = ctk.CTkSwitch(trans_menu, text="Identify Speakers", variable=self.use_spk_id_var, progress_color="#1f538d")
        self.spk_toggle.grid(row=9, column=0, sticky="w", padx=10, pady=10)
        self.spk_toggle.grid_remove() 
        
        self.trans_button = ctk.CTkButton(trans_menu, text="Transcribe", height=40, font=ctk.CTkFont(weight="bold"), command=getattr(self, "run_standalone_transcription", None), corner_radius=0)
        self.trans_button.grid(row=10, column=0, sticky="ew", padx=10, pady=(30, 10))
        
        ctk.CTkLabel(trans_menu, text="Turn ON switches in\nVocals or Instrumentals\nto process.", font=ctk.CTkFont(size=11, slant="italic")).grid(row=11, column=0, padx=10)

        if hasattr(self, 'on_trans_tool_change'): self.on_trans_tool_change()

    def create_trans_out_tab(self):
        for widget in getattr(self, "trans_out_frame", self.main_frame).winfo_children():
            widget.destroy()

        frame = self.trans_out_frame
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1) # Expand the scrollable list

        # --- PAGINATION TRACKERS ---
        if not hasattr(self, 'current_pages'):
            self.current_pages = {"trans": 0}
            
        self.ITEMS_PER_PAGE = getattr(self, 'ITEMS_PER_PAGE_TRANS', 10)
        self.trans_selection_dict = {}

        # ==========================================
        # FULL WIDTH: TRANSCRIPTIONS LIST
        # ==========================================
        header_frame1 = ctk.CTkFrame(frame, fg_color="transparent")
        header_frame1.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 0))
        ctk.CTkLabel(header_frame1, text="Transcriptions", font=ctk.CTkFont(size=18, weight="bold")).pack(side="left")
        ctk.CTkButton(header_frame1, text="Change Folder", width=100, corner_radius=0, command=lambda: self.change_output_folder("transcriptions")).pack(side="right")
        
        self.trans_list_frame = ctk.CTkScrollableFrame(frame, corner_radius=0)
        self.trans_list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(5, 5))

        self.trans_page_frame = ctk.CTkFrame(frame, fg_color="transparent", height=30)
        self.trans_page_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 15))
        self.trans_btn_prev = ctk.CTkButton(self.trans_page_frame, text="<", width=30, corner_radius=0, command=lambda: self.change_page("trans", -1))
        self.trans_btn_prev.pack(side="left")
        self.trans_lbl_page = ctk.CTkLabel(self.trans_page_frame, text="Page 1 of 1", font=ctk.CTkFont(weight="bold"))
        self.trans_lbl_page.pack(side="left", expand=True)
        self.trans_btn_next = ctk.CTkButton(self.trans_page_frame, text=">", width=30, corner_radius=0, command=lambda: self.change_page("trans", 1))
        self.trans_btn_next.pack(side="right")

    def create_file_row(self, parent_frame, file_name, file_path, item_idx, selection_dict, is_folder=False):
        is_txt = file_name.lower().endswith('.txt')

        is_input_tab = (parent_frame == self.songs_list_frame) 

        # 1. Flat base 
        row_frame = ctk.CTkFrame(parent_frame, corner_radius=0, fg_color=("gray85", "gray20"))
        row_frame.pack(fill="x", padx=2, pady=2)
        
        # 2. Selection checkbox
        if not is_txt:
            if item_idx not in selection_dict:
                var = tk.BooleanVar(value=False)
                selection_dict[item_idx] = {
                    "var": var,
                    "type": 'folder' if is_folder else 'song',
                    "data": {'name': file_name, 'path': file_path} 
                }
            chk = ctk.CTkCheckBox(row_frame, text="", variable=selection_dict[item_idx]["var"], width=24, corner_radius=0)
            chk.pack(side="left", padx=(5, 5))
            
        # 3. Action Buttons
        if is_folder:
            open_btn = ctk.CTkButton(row_frame, text="Open", width=65, corner_radius=0, 
                                     command=lambda p=file_path: self.enter_folder(p))
            open_btn.pack(side="left", padx=(0, 5))
        elif is_txt:
            read_btn = ctk.CTkButton(row_frame, text="📖", width=54, corner_radius=0,
                                     command=lambda p=file_path: self.play_audio(p))
            read_btn.pack(side="left", padx=(0, 5))
        else:
            play_btn = ctk.CTkButton(row_frame, text="▶", width=30, corner_radius=0,
                                     command=lambda p=file_path: self.play_audio(p))
            play_btn.pack(side="left", padx=(0, 5))
            
            info_btn = ctk.CTkButton(row_frame, text="\u2139", width=30, corner_radius=0,
                                     command=lambda p=file_path, n=file_name: self.show_audio_info(p, n, show_sync_controls=is_input_tab))
            info_btn.pack(side="left", padx=(0, 5))

        # 4. File icon and name
        if is_folder: display_text = f"📁 {file_name}"
        elif is_txt: display_text = f"📝 {file_name}"
        else: display_text = f"🎵 {file_name}"
            
        lbl = ctk.CTkLabel(row_frame, text=display_text, anchor="w")
        lbl.pack(side="left", fill="x", expand=True, padx=5) 
        
        # 5. Delete button
        if not is_folder:
            del_btn = ctk.CTkButton(row_frame, text="🗑", width=30, corner_radius=0, fg_color="#a83232", hover_color="#8a2929",
                                    command=lambda p=file_path: self.delete_file(p))
            del_btn.pack(side="right", padx=5)

        # 6. Stats Label
        if is_folder:
            stats_lbl = ctk.CTkLabel(row_frame, text="Calculating...", text_color="gray", font=ctk.CTkFont(size=12))
            stats_lbl.pack(side="right", padx=(10, 15))
            self.calculate_folder_size_async(file_path, stats_lbl) 
        else:
            stats_text = self.get_file_stats(file_path, is_folder, is_txt)
            stats_lbl = ctk.CTkLabel(row_frame, text=stats_text, text_color="gray", font=ctk.CTkFont(size=12))
            stats_lbl.pack(side="right", padx=(10, 15))

    def get_file_stats(self, file_path, is_folder, is_txt):
        """Returns a formatted string with duration (if audio) and file size."""
        if is_folder:
            return "Folder" # Keeping it safe and fast!
            
        try:
            # 1. Calculate File Size
            size_bytes = os.path.getsize(file_path)
            if size_bytes == 0:
                size_str = "0 B"
            else:
                size_name = ("B", "KB", "MB", "GB", "TB")
                i = int(math.floor(math.log(size_bytes, 1024)))
                p = math.pow(1024, i)
                
                # NEW: Drop the decimal if the unit is Bytes (i == 0)
                if i == 0:
                    size_str = f"{int(size_bytes)} B"
                else:
                    s = round(size_bytes / p, 1)
                    size_str = f"{s} {size_name[i]}"
                
            # 2. Calculate Audio Duration (Skip if it's a text document)
            if not is_txt:
                try:
                    import torchaudio
                    # torchaudio.info is very fast as it only reads the metadata header
                    metadata = torchaudio.info(file_path)
                    seconds = int(metadata.num_frames / metadata.sample_rate)
                    mins, secs = divmod(seconds, 60)
                    duration_str = f"{mins}:{secs:02d}"
                    return f"{duration_str}  •  {size_str}"
                except Exception:
                    pass # If torchaudio fails (e.g. unsupported format), just fall back to size

            return size_str
        except Exception:
            return "Unknown"
        
    def show_audio_info(self, file_path, file_name, show_sync_controls=True):
        """Displays advanced technical metadata with grid alignment and safe syncing."""
        try:
            import torchaudio
            import os
            metadata = torchaudio.info(file_path)
            
            sr = metadata.sample_rate
            channels = metadata.num_channels
            frames = metadata.num_frames
            bits = getattr(metadata, 'bits_per_sample', 0) 
            
            channel_str = "Stereo" if channels == 2 else "Mono" if channels == 1 else f"{channels} Ch"
            ext = os.path.splitext(file_path)[1].lower().replace(".", "")
            
            # Calculate Duration (Needed for Bitrate math)
            duration_sec = frames / sr if sr > 0 else 0
            
            # Calculate Bitrate (kbps) = (File Size in bits) / (Duration in seconds * 1000)
            if duration_sec > 0:
                file_size_bytes = os.path.getsize(file_path)
                bitrate_kbps = int((file_size_bytes * 8) / (duration_sec * 1000))
                
                # Round to nearest common standard (optional, but looks cleaner)
                standard_bitrates = [64, 96, 128, 192, 256, 320]
                closest_bitrate = min(standard_bitrates, key=lambda x: abs(x - bitrate_kbps))
                
                # If it's VBR (Variable Bitrate) it might be an odd number, so we use the raw calculation if it's far off
                if abs(closest_bitrate - bitrate_kbps) < 15:
                    bitrate_kbps = closest_bitrate
                    
                bitrate_str = f"{bitrate_kbps} kbps"
            else:
                bitrate_kbps = 0
                bitrate_str = "Unknown"
            
            bit_depth_str = f"{bits}-bit" if bits > 0 else "N/A"
            
        except Exception as e:
            logging.info(f"Could not read metadata: {e}")
            return

        # --- Pop-up Dialog ---
        dialog = ctk.CTkToplevel(self)
        dialog.title("Audio Inspector")
        dialog.geometry("380x280") 
        dialog.attributes("-topmost", True)
        
        content = ctk.CTkFrame(dialog, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=20, pady=20)
        
        ctk.CTkLabel(content, text=file_name, font=ctk.CTkFont(weight="bold", size=16), wraplength=340).pack(pady=(0, 15))
        
        # --- Grid Layout for Perfect Alignment ---
        grid_frame = ctk.CTkFrame(content, fg_color="transparent")
        grid_frame.pack(fill="x")
        
        # Safe sync functions
        def sync_format():
            if hasattr(self, 'format_var') and ext in ["wav", "mp3", "flac"]:
                self.format_var.set(ext)
                if hasattr(self, 'update_ui_state'): 
                    self.update_ui_state()

        def sync_sample_rate():
            if hasattr(self, 'sr_var'):
                self.sr_var.set(str(sr))

        def sync_channels():
            if hasattr(self, 'channel_switch_var') and channel_str in ["Mono", "Stereo"]:
                self.channel_switch_var.set(channel_str)
                if channel_str == "Stereo":
                    self.mono_label.configure(text_color="gray")
                    self.stereo_label.configure(text_color=("black", "white"))
                else:
                    self.mono_label.configure(text_color=("black", "white"))
                    self.stereo_label.configure(text_color="gray")

        def sync_bit_depth(bits):
            """Updates the UI radio buttons based on detected file bit depth."""
            if hasattr(self, 'bit_depth_var'):
                if bits == 16:
                    self.bit_depth_var.set("16-bit")
                elif bits == 24:
                    self.bit_depth_var.set("24-bit")
                elif bits == 32:
                    self.bit_depth_var.set("32-bit")

        def sync_bitrate():
            if hasattr(self, 'bitrate_var') and bitrate_kbps > 0:
                self.bitrate_var.set(str(bitrate_kbps))

        def sync_all():
            sync_format()
            sync_sample_rate()
            sync_channels()
            if ext == "mp3":
                sync_bitrate()
            else:
                sync_bit_depth(bits)
            sync_all_btn.configure(text="All Synced! ✓", fg_color="#2b7a4b")

        # Define our base rows
        details = [
            ("Format:", ext.upper(), sync_format),
            ("Sample Rate:", f"{sr} Hz", sync_sample_rate),
            ("Channels:", channel_str, sync_channels)
        ]
        
        # Dynamically append Bitrate OR Bit Depth based on format!
        if ext == "mp3":
            details.append(("Bitrate:", bitrate_str, sync_bitrate if bitrate_kbps > 0 else None))
        else:
            details.append(("Bit Depth:", bit_depth_str, sync_bit_depth if bits in [16, 24, 32] else None))
        
        # Build the grid
        for i, (lbl_text, val_text, sync_cmd) in enumerate(details):
            ctk.CTkLabel(grid_frame, text=lbl_text, text_color="gray").grid(row=i, column=0, sticky="e", padx=(0, 10), pady=5)
            ctk.CTkLabel(grid_frame, text=val_text, font=ctk.CTkFont(weight="bold")).grid(row=i, column=1, sticky="w", padx=(0, 15), pady=5)
            
            if show_sync_controls and sync_cmd:
                ctk.CTkButton(grid_frame, text="Sync", width=40, height=24, command=sync_cmd).grid(row=i, column=2, padx=5, pady=5)
                
        grid_frame.grid_columnconfigure(1, weight=1)

        # Main Sync All Button 
        if show_sync_controls:
            sync_all_btn = ctk.CTkButton(dialog, text="Sync All to Output", command=sync_all)
            sync_all_btn.pack(side="bottom", pady=10, padx=20, fill="x")

    def calculate_folder_size_async(self, folder_path, label_widget):
        """Calculates folder size safely in a background thread."""
        def calc_size():
            total_size = 0
            try:
                for dirpath, _, filenames in os.walk(folder_path):
                    for f in filenames:
                        fp = os.path.join(dirpath, f)
                        if not os.path.islink(fp):
                            total_size += os.path.getsize(fp)
                
                if total_size == 0:
                    size_str = "0 B"
                else:
                    size_name = ("B", "KB", "MB", "GB", "TB")
                    i = int(math.floor(math.log(total_size, 1024)))
                    p = math.pow(1024, i)
                    s = round(total_size / p, 1)
                    size_str = f"{s} {size_name[i]}"
                
                # SAFE WAY: Put the result in the queue instead of touching the UI directly
                self.folder_size_queue.put((label_widget, f"{size_str}"))
            except Exception:
                self.folder_size_queue.put((label_widget, "Size Unknown"))

        threading.Thread(target=calc_size, daemon=True).start()

    def _check_size_queue(self):
        """Safely updates the UI on the main thread without crashing."""
        try:
            # Process everything currently in the queue
            while True:
                label_widget, text = self.folder_size_queue.get_nowait()
                label_widget.configure(text=text)
        except queue.Empty:
            pass
        finally:
            # Check the queue again in 100 milliseconds
            self.after(100, self._check_size_queue)

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
            logging.info(f"Error playing file: {e}")

    def delete_file(self, file_path, tab_to_reload="input"):
        """Deletes a file directly from the app and refreshes the UI."""
        file_name = os.path.basename(file_path)
        if messagebox.askyesno("Delete File", f"Are you sure you want to permanently delete:\n{file_name}?"):
            try:
                os.remove(file_path)
                
                # Refresh the correct tab so the deleted file disappears from the list
                if tab_to_reload == "input":
                    self.load_input()
                elif tab_to_reload == "sep_out":
                    self.load_separation_outputs()
                elif tab_to_reload == "trans_out":
                    self.load_transcription_outputs()
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete file: {e}")

    def create_settings_tab(self):
        """!
        @brief Populates the Settings tab with directory configurations, a Model Manager, 
               and system resource controls.
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

        # Setup StringVars 
        self.settings_models_var = tk.StringVar(value=self.models_dir)
        self.settings_input_var = tk.StringVar(value=self.input_folder)
        self.settings_vocals_var = tk.StringVar(value=self.output_folders["vocals"])
        self.settings_instr_var = tk.StringVar(value=self.output_folders["instrumentals"])
        self.settings_trans_var = tk.StringVar(value=self.output_folders["transcriptions"])

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
            
            btn = ctk.CTkButton(row_frame, text="Browse", width=70, corner_radius=0,
                                command=lambda v=var: self._browse_folder(v))
            btn.pack(side="right", padx=(10, 0))
            
            if text == "AI Models Folder:": self.models_browse_btn = btn
            elif text == "Input Folder:": self.input_browse_btn = btn
            elif text == "Vocals Folder:": self.vocals_browse_btn = btn
            elif text == "Instrumentals:": self.instr_browse_btn = btn
            elif text == "Transcriptions:": self.trans_browse_btn = btn

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

        models_setup = [
            ("Demucs:", "demucs", self.separator_models.get("Demucs", []), "High quality separation (Downloads automatically on first use)"),
            ("OpenUnmix:", "openunmix", self.separator_models.get("OpenUnmix", []), "Alternative separation models"),
            ("Whisper:", "whisper", self.transcription_models.get("whisper", []), "Accurate transcription (Downloads automatically on first use)"),
            ("Wav2Vec2:", "wav2vec2", self.transcription_models.get("wav2vec2", []), "Fast transcription"),
            ("Vosk:", "vosk", self.transcription_models.get("vosk", []), "Fast offline transcription (Requires manual download or scanning)")
        ]

        for text, dict_key, model_list, desc in models_setup:
            row_frame = ctk.CTkFrame(mod_frame, fg_color="transparent")
            row_frame.pack(fill="x", padx=15, pady=5)
            
            self.model_vars[dict_key] = tk.StringVar(value=list_to_csv(model_list))
            
            input_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
            input_frame.pack(side="top", fill="x")
            
            ctk.CTkLabel(input_frame, text=text, width=100, anchor="w").pack(side="left")
            ctk.CTkEntry(input_frame, textvariable=self.model_vars[dict_key]).pack(side="left", fill="x", expand=True, padx=(5, 5))
            
            ctk.CTkLabel(row_frame, text=desc, text_color="gray", font=ctk.CTkFont(size=10)).pack(side="top", anchor="w", padx=(105, 0))

        # Action Buttons for Models
        action_mod_frame = ctk.CTkFrame(mod_frame, fg_color="transparent")
        action_mod_frame.pack(fill="x", padx=15, pady=15)

        # --- CHANGED: Assigned the download button to a variable ---
        self.download_models_btn = ctk.CTkButton(action_mod_frame, text="Download Default Models", width=180,
                                                 command=self.download_default_models)
        self.download_models_btn.pack(side="left", padx=(0, 10))
                      
        self.scan_models_btn = ctk.CTkButton(action_mod_frame, text="Scan availeble Transciption models", width=120,
                                             command=self.scan_models_directory)
        self.scan_models_btn.pack(side="left", padx=(0, 10))
        
        self.import_custom_model_btn = ctk.CTkButton(action_mod_frame, text="Import Custom Model", width=120,
                                                     command=self.import_custom_model)
        self.import_custom_model_btn.pack(side="left", padx=(0, 10))

        # ==========================================
        # SECTION 3: SYSTEM ACTIONS & MEMORY
        # ==========================================
        action_frame = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        action_frame.pack(fill="x", padx=10, pady=(10, 20))
        
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

        button_row = ctk.CTkFrame(scroll_frame, fg_color="transparent")
        button_row.pack(fill="x", pady=(15, 0))
        
        ctk.CTkButton(button_row, text="Show Tutorial", height=40, corner_radius=0,
                      command=self.show_welcome_tutorial).pack(side="left", padx=(0, 5))
        
        # Original Save and Restore buttons
        ctk.CTkButton(button_row, text="Restore Defaults", height=40, fg_color="transparent", corner_radius=0, 
                      border_width=1, text_color=("gray10", "gray90"), 
                      command=self.restore_defaults).pack(side="left", padx=5)
        
        ctk.CTkButton(button_row, text="Save Settings", height=40, font=ctk.CTkFont(weight="bold"), corner_radius=0, 
                      command=self.save_settings_changes).pack(side="left", padx=(10, 0))
         
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
            logging.info(f"Settings file corrupted: {e}. Restoring defaults.")
            self._apply_dict_to_state(defaults)
            self.save_settings()

    def _apply_dict_to_state(self, data):
        """Maps dictionary values from settings to the app's internal variables."""
        
        # 1. Directories
        self.input_folder = data.get("input_folder", "input")
        self.models_dir = data.get("models_dir", "") 
        
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

        # 5. First Run Flag
        self.is_first_run = data.get("is_first_run", True)

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
        
        # ---> ADD THIS: Capture the updated models directory from your UI
        # (Assuming you named the UI variable `self.settings_models_var`)
        if hasattr(self, 'settings_models_var'):
            self.models_dir = self.settings_models_var.get()
        
        # ---> ADD THIS: Update the environment variables instantly for lazy-loaded modules
        os.environ["HF_HOME"] = os.path.join(self.models_dir, "huggingface")
        os.environ["HF_HUB_CACHE"] = os.path.join(self.models_dir, "huggingface", "hub")
        os.environ["TORCH_HOME"] = os.path.join(self.models_dir, "hub")
        os.environ["MODEL_PATH"] = self.models_dir

        # ---> ADD THIS: Even with lazy loading, if they already ran a task this session, 
        # the class exists. We need to update its internal path.
        whisper = getattr(self, 'whisper_trans', None)
        if whisper is not None:
            whisper.models_dir = self.models_dir
        vosk = getattr(self, 'vosk_trans', None)
        if vosk is not None:
            vosk.models_dir = self.models_dir
        
        # Save the current state of the Auto-Flush switch
        self.auto_flush_memory = bool(self.flush_switch.get())
        
        # Apply model variables by parsing the CSV strings
        self.separator_models["Demucs"] = parse_csv(self.model_vars["demucs"])
        self.separator_models["OpenUnmix"] = parse_csv(self.model_vars["openunmix"])
        self.transcription_models["whisper"] = parse_csv(self.model_vars["whisper"])
        self.transcription_models["wav2vec2"] = parse_csv(self.model_vars["wav2vec2"])
        self.transcription_models["vosk"] = parse_csv(self.model_vars["vosk"])

        # Ensure folders physically exist
        os.makedirs(self.models_dir, exist_ok=True) # Ensure the models folder exists too!
        os.makedirs(self.input_folder, exist_ok=True)
        for folder in self.output_folders.values():
            os.makedirs(folder, exist_ok=True)

        # Write the updated variables to the settings.json file
        self.save_settings()
        
        messagebox.showinfo("Saved", "Settings successfully updated!")

    def save_settings(self):
        """Writes current state to disk."""
        data = {
            "is_first_run": getattr(self, "is_first_run", False), # <-- FIXED: Now saves the first-run flag
            "models_dir": getattr(self, "models_dir", ""),        # <-- FIXED: Now saves the models directory
            "input_folder": self.input_folder,
            "vocals_folder": self.output_folders["vocals"],
            "instrumentals_folder": self.output_folders["instrumentals"],
            "transcriptions_folder": self.output_folders["transcriptions"],
            "appearance_mode": getattr(self, "appearance_mode", "Dark"),
            "color_theme": getattr(self, "color_theme", "blue"),
            "scaling": getattr(self, "scaling", "100%"),
            "auto_flush_memory": getattr(self, "auto_flush_memory", True), 
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
            "is_first_run": True,                 
            "models_dir": "",                      
            "input_folder": "input",
            "vocals_folder": "output/vocals",
            "instrumentals_folder": "output/instrumentals",
            "transcriptions_folder": "output/text",
            "appearance_mode": "Dark",
            "color_theme": "blue", 
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
    
    def update_mp3_preset_label(self, value):
        self.mp3_preset_value_label.configure(text=f"Current: {int(value)}")

    def update_flac_label(self, value):
        if hasattr(self, 'flac_value_label'):
            self.flac_value_label.configure(text=f"Current: {int(value)}")

    def update_overlap_label(self, value):
        if hasattr(self, 'overlap_value_label'):
            self.overlap_value_label.configure(text=f"Current: {value:.2f}")

    def update_ui_state(self, *args):
        if not hasattr(self, 'model_frame'): 
            return # Safety check: wait until UI is drawn

        tool = self.ai_tool_var.get()
        fmt = self.format_var.get()

        # 1. Update Model Options dynamically based on the tool
        if tool != "Spleeter":
            default_models = ["mdx", "mdx_extra", "htdemucs"] if tool == "Demucs" else ["umxl", "umxhq", "umx"]
            values = getattr(self, "separator_models", {}).get(tool, default_models) or default_models
            self.model_menu.configure(values=values)
            if self.model_var.get() not in values: 
                self.model_var.set(values[0])

        # 2. Define UI Visibility Rules (True = Show, False = Hide)
        visibility_rules = {
            self.model_frame: tool != "Spleeter",
            self.bit_depth_frame: tool == "Demucs" and fmt in ["wav", "flac"],
            self.flac_frame: fmt == "flac",
            self.mp3_bitrate_frame: fmt == "mp3",
            self.mp3_preset_frame: tool == "Demucs" and fmt == "mp3",
            self.demucs_frame: tool == "Demucs"
        }

        # 3. Apply all rules instantly
        for frame, should_show in visibility_rules.items():
            frame.grid() if should_show else frame.grid_remove()

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

    def load_separation_outputs(self):
        """Only loads Vocals and Instrumentals."""
        self.vocals_selection_dict.clear()
        self.instr_selection_dict.clear()
        self.vocals.clear()
        self.instrumentals.clear()

        # Load Vocals
        vocals_dir = self.output_folders.get("vocals", "")
        if os.path.isdir(vocals_dir):
            for f in sorted(os.listdir(vocals_dir)):
                if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                    self.vocals.append({'path': os.path.join(vocals_dir, f), 'name': f})

        # Load Instrumentals
        instr_dir = self.output_folders.get("instrumentals", "")
        if os.path.isdir(instr_dir):
            for f in sorted(os.listdir(instr_dir)):
                if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                    self.instrumentals.append({'path': os.path.join(instr_dir, f), 'name': f})

        # Reset pagination and draw UI
        self.current_pages["vocals"] = 0
        self.current_pages["instr"] = 0
        
        self.after(10, lambda: self.render_page("vocals"))
        self.after(50, lambda: self.render_page("instr"))

    def load_transcription_outputs(self):
        """Only loads Transcriptions."""
        self.trans_selection_dict.clear()
        self.transcriptions.clear()

        # Load Transcriptions
        trans_dir = self.output_folders.get("transcriptions", "")
        if os.path.isdir(trans_dir):
            for f in sorted(os.listdir(trans_dir)):
                if f.lower().endswith(('.txt', '.lrc')):
                    self.transcriptions.append({'path': os.path.join(trans_dir, f), 'name': f})

        # Reset pagination and draw UI
        self.current_pages["trans"] = 0
        
        self.after(10, lambda: self.render_page("trans"))

    def change_page(self, category: str, delta: int):
        """Triggered by the < and > buttons to change the page number."""
        self.current_pages[category] += delta
        self.render_page(category)

    def render_page(self, category: str):
        self.progress_text.configure(text="Loading library and scanning files...")
        self.update_idletasks() # Force UI to show this text

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

        # Silently abort rendering this list if we are on a different tab and it doesn't exist
        if list_frame is None or not list_frame.winfo_exists():
            return  

        # Clear the frame to draw the new items
        for widget in list_frame.winfo_children():
            widget.destroy()

        # --- 1. CORE MATH ---
        # Use the recently calculated value
        items_per_page = getattr(self, "ITEMS_PER_PAGE", 10)
        total_items = len(data_list)

        # Calculate total pages based on current capacity
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
        
        # Safely get current page
        if not hasattr(self, 'current_pages'): 
            self.current_pages = {}
            
        # Clamp the current page so it never goes out of bounds
        curr_page = max(0, min(self.current_pages.get(category, 0), total_pages - 1))
        self.current_pages[category] = curr_page

        # --- 2. DRAW ROWS ---
        start_idx = curr_page * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)

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

        # --- 3. UPDATE UI LABELS & BUTTONS ---
        if self.ITEMS_PER_PAGE == 9999:
            # FULL SCREEN: Hide buttons and show total count
            lbl_page.configure(text=f"Loaded {total_items} files")
            btn_prev.pack_forget() 
            btn_next.pack_forget()
        else:
            # NORMAL WINDOW: Show buttons, update text, and configure button states
            lbl_page.configure(text=f"Page {curr_page + 1} of {total_pages}")
            
            # Pack the buttons back onto the screen
            btn_prev.pack(side="left", padx=10) 
            btn_next.pack(side="right", padx=10)
            
            # Enable/Disable buttons based on the current page
            btn_prev.configure(state="normal" if curr_page > 0 else "disabled")
            btn_next.configure(state="normal" if curr_page < total_pages - 1 else "disabled")
            
    def _on_window_configure(self, event):
        """Filters window events so we only rebuild UI when resizing STOPS."""
        # We ONLY care about the main application window resizing
        if event.widget != self:
            return

        current_width = self.winfo_width()
        current_height = self.winfo_height()

        # GUARD: Did the window actually change size? 
        # (If it only moved X/Y coordinates, do absolutely nothing!)
        if current_width == self._last_width and current_height == self._last_height:
            return

        # Update our trackers
        self._last_width = current_width
        self._last_height = current_height

        # --- THE DEBOUNCER ---
        # Cancel the previous timer if the user is still actively dragging the edge
        if self._resize_timer is not None:
            self.after_cancel(self._resize_timer)

        # Set a new timer. If the user stops dragging for 250 milliseconds, trigger the rebuild!
        self._resize_timer = self.after(250, self._recalculate_pagination)

    def _recalculate_pagination(self, is_initialization=False):
        """Dynamically calculates ITEMS_PER_PAGE based on window height."""
        is_fullscreen = self.state() == 'zoomed'
        window_height = self.winfo_height()
        
        # Fallback for startup
        if window_height < 100: window_height = 800 
        row_height = 35 

        if is_fullscreen:
            # Force reset to page 0 so items aren't hidden on a "ghost" page
            self.current_pages["input"] = 0
            self.current_pages["vocals"] = 0
            self.current_pages["instr"] = 0
            self.current_pages["trans"] = 0
            # In fullscreen mode, we can show all items without pagination
            self.ITEMS_PER_PAGE_SEP = 9999
            self.ITEMS_PER_PAGE_TRANS = 9999
            self.ITEMS_PER_PAGE_INPUT = 9999
        else:
            # --- NORMAL WINDOW MATH ---
            window_height = self.winfo_height()
            if window_height < 100: window_height = 800 
            row_height = 38 

            # Separation Tab (Vocals/Instr stacked)
            avail_sep = (window_height - 240) / 2 
            self.ITEMS_PER_PAGE_SEP = max(4, int(avail_sep / row_height))

            # Transcription/Input Tab (One tall list)
            avail_input = window_height - 200
            self.ITEMS_PER_PAGE_INPUT = max(6, int(avail_input / row_height))
            self.ITEMS_PER_PAGE_TRANS = self.ITEMS_PER_PAGE_INPUT

        if is_initialization: return

        # --- APPLY CHANGES TO ACTIVE TAB ---
        # Determine which tab is currently visible and update it
        if hasattr(self, 'input_frame') and self.input_frame.winfo_ismapped():
            self.ITEMS_PER_PAGE = self.ITEMS_PER_PAGE_INPUT
            self.render_page("input")
            
        elif hasattr(self, 'sep_out_frame') and self.sep_out_frame.winfo_ismapped():
            self.ITEMS_PER_PAGE = self.ITEMS_PER_PAGE_SEP
            self.render_page("vocals")
            self.render_page("instr")
            
        elif hasattr(self, 'trans_out_frame') and self.trans_out_frame.winfo_ismapped():
            self.ITEMS_PER_PAGE = self.ITEMS_PER_PAGE_TRANS
            self.render_page("trans")

    def change_output_folder(self, filetype):
        folder = filedialog.askdirectory(title=f"Select {filetype.capitalize()} Output Folder")
        if folder:
            self.output_folders[filetype] = folder
            # Prompt to save as default
            if messagebox.askyesno("Save as Default", f"Save this folder as the new default {filetype} folder?"):
                # Refresh Settings tab variables than save
                if filetype == "vocals":
                    self.settings_vocals_var.set(self.output_folders["vocals"])
                    self.load_separation_outputs()
                elif filetype == "instrumentals":
                    self.settings_instr_var.set(self.output_folders["instrumentals"])
                    self.load_separation_outputs()
                elif filetype == "transcriptions":
                    self.settings_trans_var.set(self.output_folders["transcriptions"])
                    self.load_transcription_outputs()
                self.save_settings()

    def check_models_directory(self):
        """
        Silently sets up the default AI models directory on startup.
        Defaults to an app-local 'Models' folder unless the user changed it in Settings.
        """
        app_dir = get_app_dir()
        default_models_dir = os.path.join(app_dir, "Models")
        self.models_dir = default_models_dir
        
        # 1. Check if the user has a custom path saved in settings.json
        if hasattr(self, 'settings_file') and os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    settings = json.load(f)
                    saved_dir = settings.get("models_dir", "")
                    if not saved_dir:  
                        self.models_dir = default_models_dir
                    else:
                        self.models_dir = saved_dir
            except json.JSONDecodeError:
                pass # Corrupted file, fallback to default
                
        # 2. Auto-create the directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 3. Lock in the environment variables globally BEFORE any AI libraries load
        os.environ["TORCH_HOME"] = os.path.join(self.models_dir, "hub")
        os.environ["HF_HOME"] = os.path.join(self.models_dir, "huggingface")
        os.environ["HF_HUB_CACHE"] = os.path.join(self.models_dir, "huggingface", "hub")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.models_dir, "huggingface")
        os.environ["MODEL_PATH"] = self.models_dir
        
        # 4. Save the confirmed path back to settings
        curr_settings = {}
        if hasattr(self, 'settings_file') and os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    curr_settings = json.load(f)
            except: pass
            
            curr_settings["models_dir"] = self.models_dir
            with open(self.settings_file, "w") as f:
                json.dump(curr_settings, f, indent=4)

    def download_default_models(self):
        """
        Unified function to download essential default models in the background 
        using the external util. Includes a status tracker and spawns a summary popup.
        """
        if hasattr(self, 'progress_bar'):
            self.progress_bar.set(0)
            self.progress_bar.grid()
            self.progress_bar.configure(mode="indeterminate")
            self.progress_bar.start()

        def update_ui(tool_name, status):
            # Update the UI text safely from the background thread
            if hasattr(self, 'progress_text'):
                self.after(0, lambda: self.progress_text.configure(text=f"{tool_name}: {status}"))

        def download_thread():
            from separators.utils import download_required_models
            
            # Fetch your model lists from settings or state (Examples below)
            # You should adapt these lists based on how you load them in your app
            demucs_list = ["htdemucs"] 
            whisper_list = ["base"]
            wav2vec2_list = ["facebook/wav2vec2-base-960h"]
            
            # Call the utils function
            final_status = download_required_models(
                models_dir=self.models_dir, 
                demucs_models=demucs_list,
                whisper_models=whisper_list,
                wav2vec2_models=wav2vec2_list,
                status_callback=update_ui
            )
            
            # Stop progress bar
            if hasattr(self, 'progress_bar'):
                self.after(0, self.progress_bar.stop)
                self.after(0, lambda: self.progress_bar.configure(mode="determinate"))
                self.after(0, lambda: self.progress_bar.set(1.0))
                self.after(3000, getattr(self.progress_bar, 'grid_remove', lambda: None))

            # Rescan models to update UI, then show popup
            self.after(0, self.scan_models_directory)
            self.after(500, lambda: self.show_model_summary(final_status))

        threading.Thread(target=download_thread, daemon=True).start()

    def show_model_summary(self, status_report):
        """
        Displays a popup window summarizing the results of the model download thread.
        """
        dialog = ctk.CTkToplevel(self)
        dialog.title("Setup Summary")
        dialog.geometry("350x300")
        dialog.attributes("-topmost", True)
        dialog.grab_set()

        ctk.CTkLabel(dialog, text="Model Setup Results", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(20, 15))

        for tool, status in status_report.items():
            text_color = "white"
            if "❌" in status: text_color = "#ff6666" 
            elif "✅" in status: text_color = "#66ff66" 
            elif "🔍" in status: text_color = "#66ccff" 

            row_frame = ctk.CTkFrame(dialog, fg_color="transparent")
            row_frame.pack(fill="x", padx=40, pady=5)
            
            ctk.CTkLabel(row_frame, text=f"{tool}:", font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")
            ctk.CTkLabel(row_frame, text=status, text_color=text_color, font=ctk.CTkFont(size=14)).pack(side="right")

        ctk.CTkButton(dialog, text="Awesome!", command=dialog.destroy, width=120).pack(pady=(25, 10))

    def scan_models_directory(self):
        """
        Scans the selected directory for strictly physical models (Vosk, Whisper, Wav2Vec2).
        Injects the required API strings for PyTorch Hub tools (Demucs, OpenUnmix).
        Saves the results directly to settings.json.
        """
        if hasattr(self, 'settings_models_var'):
            scan_dir = self.settings_models_var.get()
            self.models_dir = scan_dir  
        else:
            scan_dir = self.models_dir

        if not scan_dir or not os.path.exists(scan_dir):
            if hasattr(self, 'progress_text'):
                self.progress_text.configure(text="❌ Models directory not found.")
            return

        found_models = {
            "vosk": [],
            "whisper": [],
            "wav2vec2": [],
            "demucs": [],
            "openunmix": []
        }

        # 1. GROUP A: STRICTLY PHYSICAL SCANS
        # Vosk
        vosk_path = os.path.join(scan_dir, "vosk")
        if os.path.exists(vosk_path):
            found_models["vosk"] = [d for d in os.listdir(vosk_path) if os.path.isdir(os.path.join(vosk_path, d))]

        # Whisper
        whisper_path = os.path.join(scan_dir, "whisper")
        if os.path.exists(whisper_path):
            found_models["whisper"] = [f.replace('.pt', '') for f in os.listdir(whisper_path) if f.endswith('.pt')]

        # Wav2Vec2
        hf_hub_paths = [os.path.join(scan_dir, "huggingface", "hub"), os.path.join(scan_dir, "hub")]
        for hf_path in hf_hub_paths:
            if os.path.exists(hf_path):
                for d in os.listdir(hf_path):
                    if os.path.isdir(os.path.join(hf_path, d)) and d.startswith("models--"):
                        clean_name = d.replace("models--", "").replace("--", "/")
                        found_models["wav2vec2"].append(clean_name)

        # 2. GROUP B: CUSTOM MODEL SCANS (Official models injected below)
        demucs_path = os.path.join(scan_dir, "demucs_custom")
        if os.path.exists(demucs_path):
            found_models["demucs"] = [os.path.join(demucs_path, d).replace("\\", "/") for d in os.listdir(demucs_path) if os.path.isdir(os.path.join(demucs_path, d))]

        umx_path = os.path.join(scan_dir, "openunmix_custom")
        if os.path.exists(umx_path):
            found_models["openunmix"] = [os.path.join(umx_path, d).replace("\\", "/") for d in os.listdir(umx_path) if os.path.isdir(os.path.join(umx_path, d))]

        # 3. UPDATE UI VARIABLES (Smart Merge vs. Hard Sync)
        if hasattr(self, 'model_vars'):
            
            # Group definitions
            sync_tools = ["vosk", "whisper", "wav2vec2"]
            merge_tools = ["demucs", "openunmix"]

            for tool in found_models.keys():
                if tool in self.model_vars:
                    # Logic for Vosk/Whisper/Wav2vec2: Wipe and match disk
                    if tool in sync_tools:
                        combined = sorted(list(set(found_models[tool])))
                    
                    # Logic for Demucs/OpenUnmix: Merge with existing UI text
                    else:
                        # Grab what's currently in the text box and split by comma
                        current_text = self.model_vars[tool].get()
                        existing_list = [x.strip() for x in current_text.split(",") if x.strip()]
                        
                        # Add the "Official" safety net just in case they cleared the box
                        officials = ["htdemucs", "mdx", "mdx_extra"] if tool == "demucs" else ["umx", "umxl", "umxhq"]
                        
                        # Merge: Existing + Officials + Newly Scanned Folders
                        combined = sorted(list(set(existing_list + officials + found_models[tool])))
                    
                    # Update the UI text box
                    self.model_vars[tool].set(", ".join(combined))
                    
                    # Update internal tracking for dropdowns
                    if hasattr(self, 'transcription_models') and tool in sync_tools:
                        self.transcription_models[tool] = combined
                    if hasattr(self, 'separator_models'):
                        if tool == "demucs": self.separator_models["Demucs"] = combined
                        if tool == "openunmix": self.separator_models["OpenUnmix"] = combined

        # 4. AUTO-SAVE
        if hasattr(self, 'save_settings_changes'):
            original_showinfo = messagebox.showinfo
            messagebox.showinfo = lambda *args, **kwargs: None 
            try:
                self.save_settings_changes()
            finally:
                messagebox.showinfo = original_showinfo
        elif hasattr(self, 'save_settings'):
            self.save_settings()

        # 5. REFRESH UI
        try:
            if hasattr(self, 'on_trans_tool_change'): self.on_trans_tool_change()
        except Exception:
            pass

        # 6. EDUCATE THE USER via Popup
        if hasattr(self, 'progress_text'):
            self.progress_text.configure(text="✅ Models updated and saved.")

    def import_custom_model(self):
        """
        Spawns a small popup to ask the user which tool the custom model is for,
        then opens the file/folder picker and copies it to the correct directory.
        """
        # 1. Create a tiny popup window
        dialog = ctk.CTkToplevel(self)
        dialog.title("Import Custom Model")
        dialog.geometry("320x220")
        dialog.attributes("-topmost", True)
        dialog.grab_set()

        ctk.CTkLabel(dialog, text="Which tool is this model for?", font=ctk.CTkFont(weight="bold")).pack(pady=(20, 10))

        # We exclude Spleeter here
        tools = ["Vosk", "Whisper", "Wav2Vec2", "Demucs", "OpenUnmix"]
        selected_tool = ctk.StringVar(value=tools[0])
        ctk.CTkOptionMenu(dialog, variable=selected_tool, values=tools).pack(pady=10)

        def proceed():
            tool = selected_tool.get()
            dialog.destroy()
            self._execute_import(tool) # Pass the choice to the actual copy logic

        ctk.CTkButton(dialog, text="Next", command=proceed).pack(pady=20)

    def _execute_import(self, tool):
        """Handles the actual file dialog and copying based on the selected tool."""
        if tool == "Whisper":
            source_path = filedialog.askopenfilename(
                title="Select Custom Whisper Model (.pt)", 
                filetypes=[("PyTorch Models", "*.pt")]
            )
            is_file = True
        else:
            source_path = filedialog.askdirectory(title=f"Select Custom {tool} Model Folder")
            is_file = False

        if not source_path: 
            return 
            
        item_name = os.path.basename(source_path)
        dest_path = None # Initialize to satisfy Pylance

        if tool == "Vosk":
            dest_path = os.path.join(self.models_dir, "vosk", item_name)
        elif tool == "Whisper":
            dest_path = os.path.join(self.models_dir, "whisper", item_name)
        elif tool == "Wav2Vec2":
            dest_path = os.path.join(self.models_dir, "huggingface", "hub", f"models--custom--{item_name}")
        elif tool == "Demucs":
            dest_path = os.path.join(self.models_dir, "demucs_custom", item_name)
        elif tool == "OpenUnmix":
            dest_path = os.path.join(self.models_dir, "openunmix_custom", item_name)

        if dest_path is None:
            messagebox.showerror("Error", f"Unknown tool type: {tool}")
            return
        
        if os.path.exists(dest_path):
            messagebox.showwarning("Model Exists", f"A model named '{item_name}' is already installed for {tool}.")
            return
            
        try:
            if hasattr(self, 'progress_text'):
                self.progress_text.configure(text=f"Importing {item_name} for {tool}...")
            
            if is_file:
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(source_path, dest_path)
            else:
                shutil.copytree(source_path, dest_path)
                
            # ---> AUTOMATICALLY UPDATE UI AND JSON HERE <---
            self.scan_models_directory() 
            
            # Since scan_models_directory updates self.model_vars, we can just call your 
            # save_settings_changes() function to push those UI variables straight into the JSON file
            if hasattr(self, 'save_settings_changes'):
                self.save_settings_changes()
            elif hasattr(self, 'save_settings'):
                self.save_settings() # Fallback if save_settings_changes isn't available

            messagebox.showinfo("Success", f"Custom {tool} model '{item_name}' imported successfully!")
            
            if hasattr(self, 'progress_text'):
                self.progress_text.configure(text="✅ Import complete.")
                
        except Exception as e:
            messagebox.showerror("Import Error", f"Failed to import model.\n\nDetails: {str(e)}")

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
        if hasattr(self, 'abort_button') and self.abort_button.winfo_ismapped():
            return 

        freed_something = False

        # --- CLEAR SEPARATION MODELS ---
        # Clear if we switch to Transcriptions or Settings
        if active_tab in ["trans_out", "settings"]:
            if getattr(self, 'spleeter_sep', None) is not None:
                del self.spleeter_sep; self.spleeter_sep = None; freed_something = True
            if getattr(self, 'demucs_sep', None) is not None:
                del self.demucs_sep; self.demucs_sep = None; freed_something = True
            if getattr(self, 'openunmix_sep', None) is not None:
                del self.openunmix_sep; self.openunmix_sep = None; freed_something = True

        # --- CLEAR TRANSCRIPTION MODELS ---
        # Clear if we switch to Input, Separated Output, or Settings
        if active_tab in ["input", "sep_out", "settings"]:
            if getattr(self, 'whisper_trans', None) is not None:
                del self.whisper_trans; self.whisper_trans = None; freed_something = True
            if getattr(self, 'wav2vec2_trans', None) is not None:
                del self.wav2vec2_trans; self.wav2vec2_trans = None; freed_something = True
            if getattr(self, 'vosk_trans', None) is not None:
                del self.vosk_trans; self.vosk_trans = None; freed_something = True

        # 3. Perform the flush and update the UI (ONLY if something was actually deleted!)
        if freed_something:
            import sys
            import gc
            
            logging.info(f"[INFO] Switched to '{active_tab}'. Cleared inactive models.")
            
            # Force Python's Garbage Collector to clean up standard memory
            gc.collect()
            
            # ONLY interact with PyTorch if an AI tool has already loaded it!
            if 'torch' in sys.modules:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Update the progress text to inform the user!
            if hasattr(self, 'progress_bar') and hasattr(self, 'progress_text'):
                self.progress_bar.set(0)
                
                # Make the tab names look pretty for the UI text
                display_names = {
                    "input": "Input",
                    "sep_out": "Separation",
                    "trans_out": "Transcription",
                    "settings": "Settings"
                }
                pretty_name = display_names.get(active_tab, active_tab)
                
                self.progress_text.configure(text=f"Ready. RAM cleared for {pretty_name} workspace.")

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

    def setup_progress_bar(self, parent):
        progress_frame = ctk.CTkFrame(parent, height=40, corner_radius=0)
        # Keep the frame itself on the grid of the main window
        progress_frame.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(0, 10))
        # Important: prevents the frame from shrinking vertically when empty
        progress_frame.pack_propagate(False) 

        # 1. Pack the button to the far right
        self.abort_button = ctk.CTkButton(progress_frame, text="Abort", command=self.abort_separation_process, width=80, height=24, corner_radius=0)
        self.abort_button.pack(side="right", padx=(0, 10))
        self.abort_button.pack_forget()

        # 2. Pack the progress bar to the left, and tell it to fill empty space
        self.progress_bar = ctk.CTkProgressBar(progress_frame, mode="determinate", height=12)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(10, 10))
        self.progress_bar.set(0)
        self.progress_bar.pack_forget()

        # 3. Pack the text to the left. 
        # When the bar is hidden, this text will slide to the far left automatically!
        self.progress_text = ctk.CTkLabel(progress_frame, text="Ready", font=ctk.CTkFont(size=12))
        self.progress_text.pack(side="left", padx=(10, 10))

    def update_task_progress(self, percent, message, file_index, total_files, task_name="Task"):
        """
        @brief Unified progress updater for background tasks (Separation, Transcription, etc.)
        """
        if getattr(self, 'abort_separation', False):
            self.after(0, lambda: self.progress_text.configure(text=f"{task_name} aborted by user. (Ready)"))
            self.after(0, lambda: self.progress_bar.configure(mode="indeterminate"))
            self.after(0, lambda: self.progress_bar.pack_forget())
            self.after(0, lambda: self.abort_button.pack_forget())
            raise RuntimeError("ABORT_REQUESTED")

        # Calculate batch progress locally
        file_weight = 100.0 / total_files if total_files > 0 else 100.0
        base_progress = (file_index - 1) * file_weight
        current_file_progress = (percent / 100.0) * file_weight
        total_batch_percent = base_progress + current_file_progress

        # Format display message
        display_msg = f"[{file_index}/{total_files}] {message}" if total_files > 1 else message

        # Update UI safely
        self.after(0, lambda: self.progress_bar.pack(side="left", fill="x", expand=True, padx=(10, 10), before=self.progress_text)) 
        self.after(0, lambda: self.progress_bar.set(total_batch_percent / 100.0))
        self.after(0, lambda: self.progress_text.configure(text=display_msg))

    def abort_separation_process(self):
        self.abort_separation = True
        self.progress_text.configure(text="Aborting process, please wait...")
        self.abort_button.pack_forget()

    def separate_audio(self):
        """
        @brief Prepares a batch of selected files and folders for audio separation.
        """
        import os 
        
        # 1. Gather all selected files
        selected_files = []
        selected_paths = set() 

        valid_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

        for idx, info in self.input_selection_dict.items():
            if info["var"].get():  # If checked
                
                if info["type"] == 'song':
                    file_path = info["data"]["path"]
                    if file_path not in selected_paths:
                        selected_files.append(info["data"])
                        selected_paths.add(file_path)
                        
                elif info["type"] == 'folder':
                    folder_path = info["data"]["path"]
                    try:
                        for root, dirs, files in os.walk(folder_path):
                            for filename in files:
                                file_path = os.path.join(root, filename)
                                
                                if os.path.isfile(file_path):
                                    ext = os.path.splitext(filename)[1].lower()
                                    if ext in valid_extensions and file_path not in selected_paths:
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
        fmt = self.format_var.get()
        ai_tool = self.ai_tool_var.get()
        try:            
            sr = int(self.sr_var.get()) if fmt in ["wav", "flac"] else 44100
            shifts = int(self.shifts_var.get()) if ai_tool == "Demucs" else 1
            bitrate = f"{int(self.bitrate_var.get())}k" if fmt == "mp3" else "192k"
            mp3_preset = int(self.mp3_preset_slider.get()) if fmt == "mp3" and ai_tool == "Demucs" else 2
            flac_compression = int(self.flac_slider.get()) if fmt == "flac" else 5
            overlap = float(self.overlap_slider.get()) if ai_tool == "Demucs" else 0.25
        except ValueError:
            from tkinter import messagebox
            self.after(0, lambda: messagebox.showwarning("Invalid Input", "Using default parameters."))
            sr, shifts, mp3_preset, bitrate, flac_compression, overlap = 44100, 1, 2, "192k", 5, 0.25

        # 3. PACK THE DATACLASS
        config = SeparationSettings(
            ai_tool=self.ai_tool_var.get(),
            model=self.model_var.get(),
            channels=self.channel_var.get(),
            fmt=fmt,
            sr=sr,
            bitrate=bitrate,
            bit_depth=self.bit_depth_var.get() if fmt in ["wav", "flac"] and ai_tool == "Demucs" else None,
            mp3_preset=mp3_preset,
            shifts=shifts,
            overlap=overlap,
            flac_compression=flac_compression,
            device=self.device_var.get(),
            vocals_folder=self.output_folders["vocals"],
            instr_folder=self.output_folders["instrumentals"]
        )

        # 4. Start the thread, passing just the files list and the config object
        import threading
        thread = threading.Thread(target=self._run_separation, args=(selected_files, config))
        thread.daemon = True
        thread.start()

    def _run_separation(self, selected_files, config: SeparationSettings): 
        """
        @brief Executes audio separation on a batch of files iteratively in a background thread.
        Handles lazy loading and enforces CPU for Spleeter to ensure stability.
        """
        import os
        import logging
        total_files = len(selected_files)
        
        # UI: Show abort button
        self.after(0, lambda: self.abort_button.pack(side="right", padx=(0, 10)))

        try:
            # 1. Device Enforcement & Progress Init
            actual_device = config.device
            if config.ai_tool == "Spleeter":
                actual_device = "CPU"
                
            self.update_task_progress(5, f"Loading {config.ai_tool} on {actual_device}...", 1, total_files, "Separation")

            # 2. Environment Setup (Torch/TensorFlow behavior)
            if actual_device == "CPU":
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # 3. LAZY LOADING & TYPE CASTING
            # We use a local variable and type assertions to satisfy Pylance
            active_sep = None

            if config.ai_tool == "Spleeter":
                if getattr(self, "spleeter_sep", None) is None:
                    import separators.spleeter_separator as spleeter_mod
                    self.spleeter_sep = spleeter_mod.SpleeterSeparator()
                active_sep = self.spleeter_sep

            elif config.ai_tool == "Demucs":
                if getattr(self, "demucs_sep", None) is None:
                    import separators.demucs_separator as demucs_mod
                    self.demucs_sep = demucs_mod.DemucsSeparator()
                active_sep = self.demucs_sep

            elif config.ai_tool == "OpenUnmix":
                if getattr(self, "openunmix_sep", None) is None:
                    import separators.openunmix_separator as oumx_mod
                    self.openunmix_sep = oumx_mod.OpenUnmixSeparator()
                active_sep = self.openunmix_sep

            if active_sep is None:
                raise RuntimeError(f"Could not initialize {config.ai_tool} separator.")

            successful_files = []
            failed_files = []

            # --- BATCH LOOP ---
            for i, song in enumerate(selected_files, 1):
                if self.abort_separation: 
                    break

                input_path = song['path']
                song_name = os.path.splitext(song['name'])[0]
                cb = lambda p, m: self.update_task_progress(p, m, i, total_files, task_name="Separation")
                
                result = (False, "", "") # Safe default

                try:
                    # --- TOOL-SPECIFIC EXECUTION WITH TYPE GUARDS ---
                    
                    if config.ai_tool == "Demucs":
                        from separators.demucs_separator import DemucsSeparator
                        assert isinstance(active_sep, DemucsSeparator)
                        
                        # Use the 'or' operator to provide a default if config.bit_depth is None
                        safe_bit_depth = config.bit_depth or "16-bit"

                        result = active_sep.separate(
                            input_path=input_path, 
                            song_name=song_name, 
                            vocals_folder=config.vocals_folder, 
                            instr_folder=config.instr_folder, 
                            model=config.model, 
                            channels=config.channels, 
                            fmt=config.fmt, 
                            sr=config.sr, 
                            bitrate=config.bitrate, 
                            bit_depth=safe_bit_depth,
                            shifts=config.shifts, 
                            overlap=config.overlap, 
                            device_choice=actual_device, 
                            flac_compression=config.flac_compression, 
                            progress_callback=cb
                        )
                        
                    elif config.ai_tool == "Spleeter":
                        from separators.spleeter_separator import SpleeterSeparator
                        assert isinstance(active_sep, SpleeterSeparator)
                        
                        result = active_sep.separate(
                            input_path=input_path, 
                            song_name=song_name, 
                            vocals_folder=config.vocals_folder, 
                            instr_folder=config.instr_folder, 
                            channels=config.channels, 
                            fmt=config.fmt, 
                            sr=config.sr, 
                            bitrate=config.bitrate, 
                            device_choice=actual_device, 
                            flac_compression=config.flac_compression, 
                            progress_callback=cb
                        )
                    
                    elif config.ai_tool == "OpenUnmix":
                        from separators.openunmix_separator import OpenUnmixSeparator
                        assert isinstance(active_sep, OpenUnmixSeparator)
                        
                        result = active_sep.separate(
                            input_path=input_path, 
                            song_name=song_name, 
                            vocals_folder=config.vocals_folder, 
                            instr_folder=config.instr_folder, 
                            model=config.model, 
                            channels=config.channels, 
                            fmt=config.fmt, 
                            sr=config.sr, 
                            bitrate=config.bitrate, 
                            device_choice=actual_device, 
                            flac_compression=config.flac_compression, 
                            progress_callback=cb
                        )

                    # Validate Result
                    if isinstance(result, tuple) and len(result) >= 3 and result[0]:
                        successful_files.extend([result[1], result[2]]) 
                    else:
                        logging.error(f"Separation logic returned failure for: {song_name}")
                        failed_files.append(song_name)
                        
                except Exception as e:
                    logging.error(f"Error processing {song_name}: {e}", exc_info=True)
                    failed_files.append(song_name)

            # --- COMPLETION LOGIC ---
            if not self.abort_separation:
                success_count = len(successful_files) // 2 
                completion_text = f"Batch complete! {success_count}/{total_files} processed. (Ready)"
                
                self.after(500, lambda: self.progress_bar.pack_forget())
                self.after(1000, lambda: self.progress_text.configure(text=completion_text))

                details = ""
                if successful_files:
                    details += "✅ Generated:\n" + "\n".join(successful_files) + "\n\n"
                if failed_files:
                    details += "❌ Failed:\n" + "\n".join(failed_files) + "\n\nCheck logs for details."

                title = "Batch Finished" + (" with Errors" if failed_files else " Successfully")
                self.after(0, lambda: self.show_batch_summary_window(title, details))

        except Exception as e:
            if str(e) == "ABORT_REQUESTED":
                logging.info("Separation aborted by user.")
            else:
                self.after(0, lambda: self.progress_text.configure(text=f"Error: {str(e)} (Ready)"))
                logging.error(f"Thread error: {e}", exc_info=True)
        finally:
            self.abort_separation = False
            self.after(0, lambda: self.abort_button.pack_forget())

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
        device = self.device_var.get()

        # 3. PACK THE DATACLASS
        config = TranscriptionSettings(
            tool=self.trans_tool_var.get(),
            model=self.trans_model_var.get(),
            lang=self.trans_lang_var.get(),
            use_spk=self.use_spk_id_var.get() if self.trans_tool_var.get() == "vosk" else False,
            device=self.device_var.get(),
            output_folder=self.output_folders["transcriptions"]
        )

        # 4. Start thread
        import threading
        threading.Thread(
            target=self._exec_standalone_trans, 
            args=(selected_audio_files, config), 
            daemon=True
        ).start()

    def _exec_standalone_trans(self, selected_vocals, config: TranscriptionSettings):
        """
        @brief Executes batch transcription on selected audio files in a background thread.
        Manages the lazy loading of transcription models (Whisper, Vosk, Wav2Vec2) and
        iterates over the selected batch. It tracks successful and failed operations,
        formats output file names, and safely pushes UI updates to the main thread.
        """
        import os
        import logging
        total_files = len(selected_vocals)

        self.after(0, lambda: self.abort_button.pack(side="right", padx=(0, 10)))

        try:
            # --- 1. DEVICE ENFORCEMENT ---
            actual_device = config.device
            if config.tool == "vosk":
                actual_device = "CPU"
            
            self.update_task_progress(5, f"Initializing {config.tool} on {actual_device}...", 1, total_files, "Transcription")

            # --- 2. ENVIRONMENT SETUP ---
            if actual_device == "CPU":
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            elif "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # --- 3. LAZY LOADING & TYPE GUARDING ---
            active_trans = None

            if config.tool == "whisper":
                if getattr(self, "whisper_trans", None) is None:
                    import separators.whisper_transcription as whisper_mod
                    self.whisper_trans = whisper_mod.WhisperTranscription()
                active_trans = self.whisper_trans

            elif config.tool == "vosk":
                if getattr(self, "vosk_trans", None) is None:
                    import separators.vosk_transcription as vosk_mod
                    self.vosk_trans = vosk_mod.VoskTranscription()
                active_trans = self.vosk_trans

            elif config.tool == "wav2vec2":
                if getattr(self, "wav2vec2_trans", None) is None:
                    import separators.wav2vec2_transcription as w2v2_mod
                    self.wav2vec2_trans = w2v2_mod.Wav2Vec2Transcription()
                active_trans = self.wav2vec2_trans

            if active_trans is None:
                raise RuntimeError(f"Could not initialize {config.tool} transcription tool.")

            successful_files = []  
            failed_files = []      

            # --- 4. BATCH LOOP ---
            for i, vocal in enumerate(selected_vocals, 1):
                if getattr(self, 'abort_separation', False): 
                    break

                vocal_path = vocal['path']
                filename = vocal['name']
                base_name = os.path.splitext(filename)[0]
                out_name = f"{base_name}_{config.tool}_{config.model.replace('/', '_')}.txt"
                out_path = os.path.join(config.output_folder, out_name)
                cb = lambda p, m: self.update_task_progress(p, m, i, total_files, task_name="Transcription")
                
                result = None

                # --- TOOL-SPECIFIC EXECUTION WITH TYPE GUARDS ---
                try:
                    if config.tool == "whisper":
                        from separators.whisper_transcription import WhisperTranscription
                        assert isinstance(active_trans, WhisperTranscription)
                        
                        # Changed 'input_path' to 'audio_path' to match your error
                        result = active_trans.transcribe(
                            audio_path=vocal_path, 
                            output_path=out_path, 
                            model_name=config.model, 
                            language=config.lang, 
                            device_choice=actual_device, 
                            progress_callback=cb
                        )

                    elif config.tool == "wav2vec2":
                        from separators.wav2vec2_transcription import Wav2Vec2Transcription
                        assert isinstance(active_trans, Wav2Vec2Transcription)
                        
                        result = active_trans.transcribe(
                            audio_path=vocal_path, 
                            output_path=out_path, 
                            model_name=config.model, 
                            device_choice=actual_device, 
                            progress_callback=cb
                        )
                        
                    elif config.tool == "vosk":
                        from separators.vosk_transcription import VoskTranscription
                        assert isinstance(active_trans, VoskTranscription)
                        
                        result = active_trans.transcribe(
                            audio_path=vocal_path, 
                            output_path=out_path, 
                            model_name=config.model,        
                            use_diarization=config.use_spk,
                            device_choice=actual_device, 
                            progress_callback=cb
                        )

                    # --- RESULT VALIDATION ---
                    if (isinstance(result, tuple) and result[0]) or result is True:
                        successful_files.append(out_name)
                    else:
                        failed_files.append(filename)
                except Exception as inner_e:
                    logging.error(f"Error in {filename}: {inner_e}")
                    failed_files.append(filename)

            # --- 5. COMPLETION UI ---
            if not getattr(self, 'abort_separation', False):
                self.after(500, lambda: self.progress_bar.pack_forget())
                self.after(1000, lambda: self.progress_text.configure(text=f"Batch complete! {len(successful_files)}/{total_files} saved. (Ready)"))
                
                details = ""
                if successful_files:
                    details += "✅ Successfully transcribed:\n" + "\n".join(successful_files) + "\n\n"
                if failed_files:
                    details += "❌ Failed to transcribe:\n" + "\n".join(failed_files) + "\n\nCheck logs for details."

                title = "Transcription Summary"
                self.after(0, lambda: self.show_batch_summary_window(title, details))

        except Exception as e:
            if str(e) == "ABORT_REQUESTED":
                logging.info("Transcription aborted.")
            else:
                logging.error(f"Standalone trans error: {e}", exc_info=True)
                self.after(0, lambda: self.progress_text.configure(text="Error occurred (Ready)"))
        finally:
            self.abort_separation = False
            self.after(0, lambda: self.abort_button.pack_forget())

if __name__ == "__main__":
    multiprocessing.freeze_support()

    try:
        app = SeparationApp()
        app.mainloop()
    finally:
        # This executes even if the app crashes
        print("Cleaning up system resources...")
        # Forcefully kill any remaining child processes of this script
        for child in multiprocessing.active_children():
            child.terminate()
