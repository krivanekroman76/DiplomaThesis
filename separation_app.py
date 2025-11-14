import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import platform
import subprocess
import threading
import queue
import json 
from pkg_resources import resource_filename
<<<<<<< HEAD
import museval 
=======
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a

# Separation classes in separators directory
import separators.spleeter_separator as spleeter
import separators.demucs_separator as demucs
import separators.openunmix_separator as openunmix

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

def open_file(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.call(("open", path))
    else:
        subprocess.call(("xdg-open", path))

class ProgressWindow(ctk.CTkToplevel):
    def __init__(self, parent, title="Processing..."):
        super().__init__(parent)
        self.title(title)
        self.geometry("400x150")
        self.transient(parent)
        self.grab_set()  # Modal
        self.canceled = False

        # Center on parent
        self.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))

        label = ctk.CTkLabel(self, text="Separation in progress...", font=ctk.CTkFont(size=14))
        label.pack(pady=20)

        self.progress = ctk.CTkProgressBar(self, mode="indeterminate")
        self.progress.pack(pady=10, padx=20, fill="x")
        self.progress.start()  # Indeterminate animation

        status_label = ctk.CTkLabel(self, text="Loading model...", font=ctk.CTkFont(size=12))
        status_label.pack(pady=5)

        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=10, fill="x", padx=20)

        cancel_btn = ctk.CTkButton(btn_frame, text="Cancel", command=self.cancel)
        cancel_btn.pack(side="right", padx=10)

        self.status_label = status_label
        self.parent = parent

    def update_status(self, message):
        self.status_label.configure(text=message)

    def cancel(self):
        self.canceled = True
        self.destroy()
        messagebox.showinfo("Canceled", "Separation canceled.")

    def close(self, success=True):
        self.progress.stop()
        if success:
            self.destroy()
        else:
            self.status_label.configure(text="Error occurred. Check console.")
            # Auto-close after 3s or manual

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
        output_button.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

<<<<<<< HEAD
        self.evaluation_button = ctk.CTkButton(
            self.sidebar, 
            text="Evaluation", 
            command=self.show_evaluation,
            width=180
        )
        self.evaluation_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.evaluation_button.grid_remove()  # Hide initially

=======
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
        settings_button = ctk.CTkButton(
            self.sidebar, 
            text="Settings", 
            command=self.show_settings,
            width=180
        )
<<<<<<< HEAD
        settings_button.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

        # Appearance mode selection (bottom-aligned)
        appearance_mode_label = ctk.CTkLabel(self.sidebar, text="Appearance Mode:", anchor="w")
        appearance_mode_label.grid(row=5, column=0, padx=20, pady=(20, 0), sticky="w")
=======
        settings_button.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        # Appearance mode selection (bottom-aligned)
        appearance_mode_label = ctk.CTkLabel(self.sidebar, text="Appearance Mode:", anchor="w")
        appearance_mode_label.grid(row=4, column=0, padx=20, pady=(20, 0), sticky="w")
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a

        appearance_mode_optionemenu = ctk.CTkOptionMenu(
            self.sidebar, 
            values=["Light", "Dark", "System"],
            command=self.change_appearance_mode_event,
            width=160
        )
<<<<<<< HEAD
        appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 10), sticky="ew")
=======
        appearance_mode_optionemenu.grid(row=5, column=0, padx=20, pady=(10, 10), sticky="ew")
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
        appearance_mode_optionemenu.set("Dark")

        # UI Scaling (Zoom) selection (bottom-aligned)
        scaling_label = ctk.CTkLabel(self.sidebar, text="UI Scaling:", anchor="w")
<<<<<<< HEAD
        scaling_label.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
=======
        scaling_label.grid(row=6, column=0, padx=20, pady=(10, 0), sticky="w")
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a

        scaling_optionemenu = ctk.CTkOptionMenu(
            self.sidebar, 
            values=["80%", "90%", "100%", "110%", "120%"],
            command=self.change_scaling_event,
            width=160
        )
<<<<<<< HEAD
        scaling_optionemenu.grid(row=8, column=0, padx=20, pady=(10, 20), sticky="ew")
=======
        scaling_optionemenu.grid(row=7, column=0, padx=20, pady=(10, 20), sticky="ew")
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
        scaling_optionemenu.set("100%")

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
        self.settings_frame = ctk.CTkFrame(self.content_frame)
<<<<<<< HEAD
        self.evaluation_frame = ctk.CTkFrame(self.content_frame) 
=======
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a

        # Create tab contents
        self.create_input_tab()
        self.create_output_tab()
        self.create_settings_tab() 
<<<<<<< HEAD
        self.create_evaluation_tab()
=======
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a

        # Initially show input
        self.show_input()

        # Load initial songs and outputs
        self.load_input()
        self.load_outputs()

        # Buttons in input tab
        self.input_button = input_button
        self.output_button = output_button
<<<<<<< HEAD
        self.toggle_evaluation()
        self.settings_button = settings_button

    def load_settings(self):
        defaults = {
            "input_folder": "input",
            "vocals_folder": "output/vocals",
            "instrumentals_folder": "output/instrumentals",
            "transcriptions_folder": "output/text",
            "enable_evaluation": False,
            "separator_models": {
                "Spleeter": [],
                "Demucs": ["mdx", "mdx_extra", "htdemucs"],
                "OpenUnmix": ["umxl", "umxhq", "umx", "umxse"]
            },
            "transcription_models": {
                "whisper": ["large", "medium", "small", "tiny", "base", "turbo"],
                "wav2vec2": ["facebook/wav2vec2-base-960h", "facebook/wav2vec2-large-960h"],
                "coqui": ["model.pbmm"]
            }
=======
        self.settings_button = settings_button

    def load_settings(self):
        """Load default folders from settings.json, with fallbacks."""
        defaults = {
            "input_folder": "/input",
            "vocals_folder": "output/vocals",
            "instrumentals_folder": "output/instrumentals",
            "transcriptions_folder": "output/text"
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
        }
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, "r") as f:
                    data = json.load(f)
                self.input_folder = data.get("input_folder", defaults["input_folder"])
                self.output_folders = {
                    "vocals": data.get("vocals_folder", defaults["vocals_folder"]),
                    "instrumentals": data.get("instrumentals_folder", defaults["instrumentals_folder"]),
                    "transcriptions": data.get("transcriptions_folder", defaults["transcriptions_folder"])
                }
<<<<<<< HEAD
                self.enable_evaluation = data.get("enable_evaluation", defaults["enable_evaluation"])
                self.separator_models = data.get("separator_models", defaults["separator_models"])
                self.transcription_models = data.get("transcription_models", defaults["transcription_models"])
            except (json.JSONDecodeError, KeyError):
                self.set_defaults(defaults)
        else:
            
            self.set_defaults(defaults)
    
    def set_defaults(self, defaults):
        self.input_folder = defaults["input_folder"]
        self.output_folders = {
            "vocals": defaults["vocals_folder"],
            "instrumentals": defaults["instrumentals_folder"],
            "transcriptions": defaults["transcriptions_folder"]
        }
        self.enable_evaluation = defaults["enable_evaluation"]
        self.separator_models = defaults["separator_models"]
        self.transcription_models = defaults["transcription_models"]
        self.save_settings()
        
    def save_settings(self):
        """Save current folders, models, and settings to settings.json."""
=======
            except (json.JSONDecodeError, KeyError):
                print("Warning: Invalid settings.json, using defaults.")
                self.input_folder = defaults["input_folder"]
                self.output_folders = {k: v for k, v in defaults.items() if k != "input_folder"}
        else:
            self.input_folder = defaults["input_folder"]
            self.output_folders = {k: v for k, v in defaults.items() if k != "input_folder"}
        # Save defaults if file doesn't exist
        self.save_settings()
    
    def save_settings(self):
        """Save current folders to settings.json."""
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
        data = {
            "input_folder": self.input_folder,
            "vocals_folder": self.output_folders["vocals"],
            "instrumentals_folder": self.output_folders["instrumentals"],
<<<<<<< HEAD
            "transcriptions_folder": self.output_folders["transcriptions"],
            "enable_evaluation": self.enable_evaluation,
            "separator_models": self.separator_models,
            "transcription_models": self.transcription_models
=======
            "transcriptions_folder": self.output_folders["transcriptions"]
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
        }
        try:
            with open(self.settings_file, "w") as f:
                json.dump(data, f, indent=4)
<<<<<<< HEAD
            self.toggle_evaluation()
            print(f"Settings saved to {self.settings_file}")
=======
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
        except Exception as e:
            print(f"Error saving settings: {e}")

    def show_input(self):
        self.input_frame.grid(row=0, column=0, sticky="nsew")
        self.output_frame.grid_forget()
        self.settings_frame.grid_forget()
<<<<<<< HEAD
        self.evaluation_frame.grid_forget()
=======
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a

    def show_output(self):
        self.output_frame.grid(row=0, column=0, sticky="nsew")
        self.input_frame.grid_forget()
<<<<<<< HEAD
        self.evaluation_frame.grid_forget()
=======
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
        self.settings_frame.grid_forget()
        # Highlight active button
        self.output_button.configure(fg_color=("#DCE4EE", "#1f538d"))
        self.input_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
<<<<<<< HEAD
        self.evaluation_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        self.settings_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])

    def show_evaluation(self):
        self.evaluation_frame.grid(row=0, column=0, sticky="nsew")
        self.input_frame.grid_forget()
        self.output_frame.grid_forget()
        self.settings_frame.grid_forget()
        # Highlight active button
        self.evaluation_button.configure(fg_color=("#DCE4EE", "#1f538d"))
        self.input_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        self.output_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        self.settings_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])

=======
        self.settings_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
    
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
    def show_settings(self):
        self.settings_frame.grid(row=0, column=0, sticky="nsew")
        self.input_frame.grid_forget()
        self.output_frame.grid_forget()
<<<<<<< HEAD
        self.evaluation_frame.grid_forget()
=======
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
        # Highlight active button
        self.settings_button.configure(fg_color=("#DCE4EE", "#1f538d"))
        self.input_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        self.output_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
<<<<<<< HEAD
        self.evaluation_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
=======
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)

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
        self.songs_listbox = tk.Listbox(frame)
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
            sep_scrollable, variable=self.model_var, values=["umxl", "umxhq", "umx", "umxse"], width=200
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

        # Transcription checkbox
        self.transcript_var = tk.BooleanVar(value=False)
        self.transcript_checkbox = ctk.CTkCheckBox(sep_scrollable, text="Transcribe vocals", variable=self.transcript_var, command=self.on_trans_tool_change)
        self.transcript_checkbox.grid(row=11, column=0, sticky="w", padx=20, pady=10)

        # Transcription frame
        self.transcription_frame = ctk.CTkFrame(sep_scrollable)
        self.transcription_frame.grid(row=12, column=0, sticky="ew", padx=20, pady=5)
        self.transcription_frame.grid_remove()  # Hide initially

        # Transcription tool label
        self.trans_tool_label = ctk.CTkLabel(self.transcription_frame, text="Tool:", anchor="w")
        self.trans_tool_label.grid(row=1, column=0, sticky="w", padx=20, pady=(10,0))
        self.trans_tool_var = tk.StringVar(value="whisper")
        self.radio_whisper = ctk.CTkRadioButton(self.transcription_frame, text="whisper", variable=self.trans_tool_var, value="whisper", command=self.on_trans_tool_change)
        self.radio_wav2vec2 = ctk.CTkRadioButton(self.transcription_frame, text="wav2vec2", variable=self.trans_tool_var, value="wav2vec2", command=self.on_trans_tool_change)
        self.radio_coqui = ctk.CTkRadioButton(self.transcription_frame, text="coqui", variable=self.trans_tool_var, value="coqui", command=self.on_trans_tool_change)

        self.radio_whisper.grid(row=2, column=0, sticky="w", padx=20, pady=5)
        self.radio_wav2vec2.grid(row=3, column=0, sticky="w", padx=20, pady=5)
        self.radio_coqui.grid(row=4, column=0, sticky="w", padx=20, pady=5)

        # Transcription model label
        self.trans_model_label = ctk.CTkLabel(self.transcription_frame, text="Model:", anchor="w")
        self.trans_model_label.grid(row=5, column=0, sticky="w", padx=20, pady=(10,0))

        self.transcript_model = tk.StringVar(value="tiny")
        self.transcript_model_menu = ctk.CTkOptionMenu(self.transcription_frame, variable=self.transcript_model, width=200,
            values=["large", "medium", "small", "tiny", "base", "turbo", "large-v1", "large-v2", "large-v3", "large-v3-turbo"])
        self.transcript_model_menu.grid(row=16, column=0, sticky="ew", padx=20, pady=5)

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
                values = self.separator_models.get("OpenUnmix", ["umxl", "umxhq", "umx", "umxse"])
                self.model_menu.configure(values=values)
                self.model_var.set(values[0] if values else "umxl")
                self.model_label.grid()
                self.model_menu.grid()
        except Exception as e:
            print(f"Error updating separator models: {e}")
            # Fallback to defaults
            if tool == "Demucs":
                self.model_menu.configure(values=["mdx", "mdx_extra", "htdemucs"])
                self.model_var.set("mdx")
            elif tool == "OpenUnmix":
                self.model_menu.configure(values=["umxl", "umxhq", "umx", "umxse"])
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
        tool = self.trans_tool_var.get()
        trans_check = self.transcript_var.get()
        # Transcription frame visible
        if trans_check:
            self.transcription_frame.grid()
        else:
            self.transcription_frame.grid_remove()
        try:
            if tool == "whisper":
                values = self.transcription_models.get("whisper", ["tiny", "base"])
            elif tool == "wav2vec2":
                values = self.transcription_models.get("wav2vec2", ["facebook/wav2vec2-base-960h"])
            elif tool == "coqui":
                values = self.transcription_models.get("coqui", ["model.pbmm"])
            else:
                self.trans_model_label.grid_remove()
                self.transcript_model_menu.grid_remove()
                return
            self.transcript_model_menu.configure(values=values)
            self.transcript_model.set(values[0] if values else "")
            self.trans_model_label.grid()
            self.transcript_model_menu.grid()
        except Exception as e:
            print(f"Error updating transcription models: {e}")
            # Fallback to defaults
            self.on_trans_tool_change()

    def create_output_tab(self):
        frame = self.output_frame
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure((1,3,5), weight=1)

        # Transcriptions section
        trans_label = ctk.CTkLabel(frame, text="Transcriptions", font=ctk.CTkFont(size=18, weight="bold"))
        trans_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10,5))

        trans_btn = ctk.CTkButton(frame, text="Change Folder", command=lambda: self.change_output_folder("transcriptions"))
        trans_btn.grid(row=0, column=1, sticky="e", padx=10, pady=(10,5))

        self.trans_listbox = tk.Listbox(frame)
        self.trans_listbox.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10)
        self.trans_listbox.bind("<Double-Button-1>", self.open_selected_transcription)

        # Vocals section
        vocals_label = ctk.CTkLabel(frame, text="Vocals", font=ctk.CTkFont(size=18, weight="bold"))
        vocals_label.grid(row=2, column=0, sticky="w", padx=10, pady=(20,5))

        vocals_btn = ctk.CTkButton(frame, text="Change Folder", command=lambda: self.change_output_folder("vocals"))
        vocals_btn.grid(row=2, column=1, sticky="e", padx=10, pady=(20,5))

        self.vocals_listbox = tk.Listbox(frame)
        self.vocals_listbox.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10)
        self.vocals_listbox.bind("<Double-Button-1>", self.open_selected_vocal)

        # Instrumentals section
        instr_label = ctk.CTkLabel(frame, text="Instrumentals", font=ctk.CTkFont(size=18, weight="bold"))
        instr_label.grid(row=4, column=0, sticky="w", padx=10, pady=(20,5))

        instr_btn = ctk.CTkButton(frame, text="Change Folder", command=lambda: self.change_output_folder("instrumentals"))
        instr_btn.grid(row=4, column=1, sticky="e", padx=10, pady=(20,5))

        self.instr_listbox = tk.Listbox(frame)
        self.instr_listbox.grid(row=5, column=0, columnspan=2, sticky="nsew", padx=10)
        self.instr_listbox.bind("<Double-Button-1>", self.open_selected_instrumental)

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
                self.save_settings()
                
    def create_settings_tab(self):
        frame = self.settings_frame
<<<<<<< HEAD
        frame.grid_columnconfigure((0, 1), weight=1)
        
        # Folder label
        settings_label = ctk.CTkLabel(frame, text="Default Folders Settings", font=ctk.CTkFont(size=20, weight="bold"))
        settings_label.grid(row=0, column=0, pady=(20, 20))
=======
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        settings_label = ctk.CTkLabel(frame, text="Default Folders Settings", font=ctk.CTkFont(size=20, weight="bold"))
        settings_label.grid(row=0, column=0, pady=(20, 20))

>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
        # Input folder
        input_label = ctk.CTkLabel(frame, text="Input Folder:", anchor="w")
        input_label.grid(row=1, column=0, sticky="w", padx=20, pady=(10, 0))
        self.settings_input_var = tk.StringVar(value=self.input_folder)
        input_entry = ctk.CTkEntry(frame, textvariable=self.settings_input_var, width=400)
        input_entry.grid(row=2, column=0, sticky="ew", padx=20, pady=5)
<<<<<<< HEAD
=======

>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
        # Vocals folder
        vocals_label = ctk.CTkLabel(frame, text="Vocals Folder:", anchor="w")
        vocals_label.grid(row=3, column=0, sticky="w", padx=20, pady=(10, 0))
        self.settings_vocals_var = tk.StringVar(value=self.output_folders["vocals"])
        vocals_entry = ctk.CTkEntry(frame, textvariable=self.settings_vocals_var, width=400)
        vocals_entry.grid(row=4, column=0, sticky="ew", padx=20, pady=5)
<<<<<<< HEAD
=======

>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
        # Instrumentals folder
        instr_label = ctk.CTkLabel(frame, text="Instrumentals Folder:", anchor="w")
        instr_label.grid(row=5, column=0, sticky="w", padx=20, pady=(10, 0))
        self.settings_instr_var = tk.StringVar(value=self.output_folders["instrumentals"])
        instr_entry = ctk.CTkEntry(frame, textvariable=self.settings_instr_var, width=400)
        instr_entry.grid(row=6, column=0, sticky="ew", padx=20, pady=5)
<<<<<<< HEAD
=======

>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
        # Transcriptions folder
        trans_label = ctk.CTkLabel(frame, text="Transcriptions Folder:", anchor="w")
        trans_label.grid(row=7, column=0, sticky="w", padx=20, pady=(10, 0))
        self.settings_trans_var = tk.StringVar(value=self.output_folders["transcriptions"])
        trans_entry = ctk.CTkEntry(frame, textvariable=self.settings_trans_var, width=400)
        trans_entry.grid(row=8, column=0, sticky="ew", padx=20, pady=5)
<<<<<<< HEAD
        
=======

>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
        # Buttons
        button_frame = ctk.CTkFrame(frame)
        button_frame.grid(row=9, column=0, pady=(20, 20))
        save_btn = ctk.CTkButton(button_frame, text="Save Changes", command=self.save_settings_changes)
        save_btn.grid(row=0, column=0, padx=10)
        restore_btn = ctk.CTkButton(button_frame, text="Restore Defaults", command=self.restore_defaults)
        restore_btn.grid(row=0, column=1, padx=10)

<<<<<<< HEAD
        # Models label in second column
        model_label = ctk.CTkLabel(frame, text="Model Dropdown Menu Settings", font=ctk.CTkFont(size=20, weight="bold"))
        model_label.grid(row=0, column=1, pady=(20, 20))
        # Separator Models - Demucs
        demucs_models_label = ctk.CTkLabel(frame, text="Demucs Models (Edit/Reorder):", anchor="w")
        demucs_models_label.grid(row=1, column=1, sticky="w", padx=20, pady=(20, 0))
        self.demucs_models_text = ctk.CTkTextbox(frame, width=400, height=50)
        self.demucs_models_text.grid(row=2, column=1, sticky="ew", padx=20, pady=5)
        self.demucs_models_text.insert("0.0", json.dumps(self.separator_models.get("Demucs", [])))
        # Separator Models - OpenUnmix
        openunmix_models_label = ctk.CTkLabel(frame, text="OpenUnmix Models (Edit/Reorder):", anchor="w")
        openunmix_models_label.grid(row=3, column=1, sticky="w", padx=20, pady=(20, 0))
        self.openunmix_models_text = ctk.CTkTextbox(frame, width=400, height=50)
        self.openunmix_models_text.grid(row=4, column=1, sticky="ew", padx=20, pady=5)
        self.openunmix_models_text.insert("0.0", json.dumps(self.separator_models.get("OpenUnmix", [])))
        # Transcription Models - Whisper
        whisper_models_label = ctk.CTkLabel(frame, text="Whisper Models (Edit/Reorder):", anchor="w")
        whisper_models_label.grid(row=5, column=1, sticky="w", padx=20, pady=(20, 0))
        self.whisper_models_text = ctk.CTkTextbox(frame, width=400, height=50)
        self.whisper_models_text.grid(row=6, column=1, sticky="ew", padx=20, pady=5)
        self.whisper_models_text.insert("0.0", json.dumps(self.transcription_models.get("whisper", [])))
        # Transcription Models - Wav2Vec2
        wav2vec2_models_label = ctk.CTkLabel(frame, text="Wav2Vec2 Models (Edit/Reorder):", anchor="w")
        wav2vec2_models_label.grid(row=7, column=1, sticky="w", padx=20, pady=(20, 0))
        self.wav2vec2_models_text = ctk.CTkTextbox(frame, width=400, height=50)
        self.wav2vec2_models_text.grid(row=8, column=1, sticky="ew", padx=20, pady=5)
        self.wav2vec2_models_text.insert("0.0", json.dumps(self.transcription_models.get("wav2vec2", [])))
        # Transcription Models - Coqui
        coqui_models_label = ctk.CTkLabel(frame, text="Coqui Models (Edit/Reorder):", anchor="w")
        coqui_models_label.grid(row=9, column=1, sticky="w", padx=20, pady=(20, 0))
        self.coqui_models_text = ctk.CTkTextbox(frame, width=400, height=50)
        self.coqui_models_text.grid(row=10, column=1, sticky="ew", padx=20, pady=5)
        self.coqui_models_text.insert("0.0", json.dumps(self.transcription_models.get("coqui", [])))
        # Enable Evaluation Checkbox
        self.enable_eval_var = tk.BooleanVar(value=self.enable_evaluation)
        eval_checkbox = ctk.CTkCheckBox(frame, text="Enable Evaluation Tab", variable=self.enable_eval_var)
        eval_checkbox.grid(row=11, column=1, sticky="w", padx=20, pady=5)

=======
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
    def save_settings_changes(self):
        self.input_folder = self.settings_input_var.get()
        self.output_folders["vocals"] = self.settings_vocals_var.get()
        self.output_folders["instrumentals"] = self.settings_instr_var.get()
        self.output_folders["transcriptions"] = self.settings_trans_var.get()
<<<<<<< HEAD
        try:
            self.separator_models["Demucs"] = json.loads(self.demucs_models_text.get("0.0", "end"))
            self.separator_models["OpenUnmix"] = json.loads(self.openunmix_models_text.get("0.0", "end"))
            self.transcription_models["whisper"] = json.loads(self.whisper_models_text.get("0.0", "end"))
            self.transcription_models["wav2vec2"] = json.loads(self.wav2vec2_models_text.get("0.0", "end"))
            self.transcription_models["coqui"] = json.loads(self.coqui_models_text.get("0.0", "end"))
        except json.JSONDecodeError:
            messagebox.showerror("Error", "Invalid JSON in model fields.")
            return
=======
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
        self.save_settings()
        os.makedirs(self.input_folder, exist_ok=True)
        for folder in self.output_folders.values():
            os.makedirs(folder, exist_ok=True)
        self.load_input()
        self.load_outputs()
<<<<<<< HEAD
        # Refresh model dropdowns in the input tab to reflect new settings
        self.on_tool_change()
        self.on_trans_tool_change()
        messagebox.showinfo("Settings Saved", "Default folders and models updated and saved.")
=======
        messagebox.showinfo("Settings Saved", "Default folders updated and saved.")
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a

    def restore_defaults(self):
        defaults = {
            "input_folder": "input",
            "vocals_folder": "output/vocals",
            "instrumentals_folder": "output/instrumentals",
<<<<<<< HEAD
            "transcriptions_folder": "output/text",
            "enable_evaluation": False,
            "separator_models": {
                "Spleeter": [],
                "Demucs": ["mdx", "mdx_extra", "htdemucs"],
                "OpenUnmix": ["umxl", "umxhq", "umx", "umxse"]
            },
            "transcription_models": {
                "whisper": ["large", "medium", "small", "tiny", "base", "turbo"],
                "wav2vec2": ["facebook/wav2vec2-base-960h", "facebook/wav2vec2-large-960h"],
                "coqui": ["model.pbmm"]
            }
        }
        # Restore all attributes
        self.input_folder = defaults["input_folder"]
        self.output_folders = {
            "vocals": defaults["vocals_folder"],
            "instrumentals": defaults["instrumentals_folder"],
            "transcriptions": defaults["transcriptions_folder"]
        }
        self.enable_evaluation = defaults["enable_evaluation"]
        self.separator_models = defaults["separator_models"]
        self.transcription_models = defaults["transcription_models"]
        
        # Save to file
        self.save_settings()
        
        # Update UI variables
=======
            "transcriptions_folder": "output/text"
        }
        self.input_folder = defaults["input_folder"]
        self.output_folders = {k: v for k, v in defaults.items() if k != "input_folder"}
        self.save_settings()
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a
        self.settings_input_var.set(self.input_folder)
        self.settings_vocals_var.set(self.output_folders["vocals"])
        self.settings_instr_var.set(self.output_folders["instrumentals"])
        self.settings_trans_var.set(self.output_folders["transcriptions"])
<<<<<<< HEAD
        self.enable_eval_var.set(self.enable_evaluation)
        
        # Update model textboxes
        self.demucs_models_text.delete("0.0", "end")
        self.demucs_models_text.insert("0.0", json.dumps(self.separator_models["Demucs"]))
        self.openunmix_models_text.delete("0.0", "end")
        self.openunmix_models_text.insert("0.0", json.dumps(self.separator_models["OpenUnmix"]))
        self.whisper_models_text.delete("0.0", "end")
        self.whisper_models_text.insert("0.0", json.dumps(self.transcription_models["whisper"]))
        self.wav2vec2_models_text.delete("0.0", "end")
        self.wav2vec2_models_text.insert("0.0", json.dumps(self.transcription_models["wav2vec2"]))
        self.coqui_models_text.delete("0.0", "end")
        self.coqui_models_text.insert("0.0", json.dumps(self.transcription_models["coqui"]))
        
        # Ensure folders exist
        os.makedirs(self.input_folder, exist_ok=True)
        for folder in self.output_folders.values():
            os.makedirs(folder, exist_ok=True)
        
        # Reload data
        self.load_input()
        self.load_outputs()
        
        # Show success and switch to settings tab
        messagebox.showinfo("Defaults Restored", "All settings reset to defaults.")
        self.show_settings()

    def toggle_evaluation(self):
        self.enable_evaluation = self.enable_eval_var.get()
        if self.enable_evaluation:
            self.evaluation_button.grid()  # Add to sidebar
        else:
            # Remove evaluation button if exists
            self.evaluation_button.grid_remove()

    def create_evaluation_tab(self):
        frame = self.evaluation_frame
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        eval_label = ctk.CTkLabel(frame, text="Model Evaluation", font=ctk.CTkFont(size=20, weight="bold"))
        eval_label.grid(row=0, column=0, pady=(20, 20))
        
        # Single song multiple tools and models test
        single_song_frame = ctk.CTkFrame(frame)
        single_song_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        single_song_frame.grid_columnconfigure(1, weight=1)
        # Song selection
        song_label = ctk.CTkLabel(single_song_frame, text="Enter path to the song", anchor="w")
        song_label.grid(row=1, column=0, sticky="w", padx=20, pady=(10, 0))
        self.song_path = tk.StringVar(value=self.input_folder)
        song_path = ctk.CTkEntry(single_song_frame, textvariable=self.song_path, width=400)
        song_path.grid(row=2, column=0, sticky="ew", padx=20, pady=5)
        # Run Evaluation Button
        run_btn = ctk.CTkButton(single_song_frame, text="Run Evaluation", command=self.run_evaluation)
        run_btn.grid(row=3, column=0, pady=(20, 20))
        # Results Display
        results_label = ctk.CTkLabel(single_song_frame, text="Results:", anchor="w")
        results_label.grid(row=4, column=0, sticky="w", padx=20, pady=(10, 0))
        self.results_text = ctk.CTkTextbox(single_song_frame, width=400, height=200)
        self.results_text.grid(row=5, column=0, sticky="ew", padx=20, pady=5)

    def run_evaluation(self): 
        song_path = self.song_path.get()
        if not song_path:
            messagebox.showwarning("No Selection", "Please select a song.")
            return
        # Fixed parameters for comparison
        fmt, sr, bitrate = "wav", 44100, None
        results = {}
        for tool in ["Spleeter", "Demucs", "OpenUnmix"]:
            # Run separation (simplified, assuming no transcription)
            if tool == "Spleeter":
                success = self.spleeter_sep.separate(song['path'], os.path.splitext(song_name)[0], "/tmp/vocals", "/tmp/instr", fmt, sr, bitrate, False, None, None)
                vocals_path = "/tmp/vocals/vocals.wav"
                instr_path = "/tmp/instr/accompaniment.wav"
            # Add similar for Demucs and OpenUnmix
            if success:
                # Compute metrics (assuming ground-truth paths; adjust as needed)
                gt_vocals = song['path']  # Placeholder
                gt_instr = song['path']  # Placeholder
                sdr, sir, sar = museval.evaluate([vocals_path, instr_path], [gt_vocals, gt_instr])
                results[tool] = {"SDR": sdr.mean(), "SIR": sir.mean(), "SAR": sar.mean()}
        # Display results
        self.results_text.delete("0.0", "end")
        for tool, metrics in results.items():
            self.results_text.insert("end", f"{tool}: SDR={metrics['SDR']:.2f}, SIR={metrics['SIR']:.2f}, SAR={metrics['SAR']:.2f}\n")
=======
        os.makedirs(self.input_folder, exist_ok=True)
        for folder in self.output_folders.values():
            os.makedirs(folder, exist_ok=True)
        self.load_input()
        self.load_outputs()
        messagebox.showinfo("Defaults Restored", "Folders reset to defaults.")
>>>>>>> 92d0912b5e9008ba592c2475dd2727cd6f93247a

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

    def separate_audio(self):
        sel = self.songs_listbox.curselection()
        if not sel:
            messagebox.showwarning("No selection", "Please select a song to separate.")
            return
        idx = sel[0]
        item_type, item_data = self.all_items[idx]
        if item_type != 'song':
            messagebox.showwarning("Invalid selection", "Please select a song to separate, not a folder.")
            return

        song = item_data
        input_path = song['path']
        song_name = os.path.splitext(os.path.basename(song['name']))[0]

        ai_tool = self.ai_tool_var.get()
        do_transcribe = self.transcript_var.get()
        whisper_model = self.transcript_model.get()
        model = self.model_var.get()
        fmt = self.format_var.get()
        sr = int(self.sr_var.get()) if fmt in ["wav", "flac"] else None
        bitrate = int(self.bitrate_var.get()) if fmt == "mp3" else None
        bit_depth = self.bit_depth_var.get() if fmt == "wav" and ai_tool == "Demucs" else None
        mp3_preset = int(self.mp3_preset_slider.get()) if fmt == "mp3" and ai_tool == "Demucs" else None
        shifts = int(self.shifts_var.get()) if ai_tool == "Demucs" else None
        
        # Prepare output folders
        vocals_folder = self.output_folders["vocals"]
        instr_folder = self.output_folders["instrumentals"]
        trans_folder = self.output_folders["transcriptions"]

        # Start threaded separation with progress
        self.status_queue = queue.Queue()
        thread = threading.Thread(target=self.separate_in_thread, args=(input_path, song_name, vocals_folder, instr_folder, trans_folder, ai_tool, model, fmt, sr, bitrate, do_transcribe, whisper_model, song, bit_depth, mp3_preset, shifts))
        thread.daemon = True
        thread.start()
    
    def separate_in_thread(self, input_path, song_name, vocals_folder, instr_folder, trans_folder, ai_tool, model, fmt, sr, bitrate, do_transcribe, whisper_model, song, bit_depth, mp3_preset, shifts):
        progress = ProgressWindow(self, f"Separating with {ai_tool}...")
        try:
            self.status_queue.put("Loading model...")
            progress.update_status("Loading model")

            success = False
            if ai_tool == "Spleeter":
                self.status_queue.put("Separating using Spleeter...")
                progress.update_status("Separating using Spleeter")
                success = self.spleeter_sep.separate(input_path, song_name, vocals_folder, instr_folder, fmt, sr, bitrate, do_transcribe, trans_folder, whisper_model)
            elif ai_tool == "Demucs":
                self.status_queue.put("Separating using Demucs...")
                progress.update_status("Separating using Demucs")
                success = self.demucs_sep.separate(input_path, song_name, vocals_folder, instr_folder, model, fmt, sr, bitrate, bit_depth, mp3_preset, shifts, do_transcribe, trans_folder, whisper_model)
            elif ai_tool == "OpenUnmix":
                self.status_queue.put("Separating using OpenUnmix...")
                progress.update_status("Separating using OpenUnmix")
                success = self.openunmix_sep.separate(input_path, song_name, vocals_folder, instr_folder, model, fmt, sr, bitrate, do_transcribe, trans_folder, whisper_model)
            else:
                raise ValueError(f"Unknown AI tool: {ai_tool}")

            if progress.canceled:
                self.status_queue.put("Canceled")
                progress.update_status("Canceled")
                return

            self.status_queue.put("Saving files...")
            progress.update_status("Saving files")

            # Update UI on main thread
            self.after(0, self.load_outputs)
            self.after(0, lambda: messagebox.showinfo("Separation done", f"Separated '{song['name']}' using {ai_tool}."))
            self.status_queue.put("Success")
            progress.update_status("Separation done")
            return
        
        except Exception as e:
            print(f"Thread error: {e}")
            self.status_queue.put("Error")
            self.after(0, lambda: messagebox.showerror("Error", f"Unexpected error: {e}"))
        finally:
            progress.close(success=(self.status_queue.get() == "Success"))
                
if __name__ == "__main__":
    app = SeparationApp()
    app.mainloop()