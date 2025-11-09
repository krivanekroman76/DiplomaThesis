import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import platform
import subprocess
import threading
import queue

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

        # Default folders
        self.input_folder = os.path.abspath("input")
        self.output_folders = {
            "vocals": os.path.abspath("output/vocals"),
            "instrumentals": os.path.abspath("output/instrumentals"),
            "transcriptions": os.path.abspath("output/text")
        }
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
        self.all_items = []  # New: Tracks all listbox items (folders and songs)
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
        self.sidebar.grid_rowconfigure(2, weight=1)  # Spacer row to push settings down

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

        # Spacer row (expands to push settings to bottom)
        # (No widget here, just the configuration above)

        # Appearance mode selection (bottom-aligned)
        appearance_mode_label = ctk.CTkLabel(self.sidebar, text="Appearance Mode:", anchor="w")
        appearance_mode_label.grid(row=3, column=0, padx=20, pady=(20, 0), sticky="w")

        appearance_mode_optionemenu = ctk.CTkOptionMenu(
            self.sidebar, 
            values=["Light", "Dark", "System"],
            command=self.change_appearance_mode_event,
            width=160
        )
        appearance_mode_optionemenu.grid(row=4, column=0, padx=20, pady=(10, 10), sticky="ew")
        appearance_mode_optionemenu.set("Dark")

        # UI Scaling (Zoom) selection (bottom-aligned)
        scaling_label = ctk.CTkLabel(self.sidebar, text="UI Scaling:", anchor="w")
        scaling_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")

        scaling_optionemenu = ctk.CTkOptionMenu(
            self.sidebar, 
            values=["80%", "90%", "100%", "110%", "120%"],
            command=self.change_scaling_event,
            width=160
        )
        scaling_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 20), sticky="ew")
        scaling_optionemenu.set("100%")

        # Content frame
        self.content_frame = ctk.CTkFrame(main_frame)
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

        # Input and output frames
        self.input_frame = ctk.CTkFrame(self.content_frame)
        self.output_frame = ctk.CTkFrame(self.content_frame)

        # Create tab contents
        self.create_input_tab()
        self.create_output_tab()

        # Initially show input
        self.show_input()

        # Load initial songs and outputs
        self.load_input()
        self.load_outputs()

        # Buttons in input tab
        self.input_button = input_button
        self.output_button = output_button
    
    def show_input(self):
        self.input_frame.grid(row=0, column=0, sticky="nsew")
        self.output_frame.grid_forget()


    def show_output(self):
        self.output_frame.grid(row=0, column=0, sticky="nsew")
        self.input_frame.grid_forget()
        # Optional: Highlight active button
        self.output_button.configure(fg_color=("#DCE4EE", "#1f538d"))
        self.input_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])

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
        self.transcript_checkbox = ctk.CTkCheckBox(sep_scrollable, text="Transcribe vocals", variable=self.transcript_var, command=self.on_tool_change)
        self.transcript_checkbox.grid(row=11, column=0, sticky="w", padx=20, pady=10)

        self.transcription_frame = ctk.CTkFrame(sep_scrollable)
        self.transcription_frame.grid(row=12, column=0, sticky="ew", padx=20, pady=5)
        self.transcription_frame.grid_remove()  # Hide initially
        
        # Model selection
        self.tarns_model_label = ctk.CTkLabel(self.transcription_frame, text="Model:", anchor="w")
        self.tarns_model_label.grid(row=1, column=0, sticky="w", padx=20, pady=(10,0))
        self.transcript_model = tk.StringVar(value="tiny")
        self.transcript_model_menu = ctk.CTkOptionMenu(
            self.transcription_frame, variable=self.transcript_model, values=["large", "medium", "small", "tiny", "base", "turbo", "large-v1", "large-v2", "large-v3", "large-v3-turbo"], width=200
        )
        self.transcript_model_menu.grid(row=13, column=0, sticky="ew", padx=20, pady=5)
        self.transcript_model_menu.grid_remove()

        # Separate button
        self.separate_button = ctk.CTkButton(sep_scrollable, text="Separate", command=self.separate_audio)
        self.separate_button.grid(row=14, column=0, sticky="ew", padx=20, pady=(20,10))

        # Initial tool change to set defaults
        self.on_tool_change()

    def update_mp3_preset_label(self, value):
        self.mp3_preset_value_label.configure(text=f"Current: {int(value)}")

    def on_tool_change(self):
        tool = self.ai_tool_var.get()
        trans_tool = self.transcript_var.get()
        if tool == "Spleeter":
            self.model_menu.grid_remove()  # Hide model selection for Spleeter
        elif tool == "Demucs":
            self.model_menu.grid()  # Show model selection
            self.model_menu.configure(values=["mdx", "mdx_extra", "htdemucs"])
            self.model_var.set("mdx") # Set default model
        elif tool == "OpenUnmix":
            self.model_menu.grid()
            self.model_menu.configure(values=["umxl", "umxhq", "umx", "umxse"])
            self.model_var.set("umxl")
        #Transcription model selection
        if trans_tool:
            self.transcription_frame.grid()
            self.transcript_model_menu.grid()
        else:
            self.transcription_frame.grid_remove()
            self.transcript_model_menu.grid_remove()

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

    def change_input_folder(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_folder = folder
            self.load_input()

    def change_output_folder(self, filetype):
        folder = filedialog.askdirectory(title=f"Select {filetype.capitalize()} Output Folder")
        if folder:
            self.output_folders[filetype] = folder
            self.load_outputs()

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