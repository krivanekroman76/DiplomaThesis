import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import platform
import subprocess
import shlex  # For safe command splitting
import tempfile  # For temporary directories
from demucs.separate import main as demucs_main  # For Demucs (fallback if needed)
# Import separator classes (adjust path if separators/ is not in same dir)
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
        
        # Instantiate separator objects (ADD THIS BLOCK)
        self.spleeter_sep = spleeter.SpleeterSeparator()
        self.demucs_sep = demucs.DemucsSeparator()
        self.openunmix_sep = openunmix.OpenUnmixSeparator()

        # Data lists
        self.songs = []
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
        self.input_button = ctk.CTkButton(
            self.sidebar, 
            text="Input", 
            command=self.show_input,
            width=180
        )
        self.input_button.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        self.output_button = ctk.CTkButton(
            self.sidebar, 
            text="Output", 
            command=self.show_output,
            width=180
        )
        self.output_button.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        # Spacer row (expands to push settings to bottom)
        # (No widget here, just the configuration above)

        # Appearance mode selection (bottom-aligned)
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=3, column=0, padx=20, pady=(20, 0), sticky="w")

        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(
            self.sidebar, 
            values=["Light", "Dark", "System"],
            command=self.change_appearance_mode_event,
            width=160
        )
        self.appearance_mode_optionemenu.grid(row=4, column=0, padx=20, pady=(10, 10), sticky="ew")
        self.appearance_mode_optionemenu.set("Dark")

        # UI Scaling (Zoom) selection (bottom-aligned)
        self.scaling_label = ctk.CTkLabel(self.sidebar, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=5, column=0, padx=20, pady=(10, 0), sticky="w")

        self.scaling_optionemenu = ctk.CTkOptionMenu(
            self.sidebar, 
            values=["80%", "90%", "100%", "110%", "120%"],
            command=self.change_scaling_event,
            width=160
        )
        self.scaling_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 20), sticky="ew")
        self.scaling_optionemenu.set("100%")

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
        self.load_songs()
        self.load_outputs()

    def show_input(self):
        self.input_frame.grid(row=0, column=0, sticky="nsew")
        self.output_frame.grid_forget()
        # Optional: Highlight active button
        self.input_button.configure(fg_color=("#DCE4EE", "#1f538d"))
        self.output_button.configure(fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])

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
        frame.grid_rowconfigure(1, weight=1)

        # Songs list
        self.songs_listbox = tk.Listbox(frame)
        self.songs_listbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.songs_listbox.bind("<Double-Button-1>", self.open_selected_song)

        # Buttons frame under songs list
        btn_frame = ctk.CTkFrame(frame)
        btn_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0,10))
        btn_frame.grid_columnconfigure((0,1), weight=1)

        self.add_song_button = ctk.CTkButton(btn_frame, text="Add Song", command=self.add_song)
        self.add_song_button.grid(row=0, column=0, sticky="ew", padx=5)

        self.change_input_folder_button = ctk.CTkButton(btn_frame, text="Change Folder", command=self.change_input_folder)
        self.change_input_folder_button.grid(row=0, column=1, sticky="ew", padx=5)

        # Separation menu frame (right side)
        sep_frame = ctk.CTkFrame(frame, width=300)
        sep_frame.grid(row=1, column=1, rowspan=2, sticky="nsew", padx=10, pady=10)
        sep_frame.grid_columnconfigure(0, weight=1)

        sep_label = ctk.CTkLabel(sep_frame, text="Separation Menu", font=ctk.CTkFont(size=20, weight="bold"))
        sep_label.grid(row=0, column=0, pady=(10,20))

        self.ai_tool_var = tk.StringVar(value="Spleeter")
        self.radio_spleeter = ctk.CTkRadioButton(sep_frame, text="Spleeter", variable=self.ai_tool_var, value="Spleeter")
        self.radio_demucs = ctk.CTkRadioButton(sep_frame, text="Demucs", variable=self.ai_tool_var, value="Demucs")
        self.radio_openunmix = ctk.CTkRadioButton(sep_frame, text="OpenUnmix", variable=self.ai_tool_var, value="OpenUnmix")

        self.radio_spleeter.grid(row=1, column=0, sticky="w", padx=20, pady=5)
        self.radio_demucs.grid(row=2, column=0, sticky="w", padx=20, pady=5)
        self.radio_openunmix.grid(row=3, column=0, sticky="w", padx=20, pady=5)

        self.transcript_var = tk.BooleanVar(value=False)
        self.transcript_checkbox = ctk.CTkCheckBox(sep_frame, text="Transcribe vocals", variable=self.transcript_var)
        self.transcript_checkbox.grid(row=4, column=0, sticky="w", padx=20, pady=10)

        self.separate_button = ctk.CTkButton(sep_frame, text="Separate", command=self.separate_audio)
        self.separate_button.grid(row=5, column=0, sticky="ew", padx=20, pady=(20,10))

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

    def load_songs(self):
        self.songs_listbox.delete(0, tk.END)
        self.songs.clear()
        if not os.path.isdir(self.input_folder):
            return
        for f in sorted(os.listdir(self.input_folder)):
            if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                full_path = os.path.join(self.input_folder, f)
                self.songs.append({'path': full_path, 'name': f})
                self.songs_listbox.insert(tk.END, f)

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
        self.load_songs()

    def change_input_folder(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_folder = folder
            self.load_songs()

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
        song = self.songs[idx]
        input_path = song['path']
        song_name = os.path.splitext(os.path.basename(song['name']))[0]

        ai_tool = self.ai_tool_var.get()
        do_transcribe = self.transcript_var.get()

        # Suffix for uniqueness
        suffix_map = {"Spleeter": "_S", "Demucs": "_D", "OpenUnmix": "_O"}
        ai_suffix = suffix_map.get(ai_tool, "_S")

        # Prepare output folders
        vocals_folder = self.output_folders["vocals"]
        instr_folder = self.output_folders["instrumentals"]
        trans_folder = self.output_folders["transcriptions"]

        # Delegate to separator
        success = False
        if ai_tool == "Spleeter":
            success = self.spleeter_sep.separate(input_path, song_name, ai_suffix, vocals_folder, instr_folder)
        elif ai_tool == "Demucs":
            success = self.demucs_sep.separate(input_path, song_name, ai_suffix, vocals_folder, instr_folder)
        elif ai_tool == "OpenUnmix":
            success = self.openunmix_sep.separate(input_path, song_name, ai_suffix, vocals_folder, instr_folder)
        else:
            messagebox.showerror("Error", f"Unknown AI tool: {ai_tool}")
            return

        if not success:
            messagebox.showerror("Separation Error", f"{ai_tool} failed. Check console for details.")
            return

        # Handle transcription (placeholder)
        if do_transcribe:
            trans_path = os.path.join(trans_folder, f"{song_name}{ai_suffix}_transcription.txt")
            with open(trans_path, "w") as f:
                f.write(f"Transcription for {song_name} (using {ai_tool}):\n")
                f.write("Note: Implement actual transcription (e.g., with Whisper) here.\n")
                f.write("Placeholder: Lyrics not available yet.")

        # Update output lists
        self.load_outputs()

        msg = f"Separated '{song['name']}' using {ai_tool}."
        if do_transcribe:
            msg += "\nTranscription generated (placeholder)."
        messagebox.showinfo("Separation done", msg)

if __name__ == "__main__":
    app = SeparationApp()
    app.mainloop()