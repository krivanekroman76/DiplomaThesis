import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import platform
import subprocess

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

        # Data lists
        self.songs = []
        self.vocals = []
        self.instrumentals = []
        self.transcriptions = []

        # Create tab view for 2 pages
        self.tabview = ctk.CTkTabview(self, width=1100, height=650)
        self.tabview.pack(padx=10, pady=10, fill="both", expand=True)

        self.tabview.add("Input")
        self.tabview.add("Output")

        self.create_input_tab()
        self.create_output_tab()

        # Load initial songs and outputs
        self.load_songs()
        self.load_outputs()

    def create_input_tab(self):
        tab = self.tabview.tab("Input")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_columnconfigure(1, weight=0)
        tab.grid_rowconfigure(1, weight=1)

        # Songs list
        self.songs_listbox = tk.Listbox(tab)
        self.songs_listbox.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.songs_listbox.bind("<Double-Button-1>", self.open_selected_song)

        # Buttons frame under songs list
        btn_frame = ctk.CTkFrame(tab)
        btn_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0,10))
        btn_frame.grid_columnconfigure((0,1), weight=1)

        self.add_song_button = ctk.CTkButton(btn_frame, text="Add Song", command=self.add_song)
        self.add_song_button.grid(row=0, column=0, sticky="ew", padx=5)

        self.change_input_folder_button = ctk.CTkButton(btn_frame, text="Change Folder", command=self.change_input_folder)
        self.change_input_folder_button.grid(row=0, column=1, sticky="ew", padx=5)

        # Separation menu frame (right side)
        sep_frame = ctk.CTkFrame(tab, width=300)
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
        tab = self.tabview.tab("Output")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure((1,3,5), weight=1)

        # Transcriptions section
        trans_label = ctk.CTkLabel(tab, text="Transcriptions", font=ctk.CTkFont(size=18, weight="bold"))
        trans_label.grid(row=0, column=0, sticky="w", padx=10, pady=(10,5))

        trans_btn = ctk.CTkButton(tab, text="Change Folder", command=lambda: self.change_output_folder("transcriptions"))
        trans_btn.grid(row=0, column=1, sticky="e", padx=10, pady=(10,5))

        self.trans_listbox = tk.Listbox(tab)
        self.trans_listbox.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10)
        self.trans_listbox.bind("<Double-Button-1>", self.open_selected_transcription)

        # Vocals section
        vocals_label = ctk.CTkLabel(tab, text="Vocals", font=ctk.CTkFont(size=18, weight="bold"))
        vocals_label.grid(row=2, column=0, sticky="w", padx=10, pady=(20,5))

        vocals_btn = ctk.CTkButton(tab, text="Change Folder", command=lambda: self.change_output_folder("vocals"))
        vocals_btn.grid(row=2, column=1, sticky="e", padx=10, pady=(20,5))

        self.vocals_listbox = tk.Listbox(tab)
        self.vocals_listbox.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10)
        self.vocals_listbox.bind("<Double-Button-1>", self.open_selected_vocal)

        # Instrumentals section
        instr_label = ctk.CTkLabel(tab, text="Instrumentals", font=ctk.CTkFont(size=18, weight="bold"))
        instr_label.grid(row=4, column=0, sticky="w", padx=10, pady=(20,5))

        instr_btn = ctk.CTkButton(tab, text="Change Folder", command=lambda: self.change_output_folder("instrumentals"))
        instr_btn.grid(row=4, column=1, sticky="e", padx=10, pady=(20,5))

        self.instr_listbox = tk.Listbox(tab)
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

        ai_tool = self.ai_tool_var.get()
        do_transcribe = self.transcript_var.get()

        # Determine suffixes for AI tool
        suffix_map = {
            "Spleeter": "_S",
            "Demucs": "_D",
            "OpenUnmix": "_O"
        }
        ai_suffix = suffix_map.get(ai_tool, "_S")

        # Prepare output folders
        vocals_folder = self.output_folders["vocals"]
        instr_folder = self.output_folders["instrumentals"]
        trans_folder = self.output_folders["transcriptions"]

        # Call the appropriate separation function
        try:
            if ai_tool == "Spleeter":
                from Spleeter import separate_S
                vocals_files, instr_files, transcription_files = separate_S(
                    song['path'], vocals_folder, instr_folder, trans_folder, do_transcribe, ai_suffix)
            elif ai_tool == "Demucs":
                from demucs import separate_D
                vocals_files, instr_files, transcription_files = separate_D(
                    song['path'], vocals_folder, instr_folder, trans_folder, do_transcribe, ai_suffix)
            elif ai_tool == "OpenUnmix":
                from open_umix import separate_O
                vocals_files, instr_files, transcription_files = separate_O(
                    song['path'], vocals_folder, instr_folder, trans_folder, do_transcribe, ai_suffix)
            else:
                messagebox.showerror("Error", f"Unknown AI tool: {ai_tool}")
                return
        except Exception as e:
            messagebox.showerror("Separation Error", f"Error during separation:\n{e}")
            return

        # Update output lists
        self.load_outputs()

        msg = f"Separated '{song['name']}' using {ai_tool}."
        if do_transcribe:
            msg += "\nTranscription generated."
        messagebox.showinfo("Separation done", msg)

if __name__ == "__main__":
    app = SeparationApp()
    app.mainloop()
