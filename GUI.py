import os
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import platform
import subprocess

#import file-handle.py
#import deezer.py
#import demucs.py
#import open-umix.py

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

def open_audio_file(path):
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.call(("open", path))
    else:  # Linux and others
        subprocess.call(("xdg-open", path))

class SeparationApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Audio Separation Tool")
        self.geometry("1100x600")

        # Configure main window grid: 1 row, 2 columns
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Data storage
        self.songs = []
        self.vocals = []
        self.instrumentals = []
        self.transcriptions = []

        # --- Input widget/frame ---
        self.input_frame = ctk.CTkFrame(self, corner_radius=10)
        self.input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.input_frame.grid_rowconfigure(1, weight=1)  # Songs list expands
        self.input_frame.grid_columnconfigure(0, weight=1)

        # Songs label
        self.songs_label = ctk.CTkLabel(self.input_frame, text="Songs", font=ctk.CTkFont(size=18, weight="bold"))
        self.songs_label.grid(row=0, column=0, pady=(10,5), sticky="w", padx=10)

        # Songs listbox
        self.songs_listbox = tk.Listbox(self.input_frame, activestyle='dotbox')
        self.songs_listbox.grid(row=1, column=0, sticky="nsew", padx=10)
        self.songs_listbox.bind("<Double-Button-1>", self.open_selected_song)

        # Add song button
        self.add_song_button = ctk.CTkButton(self.input_frame, text="Add Song", command=self.add_song)
        self.add_song_button.grid(row=2, column=0, pady=10, padx=10, sticky="ew")

        # Separation menu label
        self.sep_label = ctk.CTkLabel(self.input_frame, text="Separation Menu", font=ctk.CTkFont(size=18, weight="bold"))
        self.sep_label.grid(row=3, column=0, pady=(20,10), sticky="w", padx=10)

        # AI tool radio buttons
        self.ai_tool_var = tk.StringVar(value="Spleeter")
        self.radio_spleeter = ctk.CTkRadioButton(self.input_frame, text="Spleeter", variable=self.ai_tool_var, value="Spleeter")
        self.radio_demucs = ctk.CTkRadioButton(self.input_frame, text="Demucs", variable=self.ai_tool_var, value="Demucs")
        self.radio_openunmix = ctk.CTkRadioButton(self.input_frame, text="OpenUnmix", variable=self.ai_tool_var, value="OpenUnmix")

        self.radio_spleeter.grid(row=4, column=0, sticky="w", padx=20, pady=5)
        self.radio_demucs.grid(row=5, column=0, sticky="w", padx=20, pady=5)
        self.radio_openunmix.grid(row=6, column=0, sticky="w", padx=20, pady=5)

        # Transcription checkbox
        self.transcript_var = tk.BooleanVar(value=False)
        self.transcript_checkbox = ctk.CTkCheckBox(self.input_frame, text="Transcribe vocals", variable=self.transcript_var)
        self.transcript_checkbox.grid(row=7, column=0, sticky="w", padx=20, pady=10)

        # Separate button
        self.separate_button = ctk.CTkButton(self.input_frame, text="Separate", command=self.separate_audio)
        self.separate_button.grid(row=8, column=0, sticky="ew", padx=20, pady=(10,20))

        # --- Output widget/frame ---
        self.output_frame = ctk.CTkFrame(self, corner_radius=10)
        self.output_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.output_frame.grid_rowconfigure(1, weight=1)  # Transcriptions list expands vertically
        self.output_frame.grid_rowconfigure(3, weight=1)  # Vocals list expands vertically
        self.output_frame.grid_rowconfigure(5, weight=1)  # Instrumentals list expands vertically
        self.output_frame.grid_columnconfigure(0, weight=1)

        # Transcriptions label and listbox
        self.trans_label = ctk.CTkLabel(self.output_frame, text="Transcriptions", font=ctk.CTkFont(size=18, weight="bold"))
        self.trans_label.grid(row=0, column=0, pady=(10,5), sticky="w", padx=10)

        self.trans_listbox = tk.Listbox(self.output_frame)
        self.trans_listbox.grid(row=1, column=0, sticky="nsew", padx=10)
        self.trans_listbox.bind("<Double-Button-1>", self.open_selected_transcription)

        # Vocals label and listbox
        self.vocals_label = ctk.CTkLabel(self.output_frame, text="Vocals", font=ctk.CTkFont(size=18, weight="bold"))
        self.vocals_label.grid(row=2, column=0, pady=(20,5), sticky="w", padx=10)

        self.vocals_listbox = tk.Listbox(self.output_frame)
        self.vocals_listbox.grid(row=3, column=0, sticky="nsew", padx=10)
        self.vocals_listbox.bind("<Double-Button-1>", self.open_selected_vocal)

        # Instrumentals label and listbox
        self.instr_label = ctk.CTkLabel(self.output_frame, text="Instrumentals", font=ctk.CTkFont(size=18, weight="bold"))
        self.instr_label.grid(row=4, column=0, pady=(20,5), sticky="w", padx=10)

        self.instr_listbox = tk.Listbox(self.output_frame)
        self.instr_listbox.grid(row=5, column=0, sticky="nsew", padx=10)
        self.instr_listbox.bind("<Double-Button-1>", self.open_selected_instrumental)

    def add_song(self):
        filetypes = [("Audio files", "*.mp3 *.wav *.flac *.m4a"), ("All files", "*.*")]
        paths = filedialog.askopenfilenames(title="Select audio files", filetypes=filetypes)
        for path in paths:
            name = os.path.basename(path)
            if not any(song['path'] == path for song in self.songs):
                self.songs.append({'path': path, 'name': name})
                self.songs_listbox.insert(tk.END, name)

    def open_selected_song(self, event=None):
        selection = self.songs_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        path = self.songs[index]['path']
        open_audio_file(path)

    def open_selected_vocal(self, event=None):
        selection = self.vocals_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        path = self.vocals[index]['path']
        open_audio_file(path)

    def open_selected_instrumental(self, event=None):
        selection = self.instr_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        path = self.instrumentals[index]['path']
        open_audio_file(path)

    def open_selected_transcription(self, event=None):
        selection = self.trans_listbox.curselection()
        if not selection:
            return
        index = selection[0]
        transcription_name = self.transcriptions[index]['name']
        messagebox.showinfo("Transcription", f"Transcription file:\n{transcription_name}\n\n(Opening not implemented)")

    def separate_audio(self):
        selection = self.songs_listbox.curselection()
        if not selection:
            messagebox.showwarning("No selection", "Please select a song to separate.")
            return
        index = selection[0]
        song = self.songs[index]

        ai_tool = self.ai_tool_var.get()
        do_transcribe = self.transcript_var.get()

        # Simulate separation output paths
        base, ext = os.path.splitext(song['path'])
        vocal_path = base + "_vocals" + ext
        instr_path = base + "_instrumental" + ext

        vocal_entry = {'path': vocal_path, 'name': os.path.basename(vocal_path)}
        instr_entry = {'path': instr_path, 'name': os.path.basename(instr_path)}

        if vocal_entry not in self.vocals:
            self.vocals.append(vocal_entry)
            self.vocals_listbox.insert(tk.END, vocal_entry['name'])
        if instr_entry not in self.instrumentals:
            self.instrumentals.append(instr_entry)
            self.instr_listbox.insert(tk.END, instr_entry['name'])

        if do_transcribe:
            transcription_path = base + "_transcription.txt"
            transcription_entry = {'path': transcription_path, 'name': os.path.basename(transcription_path)}
            if transcription_entry not in self.transcriptions:
                self.transcriptions.append(transcription_entry)
                self.trans_listbox.insert(tk.END, transcription_entry['name'])

        msg = f"Separated '{song['name']}' using {ai_tool}."
        if do_transcribe:
            msg += "\nTranscription generated."
        messagebox.showinfo("Separation done", msg)

if __name__ == "__main__":
    app = SeparationApp()
    app.mainloop()