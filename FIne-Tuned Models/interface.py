from PIL import Image, ImageTk
import tkinter as tk  # Import the tkinter module
from tkinter import ttk
import V2T2
import audiorec
import db
import os
import model
import threading

from datetime import datetime


def create_wind():
    global root
    root = tk.Tk()
    root.title("My Tkinter Window")
    root.configure(bg="darkgrey")
    root.geometry("1080x720")

    rec_but()  # Create button
    stop_but()
    viewt()
    out_put()
    root.mainloop()


def create_wind2():
    t = tk.Tk()
    t.title("DATABASE")
    t.configure(bg="black")
    t.geometry("1080x720")
    table(t)
    refresh(tree, db.fetch())
    t.mainloop()


def start():
    audiorec.start_recording()
    output_text.insert(tk.END, "🎙️ Recording started...\n")  # ✅ Moved here
    output_text.yview(tk.END)


def stop():
    audiorec.stop()
    output_text.insert(tk.END, "💾 Stopping and saving.\n")  # ✅ Moved here
    output_text.yview(tk.END)


def rec_but():
    global rec_pho  # Store image globally to prevent garbage collection
    rec_img = Image.open("play.png").resize((25, 25))  # Load and resize image
    rec_pho = ImageTk.PhotoImage(rec_img)  # Convert for Tkinter
    rec = tk.Button(root, image=rec_pho, text="record", compound="left",
                    command=start, font=("Arial", 20), fg="white", bg="green", padx=10, pady=5)
    rec.place(x=50, y=600)


def stop_but():
    global stop_pho  # Store image globally to prevent garbage collection
    stop_img = Image.open("pause.png").resize((25, 25))  # Load and resize image
    stop_pho = ImageTk.PhotoImage(stop_img)  # Convert for Tkinter
    stp = tk.Button(root, image=stop_pho, text="stop", compound="left",
                    command=stop, font=("Arial", 21), fg="white", bg="orange", padx=10, pady=5)
    stp.place(x=200, y=600)


def viewt():
    vt = tk.Button(root, text="ViewTable", command=create_wind2, font=("Arial", 20), fg="white", bg="blue", padx=5,
                   pady=5)
    vt.place(x=325, y=600)


def refresh(tree, rows):
    for item in tree.get_children():
        tree.delete(item)
    for row in rows:
        tree.insert("", "end", values=row)


def on_row_selected(event, tree, action_button):
    selected_item = tree.selection()
    if selected_item:
        row_data = tree.item(selected_item, "values")  # Get selected row values
        action_button.config(state=tk.NORMAL, command=lambda: perform_action(row_data))


def perform_action(row_data):
    def process_in_thread():
        file_path = row_data[2].strip()
        file_path = os.path.normpath(file_path)

        if not file_path.lower().endswith((".wav", ".mp3", ".flac")):
            file_path += ".wav"

        output_text.insert(tk.END, f"🔍 Opening: {repr(file_path)}\n")
        output_text.yview(tk.END)

        if os.path.exists(file_path):
            output_text.insert(tk.END, "🎙️ Transcribing...\n")
            n = datetime.now()
            f = n.strftime("%H:%M")
            output_text.insert(tk.END, f"\n🔍 time: {f}\n")  # ✅ Correct
            output_text.yview(tk.END)

            try:
                # Run speaker verification & transcription (SpeechBrain is now lazy-loaded inside model.py)

                res = model.process_audio(file_path, "processed_segments")


                output_text.insert(tk.END,n , "\n🔍 Transcription Results:\n")
                output_text.insert(tk.END, "-----------------------------\n")
                # Prints each transcription as it's processed

                for line in res:
                    output_text.insert(tk.END, f"{line}\n")
                elapsed_time = datetime.now() - n  # Time difference
                elapsed_seconds = elapsed_time.total_seconds()
                output_text.insert(tk.END, f"\n🔍 Time Taken: {elapsed_seconds:.2f} seconds\n")
                output_text.insert(tk.END, "-----------------------------\n")

                output_text.yview(tk.END)

            except Exception as e:
                output_text.insert(tk.END, f"❌ Error during transcription: {e}\n")

        else:
            output_text.insert(tk.END, f"❌ Error: File not found - {file_path}\n")

    threading.Thread(target=process_in_thread, daemon=True).start()


def table(t):
    global tree
    columns = ("ID", "Name", "PATH", "TIMESTAMP")
    tree = ttk.Treeview(t, columns=columns, show="headings")
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=150)
    action_button = tk.Button(t, text="transcribe", state=tk.DISABLED)
    action_button.pack(pady=10)
    tree.bind("<<TreeviewSelect>>", lambda event: on_row_selected(event, tree, action_button))
    tree.pack(expand=True, fill="both", padx=10, pady=10)
    refresh_button = tk.Button(t, text="Refresh Data", command=lambda: refresh(tree, db.fetch()))
    refresh_button.pack(pady=10)


def out_put():
    global output_text
    frame = tk.Frame(root, bg="black")
    frame.place(x=5, y=100, width=650, height=400)
    output_text = tk.Text(frame, fg="white", bg="black", wrap="word", width=65, height=20)
    output_text.pack(side=tk.LEFT, expand=True, fill="both")
    scrollbar = tk.Scrollbar(frame, command=output_text.yview)
    scrollbar.pack(side=tk.RIGHT, fill="y")
    output_text.config(yscrollcommand=scrollbar.set)
    out_head = tk.Label(root, text="output :", fg="white", bg="black")
    out_head.place(x=5, y=70)


# Start the application
create_wind()
