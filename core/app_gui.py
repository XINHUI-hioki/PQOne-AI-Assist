import queue
import threading
from typing import List
from matplotlib.axes import Axes
import tkinter as tk
from PIL import Image, ImageTk
from globals import GLOBAL  
from globals import lc_memory
from tkinter import ttk, scrolledtext, messagebox, filedialog, simpledialog 
import customtkinter as ctk 
import sys
import os
import pandas as pd
from tkhtmlview import HTMLLabel
import markdown
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from controller import start_generation_thread, append_summary_to_docx           
from usecase_extractors import (
    get_harmonics_df, 
    get_dip_df,get_thd_avg_df,
    analyze_uploaded_image,
    analyze_file,
    extract_all_rvc_events,
    get_thd_trend_df,
    get_thd_avgmax_df,
    extract_all_harmonics_from_docx,
    extract_all_voltage_dips

)
from utils.plot_utils import (                                
    plot_harmonics,
    plot_dips,
    embed_fig,
    plot_thd_avg_bar,
    plot_dip_timeline
)


CHART_REGISTRY = {
    "Harmonics Trend (Avg/Peak)": lambda d, p: plot_harmonics(d, p),
    "Voltage Dip Scatter (Depth vs Worst Voltage)":        lambda d, p: plot_dips(d, p),
    "Voltage Dip Timeline":       lambda d, p: plot_dip_timeline(d, p),
    "THD Avg Bar":                lambda d, p: plot_thd_avg_bar(d, p),
}


class ChartSelectDialog(simpledialog.Dialog):
    """Popup dialog → select desired charts via checkboxes"""

    def __init__(self, parent: tk.Misc, chart_names: List[str]) -> None:
        self._chart_names = chart_names
        self._vars: dict[str, tk.BooleanVar] = {}
        super().__init__(parent, title="Select Charts")

    def body(self, master: tk.Misc): 
        
        tk.Label(master, text="Check the charts to generate:").pack(anchor="w")
        for name in self._chart_names:
            var = tk.BooleanVar(value=False)
            tk.Checkbutton(master, text=name, variable=var).pack(anchor="w")
            self._vars[name] = var
        return None  

    def apply(self) -> None:
        self.result = [n for n, v in self._vars.items() if v.get()]

class OllamaTesterApp:

    def __init__(self, root: tk.Tk) -> None:
        self.hdf = self.ddf = self.tadf = None
        self.thd_trend_df = pd.DataFrame()
        self.uploaded_doc_paths: List[str] = []
        self.uploaded_files: List[str] = []
        self.summary_cache: list[str] = []
        self.df_dict: dict[str, pd.DataFrame] = {}
        self.last_generated_summary: str = ""

        self.root = root
        self.root.title("HiokiAssist – Power Quality AI Assistant")
        self.root.geometry("1000x850")
        self.root.configure(bg="#f5f7f9") 

        # —— Global style definition ——
        self.style = ttk.Style()
        self.style.theme_use("clam")

        # 滑Slider label font (Bold blue)
        self.style.configure("ParamLabel.TLabel",
            font=("Helvetica", 10, "bold"),
            foreground="#2f61ad",
            background="#ffffff"
        )

        # Current value text (the number on the right)
        self.style.configure("ParamValue.TLabel",
            font=("Helvetica", 10),
            background="#ffffff",
            foreground="#2f61ad"
        )

        # The control area on the left has a white background
        self.style.configure("Controls.TFrame", background="#ffffff")
        # The response area on the right is pure white
        self.style.configure("Response.TFrame", background="#ffffff")

        # Section style (light gray card with dark blue title)
        self.style.configure("Section.TLabelframe",
                             background="#fafbfc",
                             foreground="#2e5aac",
                             font=("Helvetica", 10, "bold"),
                             borderwidth=1, relief="solid")
        self.style.configure("Section.TLabelframe.Label",
                             background="#fafbfc",
                             foreground="#2e5aac",
                             font=("Helvetica", 10, "bold"))
        self.models = [
            "HiokiAssist",
            "Gennect One Assist",
            "Gennect Cross Assist",
            "*NEW: Gennect Space",
            "PQ ONE Assist"
        ]
        self.language_options = [
            "English",
            "中文",
            "日本語",
        ]

        # Main button (dark blue background with white characters)
        self.style.configure("Primary.TButton",
                             background="#2f61ad",
                             foreground="white",
                             font=("Helvetica", 10, "bold"),
                             padding=6,
                             borderwidth=0)
        self.style.map("Primary.TButton",
                       background=[("active", "#1e417a")])

        # Secondary button (white background with gray outline)
        self.style.configure("Secondary.TButton",
                             background="#ffffff",
                             foreground="#666666",
                             bordercolor="#d3d3d3",
                             borderwidth=1,
                             relief="solid",
                             font=("Helvetica", 9))
        self.style.map("Secondary.TButton",
                       background=[("active", "#f5f5f5")])
        
        self.style.configure(
            "Secondary.TCheckbutton",
            background="#ffffff",      
            foreground="#2e5aac",      
            font=("Helvetica", 9),     
            indicatorcolor="#2e5aac",  
            indicatorsize=16,          
            padding=(5, 3)             
        )
        self.style.map(
            "Secondary.TCheckbutton",
            background=[("active", "#f5f5f5")],  
            foreground=[("disabled", "#BCC6DD")]
        )

        #generalized label
        self.style.configure("TLabel",
                             background="#ffffff",
                             foreground="#2e5aac",
                             font=("Helvetica", 10))
        self.style.configure("Small.TLabel",
                             background="#ffffff",
                             foreground="#666666",
                             font=("Helvetica", 9))

        # Entry / Text Box
        self.style.configure("TEntry",
                             fieldbackground="white",
                             foreground="black",
                             font=("Helvetica", 10))

        # slider
        self.style.configure("TScale",
                             troughcolor="#d3d3d3",
                             background="#2f61ad")
        self.style.configure(
            "Thin.TScale",
            troughcolor="#e0e0e0",   
            sliderthickness=16,       
            background="#2f61ad"     
        )
        self.style.map(
            "Thin.TScale",
            background=[("active", "#1e417a")]  
        )

        # ——  Left control area  Frame ——  
        self.controls_frame = ttk.Frame(
            root,
            width=330,
            style="Controls.TFrame"
        )
        self.controls_frame.pack(
            side=tk.LEFT,
            fill=tk.Y,
            padx=20,
            pady=20,
            anchor="nw"
        )
        self.controls_frame.columnconfigure(0, weight=1)

        # —— Right response area Frame ——  
        self.response_frame = ttk.Frame(
            root,
            style="Response.TFrame"
        )
        self.response_frame.pack(
            side=tk.RIGHT,
            fill=tk.BOTH,
            expand=True,
            padx=20,
            pady=20
        )

        # ——  Start queue processing ——  
        self.response_queue: queue.Queue[str] = queue.Queue()
        self.root.after(100, self._process_queue)

        # —— Create sub-regions ——
        self._create_controls()
        self._create_response_area()

        # —— Status bar (supports logo placement ——
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Frame(  
            root,
            style="TFrame",           
            padding=5
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # To the left of the status bar: Status text
        ttk.Label(
            self.status_bar,
            textvariable=self.status_var,
            anchor="w",
            style="Small.TLabel"
        ).pack(side="left")

        self.memory_log: List[str] = []
        self._ui_queue: "queue.Queue[tuple[callable, tuple, dict]]" = queue.Queue()
        self.root.after(100, self._process_queue)

    def _create_controls(self) -> None:
        # ———Roll containers ———
        self.controls_canvas = tk.Canvas(
            self.controls_frame,
            highlightthickness=0,
            bg="#f0f4f8"
        )
        scrollbar = tk.Scrollbar(
            self.controls_frame,
            orient="vertical",
            command=self.controls_canvas.yview,
            width=16
        )
        self.scrollable_frame = ttk.Frame(
            self.controls_canvas,
            style="Controls.TFrame"
        )

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.controls_canvas.configure(
                scrollregion=self.controls_canvas.bbox("all")
            )
        )
        self.canvas_window = self.controls_canvas.create_window(
            (0, 0),
            window=self.scrollable_frame,
            anchor="nw"
        )
        self.controls_canvas.configure(yscrollcommand=scrollbar.set)
        self.controls_canvas.bind(
            "<Configure>",
            self._resize_scrollable_frame
        )

        # pack Canvas and scroll bar
        self.controls_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # ——— Prompt input ———
        prompt_frame = ttk.Frame(
            self.scrollable_frame,
            style="Controls.TFrame"
        )
        prompt_frame.pack(fill="x", padx=10, pady=(0, 8))
        prompt_frame.columnconfigure(0, weight=1)

        ttk.Label(
            prompt_frame,
            text="Enter Prompt:",
            font=("Helvetica", 11, "bold"),
            foreground="#3164c3",
            background="#ffffff"
        ).grid(row=0, column=0, sticky="w")

        ttk.Button(
            prompt_frame,
            text="Reset Conversation",
            command=self._reset_memory,
            style="Secondary.TButton"
        ).grid(row=0, column=1, sticky="e", padx=(5, 0))

        self.prompt_text = scrolledtext.ScrolledText(
            prompt_frame,
            height=10,
            wrap=tk.WORD,
            font=("Helvetica", 10),
            background="white",
            foreground="black",
            borderwidth=1,
            relief="solid"
        )
        self.prompt_text.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(5, 0)
        )

        # ——— Conversation Memory & Beginner Mode  ———
        self.memory_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.scrollable_frame,
            text="  Enable Conversation Memory",
            variable=self.memory_mode,
            style="Secondary.TCheckbutton"
        ).pack(anchor="w", padx=10, pady=2)

        self.beginner_mode = tk.BooleanVar(
            value=getattr(GLOBAL.config, "beginner_mode", False)
        )
        def _toggle_beginner_mode():
            from globals import set_beginner_mode
            set_beginner_mode(self.beginner_mode.get())

        ttk.Checkbutton(
            self.scrollable_frame,
            text="  Beginner Mode (PQ ONE help)",
            variable=self.beginner_mode,
            style="Secondary.TCheckbutton",
            command=_toggle_beginner_mode
        ).pack(anchor="w", padx=10, pady=2)

        # ——— Model & Language Select ———
        selector_row = ttk.Frame(
            self.scrollable_frame,
            style="Controls.TFrame"
        )
        selector_row.pack(fill="x", padx=10, pady=(5, 2))
        selector_row.columnconfigure(0, weight=1)
        selector_row.columnconfigure(1, weight=1)

        ttk.Label(
            selector_row,
            text="Select Model:",
            font=("Helvetica", 10),
            background="#f0f4f8",
            foreground="#2e5aac"
        ).grid(row=0, column=0, sticky="w", padx=(0, 5))
        ttk.Label(
            selector_row,
            text="Select Language:",
            font=("Helvetica", 10),
            background="#f0f4f8",
            foreground="#3164c3"
        ).grid(row=0, column=1, sticky="w", padx=(5, 0))

        menu_row = ttk.Frame(
            self.scrollable_frame,
            style="Controls.TFrame"
        )
        menu_row.pack(fill="x", padx=10, pady=(2, 10))
        menu_row.columnconfigure(0, weight=1)
        menu_row.columnconfigure(1, weight=1)

        self.models = [
            "HiokiAssist",
            "Gennect One Assist",
            "Gennect Cross Assist",
            "*NEW: Gennect Space",
            "PQ ONE Assist"
        ]
        self.model_var = tk.StringVar(value=self.models[0])
        ttk.OptionMenu(
            menu_row,
            self.model_var,
            self.models[0],
            *self.models
        ).grid(row=0, column=0, sticky="ew", padx=(0, 5))

        self.language_options = ["English", "中文", "日本語"]
        self.language_var = tk.StringVar(value=self.language_options[0])
        ttk.OptionMenu(
            menu_row,
            self.language_var,
            self.language_options[0],
            *self.language_options
        ).grid(row=0, column=1, sticky="ew", padx=(5, 0))

        # ——— parameter silder ———
        self._create_parameter_slider(
            self.scrollable_frame, "Temperature:", 0.0, 2.0, 0.8, "temp_var"
        )
        self._create_parameter_slider(
            self.scrollable_frame, "Top P:", 0.0, 1.0, 0.9, "top_p_var"
        )
        self._create_parameter_slider(
            self.scrollable_frame,
            "Frequency Penalty (Repeat Penalty):",
            0.0, 2.0, 1.1, "freq_penalty_var"
        )

        # ——— max size & upload file ———
        ttk.Label(
            self.scrollable_frame,
            text="Maximum Length:",
            font=("Helvetica", 10, "bold"),
            background="#ffffff",
            foreground="#2e5aac"
        ).pack(fill="x", padx=10, pady=(8, 3))
        self.max_length_var = tk.IntVar(value=512)
        ttk.Entry(
            self.scrollable_frame,
            textvariable=self.max_length_var
        ).pack(fill="x", padx=10)

        ttk.Label(
            self.scrollable_frame,
            text="Upload Report Files:",
            font=("Helvetica", 10, "bold"),
            background="#ffffff",
            foreground="#3164c3"
        ).pack(fill="x", padx=10, pady=(8, 3))
        ttk.Button(
            self.scrollable_frame,
            text="Select .docx Files",
            command=self._upload_files,
            style="Secondary.TButton"
        ).pack(fill="x", padx=10, pady=5)

        self.uploaded_files_label = ttk.Label(
            self.scrollable_frame,
            text="No files uploaded yet.",
            wraplength=250,
            font=("Helvetica", 9),
            foreground="gray",
            background="#f0f4f8"
        )
        self.uploaded_files_label.pack(fill="x", padx=10, pady=(3, 8))

        # ——— Generate & Analyze button ———
        self.generate_button = ttk.Button(
            self.scrollable_frame,
            text="Generate",
            command=self._on_send,
            style="TButton"
        )
        self.generate_button.pack(fill="x", padx=10, pady=(0, 5))

        self.analyze_button = ttk.Button(
            self.scrollable_frame,
            text="Auto Analyze",
            command=self.start_analysis_thread,
            style="TButton"
        )
        self.analyze_button.pack(fill="x", padx=10, pady=(0, 10))

        # ——— Dip / THD alert ———
        self._create_parameter_slider(
            self.scrollable_frame, "Dip alert voltage (V):", 80, 230, 180, "dip_threshold_var"
        )
        self._create_parameter_slider(
            self.scrollable_frame, "THD alert (%):", 0, 10, 5, "thd_threshold_var"
        )

        # ——— chart & cache  ———
        ttk.Button(
            self.scrollable_frame,
            text="Show Charts",
            command=self._show_charts,
            style="TButton"
        ).pack(fill="x", padx=10, pady=6)

        self.add_cache_button = ttk.Button(
            self.scrollable_frame,
            text="Add Last Response to Cache",
            command=self._on_add_to_cache,
            state="disabled",
            style="Secondary.TButton"
        )
        self.add_cache_button.pack(fill="x", padx=10, pady=2)

        ttk.Label(
            self.scrollable_frame,
            text="Cached Summaries:",
            font=("Helvetica", 10, "bold"),
            background="#ffffff",
            foreground="#3164c3"
        ).pack(fill="x", padx=10, pady=(6, 2))

        self.cache_listbox = tk.Listbox(
            self.scrollable_frame,
            selectmode=tk.MULTIPLE,
            height=6,
            bg="white",
            fg="black"
        )
        self.cache_listbox.pack(fill="x", padx=10, pady=(0, 6))
        self.cache_listbox.bind(
            "<<ListboxSelect>>",
            lambda e: self.append_button.config(
                state="normal" if self.cache_listbox.curselection() else "disabled"
            )
        )

        self.append_button = ttk.Button(
            self.scrollable_frame,
            text="Append Summary to Document",
            command=self._on_append_summary,
            state="disabled",
            style="Secondary.TButton"
        )
        self.append_button.pack(fill="x", padx=10, pady=2)

    def _create_response_area(self) -> None:        
        ttk.Label(
            self.response_frame,
            text="Model Response:",
            font=("Helvetica", 11, "bold"),
            foreground="#3467c7",
            background="#ffffff"
        ).pack(anchor=tk.W, padx=10, pady=(0, 5))

        # Add a card-style border to the response content
        self.response_card = ttk.Frame(
            self.response_frame,
            style="ResponseCard.TFrame"
        )
        self.response_card.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Card style (white background with thin borders)
        self.style.configure(
            "ResponseCard.TFrame",
            background="#ffffff",
            borderwidth=1,
            relief="solid"
        )

        if hasattr(self, "response_logo_img"):
            logo_h = self.response_logo_img.height()
        else:
            from PIL import Image, ImageTk
            logo_img = Image.open("assets/hioki_logo_32.png")
            self.response_logo_img = ImageTk.PhotoImage(logo_img)
            logo_h = self.response_logo_img.height()

            # logo Label
            self.response_logo = ttk.Label(
                self.response_card,
                image=self.response_logo_img,
                background="#ffffff",
                cursor="hand2"
            )
            self.response_logo.place(relx=1.0, rely=1.0, anchor="se", x=-8, y=-8)
            self.response_logo.lift()

            # click event
            def show_hioki_info(event):
                import tkinter.messagebox as msg
                msg.showinfo(
                    "Hioki Website",
                    "➡️ Hioki official website:\nhttps://www.hioki.com"
                )
        self.response_logo.bind("<Button-1>", show_hioki_info)
        if hasattr(self, "response_label"):
            self.response_label.destroy()

        from tkhtmlview import HTMLLabel
        self.response_label = HTMLLabel(
            self.response_card,
            html="",
            background="#ffffff",
            foreground="#333333"
        )
        self.response_label.pack(
            fill=tk.BOTH,
            expand=True,
            padx=8,
            pady=(8, 8 + logo_h)    
        )


    def _create_parameter_slider(self, parent, label, from_, to, initial_value, var_name):
        row = ttk.Frame(parent, style="Controls.TFrame") 
        row.pack(fill="x", padx=10, pady=(6, 2))
        ttk.Label(row, text=label, style="ParamLabel.TLabel").pack(side="left", anchor="w")

        value_label = ttk.Label(row, text=f"{initial_value:.2f}", style="ParamValue.TLabel", width=5)
        value_label.pack(side="right")

        slider = RoundSlider(parent, from_=from_, to=to, initial=initial_value, width=280,
                            command=lambda v: value_label.config(text=f"{v:.2f}"))
        slider.pack(fill="x", padx=10)

        setattr(self, var_name, slider)
        setattr(self, f"{var_name}_label", value_label)
    def _resize_scrollable_frame(self, event) -> None:
        self.controls_canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event) -> None:
        self.controls_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _reset_memory(self) -> None:
        from globals import lc_memory  
        try:
            lc_memory.chat_memory.clear()
            print("[INFO] Memory cleared.")
        except Exception as e:
            print(f"[ERROR] Failed to clear memory: {e}")


    def _upload_files(self) -> None:
        """Open file-chooser, parse selected reports / images, and store DataFrames."""
        # --- reset in-memory holders -------------------------------------------------
        self.hdf = self.ddf = self.tadf = None                  
        self.thd_trend_df = pd.DataFrame()                      
        self.uploaded_doc_paths: list[str] = []
        self.uploaded_files: list[str] = []
        self.df_dict: dict[str, dict[str, pd.DataFrame | None]] = {}

        # --- file dialog -------------------------------------------------------------
        file_paths: tuple[str, ...] = filedialog.askopenfilenames(
            title="Select PQ ONE reports or images",
            filetypes=[
                ("Word Documents", "*.docx"),
                ("PNG Image", "*.png"),
                ("JPG Image", "*.jpg"),
                ("JPEG Image", "*.jpeg"),
                ("All Files", "*.*"),
            ],
        )
        if not file_paths:
            if self.uploaded_files_label is not None:
                self.uploaded_files_label.config(
                    text="No files uploaded yet.", foreground="gray"
                )
            return

        # --- update sidebar file list ------------------------------------------------
        self.uploaded_doc_paths = list(file_paths)
        file_names: str = "\n".join(f"✓ {os.path.basename(p)}" for p in self.uploaded_doc_paths)
        if self.uploaded_files_label is not None:
            self.uploaded_files_label.config(text=f"Uploaded Files:\n{file_names}", foreground="black")

        for path in self.uploaded_doc_paths:
            ext = os.path.splitext(path)[1].lower()
            self.uploaded_files.append(path)

            if ext == ".docx":
                tbls: dict[str, pd.DataFrame | None] = {}
                print(f"\n[INFO] Parsing {os.path.basename(path)} …")

                try:
                    tbls["harmonics"], _ = get_harmonics_df(path)
                except Exception as exc:
                    print(f"[WARN] Harmonics parse failed for {path}: {exc}")
                    tbls["harmonics"] = None

                try:
                    tbls["dip"] = get_dip_df(path)
                except Exception as exc:
                    print(f"[WARN] Dip parse failed for {path}: {exc}")
                    tbls["dip"] = None

                # thd_avg  (单 df) ---------------------------------
                try:
                    _df, thd_dict = extract_all_harmonics_from_docx(path)
                    if thd_dict:
                        thd_df = pd.DataFrame([
                            {"phase": k, "THD_avg": v.get("avg"), "THD_max": v.get("max")}
                            for k, v in thd_dict.items()
                        ])
                        tbls["thd_avg"] = thd_df
                    else:
                        tbls["thd_avg"] = None
                except Exception as exc:
                    print(f"[WARN] extract_all_harmonics_from_docx failed for {path}: {exc}")
                    tbls["thd_avg"] = None

                try:
                    tbls["thd_trend"] = get_thd_trend_df(path)
                except Exception as exc:
                    print(f"[WARN] THD Trend parse failed for {path}: {exc}")
                    tbls["thd_trend"] = None

                self.df_dict[path] = tbls
                ok_tbls = [k for k, v in tbls.items() if isinstance(v, pd.DataFrame) and not v.empty]
                print("     ✔ Loaded tables:", ok_tbls if ok_tbls else "None")

            elif ext in [".png", ".jpg", ".jpeg"]:
                print(f"\n[INFO] Analyzing image {os.path.basename(path)} …")
                try:
                    ocr_result = analyze_uploaded_image(path, self.language_var.get())
                    self.df_dict[path] = {"ocr_summary": ocr_result}
                    print(f"     ✔ OCR Summary:\n{ocr_result}")
                except Exception as exc:
                    print(f"[WARN] Image analysis failed: {exc}")
                    self.df_dict[path] = {"ocr_summary": None}

        latest = self.df_dict[self.uploaded_doc_paths[-1]]

        if "harmonics" in latest:                     # latest is a .docx
            self.hdf         = latest.get("harmonics")
            self.ddf         = latest.get("dip")
            self.tadf        = latest.get("thd_avg")
            self.thd_trend_df = latest.get("thd_trend") or pd.DataFrame()
        else:                                         # latest is an image
            self.hdf = self.ddf = self.tadf = None
            self.thd_trend_df = pd.DataFrame()
            print("[INFO] Latest file is an image. Tables not loaded.")

    def _on_send(self) -> None:
        prompt = self.prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            messagebox.showwarning("Input Error", "Prompt cannot be empty.")
            return

        self.generate_button.config(state="disabled")
        self.status_var.set("Generating... Please wait.")

        self.response_label.set_html("")     
        self.add_cache_button.config(state="disabled")


        threading.Thread(
            target=start_generation_thread,
            args=(
                prompt,
                self.language_var.get(),
                self.model_var.get(),
                self.memory_mode.get(),
                self.uploaded_files,
                self.df_dict,
                self.response_queue,
                {
                    "temp": self.temp_var.get(),
                    "top_p": self.top_p_var.get(),
                    "freq_penalty": self.freq_penalty_var.get(),
                    "max_len": self.max_length_var.get(),
                    "dip_th": self.dip_threshold_var.get(),
                    "thd_th": self.thd_threshold_var.get(),
                },
            ),
            daemon=True,
        ).start()
        self.last_generated_summary = None  

    def start_analysis_thread(self) -> None:
        if not self.uploaded_doc_paths:
            messagebox.showwarning("No file", " please upload .docx report first。")
            return

        # UI 状态
        self.analyze_button.config(state="disabled")
        self.status_var.set("Analyzing… Please wait.")

        self.response_label.set_html("")
        self._response_accumulator = ""  


        self.ddf = None
        self.tadf = None
        self.rvc_df = None
        self.harm_df = None

        threading.Thread(
            target=self._analyze_in_thread, daemon=True
        ).start()

    def load_analysis_data(self, path: str) -> None:
        self.ddf, _ = get_dip_df(path)
        self.tadf   = get_thd_avgmax_df(path)
        self.rvc_df = extract_all_rvc_events(path)  
        self.harm_df, _ = get_harmonics_df(path)

    def _analyze_in_thread(self) -> None:
        """
        Parse every uploaded .docx report, rebuild cached DataFrames
        (dips / THD / RVC / harmonics), then push summary + LLM
        recommendations to the UI thread via self.response_queue.
        """
        from controller import generate_llm_recommendations
        import pandas as pd

        def _to_df(obj) -> pd.DataFrame:
            """Convert list/dict/None/DataFrame → safe DataFrame"""
            if obj is None:
                return pd.DataFrame()
            if isinstance(obj, pd.DataFrame):
                return obj.copy()
            try:
                return pd.DataFrame(obj)
            except Exception:
                return pd.DataFrame()

        try:
            lang: str = self.language_var.get()
            outputs: list[str] = []

            for path in self.uploaded_doc_paths:
                # ─────────────────────────────── 1) Voltage Dips ─────────────
                try:
                    raw_dips = extract_all_voltage_dips(path)
                except Exception as exc:
                    print(f"[Dip extraction failed] {exc}", flush=True)
                    raw_dips = None
                self.ddf = _to_df(raw_dips)

                # ─────────────────────────────── 2) THD AVG / MAX ────────────
                try:
                    raw_thd = get_thd_avgmax_df(path)
                except Exception as exc:
                    print(f"[THD extraction failed] {exc}", flush=True)
                    raw_thd = None
                self.tadf = _to_df(raw_thd)

                # ─────────────────────────────── 3) RVC Events ───────────────
                try:
                    raw_rvc = extract_all_rvc_events(path)
                except NameError:
                    raw_rvc = None
                except Exception as exc:
                    print(f"[RVC extraction failed] {exc}", flush=True)
                    raw_rvc = None
                self.rvc_df = _to_df(raw_rvc)

                # ─────────────────────────────── 4) Harmonics (full) ─────────
                try:
                    raw_harm_df, _ = extract_all_harmonics_from_docx(path)
                except NameError:
                    raw_harm_df = None
                except Exception as exc:
                    print(f"[Harmonics extraction failed] {exc}", flush=True)
                    raw_harm_df = None
                self.harm_df = _to_df(raw_harm_df)

                # ─────────────────────────────── 5) 文本摘要 + LLM 建议 ───────
                summary: str = analyze_file(path, lang)

                short_summary: str = (
                    summary.split("=== Auto-Recommendations ===")[0]
                    if "=== Auto-Recommendations ===" in summary
                    else summary
                )
                recommendations: str = generate_llm_recommendations(
                    short_summary, lang
                )

                combined_text: str = (
                    summary.strip()
                    + "\n\n=== Auto-Recommendations ===\n"
                    + recommendations.strip()
                )
                outputs.append(combined_text)

            # 汇总多份报告的输出
            final_txt: str = "\n\n" + ("─" * 10) + "\n\n".join(outputs)
            self.response_queue.put(final_txt)

        except Exception as exc:
            print("[Auto-Analyze ERROR]", exc, flush=True)
            self.response_queue.put(f"[Auto-Analyze Error] {exc}")

        finally:
            self.response_queue.put(None)
            self.analyze_button.config(state="normal")
            self.status_var.set("Ready")
   
    def _process_queue(self) -> None:
        try:
            while True:
                token = self.response_queue.get_nowait()

                if isinstance(token, str):
                    if not hasattr(self, "_response_accumulator"):
                        self._response_accumulator = ""
                    self._response_accumulator += token

                    self.add_cache_button.config(state="normal")

                    html = markdown.markdown(self._response_accumulator, extensions=["extra"])
                    self.response_label.set_html(html)
                elif token is None:
                    self.generate_button.config(state="normal")
                    self.status_var.set("Ready")

                    if hasattr(self, "_response_accumulator"):
                        self.last_generated_summary = self._response_accumulator.strip()
                        del self._response_accumulator  

                    break

        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._process_queue)

    def _show_charts(self) -> None:
        if self.hdf is None and self.ddf is None and self.tadf is None:
            messagebox.showinfo("No Data", "Please upload a valid report.")
            return


        dlg = ChartSelectDialog(self.root, list(CHART_REGISTRY.keys()))
        chosen = dlg.result or []
        if not chosen:
            return

        if hasattr(self, "_chart_win") and self._chart_win.winfo_exists():
            for c in self._chart_win.winfo_children():
                c.destroy()
        else:
            self._chart_win = tk.Toplevel(self.root)
            self._chart_win.title("PQ Charts")
            self._chart_win.geometry("960x720")

        canvas = tk.Canvas(self._chart_win, highlightthickness=0)
        vsb = tk.Scrollbar(self._chart_win, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        df_dict = {
            "harm": self.hdf,      
            "dip":  self.ddf,     
            "thd":  self.thd_trend_df,       
            "thd_avg": self.tadf, 
        }
        param_dict = {
            "dip_th":  self.dip_threshold_var.get(),
            "thd_th":  self.thd_threshold_var.get(),
        }

        for name in chosen:
            try:
                fig = CHART_REGISTRY[name](df_dict, param_dict)
                embed_fig(fig, inner)
            except Exception as e:
                import traceback, sys, tkinter.messagebox as mbox
                traceback.print_exc(file=sys.stderr)
                mbox.showerror("Plot Error", str(e))

    def _on_append_summary(self) -> None:
        if not self.uploaded_doc_paths:
            messagebox.showwarning("No File", "Please upload a .docx report first.")
            return

        selected_idx = self.cache_listbox.curselection()
        if not selected_idx:
            messagebox.showwarning("No Selection", "Please select summaries from the cache.")
            return

        selected_texts = [self.summary_cache[i] for i in selected_idx]
        full_text = "\n\n---\n\n".join(selected_texts)

        try:
            from controller import append_summary_to_docx

            latest_path = self.uploaded_doc_paths[-1]          
            new_path = append_summary_to_docx(latest_path, full_text)

            messagebox.showinfo("Success", f"Summary appended to:\n{new_path}")
            self.append_button.config(state="disabled")         
            self.cache_listbox.selection_clear(0, tk.END)      

        except Exception as exc:
            print("[Append Summary Error]", exc)
            messagebox.showerror("Error", f"Failed to append summary.\n{exc}")

    def _on_add_to_cache(self) -> None:
        if not self.last_generated_summary:
            messagebox.showwarning("No Summary", "No generated response to add.")
            return

        self.summary_cache.append(self.last_generated_summary)

        idx = len(self.summary_cache)
        self.cache_listbox.insert(tk.END, f"Summary #{idx}")

        self.cache_listbox.selection_clear(0, tk.END)
        self.cache_listbox.selection_set(tk.END)

        self.add_cache_button.config(state="disabled")

class RoundSlider(tk.Canvas):
    def __init__(self, master, from_, to, initial, width=200, command=None, **kwargs):
        super().__init__(master, width=width, height=26, bg=kwargs.get("bg", "#ffffff"), highlightthickness=0)
        self.from_ = from_
        self.to = to
        self.command = command
        self.slider_radius = 5
        self.width = width
        self.height = 26
        self.value = initial

        self.track_color = "#e6eaf0"
        self.thumb_color = "#2f61ad"
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<Button-1>", self._on_drag)

        self._draw()

    def _draw(self):
        self.delete("all")
        y = self.height // 2
        self.create_line(12, y, self.width - 12, y, fill=self.track_color, width=4)

        x = 12 + (self.width - 24) * ((self.value - self.from_) / (self.to - self.from_))
        self.create_oval(x - self.slider_radius, y - self.slider_radius,
                         x + self.slider_radius, y + self.slider_radius,
                         fill=self.thumb_color, outline="")

    def _on_drag(self, event):
        x = min(max(event.x, 12), self.width - 12)
        ratio = (x - 12) / (self.width - 24)
        self.value = self.from_ + (self.to - self.from_) * ratio
        self._draw()
        if self.command:
            self.command(self.value)

    def get(self):
        return self.value

    def set(self, value):
        self.value = value
        self._draw()

if __name__ == "__main__":
    try:
        app = OllamaTesterApp(tk.Tk())
        app.root.mainloop()
    except Exception as exc:  
        print("An error occurred while initializing the GUI.")
        print("Please ensure you have a valid desktop environment and Tkinter is installed correctly.")
        print(f"Error: {exc}")
