#!/usr/bin/env python3
"""Main Tk application shell for the DRIPP dataset debugger."""

if __package__ in (None, ""):
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    __package__ = "dripp.debugger"

from .common import *
from .csv_debugger import CsvDebuggerMixin
from .grouping_debugger import GroupingDebuggerMixin
from .preprocessing_debugger import PreprocessingDebuggerMixin

class Tooltip:
    """
    Show a small tooltip near the mouse cursor.

    Args:
        None.

    Returns:
        None.
    """
    def __init__(self, widget):
        """
        Initialize the object.

        Args:
            widget (Any): Tk widget that owns the tooltip.

        Returns:
            None.
        """
        self.widget = widget
        self.tipwindow = None

    def show(self, text, x, y):
        """
        Display the tooltip text near a screen position.

        Args:
            text (Any): Text to display.
            x (Any): Screen x-coordinate.
            y (Any): Screen y-coordinate.

        Returns:
            None.
        """
        if self.tipwindow:
            self.hide()
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        label = ttk.Label(tw, text=text, background='yellow', relief='solid', borderwidth=1)
        label.pack()
        tw.wm_geometry(f"+{x+20}+{y+20}")  # Offset from cursor

    def hide(self):
        """
        Hide the tooltip window.

        Args:
            None.

        Returns:
            None.
        """
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None

class PreprocessingDebuggerApp(GroupingDebuggerMixin, CsvDebuggerMixin, PreprocessingDebuggerMixin):
    """
    Run the Tk-based DRIPP grouping and preprocessing debugger.

    Args:
        None.

    Returns:
        None.
    """
    def __init__(self, root, csv_path=os.path.join(BASE_UNPROC, CSV_FILENAME)):
        """
        Initialize the object.

        Args:
            root (Any): Tk root window.
            csv_path (Any): Path to the datasets CSV.

        Returns:
            None.
        """
        self.root = root
        root.title("Dataset Preprocessing Debugger")
        root.geometry("1920x1080")

        # Tracks whether we're in batch mode
        self.is_batch = False

        # Remember which group is currently loaded, so we don't reload it
        self.current_group = None

        self.csv_path = csv_path
        self.metadata = self._load_metadata_or_prompt_for_csv(csv_path)
        self.original_regex = None

        # Notebook for tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: CSV editor
        self._build_csv_tab()

        # Tab 2: Regex Tester
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Regex Tester")

        style = ttk.Style(root)
        def fixed_map(option):
            """
            Return a Treeview style map without disabled-selected states.

            Args:
                option (Any): option.

            Returns:
                list: Filtered style map entries.
            """
            m = style.map('Treeview', query_opt=option)
            return [elm for elm in m if elm[:2] != ('!disabled','!selected')]
        style.map('Treeview',
            foreground=fixed_map('foreground'),
            background=fixed_map('background')
        )

        # Controls frame
        controls = ttk.Frame(self.tab1)
        controls.pack(fill="x", padx=5, pady=5)
        for i in range(1,4): controls.grid_columnconfigure(i, weight=1)

        # Dataset selector
        ttk.Label(controls, text="Dataset:").grid(row=0, column=0, sticky="w")
        self.dataset_var = tk.StringVar()
        self.ds_combo = ttk.Combobox(controls, textvariable=self.dataset_var,
                                     values=list(self.metadata.keys()), state="readonly")
        self.ds_combo.grid(row=0, column=1, sticky="we", padx=2)
        self.ds_combo.bind("<<ComboboxSelected>>", self.on_dataset_select)

        # Grouping Regex
        ttk.Label(controls, text="Grouping Regex:").grid(row=1, column=0, sticky="nw")
        self.regex_text = tk.Text(controls, height=3)
        self.regex_text.grid(row=1, column=1, columnspan=3, sticky="we", padx=2)
        self.regex_text.bind("<KeyRelease>", self.on_regex_change)

        # Strategy
        ttk.Label(controls, text="Strategy:").grid(row=0, column=2, sticky="w")
        self.strategy_var = tk.StringVar()
        strategy_combo = ttk.Combobox(controls, textvariable=self.strategy_var,
                                      values=["regex-file", "regex-folder", "filename"], state="readonly")
        strategy_combo.grid(row=0, column=3, sticky="we", padx=2)

        # Image Exts
        ttk.Label(controls, text="Image Exts:").grid(row=2, column=0, sticky="w")
        self.img_ext_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.img_ext_var).grid(row=2, column=1, sticky="we", padx=2)

        # Mask Exts
        ttk.Label(controls, text="Mask Exts:").grid(row=2, column=2, sticky="w")
        self.mask_ext_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.mask_ext_var).grid(row=2, column=3, sticky="we", padx=2)

        # Base Dir
        ttk.Label(controls, text="Base Dir:").grid(row=3, column=0, sticky="w")
        self.base_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.base_var, state="readonly").grid(row=3, column=1, columnspan=2, sticky="we", padx=2)
        ttk.Button(controls, text="Browse...", command=self.browse_directory).grid(row=3, column=3, sticky="e", padx=2)

        # Buttons frame
        btns = ttk.Frame(self.tab1)
        btns.pack(fill="x", padx=5, pady=5)
        ttk.Button(btns, text="Parse Regex", command=self.parse_configs).pack(side="left", padx=2)
        ttk.Button(btns, text="Test", command=self.test_regex).pack(side="right", padx=2)
        ttk.Label(btns, text="Sort:").pack(side="right", padx=(0,5))
        self.sort_var = tk.StringVar(value="None")
        ttk.Combobox(btns, textvariable=self.sort_var,
                     values=["None", "Subdataset Name", "Identifier"], state="readonly").pack(side="right", padx=2)
        self.save_btn = ttk.Button(btns, text="Save Regex", command=self.save_regex)
        self.save_btn.pack(side="right", padx=2)
        self.save_btn.state(['disabled'])
        self.save_json_btn = ttk.Button(btns, text="Save JSON", command=self.save_json)
        self.save_json_btn.pack(side="right", padx=2)
        self.save_json_btn.state(['disabled'])

        # Top frame: Parsed configs left, Pythex tester right
        top_frame = ttk.Frame(self.tab1)
        top_frame.pack(fill="both", padx=5, pady=5)
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)

        # Parsed configs box (now a collapsible tree)
        parsed_frame = ttk.LabelFrame(top_frame, text="Parsed SubdatasetConfigs")
        parsed_frame.grid(row=0, column=0, sticky="nsew", padx=2)

        columns = ('value',)
        self.config_tree = ttk.Treeview(parsed_frame, columns=columns, show='tree headings')
        # Configure columns
        self.config_tree.column('#0', width=120, anchor='w')
        self.config_tree.heading('#0', text='Config')
        self.config_tree.column('value', width=800, minwidth=300, stretch=True, anchor='w')
        self.config_tree.heading('value', text='Value')

        vsb = ttk.Scrollbar(parsed_frame, orient="vertical", command=self.config_tree.yview)
        self.config_tree.configure(yscrollcommand=vsb.set)
        self.config_tree.pack(side='left', fill='both', expand=True, padx=(5,0), pady=5)
        hsb = ttk.Scrollbar(parsed_frame, orient='horizontal', command=self.config_tree.xview)
        self.config_tree.configure(xscrollcommand=hsb.set)
        hsb.pack(side='bottom', fill='x', padx=(5,0), pady=(0,5))
        vsb.pack(side='right', fill='y', padx=(0,5), pady=5)

        # Right-click "Copy Value" on any row with a non-empty value
        self.config_menu = tk.Menu(self.config_tree, tearoff=0)
        self.config_menu.add_command(label="Copy Value",
                                     command=self._copy_config_value)

        def _show_config_context(event):
            """
            Show the config-tree context menu for copyable rows.

            Args:
                event (Any): Tk event object.

            Returns:
                None.
            """
            iid = self.config_tree.identify_row(event.y)
            if not iid:
                return
            vals = self.config_tree.item(iid, "values")
            if not vals or not vals[0]:
                return
            # Select the row and pop up
            self.config_tree.selection_set(iid)
            self.config_menu.tk_popup(event.x_root, event.y_root)
            self.config_menu.grab_release()

        self.config_tree.bind("<Button-3>", _show_config_context)

        # Pythex tester panel (simplified) with blue outline
        blue_frame = tk.Frame(top_frame, highlightbackground="blue", highlightthickness=1)
        blue_frame.grid(row=0, column=1, sticky="nsew", padx=2)
        px_frame = ttk.LabelFrame(blue_frame, text="Pythex Tester")
        px_frame.pack(fill="both", expand=True, padx=2, pady=2)
        ttk.Label(px_frame, text="Regex:").pack(anchor="w", padx=5, pady=(5,0))
        self.px_regex = tk.Text(px_frame, height=3)
        self.px_regex.pack(fill="x", padx=5)
        ttk.Label(px_frame, text="Test String:").pack(anchor="w", padx=5, pady=(5,0))
        self.px_test_str = scrolledtext.ScrolledText(px_frame, wrap="word", height=4)
        self.px_test_str.pack(fill="both", expand=True, padx=5, pady=(0,5))
        ttk.Button(px_frame, text="Test", command=self.px_test).pack(pady=(0,5))
        self.px_results = scrolledtext.ScrolledText(px_frame, wrap="none", height=6)
        self.px_results.pack(fill="both", expand=True, padx=5, pady=(0,5))

        # Three columns layout
        content = ttk.Frame(self.tab1)
        content.pack(fill="both", expand=True, padx=5, pady=5)
        for i in range(3): content.columnconfigure(i, weight=1)
        content.rowconfigure(0, weight=1)

        # Resolved
        res_frame = ttk.LabelFrame(content, text="Resolved Folders & Files")
        res_frame.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        self.resolved_tree = ttk.Treeview(res_frame)
        self._configure_tags(self.resolved_tree)
        res_scroll = ttk.Scrollbar(res_frame, orient="vertical", command=self.resolved_tree.yview)
        self.resolved_tree.configure(yscrollcommand=res_scroll.set)
        self.resolved_tree.pack(side="left", fill="both", expand=True)
        res_scroll.pack(side="right", fill="y")

        # Training
        train_frame = ttk.LabelFrame(content, text="Training Groups")
        train_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        self.train_tree = ttk.Treeview(train_frame)
        self._configure_tags(self.train_tree)
        tr_scroll = ttk.Scrollbar(train_frame, orient="vertical", command=self.train_tree.yview)
        self.train_tree.configure(yscrollcommand=tr_scroll.set)
        self.train_tree.pack(side="left", fill="both", expand=True)
        tr_scroll.pack(side="right", fill="y")

        # Testing
        test_frame = ttk.LabelFrame(content, text="Testing Groups")
        test_frame.grid(row=0, column=2, sticky="nsew", padx=2, pady=2)
        self.test_tree = ttk.Treeview(test_frame)
        self._configure_tags(self.test_tree)
        te_scroll = ttk.Scrollbar(test_frame, orient="vertical", command=self.test_tree.yview)
        self.test_tree.configure(yscrollcommand=te_scroll.set)
        self.test_tree.pack(side="left", fill="both", expand=True)
        te_scroll.pack(side="right", fill="y")

        self._configure_tags(self.resolved_tree)
        self._add_tree_context_menu(self.resolved_tree)

        self._configure_tags(self.train_tree)
        self._add_tree_context_menu(self.train_tree)

        self._configure_tags(self.test_tree)
        self._add_tree_context_menu(self.test_tree)

        # -- Tab 3: Preprocessing UI -------------------------------------
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Preprocessing")

        # Header: Dataset dropdown + Preprocess button
        self.preproc_controls = ttk.Frame(self.tab2)
        self.preproc_controls.pack(fill="x", padx=10, pady=5)
        ttk.Label(self.preproc_controls, text="Dataset:").pack(side="left")
        self.preproc_dataset_var = tk.StringVar()
        self.preproc_ds_combo = ttk.Combobox(
            self.preproc_controls,
            textvariable=self.preproc_dataset_var,
            values=list(self.metadata.keys()),
            state="readonly",
            width=50
        )
        self.preproc_ds_combo.pack(side="left", padx=5)
        self.preproc_ds_combo.bind("<<ComboboxSelected>>", self.on_preproc_dataset_select)

        # Load already-preprocessed groups (enabled only if folder exists)
        self.load_pre_btn = ttk.Button(
            self.preproc_controls,
            text="Load Preprocessed",
            state="disabled",
            command=self.load_preprocessed
        )
        self.load_pre_btn.pack(side="left", padx=(5, 20))

        # -- Max Groups spinner -----------------------------
        self.max_groups_var = tk.IntVar(value=2)
        ttk.Label(self.preproc_controls, text="Max Groups:").pack(side="left", padx=5)
        ttk.Spinbox(
            self.preproc_controls,
            from_=0, to=9999,
            textvariable=self.max_groups_var,
            width=5
        ).pack(side="left", padx=(0,10))
        # ---------------------------------------------------

        self.preprocess_btn = ttk.Button(
            self.preproc_controls,
            text="Preprocess",
            command=self.preprocess_dataset
        )
        self.preprocess_btn.pack(side="right")

        # Preprocess every dataset in the CSV
        self.preprocess_all_btn = ttk.Button(
            self.preproc_controls,
            text="Preprocess All",
            command=self.preprocess_all
        )
        self.preprocess_all_btn.pack(side="right", padx=5)

        # Main content: 2x2 grid
        content = ttk.Frame(self.tab2)
        content.pack(fill="both", expand=True, padx=10, pady=5)
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=5)
        content.rowconfigure(0, weight=1)
        content.rowconfigure(1, weight=1)

        # -- Row 0, Col 0: single "Preprocessed Groups" treeview --------
        groups_frame = ttk.LabelFrame(content, text="Preprocessed Groups")
        groups_frame.grid(row=0, column=0, sticky="nsew", padx=(0,5), pady=(0,5))

        self.group_tree = ttk.Treeview(groups_frame)
        self._configure_tags(self.group_tree)
        g_scroll = ttk.Scrollbar(groups_frame, orient="vertical", command=self.group_tree.yview)
        self.group_tree.configure(yscrollcommand=g_scroll.set)
        self.group_tree.pack(side="left", fill="both", expand=True)
        g_scroll.pack(side="right", fill="y")
        self._add_tree_context_menu(self.group_tree)
        self.group_tree.bind("<<TreeviewSelect>>", self.view_selected_group)

        # -- Row 1, Col 0: Dataset Log -----------------------------------
        log_frame = ttk.LabelFrame(content, text="Dataset Log")
        log_frame.grid(row=1, column=0, sticky="nsew", padx=(0,5), pady=(5,0))
        self.dataset_log = scrolledtext.ScrolledText(log_frame)
        self.dataset_log.pack(fill="both", expand=True)

        # -- Col 1, Rows 0-1: Group Viewer ------------------------------
        viewer_frame = ttk.LabelFrame(content, text="Group Viewer")
        viewer_frame.grid(row=0, column=1, rowspan=2, sticky="nsew",
                        padx=(5,0), pady=5)
        # Viewer canvas
        self.viewer_canvas = tk.Canvas(viewer_frame)
        self.viewer_canvas.pack(fill="both", expand=True)

        # -- NIfTI 3-view container (hidden until needed) ---------------------
        self.nifti_frame = ttk.Frame(viewer_frame)
        # Boolean var to control mask overlay
        self.show_masks = tk.BooleanVar(value=True)
        for i in (0, 1):
            self.nifti_frame.grid_rowconfigure(i, weight=1)
            self.nifti_frame.grid_columnconfigure(i, weight=1)
        self.nifti_canvases = {}
        axes = ['Axial','Coronal','Sagittal']
        for i, ax in enumerate(axes):
            r, c = divmod(i, 2)
            f = ttk.Frame(self.nifti_frame)
            f.grid(row=r, column=c, sticky='nsew', padx=2, pady=2)
            # Make the canvas row stretch
            f.grid_rowconfigure(1, weight=1)
            f.grid_columnconfigure(0, weight=1)
            # 1) Axis label
            lbl = ttk.Label(f, text=ax, font=('TkDefaultFont', 12, 'bold'))
            lbl.pack(side='top', pady=(0, 5))
            # 2) The image canvas
            cv = tk.Canvas(f, background='black')
            cv.pack(fill='both', expand=True)
            self.nifti_canvases[ax] = cv
            # Slider for each axis, hooked to our new handler
            slider = ttk.Scale(
                f,
                from_=0, to=0,
                orient='horizontal',
                command=lambda v, a=ax: self._on_slider_move(a, int(float(v)))
            )
            slider.pack(fill='x', pady=(5,0))
            setattr(self, f"{ax.lower()}_slider", slider)

            # Label showing current slice index
            val_lbl = ttk.Label(f, text="0")
            val_lbl.pack(pady=(2,0))
            setattr(self, f"{ax.lower()}_val_label", val_lbl)

        # Hide until needed
        self.nifti_frame.pack_forget()

        # Tooltip instance
        self.tooltip = Tooltip(self.viewer_canvas)

        # Keep track of the current mask data for hover tests
        self._last_2d_mask_img = None         # Will be a PIL mask image resized to fit the canvas
        self._last_2d_mask_meta = None        # Dict with {'path': ..., 'class': ...}

        self._last_slice_masks = []           # For 3D: list of 2D numpy slices
        self._last_slice_mask_meta = []       # For 3D: list of dicts {'path': ..., 'class': ...}

        # Bind motion and leave on the 2D canvas
        self.viewer_canvas.bind("<Motion>", self._on_2d_motion)
        self.viewer_canvas.bind("<Leave>", lambda e: self.tooltip.hide())

        # Bind motion and leave on each 3D slice canvas
        for cv in self.nifti_canvases.values():
            cv.bind("<Motion>", self._on_3d_motion)
            cv.bind("<Leave>", lambda e: self.tooltip.hide())

        # -- Navigation: 2x2 Prev/Next Image & Mask (right-aligned) ----
        self.nav_frame = ttk.Frame(viewer_frame)
        self.nav_frame.pack(side="right", padx=5, pady=5)  # Move to right

        # Image Prev/Next
        self.img_prev_btn = ttk.Button(self.nav_frame, text="Prev Image",
                                       state="disabled",
                                       command=self.show_prev_image)
        self.img_prev_btn.grid(row=0, column=0, padx=5, pady=2)
        self.img_next_btn = ttk.Button(self.nav_frame, text="Next Image",
                                       state="disabled",
                                       command=self.show_next_image)
        self.img_next_btn.grid(row=0, column=1, padx=5, pady=2)

        # Mask Prev/Next
        self.msk_prev_btn = ttk.Button(self.nav_frame, text="Prev Mask",
                                       state="disabled",
                                       command=self.show_prev_mask)
        self.msk_prev_btn.grid(row=1, column=0, padx=5, pady=2)
        self.msk_next_btn = ttk.Button(self.nav_frame, text="Next Mask",
                                       state="disabled",
                                       command=self.show_next_mask)
        self.msk_next_btn.grid(row=1, column=1, padx=5, pady=2)

        # Combined index label
        self.index_label = ttk.Label(viewer_frame, text="Img -/- | Msk -/-")
        self.index_label.pack(pady=(2,0))

        # -- Filenames display -----------------------------------------
        self.img_file_label  = ttk.Label(viewer_frame, text="Image: ")
        self.img_file_label.pack(anchor="e")  # Right-align under nav.
        self.mask_file_label = ttk.Label(viewer_frame, text="Mask:  ")
        self.mask_file_label.pack(anchor="e")
        # ----------------------------------------------------------------

        # For tracking current group's image/mask pairs
        self.current_images = []       # List of (img_path, mask_path)
        self.current_index = 0
        self.current_masks = []
        self.current_mask_index = 0

    def _add_tree_context_menu(self, tree):
        """
        Add an Explorer context menu to a tree view.

        Args:
            tree (Any): Treeview to configure or inspect.

        Returns:
            None.
        """
        menu = tk.Menu(tree, tearoff=False)
        menu.add_command(label="Open File Location",
                         command=lambda: self._open_file_location(tree))

        def on_right_click(event):
            """
            Show the file-location context menu for a tree row.

            Args:
                event (Any): Tk event object.

            Returns:
                None.
            """
            iid = tree.identify_row(event.y)
            if not iid:
                return
            path = tree.item(iid, "text")
            if os.path.isfile(path):
                tree.selection_set(iid)
                menu.tk_popup(event.x_root, event.y_root)

        tree.bind("<Button-3>", on_right_click)

    def _open_file_location(self, tree):
        """
        Open Explorer with the selected file highlighted.

        Args:
            tree (Any): Treeview to configure or inspect.

        Returns:
            None.
        """
        sel = tree.selection()
        if not sel:
            return
        path = tree.item(sel[0], "text")
        if not os.path.exists(path):
            messagebox.showerror("Path Not Found", f"The path does not exist:\n{path}")
            return

        path_norm = os.path.normpath(path)
        pidl = ctypes.c_void_p()

        # Turn the full file path into a PIDL
        hr = SHILCreateFromPath(path_norm, ctypes.byref(pidl), 0)
        if hr == 0 and pidl.value:
            # Open Explorer and select the item
            SHOpenFolderAndSelectItems(pidl, 0, None, 0)
            return

        # Fallback: open the folder without selection
        folder = os.path.dirname(path_norm)
        try:
            os.startfile(folder)
        except Exception as e:
            messagebox.showerror("Unable to Open", f"Could not open folder:\n{e}")

    def _configure_tags(self, tree):
        """
        Configure shared Treeview row styles.

        Args:
            tree (Any): Treeview to configure or inspect.

        Returns:
            None.
        """
        tree.tag_configure('folder',  foreground='gray')
        tree.tag_configure('train',   foreground='blue')
        tree.tag_configure('test',    foreground='green')
        tree.tag_configure('images',  foreground='purple')
        tree.tag_configure('masks',   foreground='orange')
        tree.tag_configure('error',   background='red', foreground='white')

    def show_loading(self):
        """
        Display a modal loading overlay.

        Args:
            None.

        Returns:
            None.
        """
        # Create a borderless transient window
        overlay = tk.Toplevel(self.root)
        overlay.transient(self.root)
        overlay.grab_set()
        overlay.overrideredirect(True)

        # Center it over the root window
        w = 200
        h = 100
        x = self.root.winfo_rootx() + (self.root.winfo_width() - w) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - h) // 2
        overlay.geometry(f"{w}x{h}+{x}+{y}")

        # Frame with label and indeterminate bar
        frame = ttk.Frame(overlay, relief="raised", borderwidth=1, padding=10)
        frame.pack(expand=True, fill="both")
        ttk.Label(frame, text="Loading...", font=('TkDefaultFont', 12)).pack(pady=(0,8))
        pb = ttk.Progressbar(frame, mode='indeterminate')
        pb.pack(fill="x")
        pb.start(50)

        self._loading_overlay = overlay

    def hide_loading(self):
        """
        Remove the loading overlay.

        Args:
            None.

        Returns:
            None.
        """
        if hasattr(self, "_loading_overlay"):
            self._loading_overlay.destroy()
            del self._loading_overlay

if __name__=='__main__':
    root=tk.Tk()
    app=PreprocessingDebuggerApp(root)
    root.mainloop()
