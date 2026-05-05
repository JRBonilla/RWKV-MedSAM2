"""CSV viewer/editor callbacks for the DRIPP debugger app."""

import configparser

from dripp.config import LOCAL_CONFIG_FILENAME

from .common import *


class CsvDebuggerMixin:
    """
    Provide callbacks for viewing, creating, and editing DRIPP metadata CSVs.

    Args:
        None.

    Returns:
        None.
    """

    DEFAULT_CSV_COLUMNS = [
        "Dataset Name",
        "Modality",
        "Image File Type",
        "Mask File Type",
        "Root Folder",
        "Train Folders",
        "Test Folders",
        "Mask Folders",
        "Mask Key",
        "Grouping Strategy",
        "Grouping Regex",
        "Grouping Distance",
        "Mask Classes",
        "Segmentation Tasks",
        "Background Value",
        "Preprocessed?",
    ]
    MODALITY_COLORS = {
        "ct": "#d8ecff",
        "mri": "#eadcff",
        "x-ray": "#fff3c7",
        "xray": "#fff3c7",
        "ultrasound": "#d7f5e5",
        "histopathology": "#ffdce8",
        "default": "#eeeeee",
    }
    FILE_TYPE_COLORS = {
        "volume": "#cfe4ff",
        "image": "#d8f2dd",
        "video": "#ffe2c2",
        "mixed": "#f1e1ff",
        "unknown": "#eeeeee",
    }
    GROUPING_STRATEGY_COLORS = {
        "regex-file": "#dceeff",
        "regex-folder": "#e8ddff",
        "regex": "#fff0c2",
        "filename": "#d8f0dc",
        "unknown": "#eeeeee",
    }
    GROUPING_REGEX_COLORS = {
        "none": "#eeeeee",
        "single": "#d8f0dc",
        "multiple": "#dceeff",
    }
    VOLUME_FILE_TYPES = {".nii", ".nii.gz", ".nrrd", ".mha", ".mhd", ".dcm", ".dicom"}
    IMAGE_FILE_TYPES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".pgm"}
    VIDEO_FILE_TYPES = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}

    CSV_ROW_HEIGHT = 26
    CSV_HEADER_HEIGHT = 30
    CSV_MIN_COL_WIDTH = 130
    CSV_WIDE_COL_WIDTH = 240
    CSV_CELL_CHAR_PX = 7
    CSV_LINE_HEIGHT = 17
    CSV_MAX_ROW_HEIGHT = 180

    def _load_metadata_or_prompt_for_csv(self, csv_path):
        """
        Load dataset metadata, prompting for a CSV if the configured file is missing.

        Args:
            csv_path (Any): Configured path to the datasets CSV.

        Returns:
            dict: Loaded dataset metadata, or an empty dict if unresolved.
        """
        attempted_path = csv_path
        should_persist = False
        while True:
            try:
                metadata = load_dataset_metadata(attempted_path)
                self.csv_path = normalize_path(attempted_path)
                if should_persist:
                    self._persist_csv_location(self.csv_path)
                return metadata
            except Exception as e:
                messagebox.showwarning(
                    "CSV Not Found",
                    "DRIPP could not load the configured datasets CSV:\n\n"
                    f"{attempted_path}\n\n"
                    f"{e}\n\n"
                    "Please choose where to look for the datasets CSV."
                )
                selected_path = filedialog.askopenfilename(
                    title="Locate DRIPP datasets CSV",
                    filetypes=[
                        ("CSV files", "*.csv"),
                        ("All files", "*.*"),
                    ],
                )
                if not selected_path:
                    messagebox.showwarning(
                        "CSV Required",
                        "No replacement CSV was selected. The debugger will stay open, but no datasets are loaded."
                    )
                    self.csv_path = normalize_path(attempted_path)
                    return {}
                attempted_path = normalize_path(selected_path)
                should_persist = True

    def _build_csv_tab(self):
        """
        Build the CSV viewer/editor tab.

        Args:
            None.

        Returns:
            None.
        """
        self.csv_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.csv_tab, text="Datasets CSV")

        toolbar = ttk.Frame(self.csv_tab)
        toolbar.pack(fill="x", padx=8, pady=6)

        ttk.Button(toolbar, text="Open...", command=self._open_csv_from_editor).pack(side="left", padx=2)
        ttk.Button(toolbar, text="New CSV", command=self._new_csv_from_editor).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Add Row", command=self._add_csv_row).pack(side="left", padx=(14, 2))
        ttk.Button(toolbar, text="Delete Row", command=self._delete_csv_rows).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Reload", command=lambda: self._load_csv_table(show_errors=True)).pack(side="right", padx=2)
        ttk.Button(toolbar, text="Save As...", command=self._save_csv_as_from_editor).pack(side="right", padx=2)
        ttk.Button(toolbar, text="Save", command=self._save_csv_from_editor).pack(side="right", padx=2)

        self.csv_status_var = tk.StringVar(value="")
        ttk.Label(self.csv_tab, textvariable=self.csv_status_var, anchor="w").pack(fill="x", padx=10)
        self.csv_signal_frame = ttk.Frame(self.csv_tab)
        self.csv_signal_frame.pack(fill="x", padx=8, pady=(4, 0))

        table_frame = ttk.Frame(self.csv_tab)
        table_frame.pack(fill="both", expand=True, padx=8, pady=(6, 8))
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.csv_canvas = tk.Canvas(table_frame, highlightthickness=0, bg="white")
        self.csv_canvas.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.csv_canvas.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.csv_canvas.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        self.csv_canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.csv_canvas.bind("<Button-1>", self._on_csv_canvas_click)
        self.csv_canvas.bind("<Double-1>", self._begin_csv_cell_edit)
        self.csv_canvas.bind("<Configure>", lambda _e: self._draw_csv_grid())
        self.csv_canvas.bind("<Enter>", self._bind_csv_mousewheel)
        self.csv_canvas.bind("<Leave>", self._unbind_csv_mousewheel)

        self.csv_dirty = False
        self.csv_columns = []
        self.csv_rows = []
        self.csv_col_widths = []
        self.csv_col_offsets = []
        self.csv_row_heights = []
        self.csv_row_offsets = []
        self.csv_selected_cell = None
        self.csv_selected_rows = set()
        self.csv_edit_entry = None
        self._load_csv_table(show_errors=False)

    def _bind_csv_mousewheel(self, _event=None):
        """
        Bind mouse-wheel scrolling while the pointer is over the CSV table.

        Args:
            _event (Any, optional): Tk event object.

        Returns:
            None.
        """
        self.csv_canvas.bind_all("<MouseWheel>", self._on_csv_mousewheel)
        self.csv_canvas.bind_all("<Shift-MouseWheel>", self._on_csv_shift_mousewheel)
        self.csv_canvas.bind_all("<Button-4>", self._on_csv_mousewheel)
        self.csv_canvas.bind_all("<Button-5>", self._on_csv_mousewheel)

    def _unbind_csv_mousewheel(self, _event=None):
        """
        Unbind CSV mouse-wheel scrolling after the pointer leaves the table.

        Args:
            _event (Any, optional): Tk event object.

        Returns:
            None.
        """
        self.csv_canvas.unbind_all("<MouseWheel>")
        self.csv_canvas.unbind_all("<Shift-MouseWheel>")
        self.csv_canvas.unbind_all("<Button-4>")
        self.csv_canvas.unbind_all("<Button-5>")

    def _on_csv_mousewheel(self, event):
        """
        Scroll the CSV table vertically with the mouse wheel.

        Args:
            event (Any): Tk event object.

        Returns:
            str: Tk break marker.
        """
        if getattr(event, "num", None) == 4:
            units = -3
        elif getattr(event, "num", None) == 5:
            units = 3
        else:
            units = -1 * int(event.delta / 120) if event.delta else 0
        self.csv_canvas.yview_scroll(units, "units")
        return "break"

    def _on_csv_shift_mousewheel(self, event):
        """
        Scroll the CSV table horizontally with Shift+mouse wheel.

        Args:
            event (Any): Tk event object.

        Returns:
            str: Tk break marker.
        """
        units = -1 * int(event.delta / 120) if event.delta else 0
        self.csv_canvas.xview_scroll(units, "units")
        return "break"

    def _persist_csv_location(self, path):
        """
        Persist the active CSV location to local dripp.ini.

        Args:
            path (str): CSV path to persist.

        Returns:
            None.
        """
        config_path = os.path.join(os.getcwd(), LOCAL_CONFIG_FILENAME)
        parser = configparser.ConfigParser()
        parser.read(config_path)
        if not parser.has_section("paths"):
            parser.add_section("paths")
        parser.set("paths", "base_unproc", normalize_path(os.path.dirname(path)))
        parser.set("paths", "csv_filename", os.path.basename(path))
        with open(config_path, "w") as f:
            parser.write(f)

    def _set_csv_table(self, df):
        """
        Store and render a dataframe in the CSV editor.

        Args:
            df (pandas.DataFrame): CSV data to render.

        Returns:
            None.
        """
        self.csv_columns = list(df.columns) if len(df.columns) else list(self.DEFAULT_CSV_COLUMNS)
        self.csv_rows = [
            [str(row.get(col, "")) for col in self.csv_columns]
            for _, row in df.fillna("").iterrows()
        ]
        self.csv_selected_cell = None
        self.csv_selected_rows = set()
        self._measure_csv_columns()
        self._measure_csv_rows()
        self._draw_csv_grid()

    def _measure_csv_columns(self):
        """
        Measure CSV columns once for the current table.

        Args:
            None.

        Returns:
            None.
        """
        wide_columns = {"Grouping Regex", "Mask Classes", "Segmentation Tasks"}
        self.csv_col_widths = [
            self.CSV_WIDE_COL_WIDTH if col in wide_columns else self.CSV_MIN_COL_WIDTH
            for col in self.csv_columns
        ]
        self.csv_col_offsets = []
        offset = 0
        for width in self.csv_col_widths:
            self.csv_col_offsets.append(offset)
            offset += width

    def _measure_csv_rows(self):
        """
        Measure row heights from wrapped cell content.

        Args:
            None.

        Returns:
            None.
        """
        self.csv_row_heights = []
        self.csv_row_offsets = []
        offset = self.CSV_HEADER_HEIGHT
        for row in self.csv_rows:
            values = row + [""] * (len(self.csv_columns) - len(row))
            max_lines = 1
            for col_index, value in enumerate(values[:len(self.csv_columns)]):
                usable_width = max(20, self.csv_col_widths[col_index] - 10)
                chars_per_line = max(1, usable_width // self.CSV_CELL_CHAR_PX)
                text = str(value)
                lines = 1
                for part in text.splitlines() or [""]:
                    lines += max(0, (len(part) - 1) // chars_per_line)
                max_lines = max(max_lines, lines)
            height = min(
                self.CSV_MAX_ROW_HEIGHT,
                max(self.CSV_ROW_HEIGHT, 8 + max_lines * self.CSV_LINE_HEIGHT)
            )
            self.csv_row_offsets.append(offset)
            self.csv_row_heights.append(height)
            offset += height

    def _draw_csv_grid(self):
        """
        Draw the CSV table on the canvas.

        Args:
            None.

        Returns:
            None.
        """
        if not hasattr(self, "csv_canvas"):
            return

        self.csv_canvas.delete("all")
        total_width = sum(self.csv_col_widths)
        total_height = (
            self.CSV_HEADER_HEIGHT
            if not self.csv_row_offsets
            else self.csv_row_offsets[-1] + self.csv_row_heights[-1]
        )
        self.csv_canvas.configure(scrollregion=(0, 0, total_width, total_height))

        for col_index, column in enumerate(self.csv_columns):
            x0 = self.csv_col_offsets[col_index]
            x1 = x0 + self.csv_col_widths[col_index]
            self.csv_canvas.create_rectangle(x0, 0, x1, self.CSV_HEADER_HEIGHT, fill="#d9d9d9", outline="#999999")
            self.csv_canvas.create_text(x0 + 6, self.CSV_HEADER_HEIGHT // 2, text=column, anchor="w", fill="#111111")

        for row_index, row in enumerate(self.csv_rows):
            y0 = self.csv_row_offsets[row_index]
            y1 = y0 + self.csv_row_heights[row_index]
            values = row + [""] * (len(self.csv_columns) - len(row))
            for col_index, column in enumerate(self.csv_columns):
                x0 = self.csv_col_offsets[col_index]
                x1 = x0 + self.csv_col_widths[col_index]
                color = self._csv_cell_color(column, values)
                outline = "#222222" if self.csv_selected_cell == (row_index, col_index) else "#cccccc"
                width = 2 if self.csv_selected_cell == (row_index, col_index) else 1
                self.csv_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=outline, width=width)
                text = values[col_index] if col_index < len(values) else ""
                self.csv_canvas.create_text(
                    x0 + 5,
                    y0 + 4,
                    text=text,
                    anchor="nw",
                    fill="#111111",
                    width=max(20, self.csv_col_widths[col_index] - 10),
                )
        self._update_csv_signal_chips()

    def _cell_at_canvas_position(self, event):
        """
        Identify the cell at a canvas event position.

        Args:
            event (Any): Tk event object.

        Returns:
            tuple[int, int] or None: Row and column indexes.
        """
        x = self.csv_canvas.canvasx(event.x)
        y = self.csv_canvas.canvasy(event.y)
        if y < self.CSV_HEADER_HEIGHT:
            return None

        row_index = None
        for index, y0 in enumerate(self.csv_row_offsets):
            if y0 <= y < y0 + self.csv_row_heights[index]:
                row_index = index
                break
        if row_index is None:
            return None

        col_index = None
        for index, x0 in enumerate(self.csv_col_offsets):
            if x0 <= x < x0 + self.csv_col_widths[index]:
                col_index = index
                break
        if col_index is None:
            return None
        return row_index, col_index

    def _on_csv_canvas_click(self, event):
        """
        Select a CSV cell from a canvas click.

        Args:
            event (Any): Tk event object.

        Returns:
            None.
        """
        cell = self._cell_at_canvas_position(event)
        if not cell:
            return
        self._commit_csv_edit()
        self.csv_selected_cell = cell
        self.csv_selected_rows = {cell[0]}
        self._draw_csv_grid()

    def _begin_csv_cell_edit(self, event):
        """
        Begin editing the clicked CSV cell.

        Args:
            event (Any): Tk event object.

        Returns:
            None.
        """
        cell = self._cell_at_canvas_position(event)
        if not cell:
            return
        self._commit_csv_edit()
        row_index, col_index = cell
        self.csv_selected_cell = cell
        self.csv_selected_rows = {row_index}

        x0 = self.csv_col_offsets[col_index]
        y0 = self.csv_row_offsets[row_index]
        value = self.csv_rows[row_index][col_index] if col_index < len(self.csv_rows[row_index]) else ""

        entry = ttk.Entry(self.csv_canvas)
        entry.insert(0, value)
        entry.select_range(0, tk.END)
        entry.focus_set()
        window_id = self.csv_canvas.create_window(
            x0,
            y0,
            width=self.csv_col_widths[col_index],
            height=self.csv_row_heights[row_index],
            anchor="nw",
            window=entry,
        )
        self.csv_edit_entry = (entry, window_id, row_index, col_index)
        entry.bind("<Return>", lambda _e: self._commit_csv_edit())
        entry.bind("<FocusOut>", lambda _e: self._commit_csv_edit())
        entry.bind("<Escape>", lambda _e: self._cancel_csv_edit())

    def _commit_csv_edit(self):
        """
        Commit an active CSV cell edit.

        Args:
            None.

        Returns:
            None.
        """
        if not self.csv_edit_entry:
            return
        entry, window_id, row_index, col_index = self.csv_edit_entry
        if row_index < len(self.csv_rows) and col_index < len(self.csv_columns):
            self.csv_rows[row_index] += [""] * (len(self.csv_columns) - len(self.csv_rows[row_index]))
            self.csv_rows[row_index][col_index] = entry.get()
            self.csv_dirty = True
        self.csv_canvas.delete(window_id)
        self.csv_edit_entry = None
        self._measure_csv_rows()
        self._draw_csv_grid()

    def _cancel_csv_edit(self):
        """
        Cancel an active CSV cell edit.

        Args:
            None.

        Returns:
            None.
        """
        if not self.csv_edit_entry:
            return
        _entry, window_id, _row_index, _col_index = self.csv_edit_entry
        self.csv_canvas.delete(window_id)
        self.csv_edit_entry = None
        self._draw_csv_grid()

    def _hex_to_rgb(self, color):
        color = color.lstrip("#")
        return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))

    def _rgb_to_hex(self, rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def _mix_colors(self, colors):
        if not colors:
            return "#ffffff"
        rgbs = [self._hex_to_rgb(color) for color in colors]
        mixed = tuple(round(sum(rgb[i] for rgb in rgbs) / len(rgbs)) for i in range(3))
        return self._rgb_to_hex(mixed)

    def _blend_colors(self, base_color, overlay_color, overlay_weight):
        overlay_weight = max(0.0, min(1.0, overlay_weight))
        base = self._hex_to_rgb(base_color)
        overlay = self._hex_to_rgb(overlay_color)
        blended = tuple(
            round(base[i] * (1.0 - overlay_weight) + overlay[i] * overlay_weight)
            for i in range(3)
        )
        return self._rgb_to_hex(blended)

    def _value_for_column(self, columns, values, column):
        if column not in columns:
            return ""
        index = columns.index(column)
        return values[index] if index < len(values) else ""

    def _split_csv_tokens(self, value):
        return [item.strip().lower() for item in str(value).split(",") if item.strip()]

    def _modality_color(self, modality_value):
        modalities = self._split_csv_tokens(modality_value) or ["default"]
        colors = [self.MODALITY_COLORS.get(mod, self.MODALITY_COLORS["default"]) for mod in modalities]
        return self._mix_colors(colors)

    def _file_type_category(self, value):
        extensions = set(self._split_csv_tokens(value))
        if not extensions:
            return "unknown"
        categories = set()
        for ext in extensions:
            ext = ext if ext.startswith(".") else f".{ext}"
            if ext in self.VOLUME_FILE_TYPES:
                categories.add("volume")
            elif ext in self.IMAGE_FILE_TYPES:
                categories.add("image")
            elif ext in self.VIDEO_FILE_TYPES:
                categories.add("video")
            else:
                categories.add("unknown")
        known = categories - {"unknown"}
        if len(known) > 1 or (known and "unknown" in categories):
            return "mixed"
        return next(iter(known or categories))

    def _grouping_regex_category(self, value):
        regex = str(value).strip()
        if not regex:
            return "none"
        return "multiple" if regex.startswith("[") else "single"

    def _grouping_distance_value(self, columns, values):
        for column in ("Grouping Distance", "Grouping distance", "grouping_distance", "Distance", "distance"):
            raw = self._value_for_column(columns, values, column).strip()
            if not raw:
                continue
            try:
                return max(0.0, float(raw))
            except ValueError:
                return None
        return None

    def _grouping_distance_color(self, distance):
        if distance is None:
            return "#eeeeee"
        ratio = min(distance / 10.0, 1.0)
        return self._blend_colors("#d8f0dc", "#ffd6d6", ratio)

    def _csv_cell_color(self, column, values):
        """
        Return the per-cell color for a CSV column.

        Args:
            column (str): Column name.
            values (list[str]): Row values.

        Returns:
            str: Hex color.
        """
        if column == "Modality":
            return self._modality_color(self._value_for_column(self.csv_columns, values, column))
        if column in ("Image File Type", "Mask File Type"):
            category = self._file_type_category(self._value_for_column(self.csv_columns, values, column))
            return self.FILE_TYPE_COLORS.get(category, self.FILE_TYPE_COLORS["unknown"])
        if column == "Grouping Strategy":
            strategy = self._value_for_column(self.csv_columns, values, column).strip().lower() or "unknown"
            return self.GROUPING_STRATEGY_COLORS.get(strategy, self.GROUPING_STRATEGY_COLORS["unknown"])
        if column == "Grouping Regex":
            category = self._grouping_regex_category(self._value_for_column(self.csv_columns, values, column))
            return self.GROUPING_REGEX_COLORS[category]
        if column in ("Grouping Distance", "Grouping distance", "grouping_distance", "Distance", "distance"):
            return self._grouping_distance_color(self._grouping_distance_value(self.csv_columns, values))
        return "#ffffff"

    def _chip(self, label, value, color):
        text = f"{label}: {value}" if value else label
        tk.Label(
            self.csv_signal_frame,
            text=text,
            bg=color,
            fg="#111111",
            relief="solid",
            borderwidth=1,
            padx=6,
            pady=2,
        ).pack(side="left", padx=2, pady=2)

    def _update_csv_signal_chips(self):
        if not hasattr(self, "csv_signal_frame"):
            return
        for child in self.csv_signal_frame.winfo_children():
            child.destroy()

        if not self.csv_selected_rows:
            self._chip("Select a row to inspect colors", "", "#eeeeee")
            return

        row_index = min(self.csv_selected_rows)
        if row_index >= len(self.csv_rows):
            self._chip("Select a row to inspect colors", "", "#eeeeee")
            return

        values = self.csv_rows[row_index] + [""] * (len(self.csv_columns) - len(self.csv_rows[row_index]))
        modality = self._value_for_column(self.csv_columns, values, "Modality")
        image_types = self._value_for_column(self.csv_columns, values, "Image File Type")
        mask_types = self._value_for_column(self.csv_columns, values, "Mask File Type")
        strategy = self._value_for_column(self.csv_columns, values, "Grouping Strategy").strip().lower() or "unknown"
        regex_category = self._grouping_regex_category(self._value_for_column(self.csv_columns, values, "Grouping Regex"))
        distance = self._grouping_distance_value(self.csv_columns, values)

        image_category = self._file_type_category(image_types)
        mask_category = self._file_type_category(mask_types)
        strategy_color = self.GROUPING_STRATEGY_COLORS.get(strategy, self.GROUPING_STRATEGY_COLORS["unknown"])
        distance_label = "n/a" if distance is None else f"{distance:g}"

        self._chip("Modality", modality or "default", self._modality_color(modality))
        self._chip("Image Types", image_category, self.FILE_TYPE_COLORS.get(image_category, self.FILE_TYPE_COLORS["unknown"]))
        self._chip("Mask Types", mask_category, self.FILE_TYPE_COLORS.get(mask_category, self.FILE_TYPE_COLORS["unknown"]))
        self._chip("Strategy", strategy, strategy_color)
        self._chip("Regex", regex_category, self.GROUPING_REGEX_COLORS[regex_category])
        self._chip("Distance", distance_label, self._grouping_distance_color(distance))

    def _load_csv_table(self, show_errors=True):
        """
        Load the current CSV path into the editor tab.

        Args:
            show_errors (bool): Whether to show load failures in a dialog.

        Returns:
            None.
        """
        self._commit_csv_edit()
        if self.csv_path and os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path, dtype=str, keep_default_na=False)
                self._set_csv_table(df)
                self.csv_status_var.set(f"CSV: {self.csv_path}")
                self.csv_dirty = False
                return
            except Exception as e:
                if show_errors:
                    messagebox.showerror("CSV Load Error", f"Could not load CSV:\n\n{e}")

        self._set_csv_table(pd.DataFrame(columns=self.DEFAULT_CSV_COLUMNS))
        self.csv_status_var.set(f"CSV: {self.csv_path} (not loaded)")
        self.csv_dirty = False

    def _csv_dataframe_from_tree(self):
        """
        Build a dataframe from the CSV editor table.

        Args:
            None.

        Returns:
            pandas.DataFrame: Edited CSV data.
        """
        self._commit_csv_edit()
        rows = []
        for row in self.csv_rows:
            values = row + [""] * (len(self.csv_columns) - len(row))
            values = values[:len(self.csv_columns)]
            if any(v.strip() for v in values):
                rows.append(values)
        return pd.DataFrame(rows, columns=self.csv_columns)

    def _refresh_metadata_from_csv_editor(self):
        """
        Refresh debugger dataset controls from the current CSV.

        Args:
            None.

        Returns:
            bool: True if metadata refreshed successfully.
        """
        try:
            self._reload_metadata()
        except Exception as e:
            messagebox.showwarning(
                "CSV Saved",
                "The CSV was saved, but DRIPP could not load it as dataset metadata:\n\n"
                f"{e}"
            )
            return False
        self.dataset_var.set("")
        self.preproc_dataset_var.set("")
        self.base_var.set("")
        self.save_json_btn.state(["disabled"])
        return True

    def _save_csv_to_path(self, path):
        """
        Save the editor table to a CSV path.

        Args:
            path (str): Destination CSV path.

        Returns:
            bool: True when saved.
        """
        if not path:
            return False
        try:
            df = self._csv_dataframe_from_tree()
            df.to_csv(path, index=False)
            self.csv_path = normalize_path(path)
            self._persist_csv_location(self.csv_path)
            self.csv_status_var.set(f"CSV: {self.csv_path}")
            self.csv_dirty = False
            self._refresh_metadata_from_csv_editor()
            messagebox.showinfo("Saved CSV", f"Saved CSV to:\n{self.csv_path}")
            return True
        except Exception as e:
            messagebox.showerror("CSV Save Error", f"Could not save CSV:\n\n{e}")
            return False

    def _open_csv_from_editor(self):
        """
        Open a CSV file into the editor.

        Args:
            None.

        Returns:
            None.
        """
        path = filedialog.askopenfilename(
            title="Open DRIPP datasets CSV",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.csv_path = normalize_path(path)
        self._persist_csv_location(self.csv_path)
        self._load_csv_table(show_errors=True)
        self._refresh_metadata_from_csv_editor()

    def _new_csv_from_editor(self):
        """
        Start a new DRIPP metadata CSV in the editor.

        Args:
            None.

        Returns:
            None.
        """
        self._set_csv_table(pd.DataFrame(columns=self.DEFAULT_CSV_COLUMNS))
        self.csv_path = ""
        self.csv_status_var.set("New DRIPP datasets CSV (unsaved)")
        self.csv_dirty = True

    def _save_csv_from_editor(self):
        """
        Save the editor contents to the current CSV path.

        Args:
            None.

        Returns:
            None.
        """
        if not self.csv_path or not os.path.isdir(os.path.dirname(os.path.abspath(self.csv_path))):
            self._save_csv_as_from_editor()
            return
        self._save_csv_to_path(self.csv_path)

    def _save_csv_as_from_editor(self):
        """
        Save the editor contents to a user-selected CSV path.

        Args:
            None.

        Returns:
            None.
        """
        path = filedialog.asksaveasfilename(
            title="Save DRIPP datasets CSV",
            defaultextension=".csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._save_csv_to_path(path)

    def _add_csv_row(self):
        """
        Add an empty row to the CSV editor.

        Args:
            None.

        Returns:
            None.
        """
        self._commit_csv_edit()
        self.csv_rows.append([""] * len(self.csv_columns))
        self.csv_dirty = True
        self._measure_csv_rows()
        self._draw_csv_grid()

    def _delete_csv_rows(self):
        """
        Delete selected rows from the CSV editor.

        Args:
            None.

        Returns:
            None.
        """
        self._commit_csv_edit()
        if not self.csv_selected_rows:
            return
        selected = set(self.csv_selected_rows)
        self.csv_rows = [row for index, row in enumerate(self.csv_rows) if index not in selected]
        self.csv_selected_cell = None
        self.csv_selected_rows = set()
        self.csv_dirty = True
        self._measure_csv_rows()
        self._draw_csv_grid()
