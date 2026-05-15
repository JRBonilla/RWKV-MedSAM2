"""Settings tab for the DRIPP debugger app."""

import configparser
import logging
import os

import dripp.config as config
import dripp.dataset as dataset_module
import dripp.debugger.common as debugger_common
import dripp.debugger.grouping_debugger as grouping_debugger
import dripp.debugger.preprocessing_debugger as preprocessing_debugger
import dripp.helpers as helpers_module
import dripp.indexer as indexer_module
import dripp.preprocessor as preprocessor_module
from dripp.config import LOCAL_CONFIG_FILENAME
from dripp.output_structure import (
    DEFAULT_GROUP_FOLDER_TEMPLATE,
    OutputStructureError,
    render_preview,
    validate_group_folder_template,
    validate_leaf_folder,
)
from dripp.output_filenames import (
    DEFAULT_IMAGE_FILENAME_SEGMENTS,
    DEFAULT_MASK_FILENAME_SEGMENTS,
    FILENAME_SEGMENT_LABELS,
    IMAGE_FILENAME_SEGMENTS,
    MASK_FILENAME_SEGMENTS,
    OutputFilenameError,
    render_filename_preview,
    validate_filename_separator,
    validate_output_filenames,
)

from .common import *


class SettingsDebuggerMixin:
    """
    Provide a debugger tab for editing local DRIPP settings.

    Args:
        None.

    Returns:
        None.
    """

    SETTINGS_PATH_FIELDS = [
        ("preprocessing_log_dir", "Preprocessing Log Dir"),
        ("base_unproc", "Raw Datasets Dir"),
        ("base_proc", "Preprocessed Output Dir"),
        ("groups_dir", "Groups Dir"),
        ("index_dir", "Index Dir"),
    ]
    SETTINGS_OUTPUT_FIELDS = [
        ("2d_image_format", "2D Image"),
        ("2d_mask_format", "2D Mask"),
        ("3d_image_format", "3D Image"),
        ("3d_mask_format", "3D Mask"),
        ("video_frame_format", "Video Frame"),
        ("video_mask_format", "Video Mask"),
    ]
    SETTINGS_OUTPUT_RUNTIME_KEYS = {
        "2d_image_format": "2d_image",
        "2d_mask_format": "2d_mask",
        "3d_image_format": "3d_image",
        "3d_mask_format": "3d_mask",
        "video_frame_format": "video_frame",
        "video_mask_format": "video_mask",
    }
    SETTINGS_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    SETTINGS_OUTPUT_STRUCTURE_FIELDS = [
        ("group_folder_template", "Group Folder Template"),
        ("images_folder", "Images Folder"),
        ("masks_folder", "Masks Folder"),
    ]
    SETTINGS_OUTPUT_STRUCTURE_REQUIRED_PREFIX = "{dataset}"
    SETTINGS_OUTPUT_STRUCTURE_TOKENS = [
        ("Modality", "{modality}", "#dcfce7"),
        ("Subdataset", "{subdataset}", "#fef3c7"),
        ("Split", "{split}", "#fee2e2"),
        ("ID", "{id}", "#ede9fe"),
        ("ID Parts", "{id_parts}", "#cffafe"),
    ]
    SETTINGS_OUTPUT_STRUCTURE_TOOLTIPS = {
        "{dataset}": "Dataset name. This is always the first folder.",
        "{modality}": "Image modality folder, such as ct, mri, xray, or ultrasound.",
        "{subdataset}": "Subdataset folder. Uses default when the dataset has no subdataset.",
        "{split}": "Dataset split folder, such as train, val, or test.",
        "{id}": "One folder from the matched group ID, e.g. Patient42_Study7_Slice003.",
        "{id_parts}": "Nested folders from the group ID parts, e.g. Patient42/Study7/Slice003.",
    }
    SETTINGS_OUTPUT_STRUCTURE_EXCLUSIVE_TOKENS = {"{id}", "{id_parts}"}
    SETTINGS_FILENAME_SEGMENT_COLORS = {
        "short_id": "#dbeafe",
        "image_number": "#dcfce7",
        "source_tag": "#fef3c7",
        "mask_number": "#fee2e2",
        "class_name": "#ede9fe",
        "label_value": "#cffafe",
        "component_number": "#fce7f3",
    }
    SETTINGS_FILENAME_SEGMENT_TOOLTIPS = {
        "short_id": "Stable short version of the group ID used to keep filenames compact.",
        "image_number": "Image or frame index, such as img003 or frame0003.",
        "source_tag": "Optional image or mask source tag, written like ~lesionA~ when present.",
        "mask_number": "Mask index for the current image, such as mask002.",
        "class_name": "Mask class name, written like %tumor%.",
        "label_value": "3D label value, such as label007. Omitted when not needed.",
        "component_number": "Connected component index, such as comp005.",
    }

    def _build_settings_tab(self):
        """
        Register the settings editor tab.

        Args:
            None.

        Returns:
            None.
        """
        self._configure_settings_tab_style()
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text="Settings")
        self._settings_tab_built = False
        self._settings_tab_build_after_id = None
        self._settings_tab_placeholder = ttk.Label(
            self.settings_tab,
            text="Settings will load when this tab is opened.",
            anchor="center",
        )
        self._settings_tab_placeholder.pack(fill="both", expand=True, padx=12, pady=12)
        self.notebook.bind("<<NotebookTabChanged>>", self._on_settings_tab_selected, add="+")

    def _on_settings_tab_selected(self, _event=None):
        """
        Lazily build Settings when the tab is first selected.

        Args:
            _event (Any): Tk event object.

        Returns:
            None.
        """
        if getattr(self, "_settings_tab_built", False):
            return
        selected = self.notebook.select()
        if selected and selected == str(self.settings_tab) and self._settings_tab_build_after_id is None:
            self._settings_tab_build_after_id = self.root.after_idle(self._build_settings_tab_if_selected)

    def _build_settings_tab_if_selected(self):
        """
        Build Settings after the notebook has painted the selected tab.

        Args:
            None.

        Returns:
            None.
        """
        self._settings_tab_build_after_id = None
        selected = self.notebook.select()
        if selected and selected == str(self.settings_tab):
            self._ensure_settings_tab_built()

    def _ensure_settings_tab_built(self):
        """
        Build the Settings tab contents once.

        Args:
            None.

        Returns:
            None.
        """
        if getattr(self, "_settings_tab_built", False):
            return
        if hasattr(self, "_settings_tab_placeholder"):
            self._settings_tab_placeholder.destroy()
        self._settings_tab_built = True

        toolbar = ttk.Frame(self.settings_tab)
        toolbar.pack(fill="x", padx=10, pady=8)
        ttk.Button(toolbar, text="Save", command=self._save_settings).pack(side="right", padx=2)
        ttk.Button(toolbar, text="Reload", command=self._load_settings_values).pack(side="right", padx=2)
        ttk.Button(toolbar, text="Reset Unsaved", command=self._reset_unsaved_settings).pack(side="right", padx=2)

        self.settings_status_var = tk.StringVar()
        ttk.Label(
            self.settings_tab,
            textvariable=self.settings_status_var,
            anchor="w",
        ).pack(fill="x", padx=12)

        scroll_frame = ttk.Frame(self.settings_tab)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=8)
        scroll_frame.columnconfigure(0, weight=1)
        scroll_frame.rowconfigure(0, weight=1)

        self.settings_canvas = tk.Canvas(scroll_frame, highlightthickness=0, borderwidth=0)
        settings_scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical", command=self.settings_canvas.yview)
        self.settings_canvas.configure(yscrollcommand=settings_scrollbar.set)
        self.settings_canvas.grid(row=0, column=0, sticky="nsew")
        settings_scrollbar.grid(row=0, column=1, sticky="ns")

        body = ttk.Frame(self.settings_canvas)
        self.settings_canvas_window = self.settings_canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", self._on_settings_body_configure)
        self.settings_canvas.bind("<Configure>", self._on_settings_canvas_configure)
        self.settings_canvas.bind("<Enter>", lambda _event: self._bind_settings_mousewheel())
        self.settings_canvas.bind("<Leave>", lambda _event: self._unbind_settings_mousewheel())
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        self.settings_vars = {}
        self._settings_preview_after_ids = {}
        self._settings_preview_suspended = False
        self._settings_scrollregion_after_id = None
        self._settings_scrollregion = None
        self._settings_canvas_width = None
        self._settings_canvas_resize_after_id = None
        self._settings_pending_canvas_width = None
        self._settings_tooltip = {"window": None, "after_id": None}

        paths_frame = ttk.LabelFrame(body, text="Paths", style="SettingsSubsection.TLabelframe")
        paths_frame.grid(row=0, column=0, columnspan=2, sticky="new", padx=2, pady=(0, 8))
        paths_frame.columnconfigure(1, weight=1)

        row = 0
        for key, label in self.SETTINGS_PATH_FIELDS:
            ttk.Label(paths_frame, text=f"{label}:").grid(row=row, column=0, sticky="w", padx=6, pady=3)
            var = tk.StringVar()
            self.settings_vars[key] = var
            ttk.Entry(paths_frame, textvariable=var).grid(row=row, column=1, sticky="we", padx=4, pady=3)
            ttk.Button(
                paths_frame,
                text="Browse...",
                command=lambda k=key: self._browse_settings_directory(k)
            ).grid(row=row, column=2, sticky="e", padx=6, pady=3)
            row += 1

        ttk.Label(paths_frame, text="CSV Filename:").grid(row=row, column=0, sticky="w", padx=6, pady=3)
        csv_var = tk.StringVar()
        self.settings_vars["csv_filename"] = csv_var
        ttk.Entry(paths_frame, textvariable=csv_var).grid(row=row, column=1, sticky="we", padx=4, pady=3)
        ttk.Button(paths_frame, text="Locate CSV...", command=self._browse_settings_csv).grid(
            row=row, column=2, sticky="e", padx=6, pady=3
        )

        preproc_frame = ttk.LabelFrame(body, text="Preprocessing", style="SettingsSubsection.TLabelframe")
        preproc_frame.grid(row=1, column=0, sticky="new", padx=(2, 6), pady=2)
        preproc_frame.columnconfigure(1, weight=1)
        self.settings_vars["target_size"] = tk.StringVar()
        self.settings_vars["min_component_size"] = tk.StringVar()
        ttk.Label(preproc_frame, text="Target Size:").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(preproc_frame, textvariable=self.settings_vars["target_size"]).grid(
            row=0, column=1, sticky="we", padx=4, pady=3
        )
        ttk.Label(preproc_frame, text="Min Component Size:").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(preproc_frame, textvariable=self.settings_vars["min_component_size"]).grid(
            row=1, column=1, sticky="we", padx=4, pady=3
        )

        runtime_frame = ttk.LabelFrame(body, text="Runtime & Logging", style="SettingsSubsection.TLabelframe")
        runtime_frame.grid(row=1, column=1, sticky="new", padx=(6, 2), pady=2)
        runtime_frame.columnconfigure(1, weight=1)
        self.settings_vars["gpu_enabled"] = tk.BooleanVar()
        self.settings_vars["default_log_level"] = tk.StringVar()
        ttk.Checkbutton(
            runtime_frame,
            text="GPU Enabled",
            variable=self.settings_vars["gpu_enabled"],
            state="disabled",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=3)
        ttk.Label(runtime_frame, text="Log Level:").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Combobox(
            runtime_frame,
            textvariable=self.settings_vars["default_log_level"],
            values=self.SETTINGS_LOG_LEVELS,
            state="readonly",
        ).grid(row=1, column=1, sticky="we", padx=4, pady=3)

        output_frame = ttk.LabelFrame(body, text="Output Formats", style="SettingsSubsection.TLabelframe")
        output_frame.grid(row=2, column=0, columnspan=2, sticky="new", padx=2, pady=(8, 2))
        output_frame.columnconfigure(1, weight=1)
        output_frame.columnconfigure(3, weight=1)
        two_d_output_fields = {"2d_image_format", "2d_mask_format"}
        image_output_fields = {"video_frame_format", "video_mask_format"}
        for idx, (key, label) in enumerate(self.SETTINGS_OUTPUT_FIELDS):
            row, col = divmod(idx, 2)
            base_col = col * 2
            var = tk.StringVar()
            self.settings_vars[key] = var
            ttk.Label(output_frame, text=f"{label}:").grid(row=row, column=base_col, sticky="w", padx=6, pady=3)
            values = (
                sorted(config.SUPPORTED_2D_OUTPUT_EXTS)
                if key in two_d_output_fields
                else sorted(config.SUPPORTED_IMAGE_OUTPUT_EXTS)
                if key in image_output_fields
                else sorted(config.SUPPORTED_VOLUME_OUTPUT_EXTS)
            )
            ttk.Combobox(
                output_frame,
                textvariable=var,
                values=values,
                state="readonly",
                width=16,
            ).grid(
                row=row, column=base_col + 1, sticky="we", padx=4, pady=3
            )

        structure_frame = ttk.LabelFrame(body, text="Output Structure", style="SettingsSubsection.TLabelframe")
        structure_frame.grid(row=3, column=0, columnspan=2, sticky="new", padx=2, pady=(8, 2))
        structure_frame.columnconfigure(1, weight=1)
        self.settings_vars["group_folder_template"] = tk.StringVar()
        self.settings_vars["images_folder"] = tk.StringVar()
        self.settings_vars["masks_folder"] = tk.StringVar()
        self.output_structure_selected_tokens = []
        self.output_structure_active_token = None
        self.output_structure_token_buttons = {}
        self.output_structure_selected_chips = {}

        ttk.Label(structure_frame, text="Folder Tags:").grid(row=0, column=0, sticky="nw", padx=6, pady=3)
        self.output_structure_selected_frame = ttk.Frame(structure_frame)
        self.output_structure_selected_frame.grid(row=0, column=1, sticky="we", padx=4, pady=3)
        label, color = "Dataset", "#dbeafe"
        self.output_structure_dataset_chip = tk.Button(
            self.output_structure_selected_frame,
            text=label,
            background=color,
            activebackground=color,
            relief="sunken",
            borderwidth=2,
            state="disabled",
        )
        self.output_structure_dataset_chip.grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self._attach_settings_tooltip(
            self.output_structure_dataset_chip,
            self.SETTINGS_OUTPUT_STRUCTURE_TOOLTIPS["{dataset}"],
        )

        ttk.Label(structure_frame, text="Add Tag:").grid(row=1, column=0, sticky="nw", padx=6, pady=3)
        self.output_structure_available_frame = ttk.Frame(structure_frame)
        self.output_structure_available_frame.grid(row=1, column=1, sticky="we", padx=4, pady=3)
        for idx, (label, token, color) in enumerate(self.SETTINGS_OUTPUT_STRUCTURE_TOKENS):
            btn = tk.Button(
                self.output_structure_available_frame,
                text=label,
                background=color,
                activebackground=color,
                relief="raised",
                borderwidth=1,
                command=lambda t=token: self._add_output_structure_token(t),
            )
            btn.grid(row=idx // 3, column=idx % 3, sticky="we", padx=2, pady=2)
            self._attach_settings_tooltip(btn, self.SETTINGS_OUTPUT_STRUCTURE_TOOLTIPS[token])
            self.output_structure_token_buttons[token] = btn
        for col in range(3):
            self.output_structure_available_frame.columnconfigure(col, weight=1)

        id_help = ttk.Frame(structure_frame)
        id_help.grid(row=2, column=1, sticky="we", padx=4, pady=(0, 4))
        ttk.Label(id_help, text="ID: one folder from the matched group id, e.g. Patient42_Study7_Slice003").pack(anchor="w")
        ttk.Label(id_help, text="ID Parts: nested folders from the composite id, e.g. Patient42/Study7/Slice003").pack(anchor="w")

        control_frame = ttk.Frame(structure_frame)
        control_frame.grid(row=3, column=1, sticky="w", padx=4, pady=(0, 4))
        self.output_structure_up_btn = ttk.Button(
            control_frame,
            text="Move Up",
            command=lambda: self._move_output_structure_token(-1),
        )
        self.output_structure_up_btn.pack(side="left", padx=(0, 4))
        self.output_structure_down_btn = ttk.Button(
            control_frame,
            text="Move Down",
            command=lambda: self._move_output_structure_token(1),
        )
        self.output_structure_down_btn.pack(side="left", padx=4)
        self.output_structure_remove_btn = ttk.Button(
            control_frame,
            text="Remove",
            command=self._remove_output_structure_token,
        )
        self.output_structure_remove_btn.pack(side="left", padx=4)
        ttk.Button(
            control_frame,
            text="Default",
            command=self._reset_output_structure_template,
        ).pack(side="left", padx=4)

        self.output_structure_builder_status_var = tk.StringVar()
        ttk.Label(
            structure_frame,
            textvariable=self.output_structure_builder_status_var,
            anchor="w",
        ).grid(row=4, column=1, sticky="we", padx=4, pady=(0, 4))

        ttk.Label(structure_frame, text="Group Folder Template:").grid(
            row=5, column=0, sticky="w", padx=6, pady=3
        )
        ttk.Entry(
            structure_frame,
            textvariable=self.settings_vars["group_folder_template"],
            state="readonly",
        ).grid(row=5, column=1, sticky="we", padx=4, pady=3)
        self.settings_vars["group_folder_template"].trace_add(
            "write",
            lambda *_args: self._schedule_settings_preview(
                "output_structure",
                self._update_output_structure_preview,
            ),
        )

        for idx, (key, label) in enumerate(
            (
                ("images_folder", "Images Folder"),
                ("masks_folder", "Masks Folder"),
            ),
            start=6,
        ):
            ttk.Label(structure_frame, text=f"{label}:").grid(row=idx, column=0, sticky="w", padx=6, pady=3)
            ttk.Entry(
                structure_frame,
                textvariable=self.settings_vars[key],
                width=20,
            ).grid(row=idx, column=1, sticky="w", padx=4, pady=3)
            self.settings_vars[key].trace_add(
                "write",
                lambda *_args: self._schedule_settings_preview(
                    "output_structure",
                    self._update_output_structure_preview,
                ),
            )

        self.output_structure_preview_var = tk.StringVar()
        ttk.Label(
            structure_frame,
            textvariable=self.output_structure_preview_var,
            anchor="w",
            justify="left",
        ).grid(
            row=8,
            column=0,
            columnspan=2,
            sticky="we",
            padx=6,
            pady=(6, 3),
        )

        filename_frame = ttk.LabelFrame(body, text="Output Filenames", style="SettingsSubsection.TLabelframe")
        filename_frame.grid(row=4, column=0, columnspan=2, sticky="new", padx=2, pady=(8, 2))
        filename_frame.columnconfigure(1, weight=1)
        self.settings_vars["filename_separator"] = tk.StringVar()
        self.filename_builders = {}
        self._build_filename_tag_builder(
            filename_frame,
            "image",
            "Image Filename Tags:",
            IMAGE_FILENAME_SEGMENTS,
            row=0,
        )
        self._build_filename_tag_builder(
            filename_frame,
            "mask",
            "Mask Filename Tags:",
            MASK_FILENAME_SEGMENTS,
            row=3,
        )
        ttk.Label(filename_frame, text="Separator:").grid(row=6, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(
            filename_frame,
            textvariable=self.settings_vars["filename_separator"],
            width=8,
        ).grid(row=6, column=1, sticky="w", padx=4, pady=3)
        self.settings_vars["filename_separator"].trace_add(
            "write",
            lambda *_args: self._schedule_settings_preview(
                "filename",
                self._update_filename_preview,
            ),
        )
        self.filename_preview_var = tk.StringVar()
        ttk.Label(
            filename_frame,
            textvariable=self.filename_preview_var,
            anchor="w",
            justify="left",
        ).grid(row=7, column=0, columnspan=2, sticky="we", padx=6, pady=(6, 3))

        self._load_settings_values()

    def _configure_settings_tab_style(self):
        """
        Configure local Settings tab styles.

        Args:
            None.

        Returns:
            None.
        """
        style = ttk.Style(self.root)
        style.configure("SettingsSubsection.TLabelframe", borderwidth=2, relief="solid")
        style.configure("SettingsSubsection.TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))

    def _on_settings_body_configure(self, _event=None):
        """
        Schedule a Settings scrollregion update when the body size changes.

        Args:
            _event (Any): Tk event object.

        Returns:
            None.
        """
        if hasattr(self, "settings_canvas"):
            if self._settings_scrollregion_after_id is not None:
                try:
                    self.root.after_cancel(self._settings_scrollregion_after_id)
                except tk.TclError:
                    pass
            self._settings_scrollregion_after_id = self.root.after(35, self._update_settings_scrollregion)

    def _update_settings_scrollregion(self):
        """
        Apply the pending Settings scrollregion update.

        Args:
            None.

        Returns:
            None.
        """
        self._settings_scrollregion_after_id = None
        if hasattr(self, "settings_canvas"):
            scrollregion = self.settings_canvas.bbox("all")
            if scrollregion != self._settings_scrollregion:
                self._settings_scrollregion = scrollregion
                self.settings_canvas.configure(scrollregion=scrollregion)

    def _on_settings_canvas_configure(self, event):
        """
        Keep the scrollable Settings body the same width as the canvas.

        Args:
            event (Any): Tk event object.

        Returns:
            None.
        """
        if not hasattr(self, "settings_canvas_window"):
            return
        if event.width == self._settings_canvas_width:
            return

        self._settings_pending_canvas_width = event.width
        if self._settings_canvas_width is None:
            self._apply_settings_canvas_width()
            return

        if self._settings_canvas_resize_after_id is not None:
            try:
                self.root.after_cancel(self._settings_canvas_resize_after_id)
            except tk.TclError:
                pass
        self._settings_canvas_resize_after_id = self.root.after(60, self._apply_settings_canvas_width)

    def _apply_settings_canvas_width(self):
        """
        Apply the latest pending Settings canvas width.

        Args:
            None.

        Returns:
            None.
        """
        self._settings_canvas_resize_after_id = None
        width = self._settings_pending_canvas_width
        if width is None or width == self._settings_canvas_width:
            return
        if not hasattr(self, "settings_canvas_window"):
            return
        self._settings_canvas_width = width
        self.settings_canvas.itemconfigure(self.settings_canvas_window, width=width)

    def _bind_settings_mousewheel(self):
        """
        Bind mousewheel scrolling while the pointer is over Settings.

        Args:
            None.

        Returns:
            None.
        """
        self.settings_canvas.bind_all("<MouseWheel>", self._on_settings_mousewheel)

    def _unbind_settings_mousewheel(self):
        """
        Remove Settings mousewheel binding when the pointer leaves.

        Args:
            None.

        Returns:
            None.
        """
        self.settings_canvas.unbind_all("<MouseWheel>")

    def _on_settings_mousewheel(self, event):
        """
        Scroll the Settings canvas with the mousewheel.

        Args:
            event (Any): Tk event object.

        Returns:
            None.
        """
        self.settings_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _schedule_settings_preview(self, key, callback):
        """
        Debounce Settings preview updates triggered by text variables.

        Args:
            key (str): Preview group key.
            callback (Callable): Preview update callback.

        Returns:
            None.
        """
        if getattr(self, "_settings_preview_suspended", False):
            return
        if not hasattr(self, "_settings_preview_after_ids"):
            callback()
            return
        previous = self._settings_preview_after_ids.pop(key, None)
        if previous is not None:
            try:
                self.root.after_cancel(previous)
            except tk.TclError:
                pass

        def run():
            self._settings_preview_after_ids.pop(key, None)
            callback()

        self._settings_preview_after_ids[key] = self.root.after(80, run)

    def _attach_settings_tooltip(self, widget, text):
        """
        Attach a small hover tooltip to a Settings widget.

        Args:
            widget (tk.Widget): Widget that owns the tooltip.
            text (str): Tooltip text.

        Returns:
            None.
        """
        tooltip = self._settings_tooltip

        def show(target=widget, message=text):
            if not target.winfo_exists():
                return
            hide()
            x = target.winfo_rootx() + 12
            y = target.winfo_rooty() + target.winfo_height() + 6
            try:
                window = tk.Toplevel(target)
            except tk.TclError:
                return
            window.wm_overrideredirect(True)
            window.wm_geometry(f"+{x}+{y}")
            label = tk.Label(
                window,
                text=message,
                justify="left",
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
                padx=6,
                pady=3,
                wraplength=320,
            )
            label.pack()
            tooltip["window"] = window

        def schedule(_event=None):
            cancel()
            tooltip["after_id"] = widget.after(450, show)

        def hide(_event=None):
            cancel()
            if tooltip["window"] is not None:
                try:
                    tooltip["window"].destroy()
                except tk.TclError:
                    pass
                tooltip["window"] = None

        def cancel():
            if tooltip["after_id"] is not None:
                try:
                    widget.after_cancel(tooltip["after_id"])
                except tk.TclError:
                    pass
                tooltip["after_id"] = None

        widget.bind("<Enter>", schedule, add="+")
        widget.bind("<Leave>", hide, add="+")
        widget.bind("<ButtonPress>", hide, add="+")
        widget.bind("<Destroy>", hide, add="+")

    def _settings_config_path(self):
        """
        Return the local DRIPP config path edited by this tab.

        Args:
            None.

        Returns:
            str: Absolute config path.
        """
        return os.path.join(os.getcwd(), LOCAL_CONFIG_FILENAME)

    def _read_settings_parser(self):
        """
        Read the local settings file into a parser.

        Args:
            None.

        Returns:
            configparser.ConfigParser: Parsed local config.
        """
        parser = configparser.ConfigParser()
        parser.read(self._settings_config_path())
        return parser

    def _settings_get(self, parser, section, option, fallback):
        """
        Return a local config value with an effective runtime fallback.

        Args:
            parser (configparser.ConfigParser): Local config parser.
            section (str): Config section.
            option (str): Config option.
            fallback (Any): Fallback value.

        Returns:
            str: Config value.
        """
        if parser.has_option(section, option):
            return parser.get(section, option)
        if isinstance(fallback, (tuple, list)):
            return ", ".join(str(item) for item in fallback)
        return str(fallback)

    def _build_filename_tag_builder(self, parent, builder_key, label, allowed_segments, row):
        """
        Build a color-tag filename segment editor.
        """
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="nw", padx=6, pady=3)
        selected_frame = ttk.Frame(parent)
        selected_frame.grid(row=row, column=1, sticky="we", padx=4, pady=3)

        ttk.Label(parent, text="Add Tag:").grid(row=row + 1, column=0, sticky="nw", padx=6, pady=3)
        available_frame = ttk.Frame(parent)
        available_frame.grid(row=row + 1, column=1, sticky="we", padx=4, pady=3)

        buttons = {}
        for idx, segment in enumerate(sorted(allowed_segments, key=lambda item: list(FILENAME_SEGMENT_LABELS).index(item))):
            color = self.SETTINGS_FILENAME_SEGMENT_COLORS[segment]
            btn = tk.Button(
                available_frame,
                text=FILENAME_SEGMENT_LABELS[segment],
                background=color,
                activebackground=color,
                relief="raised",
                borderwidth=1,
                command=lambda b=builder_key, s=segment: self._add_filename_segment(b, s),
            )
            btn.grid(row=idx // 4, column=idx % 4, sticky="we", padx=2, pady=2)
            self._attach_settings_tooltip(btn, self.SETTINGS_FILENAME_SEGMENT_TOOLTIPS[segment])
            buttons[segment] = btn
        for col in range(4):
            available_frame.columnconfigure(col, weight=1)

        controls = ttk.Frame(parent)
        controls.grid(row=row + 2, column=1, sticky="w", padx=4, pady=(0, 4))
        up_btn = ttk.Button(controls, text="Move Up", command=lambda b=builder_key: self._move_filename_segment(b, -1))
        down_btn = ttk.Button(controls, text="Move Down", command=lambda b=builder_key: self._move_filename_segment(b, 1))
        remove_btn = ttk.Button(controls, text="Remove", command=lambda b=builder_key: self._remove_filename_segment(b))
        default_btn = ttk.Button(controls, text="Default", command=lambda b=builder_key: self._reset_filename_segments(b))
        up_btn.pack(side="left", padx=(0, 4))
        down_btn.pack(side="left", padx=4)
        remove_btn.pack(side="left", padx=4)
        default_btn.pack(side="left", padx=4)

        self.filename_builders[builder_key] = {
            "allowed": set(allowed_segments),
            "segments": [],
            "active": None,
            "selected_frame": selected_frame,
            "buttons": buttons,
            "chips": {},
            "up_btn": up_btn,
            "down_btn": down_btn,
            "remove_btn": remove_btn,
        }

    def _filename_segment_meta(self, segment):
        return FILENAME_SEGMENT_LABELS[segment], self.SETTINGS_FILENAME_SEGMENT_COLORS[segment]

    def _render_filename_builder(self, builder_key):
        if not hasattr(self, "filename_builders") or builder_key not in self.filename_builders:
            return
        builder = self.filename_builders[builder_key]
        frame = builder["selected_frame"]
        for child in frame.winfo_children():
            child.destroy()
        builder["chips"] = {}

        for idx, segment in enumerate(builder["segments"]):
            label, color = self._filename_segment_meta(segment)
            relief = "sunken" if segment == builder["active"] else "raised"
            chip = tk.Button(
                frame,
                text=label,
                background=color,
                activebackground=color,
                relief=relief,
                borderwidth=2,
                command=lambda b=builder_key, s=segment: self._select_filename_segment(b, s),
            )
            chip.grid(row=0, column=idx, sticky="w", padx=2, pady=2)
            self._attach_settings_tooltip(chip, self.SETTINGS_FILENAME_SEGMENT_TOOLTIPS[segment])
            builder["chips"][segment] = chip

        if not builder["segments"]:
            ttk.Label(frame, text="No filename tags selected.").grid(row=0, column=0, sticky="w", padx=2, pady=2)

        used = set(builder["segments"])
        for segment, button in builder["buttons"].items():
            button.configure(state=("disabled" if segment in used else "normal"))

        self._update_filename_builder_controls(builder_key)

    def _update_filename_builder_controls(self, builder_key):
        builder = self.filename_builders[builder_key]
        used = set(builder["segments"])
        for segment, chip in builder.get("chips", {}).items():
            relief = "sunken" if segment == builder["active"] else "raised"
            chip.configure(relief=relief)

        active = builder["active"]
        has_active = active in used
        active_index = builder["segments"].index(active) if has_active else -1
        builder["up_btn"].state(["!disabled"] if active_index > 0 else ["disabled"])
        builder["down_btn"].state(
            ["!disabled"] if has_active and active_index < len(builder["segments"]) - 1 else ["disabled"]
        )
        builder["remove_btn"].state(["!disabled"] if has_active else ["disabled"])

    def _set_filename_segments(self, builder_key, segments):
        builder = self.filename_builders[builder_key]
        builder["segments"] = list(segments)
        if builder["active"] not in builder["segments"]:
            builder["active"] = builder["segments"][-1] if builder["segments"] else None
        self._render_filename_builder(builder_key)
        self._update_filename_preview()

    def _add_filename_segment(self, builder_key, segment):
        builder = self.filename_builders[builder_key]
        if segment in builder["segments"]:
            return
        builder["segments"].append(segment)
        builder["active"] = segment
        self._render_filename_builder(builder_key)
        self._update_filename_preview()

    def _select_filename_segment(self, builder_key, segment):
        builder = self.filename_builders[builder_key]
        if builder["active"] == segment:
            return
        builder["active"] = segment
        self._update_filename_builder_controls(builder_key)

    def _move_filename_segment(self, builder_key, delta):
        builder = self.filename_builders[builder_key]
        segment = builder["active"]
        if segment not in builder["segments"]:
            return
        old_index = builder["segments"].index(segment)
        new_index = old_index + delta
        if new_index < 0 or new_index >= len(builder["segments"]):
            return
        builder["segments"].pop(old_index)
        builder["segments"].insert(new_index, segment)
        self._render_filename_builder(builder_key)
        self._update_filename_preview()

    def _remove_filename_segment(self, builder_key):
        builder = self.filename_builders[builder_key]
        segment = builder["active"]
        if segment not in builder["segments"]:
            return
        index = builder["segments"].index(segment)
        builder["segments"].pop(index)
        builder["active"] = builder["segments"][min(index, len(builder["segments"]) - 1)] if builder["segments"] else None
        self._render_filename_builder(builder_key)
        self._update_filename_preview()

    def _reset_filename_segments(self, builder_key):
        defaults = DEFAULT_IMAGE_FILENAME_SEGMENTS if builder_key == "image" else DEFAULT_MASK_FILENAME_SEGMENTS
        self._set_filename_segments(builder_key, defaults)

    def _update_filename_preview(self):
        if getattr(self, "_settings_preview_suspended", False):
            return
        if not hasattr(self, "filename_preview_var") or not hasattr(self, "filename_builders"):
            return
        try:
            image_name, mask_name, volume_mask_name = render_filename_preview(
                self.filename_builders["image"]["segments"],
                self.filename_builders["mask"]["segments"],
                self.settings_vars["filename_separator"].get(),
            )
            self.filename_preview_var.set(
                "Preview:\n"
                f"  Image:      {image_name}\n"
                f"  Mask:       {mask_name}\n"
                f"  3D Mask:    {volume_mask_name}"
            )
        except OutputFilenameError as exc:
            self.filename_preview_var.set(f"Preview unavailable: {exc}")

    def _output_structure_token_meta(self, token):
        """
        Return display metadata for an output-structure token.

        Args:
            token (str): Template token.

        Returns:
            tuple[str, str]: Label and color.
        """
        for label, candidate, color in self.SETTINGS_OUTPUT_STRUCTURE_TOKENS:
            if candidate == token:
                return label, color
        return token, "#eeeeee"

    def _parse_output_structure_tokens(self, template):
        """
        Parse a template into selectable tag tokens.

        Args:
            template (str): Group folder template.

        Returns:
            list[str] or None: Tokens when the template can be represented by the builder.
        """
        known = {token for _label, token, _color in self.SETTINGS_OUTPUT_STRUCTURE_TOKENS}
        parts = [part for part in template.strip().strip("/").split("/") if part]
        if not parts or parts[0] != self.SETTINGS_OUTPUT_STRUCTURE_REQUIRED_PREFIX:
            return None
        editable_parts = parts[1:]
        if len(set(editable_parts)) != len(editable_parts):
            return None
        if self.SETTINGS_OUTPUT_STRUCTURE_REQUIRED_PREFIX in editable_parts:
            return None
        if any(part not in known for part in editable_parts):
            return None
        if self.SETTINGS_OUTPUT_STRUCTURE_EXCLUSIVE_TOKENS.issubset(set(editable_parts)):
            return None
        return editable_parts

    def _sync_output_structure_template_from_tags(self):
        """
        Update the saved template field from the current tag order.

        Args:
            None.

        Returns:
            None.
        """
        parts = [self.SETTINGS_OUTPUT_STRUCTURE_REQUIRED_PREFIX] + self.output_structure_selected_tokens
        self.settings_vars["group_folder_template"].set("/".join(parts))

    def _render_output_structure_tag_builder(self):
        """
        Refresh selected chips, disabled available tags, and controls.

        Args:
            None.

        Returns:
            None.
        """
        if not hasattr(self, "output_structure_selected_frame"):
            return

        for child in self.output_structure_selected_frame.winfo_children():
            child.destroy()
        self.output_structure_selected_chips = {}

        label, color = "Dataset", "#dbeafe"
        dataset_chip = tk.Button(
            self.output_structure_selected_frame,
            text=label,
            background=color,
            activebackground=color,
            relief="sunken",
            borderwidth=2,
            state="disabled",
        )
        dataset_chip.grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self._attach_settings_tooltip(
            dataset_chip,
            self.SETTINGS_OUTPUT_STRUCTURE_TOOLTIPS["{dataset}"],
        )

        for idx, token in enumerate(self.output_structure_selected_tokens):
            label, color = self._output_structure_token_meta(token)
            relief = "sunken" if token == self.output_structure_active_token else "raised"
            chip = tk.Button(
                self.output_structure_selected_frame,
                text=label,
                background=color,
                activebackground=color,
                relief=relief,
                borderwidth=2,
                command=lambda t=token: self._select_output_structure_token(t),
            )
            chip.grid(row=0, column=idx + 1, sticky="w", padx=2, pady=2)
            self._attach_settings_tooltip(chip, self.SETTINGS_OUTPUT_STRUCTURE_TOOLTIPS[token])
            self.output_structure_selected_chips[token] = chip

        if not self.output_structure_selected_tokens:
            ttk.Label(self.output_structure_selected_frame, text="Add folders after Dataset.").grid(
                row=0, column=1, sticky="w", padx=2, pady=2
            )

        used = set(self.output_structure_selected_tokens)
        for token, button in self.output_structure_token_buttons.items():
            disabled = token in used or (
                token in self.SETTINGS_OUTPUT_STRUCTURE_EXCLUSIVE_TOKENS
                and bool((self.SETTINGS_OUTPUT_STRUCTURE_EXCLUSIVE_TOKENS - {token}) & used)
            )
            button.configure(state=("disabled" if disabled else "normal"))

        self._update_output_structure_controls()

    def _update_output_structure_controls(self):
        used = set(self.output_structure_selected_tokens)
        for token, chip in getattr(self, "output_structure_selected_chips", {}).items():
            relief = "sunken" if token == self.output_structure_active_token else "raised"
            chip.configure(relief=relief)

        active = self.output_structure_active_token
        has_active = active in used
        active_index = self.output_structure_selected_tokens.index(active) if has_active else -1
        self.output_structure_up_btn.state(["!disabled"] if active_index > 0 else ["disabled"])
        self.output_structure_down_btn.state(
            ["!disabled"]
            if has_active and active_index < len(self.output_structure_selected_tokens) - 1
            else ["disabled"]
        )
        self.output_structure_remove_btn.state(["!disabled"] if has_active else ["disabled"])

    def _set_output_structure_tokens(self, tokens, status=""):
        """
        Set the tag list and refresh the generated template.

        Args:
            tokens (list[str]): Selected tokens.
            status (str): Optional status text.

        Returns:
            None.
        """
        self.output_structure_selected_tokens = list(tokens)
        if self.output_structure_active_token not in self.output_structure_selected_tokens:
            self.output_structure_active_token = (
                self.output_structure_selected_tokens[-1]
                if self.output_structure_selected_tokens
                else None
            )
        self.output_structure_builder_status_var.set(status)
        self._sync_output_structure_template_from_tags()
        self._render_output_structure_tag_builder()

    def _add_output_structure_token(self, token):
        """
        Add a token chip to the folder layout.

        Args:
            token (str): Template token.

        Returns:
            None.
        """
        if token in self.output_structure_selected_tokens:
            return
        if (
            token in self.SETTINGS_OUTPUT_STRUCTURE_EXCLUSIVE_TOKENS
            and bool((self.SETTINGS_OUTPUT_STRUCTURE_EXCLUSIVE_TOKENS - {token}) & set(self.output_structure_selected_tokens))
        ):
            return
        self.output_structure_selected_tokens.append(token)
        self.output_structure_active_token = token
        self.output_structure_builder_status_var.set("")
        self._sync_output_structure_template_from_tags()
        self._render_output_structure_tag_builder()

    def _select_output_structure_token(self, token):
        """
        Select a tag chip for moving or removal.

        Args:
            token (str): Template token.

        Returns:
            None.
        """
        if self.output_structure_active_token == token:
            return
        self.output_structure_active_token = token
        self._update_output_structure_controls()

    def _move_output_structure_token(self, delta):
        """
        Move the selected tag by one position.

        Args:
            delta (int): -1 to move left/up, 1 to move right/down.

        Returns:
            None.
        """
        token = self.output_structure_active_token
        if token not in self.output_structure_selected_tokens:
            return
        old_index = self.output_structure_selected_tokens.index(token)
        new_index = old_index + delta
        if new_index < 0 or new_index >= len(self.output_structure_selected_tokens):
            return
        self.output_structure_selected_tokens.pop(old_index)
        self.output_structure_selected_tokens.insert(new_index, token)
        self._sync_output_structure_template_from_tags()
        self._render_output_structure_tag_builder()

    def _remove_output_structure_token(self):
        """
        Remove the selected tag from the folder layout.

        Args:
            None.

        Returns:
            None.
        """
        token = self.output_structure_active_token
        if token not in self.output_structure_selected_tokens:
            return
        index = self.output_structure_selected_tokens.index(token)
        self.output_structure_selected_tokens.pop(index)
        if self.output_structure_selected_tokens:
            self.output_structure_active_token = self.output_structure_selected_tokens[
                min(index, len(self.output_structure_selected_tokens) - 1)
            ]
        else:
            self.output_structure_active_token = None
        self._sync_output_structure_template_from_tags()
        self._render_output_structure_tag_builder()

    def _reset_output_structure_template(self):
        """
        Restore the default output folder tags.

        Args:
            None.

        Returns:
            None.
        """
        tokens = self._parse_output_structure_tokens(DEFAULT_GROUP_FOLDER_TEMPLATE) or []
        self._set_output_structure_tokens(tokens)

    def _update_output_structure_preview(self):
        """
        Update the example output-structure preview.

        Args:
            None.

        Returns:
            None.
        """
        if getattr(self, "_settings_preview_suspended", False):
            return
        if not hasattr(self, "output_structure_preview_var"):
            return
        try:
            img_path, mask_path = render_preview(
                self.settings_vars["group_folder_template"].get(),
                self.settings_vars["images_folder"].get(),
                self.settings_vars["masks_folder"].get(),
            )
            complex_img_path, complex_mask_path = render_preview(
                self.settings_vars["group_folder_template"].get(),
                self.settings_vars["images_folder"].get(),
                self.settings_vars["masks_folder"].get(),
                identifier="Patient42_Study7_Slice003",
            )
            self.output_structure_preview_var.set(
                "Preview:\n"
                f"  Images: {img_path}\n"
                f"  Masks:  {mask_path}\n"
                "Complex ID example:\n"
                f"  Images: {complex_img_path}\n"
                f"  Masks:  {complex_mask_path}"
            )
        except OutputStructureError as exc:
            self.output_structure_preview_var.set(f"Preview unavailable: {exc}")

    def _load_settings_values(self):
        """
        Load current effective config values into the settings form.

        Args:
            None.

        Returns:
            None.
        """
        self._settings_preview_suspended = True
        parser = self._read_settings_parser()
        path_fallbacks = {
            "preprocessing_log_dir": config.PREPROCESSING_LOG_DIR,
            "base_unproc": config.BASE_UNPROC,
            "base_proc": config.BASE_PROC,
            "groups_dir": config.GROUPS_DIR,
            "index_dir": config.INDEX_DIR,
            "csv_filename": config.CSV_FILENAME,
        }
        for key, _label in self.SETTINGS_PATH_FIELDS:
            self.settings_vars[key].set(self._settings_get(parser, "paths", key, path_fallbacks[key]))
        self.settings_vars["csv_filename"].set(
            self._settings_get(parser, "paths", "csv_filename", path_fallbacks["csv_filename"])
        )
        self.settings_vars["target_size"].set(
            self._settings_get(parser, "preprocessing", "target_size", config.DEFAULT_TARGET_SIZE)
        )
        self.settings_vars["min_component_size"].set(
            self._settings_get(parser, "preprocessing", "min_component_size", config.MIN_COMPONENT_SIZE)
        )
        for key, _label in self.SETTINGS_OUTPUT_FIELDS:
            runtime_key = self.SETTINGS_OUTPUT_RUNTIME_KEYS[key]
            self.settings_vars[key].set(
                self._settings_get(parser, "output_formats", key, config.OUTPUT_FORMATS[runtime_key])
            )
        structure_values = {
            key: self._settings_get(parser, "output_structure", key, config.OUTPUT_STRUCTURE[key])
            for key, _label in self.SETTINGS_OUTPUT_STRUCTURE_FIELDS
        }
        for key, value in structure_values.items():
            self.settings_vars[key].set(value)

        tokens = self._parse_output_structure_tokens(structure_values["group_folder_template"])
        if tokens is None:
            self.output_structure_selected_tokens = []
            self.output_structure_active_token = None
            self.output_structure_builder_status_var.set(
                "Custom template loaded from dripp.ini. Add tags or use Default to replace it."
            )
            self._render_output_structure_tag_builder()
        else:
            self._set_output_structure_tokens(tokens)
        filename_values = {
            "image_segments": self._settings_get(
                parser,
                "output_filenames",
                "image_segments",
                config.OUTPUT_FILENAMES["image_segments"],
            ),
            "mask_segments": self._settings_get(
                parser,
                "output_filenames",
                "mask_segments",
                config.OUTPUT_FILENAMES["mask_segments"],
            ),
            "separator": self._settings_get(
                parser,
                "output_filenames",
                "separator",
                config.OUTPUT_FILENAMES["separator"],
            ),
        }
        try:
            filename_config = validate_output_filenames(
                filename_values["image_segments"],
                filename_values["mask_segments"],
                filename_values["separator"],
            )
            self._set_filename_segments("image", filename_config["image_segments"])
            self._set_filename_segments("mask", filename_config["mask_segments"])
            self.settings_vars["filename_separator"].set(filename_config["separator"])
        except OutputFilenameError:
            self._set_filename_segments("image", DEFAULT_IMAGE_FILENAME_SEGMENTS)
            self._set_filename_segments("mask", DEFAULT_MASK_FILENAME_SEGMENTS)
            self.settings_vars["filename_separator"].set(config.OUTPUT_FILENAMES["separator"])
        self.settings_vars["gpu_enabled"].set(False)
        self.settings_vars["default_log_level"].set(
            self._settings_get(
                parser,
                "logging",
                "default_log_level",
                logging.getLevelName(config.DEFAULT_LOG_LEVEL),
            ).upper()
        )
        self.settings_status_var.set(f"Config: {self._settings_config_path()}")
        self._settings_preview_suspended = False
        self._update_output_structure_preview()
        self._update_filename_preview()

    def _reset_unsaved_settings(self):
        """
        Reset the form to the currently saved settings.

        Args:
            None.

        Returns:
            None.
        """
        self._load_settings_values()

    def _browse_settings_directory(self, key):
        """
        Select a directory for a settings path.

        Args:
            key (str): Settings variable key to update.

        Returns:
            None.
        """
        initial = self.settings_vars[key].get().strip()
        kwargs = {"initialdir": initial} if initial else {}
        selected = filedialog.askdirectory(**kwargs)
        if selected:
            self.settings_vars[key].set(normalize_path(selected))

    def _browse_settings_csv(self):
        """
        Select a datasets CSV and map it to base_unproc plus csv_filename.

        Args:
            None.

        Returns:
            None.
        """
        initial = self.settings_vars["base_unproc"].get().strip()
        kwargs = {"initialdir": initial} if initial else {}
        selected = filedialog.askopenfilename(
            title="Locate DRIPP datasets CSV",
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
            **kwargs,
        )
        if selected:
            selected = normalize_path(selected)
            self.settings_vars["base_unproc"].set(normalize_path(os.path.dirname(selected)))
            self.settings_vars["csv_filename"].set(os.path.basename(selected))

    def _normalize_settings_ext(self, value):
        """
        Normalize a file extension value.

        Args:
            value (str): Raw extension.

        Returns:
            str: Normalized extension.
        """
        value = value.strip().lower()
        if not value:
            return ""
        return value if value.startswith(".") else f".{value}"

    def _validate_settings(self):
        """
        Validate settings form values.

        Args:
            None.

        Returns:
            tuple[dict, list[str]]: Validated values and error messages.
        """
        errors = []
        values = {
            "paths": {},
            "preprocessing": {},
            "output_formats": {},
            "output_structure": {},
            "output_filenames": {},
            "runtime": {},
            "logging": {},
        }

        for key, label in self.SETTINGS_PATH_FIELDS:
            raw = self.settings_vars[key].get().strip()
            if not raw:
                errors.append(f"{label} is required.")
            values["paths"][key] = normalize_path(raw) if raw else raw

        csv_filename = self.settings_vars["csv_filename"].get().strip()
        if not csv_filename:
            errors.append("CSV Filename is required.")
        values["paths"]["csv_filename"] = csv_filename

        target_raw = self.settings_vars["target_size"].get().strip()
        parts = [part.strip() for part in target_raw.lower().replace("x", ",").split(",") if part.strip()]
        try:
            target_size = tuple(int(part) for part in parts)
            if len(target_size) != 2 or any(part <= 0 for part in target_size):
                raise ValueError
            values["preprocessing"]["target_size"] = target_size
        except ValueError:
            errors.append("Target Size must be exactly two positive integers, e.g. 512, 512.")

        min_raw = self.settings_vars["min_component_size"].get().strip()
        try:
            min_component_size = int(min_raw)
            if min_component_size < 0:
                raise ValueError
            values["preprocessing"]["min_component_size"] = min_component_size
        except ValueError:
            errors.append("Min Component Size must be a non-negative integer.")

        two_d_fields = {"2d_image_format", "2d_mask_format"}
        image_fields = {"video_frame_format", "video_mask_format"}
        for key, label in self.SETTINGS_OUTPUT_FIELDS:
            ext = self._normalize_settings_ext(self.settings_vars[key].get())
            allowed = (
                config.SUPPORTED_2D_OUTPUT_EXTS
                if key in two_d_fields
                else config.SUPPORTED_IMAGE_OUTPUT_EXTS
                if key in image_fields
                else config.SUPPORTED_VOLUME_OUTPUT_EXTS
            )
            if ext not in allowed:
                errors.append(f"{label} must be one of: {', '.join(sorted(allowed))}.")
            values["output_formats"][key] = ext

        try:
            values["output_structure"]["group_folder_template"] = validate_group_folder_template(
                self.settings_vars["group_folder_template"].get()
            )
        except OutputStructureError as exc:
            errors.append(str(exc))
        try:
            values["output_structure"]["images_folder"] = validate_leaf_folder(
                self.settings_vars["images_folder"].get(),
                "Images Folder",
            )
        except OutputStructureError as exc:
            errors.append(str(exc))
        try:
            values["output_structure"]["masks_folder"] = validate_leaf_folder(
                self.settings_vars["masks_folder"].get(),
                "Masks Folder",
            )
        except OutputStructureError as exc:
            errors.append(str(exc))

        try:
            filename_config = validate_output_filenames(
                self.filename_builders["image"]["segments"],
                self.filename_builders["mask"]["segments"],
                self.settings_vars["filename_separator"].get(),
            )
            values["output_filenames"] = filename_config
        except OutputFilenameError as exc:
            errors.append(str(exc))

        values["runtime"]["gpu_enabled"] = False
        level = self.settings_vars["default_log_level"].get().strip().upper()
        if level not in self.SETTINGS_LOG_LEVELS:
            errors.append("Log Level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL.")
        values["logging"]["default_log_level"] = level

        return values, errors

    def _ensure_settings_sections(self, parser):
        """
        Ensure sections edited by the settings tab exist.

        Args:
            parser (configparser.ConfigParser): Parser to update.

        Returns:
            None.
        """
        for section in (
            "paths",
            "preprocessing",
            "output_formats",
            "output_structure",
            "output_filenames",
            "runtime",
            "logging",
        ):
            if not parser.has_section(section):
                parser.add_section(section)

    def _write_settings(self, values):
        """
        Write validated settings to local dripp.ini.

        Args:
            values (dict): Validated settings values.

        Returns:
            None.
        """
        parser = self._read_settings_parser()
        self._ensure_settings_sections(parser)
        for key, value in values["paths"].items():
            parser.set("paths", key, str(value))
        target_size = values["preprocessing"]["target_size"]
        parser.set("preprocessing", "target_size", f"{target_size[0]}, {target_size[1]}")
        parser.set("preprocessing", "min_component_size", str(values["preprocessing"]["min_component_size"]))
        for key, value in values["output_formats"].items():
            parser.set("output_formats", key, value)
        for key, value in values["output_structure"].items():
            parser.set("output_structure", key, value)
        parser.set("output_filenames", "image_segments", ", ".join(values["output_filenames"]["image_segments"]))
        parser.set("output_filenames", "mask_segments", ", ".join(values["output_filenames"]["mask_segments"]))
        parser.set("output_filenames", "separator", values["output_filenames"]["separator"])
        parser.remove_option("runtime", "gpu_enabled")
        parser.set("logging", "default_log_level", values["logging"]["default_log_level"])
        with open(self._settings_config_path(), "w") as f:
            parser.write(f)

    def _apply_settings_to_runtime(self, values):
        """
        Apply validated settings to modules already imported by the debugger.

        Args:
            values (dict): Validated settings values.

        Returns:
            set[str]: Names of path values that changed.
        """
        old_paths = {
            "base_unproc": config.BASE_UNPROC,
            "base_proc": config.BASE_PROC,
            "groups_dir": config.GROUPS_DIR,
            "index_dir": config.INDEX_DIR,
            "output_structure": dict(config.OUTPUT_STRUCTURE),
            "output_filenames": dict(config.OUTPUT_FILENAMES),
        }

        paths = values["paths"]
        config.PREPROCESSING_LOG_DIR = paths["preprocessing_log_dir"]
        config.BASE_UNPROC = paths["base_unproc"]
        config.BASE_PROC = paths["base_proc"]
        config.GROUPS_DIR = paths["groups_dir"]
        config.INDEX_DIR = paths["index_dir"]
        config.CSV_FILENAME = paths["csv_filename"]
        config.DEFAULT_TARGET_SIZE = values["preprocessing"]["target_size"]
        config.MIN_COMPONENT_SIZE = values["preprocessing"]["min_component_size"]
        config.OUTPUT_FORMATS = {
            runtime_key: values["output_formats"][option_key]
            for option_key, runtime_key in self.SETTINGS_OUTPUT_RUNTIME_KEYS.items()
        }
        config.OUTPUT_EXTS = set(config.OUTPUT_FORMATS.values())
        config.IMAGE_OUTPUT_EXTS = {
            ext for ext in config.OUTPUT_EXTS
            if ext in config.SUPPORTED_IMAGE_OUTPUT_EXTS
        }
        config.VOLUME_OUTPUT_EXTS = {
            ext for ext in config.OUTPUT_EXTS
            if ext in config.SUPPORTED_VOLUME_OUTPUT_EXTS
        }
        config.OUTPUT_STRUCTURE = dict(values["output_structure"])
        config.OUTPUT_FILENAMES = dict(values["output_filenames"])
        config.GPU_ENABLED = values["runtime"]["gpu_enabled"]
        config.DEFAULT_LOG_LEVEL = getattr(logging, values["logging"]["default_log_level"])

        for module in (debugger_common, grouping_debugger, preprocessing_debugger):
            module.BASE_UNPROC = config.BASE_UNPROC
            module.CSV_FILENAME = config.CSV_FILENAME
            module.INDEX_DIR = config.INDEX_DIR

        indexer_module.BASE_UNPROC = config.BASE_UNPROC
        indexer_module.INDEX_DIR = config.INDEX_DIR
        indexer_module.CSV_FILENAME = config.CSV_FILENAME
        indexer_module.DEFAULT_LOG_LEVEL = config.DEFAULT_LOG_LEVEL
        indexer_module.logger.setLevel(config.DEFAULT_LOG_LEVEL)
        dataset_module.BASE_UNPROC = config.BASE_UNPROC
        dataset_module.BASE_PROC = config.BASE_PROC
        dataset_module.INDEX_DIR = config.GROUPS_DIR
        dataset_module.CSV_FILENAME = config.CSV_FILENAME
        dataset_module.DEFAULT_LOG_LEVEL = config.DEFAULT_LOG_LEVEL
        dataset_module.GPU_ENABLED = config.GPU_ENABLED
        preprocessor_module.PREPROCESSING_LOG_DIR = config.PREPROCESSING_LOG_DIR
        preprocessor_module.DEFAULT_TARGET_SIZE = config.DEFAULT_TARGET_SIZE
        preprocessor_module.MIN_COMPONENT_SIZE = config.MIN_COMPONENT_SIZE
        preprocessor_module.DEFAULT_LOG_LEVEL = config.DEFAULT_LOG_LEVEL
        preprocessor_module.main_logger.setLevel(config.DEFAULT_LOG_LEVEL)
        helpers_module.GPU_ENABLED = config.GPU_ENABLED

        changed_paths = {
            key for key, old_value in old_paths.items()
            if old_value != (
                config.OUTPUT_STRUCTURE
                if key == "output_structure"
                else config.OUTPUT_FILENAMES
                if key == "output_filenames"
                else paths[key]
            )
        }
        return changed_paths

    def _refresh_after_settings_save(self):
        """
        Refresh open debugger state after settings are saved.

        Args:
            None.

        Returns:
            bool: True when metadata refreshed successfully.
        """
        self.csv_path = normalize_path(os.path.join(config.BASE_UNPROC, config.CSV_FILENAME))
        try:
            self.metadata = load_dataset_metadata(self.csv_path)
        except Exception as e:
            self.metadata = {}
            messagebox.showwarning(
                "Settings Saved",
                "Settings were saved, but DRIPP could not load the configured datasets CSV:\n\n"
                f"{self.csv_path}\n\n{e}"
            )

        if hasattr(self, "ds_combo"):
            self.ds_combo["values"] = list(self.metadata.keys())
        if hasattr(self, "preproc_ds_combo"):
            self.preproc_ds_combo["values"] = list(self.metadata.keys())
        if hasattr(self, "dataset_var"):
            self.dataset_var.set("")
        if hasattr(self, "preproc_dataset_var"):
            self.preproc_dataset_var.set("")
        if hasattr(self, "base_var"):
            self.base_var.set("")
        if hasattr(self, "save_json_btn"):
            self.save_json_btn.state(["disabled"])
        if hasattr(self, "load_pre_btn"):
            self.load_pre_btn.state(["disabled"])
        if hasattr(self, "group_tree"):
            self.group_tree.delete(*self.group_tree.get_children())
        if hasattr(self, "current_images"):
            self.current_images.clear()
        if hasattr(self, "current_masks"):
            self.current_masks.clear()
        self.current_group = None
        if hasattr(self, "csv_status_var"):
            self._load_csv_table(show_errors=False)
        if hasattr(self, "update_nav_buttons"):
            self.update_nav_buttons()
        return bool(self.metadata)

    def _save_settings(self):
        """
        Validate, save, and hot-apply debugger settings.

        Args:
            None.

        Returns:
            None.
        """
        values, errors = self._validate_settings()
        if errors:
            messagebox.showerror("Invalid Settings", "\n".join(errors))
            return

        selected_dataset = self.dataset_var.get() if hasattr(self, "dataset_var") else ""
        selected_preproc = self.preproc_dataset_var.get() if hasattr(self, "preproc_dataset_var") else ""
        had_loaded_state = bool(selected_dataset or selected_preproc or getattr(self, "current_group", None))

        try:
            self._write_settings(values)
            changed_paths = self._apply_settings_to_runtime(values)
            self._refresh_after_settings_save()
            self._load_settings_values()
        except Exception as e:
            messagebox.showerror("Settings Save Error", f"Could not save settings:\n\n{e}")
            return

        if had_loaded_state and changed_paths & {
            "base_unproc",
            "base_proc",
            "groups_dir",
            "index_dir",
            "output_structure",
            "output_filenames",
        }:
            messagebox.showwarning(
                "Settings Applied",
                "Settings were saved and applied. Output or indexing paths changed while a dataset was loaded; "
                "please reselect or reload the dataset before running more debugger operations."
            )
        else:
            messagebox.showinfo("Settings Saved", f"Saved settings to:\n{self._settings_config_path()}")
