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

    def _build_settings_tab(self):
        """
        Build the settings editor tab.

        Args:
            None.

        Returns:
            None.
        """
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text="Settings")

        toolbar = ttk.Frame(self.settings_tab)
        toolbar.pack(fill="x", padx=10, pady=8)
        ttk.Button(toolbar, text="Save", command=self._save_settings).pack(side="right", padx=2)
        ttk.Button(toolbar, text="Reload", command=self._load_settings_values).pack(side="right", padx=2)
        ttk.Button(toolbar, text="Reset Unsaved", command=self._reset_unsaved_settings).pack(side="right", padx=2)

        self.settings_status_var = tk.StringVar()
        ttk.Label(self.settings_tab, textvariable=self.settings_status_var, anchor="w").pack(fill="x", padx=12)

        body = ttk.Frame(self.settings_tab)
        body.pack(fill="both", expand=True, padx=10, pady=8)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        self.settings_vars = {}

        paths_frame = ttk.LabelFrame(body, text="Paths")
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

        preproc_frame = ttk.LabelFrame(body, text="Preprocessing")
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

        runtime_frame = ttk.LabelFrame(body, text="Runtime & Logging")
        runtime_frame.grid(row=1, column=1, sticky="new", padx=(6, 2), pady=2)
        runtime_frame.columnconfigure(1, weight=1)
        self.settings_vars["gpu_enabled"] = tk.BooleanVar()
        self.settings_vars["default_log_level"] = tk.StringVar()
        ttk.Checkbutton(
            runtime_frame,
            text="GPU Enabled",
            variable=self.settings_vars["gpu_enabled"]
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=3)
        ttk.Label(runtime_frame, text="Log Level:").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        ttk.Combobox(
            runtime_frame,
            textvariable=self.settings_vars["default_log_level"],
            values=self.SETTINGS_LOG_LEVELS,
            state="readonly",
        ).grid(row=1, column=1, sticky="we", padx=4, pady=3)

        output_frame = ttk.LabelFrame(body, text="Output Formats")
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

        self._load_settings_values()

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

    def _load_settings_values(self):
        """
        Load current effective config values into the settings form.

        Args:
            None.

        Returns:
            None.
        """
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
        self.settings_vars["gpu_enabled"].set(
            parser.getboolean("runtime", "gpu_enabled", fallback=config.GPU_ENABLED)
        )
        self.settings_vars["default_log_level"].set(
            self._settings_get(
                parser,
                "logging",
                "default_log_level",
                logging.getLevelName(config.DEFAULT_LOG_LEVEL),
            ).upper()
        )
        self.settings_status_var.set(f"Config: {self._settings_config_path()}")

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
        values = {"paths": {}, "preprocessing": {}, "output_formats": {}, "runtime": {}, "logging": {}}

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

        values["runtime"]["gpu_enabled"] = bool(self.settings_vars["gpu_enabled"].get())
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
        for section in ("paths", "preprocessing", "output_formats", "runtime", "logging"):
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
        parser.set("runtime", "gpu_enabled", str(values["runtime"]["gpu_enabled"]).lower())
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
            if old_value != paths[key]
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

        if had_loaded_state and changed_paths & {"base_unproc", "base_proc", "groups_dir", "index_dir"}:
            messagebox.showwarning(
                "Settings Applied",
                "Settings were saved and applied. Output or indexing paths changed while a dataset was loaded; "
                "please reselect or reload the dataset before running more debugger operations."
            )
        else:
            messagebox.showinfo("Settings Saved", f"Saved settings to:\n{self._settings_config_path()}")
