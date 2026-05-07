"""Shared imports and utilities for the Tk-based DRIPP debugger."""

import ctypes
import json
import logging
import os
import re
import subprocess
import tkinter as tk
from ctypes import wintypes
from logging import Formatter
from tkinter import filedialog, messagebox, scrolledtext, ttk

import numpy as np
import pandas as pd
import SimpleITK as sitk
from PIL import Image, ImageTk

from dripp.config import BASE_UNPROC, CSV_FILENAME, INDEX_DIR
from dripp.dataset import DatasetManager
from dripp.helpers import (
    collect_files,
    extract_mask_class,
    get_composite_identifier,
    get_preprocessing_options,
    get_extension,
    get_regex_configs,
    load_dataset_metadata,
    normalize_path,
    parse_mask_classes,
    resolve_folder,
    resolve_mask_folders,
    set_indexing_log,
)
from dripp.indexer import build_groupings, index_dataset
from dripp.preprocessor import Preprocessor


if os.name == "nt":
    SHILCreateFromPath = ctypes.windll.shell32.SHILCreateFromPath
    SHILCreateFromPath.argtypes = [
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
    ]
    SHILCreateFromPath.restype = ctypes.HRESULT

    SHOpenFolderAndSelectItems = ctypes.windll.shell32.SHOpenFolderAndSelectItems
    SHOpenFolderAndSelectItems.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_uint,
    ]
    SHOpenFolderAndSelectItems.restype = ctypes.HRESULT
else:
    SHILCreateFromPath = None
    SHOpenFolderAndSelectItems = None


class TextHandler(logging.Handler):
    """
    Forward log records into a Tk text widget.

    Args:
        None.

    Returns:
        None.
    """

    def __init__(self, text_widget):
        """
        Initialize the object.

        Args:
            text_widget (Any): Text or ScrolledText widget that receives log output.

        Returns:
            None.
        """
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        """
        Append a formatted log record to the text widget.

        Args:
            record (Any): Logging record to emit.

        Returns:
            None.
        """
        msg = self.format(record)
        # Insert message at the end, then scroll to it.
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.configure(state='disabled')
        self.text_widget.see(tk.END)
