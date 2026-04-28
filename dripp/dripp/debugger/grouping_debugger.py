"""Grouping and regex-debugger callbacks for the DRIPP debugger app."""

from .common import *


class GroupingDebuggerMixin:
    """
    Provide callbacks for the grouping and regex debugger tab.

    Args:
        None.

    Returns:
        None.
    """

    def save_json(self):
        """
        Save the current dataset index and group JSON files.

        Args:
            None.

        Returns:
            None.
        """
        name = self.dataset_var.get()
        if not name:
            messagebox.showwarning("No Dataset", "Please select a dataset first.")
            return

        # Re-build index_data & groups exactly as in test_regex()
        base_dir = self.base_var.get()
        meta = self.metadata[name]

        # Collect files
        img_types  = [e.strip() for e in self.img_ext_var.get().split(',') if e.strip()]
        mask_types = [e.strip() for e in self.mask_ext_var.get().split(',') if e.strip()]

        train_recs = collect_files("", self.train_paths, img_types,
                                   use_folder_identifier=False,
                                   exclude_folders=meta.get("mask_folders"),
                                   mask_key=meta.get("mask_key"))
        train_recs = sorted(train_recs, key=lambda x: x["path"])

        test_recs = collect_files("", self.test_paths, img_types,
                                  use_folder_identifier=False,
                                  exclude_folders=meta.get("mask_folders"),
                                  mask_key=meta.get("mask_key"))
        test_recs = sorted(test_recs, key=lambda x: x["path"])

        mask_recs = collect_files("", self.mask_paths, mask_types,
                                  is_mask=True,
                                  mask_key=meta.get("mask_key"))
        mask_recs = sorted(mask_recs, key=lambda x: x["path"])

        index_data = {
            "images_train": train_recs,
            "images_test":  test_recs,
            "mask_files":   mask_recs
        }

        groups = build_groupings(index_data, base_dir, meta)
        # `Groups` is a dict { "subdatasets": [...] }; indexer writes groups=subdatasets
        subdatasets = groups.get("subdatasets", [])
        index_data_full = {
            **index_data,
            "groups": subdatasets
        }

        # Ask for folder
        folder = filedialog.askdirectory(title="Select folder to save JSONs")
        if not folder:
            return

        # Write index JSON
        idx_path = os.path.join(folder, f"{name}_index.json")
        with open(idx_path, "w") as f:
            json.dump(index_data_full, f, indent=2)

        # Write groups-only JSON
        grp_path = os.path.join(folder, f"{name}_groups.json")
        with open(grp_path, "w") as f:
            json.dump(subdatasets, f, indent=2)

        messagebox.showinfo(
            "Saved JSON",
            f"Index file saved to:\n  {idx_path}\n\n"
            f"Groups file saved to:\n  {grp_path}"
        )

    def px_test(self):
        """
        Run the Pythex-style regex test panel.

        Args:
            None.

        Returns:
            None.
        """
        self.px_results.delete('1.0', tk.END)
        pattern = self.px_regex.get('1.0', tk.END).strip()
        text = self.px_test_str.get('1.0', tk.END)
        if not pattern:
            messagebox.showwarning('Input Required', 'Please enter a regex.')
            return
        try:
            regex = re.compile(pattern)
        except re.error as e:
            self.px_results.insert(tk.END, f'Regex error: {e}\n')
            return
        matches = list(regex.finditer(text))
        if not matches:
            self.px_results.insert(tk.END, 'No matches found.\n')
            return
        for i, m in enumerate(matches, 1):
            self.px_results.insert(tk.END, f'Match {i}: {m.group(0)}\n')
            for name, val in m.groupdict().items():
                self.px_results.insert(tk.END, f'  {name}: {val}\n')
            self.px_results.insert(tk.END, '\n')

    def on_dataset_select(self, event):
        """
        Load dataset metadata into the grouping tab controls.

        Args:
            event (Any): Tk event object.

        Returns:
            None.
        """
        name = self.dataset_var.get()
        meta = self.metadata.get(name, {})
        # Load regex
        self.regex_text.delete('1.0', tk.END)
        regex = meta.get('grouping_regex') or ''
        self.regex_text.insert(tk.END, regex)
        self.original_regex = regex
        self.save_btn.state(['disabled'])
        # Load other fields
        self.strategy_var.set(meta.get('grouping_strategy','regex-file'))
        self.img_ext_var.set(','.join(meta.get('image_file_types',[])))
        self.mask_ext_var.set(','.join(meta.get('mask_file_types',[])))
        rf = meta.get('root_directory')
        if not isinstance(rf,str): rf = ''
        dataset_dir = normalize_path(os.path.join(BASE_UNPROC,name,rf))
        self.base_var.set(dataset_dir)
        # Resolve folders
        self.train_paths=[]
        self.test_paths=[]
        self.mask_paths=[]
        if not self.mask_paths: self.mask_paths=[dataset_dir]
        self.save_json_btn.state(['disabled'])
        # Clear UI
        for iid in self.config_tree.get_children():
            self.config_tree.delete(iid)
        for t in (self.resolved_tree,self.train_tree,self.test_tree): t.delete(*t.get_children())

    def on_regex_change(self, event):
        """
        Mark the grouping regex as modified.

        Args:
            event (Any): Tk event object.

        Returns:
            None.
        """
        current=self.regex_text.get('1.0',tk.END).rstrip('\n')
        if self.original_regex is not None and current!=self.original_regex:
            self.save_btn.state(['!disabled'])
        else:
            self.save_btn.state(['disabled'])

    def save_regex(self):
        """
        Persist the edited grouping regex to the datasets CSV.

        Args:
            None.

        Returns:
            None.
        """
        name          = self.dataset_var.get()
        new_regex     = self.regex_text.get('1.0', tk.END).strip()
        new_strategy  = self.strategy_var.get().strip()
        new_img_ext   = self.img_ext_var.get().strip()
        new_mask_ext  = self.mask_ext_var.get().strip()
        new_base_full = self.base_var.get().strip()

        # Compute relative root (after "F:/Datasets/")
        prefix = os.path.normpath("F:/Datasets/").replace("\\","/") + "/"
        norm_base = os.path.normpath(new_base_full).replace("\\","/")
        if norm_base.startswith(prefix):
            rel = norm_base[len(prefix):]
        else:
            rel = norm_base
        rel_to_save = "" if rel == name else rel

        try:
            df = pd.read_csv(self.csv_path)

            # Update known columns
            df.loc[df['Dataset Name'] == name, 'Grouping Regex']    = new_regex
            df.loc[df['Dataset Name'] == name, 'Grouping Strategy'] = new_strategy
            df.loc[df['Dataset Name'] == name, 'Image File Type']   = new_img_ext
            df.loc[df['Dataset Name'] == name, 'Mask File Type']    = new_mask_ext

            # Ensure a 'Root Folder' column exists
            if 'Root Folder' not in df.columns:
                df['Root Folder'] = ""
            df.loc[df['Dataset Name'] == name, 'Root Folder'] = rel_to_save

            # Write back and refresh
            df.to_csv(self.csv_path, index=False)
            self.metadata = load_dataset_metadata(self.csv_path)
            self.original_regex = new_regex
            self.save_btn.state(['disabled'])

            messagebox.showinfo('Saved',
                f"Updated '{name}' with:\n"
                f" - Grouping Regex\n"
                f" - Grouping Strategy\n"
                f" - Image File Type\n"
                f" - Mask File Type\n"
                f" - Root Folder (rel: '{rel_to_save}')"
            )
        except Exception as e:
            messagebox.showerror('Save Error', f'Failed to save CSV: {e}')

    def browse_directory(self):
        """
        Select a base directory for the current dataset.

        Args:
            None.

        Returns:
            None.
        """
        p=filedialog.askdirectory()
        if p: self.base_var.set(p)

    def parse_configs(self):
        """
        Parse and display subdataset regex configurations.

        Args:
            None.

        Returns:
            None.
        """
        # Clear old
        for i in self.config_tree.get_children():
            self.config_tree.delete(i)

        from dripp.helpers import get_regex_configs
        regex = self.regex_text.get('1.0', tk.END).strip()
        if not regex:
            messagebox.showwarning('Input Required','Please enter a grouping regex.')
            return

        configs = get_regex_configs(regex,{})
        if not configs:
            # Show a dummy node
            self.config_tree.insert('', 'end', text='No valid configs parsed.')
            return

        for idx, cfg in enumerate(configs, start=1):
            # Top-level node: use the name if available
            label = getattr(cfg, 'name', None) or f'Config {idx}'
            parent = self.config_tree.insert('', 'end', text=label, open=False)
            # Add each field of the config as a child
            for field_name, field_val in cfg.__dict__.items():
                if field_name.startswith('_'):
                    continue
                self.config_tree.insert(parent, 'end', text=field_name, values=(repr(field_val),))

    def test_regex(self):
        """
        Resolve files, build groups, and populate grouping trees.

        Args:
            None.

        Returns:
            None.
        """
        # 1) Clear existing entries in all trees
        for tree in (self.resolved_tree, self.train_tree, self.test_tree):
            tree.delete(*tree.get_children())

        # 2) Ensure a dataset is selected
        dataset_name = self.dataset_var.get()
        if not dataset_name:
            messagebox.showwarning('Select Dataset', 'Please select a dataset.')
            return

        # 3) Redirect all helper-logger output into this dataset's own log
        set_indexing_log(INDEX_DIR, dataset_name)

        # 4) Run the central indexing routine
        meta = self.metadata.get(dataset_name, {})
        result = index_dataset(dataset_name, meta)
        if result is None:
            messagebox.showerror(
                'Indexing Error',
                f'No indexing result for "{dataset_name}".'
            )
            return

        # 4A) Resolve dataset folder structure
        root = meta.get('root_directory') or ''
        dataset_dir = normalize_path(os.path.join(BASE_UNPROC, dataset_name, root))
        self.train_paths = []
        for fld in meta.get('train_folders', []):
            self.train_paths += resolve_folder(dataset_dir, fld)
        self.test_paths = []
        for fld in meta.get('test_folders', []):
            self.test_paths += resolve_folder(dataset_dir, fld)
        self.mask_paths = resolve_mask_folders(dataset_dir, meta.get('mask_folders', []))

        # 5) Warn about any unannotated (no-mask) groups
        unannotated = result.get("unannotated_groups", [])
        if unannotated:
            messagebox.showwarning(
                "Unannotated Groups",
                "The following groups were dropped (no masks found):\n\n"
                + "\n".join(unannotated)
            )

        # 6) Populate the "Resolved" tree with folders _and_ files
        root_id = self.resolved_tree.insert('', 'end', text='Resolved', open=True)

        # 6A) Folders
        fld_id = self.resolved_tree.insert(root_id, 'end', text='Folders', tags=('folder',), open=True)
        tf_id  = self.resolved_tree.insert(fld_id, 'end',
                                        text=f'Train Folders ({len(self.train_paths)})',
                                        tags=('folder',))
        for p in self.train_paths:
            self.resolved_tree.insert(tf_id, 'end', text=p, tags=('folder',))

        tstf_id = self.resolved_tree.insert(fld_id, 'end',
                                            text=f'Test Folders ({len(self.test_paths)})',
                                            tags=('folder',))
        for p in self.test_paths:
            self.resolved_tree.insert(tstf_id, 'end', text=p, tags=('folder',))

        mskf_id = self.resolved_tree.insert(fld_id, 'end',
                                            text=f'Mask Folders ({len(self.mask_paths)})',
                                            tags=('folder',))
        for p in self.mask_paths:
            self.resolved_tree.insert(mskf_id, 'end', text=p, tags=('folder',))

        # 6B) Files
        files_id = self.resolved_tree.insert(root_id, 'end', text='Files', open=True)
        # Images
        all_images = result['images_train'] + result['images_test']
        imgs_id = self.resolved_tree.insert(
            files_id, 'end',
            text=f'Images ({len(all_images)})',
            tags=('images',)
        )
        for rec in all_images:
            self.resolved_tree.insert(imgs_id, 'end', text=rec['path'], tags=('images',))
        # Masks
        masks = result['mask_files']
        msks_id = self.resolved_tree.insert(
            files_id, 'end',
            text=f'Masks ({len(masks)})',
            tags=('masks',)
        )
        for rec in masks:
            self.resolved_tree.insert(msks_id, 'end', text=rec['path'], tags=('masks',))

        # 7) Populate the training and testing group trees
        train_count = test_count = 0
        for group in result['groups']:
            for split, tree in (('train', self.train_tree), ('test', self.test_tree)):
                for subgroup in group[split]:
                    if split == 'train':
                        train_count += 1
                    else:
                        test_count += 1

                    node = tree.insert(
                        '', 'end',
                        text=subgroup['identifier'],
                        tags=(split, 'error') if not subgroup['masks'] else (split,)
                    )
                    # Images
                    img_node = tree.insert(
                        node, 'end',
                        text=f"Images ({len(subgroup['images'])})",
                        tags=('images',)
                    )
                    for r in subgroup['images']:
                        tree.insert(img_node, 'end', text=r['path'], tags=('images',))
                    # Masks
                    m_node = tree.insert(
                        node, 'end',
                        text=f"Masks ({len(subgroup['masks'])})",
                        tags=('masks',)
                    )
                    for r in subgroup['masks']:
                        tree.insert(m_node, 'end', text=r['path'], tags=('masks',))

        # 8) Update headers and enable the "Save JSON" button
        self.train_tree.master.config(text=f"Training Groups ({train_count})")
        self.test_tree.master.config(text=f"Testing Groups ({test_count})")
        self.save_json_btn.state(['!disabled'])

    def _copy_config_value(self):
        """
        Copy the selected config-tree value to the clipboard.

        Args:
            None.

        Returns:
            None.
        """
        sel = self.config_tree.selection()
        if not sel:
            return
        raw = self.config_tree.item(sel[0], "values")[0]
        # If it's repr-quoted, strip the quotes
        if (raw.startswith(("'", '"')) and raw.endswith(("'", '"'))):
            val = raw[1:-1]
        else:
            val = raw
        self.root.clipboard_clear()
        self.root.clipboard_append(val)
        messagebox.showinfo("Copied", f"Copied to clipboard:\n{val}")

