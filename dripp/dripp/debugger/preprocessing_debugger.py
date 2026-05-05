"""Preprocessing and group-viewer callbacks for the DRIPP debugger app."""

import dripp.config as config

from .common import *


class PreprocessingDebuggerMixin:
    """
    Provide callbacks for preprocessing and group viewing.

    Args:
        None.

    Returns:
        None.
    """

    def preprocess_dataset(self):
        """
        Preprocess sampled groups for the selected dataset.

        Args:
            None.

        Returns:
            None.
        """
        # 1) Get the selected dataset.
        selected = self.preproc_dataset_var.get()
        if not selected:
            messagebox.showwarning("No Dataset", "Please select a dataset first.")
            return

        # 2) Load manager & metadata
        manager = DatasetManager(self.csv_path)
        meta = manager.metadata[selected]

        self.mask_classes = parse_mask_classes(meta.get("mask_classes") or "")

        # 3) Build subdataset_configs exactly like SegmentationDataset.process()
        grouping_regex = meta.get("grouping_regex")
        subdataset_configs = (get_regex_configs(grouping_regex, meta)
                              if grouping_regex else None)

        # 4) Load the groups JSON
        groups_file = os.path.join(INDEX_DIR, "Groups", f"{selected}_groups.json")
        try:
            with open(groups_file, "r") as f:
                groups_data = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Unable to load groups file:\n{e}")
            return
        subdatasets = groups_data.get("subdatasets", [])

        # 5) Flatten & (optionally) sample
        flat_groups = []
        for sub in subdatasets:
            sub_name = sub.get("name", "default")
            sub_modality = sub.get("modality",
                                   (meta.get("modalities") or ["default"])[0])
            for split in ("train", "test"):
                for grp in sub.get(split, []):
                    grp["subdataset_name"] = sub_name
                    grp["subdataset_modality"] = sub_modality
                    grp["axis"] = grp.get("axis")
                    flat_groups.append(grp)

        # 6) Sample up to max_groups per (subdataset_name, split)
        max_g = self.max_groups_var.get()
        if max_g > 0:
            from collections import defaultdict
            import random

            groups_by_key = defaultdict(list)
            for g in flat_groups:
                key = (g["subdataset_name"], g["split"])
                groups_by_key[key].append(g)

            sampled = []
            for grp_list in groups_by_key.values():
                if len(grp_list) > max_g:
                    sampled.extend(random.sample(grp_list, max_g))
                else:
                    sampled.extend(grp_list)
            flat_groups = sampled

        # 7) Build identifiers dict for processing
        identifiers = {
            (g["identifier"], g["split"], g.get("subdataset_name")): g
            for g in flat_groups
        }

        # 8) Hook up both the Preprocessor's main_logger and this dataset_logger
        # First, clear the widget
        self.dataset_log.configure(state='normal')
        self.dataset_log.delete('1.0', tk.END)
        self.dataset_log.configure(state='disabled')

        # Dataset-specific logger
        dataset_logger = logging.getLogger(f"RegexTester.{selected}")
        dataset_logger.setLevel(logging.INFO)
        ds_handler = TextHandler(self.dataset_log)
        ds_handler.setFormatter(Formatter("%(asctime)s %(levelname)s: %(message)s"))
        if not any(isinstance(h, TextHandler) for h in dataset_logger.handlers):
            dataset_logger.addHandler(ds_handler)

        # Preprocessor main logger
        pre_logger = logging.getLogger("Preprocessor")
        pre_logger.setLevel(logging.INFO)
        pre_handler = TextHandler(self.dataset_log)
        pre_handler.setFormatter(Formatter("%(asctime)s %(levelname)s: %(message)s"))
        if not any(isinstance(h, TextHandler) for h in pre_logger.handlers):
            pre_logger.addHandler(pre_handler)

        dataset_logger.info(f"Processing dataset '{selected}'...")
        pre_logger.info(f"\nProcessing dataset '{selected}'...")

        # 8.5) Mirror SegmentationDataset: grab its output_dir & preprocessor
        # Find the actual dataset object to re-use its output_dir
        dataset_obj = next(ds for ds in DatasetManager(self.csv_path).datasets
                           if ds.dataset_name == selected)
        self.output_dir = dataset_obj.output_dir

        # CT-aware preprocessor: load per-dataset CT stats if modality includes CT
        if any(m.lower() == "ct" for m in meta.get("modalities", [])):
            stats_path = os.path.join(INDEX_DIR, "CTStats", f"{selected}_ct_stats.json")
            try:
                with open(stats_path, "r") as sf:
                    ct_stats = json.load(sf)
                global_stats = (ct_stats["mean"], ct_stats["std"])
                dataset_logger.info(f"Loaded CT stats: mean={global_stats[0]:.4f}, std={global_stats[1]:.4f}")
            except Exception as e:
                dataset_logger.warning(f"Could not load CT stats from {stats_path}: {e}")
                global_stats = None

            self.preprocessor = Preprocessor(
                target_size=(512,512),
                dataset_logger=dataset_logger,
                global_ct_stats=global_stats,
                dataset_name=selected,
                background_value=meta.get("background_value", 0)
            )
        else:
            self.preprocessor = Preprocessor(
                target_size=(512,512),
                dataset_logger=dataset_logger,
                dataset_name=selected,
                background_value=meta.get("background_value", 0)
            )

        # Save dataset-specific logs
        log_dir = os.path.join(self.output_dir, ".preprocessing_logs")
        os.makedirs(log_dir, exist_ok=True)
        file_log_path = os.path.join(log_dir, f"{selected}.log")
        if not any(isinstance(h, logging.FileHandler) for h in dataset_logger.handlers):
            fh = logging.FileHandler(file_log_path)
            fh.setFormatter(Formatter("%(asctime)s %(levelname)s: %(message)s"))
            dataset_logger.addHandler(fh)

        # 9) Call the inlined copy logic
        grouping_metadata = self._process_grouping_and_copy(
            identifiers,
            subdataset_configs,
            dataset_logger
        )
        # Remember for lookup on selection
        self.grouping_metadata_list = grouping_metadata

        # Calculate how many groups were dropped (returned None)
        total_groups = len(identifiers)
        processed_groups = len(grouping_metadata)
        dropped_groups = total_groups - processed_groups

        # -- Persist groupings.json so Load Preprocessed can find it --
        grouping_meta_path = os.path.join(self.output_dir, "groupings.json")
        try:
            with open(grouping_meta_path, "w") as f:
                json.dump(self.grouping_metadata_list, f, indent=2)
        except Exception as e:
            messagebox.showwarning("Warning", f"Could not write groupings.json:\n{e}")
        # Now that it exists, enable the Load Preprocessed button
        self.load_pre_btn.state(['!disabled'])
        # -----------------------------------------------------------

        # 10) Populate the treeview ...
        self.group_tree.delete(*self.group_tree.get_children())

        for entry in grouping_metadata:
            grp_iid = self.group_tree.insert(
                "", "end",
                text=entry["identifier"],
                open=False
            )

            # Pull out the processed image paths from preprocessing_metadata
            proc_imgs = entry.get("proc_images", [])
            first_ext = get_extension(proc_imgs[0]).lower() if proc_imgs else ""

            # If first_ext is a volumetric extension, treat as 3D; otherwise 2D
            if first_ext in config.VOLUME_OUTPUT_EXTS:
                # Use the single volume under proc_images[0]
                img_nif = proc_imgs[0]
                raw_masks = entry.get("proc_masks", [])
                proc_msks = [
                    m["path"] if isinstance(m, dict) else m
                    for m in raw_masks
                ]

                vol_iid = self.group_tree.insert(
                    grp_iid, "end",
                    text="Volume",
                    tags=("volume",)
                )
                # Show the volume-image
                self.group_tree.insert(
                    vol_iid, "end",
                    text=img_nif,
                    tags=("images",)
                )

                # List each volume mask
                masks_parent = self.group_tree.insert(
                    vol_iid, "end",
                    text=f"Masks ({len(proc_msks)})",
                    tags=("masks",)
                )
                for mfp in proc_msks:
                    self.group_tree.insert(
                        masks_parent, "end",
                        text=mfp,
                        tags=("masks",)
                    )
            else:
                # 2D image/mask nodes as before
                imgs = entry.get("proc_images", entry.get("images", []))
                img_iid = self.group_tree.insert(
                    grp_iid, "end",
                    text=f"Images ({len(imgs)})",
                    tags=("images",)
                )
                for it in imgs:
                    path = it["path"] if isinstance(it, dict) else it
                    self.group_tree.insert(
                        img_iid, "end",
                        text=path,
                        tags=("images",)
                    )

                raw_mks = entry.get("proc_masks", entry.get("masks", []))
                mks = [
                    m["path"] if isinstance(m, dict) else m
                    for m in raw_mks
                ]
                msk_iid = self.group_tree.insert(
                    grp_iid, "end",
                    text=f"Masks ({len(mks)})",
                    tags=("masks",)
                )
                for it in mks:
                    path = it["path"] if isinstance(it, dict) else it
                    self.group_tree.insert(
                        msk_iid, "end",
                        text=path,
                        tags=("masks",)
                    )

        # Update the Load Preprocessed button
        self.update_load_preproc_button()

        # 11) Prepare navigation
        # Flatten image/mask paths in the order returned
        self.current_images = [
            (img if isinstance(img, str) else img["path"],
             msk if isinstance(msk, str) else msk["path"])
            for e in grouping_metadata
            for img, msk in zip(e["images"], e["masks"])
        ]
        self.current_index = 0
        self.update_nav_buttons()

        # Show a popup reporting the total number of groups processed and dropped, unless it's a batch
        if not self.is_batch:
            messagebox.showinfo(
                "Preprocessing Complete",
                f"Preprocessed {processed_groups} group{'s' if processed_groups != 1 else ''} "
                f"for dataset '{selected}'.\nDropped {dropped_groups} group{'s' if dropped_groups != 1 else ''}."
            )

    def on_preproc_dataset_select(self, event):
        """
        Reset preprocessing UI state after dataset selection.

        Args:
            event (Any): Tk event object.

        Returns:
            None.
        """
        # Clear the group selector
        self.group_tree.delete(*self.group_tree.get_children())
        # Reset the image list + index
        self.current_images.clear()
        self.current_index = 0
        # Update Prev/Next buttons & index label
        self.update_nav_buttons()
        # Clear the log
        self.dataset_log.delete('1.0', tk.END)
        # Now toggle Load Preprocessed
        self.update_load_preproc_button()

    def update_load_preproc_button(self):
        """
        Enable loading when preprocessed output exists.

        Args:
            None.

        Returns:
            None.
        """
        selected = self.preproc_dataset_var.get()
        if not selected:
            self.load_pre_btn.state(['disabled'])
            return

        # Find the output_dir from DatasetManager
        manager = DatasetManager(self.csv_path)
        ds = next((d for d in manager.datasets if d.dataset_name == selected), None)
        if not ds:
            self.load_pre_btn.state(['disabled'])
            return

        grp_path = os.path.join(ds.output_dir, "groupings.json")
        if os.path.exists(grp_path):
            self.load_pre_btn.state(['!disabled'])
        else:
            self.load_pre_btn.state(['disabled'])

    def load_preprocessed(self):
        """
        Load preprocessed group metadata into the viewer.

        Args:
            None.

        Returns:
            None.
        """
        # 1) Identify selected dataset and locate its groupings.json
        selected = self.preproc_dataset_var.get()
        manager   = DatasetManager(self.csv_path)
        ds       = next((d for d in manager.datasets if d.dataset_name == selected), None)
        if not ds:
            return

        grp_path = os.path.join(ds.output_dir, "groupings.json")
        # 2) Read the JSON file
        try:
            with open(grp_path, "r") as f:
                grouping_metadata = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read {grp_path}:\n{e}")
            return

        # 3) Store for later and clear the treeview
        self.grouping_metadata_list = grouping_metadata
        self.group_tree.delete(*self.group_tree.get_children())

        # 4) Populate each group node ...
        self.group_tree.delete(*self.group_tree.get_children())

        for entry in grouping_metadata:
            grp_iid = self.group_tree.insert(
                "", "end",
                text=entry["identifier"],
                open=False
            )

            proc_imgs = entry.get("proc_images", [])
            first_ext = get_extension(proc_imgs[0]).lower() if proc_imgs else ""

            if first_ext in config.VOLUME_OUTPUT_EXTS:
                # 3D-volume mode
                img_iid = self.group_tree.insert(
                    grp_iid, "end",
                    text="Images (1)",
                    tags=("images",)
                )
                self.group_tree.insert(
                    img_iid, "end",
                    text=proc_imgs[0],
                    tags=("images",)
                )

                raw_masks = entry.get("proc_masks", [])
                proc_msks = [
                    m["path"] if isinstance(m, dict) else m
                    for m in raw_masks
                ]
                msk_iid = self.group_tree.insert(
                    grp_iid, "end",
                    text=f"Masks ({len(proc_msks)})",
                    tags=("masks",)
                )
                for mpath in proc_msks:
                    self.group_tree.insert(
                        msk_iid, "end",
                        text=mpath,
                        tags=("masks",)
                    )
            else:
                # 2D image fallback
                imgs = entry.get("proc_images", entry.get("images", []))
                img_iid = self.group_tree.insert(
                    grp_iid, "end",
                    text=f"Images ({len(imgs)})",
                    tags=("images",)
                )
                for p in imgs:
                    path = p["path"] if isinstance(p, dict) else p
                    self.group_tree.insert(
                        img_iid, "end",
                        text=path,
                        tags=("images",)
                    )

                raw_msks = entry.get("proc_masks", entry.get("masks", []))
                msks = [
                    m["path"] if isinstance(m, dict) else m
                    for m in raw_msks
                ]
                msk_iid = self.group_tree.insert(
                    grp_iid, "end",
                    text=f"Masks ({len(msks)})",
                    tags=("masks",)
                )
                for p in msks:
                    path = p["path"] if isinstance(p, dict) else p
                    self.group_tree.insert(
                        msk_iid, "end",
                        text=path,
                        tags=("masks",)
                    )

        # 5) Re-enable the Load Preprocessed button
        self.update_load_preproc_button()

    def view_selected_group(self, event):
        """
        Load the selected group into the image or NIfTI viewer.

        Args:
            event (Any): Tk event object.

        Returns:
            None.
        """
        # Start loading indicator
        self.show_loading()
        self.root.update_idletasks()

        try:
            # 1) Determine which tree item was clicked
            sel = self.group_tree.selection()
            if not sel:
                return
            node = sel[0]

            # 2) Climb to find the root-group node
            root = node
            while self.group_tree.parent(root):
                root = self.group_tree.parent(root)
            identifier = self.group_tree.item(root, "text")

            # If it's the same group as we already have loaded, do nothing
            if identifier == self.current_group:
                self.hide_loading()
                return
            # Otherwise remember this group and proceed
            self.current_group = identifier

            # 3) Look up that group's metadata
            entry = next((e for e in self.grouping_metadata_list
                          if e["identifier"] == identifier), None)
            if not entry:
                return

            # 4) If this group has full-volume outputs, switch to 4-view mode
            img_nii    = entry.get("proc_images")
            raw_masks   = entry.get("proc_masks", [])
            msk_niftis  = [
                m["path"] if isinstance(m, dict) else m
                for m in raw_masks
            ]
            if (img_nii and msk_niftis
                and get_extension(img_nii[0]).lower() in config.VOLUME_OUTPUT_EXTS
                and get_extension(msk_niftis[0]).lower() in config.VOLUME_OUTPUT_EXTS):
                self.viewer_canvas.pack_forget()
                self.nifti_frame.pack(fill="both", expand=True)
                # Pass the entire list of mask files to _init_nifti_volumes(...)
                self._init_nifti_volumes(img_nii[0], msk_niftis)
                self.nav_frame.pack_forget()
                self.index_label.pack_forget()
                self.img_file_label.pack_forget()
                self.mask_file_label.pack_forget()
                return

            # 5) Otherwise, fall back to the 2D image viewer
            self.nifti_frame.pack_forget()
            self.nifti_msk_list = []
            self.nifti_img = None
            self.viewer_canvas.pack(fill="both", expand=True)
            self.nav_frame.pack(side="right", padx=5, pady=5)
            self.index_label.pack(pady=(2,0))
            self.img_file_label.pack(anchor="e")
            self.mask_file_label.pack(anchor="e")

            # 5A) Reset and load the image/mask lists
            self.current_images = entry.get("proc_images", entry.get("images", []))
            msks = entry.get("proc_masks", entry.get("masks", []))
            self.current_masks = [
                m["path"] if isinstance(m, dict) else m
                for m in msks
            ]
            self.current_index = self.current_mask_index = 0

            # 5B) Render the first pair and update nav
            self.render_current_pair()
            self.update_nav_buttons()

        finally:
            # Always hide loading indicator
            self.hide_loading()

    def show_prev_image(self):
        """
        Navigate to the previous processed image.

        Args:
            None.

        Returns:
            None.
        """
        if self.current_index > 0:
            self.current_index -= 1
            self.render_current_pair()
            self.update_nav_buttons()

    def show_next_image(self):
        """
        Navigate to the next processed image.

        Args:
            None.

        Returns:
            None.
        """
        if self.current_index < len(self.current_images)-1:
            self.current_index += 1
            self.render_current_pair()
            self.update_nav_buttons()

    def show_prev_mask(self):
        """
        Navigate to the previous mask for the current image.

        Args:
            None.

        Returns:
            None.
        """
        if getattr(self, "nifti_msk_list", []):
            # 3D mode: cycle through mask volumes
            if self.current_mask_index > 0:
                self.current_mask_index -= 1
                self.update_nifti_display(self.current_axis, getattr(self, f"{self.current_axis.lower()}_slider").get())
                self.msk_prev_btn.config(state="normal" if self.current_mask_index > 0 else "disabled")
                self.msk_next_btn.config(
                    state="normal" if self.current_mask_index < len(self.nifti_msk_list) - 1 else "disabled"
                )
        else:
            # 2D image fallback
            if self.current_mask_index > 0:
                self.current_mask_index -= 1
                self.render_current_pair()
                self.update_nav_buttons()

    def show_next_mask(self):
        """
        Navigate to the next mask for the current image.

        Args:
            None.

        Returns:
            None.
        """
        if getattr(self, "nifti_msk_list", []):
            # 3D mode: cycle through mask volumes
            if self.current_mask_index < len(self.nifti_msk_list) - 1:
                self.current_mask_index += 1
                # Update the displayed slice for the current axis
                slider = getattr(self, f"{self.current_axis.lower()}_slider")
                self.update_nifti_display(self.current_axis, slider.get())
                # Enable or disable mask navigation buttons
                self.msk_prev_btn.config(
                    state="normal" if self.current_mask_index > 0 else "disabled"
                )
                self.msk_next_btn.config(
                    state="normal" if self.current_mask_index < len(self.nifti_msk_list) - 1 else "disabled"
                )
        else:
            # 2D image fallback: advance through 2D masks
            if self.current_mask_index < len(self.current_masks) - 1:
                self.current_mask_index += 1
                self.render_current_pair()
                self.update_nav_buttons()

    def update_nav_buttons(self):
        """
        Update viewer navigation controls.

        Args:
            None.

        Returns:
            None.
        """
        total_imgs = len(self.current_images)
        total_msks = len(self.current_masks)

        # Update label
        img_text = f"{self.current_index+1}/{total_imgs}" if total_imgs else "-/-"
        msk_text = f"{self.current_mask_index+1}/{total_msks}" if total_msks else "-/-"
        self.index_label.config(text=f"Img {img_text} | Msk {msk_text}")

        # Image buttons
        self.img_prev_btn.config(state="normal" if self.current_index > 0 else "disabled")
        self.img_next_btn.config(state="normal" if self.current_index < total_imgs-1 else "disabled")

        # Mask buttons
        self.msk_prev_btn.config(state="normal" if self.current_mask_index > 0 else "disabled")
        self.msk_next_btn.config(state="normal" if self.current_mask_index < total_msks-1 else "disabled")

    def render_current_pair(self):
        """
        Render the current 2D image and mask overlay.

        Args:
            None.

        Returns:
            None.
        """
        if not self.current_images:
            return

        # Load image & update filename label
        img_path = self.current_images[self.current_index]
        self.img_file_label.config(text=f"Image: {os.path.basename(img_path)}")
        try:
            img = Image.open(img_path).convert('RGBA')
        except Exception as e:
            messagebox.showerror("Error Loading Image", str(e))
            return

        # Load mask (if any)
        msk = None
        if self.current_masks:
            msk_path = self.current_masks[
                min(self.current_mask_index, len(self.current_masks)-1)
            ]
            # Update mask filename label
            self.mask_file_label.config(text=f"Mask:  {os.path.basename(msk_path)}")

            try:
                msk = Image.open(msk_path).convert('RGBA')
                # Make mask semi-transparent
                a = msk.split()[3]
                msk.putalpha(a.point(lambda p: p//2))
            except Exception as e:
                messagebox.showerror("Error Loading Mask", str(e))
                msk = None

        # Composite
        if msk:
            if msk.size != img.size:
                msk = msk.resize(img.size)
            comp = Image.alpha_composite(img, msk)
        else:
            comp = img

        # Scale to fit both canvas width and height
        self.viewer_canvas.delete('all')
        self.viewer_canvas.update_idletasks()
        cw = self.viewer_canvas.winfo_width() or self.viewer_canvas.winfo_reqwidth()
        ch = self.viewer_canvas.winfo_height() or self.viewer_canvas.winfo_reqheight()

        w, h = comp.size
        # Scale factor so image fits within canvas
        scale = min(cw / w, ch / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        # --- Store mask overlay for hover testing ---
        if msk:
            mask_resized = msk.resize((new_w, new_h), Image.LANCZOS)
            self._last_2d_mask_img = mask_resized
            entry = self.current_masks[self.current_mask_index]
            self._last_2d_mask_meta = {'path': entry, 'class': extract_mask_class(entry)}
        else:
            self._last_2d_mask_img = None
            self._last_2d_mask_meta = None

        comp_resized = comp.resize((new_w, new_h), Image.LANCZOS)

        # Draw centered in the canvas
        self.tk_img = ImageTk.PhotoImage(comp_resized)
        self.viewer_canvas.create_image(
            cw // 2, ch // 2,
            anchor='center',
            image=self.tk_img
        )
        # Ensure the full image is visible
        self.viewer_canvas.config(scrollregion=(0, 0, cw, ch))

    def _process_grouping_and_copy(self, identifiers, subdataset_configs, logger):
        """
        Preprocess groups and build output group metadata.

        Args:
            identifiers (Any): Grouped dataset identifiers to preprocess.
            subdataset_configs (Any): Parsed subdataset configuration objects.
            logger (Any): Logger used for preprocessing progress.

        Returns:
            list[dict]: Preprocessed group metadata entries.
        """
        grouping_metadata = []
        for key, entry in identifiers.items():
            if not entry["images"]:
                logger.warning(f"Group {key} has no images. Skipping...")
                continue

            # Extract file paths.
            image_paths = [item["path"] if isinstance(item, dict) else item
                           for item in entry["images"]]
            mask_paths = [item["path"] if isinstance(item, dict) else item
                          for item in entry["masks"]]
            image_paths.sort()
            mask_paths.sort()

            composite_id = get_composite_identifier(entry)

            # Build output dirs
            sub_name = entry.get("subdataset_name") or entry.get("name")
            sub_modality = (entry.get("subdataset_modality")
                            or (self.metadata.get("modalities") or ["default"])[0])
            sub_pipeline = entry.get("subdataset_pipeline")

            # Only use subdataset name if it exists
            if sub_name:
                out_dir = os.path.join(self.output_dir,
                                       sub_modality,
                                       sub_name,
                                       entry["split"])
            else:
                out_dir = os.path.join(self.output_dir,
                                       sub_modality,
                                       entry["split"])

            # If composite_id starts with the split name, drop that prefix so
            # We don't end up with "train/train/..."
            id_subfolders = composite_id.split("_")
            if id_subfolders and id_subfolders[0] == entry["split"]:
                id_subfolders = id_subfolders[1:]

            out_dir = normalize_path(os.path.join(out_dir, *id_subfolders))

            img_out_dir = os.path.join(out_dir, "images")
            mask_out_dir = os.path.join(out_dir, "masks")
            os.makedirs(img_out_dir, exist_ok=True)
            os.makedirs(mask_out_dir, exist_ok=True)

            try:
                preprocessing_metadata = self.preprocessor.preprocess_group(
                    sub_name,
                    sub_pipeline,
                    image_paths,
                    mask_paths,
                    sub_modality,
                    img_out_dir,
                    mask_out_dir,
                    composite_id,
                    self.mask_classes
                )
            except Exception as e:
                logger.error(f"Error preprocessing group {key}: {e}", exc_info=True)
                continue

            # Gather all processed files from img_out_dir and mask_out_dir
            proc_imgs = sorted(
                os.path.join(img_out_dir, fn)
                for fn in os.listdir(img_out_dir)
                if fn.lower().endswith(tuple(config.OUTPUT_EXTS))
            )
            mask_files = sorted(
                os.path.join(mask_out_dir, fn)
                for fn in os.listdir(mask_out_dir)
                if fn.lower().endswith(tuple(config.OUTPUT_EXTS))
            )
            proc_masks = [
                {"path": p, "class": extract_mask_class(p)}
                for p in mask_files
            ]

            # Drop this group if no masks were produced
            if not proc_masks:
                logger.warning(f"Group {key} produced no processed masks. Skipping group.")
                continue

            # If preprocess_group returned multiple modality volumes, split into separate entries
            if (isinstance(preprocessing_metadata, dict)
                    and "image_niftis" in preprocessing_metadata
                    and len(preprocessing_metadata["image_niftis"]) > 1):
                for idx, img_nif in enumerate(preprocessing_metadata["image_niftis"]):
                    new_id = f"{composite_id}_modality{idx}"
                    mask_niftis = preprocessing_metadata.get("mask_niftis", [])
                    proc_masks = [{"path": p, "class": extract_mask_class(p)} for p in mask_niftis]
                    split_entry = {
                        "identifier":       new_id,
                        "short_id":         self.preprocessor._short_id(new_id),
                        "images":           entry["images"],
                        "masks":            entry["masks"],  # Original raw-mask paths
                        "proc_images":      [img_nif],          # Put the volume path here
                        "proc_masks":       proc_masks,
                        "preprocessing_metadata": {
                            "resize_shape": preprocessing_metadata.get("resize_shape"),
                            "volume_shape": preprocessing_metadata.get("volume_shape")
                        }
                    }
                    grouping_metadata.append(split_entry)
            elif self.preproc_dataset_var.get() == "QUBIQ2021" and sub_name == "brain-tumor":
                for idx, img in enumerate(proc_imgs):
                    new_id = f"{composite_id}_modality{idx}"
                    entry_dict = {
                        "identifier":       new_id,
                        "short_id":         self.preprocessor._short_id(new_id),
                        "images":           entry["images"],
                        "masks":            entry["masks"],  # Original raw-mask paths
                        "proc_images":      [img],       # Processed image
                        "proc_masks":       proc_masks,
                        "preprocessing_metadata": {}
                    }
                    grouping_metadata.append(entry_dict)
            else:
                entry_dict = {
                    "identifier":             composite_id,
                    "short_id":               self.preprocessor._short_id(composite_id),
                    "images":                 entry["images"],
                    "masks":                  entry["masks"],
                    "proc_images":            proc_imgs,
                    "proc_masks":             proc_masks,
                    "preprocessing_metadata": {}
                }

                # Always include resize_shape if present
                if isinstance(preprocessing_metadata, dict) and "resize_shape" in preprocessing_metadata:
                    entry_dict["preprocessing_metadata"]["resize_shape"] = preprocessing_metadata["resize_shape"]

                # If video metadata is present, include fps & num_frames
                if isinstance(preprocessing_metadata, dict) and "fps" in preprocessing_metadata and "num_frames" in preprocessing_metadata:
                    entry_dict["preprocessing_metadata"]["fps"] = preprocessing_metadata["fps"]
                    entry_dict["preprocessing_metadata"]["num_frames"] = preprocessing_metadata["num_frames"]

                # If 3D metadata is present, include volume_shape and volume paths
                if isinstance(preprocessing_metadata, dict) and "volume_shape" in preprocessing_metadata:
                    entry_dict["preprocessing_metadata"]["volume_shape"] = preprocessing_metadata["volume_shape"]

                grouping_metadata.append(entry_dict)

        return grouping_metadata

    def preprocess_all(self):
        """
        Preprocess every dataset listed in the combo box.

        Args:
            None.

        Returns:
            None.
        """
        # Grab the full list of dataset names
        all_names = list(self.preproc_ds_combo['values'])

        # Set the batch flag
        self.is_batch = True
        for name in all_names:
            # Skip PAIP2019. Temporary: Takes too long.
            if name == "PAIP2019":
                continue

            # Select it in the UI
            self.preproc_dataset_var.set(name)
            # Clear UI and enable Load Preprocessed if needed
            self.on_preproc_dataset_select(None)
            # Run the single-dataset logic
            try:
                self.preprocess_dataset()
            except Exception as e:
                messagebox.showerror(
                    f"Error processing {name}",
                    f"An error occurred while preprocessing '{name}':\n{e}"
                )
                # Continue with the next dataset
        # Final refresh of the Load Preprocessed button for whichever is selected now
        self.is_batch = False
        self.update_load_preproc_button()
        messagebox.showinfo("Done", "All datasets have been preprocessed.")

    def _init_nifti_volumes(self, img_path, msk_paths):
        """
        Load NIfTI image and mask volumes for triplanar viewing.

        Args:
            img_path (Any): Path to the image volume.
            msk_paths (Any): Mask volume paths.

        Returns:
            None.
        """
        # Load the NIfTI image and its masks
        img_itk = sitk.ReadImage(img_path)
        img_itk = sitk.DICOMOrient(img_itk, 'RAS')
        self.spacing_x, self.spacing_y, self.spacing_z = img_itk.GetSpacing()
        self.nifti_img = sitk.GetArrayFromImage(img_itk)  # Shape (Z, Y, X)
        # Store the 3x3 direction cosines (tuple of 9 floats)
        self.direction = img_itk.GetDirection()

        # Load each mask file into a separate 3D SimpleITK image and NumPy array
        self.nifti_msk_list   = []
        self.nifti_mask_paths = msk_paths
        all_labels = set()
        for mfp in msk_paths:
            mask_itk = sitk.ReadImage(mfp)
            mask_itk = sitk.DICOMOrient(mask_itk, 'RAS')
            mask_arr = sitk.GetArrayFromImage(mask_itk).astype(np.int32)
            self.nifti_msk_list.append(mask_arr)
            unique = np.unique(mask_arr)
            for l in unique:
                if l != 0:
                    all_labels.add(int(l))

        # Assign a distinct color to each label across all mask volumes
        palette = [
            (255,   0,   0),  # Red
            (  0, 255,   0),  # Green
            (  0,   0, 255),  # Blue
            (255, 255,   0),  # Yellow
            (255,   0, 255),  # Magenta
            (  0, 255, 255),  # Cyan
            (128,   0, 128),  # Purple
            (128, 128,   0),  # Olive
            (  0, 128, 128)   # Teal
        ]
        sorted_labels = sorted(all_labels)
        self.label_colors = {lbl: palette[i % len(palette)] for i, lbl in enumerate(sorted_labels)}

        # Configure sliders based on volume dimensions
        Z, Y, X = self.nifti_img.shape
        self.axial_slider.config(to=Z - 1)
        self.coronal_slider.config(to=Y - 1)
        self.sagittal_slider.config(to=X - 1)

        # Start each axis at the center slice
        self.axial_slider.set(Z // 2)
        self.coronal_slider.set(Y // 2)
        self.sagittal_slider.set(X // 2)

        # Default to axial view
        self.current_axis = 'Axial'
        self.current_mask_index = 0  # No single-mask navigation in combined mode

        # Initialize display for each view
        self.update_nifti_display('Axial',   Z // 2)
        self.update_nifti_display('Coronal', Y // 2)
        self.update_nifti_display('Sagittal',X // 2)

        # Disable mask-prev/next buttons since all masks are shown simultaneously
        self.msk_prev_btn.config(state="disabled")
        self.msk_next_btn.config(state="disabled")

        # Keep image navigation disabled in 3D mode
        self.img_prev_btn.config(state="disabled")
        self.img_next_btn.config(state="disabled")


        # Add a "Show Masks" checkbox to toggle overlays
        if not hasattr(self, "_mask_toggle_cb"):
            cb = ttk.Checkbutton(
                self.nifti_frame,
                text="Show Masks",
                variable=self.show_masks,
                command=lambda: self.update_nifti_display(
                    self.current_axis,
                    getattr(self, f"{self.current_axis.lower()}_slider").get()
                )
            )
            # Place it in the UI, e.g. above sliders
            cb.grid(row=2, column=0, columnspan=2, pady=(5,0))
            self._mask_toggle_cb = cb

    def update_nifti_display(self, axis, idx):
        """
        Render one NIfTI slice view with mask overlays.

        Args:
            axis (Any): View axis name.
            idx (Any): Slice index.

        Returns:
            None.
        """
        # Determine the image slice for the given axis
        if axis == 'Axial':
            slice_img = self.nifti_img[idx, :, :]
        elif axis == 'Coronal':
            slice_img = self.nifti_img[:, idx, :]
        else:  # 'Sagittal'
            slice_img = self.nifti_img[:, :, idx]

        # Normalize the image slice to 0-255
        mn, mx = float(slice_img.min()), float(slice_img.max())
        if mx > mn:
            norm = ((slice_img - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(slice_img, dtype=np.uint8)
        rgb = np.stack([norm] * 3, axis=-1)

        # Overlay every mask volume onto the RGB image
        # Conditionally overlay masks
        if self.show_masks.get():
            alpha = 0.5
            for mask_vol in self.nifti_msk_list:
                if axis == 'Axial':
                    slice_msk = mask_vol[idx, :, :]
                elif axis == 'Coronal':
                    slice_msk = mask_vol[:, idx, :]
                else:  # 'Sagittal'
                    slice_msk = mask_vol[:, :, idx]

                # For each label in this slice (excluding background), paint with its assigned color
                unique_labels = np.unique(slice_msk)
                for lbl in unique_labels:
                    if lbl == 0:
                        continue
                    color = self.label_colors.get(int(lbl), (255, 255, 255))
                    mask_idxs = (slice_msk == lbl)
                    for c in range(3):
                        rgb[mask_idxs, c] = (1 - alpha) * rgb[mask_idxs, c] + alpha * color[c]

        # Convert to PIL
        pil = Image.fromarray(rgb.astype(np.uint8))
        pil = pil.transpose(Image.FLIP_TOP_BOTTOM)    # Superior at top
        pil = pil.transpose(Image.FLIP_LEFT_RIGHT)    # A (or R) on left

        # Determine physical pixel spacing for aspect ratio
        if axis == 'Axial':
            # Width=X, height=Y
            px_w, px_h = self.spacing_x, self.spacing_y
        elif axis == 'Coronal':
            # Width=X, height=Z
            px_w, px_h = self.spacing_x, self.spacing_z
        else:  # 'Sagittal'
            # Width=Y, height=Z
            px_w, px_h = self.spacing_y, self.spacing_z

        orig_w, orig_h = pil.size
        phys_w = orig_w * px_w
        phys_h = orig_h * px_h
        self._last_orig_dimensions = (orig_w, orig_h)

        # Resize to fit canvas while preserving aspect ratio
        cv = self.nifti_canvases[axis]
        cw, ch = cv.winfo_width(), cv.winfo_height()
        if cw > 1 and ch > 1:
            scale = min(cw / phys_w, ch / phys_h)
            new_w = max(1, int(phys_w * scale))
            new_h = max(1, int(phys_h * scale))
            pil = pil.resize((new_w, new_h), Image.NEAREST)

            # --- Store slice masks & metadata for hover testing ---
            self._last_slice_masks = []
            self._last_slice_mask_meta = []
            for i, mask_vol in enumerate(self.nifti_msk_list):
                if axis == 'Axial':
                    slice_msk = mask_vol[idx, :, :]
                elif axis == 'Coronal':
                    slice_msk = mask_vol[:, idx, :]
                else:  # 'Sagittal'
                    slice_msk = mask_vol[:, :, idx]
                self._last_slice_masks.append(slice_msk)

                # Use the same index into the list of paths you saved
                mask_path = self.nifti_mask_paths[i]
                self._last_slice_mask_meta.append({
                    'path': mask_path,
                    'class': extract_mask_class(mask_path)
                })

            # Stash for coordinate-mapping
            self._last_resized_dimensions = (new_w, new_h)
            self._last_scale = scale

        tkimg = ImageTk.PhotoImage(pil)
        cv.delete('all')
        cv.create_image(cw // 2, ch // 2, anchor='center', image=tkimg)
        cv.image = tkimg

    def _on_slider_move(self, axis, idx):
        """
        Handle triplanar slice slider movement.

        Args:
            axis (Any): View axis name.
            idx (Any): Slice index.

        Returns:
            None.
        """
        # 1) Update the little value label
        lbl = getattr(self, f"{axis.lower()}_val_label")
        lbl.config(text=str(idx))

        # 2) Redraw the image slice
        self.update_nifti_display(axis, idx)

    def _on_2d_motion(self, event):
        """
        Show mask-class tooltip for 2D overlay hover.

        Args:
            event (Any): Tk event object.

        Returns:
            None.
        """
        if not self._last_2d_mask_img:
            self.tooltip.hide()
            return

        # Map canvas coords -> mask image coords
        cw = self.viewer_canvas.winfo_width()
        ch = self.viewer_canvas.winfo_height()
        mw, mh = self._last_2d_mask_img.size
        # Image is centered
        x0 = (cw - mw)//2
        y0 = (ch - mh)//2
        mx, my = event.x - x0, event.y - y0
        if not (0 <= mx < mw and 0 <= my < mh):
            self.tooltip.hide()
            return

        # Check mask alpha
        a = self._last_2d_mask_img.getchannel('A').getpixel((mx, my))
        if a > 0:
            meta = self._last_2d_mask_meta
            fn = os.path.basename(meta['path'])
            cls = meta.get('class') or 'unknown'
            self.tooltip.show(f"{cls}\n{fn}", event.x_root, event.y_root)
        else:
            self.tooltip.hide()

    def _on_3d_motion(self, event):
        """
        Show mask-class tooltip for 3D slice hover.

        Args:
            event (Any): Tk event object.

        Returns:
            None.
        """
        if not self._last_slice_masks:
            self.tooltip.hide()
            return

        cv = event.widget
        cw, ch = cv.winfo_width(), cv.winfo_height()
        new_w, new_h = self._last_resized_dimensions
        x0 = (cw - new_w) // 2
        y0 = (ch - new_h) // 2

        mx, my = event.x - x0, event.y - y0
        # If outside the displayed image, hide
        if not (0 <= mx < new_w and 0 <= my < new_h):
            self.tooltip.hide()
            return

        orig_w, orig_h = self._last_orig_dimensions
        # Map canvas coords -> original image coords
        sx = int(mx * orig_w  / new_w)
        # Because you flipped top/bottom, invert y
        sy = orig_h - 1 - int(my * orig_h / new_h)

        # Guard again
        if not (0 <= sx < orig_w and 0 <= sy < orig_h):
            self.tooltip.hide()
            return

        # Now test each mask slice
        for slice_msk, meta in zip(self._last_slice_masks, self._last_slice_mask_meta):
            # Slice_msk shape is (orig_h, orig_w)
            if slice_msk[sy, sx] != 0:
                fn = os.path.basename(meta['path'])
                cls = meta.get('class') or 'unknown'
                self.tooltip.show(f"{cls}\n{fn}", event.x_root, event.y_root)
                return

        self.tooltip.hide()

