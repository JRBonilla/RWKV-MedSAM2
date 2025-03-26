import os, gzip, io
import numpy as np
from PIL import Image, ImageDraw
import nibabel as nib
import scipy.ndimage as ndimage
import SimpleITK as sitk
import h5py
import json
import xml.etree.ElementTree as ET
import logging

# Set up logging to record any errors
logging.basicConfig(
    filename="processing_errors.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s: %(message)s"
)

class Preprocessor:
    def __init__(self, target_size=(256, 256)):
        """
        Initialize the Preprocessor with the target image size.

        Args:
            target_size (tuple, optional): The target image size for the output. Defaults to (256, 256).
        """
        self.target_size = target_size

        # Define modalities for which grayscale processing (NIfTI output) is expected.
        self.grayscale_modalities = { 'ct', 'mri', 'x-ray', 'ultrasound' }

    def convert_file(self, file_path, modality=None, is_mask=False):
        """
        Convert files to a standard format before further processing.
          - For non-mask files: if file is a 3D dataset (.dcm, .nrrd, .mhd) convert to NIfTI
          - For mask files: if not already PNG, convert to PNG.
          - Otherwise, simply return the original path.

        Args:
            file_path (str): The path to the file to be converted.
            modality (str, optional): The modality of the file. Defaults to None.
            is_mask (bool, optional): Whether the file is a mask file. Defaults to False.

        Returns:
            str: The path to the converted file.
        """
        ext = os.path.splitext(file_path)[1].lower()
        if not is_mask:
            if ext in {'.dcm', '.nrrd', '.mhd'}:
                print(f"Converting 3D image {file_path} to NIfTI format...")
                try:
                    img = sitk.ReadImage(file_path)
                    new_path = os.path.splitext(file_path)[0] + '.nii.gz'
                    sitk.WriteImage(img, new_path)
                    return new_path
                except Exception as e:
                    logging.exception(f"Error converting {file_path} to NIfTI: {e}")
                    return file_path
            else:
                # For already-converted files or 2D images, return original.
                return file_path
        elif ext == '.avi':
            print(f"Extracting frame from video {file_path}...")
            try:
                import cv2
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    new_path = os.path.splitext(file_path)[0] + '.png'
                    cv2.imwrite(new_path, frame)
                    return new_path
                else:
                    print(f"Failed to read frame from {file_path}")
                    return file_path
            except Exception as e:
                logging.exception(f"Error processing video file {file_path}: {e}")
                return file_path
        elif ext == '.npy':
            print(f"Converting .npy image file {file_path} to PNG...")
            try:
                arr = np.load(file_path)
                # Normalize array to [0,255]
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
                new_path = os.path.splitext(file_path)[0] + '.png'
                Image.fromarray(arr.astype(np.uint8)).save(new_path)
                return new_path
            except Exception as e:
                logging.exception(f"Error converting npy file {file_path}: {e}")
                return file_path
        elif ext in {'.ppm', '.ppm.gz'}:
            print(f"Converting {ext} image file {file_path} to PNG...")
            try:
                if ext == '.ppm.gz':
                    with gzip.open(file_path, 'rb') as f:
                        data = f.read()
                    img = Image.open(io.BytesIO(data))
                else:
                    img = Image.open(file_path)
                new_path = os.path.splitext(file_path)[0] + '.png'
                img.save(new_path)
                return new_path
            except Exception as e:
                logging.exception(f"Error converting ppm file {file_path}: {e}")
                return file_path
        else:
            # For masks, ensure PNG format
            if ext == '.mat':
                print(f"Converting .mat mask file {file_path} to PNG...")
                try:
                    from scipy.io import loadmat
                    mat_data = loadmat(file_path)
                    mask = None
                    if 'mask' in mat_data:
                        mask = mat_data['mask']
                    else:
                        # If key 'mask' not present, take the first numeric array.
                        for key in mat_data:
                            if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                                mask = mat_data[key]
                                break
                    if mask is None:
                        print(f"No valid mask found in {file_path}")
                        return file_path
                    mask = (mask > 0).astype(np.uint8) * 255
                    new_path = os.path.splitext(file_path)[0] + '.png'
                    Image.fromarray(mask).save(new_path)
                    return new_path
                except Exception as e:
                    logging.exception(f"Error converting mat mask file {file_path}: {e}")
                    return file_path
            elif ext == '.npy':
                print(f"Converting .npy mask file {file_path} to PNG...")
                try:
                    arr = np.load(file_path)
                    mask = (arr > 0).astype(np.uint8) * 255
                    new_path = os.path.splitext(file_path)[0] + '.png'
                    Image.fromarray(mask).save(new_path)
                    return new_path
                except Exception as e:
                    logging.exception(f"Error converting npy mask file {file_path}: {e}")
                    return file_path
            elif ext in {'.ppm', '.ppm.gz'}:
                print(f"Converting {ext} mask file {file_path} to PNG...")
                try:
                    if ext == '.ppm.gz':
                        with gzip.open(file_path, 'rb') as f:
                            data = f.read()
                        img = Image.open(io.BytesIO(data))
                    else:
                        img = Image.open(file_path)
                    new_path = os.path.splitext(file_path)[0] + '.png'
                    img.save(new_path)
                    return new_path
                except Exception as e:
                    logging.exception(f"Error converting ppm mask file {file_path}: {e}")
                    return file_path
            elif ext != '.png':
                print(f"Converting mask {file_path} to PNG format...")
                try:
                    with Image.open(file_path) as img:
                        new_path = os.path.splitext(file_path)[0] + '.png'
                        img.save(new_path)
                        return new_path
                except Exception as e:
                    logging.exception(f"Error converting {file_path} to PNG: {e}")
                    return file_path

    def load_image(self, file_path):
        """
        Load an image as a NumPy array.
          - Uses nibabel for NIfTI files.
          - Uses PIL for PNG files.
          
        Args:
            file_path (str): The path to the image file.

        Returns:
            np.ndarray: The image data as a NumPy array.
        """
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext in {'.nii', '.nii.gz'}:
                nii = nib.load(file_path)
                return nii.get_fdata()
            else:
                with Image.open(file_path) as img:
                    return np.array(img)
        except Exception as e:
            logging.exception(f"Error loading {file_path}: {e}")
            return None
        
    def handle_3d_image(self, img_array):
        """
        Extract a representative 2D slice from a 3D image.
          - If isotropic, use the middle slice along axis 0.
          - Otherwise, assume the smallest dimension is out-of-plane.
          - If the resulting slice is single channel, replicate it to three channels.

        Args:
            img_array (np.ndarray): The 3D image data as a NumPy array.

        Returns:
            np.ndarray: The 2D slice as a NumPy array.
        """
        shape = img_array.shape
        if len(shape) != 3:
            return img_array
        
        # Check if all dimensions are similar (isotropic)
        if np.allclose(shape, shape[0]):
            slice_idx = shape[0] // 2
            slice_2d = img_array[slice_idx, :, :]
        else:
            out_axis = np.argmin(shape)
            slice_idx = shape[out_axis] // 2
            slice_2d = np.take(img_array, slice_idx, axis=out_axis)

        if slice_2d.ndim == 2:
            slice_2d = np.stack([slice_2d] * 3, axis=-1)
        
        return slice_2d
    
    def normalize_image(self, img_array, modalities):
        """
        Normalize intensities:
        - For CT images, apply windowing using preset level/width.
        - For other modalities, clip intensities between the 0.5th and 99.5th percentile.
        - Then subtract the mean and divide by the standard deviation of the foreground.
        
        Args:
            img_array (np.ndarray): The image data as a NumPy array.
            modalities (list or str): The modality or modalities of the image.
        
        Returns:
            np.ndarray: The normalized image.
        """
        # Use the first modality if a list is provided.
        modality = modalities[0] if isinstance(modalities, list) and modalities else modalities
        foreground = img_array[img_array > 0]
        if foreground.size == 0:
            foreground = img_array.flatten()
        
        if modality and modality.lower() == 'ct':
            window_level = 50
            window_width = 350
            lower = window_level - window_width // 2
            upper = window_level + window_width // 2
            img_array = np.clip(img_array, lower, upper)
        else:
            lower = np.percentile(foreground, 0.5)
            upper = np.percentile(foreground, 99.5)
            img_array = np.clip(img_array, lower, upper)
        
        fg = img_array[img_array > 0]
        if fg.size == 0:
            fg = img_array.flatten()
        mean = np.mean(fg)
        std = np.std(fg) if np.std(fg) > 0 else 1.0
        normalized = (img_array - mean) / std
        return normalized

    
    def extract_h5_content(self, h5_file_path):
        """
        Process an .h5 file by extracting image and label data based on keywords.
        Looks for keys containing 'image' or 'img' for images and 'label' or 'mask' for labels.

        Args:
            h5_file_path (str): The path to the .h5 file.

        Returns:
            tuple: A tuple containing the image data and label data.
        """
        try:
            with h5py.File(h5_file_path, 'r') as h5_file:
                image_data = None
                label_data = None
                for key in h5_file.keys():
                    key_lower = key.lower()
                    if 'image' in key_lower or 'img' in key_lower:
                        image_data = h5_file[key][()]
                    elif 'label' in key_lower or 'mask' in key_lower:
                        label_data = h5_file[key][()]
                return image_data, label_data
        except Exception as e:
            logging.exception(f"Error processing h5 file {h5_file_path}: {e}")
            return None, None

    def convert_annotation_to_mask(self, annotation_file, image_shape):
        """
        Convert an annotation file (XML or txt) to a binary mask.
        For XML, looks for <svg> elements containing JSON strings representing polygons.
        For txt, each nonempty line is assumed to contain an (x, y) coordinate.

        Args:
            annotation_file (str): The path to the annotation file.
            image_shape (tuple): The shape of the image to which the mask should be applied.

        Returns:
            np.ndarray: The binary mask as a NumPy array.
        """
        ext = os.path.splitext(annotation_file)[1].lower()
        mask_img = Image.new("L", image_shape, 0)
        draw = ImageDraw.Draw(mask_img)
        
        if ext == '.xml':
            try:
                tree = ET.parse(annotation_file)
                root = tree.getroot()
                svg_elements = root.findall('.//svg')
                for svg_elem in svg_elements:
                    svg_content = svg_elem.text
                    if svg_content:
                        annotations = json.loads(svg_content)
                        for anno in annotations:
                            points = [(pt["x"], pt["y"]) for pt in anno.get("points", [])]
                            if points:
                                draw.polygon(points, outline=1, fill=1)
                return np.array(mask_img)
            except Exception as e:
                logging.exception(f"Error converting XML to mask for {annotation_file}: {e}")
                return None
        elif ext == '.txt':
            try:
                points = []
                with open(annotation_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            x, y = float(parts[0]), float(parts[1])
                            points.append((x, y))
                if points:
                    draw.polygon(points, outline=1, fill=1)
                return np.array(mask_img)
            except Exception as e:
                logging.exception(f"Error converting TXT to mask for {annotation_file}: {e}")
                return None
        else:
            print(f"Unsupported annotation file type: {annotation_file}")
            return None

    def crop_to_nonzero(self, img_array):
        """
        Crop the image to its nonzero region. If the crop reduces average size
        by 25% or more, then use the central 50% region for re-normalization.

        Args:
            img_array (np.ndarray): The image data as a NumPy array.

        Returns:
            np.ndarray: The cropped image data as a NumPy array.
        """
        nonzero_coords = np.argwhere(img_array)
        if nonzero_coords.size == 0:
            return img_array
        min_coords = nonzero_coords.min(axis=0)
        max_coords = nonzero_coords.max(axis=0) + 1
        cropped = img_array[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1]]
        orig_shape = np.array(img_array.shape[:2])
        crop_shape = np.array(cropped.shape[:2])
        if np.mean(crop_shape) < 0.75 * np.mean(orig_shape):
            print("Significant crop detected; applying central nonzero normalization.")
            h, w = crop_shape
            h_start = h // 4
            w_start = w // 4
            h_end = h_start + h // 2
            w_end = w_start + w // 2
            central_region = cropped[h_start:h_end, w_start:w_end]
            fg = central_region[central_region > 0]
            if fg.size == 0:
                fg = central_region.flatten()
            center_mean = np.mean(fg)
            center_std = np.std(fg) if np.std(fg) > 0 else 1.0
            cropped = (cropped - center_mean) / center_std
        return cropped
    
    def resize_image(self, img_array, is_mask):
        """
        Resize the image to the target size (default: 256x256).
          - Uses bicubic interpolation for images.
          - Uses nearest neighbor interpolation for masks.

        Args:
            img_array (np.ndarray): The image data as a NumPy array.
            is_mask (bool): True if the image is a mask, False otherwise.

        Returns:
        """
        target_size = self.target_size
        try:
            # Normalize the data to uint8 for PIL conversion if needed.
            if img_array.dtype != np.uint8:
                img_min, img_max = img_array.min(), img_array.max()
                img_norm = ((img_array - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)
            else:
                img_norm = img_array
            
            pil_img = Image.fromarray(img_norm)
            interp = Image.NEAREST if is_mask else Image.BICUBIC
            resized = pil_img.resize(target_size, resample=interp)
            return np.array(resized)
        except Exception as e:
            logging.exception(f"Error resizing image: {e}")
            return img_array

    def process_mask_components(self, mask_array):
        """
        For multi-class masks, split into individual binary masks.
        Dissect masks with multiple connected components into separate pieces.

        Args:
            mask_array (np.ndarray): The mask data as a NumPy array.

        Returns:
            masks (list): A list of binary masks for each connected component.
        """
        unique_vals = np.unique(mask_array)
        masks = []
        for val in unique_vals:
            if val == 0:
                continue
            binary_mask = (mask_array == val).astype(np.uint8) * 255
            labeled, num_features = ndimage.label(binary_mask)
            for i in range(1, num_features + 1):
                component = (labeled == i).astype(np.uint8) * 255
                masks.append(component)
        if not masks:
            masks = [mask_array]
        return masks
    
    def is_mask_large_enough(self, mask):
        """
        Returns True if the mask has at least 100 nonzero pixels.
        (100 pixels at 256x256 roughly corresponds to 0.153% of the image.)

        Args:
            mask (np.ndarray): The mask data as a NumPy array.

        Returns:
            bool: Whether the mask is large enough.
        """
        return np.sum(mask > 0) >= 100

    def convert_output_format(self, img_array, modalities, is_mask):
        """
        Convert the processed array into its final output format:
        - For masks, output a PNG (as a PIL Image).
        - For grayscale modalities, output a NIfTI image.
        - For RGB images, output as a PNG.
        """
        # Use the first modality if a list is provided.
        modality = modalities[0] if isinstance(modalities, list) and modalities else modalities
        if is_mask:
            return Image.fromarray(img_array)
        if modality and modality.lower() in self.grayscale_modalities:
            return nib.Nifti1Image(img_array, affine=np.eye(4))
        else:
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            return Image.fromarray(img_array.astype(np.uint8))


    def process_file(self, file_path, modalities=None, is_mask=False):
        """
        Execute the full preprocessing pipeline on a single file:
        1. Convert the file to a standard format.
        2. Load the file into a NumPy array.
        3. If 3D, extract a representative 2D slice.
        4. For images, perform intensity normalization.
        5. Crop to nonzero regions (with a central-region normalization if needed).
        6. Resize to the target size.
        7. For masks, split multi-class components and exclude small ones.
        8. Convert the processed array into the final output format.
        
        Args:
            file_path (str): The path to the input file.
            modalities (list or str, optional): The modality or modalities. Default is None.
            is_mask (bool, optional): True if the file is a mask, False otherwise.
        
        Returns:
            - For images: either a nibabel NIfTI image or a PIL Image.
            - For masks: a list of PIL Images.
        """
        # Step 1: Convert file
        conv_path = self.convert_file(file_path, modalities, is_mask)
        # Step 2: Load image data
        img_array = self.load_image(conv_path)
        if img_array is None:
            return None
        # Step 3: Handle potential 3D images
        if img_array.ndim == 3:
            img_array = self.handle_3d_image(img_array)
        # Step 4: Normalize image intensities (skip for masks)
        if not is_mask:
            img_array = self.normalize_image(img_array, modalities)
        # Step 5: Crop to nonzero regions
        img_array = self.crop_to_nonzero(img_array)
        # Step 6: Resize to the target size
        img_array = self.resize_image(img_array, is_mask)
        # Step 7 & 8: Process masks or convert output format
        if is_mask:
            mask_list = self.process_mask_components(img_array)
            mask_list = [mask for mask in mask_list if self.is_mask_large_enough(mask)]
            return [Image.fromarray(mask) for mask in mask_list]
        return self.convert_output_format(img_array, modalities, is_mask)