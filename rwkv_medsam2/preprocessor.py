import os, gzip, io, json, logging
import numpy as np
from PIL import Image, ImageDraw
import nibabel as nib
import scipy.ndimage as ndimage
import SimpleITK as sitk
import h5py
import xml.etree.ElementTree as ET
import cv2
from scipy.io import loadmat

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
        logging.debug(f"Preprocessor initialized with target size: {target_size}")

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
        lower_fp = file_path.lower()
        if lower_fp.endswith('.nii.gz'):
            ext = '.nii.gz'
        else:
            ext = os.path.splitext(file_path)[1].lower()
        logging.debug(f"Converting file: {file_path} (ext={ext}, is_mask={is_mask}, modality={modality})")
        if not is_mask:
            if ext in {'.dcm', '.nrrd', '.mhd'}:
                logging.info(f"Converting 3D image {file_path} to NIfTI format...")
                try:
                    img = sitk.ReadImage(file_path)
                    new_path = os.path.splitext(file_path)[0] + '.nii.gz'
                    sitk.WriteImage(img, new_path)
                    logging.debug(f"Converted 3D image saved to {new_path}")
                    return new_path
                except Exception as e:
                    logging.exception(f"Error converting {file_path} to NIfTI: {e}")
                    return file_path
            else:
                logging.debug(f"No conversion needed for {file_path}")
                return file_path
        else:
            if ext == '.avi':
                logging.info(f"Extracting frame from video {file_path}...")
                try:
                    cap = cv2.VideoCapture(file_path)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        new_path = os.path.splitext(file_path)[0] + '.png'
                        cv2.imwrite(new_path, frame)
                        logging.debug(f"Extracted frame saved to {new_path}")
                        return new_path
                    else:
                        logging.error(f"Failed to read frame from {file_path}")
                        return file_path
                except Exception as e:
                    logging.exception(f"Error processing video file {file_path}: {e}")
                    return file_path
            elif ext == '.npy':
                logging.info(f"Converting .npy file {file_path} to PNG...")
                try:
                    arr = np.load(file_path)
                    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
                    new_path = os.path.splitext(file_path)[0] + '.png'
                    Image.fromarray(arr.astype(np.uint8)).save(new_path)
                    logging.debug(f".npy file converted and saved to {new_path}")
                    return new_path
                except Exception as e:
                    logging.exception(f"Error converting npy file {file_path}: {e}")
                    return file_path
            elif ext in {'.ppm', '.ppm.gz'}:
                logging.info(f"Converting {ext} file {file_path} to PNG...")
                try:
                    if ext == '.ppm.gz':
                        with gzip.open(file_path, 'rb') as f:
                            data = f.read()
                        img = Image.open(io.BytesIO(data))
                    else:
                        img = Image.open(file_path)
                    new_path = os.path.splitext(file_path)[0] + '.png'
                    img.save(new_path)
                    logging.debug(f"Converted {ext} file saved to {new_path}")
                    return new_path
                except Exception as e:
                    logging.exception(f"Error converting {ext} file {file_path}: {e}")
                    return file_path
            else:
                if ext == '.mat':
                    logging.info(f"Converting .mat mask file {file_path} to PNG...")
                    try:
                        mat_data = loadmat(file_path)
                        mask = None
                        if 'mask' in mat_data:
                            mask = mat_data['mask']
                        else:
                            for key in mat_data:
                                if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                                    mask = mat_data[key]
                                    break
                        if mask is None:
                            logging.error(f"No valid mask found in {file_path}")
                            return file_path
                        mask = (mask > 0).astype(np.uint8) * 255
                        new_path = os.path.splitext(file_path)[0] + '.png'
                        Image.fromarray(mask).save(new_path)
                        logging.debug(f".mat mask converted and saved to {new_path}")
                        return new_path
                    except Exception as e:
                        logging.exception(f"Error converting mat file {file_path}: {e}")
                        return file_path
                elif ext == '.npy':
                    logging.info(f"Converting .npy mask file {file_path} to PNG...")
                    try:
                        arr = np.load(file_path)
                        mask = (arr > 0).astype(np.uint8) * 255
                        new_path = os.path.splitext(file_path)[0] + '.png'
                        Image.fromarray(mask).save(new_path)
                        logging.debug(f".npy mask converted and saved to {new_path}")
                        return new_path
                    except Exception as e:
                        logging.exception(f"Error converting npy mask file {file_path}: {e}")
                        return file_path
                elif ext in {'.ppm', '.ppm.gz'}:
                    logging.info(f"Converting {ext} mask file {file_path} to PNG...")
                    try:
                        if ext == '.ppm.gz':
                            with gzip.open(file_path, 'rb') as f:
                                data = f.read()
                            img = Image.open(io.BytesIO(data))
                        else:
                            img = Image.open(file_path)
                        new_path = os.path.splitext(file_path)[0] + '.png'
                        img.save(new_path)
                        logging.debug(f"Converted {ext} mask saved to {new_path}")
                        return new_path
                    except Exception as e:
                        logging.exception(f"Error converting {ext} mask file {file_path}: {e}")
                        return file_path
                elif ext != '.png':
                    logging.info(f"Converting mask {file_path} to PNG...")
                    try:
                        with Image.open(file_path) as img:
                            new_path = os.path.splitext(file_path)[0] + '.png'
                            img.save(new_path)
                        logging.debug(f"Mask converted to PNG and saved to {new_path}")
                        return new_path
                    except Exception as e:
                        logging.exception(f"Error converting {file_path} to PNG: {e}")
                        return file_path
                else:
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
        lower_fp = file_path.lower()
        if lower_fp.endswith('.nii.gz'):
            ext = '.nii.gz'
        else:
            ext = os.path.splitext(file_path)[1].lower()
        logging.debug(f"Loading image from {file_path} with extension {ext}")
        try:
            if ext in {'.nii', '.nii.gz'}:
                logging.debug(f"Using nibabel to load NIfTI file: {file_path}")
                nii = nib.load(file_path)
                data = nii.get_fdata()
                logging.debug(f"NIfTI image shape: {data.shape}")
                return data
            else:
                logging.debug(f"Using PIL to load image: {file_path}")
                with Image.open(file_path) as img:
                    data = np.array(img)
                logging.debug(f"PIL image shape: {data.shape}")
                return data
        except Exception as e:
            logging.exception(f"Error loading image {file_path}: {e}")
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
        if img_array.ndim != 3:
            logging.debug("Image is not 3D; skipping slice extraction.")
            return img_array
        shape = img_array.shape
        logging.debug(f"Handling 3D image with shape: {shape}")
        if np.allclose(shape, shape[0]):
            slice_idx = shape[0] // 2
            slice_2d = img_array[slice_idx, :, :]
            logging.debug(f"Extracted isotropic middle slice at index {slice_idx}")
        else:
            out_axis = np.argmin(shape)
            slice_idx = shape[out_axis] // 2
            slice_2d = np.take(img_array, slice_idx, axis=out_axis)
            logging.debug(f"Extracted slice along axis {out_axis} at index {slice_idx}")
        if slice_2d.ndim == 2:
            slice_2d = np.stack([slice_2d] * 3, axis=-1)
            logging.debug("Converted single-channel slice to 3 channels")
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
        logging.debug(f"Normalizing image with modality: {modality}")
        foreground = img_array[img_array > 0]
        if foreground.size == 0:
            foreground = img_array.flatten()
        if modality and modality.lower() == 'ct':
            window_level = 50
            window_width = 350
            lower = window_level - window_width // 2
            upper = window_level + window_width // 2
            img_array = np.clip(img_array, lower, upper)
            logging.debug(f"Applied CT windowing: lower={lower}, upper={upper}")
        else:
            lower = np.percentile(foreground, 0.5)
            upper = np.percentile(foreground, 99.5)
            img_array = np.clip(img_array, lower, upper)
            logging.debug(f"Applied percentile clipping: lower={lower}, upper={upper}")
        fg = img_array[img_array > 0]
        if fg.size == 0:
            fg = img_array.flatten()
        mean = np.mean(fg)
        std = np.std(fg) if np.std(fg) > 0 else 1.0
        normalized = (img_array - mean) / std
        logging.debug(f"Normalized image: mean={mean}, std={std}")
        return normalized

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
        logging.debug(f"Converting annotation file {annotation_file} to mask with target shape {image_shape}")
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
                logging.debug("Successfully converted XML annotation to mask")
                return np.array(mask_img)
            except Exception as e:
                logging.exception(f"Error converting XML annotation {annotation_file}: {e}")
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
                logging.debug("Successfully converted TXT annotation to mask")
                return np.array(mask_img)
            except Exception as e:
                logging.exception(f"Error converting TXT annotation {annotation_file}: {e}")
                return None
        else:
            logging.error(f"Unsupported annotation file type: {annotation_file}")
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
            logging.debug("No nonzero regions found; skipping cropping.")
            return img_array
        min_coords = nonzero_coords.min(axis=0)
        max_coords = nonzero_coords.max(axis=0) + 1
        cropped = img_array[min_coords[0]:max_coords[0], min_coords[1]:max_coords[1]]
        orig_shape = np.array(img_array.shape[:2])
        crop_shape = np.array(cropped.shape[:2])
        logging.debug(f"Cropping image from shape {orig_shape} to {crop_shape}")
        if np.mean(crop_shape) < 0.75 * np.mean(orig_shape):
            logging.info("Significant crop detected; applying central region normalization.")
            h, w = crop_shape
            h_start = h // 4
            w_start = w // 4
            h_end = h_start + h // 2
            w_end = w_start + w // 2
            central = cropped[h_start:h_end, w_start:w_end]
            fg = central[central > 0]
            if fg.size == 0:
                fg = central.flatten()
            center_mean = np.mean(fg)
            center_std = np.std(fg) if np.std(fg) > 0 else 1.0
            cropped = (cropped - center_mean) / center_std
            logging.debug(f"Central normalization applied: mean={center_mean}, std={center_std}")
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
        try:
            if img_array.dtype != np.uint8:
                img_min, img_max = img_array.min(), img_array.max()
                img_norm = ((img_array - img_min) / (img_max - img_min + 1e-8) * 255).astype(np.uint8)
            else:
                img_norm = img_array
            pil_img = Image.fromarray(img_norm)
            interp = Image.NEAREST if is_mask else Image.BICUBIC
            resized = pil_img.resize(self.target_size, resample=interp)
            logging.debug(f"Resized image to {self.target_size} using {'NEAREST' if is_mask else 'BICUBIC'} interpolation")
            return np.array(resized)
        except Exception as e:
            logging.exception(f"Error resizing image: {e}")
            return img_array
    
    def is_mask_large_enough(self, mask):
        """
        Returns True if the mask has at least 100 nonzero pixels.
        (100 pixels at 256x256 roughly corresponds to 0.153% of the image.)

        Args:
            mask (np.ndarray): The mask data as a NumPy array.

        Returns:
            bool: Whether the mask is large enough.
        """
        pixel_count = np.sum(mask > 0)
        logging.debug(f"Mask nonzero pixel count: {pixel_count}")
        return pixel_count >= 100

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
            logging.debug("Converting mask array to PIL Image for PNG output")
            return Image.fromarray(img_array)
        if modality and modality.lower() in self.grayscale_modalities:
            logging.debug("Converting image array to NIfTI for grayscale modality")
            return nib.Nifti1Image(img_array, affine=np.eye(4))
        else:
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            logging.debug("Converting image array to PIL Image for RGB PNG output")
            return Image.fromarray(img_array.astype(np.uint8))

    def process_file(self, file_path, modalities=None, is_mask=False):
        """
        Execute the full preprocessing pipeline on a single file:
        1. Convert the file to a standard format.
        2. Load the file into a NumPy array.
        3. If 3D, extract a representative 2D slice.
        4. For images, perform intensity normalization.
        5. Crop to nonzero regions (with central-region normalization if needed).
        6. Resize to the target size.
        7. For masks, directly convert the processed array into the final output format
           (without splitting into components).
        
        Args:
            file_path (str): The path to the input file.
            modalities (list or str, optional): The modality or modalities.
            is_mask (bool, optional): True if the file is a mask, False otherwise.
        
        Returns:
            For images: either a nibabel NIfTI image or a PIL Image.
            For masks: a list containing a single PIL Image.
        """
        logging.debug(f"Processing file: {file_path} (is_mask={is_mask})")
        conv_path = self.convert_file(file_path, modalities, is_mask)
        img_array = self.load_image(conv_path)
        if img_array is None:
            logging.error(f"Image array is None for file: {file_path}")
            return None
        if img_array.ndim == 3:
            img_array = self.handle_3d_image(img_array)
        if not is_mask:
            img_array = self.normalize_image(img_array, modalities)
        #img_array = self.crop_to_nonzero(img_array)
        img_array = self.resize_image(img_array, is_mask)
        if is_mask:
            # Instead of splitting mask components, simply convert the full mask.
            if not self.is_mask_large_enough(img_array):
                logging.warning(f"Mask is too small: {file_path}")
            processed = self.convert_output_format(img_array, modalities, is_mask)
            logging.debug("Processed mask without splitting into components")
            return [processed]
        else:
            result = self.convert_output_format(img_array, modalities, is_mask)
            logging.debug("Processed image file successfully")
            return result