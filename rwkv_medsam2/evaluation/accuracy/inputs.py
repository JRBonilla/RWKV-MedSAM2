"""Bounded, read-only input reuse with frozen hashes and change detection.

Each process verifies a file once per stat signature. Input files must remain
immutable during evaluation. Arrays are never shared between GPU processes.
"""
from collections import OrderedDict
from pathlib import Path

from ..shared.data import file_hash


def signature(path):
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


class InputCache:
    def __init__(self, max_bytes=2 * 1024**3):
        if max_bytes < 0:
            raise ValueError('Input cache size must be nonnegative')
        self.max_bytes = max_bytes
        self.bytes = 0
        self.arrays = OrderedDict()
        self.verified = OrderedDict()
        self.stats = dict(hash_reads=0, array_loads=0, array_hits=0)

    def verify(self, path, expected):
        path = Path(path).resolve()
        before = signature(path)
        key = (str(path), expected)
        if self.verified.get(key) != before:
            if file_hash(path) != expected or signature(path) != before:
                raise ValueError(f'Frozen input changed: {path}')
            self.stats['hash_reads'] += 1
            self.verified[key] = before
        self.verified.move_to_end(key)
        while len(self.verified) > 8192:
            self.verified.popitem(last=False)
        return path, before

    def verify_case(self, case, root):
        return {k: self.verify(Path(case[k]) if Path(case[k]).is_absolute()
                               else Path(root) / case[k], case['file_sha256'][k])
                for k in ('image', 'mask')}

    def _get(self, role, case, verified, loader):
        path, stamp = verified
        key = (role, str(path), stamp, case['dim'], case.get('axis_zyx'),
               case.get('reverse_slices'), case.get('label') if role == 'mask' else None)
        if key in self.arrays:
            self.stats['array_hits'] += 1
            self.arrays.move_to_end(key)
            return self.arrays[key]
        value = loader(case, path)
        if signature(path) != stamp:
            raise ValueError(f'Input changed while loading: {path}')
        self.stats['array_loads'] += 1
        value[0].setflags(write=False)
        size = value[0].nbytes
        if size <= self.max_bytes:
            while self.arrays and self.bytes + size > self.max_bytes:
                _, old = self.arrays.popitem(last=False)
                self.bytes -= old[0].nbytes
            self.arrays[key] = value
            self.bytes += size
        return value

    def read_case(self, case, root):
        paths = self.verify_case(case, root)
        frames, geometry, limits = self._get('image', case, paths['image'], _image)
        masks, mask_geometry = self._get('mask', case, paths['mask'], _mask)
        _match_geometry(geometry, mask_geometry)
        if frames.shape[:3] != masks.shape:
            raise ValueError('Image/mask array shape differs')
        return frames, masks, dict(geometry=geometry, intensity_limits=limits, frames=len(frames))

    def read_target(self, case, root):
        # Verify image content, but do not decompress it or recompute percentiles.
        paths = self.verify_case(case, root)
        target, geometry = self._get('mask', case, paths['mask'], _mask)
        transform = case['transform']
        _match_geometry(transform['geometry'], geometry)
        if len(target) != transform['frames']:
            raise ValueError('Stored target frame count changed')
        return target, transform


def _volume(case, path):
    import numpy as np
    import SimpleITK as sitk
    v = sitk.ReadImage(str(path))
    if v.GetDimension() != 3 or v.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError('Expected scalar 3D image/mask')
    array = np.moveaxis(sitk.GetArrayFromImage(v), case['axis_zyx'], 0)
    if case['reverse_slices']:
        array = array[::-1]
    geometry = {name: list(getattr(v, accessor)()) for name, accessor in
                [('size_xyz', 'GetSize'), ('spacing_xyz', 'GetSpacing'),
                 ('origin_xyz', 'GetOrigin'), ('direction', 'GetDirection')]}
    geometry.update(axis_zyx=case['axis_zyx'], reverse_slices=case['reverse_slices'])
    return array, geometry


def _image(case, path):
    import numpy as np
    from PIL import Image
    if case['dim'] == 2:
        with Image.open(path) as im:
            if im.mode not in ('L', 'RGB'):
                raise ValueError('2D input must be byte L/RGB')
            x = np.asarray(im).copy()
        if x.dtype != np.uint8:
            raise ValueError('Unexpected 2D dtype')
        geometry = dict(original_hw=list(x.shape[:2]), units='processed_pixels')
        if x.ndim == 2:
            x = np.repeat(x[..., None], 3, axis=-1)
        return x[None], geometry, None
    frames, geometry = _volume(case, path)
    if not np.isfinite(frames).all():
        raise ValueError('Nonfinite volume intensities')
    return frames, geometry, [float(x) for x in np.percentile(frames, [0.5, 99.5])]


def _mask(case, path):
    import numpy as np
    from PIL import Image
    if case['dim'] == 2:
        with Image.open(path) as im:
            raw = np.asarray(im).copy()
        if raw.ndim != 2:
            raise ValueError('Expected single-channel annotation')
        geometry = dict(original_hw=list(raw.shape), units='processed_pixels')
        raw = raw[None]
    else:
        raw, geometry = _volume(case, path)
    integer = np.issubdtype(raw.dtype, np.integer) or raw.dtype == np.bool_
    if not integer and (not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all()):
        raise ValueError('Mask labels must be finite integers')
    return raw == case['label'], geometry


def _match_geometry(expected, actual):
    import numpy as np
    if set(expected) != set(actual):
        raise ValueError('Stored geometry fields differ')
    for key, value in expected.items():
        same = value == actual[key] if isinstance(value, str) else np.allclose(value, actual[key], rtol=0, atol=1e-5)
        if not same:
            raise ValueError(f'Stored geometry differs: {key}')
