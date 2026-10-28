"""
Safe reader for the DrivaerML-PCTR `.pt` files (third-party PyTorch files).

A `.pt` file is a zip holding a pickle plus raw tensor bytes. A normal
pickle/torch.load can run arbitrary code from the file, so this reader uses a
restricted unpickler that only allows the three names these files need
(the tensor-rebuild function, the float/long storage types and OrderedDict)
and rebuilds every tensor as a NumPy array straight from the raw bytes.
Anything else in the pickle stops the load with an error. PyTorch is not
needed.
"""

import collections
import pickle
import zipfile

import numpy as np

_DTYPES = {"FloatStorage": np.float32, "LongStorage": np.int64, "DoubleStorage": np.float64, "IntStorage": np.int32}


def _rebuild_tensor(storage, offset, size, stride, *unused):
    size = tuple(int(s) for s in size)
    n = int(np.prod(size)) if size else 1
    expected = tuple(int(np.prod(size[i + 1:])) for i in range(len(size)))
    if tuple(int(s) for s in stride) != expected:
        raise pickle.UnpicklingError("non-contiguous tensor; not supported by the safe reader")
    return storage[offset:offset + n].reshape(size)


class _StorageType:
    def __init__(self, name):
        self.name = name


class _SafeUnpickler(pickle.Unpickler):
    def __init__(self, fh, archive, prefix):
        super().__init__(fh)
        self.archive, self.prefix = archive, prefix

    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor
        if module == "torch" and name in _DTYPES:
            return _StorageType(name)
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"blocked: {module}.{name}")

    def persistent_load(self, pid):
        kind, storage_type, key, _location, numel = pid
        if kind != "storage" or not isinstance(storage_type, _StorageType):
            raise pickle.UnpicklingError("unexpected persistent id")
        raw = self.archive.read(f"{self.prefix}/data/{key}")
        arr = np.frombuffer(raw, dtype=np.dtype(_DTYPES[storage_type.name]).newbyteorder("<"))
        if arr.size != numel:
            raise pickle.UnpicklingError("storage size mismatch")
        return arr.astype(_DTYPES[storage_type.name])


def read_pt(path):
    """Return the file's dict with every tensor as a NumPy array."""
    with zipfile.ZipFile(path) as archive:
        pkl = [n for n in archive.namelist() if n.endswith("/data.pkl")]
        if len(pkl) != 1:
            raise ValueError(f"{path}: expected one data.pkl")
        prefix = pkl[0][: -len("/data.pkl")]
        if archive.read(f"{prefix}/byteorder").strip() != b"little":
            raise ValueError(f"{path}: not little-endian")
        with archive.open(pkl[0]) as fh:
            return _SafeUnpickler(fh, archive, prefix).load()
