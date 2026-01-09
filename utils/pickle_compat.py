import pickle


class _NumpyCompatUnpickler(pickle.Unpickler):
    """
    Unpickler that remaps NumPy 2.x internal module paths (numpy._core.*)
    to NumPy 1.x equivalents (numpy.core.*) so pickles created under
    NumPy 2.x can be loaded in environments with NumPy 1.x.
    """

    def find_class(self, module, name):
        # Remap any numpy._core.* references to numpy.core.*
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def safe_pickle_load(file_obj):
    """
    Load a pickle, falling back to a NumPy 2.x -> 1.x compatibility
    unpickler if needed.
    """
    try:
        pos = None
        try:
            pos = file_obj.tell()
        except Exception:
            pass
        return pickle.load(file_obj)
    except ModuleNotFoundError as exc:
        # Common when a pickle was created with NumPy 2.x and contains
        # references to numpy._core.*, which do not exist in NumPy 1.x.
        if "numpy._core" in str(exc):
            # Retry same object with compat unpickler
            try:
                if pos is not None:
                    file_obj.seek(pos)
            except Exception:
                # If seeking fails, fall through and attempt load anyway
                pass
            return _NumpyCompatUnpickler(file_obj).load()
        raise


def safe_pickle_load_path(path: str):
    with open(path, "rb") as f:
        return safe_pickle_load(f)
