import numpy as np
from scipy.ndimage import rotate, gaussian_filter


def transform_test_set(
    data: dict,
    test_idx: dict,
    operation: str = "rotate",
    replace: bool = True,
    **kwargs,
) -> tuple[dict, dict]:
    """
    Apply transformations to test set images.

    Parameters
    ----------
        operation: Type of transformation - "rotate", "noise", "blur", "flip", "brightness"
        replace: If True, replace original test images. If False, add augmented as new images
        **kwargs: Additional parameters for specific operations
            - angle: rotation angle in degrees (default: 15)
            - noise_std: standard deviation for gaussian noise (default: 10.0)
            - blur_std: standard deviation for gaussian blur (default: 1.0)
            - flip_direction: "horizontal" or "vertical" (default: "horizontal")
            - brightness_factor: multiplicative factor (default: 1.2)
    """

    new_data = {sid: imgs.copy() for sid, imgs in data.items()}
    new_test_idx = {sid: idx.copy() for sid, idx in test_idx.items()}

    for sid in data.keys():
        test_indices = test_idx[sid]
        test_images = data[sid][test_indices]

        if operation == "rotate":
            angle = kwargs.get("angle", 15)
            transformed = np.array(
                [rotate(img, angle, reshape=False, order=1) for img in test_images]
            )

        elif operation == "noise":
            std = kwargs.get("noise_std", 10.0)
            noise = np.random.normal(0, std, test_images.shape)
            transformed = np.clip(test_images + noise, 0, 255)

        elif operation == "blur":
            std = kwargs.get("blur_std", 1.0)
            transformed = np.array(
                [gaussian_filter(img, sigma=std) for img in test_images]
            )

        elif operation == "flip":
            direction = kwargs.get("flip_direction", "horizontal")
            axis = 2 if direction == "horizontal" else 1
            transformed = np.array([np.flip(img, axis=axis) for img in test_images])

        elif operation == "brightness":
            factor = kwargs.get("brightness_factor", 1.2)
            transformed = np.clip(test_images * factor, 0, 255)

        else:
            raise ValueError(f"Unsupported operation: {operation}")

        if replace:
            new_data[sid][test_indices] = transformed

        else:
            n = new_data[sid].shape[0]
            new_data[sid] = np.vstack([new_data[sid], transformed], axis=0)

            new_indices = np.arange(n, n + transformed.shape[0])
            new_test_idx[sid] = np.concatenate([new_test_idx[sid], new_indices])

    return new_data, new_test_idx
