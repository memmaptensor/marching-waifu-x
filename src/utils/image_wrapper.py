import cv2
import numpy as np
import PIL.Image


class image_wrapper:
    def __init__(self, img, format):
        if format == "pil":
            self.img = img
        elif format == "np":
            self.img = PIL.Image.fromarray(img)
        elif format == "cv2":
            self.img = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            raise TypeError(f"Unsupported type: {format}")

    def to_pil(self):
        return self.img

    def to_np(self):
        return np.array(self.img)

    def to_cv2(self):
        return cv2.cvtColor(self.to_np(), cv2.COLOR_RGB2BGR)

    def resize(self, width, height):
        self.img = self.img.resize((width, height), PIL.Image.LANCZOS)
        return self

    def scale(self, scale):
        return self.resize(int(self.img.width * scale), int(self.img.height * scale))

    def concatenate(self, other, axis=0):
        width, height = self.img.width, self.img.height
        width, height = (
            width + other.img.width if axis == 0 else width,
            height + other.img.height if axis == 1 else height,
        )

        new_image = PIL.Image.new(other.img.mode, (width, height))
        new_image.paste(self.img, (0, 0))
        new_image.paste(
            other.img,
            (
                self.img.width if axis == 0 else 0,
                self.img.height if axis == 1 else 0,
            ),
        )

        self.img = new_image
        return self
