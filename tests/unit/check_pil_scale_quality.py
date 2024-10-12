import PIL.Image as Image

from marie.logging_core.profile import TimeContext

if __name__ == "__main__":

    # read pil image and rescale
    image = Image.open(
        "/home/gbugaj/datasets/private/overlay_ssim/EOB_CLEAN/159000444_3_00001.tif"
    )

    size = image.size
    # warmup
    r_bicubic = image.resize(size, Image.BICUBIC)
    r_lanczos = image.resize(size, Image.LANCZOS)
    r_bilinear = image.resize(size, Image.BILINEAR)

    with TimeContext(f"### BICUBIC"):
        for i in range(100):
            r_bicubic = image.resize(size, Image.BICUBIC)
    with TimeContext(f"### LANCZOS"):
        for i in range(100):
            r_lanczos = image.resize(size, Image.LANCZOS)
    with TimeContext(f"### BILINEAR"):
        for i in range(100):
            r_bilinear = image.resize(size, Image.BILINEAR)

    r_bicubic.save("/tmp/r_bicubic.png")
    r_lanczos.save("/tmp/r_lanczos.png")
    r_bilinear.save("/tmp/r_bilinear.png")
