from .image_watermark import ImageWaterMarkVerifyView, ImageWaterMarkView


def bind_handle(add_route):
    add_route("image/watermark/fetch/", ImageWaterMarkView)
    add_route("image/watermark/verify/", ImageWaterMarkVerifyView)
    add_route("image/watermark/", ImageWaterMarkView)
