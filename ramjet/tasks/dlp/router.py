from .image_watermark import (ImageWaterMarkSignView, ImageWaterMarkVerifyView,
                              ImageWaterMarkView)


def bind_handle(add_route):
    add_route("image/watermark/fetch/", ImageWaterMarkSignView)
    add_route("image/watermark/verify/", ImageWaterMarkVerifyView)
    add_route("image/watermark/", ImageWaterMarkView)
