from .views import UploadFileView


def bind_handle(add_route):
    add_route("/proto/", UploadFileView)
