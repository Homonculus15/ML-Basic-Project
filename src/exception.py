import sys
import types

def error_message_detail(error, error_detail: types.ModuleType):
    _, _, exc_tb = error_detail.exc_info()

    # exc_tb may be None; guard against accessing attributes on None
    if exc_tb is None:
        file_name = "<unknown>"
        line_no = 0
    else:
        # use getattr to be defensive in case intermediate attributes are missing
        frame = getattr(exc_tb, 'tb_frame', None)
        code = getattr(frame, 'f_code', None)
        file_name = getattr(code, 'co_filename', '<unknown>')
        line_no = getattr(exc_tb, 'tb_lineno', 0)

    error_message = (
        f"Error occurred in Python Script name [{file_name}] "
        f"line number [{line_no}] "
        f"error message [{str(error)}]"
    )

    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: types.ModuleType):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message