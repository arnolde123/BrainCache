"""
Minimal readline stub for Windows. The Pinecone SDK imports readline unconditionally;
on Windows the stdlib readline module does not exist. This stub satisfies the import
so the app starts without installing pyreadline (which is broken on Python 3.10+).
"""


def get_completer():
    return None


def set_completer(*args, **kwargs):
    pass


def get_history_length():
    return 0


def get_current_history_length():
    return 0


def read_history_file(*args, **kwargs):
    pass


def write_history_file(*args, **kwargs):
    pass


def clear_history():
    pass


def add_history(*args, **kwargs):
    pass


def insert_text(*args, **kwargs):
    pass


def parse_and_bind(*args, **kwargs):
    pass
