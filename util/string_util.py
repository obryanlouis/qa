"""Utility methods for strings.
"""

def clear_printed_line(num_chars_to_clear=200):
    print(" " * num_chars_to_clear, end="\r", flush=True)

# Encode a string in utf8 encoding. Mostly, this is used so that it can be
# printed.
def utf8_str(obj):
    return str(str(obj).encode("utf-8"))
