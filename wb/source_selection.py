"""Choose a native image page without converting or discarding originals."""
from wb.core import sha


def preferred_source_ids(sources):
    groups = {}
    for key, source in sources.items():
        groups.setdefault(sha(source.original), []).append(key)
    def priority(key):
        source = sources[key]
        bits = source.gray.dtype.itemsize * 8
        grayscale = 'palette' not in source.conversion
        return (grayscale and bits == 16, grayscale, bits, -source.page)
    return [max(keys, key=priority) for keys in groups.values()]
