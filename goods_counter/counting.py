"""
Counting logic - no line. New tracked id -> count it, once.
Kept as its own tiny module since it's a distinct concern from rendering
or inference, and is easy to unit-test in isolation.
"""


class GoodsCounter:
    def __init__(self, cargo_classes):
        self.counted_ids = {code: set() for code in cargo_classes}
        self.total_count = {code: 0 for code in cargo_classes}

    def register(self, code, obj_id):
        """If obj_id hasn't been seen before for this class, count it.
        Returns True if this call incremented the count."""
        if obj_id not in self.counted_ids[code]:
            self.counted_ids[code].add(obj_id)
            self.total_count[code] += 1
            print(f"Counted {code} #{self.total_count[code]} (object id {obj_id})")
            return True
        return False

    def grand_total(self):
        return sum(self.total_count.values())
