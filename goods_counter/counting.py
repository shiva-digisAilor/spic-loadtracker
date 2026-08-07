"""
Counting logic -- grand total is "the most objects of this class ever
seen visible at the same time," same idea as before, but now a proposed
NEW higher count must hold steady for several consecutive frames before
it's accepted. This filters out one-frame noise spikes (a shadow briefly
splitting a mask, a borderline-confidence false detection, motion blur)
that would otherwise permanently and incorrectly bump the total up.

A genuine new object stays in frame continuously, so it easily clears the
required streak. A flicker lasting one or two frames does not.
"""


class GoodsCounter:
    def __init__(self, cargo_classes, confirm_frames=8):
        self.total_count = {code: 0 for code in cargo_classes}
        self.confirm_frames = confirm_frames

        # Tracks a candidate higher count that hasn't been accepted yet.
        self._pending_count = {code: 0 for code in cargo_classes}
        self._pending_streak = {code: 0 for code in cargo_classes}

    def update(self, code, currently_visible_count):
        """Call once per class, per frame, with how many are visible right
        now. Only raises the running total once the SAME higher count has
        been seen for `confirm_frames` consecutive calls in a row.
        Returns True if the total just changed."""
        if currently_visible_count <= self.total_count[code]:
            # Nothing higher than what we already have -- no candidate to
            # track, reset any in-progress streak (a spike that didn't
            # even last one more frame doesn't count as "holding steady").
            self._pending_count[code] = 0
            self._pending_streak[code] = 0
            return False

        if currently_visible_count == self._pending_count[code]:
            self._pending_streak[code] += 1
        else:
            # A different candidate than last frame -- restart the streak.
            self._pending_count[code] = currently_visible_count
            self._pending_streak[code] = 1

        if self._pending_streak[code] >= self.confirm_frames:
            old = self.total_count[code]
            self.total_count[code] = currently_visible_count
            print(f"{code}: grand total updated {old} -> {currently_visible_count} "
                  f"(confirmed over {self.confirm_frames} frames)")
            self._pending_count[code] = 0
            self._pending_streak[code] = 0
            return True

        return False

    def grand_total(self):
        return sum(self.total_count.values())