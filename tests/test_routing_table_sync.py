"""Parity checks for the actual release docs and the card staging step."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.hub import sync_routing_table as sync


class RoutingTableTests(unittest.TestCase):
    def test_checked_in_cards_match_the_release_table(self):
        sync.sync_local(sync.canonical(sync.ROOT), sync.card_paths(sync.ROOT))

    def test_stale_staging_table_is_detected_and_only_its_table_is_updated(self):
        table = sync.canonical(sync.ROOT)
        with tempfile.TemporaryDirectory() as temp:
            paths = sync.card_paths(sync.ROOT, Path(temp))[-2:]
            for path in paths:
                path.parent.mkdir()
                path.write_bytes(b"Own architecture\n" + sync.BEGIN + b"old rows" + sync.END + b"\nOwn figures\n")
            with self.assertRaisesRegex(ValueError, "routing table differs"):
                sync.sync_local(table, paths)
            sync.sync_local(table, paths, write=True)
            for path in paths:
                self.assertEqual(path.read_bytes(), b"Own architecture\n" + sync.BEGIN + table
                                 + sync.END + b"\nOwn figures\n")

    def test_live_check_rejects_a_partial_card_even_when_its_results_image_exists(self):
        table = sync.canonical(sync.ROOT)
        good = sync.BEGIN + table + sync.END
        bad = sync.BEGIN + b"|incomplete table|" + sync.END + b"\n![All five runs](figures/routing-accuracy.png)"
        with patch.object(sync, "urlopen", side_effect=[io.BytesIO(good), io.BytesIO(bad)]):
            with self.assertRaisesRegex(ValueError, "Joey-1-Compact-Loop: published routing table differs"):
                sync.check_live(table, "example")

    def test_missing_marker_does_not_overwrite_another_staged_card(self):
        with tempfile.TemporaryDirectory() as temp:
            a, b = (Path(temp) / f"{n}.md" for n in ("full", "compact"))
            old = sync.BEGIN + b"old" + sync.END
            a.write_bytes(old)
            b.write_bytes(b"unrelated card")
            with self.assertRaisesRegex(ValueError, "expected one marked routing table"):
                sync.sync_local(b"new", [a, b], write=True)
            self.assertEqual(a.read_bytes(), old)
