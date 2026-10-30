"""Extractor integrity tests using small in-memory arrays and a local byte file."""
import tempfile
import unittest
from pathlib import Path
import awkward as ak
import numpy as np

from atmospheric_response.v2_extract import (
    BoundedReader, expand_responses, digest_file, validate_resume_metadata,
    validate_cached_record, inventory_source_directory, SCHEMA_NAME,
)


class ExtractorTests(unittest.TestCase):
    def test_relative_source_directory_is_anchored_to_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            inventory = Path(tmp) / "manifests" / "inventory.json"
            expected = (Path(tmp) / "data").resolve()
            resolved = inventory_source_directory({"directory": "../data"}, inventory)
            self.assertEqual(resolved, expected)
            self.assertTrue(resolved.is_absolute())
            self.assertEqual(inventory_source_directory({"directory": str(expected)}, inventory), expected)
            with self.assertRaisesRegex(ValueError, "nonempty"):
                inventory_source_directory({"directory": ""}, inventory)

    def test_changed_inventory_or_dial_metadata_aborts_resume(self):
        expected = {"inventory_sha256": "inventory-A", "dial_metadata_sha256": "dials-A"}
        index = {"schema_name": SCHEMA_NAME, "extractor_and_physics_sha256": "code", **expected}
        validate_resume_metadata(index, expected, "code")
        for field in expected:
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, field):
                validate_resume_metadata(index, {**expected, field: "changed"}, "code")

    def test_same_size_cache_corruption_aborts_before_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, cache = Path(tmp) / "source.root", Path(tmp) / "part_000.npz"
            source.write_bytes(b"ROOT is never opened by the resume guard")
            cache.write_bytes(b"original-cache-bytes")
            stat = source.stat()
            hashes = {"inventory_sha256": "inventory-A", "dial_metadata_sha256": "dials-A"}
            record = {"schema_name": SCHEMA_NAME, "extractor_and_physics_sha256": "code",
                "source": str(source), "cache": str(cache), "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns, "cache_bytes": cache.stat().st_size,
                "cache_sha256": digest_file(cache), **hashes}
            validate_cached_record(record, source, cache, "code", hashes)
            before = cache.read_bytes()
            cache.write_bytes(bytes([before[0] ^ 1]) + before[1:])
            self.assertEqual(cache.stat().st_size, record["cache_bytes"])
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                validate_cached_record(record, source, cache, "code", hashes)

    def test_orphan_sidecar_cannot_bypass_metadata_guard(self):
        hashes = {"inventory_sha256": "inventory-A", "dial_metadata_sha256": "dials-A"}
        sidecar = {"schema_name": SCHEMA_NAME, "extractor_and_physics_sha256": "code", **hashes}
        # Validation must fail on metadata before accessing even source stats.
        with self.assertRaisesRegex(RuntimeError, "dial_metadata_sha256"):
            validate_cached_record(sidecar, Path("does-not-exist.root"), Path("absent.npz"),
                                   "code", {**hashes, "dial_metadata_sha256": "changed"})

    def test_missing_dials_expand_to_unit_without_modifying_bad_weights(self):
        lengths = np.array([[2, 0], [0, 3], [2, 3]])
        offsets = np.array([[0, 2], [0, 0], [0, 2]])
        raw = ak.Array([[2., -1.], [np.nan, 3., 4.], [1., 1., 1., 1., 1.]])
        expected = [[2, -1, 1, 1, 1], [1, 1, np.nan, 3, 4], [1, 1, 1, 1, 1]]
        np.testing.assert_allclose(expand_responses(lengths, offsets, raw, [2, 3]),
                                   expected, equal_nan=True)

    def test_bad_offsets_and_header_lengths_fail_closed(self):
        with self.assertRaisesRegex(RuntimeError, "Noncontiguous"):
            expand_responses(np.array([[2, 3]]), np.array([[0, 1]]), ak.Array([[1]*5]), [2, 3])
        with self.assertRaisesRegex(RuntimeError, "closure"):
            expand_responses(np.array([[2, 3]]), np.array([[0, 2]]), ak.Array([[1]*4]), [2, 3])
        with self.assertRaisesRegex(RuntimeError, "variation count"):
            expand_responses(np.array([[2, 2]]), np.array([[0, 2]]), ak.Array([[1]*4]), [2, 3])

    def test_reader_caps_total_reads_not_only_source_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bytes"
            path.write_bytes(b"12345678")
            reader = BoundedReader(path, cap=5)
            try:
                self.assertEqual(reader.read(3), b"123")
                reader.seek(0)
                with self.assertRaisesRegex(RuntimeError, "Bounded read"):
                    reader.read(3)
                self.assertEqual(reader.n, 3)
                self.assertEqual(reader.read(2), b"12")
            finally:
                reader.close()


if __name__ == "__main__":
    unittest.main()
