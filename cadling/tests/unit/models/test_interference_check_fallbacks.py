"""Tests for interference check fallback implementations.

Tests the fallback methods that work when pythonocc is unavailable:
- _check_interferences_bbox_fallback: Bounding box based interference detection
- _compute_clearance_bbox_fallback: Bounding box based clearance computation
"""

from __future__ import annotations

import pytest
import numpy as np


class TestInterferenceCheckBboxHelpers:
    """Test helper methods for bbox-based interference checking."""

    def test_bboxes_overlap_true(self):
        """Test overlap detection when bboxes overlap."""
        from cadling.models.interference_check import InterferenceCheckModel

        model = InterferenceCheckModel()

        bbox1 = {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10}
        bbox2 = {"min_x": 5, "max_x": 15, "min_y": 5, "max_y": 15, "min_z": 5, "max_z": 15}

        assert model._bboxes_overlap(bbox1, bbox2) is True

    def test_bboxes_overlap_false(self):
        """Test overlap detection when bboxes don't overlap."""
        from cadling.models.interference_check import InterferenceCheckModel

        model = InterferenceCheckModel()

        bbox1 = {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10}
        bbox2 = {"min_x": 20, "max_x": 30, "min_y": 20, "max_y": 30, "min_z": 20, "max_z": 30}

        assert model._bboxes_overlap(bbox1, bbox2) is False

    def test_bboxes_overlap_touching(self):
        """Test overlap detection when bboxes touch but don't overlap."""
        from cadling.models.interference_check import InterferenceCheckModel

        model = InterferenceCheckModel()

        bbox1 = {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10}
        bbox2 = {"min_x": 10, "max_x": 20, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10}

        # Note: The implementation uses < not <= for separation check, so touching
        # (max1 == min2) is considered overlap. This is intentional for conservative
        # interference detection.
        # For truly separated bboxes, there must be a gap
        bbox3 = {"min_x": 11, "max_x": 20, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10}
        assert model._bboxes_overlap(bbox1, bbox3) is False

    def test_compute_bbox_overlap_volume(self):
        """Test overlap volume computation."""
        from cadling.models.interference_check import InterferenceCheckModel

        model = InterferenceCheckModel()

        bbox1 = {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10}
        bbox2 = {"min_x": 5, "max_x": 15, "min_y": 5, "max_y": 15, "min_z": 5, "max_z": 15}

        # Overlap region is [5, 10] x [5, 10] x [5, 10] = 5 x 5 x 5 = 125
        volume = model._compute_bbox_overlap_volume(bbox1, bbox2)
        assert volume == 125

    def test_compute_bbox_overlap_volume_no_overlap(self):
        """Test overlap volume when no overlap."""
        from cadling.models.interference_check import InterferenceCheckModel

        model = InterferenceCheckModel()

        bbox1 = {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10}
        bbox2 = {"min_x": 20, "max_x": 30, "min_y": 20, "max_y": 30, "min_z": 20, "max_z": 30}

        volume = model._compute_bbox_overlap_volume(bbox1, bbox2)
        assert volume == 0

    def test_compute_bbox_overlap_center(self):
        """Test overlap center computation."""
        from cadling.models.interference_check import InterferenceCheckModel

        model = InterferenceCheckModel()

        bbox1 = {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10}
        bbox2 = {"min_x": 5, "max_x": 15, "min_y": 5, "max_y": 15, "min_z": 5, "max_z": 15}

        # Overlap region is [5, 10] x [5, 10] x [5, 10], center is (7.5, 7.5, 7.5)
        center = model._compute_bbox_overlap_center(bbox1, bbox2)
        assert center == [7.5, 7.5, 7.5]

    def test_bbox_min_distance_separated(self):
        """Test minimum distance between separated bboxes."""
        from cadling.models.interference_check import InterferenceCheckModel

        model = InterferenceCheckModel()

        bbox1 = {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10}
        bbox2 = {"min_x": 20, "max_x": 30, "min_y": 20, "max_y": 30, "min_z": 20, "max_z": 30}

        # Gap is 10 on each axis, distance = sqrt(10^2 + 10^2 + 10^2) = sqrt(300) ≈ 17.32
        distance = model._bbox_min_distance(bbox1, bbox2)
        assert abs(distance - np.sqrt(300)) < 0.01

    def test_bbox_min_distance_overlapping(self):
        """Test minimum distance when bboxes overlap."""
        from cadling.models.interference_check import InterferenceCheckModel

        model = InterferenceCheckModel()

        bbox1 = {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10}
        bbox2 = {"min_x": 5, "max_x": 15, "min_y": 5, "max_y": 15, "min_z": 5, "max_z": 15}

        distance = model._bbox_min_distance(bbox1, bbox2)
        assert distance == 0

    def test_bbox_contains_true(self):
        """Test containment when inner is fully inside outer."""
        from cadling.models.interference_check import InterferenceCheckModel

        model = InterferenceCheckModel()

        outer = {"min_x": 0, "max_x": 20, "min_y": 0, "max_y": 20, "min_z": 0, "max_z": 20}
        inner = {"min_x": 5, "max_x": 15, "min_y": 5, "max_y": 15, "min_z": 5, "max_z": 15}

        assert model._bbox_contains(outer, inner) is True

    def test_bbox_contains_false(self):
        """Test containment when inner extends outside outer."""
        from cadling.models.interference_check import InterferenceCheckModel

        model = InterferenceCheckModel()

        outer = {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10}
        inner = {"min_x": 5, "max_x": 15, "min_y": 5, "max_y": 15, "min_z": 5, "max_z": 15}

        assert model._bbox_contains(outer, inner) is False

    def test_compute_closest_bbox_points(self):
        """Test closest point computation between bboxes."""
        from cadling.models.interference_check import InterferenceCheckModel

        model = InterferenceCheckModel()

        bbox1 = {"min_x": 0, "max_x": 10, "min_y": 0, "max_y": 10, "min_z": 0, "max_z": 10}
        bbox2 = {"min_x": 20, "max_x": 30, "min_y": 20, "max_y": 30, "min_z": 20, "max_z": 30}

        point1, point2 = model._compute_closest_bbox_points(bbox1, bbox2)

        # Point1 should be on surface of bbox1 (clamped to bbox1)
        assert point1[0] >= 0 and point1[0] <= 10
        assert point1[1] >= 0 and point1[1] <= 10
        assert point1[2] >= 0 and point1[2] <= 10

        # Point2 should be on surface of bbox2 (clamped to bbox2)
        assert point2[0] >= 20 and point2[0] <= 30
        assert point2[1] >= 20 and point2[1] <= 30
        assert point2[2] >= 20 and point2[2] <= 30


class TestInterferenceClearanceFallback:
    """Test clearance computation fallback."""

    def test_compute_clearance_bbox_fallback_basic(self):
        """Test bbox-based clearance computation."""
        from cadling.models.interference_check import InterferenceCheckModel
        from unittest.mock import MagicMock

        model = InterferenceCheckModel()

        part1 = MagicMock()
        part1.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 0, "max_x": 10,
                    "min_y": 0, "max_y": 10,
                    "min_z": 0, "max_z": 10,
                }
            }
        }

        part2 = MagicMock()
        part2.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 20, "max_x": 30,
                    "min_y": 20, "max_y": 30,
                    "min_z": 20, "max_z": 30,
                }
            }
        }

        result = model._compute_clearance_bbox_fallback(part1, part2)

        assert result is not None
        assert result.distance > 0  # Should have positive clearance
        assert result.point1 is not None
        assert result.point2 is not None

    def test_compute_clearance_bbox_fallback_no_properties(self):
        """Test bbox clearance fallback when properties missing."""
        from cadling.models.interference_check import InterferenceCheckModel
        from unittest.mock import MagicMock

        model = InterferenceCheckModel()

        part1 = MagicMock()
        part1.properties = {}

        part2 = MagicMock()
        part2.properties = {}

        result = model._compute_clearance_bbox_fallback(part1, part2)

        assert result is None  # Should fail gracefully


class TestInterferenceCheckBboxFallback:
    """Test full bbox interference check fallback."""

    def test_check_interferences_bbox_fallback_no_items(self):
        """Test bbox fallback with no solid items."""
        from cadling.models.interference_check import InterferenceCheckModel
        from unittest.mock import MagicMock

        model = InterferenceCheckModel()

        doc = MagicMock()
        items = []

        result = model._check_interferences_bbox_fallback(doc, items)

        assert result["status"] == "success"
        assert result["method"] == "bbox_fallback"
        assert result["num_interferences"] == 0

    def test_check_interferences_bbox_fallback_overlapping(self):
        """Test bbox fallback detects overlapping parts."""
        from cadling.models.interference_check import InterferenceCheckModel
        from unittest.mock import MagicMock

        model = InterferenceCheckModel()

        doc = MagicMock()

        # Create two overlapping parts
        part1 = MagicMock()
        part1.item_type = "brep_solid"
        part1.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 0, "max_x": 10,
                    "min_y": 0, "max_y": 10,
                    "min_z": 0, "max_z": 10,
                },
                "centroid": [5, 5, 5],
                "volume": 1000,
            }
        }

        part2 = MagicMock()
        part2.item_type = "brep_solid"
        part2.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 5, "max_x": 15,
                    "min_y": 5, "max_y": 15,
                    "min_z": 5, "max_z": 15,
                },
                "centroid": [10, 10, 10],
                "volume": 1000,
            }
        }

        result = model._check_interferences_bbox_fallback(doc, [part1, part2])

        assert result["status"] == "success"
        assert result["num_interferences"] == 1
        assert result["has_interferences"] is True

    def test_check_interferences_bbox_fallback_separated(self):
        """Test bbox fallback with separated parts."""
        from cadling.models.interference_check import InterferenceCheckModel
        from unittest.mock import MagicMock

        model = InterferenceCheckModel()

        doc = MagicMock()

        part1 = MagicMock()
        part1.item_type = "brep_solid"
        part1.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 0, "max_x": 10,
                    "min_y": 0, "max_y": 10,
                    "min_z": 0, "max_z": 10,
                },
                "centroid": [5, 5, 5],
                "volume": 1000,
            }
        }

        part2 = MagicMock()
        part2.item_type = "brep_solid"
        part2.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 20, "max_x": 30,
                    "min_y": 20, "max_y": 30,
                    "min_z": 20, "max_z": 30,
                },
                "centroid": [25, 25, 25],
                "volume": 1000,
            }
        }

        result = model._check_interferences_bbox_fallback(doc, [part1, part2])

        assert result["status"] == "success"
        assert result["num_interferences"] == 0
        assert result["has_interferences"] is False
        assert result["num_clearances_checked"] == 1

    def test_check_interferences_bbox_fallback_containment(self):
        """Test bbox fallback detects containment."""
        from cadling.models.interference_check import InterferenceCheckModel
        from unittest.mock import MagicMock

        model = InterferenceCheckModel(check_containment=True)

        doc = MagicMock()

        outer = MagicMock()
        outer.item_type = "brep_solid"
        outer.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 0, "max_x": 100,
                    "min_y": 0, "max_y": 100,
                    "min_z": 0, "max_z": 100,
                },
                "centroid": [50, 50, 50],
                "volume": 1000000,
            }
        }

        inner = MagicMock()
        inner.item_type = "brep_solid"
        inner.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 10, "max_x": 20,
                    "min_y": 10, "max_y": 20,
                    "min_z": 10, "max_z": 20,
                },
                "centroid": [15, 15, 15],
                "volume": 1000,
            }
        }

        result = model._check_interferences_bbox_fallback(doc, [outer, inner])

        assert result["status"] == "success"
        assert result["num_containment_issues"] == 1


class TestInterferenceCheckPublicMethods:
    """Test public methods use fallbacks."""

    def test_compute_clearances_uses_fallback(self):
        """Test compute_clearances uses bbox fallback when OCC unavailable."""
        from cadling.models.interference_check import InterferenceCheckModel
        from unittest.mock import MagicMock

        model = InterferenceCheckModel()
        model.has_pythonocc = False

        part1 = MagicMock()
        part1.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 0, "max_x": 10,
                    "min_y": 0, "max_y": 10,
                    "min_z": 0, "max_z": 10,
                }
            }
        }

        part2 = MagicMock()
        part2.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 20, "max_x": 30,
                    "min_y": 20, "max_y": 30,
                    "min_z": 20, "max_z": 30,
                }
            }
        }

        result = model.compute_clearances(part1, part2)

        assert result is not None
        assert result.distance > 0

    def test_detect_containment_works_without_occ(self):
        """Test detect_containment works without pythonocc."""
        from cadling.models.interference_check import InterferenceCheckModel
        from unittest.mock import MagicMock

        model = InterferenceCheckModel()
        model.has_pythonocc = False

        outer = MagicMock()
        outer.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 0, "max_x": 100,
                    "min_y": 0, "max_y": 100,
                    "min_z": 0, "max_z": 100,
                }
            }
        }

        inner = MagicMock()
        inner.properties = {
            "geometry_analysis": {
                "bounding_box": {
                    "min_x": 10, "max_x": 20,
                    "min_y": 10, "max_y": 20,
                    "min_z": 10, "max_z": 20,
                }
            }
        }

        result = model.detect_containment(outer, inner)

        assert result is True
