"""
UV-Grid Extraction Module

Extracts industry-standard UV-grids from CAD faces and edges following the UV-Net specification:
- Face UV-grids: 10×10×7 (point coords, normals, trimming mask)
- Edge UV-grids: 10×6 (point coords, tangent vectors)

Based on research from UV-Net (Autodesk AI Lab):
https://github.com/AutodeskAILab/UV-Net
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Add resources/occwl to path for imports
OCCWL_PATH = Path(__file__).parent.parent.parent.parent.parent / "resources" / "occwl" / "src"
if str(OCCWL_PATH) not in sys.path:
    sys.path.insert(0, str(OCCWL_PATH))

try:
    from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Edge
    from OCC.Core.TopAbs import TopAbs_IN, TopAbs_OUT
    from occwl.face import Face
    from occwl.edge import Edge
    from occwl.uvgrid import uvgrid, ugrid
    HAS_OCC = True
except ImportError:
    HAS_OCC = False
    logging.warning("OpenCASCADE (pythonocc-core) not available. UV-grid extraction will be disabled.")


logger = logging.getLogger(__name__)


class FaceUVGridExtractor:
    """Extract 10×10×7 UV-grids from faces following UV-Net specification.

    UV-grid format:
    - Shape: [10, 10, 7]
    - Channels 0-2: Point coordinates (x, y, z)
    - Channels 3-5: Normal vectors (nx, ny, nz)
    - Channel 6: Trimming mask (0=outside trimmed region, 1=inside)
    """

    @staticmethod
    def extract_uv_grid(
        occ_face: TopoDS_Face,
        num_u: int = 10,
        num_v: int = 10
    ) -> Optional[np.ndarray]:
        """
        Extract UV-grid from a single face.

        Args:
            occ_face: OpenCASCADE TopoDS_Face object
            num_u: Number of samples in U direction (default: 10)
            num_v: Number of samples in V direction (default: 10)

        Returns:
            np.ndarray of shape [num_u, num_v, 7] or None if extraction fails
            - Channels 0-2: Point coordinates (x, y, z)
            - Channels 3-5: Normal vectors (nx, ny, nz)
            - Channel 6: Trimming mask (0 or 1)
        """
        if not HAS_OCC:
            logger.error("OpenCASCADE not available")
            return None

        try:
            # Wrap in occwl.Face for easier manipulation
            face = Face(occ_face)

            # Extract points using uvgrid function
            points_grid = uvgrid(face, num_u=num_u, num_v=num_v, method="point")
            if points_grid is None:
                logger.warning("Failed to extract point grid from face")
                return None

            # Extract normals using uvgrid function
            normals_grid = uvgrid(face, num_u=num_u, num_v=num_v, method="normal")
            if normals_grid is None:
                logger.warning("Failed to extract normal grid from face")
                return None

            # Compute trimming mask
            # Get UV values where we sampled
            _, uv_values = uvgrid(face, num_u=num_u, num_v=num_v, method="point", uvs=True)

            # Create trimming mask by checking if each UV point is inside the trimmed region
            trimming_mask = np.zeros((num_u, num_v, 1), dtype=np.float32)
            for i in range(num_u):
                for j in range(num_v):
                    uv = uv_values[i, j]
                    # Use visibility_status: 0=IN, 1=OUT, 2=ON, 3=UNKNOWN
                    status = face.visibility_status(uv)
                    # Mark as 1 if inside (TopAbs_IN = 0)
                    trimming_mask[i, j, 0] = 1.0 if status == TopAbs_IN else 0.0

            # Stack into [num_u, num_v, 7] array
            uv_grid = np.concatenate([
                points_grid,      # [num_u, num_v, 3]
                normals_grid,     # [num_u, num_v, 3]
                trimming_mask     # [num_u, num_v, 1]
            ], axis=2)

            return uv_grid.astype(np.float32)

        except Exception as e:
            logger.warning(f"Failed to extract UV-grid from face: {e}")
            return None

    @staticmethod
    def extract_batch_uv_grids(
        occ_faces: List[TopoDS_Face],
        num_u: int = 10,
        num_v: int = 10
    ) -> Dict[int, np.ndarray]:
        """
        Extract UV-grids for multiple faces with error handling.

        Args:
            occ_faces: List of OpenCASCADE TopoDS_Face objects
            num_u: Number of samples in U direction (default: 10)
            num_v: Number of samples in V direction (default: 10)

        Returns:
            Dict mapping face index to UV-grid array [10, 10, 7]
            Only includes faces where extraction succeeded
        """
        uv_grids = {}

        for idx, occ_face in enumerate(occ_faces):
            uv_grid = FaceUVGridExtractor.extract_uv_grid(occ_face, num_u, num_v)
            if uv_grid is not None:
                uv_grids[idx] = uv_grid
            else:
                logger.debug(f"Skipping face {idx} - UV-grid extraction failed")

        logger.info(f"Extracted UV-grids for {len(uv_grids)}/{len(occ_faces)} faces "
                   f"({len(uv_grids)/len(occ_faces)*100:.1f}% coverage)")

        return uv_grids


class EdgeUVGridExtractor:
    """Extract 10×6 UV-grids from edges following UV-Net specification.

    UV-grid format:
    - Shape: [10, 6]
    - Channels 0-2: Point coordinates (x, y, z)
    - Channels 3-5: Tangent vectors (tx, ty, tz)
    """

    @staticmethod
    def extract_uv_grid(
        occ_edge: TopoDS_Edge,
        num_u: int = 10
    ) -> Optional[np.ndarray]:
        """
        Extract UV-grid from a single edge.

        Args:
            occ_edge: OpenCASCADE TopoDS_Edge object
            num_u: Number of samples along the curve (default: 10)

        Returns:
            np.ndarray of shape [num_u, 6] or None if extraction fails
            - Channels 0-2: Point coordinates (x, y, z)
            - Channels 3-5: Tangent vectors (tx, ty, tz)
        """
        if not HAS_OCC:
            logger.error("OpenCASCADE not available")
            return None

        try:
            # Wrap in occwl.Edge for easier manipulation
            edge = Edge(occ_edge)

            # Extract points using ugrid function
            points_grid = ugrid(edge, num_u=num_u, method="point")
            if points_grid is None:
                logger.warning("Failed to extract point grid from edge")
                return None

            # Extract tangents using ugrid function
            tangents_grid = ugrid(edge, num_u=num_u, method="tangent")
            if tangents_grid is None:
                logger.warning("Failed to extract tangent grid from edge")
                return None

            # Stack into [num_u, 6] array
            uv_grid = np.concatenate([
                points_grid,    # [num_u, 3]
                tangents_grid   # [num_u, 3]
            ], axis=1)

            return uv_grid.astype(np.float32)

        except Exception as e:
            logger.warning(f"Failed to extract UV-grid from edge: {e}")
            return None

    @staticmethod
    def extract_batch_uv_grids(
        occ_edges: List[TopoDS_Edge],
        num_u: int = 10
    ) -> Dict[int, np.ndarray]:
        """
        Extract UV-grids for multiple edges with error handling.

        Args:
            occ_edges: List of OpenCASCADE TopoDS_Edge objects
            num_u: Number of samples along each curve (default: 10)

        Returns:
            Dict mapping edge index to UV-grid array [10, 6]
            Only includes edges where extraction succeeded
        """
        uv_grids = {}

        for idx, occ_edge in enumerate(occ_edges):
            uv_grid = EdgeUVGridExtractor.extract_uv_grid(occ_edge, num_u)
            if uv_grid is not None:
                uv_grids[idx] = uv_grid
            else:
                logger.debug(f"Skipping edge {idx} - UV-grid extraction failed")

        logger.info(f"Extracted UV-grids for {len(uv_grids)}/{len(occ_edges)} edges "
                   f"({len(uv_grids)/len(occ_edges)*100:.1f}% coverage)")

        return uv_grids
