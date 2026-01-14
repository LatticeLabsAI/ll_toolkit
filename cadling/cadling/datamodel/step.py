"""STEP format data models.

This module provides data models specific to STEP (ISO 10303-21) files,
including entity representations, topology graphs, and document structure.

Classes:
    STEPEntityItem: Individual STEP entity (CARTESIAN_POINT, CIRCLE, etc.)
    STEPHeader: STEP file header information
    STEPDocument: STEP-specific document structure
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from cadling.datamodel.base_models import (
    CADDocumentOrigin,
    CADItem,
    CADItemLabel,
    CADlingDocument,
    InputFormat,
)


class STEPEntityItem(CADItem):
    """STEP entity item.

    Represents a single STEP entity like CARTESIAN_POINT, CIRCLE,
    CYLINDRICAL_SURFACE, ADVANCED_FACE, etc.

    STEP entities have:
    - An ID (e.g., #31)
    - A type (e.g., "CARTESIAN_POINT")
    - Numeric parameters (e.g., coordinates [0.0, 0.0, 1.0])
    - Reference parameters (e.g., references to other entities [#15, #22])

    Attributes:
        item_type: Always "step_entity"
        entity_id: Entity ID number (e.g., 31 for #31)
        entity_type: Entity type name (e.g., "CARTESIAN_POINT")
        numeric_params: List of numeric parameters
        reference_params: List of entity ID references
        features: ll_stepnet extracted features (geometric properties)
        raw_text: Original STEP entity text
    """

    item_type: str = "step_entity"

    # STEP-specific fields
    entity_id: int
    entity_type: str
    numeric_params: List[float] = Field(default_factory=list)
    reference_params: List[int] = Field(default_factory=list)

    # ll_stepnet features
    features: Optional[Dict[str, Any]] = None

    # Raw representation
    raw_text: Optional[str] = None

    @classmethod
    def from_step_line(
        cls,
        entity_id: int,
        entity_type: str,
        line: str,
        features: Optional[Dict[str, Any]] = None,
    ) -> STEPEntityItem:
        """Create STEPEntityItem from a STEP file line.

        Args:
            entity_id: Entity ID number
            entity_type: Entity type name
            line: Raw STEP line
            features: Optional ll_stepnet features

        Returns:
            STEPEntityItem instance

        Example:
            item = STEPEntityItem.from_step_line(
                entity_id=31,
                entity_type="CARTESIAN_POINT",
                line="#31=CARTESIAN_POINT('',(0.,0.,1.));",
                features={"coordinates": [0.0, 0.0, 1.0]}
            )
        """
        return cls(
            entity_id=entity_id,
            entity_type=entity_type,
            label=CADItemLabel(text=f"#{entity_id} {entity_type}"),
            text=line,
            raw_text=line,
            features=features,
        )


# Type alias for STEP header information
# The STEP header contains metadata about the file, including:
# - FILE_DESCRIPTION
# - FILE_NAME
# - FILE_SCHEMA
# - Timestamps, authors, organizations, etc.
#
# Example:
#     header = {
#         "file_description": ["CAD Model"],
#         "file_name": "part.step",
#         "time_stamp": "2024-01-01T12:00:00",
#         "author": ["Engineer"],
#         "organization": ["Company"],
#         "file_schema": "AUTOMOTIVE_DESIGN"
#     }
STEPHeader = Dict[str, Any]


class STEPDocument(CADlingDocument):
    """STEP-specific document structure.

    Extends CADlingDocument with STEP-specific fields like header
    information and entity index for fast lookups.

    Attributes:
        format: Always InputFormat.STEP
        header: STEP header information
        entity_index: Dict mapping entity ID to STEPEntityItem
        step_schema: STEP schema name (e.g., "AUTOMOTIVE_DESIGN")
    """

    format: InputFormat = InputFormat.STEP

    # STEP-specific fields
    header: Optional[STEPHeader] = None
    entity_index: Dict[int, STEPEntityItem] = Field(default_factory=dict)
    step_schema: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def add_entity(self, entity: STEPEntityItem):
        """Add a STEP entity to the document.

        This also updates the entity_index for fast lookups.

        Args:
            entity: STEPEntityItem to add
        """
        self.add_item(entity)
        self.entity_index[entity.entity_id] = entity

    def get_entity(self, entity_id: int) -> Optional[STEPEntityItem]:
        """Get entity by ID.

        Args:
            entity_id: Entity ID number

        Returns:
            STEPEntityItem if found, None otherwise
        """
        return self.entity_index.get(entity_id)

    def get_entities_by_type(self, entity_type: str) -> List[STEPEntityItem]:
        """Get all entities of a specific type.

        Args:
            entity_type: Entity type name (e.g., "CARTESIAN_POINT")

        Returns:
            List of matching STEPEntityItem objects
        """
        return [
            item
            for item in self.items
            if isinstance(item, STEPEntityItem) and item.entity_type == entity_type
        ]

    def resolve_reference(self, entity_id: int) -> Optional[STEPEntityItem]:
        """Resolve an entity reference.

        Args:
            entity_id: Referenced entity ID

        Returns:
            Referenced STEPEntityItem if found, None otherwise
        """
        return self.get_entity(entity_id)
