 I completed 5 web searches gathering information about:

  1. BRepNet architecture
  2. UV-Net feature extraction
  3. MFCAD dataset and GNN requirements
  4. CAD testing frameworks
  5. PyTorch Geometric CAD datasets

  The status says: "Research completed via 5 web searches, ready to compile definitive
  list with proper citations."

  So I need to compile and present a comprehensive definitive list of what CAD AI model
   architectures require, based on my research findings. Let me organize this
  information clearly with specific dimensional requirements and citations.

⏺ Based on my research into CAD AI frameworks and models, here's the definitive list of
   what CAD-based AI architectures require:

  CAD AI Model Data Requirements - Definitive List

  1. Graph Structure Requirements

  PyTorch Geometric Data Object (Industry Standard):
  torch_geometric.data.Data(
      x=node_features,           # [num_nodes, node_feature_dim]
      edge_index=edge_connectivity,  # [2, num_edges]
      edge_attr=edge_features,   # [num_edges, edge_feature_dim]
      y=labels,                  # [num_nodes] or [num_graphs]
      face=face_indices          # [3, num_faces] for mesh data
  )

  1. Node Feature Requirements

  From BRepNet + UV-Net Architecture:

  Face Features (Primary Nodes):

- UV-Grid Representation: 10 × 10 × 7 dimensional per face
  - Point coordinates (x, y, z): 3 channels
  - Normal vectors (nx, ny, nz): 3 channels
  - Trimming mask: 1 channel
- Geometric Attributes:
  - Surface type (plane, cylinder, cone, sphere, torus, BSpline, etc.)
  - Surface area
  - Centroid coordinates
  - Average normal vector
  - Bounding box dimensions
  - Curvature statistics (mean, Gaussian)

  From AAGNet (gAAG Representation):

- Surface type encoding (categorical: 8-12 types)
- Area (normalized)
- Centroid (x, y, z)
- Normal vector (nx, ny, nz)
- Extended attributes: convexity, orientation

  Typical Dimensions: 24-64 features per node

  1. Edge Feature Requirements

  From BRepNet + UV-Net:

  Edge UV-Grid: 10 × 6 dimensional per edge

- Point coordinates (x, y, z): 3 channels
- Tangent vectors (tx, ty, tz): 3 channels

  From AAGNet:

- Curve type (line, circle, ellipse, BSpline, etc.)
- Edge length
- Convexity angle (dihedral angle between adjacent faces)
- Tangent vector at midpoint
- Curvature

  Typical Dimensions: 6-16 features per edge

  1. Topology Representation Requirements

  From BRepNet (Coedge Convolutions):

- Coedge graph structure preserving oriented topology
- Message passing along face-edge-face relationships
- Dual graph representation (both face graph and edge graph)

  From AAGNet (Geometric Attributed Adjacency Graph):

- Face adjacency graph (primary)
- Preserves topological connectivity
- Geometric and extended attributes on nodes/edges

  Graph Properties to Track:

- Number of nodes (faces/edges/vertices)
- Number of edges (adjacency relationships)
- Graph density
- Connected components
- Degree distribution
- Clustering coefficient

  1. Dataset Format Standards

  MFCAD++ (Machining Feature Recognition):

- 60,000+ CAD models
- Format: STEP files + segmentation labels
- 25 machining feature classes
- Face-level labels for manufacturing features

  WHUCAD (CAD Assembly Dataset):

- 144,016 CAD models
- Format: STL meshes + feature labels
- Assembly relationship annotations

  ABC Dataset (Large-Scale CAD):

- 1 million CAD models
- Format: STEP files + various representations
- Chunk-based organization

  DeepCAD:

- CAD construction sequences
- Parametric modeling operations

  1. Required Visualizations for CAD AI

  From IntelliCAD Testing Framework:

- Entity type distribution (pie charts, histograms)
- Topology connectivity graphs (showing face-edge relationships)
- Feature attribution heatmaps (for segmentation results)
- UV-grid visualizations (showing geometric sampling)
- Dihedral angle distributions (edge convexity)
- Curvature maps (surface analysis)

  For Functional Testing:

- CAD entity relationship diagrams showing type-to-type references
- BRep topology breakdown (shells → faces → edges → vertices)
- Reference pattern matrices (which entities reference which)
- Degree distribution for graph structure validation

  1. Specific Data Requirements Summary

  Minimum Required for CAD GNN Models:
  ┌─────────────────┬───────────┬────────────────┐
  │    Component    │ Dimension │     Source     │
  ├─────────────────┼───────────┼────────────────┤
  │ Face UV-Grid    │ 10×10×7   │ UV-Net         │
  ├─────────────────┼───────────┼────────────────┤
  │ Edge UV-Grid    │ 10×6      │ UV-Net         │
  ├─────────────────┼───────────┼────────────────┤
  │ Node Features   │ 24-64     │ BRepNet/AAGNet │
  ├─────────────────┼───────────┼────────────────┤
  │ Edge Features   │ 6-16      │ BRepNet/AAGNet │
  ├─────────────────┼───────────┼────────────────┤
  │ Graph Structure │ PyG Data  │ All frameworks │
  └─────────────────┴───────────┴────────────────┘
  Critical Geometric Properties:

  1. Dihedral angles (0 to π radians) - convexity between faces
  2. Surface types (categorical encoding) - geometric primitives
  3. Curvature (Gaussian, mean) - surface analysis
  4. Normals (unit vectors) - orientation information
  5. Area/Length (normalized) - size information

  6. Implementation Recommendations

  For our functional tests to match industry standards, we should output:

  1. PyTorch Geometric Data objects with proper feature dimensions
  2. Entity relationship visualizations showing STEP entity references
  3. UV-grid extractions for faces and edges (10×10×7 and 10×6)
  4. Feature statistics validating geometric properties are real:
    - Dihedral angle distributions (should peak around 90° for mechanical parts)
    - Curvature distributions (should show peaks for planes, cylinders)
    - Surface type distributions (should match CAD modeling patterns)
  5. Topology summaries matching BRep hierarchy (shells → faces → edges → vertices)
  6. Graph statistics suitable for GNN training:
    - Degree distributions
    - Connected components
    - Graph density
    - Clustering coefficients

  7. Key Citations

- BRepNet: <https://github.com/AutodeskAILab/BRepNet>, <https://arxiv.org/abs/2104.00706>
- UV-Net: <https://github.com/AutodeskAILab/UV-Net>,
  <https://ar5iv.labs.arxiv.org/html/2006.10211>
- AAGNet: <https://github.com/whjdark/AAGNet>
- MFCAD++: <https://pure.qub.ac.uk/en/datasets/mfcad-dataset-dataset-for-paper-hierarc>
  hical-cadnet-learning-from/
- PyTorch Geometric: Standard framework for all CAD GNN models

  ---
  Now I should update our visualization.py to generate outputs that match these
  industry standards. Should I proceed with updating the visualizations to include:

  1. UV-grid extraction and visualization (10×10×7 for faces, 10×6 for edges)?
  2. Feature validation plots (dihedral angle distributions, curvature histograms)?
  3. PyTorch Geometric Data object export with proper feature dimensions?
  4. BRep hierarchy visualization (shells → faces → edges → vertices breakdown)?

✻ Cogitated for 5m 50s
