"""STL-based 3D comparison helpers for the viewer."""

import io
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import trimesh
from streamlit.runtime.uploaded_file_manager import UploadedFile


@dataclass(frozen=True)
class MeshComparisonSummary:
    """Mesh-level summary metrics for quick pre/post comparison."""

    pre_vertices: int
    post_vertices: int
    pre_faces: int
    post_faces: int
    pre_volume: float
    post_volume: float
    volume_delta_percent: float


def load_stl_mesh(uploaded_file: UploadedFile) -> trimesh.Trimesh:
    """Load one STL from a Streamlit upload and return a cleaned Trimesh."""
    mesh = trimesh.load(io.BytesIO(uploaded_file.getvalue()), file_type="stl")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Expected a single STL mesh.")
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError("STL mesh is empty.")
    return mesh


def build_mesh_comparison_summary(pre_mesh: trimesh.Trimesh, post_mesh: trimesh.Trimesh) -> MeshComparisonSummary:
    """Compute simple pre/post mesh metrics for the UI."""
    pre_volume = float(abs(pre_mesh.volume))
    post_volume = float(abs(post_mesh.volume))
    if pre_volume == 0:
        volume_delta_percent = 0.0
    else:
        volume_delta_percent = ((post_volume - pre_volume) / pre_volume) * 100

    return MeshComparisonSummary(
        pre_vertices=int(pre_mesh.vertices.shape[0]),
        post_vertices=int(post_mesh.vertices.shape[0]),
        pre_faces=int(pre_mesh.faces.shape[0]),
        post_faces=int(post_mesh.faces.shape[0]),
        pre_volume=pre_volume,
        post_volume=post_volume,
        volume_delta_percent=volume_delta_percent,
    )


def build_overlay_figure(pre_mesh: trimesh.Trimesh, post_mesh: trimesh.Trimesh) -> go.Figure:
    """Build an interactive Plotly overlay of pre/post meshes."""
    pre_vertices = pre_mesh.vertices
    pre_faces = pre_mesh.faces
    post_vertices = post_mesh.vertices
    post_faces = post_mesh.faces

    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=pre_vertices[:, 0],
            y=pre_vertices[:, 1],
            z=pre_vertices[:, 2],
            i=pre_faces[:, 0],
            j=pre_faces[:, 1],
            k=pre_faces[:, 2],
            color="royalblue",
            opacity=0.35,
            name="Pre-treatment",
        )
    )
    figure.add_trace(
        go.Mesh3d(
            x=post_vertices[:, 0],
            y=post_vertices[:, 1],
            z=post_vertices[:, 2],
            i=post_faces[:, 0],
            j=post_faces[:, 1],
            k=post_faces[:, 2],
            color="orangered",
            opacity=0.35,
            name="Post-treatment",
        )
    )
    figure.update_layout(
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        title="STL Overlay Prototype",
    )
    return figure


def load_fusion_points(uploaded_file: UploadedFile, vertex_count: int) -> pd.DataFrame:
    """Load vertex-level fusion points from CSV with boundary checks."""
    data_frame = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
    data_frame.columns = [column.strip().lower() for column in data_frame.columns]
    required_columns = {"vertex_index", "segment"}
    missing_columns = sorted(required_columns - set(data_frame.columns))
    if missing_columns:
        raise ValueError("Fusion CSV missing required columns: " + ", ".join(missing_columns))

    data_frame["vertex_index"] = pd.to_numeric(data_frame["vertex_index"], errors="coerce")
    if data_frame["vertex_index"].isna().any():
        raise ValueError("Fusion CSV has invalid vertex_index values.")

    data_frame["vertex_index"] = data_frame["vertex_index"].astype(int)
    if ((data_frame["vertex_index"] < 0) | (data_frame["vertex_index"] >= vertex_count)).any():
        raise ValueError("Fusion CSV vertex_index is out of mesh bounds.")

    if "change" in data_frame.columns:
        data_frame["change"] = pd.to_numeric(data_frame["change"], errors="coerce")
    return data_frame


def build_fusion_figure(
    post_mesh: trimesh.Trimesh,
    fusion_points_data_frame: pd.DataFrame,
    color_mode: str,
) -> go.Figure:
    """Build anatomical fusion visualization as points over the post mesh."""
    vertices = post_mesh.vertices
    faces = post_mesh.faces
    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="lightgray",
            opacity=0.25,
            name="Post mesh",
            showscale=False,
        )
    )

    selected_vertices = vertices[fusion_points_data_frame["vertex_index"].to_numpy()]
    if color_mode == "Change magnitude" and "change" in fusion_points_data_frame.columns:
        figure.add_trace(
            go.Scatter3d(
                x=selected_vertices[:, 0],
                y=selected_vertices[:, 1],
                z=selected_vertices[:, 2],
                mode="markers",
                marker={
                    "size": 3,
                    "color": fusion_points_data_frame["change"],
                    "colorscale": "RdYlGn_r",
                    "showscale": True,
                    "colorbar": {"title": "Change"},
                },
                name="Fusion points",
            )
        )
    else:
        for segment_name, segment_rows in fusion_points_data_frame.groupby("segment"):
            segment_vertices = vertices[segment_rows["vertex_index"].to_numpy()]
            figure.add_trace(
                go.Scatter3d(
                    x=segment_vertices[:, 0],
                    y=segment_vertices[:, 1],
                    z=segment_vertices[:, 2],
                    mode="markers",
                    marker={"size": 3},
                    name=str(segment_name).title(),
                )
            )

    figure.update_layout(
        scene={
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
            "aspectmode": "data",
        },
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        title="Anatomical Fusion Prototype",
        legend={"itemsizing": "constant"},
    )
    return figure
