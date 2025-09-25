/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/vision/scene_graph.rs
 *
 * This module is responsible for constructing a semantic scene graph from the
 * results of the vision pipeline. Instead of just a list of objects, it
 * produces a structured representation of the scene, capturing the relationships
 * between objects (e.g., "cup is on top of table").
 *
 * This is a critical component for enabling advanced reasoning and interaction.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::object_analysis::EnrichedObject;
use nalgebra::Point3;
use serde::{Serialize};

/// Represents a node in the scene graph, typically a detected object.
#[derive(Serialize, Debug)]
pub struct Node {
    pub label: String,
    pub class_id: u32,
    pub center_3d: Point3<f32>,
    pub bbox_2d: (i32, i32, i32, i32),
}

/// Represents the type of relationship between two nodes.
#[derive(Serialize, Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Relationship {
    OnTopOf,
    NextTo,
    // Future relationships can be added here, e.g., Inside, Facing
}

/// Represents a directed edge in the scene graph.
#[derive(Serialize, Debug)]
pub struct Edge {
    /// The index of the source node in the `nodes` vector.
    pub source_index: usize,
    /// The index of the target node in the `nodes` vector.
    pub target_index: usize,
    /// The type of relationship.
    pub relationship: Relationship,
}

/// Represents the entire scene as a graph structure.
#[derive(Serialize, Debug, Default)]
pub struct SceneGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

use super::point_cloud::Point3D;

/// Builds a semantic scene graph from a list of enriched objects.
///
/// # Arguments
/// * `objects` - A slice of `EnrichedObject`s that have been fused with depth data.
/// * `point_cloud` - The full 3D point cloud of the scene.
/// * `depth_map_width`, `depth_map_height` - Dimensions of the depth map for coordinate mapping.
///
/// # Returns
/// A `SceneGraph` representing the spatial relationships between the objects.
pub fn build_scene_graph(
    objects: &[EnrichedObject],
    point_cloud: &[Point3D],
    depth_map_width: u32,
    depth_map_height: u32,
) -> SceneGraph {
    let mut graph = SceneGraph::default();

    // 1. Create a node for each object
    for obj in objects {
        // We only add objects with valid 3D information to the graph
        if obj.distance_meters > 0.0 {
            // Find the 3D centroid of the points within the object's bounding box
            let points_in_bbox: Vec<&Point3D> = point_cloud.iter().filter(|p| {
                // This mapping is approximate and assumes the point cloud and image are aligned.
                // A more robust solution would use the camera projection matrix.
                let u = (p.x / p.z) * 500.0 + (depth_map_width as f32 / 2.0); // Simplified projection
                let v = (p.y / p.z) * 500.0 + (depth_map_height as f32 / 2.0);
                u >= obj.bbox.x as f32 && u < (obj.bbox.x + obj.bbox.w) as f32 &&
                v >= obj.bbox.y as f32 && v < (obj.bbox.y + obj.bbox.h) as f32
            }).collect();

            let centroid = if !points_in_bbox.is_empty() {
                points_in_bbox.iter().fold(Point3::origin(), |sum, p| sum + p.coords) / (points_in_bbox.len() as f32)
            } else {
                // Fallback to the old approximation if no points are found
                Point3::new(0.0, 0.0, obj.distance_meters)
            };

            graph.nodes.push(Node {
                label: "unknown".to_string(), // The label is on the C side, needs to be passed in
                class_id: obj.class_id,
                center_3d: centroid,
                bbox_2d: (obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h),
            });
        }
    }

    // 2. Infer relationships and create edges
    // This is a naive O(n^2) approach, which is fine for a small number of objects.
    for i in 0..graph.nodes.len() {
        for j in 0..graph.nodes.len() {
            if i == j { continue; }

            let node_a = &graph.nodes[i];
            let node_b = &graph.nodes[j];

            // Infer "OnTopOf" relationship
            // Condition: BBox of A is mostly inside BBox of B, and A is physically higher than B.
            if is_on_top_of(node_a, node_b) {
                graph.edges.push(Edge {
                    source_index: i,
                    target_index: j,
                    relationship: Relationship::OnTopOf,
                });
            }
            // Infer "NextTo" relationship
            if is_next_to(node_a, node_b) {
                graph.edges.push(Edge {
                    source_index: i,
                    target_index: j,
                    relationship: Relationship::NextTo,
                });
            }
        }
    }

    graph
}

/// Checks if node A is on top of node B.
fn is_on_top_of(a: &Node, b: &Node) -> bool {
    let (ax, ay, aw, ah) = a.bbox_2d;
    let (bx, by, bw, bh) = b.bbox_2d;

    // Check for vertical alignment (bottom of A is near the top of B)
    let vertical_alignment = (ay + ah) > (by - 10) && (ay + ah) < (by + 10);

    // Check for horizontal overlap
    let horizontal_overlap = (ax > bx) && ((ax + aw) < (bx + bw));

    // Check that A is physically higher than B (in 3D space)
    let is_higher = a.center_3d.y > b.center_3d.y;

    vertical_alignment && horizontal_overlap && is_higher
}

/// Checks if node A is next to node B.
fn is_next_to(a: &Node, b: &Node) -> bool {
    let (ax, ay, aw, ah) = a.bbox_2d;
    let (bx, by, bw, bh) = b.bbox_2d;

    // Check for vertical overlap in 2D (they are at a similar vertical level on screen)
    let y_overlap = (ay < by + bh) && (ay + ah > by);

    // Check for horizontal proximity in 2D (they are close to each other horizontally)
    let x_proximity = (ax + aw > bx - 20) && (ax < bx + bw + 20);

    // Check for similar height in 3D space
    let similar_height = (a.center_3d.y - b.center_3d.y).abs() < 0.2; // 20cm tolerance

    y_overlap && x_proximity && similar_height
}