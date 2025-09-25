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

/// Builds a semantic scene graph from a list of enriched objects.
///
/// # Arguments
/// * `objects` - A slice of `EnrichedObject`s that have been fused with depth data.
///
/// # Returns
/// A `SceneGraph` representing the spatial relationships between the objects.
pub fn build_scene_graph(objects: &[EnrichedObject]) -> SceneGraph {
    let mut graph = SceneGraph::default();

    // 1. Create a node for each object
    for obj in objects {
        // We only add objects with valid 3D information to the graph
        if obj.distance_meters > 0.0 {
            graph.nodes.push(Node {
                label: "unknown".to_string(), // The label is on the C side, needs to be passed in
                class_id: obj.class_id,
                // Approximate 3D center
                center_3d: Point3::new(
                    (obj.bbox.x as f32 + obj.bbox.w as f32 / 2.0), // These are still 2D coords
                    (obj.bbox.y as f32 + obj.bbox.h as f32 / 2.0),
                    obj.distance_meters
                ),
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
            // TODO: Infer "NextTo" relationship
        }
    }

    graph
}

/// Checks if node A is on top of node B.
fn is_on_top_of(a: &Node, b: &Node) -> bool {
    let bbox_a = a.bbox_2d;
    let bbox_b = b.bbox_2d;

    // Check for vertical alignment: bottom of A is near the top of B
    let vertical_alignment = (bbox_a.1 + bbox_a.3) > (bbox_b.1 - 10) && (bbox_a.1 + bbox_a.3) < (bbox_b.1 + 10);

    // Check for horizontal overlap
    let horizontal_overlap = (bbox_a.0 > bbox_b.0) && ((bbox_a.0 + bbox_a.2) < (bbox_b.0 + bbox_b.2));

    // Check that A is closer (smaller Z value) than B, assuming Z is distance
    let closer = a.center_3d.z < b.center_3d.z;

    vertical_alignment && horizontal_overlap && closer
}