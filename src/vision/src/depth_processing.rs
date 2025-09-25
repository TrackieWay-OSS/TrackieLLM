/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/vision/depth_processing.rs
 *
 * This file contains advanced analysis logic for depth maps, focusing on
 * extracting navigation-related cues for the TrackieLLM system. It is
 * responsible for identifying traversable ground, potential hazards like
 * holes and steps, and other features that can inform the Cortex about
 * the safety of the environment.
 *
 * This logic is designed to be called from the C-level pipeline via FFI.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::ffi;
use super::point_cloud::{self, Point3D};
use super::ransac;
use trackiellm_event_bus::{GroundPlaneStatus, NavigationCues, VerticalChange};
use std::slice;

// --- Constants for 3D Navigation Analysis ---
const GRID_DIMS: (u32, u32) = (5, 5);
const RANSAC_ITERATIONS: usize = 50;
const RANSAC_DISTANCE_THRESHOLD: f32 = 0.05; // 5cm for a point to be on the plane
const OBSTACLE_HEIGHT_THRESHOLD_M: f32 = 0.15; // 15cm above the plane is an obstacle
const HOLE_DEPTH_THRESHOLD_M: f32 = -0.10;     // 10cm below the plane is a hole

/// Analyzes a depth map to extract navigation cues using 3D point cloud analysis.
pub fn analyze_navigation_cues(depth_map: &ffi::tk_vision_depth_map_t) -> Option<NavigationCues> {
    if depth_map.data.is_null() || depth_map.width == 0 || depth_map.height == 0 {
        return None;
    }

    // --- 1. Unproject Depth Map to 3D Point Cloud ---
    // For now, assume camera intrinsics are fixed. In a real system, these would
    // be part of the configuration.
    let principal_point_x = depth_map.width as f32 / 2.0;
    let principal_point_y = depth_map.height as f32 / 2.0;
    let focal_length = 300.0; // A reasonable guess for a webcam-like camera

    let point_cloud = point_cloud::unproject_to_point_cloud(
        depth_map,
        focal_length,
        focal_length,
        principal_point_x,
        principal_point_y,
    );

    if point_cloud.is_empty() {
        return None;
    }

    // --- 2. Find the Ground Plane using RANSAC ---
    let ground_plane = ransac::find_plane_ransac(
        &point_cloud,
        RANSAC_ITERATIONS,
        RANSAC_DISTANCE_THRESHOLD,
    )?;

    // --- 3. Classify Grid Cells based on the Ground Plane ---
    let mut traversability_grid = vec![GroundPlaneStatus::Unknown; (GRID_DIMS.0 * GRID_DIMS.1) as usize];
    let cell_width_3d = (principal_point_x * 2.0 / focal_length * 5.0) / GRID_DIMS.0 as f32; // Approx 3D width of a cell at 5m
    let cell_depth_3d = 5.0 / GRID_DIMS.1 as f32; // Approx 3D depth of a cell

    for gy in 0..GRID_DIMS.1 {
        for gx in 0..GRID_DIMS.0 {
            let cell_center_x = (gx as f32 - (GRID_DIMS.0 as f32 / 2.0)) * cell_width_3d;
            let cell_center_z = (GRID_DIMS.1 - gy) as f32 * cell_depth_3d;

            let points_in_cell: Vec<&Point3D> = point_cloud
                .iter()
                .filter(|p| {
                    (p.x - cell_center_x).abs() < cell_width_3d / 2.0
                        && (p.z - cell_center_z).abs() < cell_depth_3d / 2.0
                })
                .collect();

            if points_in_cell.is_empty() { continue; }

            let avg_height_from_plane: f32 = points_in_cell
                .iter()
                .map(|p| p.y - (-ground_plane.normal.x * p.x - ground_plane.normal.z * p.z + ground_plane.d) / ground_plane.normal.y)
                .sum::<f32>() / points_in_cell.len() as f32;

            let grid_idx = (gy * GRID_DIMS.0 + gx) as usize;
            if avg_height_from_plane > OBSTACLE_HEIGHT_THRESHOLD_M {
                traversability_grid[grid_idx] = GroundPlaneStatus::Obstacle;
            } else if avg_height_from_plane < HOLE_DEPTH_THRESHOLD_M {
                traversability_grid[grid_idx] = GroundPlaneStatus::Hole;
            } else {
                traversability_grid[grid_idx] = GroundPlaneStatus::Flat;
            }
        }
    }

    Some(NavigationCues {
        traversability_grid,
        grid_dimensions: GRID_DIMS,
        detected_vertical_changes: Vec::new(), // To be implemented later
    })
}
