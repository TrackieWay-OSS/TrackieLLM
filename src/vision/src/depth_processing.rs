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
pub fn analyze_navigation_cues(
    point_cloud: &[Point3D],
    depth_map: &ffi::tk_vision_depth_map_t
) -> Option<NavigationCues> {
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

    let mut detected_vertical_changes = detect_curbs(&point_cloud, &ground_plane);

    Some(NavigationCues {
        traversability_grid,
        grid_dimensions: GRID_DIMS,
        detected_vertical_changes,
    })
}

/// Analyzes the point cloud to detect vertical, linear structures like curbs.
fn detect_curbs(point_cloud: &[Point3D], ground_plane: &ransac::Plane) -> Vec<VerticalChange> {
    const CURB_MIN_HEIGHT: f32 = 0.08; // 8cm
    const CURB_MAX_HEIGHT: f32 = 0.25; // 25cm
    const POINT_DENSITY_THRESHOLD: usize = 10;

    // Remove the ground plane to isolate obstacles
    let obstacle_points: Vec<Point3D> = point_cloud
        .iter()
        .filter(|p| {
            let height_from_plane = p.y - (-ground_plane.normal.x * p.x - ground_plane.normal.z * p.z + ground_plane.d) / ground_plane.normal.y;
            height_from_plane > CURB_MIN_HEIGHT
        })
        .cloned()
        .collect();

    if obstacle_points.is_empty() {
        return Vec::new();
    }

    // Grid the obstacle points to find dense vertical clusters
    let mut grid: std::collections::HashMap<(i32, i32), Vec<f32>> = std::collections::HashMap::new();
    for p in &obstacle_points {
        let grid_x = (p.x / 0.1).round() as i32;
        let grid_z = (p.z / 0.2).round() as i32;
        grid.entry((grid_x, grid_z)).or_default().push(p.y);
    }

    let mut detected_curbs = Vec::new();
    for ((gx, gz), ys) in grid.iter() {
        if ys.len() < POINT_DENSITY_THRESHOLD { continue; }

        let min_y = ys.iter().fold(f32::MAX, |a, &b| a.min(b));
        let max_y = ys.iter().fold(f32::MIN, |a, &b| a.max(b));
        let height = max_y - min_y;

        if height > CURB_MIN_HEIGHT && height < CURB_MAX_HEIGHT {
            // Found a potential curb segment
            detected_curbs.push(VerticalChange {
                height_m: height,
                status: GroundPlaneStatus::Obstacle,
                // Approximate the grid index from the 3D grid coordinates
                grid_index: (*gx as u32, *gz as u32),
            });
        }
    }

    // In a full implementation, we would merge adjacent curb segments into a single line.
    detected_curbs
}
