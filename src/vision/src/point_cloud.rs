/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/vision/point_cloud.rs
 *
 * This module provides functionalities for creating and processing 3D point clouds
 * from 2D depth maps. This is a fundamental step for advanced navigation analysis,
 * enabling robust ground plane detection and obstacle avoidance.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use nalgebra::Point3;
use super::ffi;
use std::slice;

/// Represents a point in 3D space.
pub type Point3D = Point3<f32>;

/// Unprojects a 2D depth map into a 3D point cloud.
///
/// This function transforms each pixel of the depth map into a 3D coordinate
/// in the camera's space using the camera's intrinsic parameters.
///
/// # Arguments
/// * `depth_map` - The depth map data from the depth estimator.
/// * `focal_length_x`, `focal_length_y` - Camera focal lengths.
/// * `principal_point_x`, `principal_point_y` - The camera's principal point (usually center of the image).
///
/// # Returns
/// A `Vec<Point3D>` representing the generated point cloud.
pub fn unproject_to_point_cloud(
    depth_map: &ffi::tk_vision_depth_map_t,
    focal_length_x: f32,
    focal_length_y: f32,
    principal_point_x: f32,
    principal_point_y: f32,
) -> Vec<Point3D> {
    let depth_data = unsafe {
        slice::from_raw_parts(depth_map.data, (depth_map.width * depth_map.height) as usize)
    };

    let mut point_cloud = Vec::with_capacity(depth_data.len());

    for y in 0..depth_map.height {
        for x in 0..depth_map.width {
            let index = (y * depth_map.width + x) as usize;
            let depth = depth_data[index];

            // Only unproject valid depth points
            if depth > 0.1 && depth < 100.0 {
                let x_3d = (x as f32 - principal_point_x) * depth / focal_length_x;
                // The Y-axis in camera coordinates typically points down, so we invert it for a more intuitive "up" direction.
                let y_3d = -(y as f32 - principal_point_y) * depth / focal_length_y;
                let z_3d = depth;

                point_cloud.push(Point3D::new(x_3d, y_3d, z_3d));
            }
        }
    }
    point_cloud
}