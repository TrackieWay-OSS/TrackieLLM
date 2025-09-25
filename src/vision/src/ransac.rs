/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/vision/ransac.rs
 *
 * This module provides a generic implementation of the RANSAC (RANdom SAmple
 * Consensus) algorithm for fitting models to data containing outliers. It is
 * specifically used here for robustly detecting the ground plane in a 3D
 * point cloud.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use nalgebra::{Vector3, Point3};
use rand::seq::SliceRandom;

/// Represents a 3D plane defined by its normal vector and distance from the origin.
/// The plane equation is `normal.dot(p) - d = 0`.
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub normal: Vector3<f32>,
    pub d: f32,
}

impl Plane {
    /// Creates a plane from three non-collinear points.
    pub fn from_points(p1: &Point3<f32>, p2: &Point3<f32>, p3: &Point3<f32>) -> Option<Self> {
        let v1 = p2 - p1;
        let v2 = p3 - p1;
        let mut normal = v1.cross(&v2);

        if normal.normalize_mut() == 0.0 {
            return None; // Points are collinear
        }

        // Ensure the normal vector points upwards (positive Y)
        if normal.y < 0.0 {
            normal = -normal;
        }

        let d = normal.dot(&p1.coords);
        Some(Plane { normal, d })
    }

    /// Calculates the perpendicular distance from a point to the plane.
    pub fn distance_to_point(&self, p: &Point3<f32>) -> f32 {
        (self.normal.dot(&p.coords) - self.d).abs()
    }
}

/// Finds the best-fit plane in a point cloud using the RANSAC algorithm.
///
/// # Arguments
/// * `points` - The input 3D point cloud.
/// * `max_iterations` - The number of RANSAC iterations to perform.
/// * `distance_threshold` - The maximum distance a point can be from the plane to be considered an inlier.
///
/// # Returns
/// An `Option<Plane>` representing the best plane found. Returns `None` if no suitable plane is found.
pub fn find_plane_ransac(
    points: &[Point3<f32>],
    max_iterations: usize,
    distance_threshold: f32,
) -> Option<Plane> {
    if points.len() < 3 {
        return None;
    }

    let mut best_plane = None;
    let mut max_inliers = 0;
    let mut rng = rand::thread_rng();

    for _ in 0..max_iterations {
        // 1. Randomly sample 3 points
        let sample = points.choose_multiple(&mut rng, 3).collect::<Vec<_>>();
        if sample.len() < 3 { continue; }

        // 2. Fit a plane to the sample
        if let Some(plane) = Plane::from_points(sample[0], sample[1], sample[2]) {
            // 3. Count inliers
            let inliers: Vec<&Point3<f32>> = points
                .iter()
                .filter(|p| plane.distance_to_point(p) < distance_threshold)
                .collect();

            // 4. Update best model if this one is better
            if inliers.len() > max_inliers {
                max_inliers = inliers.len();
                best_plane = Some(plane);
            }
        }
    }

    // Optional: Refit the plane using all inliers from the best model found
    // This can provide a more accurate final plane model.
    if let Some(plane) = best_plane {
        let all_inliers: Vec<&Point3<f32>> = points
            .iter()
            .filter(|p| plane.distance_to_point(p) < distance_threshold)
            .collect();

        if all_inliers.len() > max_inliers {
            // Refit logic would go here (e.g., using PCA or least squares on the inliers)
            // For now, we return the plane from the best sample.
        }
    }

    best_plane
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    #[test]
    fn test_ransac_finds_simple_plane() {
        // Create a synthetic point cloud representing a flat plane (y=0.5)
        // with some outliers.
        let mut points = Vec::new();
        for i in 0..100 {
            points.push(Point3::new((i as f32) / 10.0, 0.5, (i as f32 * 2.0) / 10.0));
        }
        // Add outliers
        points.push(Point3::new(1.0, 5.0, 1.0));
        points.push(Point3::new(2.0, -3.0, 2.0));
        points.push(Point3::new(3.0, 4.0, 3.0));

        let plane = find_plane_ransac(&points, 100, 0.1).expect("RANSAC should find a plane");

        // The normal vector should be very close to (0, 1, 0) or (0, -1, 0)
        assert!(plane.normal.y.abs() > 0.99);
        // The distance from origin should be close to 0.5
        assert!((plane.d.abs() - 0.5).abs() < 0.01);
    }
}