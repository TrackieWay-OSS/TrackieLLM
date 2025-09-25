/*
 * Copyright (C) 2025 Pedro Henrique / phdev13
 *
 * File: src/vision/object_analysis.rs
 *
 * This file contains the core data fusion logic for the vision pipeline,
 * written in safe Rust. Its primary responsibility is to take the raw outputs
 * from the object detector and the depth estimator and combine them into a
 * single, more meaningful representation: a list of "enriched" objects that
 * include an estimated distance.
 *
 * This logic is designed to be called from the C-level pipeline via FFI.
 *
 * SPDX-License-Identifier: AGPL-3.0 license
 */

use super::ffi;
use std::slice;
use std::collections::HashMap;
use std::sync::Mutex;

use uuid::Uuid;
use nalgebra::{Matrix1, Matrix1x2, Matrix2, Vector1, Vector2};
use kalman_rs::filter::KalmanFilter;

use lazy_static::lazy_static;

// --- Type definitions for our 1D Kalman Filter (tracking distance) ---
// State: [distance]
// Control: Not used
// Observation: [measured_distance]
type DistanceKalmanFilter = KalmanFilter<f32, 1, 0, 1>;
type State = Vector1<f32>;
type Control = nalgebra::allocator::Void; // No control input
type Observation = Vector1<f32>;

/// Represents a tracked object across multiple frames.
struct ObjectTracker {
    id: Uuid,
    class_id: u32,
    last_bbox: ffi::tk_rect_t,
    kf: DistanceKalmanFilter,
    frames_without_detection: u32,
}

// Global state to maintain trackers between FFI calls.
// This is a common pattern for adding state to a library that is
// primarily called from a stateless C API.
lazy_static! {
    static ref TRACKERS: Mutex<HashMap<Uuid, ObjectTracker>> = Mutex::new(HashMap::new());
}


/// Represents a detected object with its distance calculated.
/// This struct is C-compatible and will be returned to the C layer.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EnrichedObject {
    pub class_id: u32,
    pub confidence: f32,
    pub bbox: ffi::tk_rect_t,
    pub distance_meters: f32,
    pub width_meters: f32,
    pub height_meters: f32,
    pub is_partially_occluded: bool,
}

// --- Helper Functions for Tracking ---

/// Calculates the Intersection over Union (IoU) of two bounding boxes.
fn calculate_iou(box1: &ffi::tk_rect_t, box2: &ffi::tk_rect_t) -> f32 {
    let x_left = std::cmp::max(box1.x, box2.x);
    let y_top = std::cmp::max(box1.y, box2.y);
    let x_right = std::cmp::min(box1.x + box1.w, box2.x + box2.w);
    let y_bottom = std::cmp::min(box1.y + box1.h, box2.y + box2.h);

    if x_right < x_left || y_bottom < y_top {
        return 0.0;
    }

    let intersection_area = (x_right - x_left) as f32 * (y_bottom - y_top) as f32;
    let box1_area = (box1.w * box1.h) as f32;
    let box2_area = (box2.w * box2.h) as f32;
    let union_area = box1_area + box2_area - intersection_area;

    if union_area > 0.0 {
        intersection_area / union_area
    } else {
        0.0
    }
}

/// Creates a new ObjectTracker with a configured Kalman filter.
fn create_new_tracker(class_id: u32, bbox: ffi::tk_rect_t, initial_distance: f32) -> ObjectTracker {
    // A: State Transition Matrix. Assumes distance is constant between frames.
    let f_matrix = Matrix1::new(1.0);
    // B: Control Matrix. Not used.
    let b_matrix: Control = nalgebra::allocator::Void::new_allocator_for_value(0,0).reallocate_from_void(0,0);
    // H: Observation Matrix. We directly observe the distance.
    let h_matrix = Matrix1::new(1.0);
    // Q: Process Noise Covariance. How much we trust our "constant distance" model.
    let q_matrix = Matrix1::new(0.1);
    // R: Observation Noise Covariance. How much we trust the sensor measurement.
    let r_matrix = Matrix1::new(0.5);
    // P: Initial State Covariance. Initial uncertainty.
    let p_matrix = Matrix1::new(1.0);

    let kf = DistanceKalmanFilter::new(f_matrix, b_matrix, h_matrix, q_matrix, r_matrix, p_matrix);

    let mut tracker = ObjectTracker {
        id: Uuid::new_v4(),
        class_id,
        last_bbox: bbox,
        kf,
        frames_without_detection: 0,
    };

    // Set the initial state
    tracker.kf.x = State::new(initial_distance);
    tracker
}

/// Fuses object detection results with a depth map to calculate the distance
/// to each object. This is a safe Rust implementation.
///
/// # Arguments
/// * `detections` - A slice of raw detection results from the object detector.
/// * `depth_map` - The depth map data from the depth estimator.
/// * `frame_width`, `frame_height` - Dimensions of the original video frame.
/// * `focal_length_x`, `focal_length_y` - Camera focal lengths for size estimation.
///
/// # Returns
/// A `Vec<EnrichedObject>` containing the fused data.
pub fn fuse_object_and_depth_data(
    detections: &[ffi::tk_detection_result_t],
    depth_map: &ffi::tk_vision_depth_map_t,
    frame_width: u32,
    frame_height: u32,
    focal_length_x: f32,
    focal_length_y: f32,
) -> Vec<EnrichedObject> {
    const IOU_THRESHOLD: f32 = 0.4;
    const MAX_FRAMES_WITHOUT_DETECTION: u32 = 5;

    let mut trackers = TRACKERS.lock().unwrap();
    let mut enriched_objects = Vec::new();
    let mut matched_tracker_ids = std::collections::HashSet::new();

    // --- 1. Association: Match new detections to existing trackers ---
    for detection in detections {
        let raw_distance = calculate_raw_distance(detection, depth_map, frame_width, frame_height);
        if raw_distance < 0.0 { continue; } // Skip if no valid depth data

        let mut best_match: Option<(Uuid, f32)> = None;

        for (id, tracker) in trackers.iter() {
            let iou = calculate_iou(&detection.bbox, &tracker.last_bbox);
            if iou > IOU_THRESHOLD && iou > best_match.map_or(0.0, |(_, score)| score) {
                best_match = Some((*id, iou));
            }
        }

        // --- 2. Update or Create ---
        if let Some((tracker_id, _)) = best_match {
            // Matched an existing tracker
            if let Some(tracker) = trackers.get_mut(&tracker_id) {
                // Predict the next state
                tracker.kf.predict(None);

                // Update with the new measurement
                let measurement = Observation::new(raw_distance);
                tracker.kf.update(&measurement);

                tracker.last_bbox = detection.bbox;
                tracker.frames_without_detection = 0;
                matched_tracker_ids.insert(tracker_id);
            }
        } else {
            // No match found, create a new tracker
            let new_tracker = create_new_tracker(detection.class_id, detection.bbox, raw_distance);
            matched_tracker_ids.insert(new_tracker.id);
            trackers.insert(new_tracker.id, new_tracker);
        }
    }

    // --- 3. Cleanup and Result Generation ---
    let mut trackers_to_remove = Vec::new();
    for (id, tracker) in trackers.iter_mut() {
        if matched_tracker_ids.contains(id) {
            // This tracker was seen, generate result from its smoothed state
            let smoothed_distance = tracker.kf.x[0];
            let (width_meters, height_meters) = if smoothed_distance > 0.0 {
                ((tracker.last_bbox.w as f32 * smoothed_distance) / focal_length_x,
                 (tracker.last_bbox.h as f32 * smoothed_distance) / focal_length_y)
            } else {
                (-1.0, -1.0)
            };

            enriched_objects.push(EnrichedObject {
                class_id: tracker.class_id,
                confidence: 1.0, // Confidence is now part of the tracker, not detection
                bbox: tracker.last_bbox,
                distance_meters: smoothed_distance,
                width_meters,
                height_meters,
                is_partially_occluded: false, // Occlusion logic can be re-integrated here
            });
        } else {
            // This tracker was not seen in this frame
            tracker.frames_without_detection += 1;
            if tracker.frames_without_detection > MAX_FRAMES_WITHOUT_DETECTION {
                trackers_to_remove.push(*id);
            }
        }
    }

    // Remove old, untracked objects
    for id in trackers_to_remove {
        trackers.remove(&id);
    }

    enriched_objects
}

/// Helper function to calculate the raw, unfiltered distance for a detection.
/// This contains the logic from the original `fuse_object_and_depth_data`.
fn calculate_raw_distance(
    detection: &ffi::tk_detection_result_t,
    depth_map: &ffi::tk_vision_depth_map_t,
    frame_width: u32,
    frame_height: u32
) -> f32 {
    const MIN_DEPTH_POINTS_FOR_STATS: usize = 10;
    let depth_data = unsafe {
        slice::from_raw_parts(depth_map.data, (depth_map.width * depth_map.height) as usize)
    };

    let bbox = detection.bbox;
    let norm_x_min = bbox.x as f32 / frame_width as f32;
    let norm_y_min = bbox.y as f32 / frame_height as f32;
    let norm_x_max = (bbox.x + bbox.w) as f32 / frame_width as f32;
    let norm_y_max = (bbox.y + bbox.h) as f32 / frame_height as f32;

    let depth_x_min = (norm_x_min * (depth_map.width - 1) as f32).round() as u32;
    let depth_y_min = (norm_y_min * (depth_map.height - 1) as f32).round() as u32;
    let depth_x_max = (norm_x_max * (depth_map.width - 1) as f32).round() as u32;
    let depth_y_max = (norm_y_max * (depth_map.height - 1) as f32).round() as u32;

    if depth_x_min >= depth_map.width || depth_y_min >= depth_map.height ||
       depth_x_max >= depth_map.width || depth_y_max >= depth_map.height ||
       depth_x_min >= depth_x_max || depth_y_min >= depth_y_max {
        return -1.0;
    }

    let mut valid_depths: Vec<f32> = (depth_y_min..=depth_y_max)
        .flat_map(|y| (depth_x_min..=depth_x_max)
            .map(move |x| (y, x)))
        .filter_map(|(y, x)| {
            let index = (y * depth_map.width + x) as usize;
            depth_data.get(index).and_then(|&d| if d > 0.1 && d < 100.0 { Some(d) } else { None })
        })
        .collect();

    if valid_depths.len() >= MIN_DEPTH_POINTS_FOR_STATS {
        valid_depths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q1 = valid_depths[valid_depths.len() / 4];
        let q3 = valid_depths[valid_depths.len() * 3 / 4];
        let iqr = q3 - q1;
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        let filtered_depths: Vec<f32> = valid_depths.into_iter().filter(|&d| d >= lower_bound && d <= upper_bound).collect();
        if !filtered_depths.is_empty() {
            return filtered_depths.iter().sum::<f32>() / filtered_depths.len() as f32;
        }
    }

    -1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use ffi::{tk_detection_result_t, tk_vision_depth_map_t, tk_rect_t};

    #[test]
    fn test_kalman_filter_smoothes_distance() {
        // --- Frame 1: Initial detection ---
        let detection1 = tk_detection_result_t {
            class_id: 1,
            label: std::ptr::null(),
            confidence: 0.9,
            bbox: tk_rect_t { x: 10, y: 10, w: 20, h: 20 },
        };
        let detections1 = vec![detection1];

        // Mock a depth map where the object's area has a depth of 10.0m
        let mut depth_data1 = vec![0.0f32; 100 * 100];
        for y in 10..30 {
            for x in 10..30 {
                depth_data1[y * 100 + x] = 10.0;
            }
        }
        let depth_map1 = tk_vision_depth_map_t {
            width: 100,
            height: 100,
            data: depth_data1.as_mut_ptr(),
        };

        let enriched1 = fuse_object_and_depth_data(&detections1, &depth_map1, 100, 100, 300.0, 300.0);
        assert_eq!(enriched1.len(), 1);
        // The first value should be close to the measurement
        assert!((enriched1[0].distance_meters - 10.0).abs() < 0.1);


        // --- Frame 2: New detection with slightly different distance ---
        let detection2 = tk_detection_result_t {
            class_id: 1,
            label: std::ptr::null(),
            confidence: 0.9,
            bbox: tk_rect_t { x: 11, y: 11, w: 20, h: 20 }, // Slightly moved
        };
        let detections2 = vec![detection2];

        // Mock a depth map where the object's area now has a depth of 12.0m
        let mut depth_data2 = vec![0.0f32; 100 * 100];
         for y in 11..31 {
            for x in 11..31 {
                depth_data2[y * 100 + x] = 12.0;
            }
        }
        let depth_map2 = tk_vision_depth_map_t {
            width: 100,
            height: 100,
            data: depth_data2.as_mut_ptr(),
        };

        let enriched2 = fuse_object_and_depth_data(&detections2, &depth_map2, 100, 100, 300.0, 300.0);
        assert_eq!(enriched2.len(), 1);

        // The key check: The new distance should be smoothed by the Kalman filter,
        // so it should be somewhere between the old state (10.0) and the new measurement (12.0).
        // The exact value depends on the Kalman gain, but it MUST be greater than 10.0 and less than 12.0.
        let smoothed_distance = enriched2[0].distance_meters;
        println!("Smoothed distance: {}", smoothed_distance);
        assert!(smoothed_distance > 10.0 && smoothed_distance < 12.0);
    }
}
