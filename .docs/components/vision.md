# Vision Subsystem

The Vision Subsystem is a cornerstone of the TrackieLLM project, responsible for transforming raw video frames into a structured, semantic understanding of the environment. It has been engineered to be a **production-ready, robust, and highly intelligent** perception system, providing the Cortex with the critical data needed for advanced reasoning and high-level assistance.

## Core Capabilities

-   **Multi-Model Inference**: Utilizes AI models like YOLOv5nu for object detection and MiDaS for monocular depth estimation. The system is resilient to model loading failures, allowing it to operate in a degraded state.
-   **Dynamic Configuration**: The pipeline can be reconfigured at runtime. Parameters like confidence thresholds or even entire models can be enabled/disabled on-the-fly, allowing the system to adapt to different environments and performance requirements without interruption.
-   **Temporal Object Tracking**: Goes beyond single-frame detection by implementing object tracking with a **Kalman Filter**. This provides a stable, smoothed estimate of object distances over time, eliminating jitter and providing a more consistent perception of the world.
-   **3D Point Cloud Analysis**: Instead of a simple 2D grid analysis, the system converts depth maps into a 3D point cloud. It then uses the robust **RANSAC algorithm** to find the ground plane, allowing for highly accurate obstacle and hole detection based on real-world height differences.
-   **Intention-Oriented OCR**: OCR is no longer limited to pre-defined object classes. The Cortex can now request text recognition on **any arbitrary region of interest (ROI)**, enabling a much more flexible and intelligent interaction with text in the environment.
-   **Semantic Scene Graph Construction**: This is the system's most advanced feature. It moves beyond a simple list of objects to build a **semantic graph** that describes the relationships between them (e.g., `cup on_top_of table`). This structured understanding is crucial for genuine scene comprehension.
-   **Object Attribute Classification**: Using efficient, classical computer vision algorithms, the system can classify key attributes of objects, such as their **dominant color**, without the need for additional heavy AI models.

## Data Flow

1.  **Frame Input**: The `Cortex` captures a video frame and passes it to `tk_vision_pipeline_process_frame`, along with flags indicating which analyses are required and an optional ROI for OCR.
2.  **Parallel Analysis & Resilience**: The pipeline runs object detection and depth estimation. If a model has failed to load or an inference fails, it is gracefully skipped, and the final result indicates which data is missing.
3.  **Attribute & Text Analysis**: For each detected object, attributes like color are classified. If OCR is requested (by flag, ROI, or detection of a text-like object), the Tesseract engine is run.
4.  **3D Navigation Analysis**: The depth map is converted to a 3D point cloud. RANSAC is used to find the ground plane, and the traversability of the scene is determined based on this robust 3D analysis.
5.  **Temporal Fusion & Tracking**: The list of current detections is fused with the state of objects tracked from previous frames. A Kalman filter updates the distance and position of each tracked object, providing a smoothed, stable output.
6.  **Scene Graph Construction**: The final list of tracked, enriched objects is passed to a Rust module that infers spatial relationships between them, building a semantic scene graph.
7.  **Structured Output**: The final result, `tk_vision_result_t`, is a comprehensive, multi-layered understanding of the scene. It contains the list of tracked objects (with smoothed distance and attributes), recognized text blocks, navigation data, and the **serialized scene graph**. This rich, structured data is the foundation for the Cortex's reasoning engine.

## Current Status

**100% Production-Ready and Intelligent.** The Vision Subsystem has been elevated far beyond its initial requirements. It is now a robust, resilient, and deeply intelligent perception system. All advanced features—including Kalman filter tracking, RANSAC-based 3D analysis, and semantic scene graph construction—have been implemented, tested, and documented, making it a true cornerstone of the application.