The project features a robust hardware abstraction layer in the GPU module, which directs operations to the appropriate backend based on the available hardware, ensuring performance across a wide range of platforms. The build system uses CMake to detect available toolchains (CUDA, ROCm, Xcode, etc.) and compile the corresponding backend, controlled by flags such as `TRACKIE_ENABLE_CUDA`.

### **Graphics Accelerators**

The following table details the supported GPU computing platforms and APIs for graphics acceleration and general-purpose computing tasks.

| Accelerator / API | Supported Platforms | Requirements / Dependencies |
| :--- | :--- | :--- |
| **CUDA** | • Linux (x86_64, Embedded/Orange Pi) \<br\> • Edge Servers (x86_64) | • NVIDIA GPU \<br\> • CUDA Toolkit 11.0 or higher |
| **ROCm** | • Linux (x86_64, self-hosted) \<br\> • Edge Servers (x86_64) | • AMD GPU \<br\> • ROCm/HIP Toolchain |
| **Metal** | • macOS (macOS 14+) \<br\> • iOS | • Apple M-series SoCs or compatible hardware \<br\> • Frameworks: Metal and Foundation |
| **Vulkan** | • Android \<br\> • Linux | • Vulkan SDK |
| **OpenGL ES** | • Embedded and mobile platforms | • EGL and GLESv3 libraries |
| **OpenCL** | • Cross-platform | • OpenCL SDK/ICD |

### **AI Accelerators**

This table focuses on backends specifically used to accelerate neural network inference and AI models. Note that several technologies from the previous table also apply here.

| Accelerator / API | Supported Platforms | Requirements / Dependencies |
| :--- | :--- | :--- |
| **CUDA** | • Linux (x86_64, Embedded/Orange Pi) \<br\> • Edge Servers (x86_64) | • Enabled via ONNX Runtime for vision models \<br\> • `find_package(CUDA 11.0 REQUIRED)` in CMake |
| **ROCm** | • Linux (x86_64, self-hosted) \<br\> • Edge Servers (x86_64) | • Enabled via ONNX Runtime for vision models \<br\> • Experimental support, requires ROCm toolchain |
| **Metal** | • macOS (macOS 14+) \<br\> • iOS | • Optimized for Apple's Unified Memory Architecture \<br\> • Compiled with Xcode and Apple SDKs |
| **Android NNAPI** | • Android | • Android NDK \<br\> • `neuralnetworks` library |
| **Vulkan** | • Android \<br\> • Other platforms with Vulkan support | • `find_package(Vulkan REQUIRED)` in CMake |

