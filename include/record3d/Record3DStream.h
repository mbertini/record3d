#ifndef CPP_RECORD3DSTREAM_H
#define CPP_RECORD3DSTREAM_H

#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <array>
#include "Record3DStructs.h"
#include <stdint.h>
#include <functional>
#include <string.h>

#ifdef PYTHON_BINDINGS_BUILD
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
#endif


namespace Record3D
{
    using BufferRGB = std::vector<uint8_t>;
    using BufferDepth = std::vector<uint8_t>;
    using BufferConfidence = std::vector<uint8_t>;
    using BufferMisc = std::vector<uint8_t>;

    class Record3DStream
    {
    public:
        /**
         * Represents single connection to a device. Each instance represents connection to an individual device.
         * Upon connecting to a selected device, a callback function is specified that will be passed the RGB and Depth
         * frames.
         */
        Record3DStream();

        ~Record3DStream();

        /**
         * Returns list of currently attached USB devices which can be later passed into the `ConnectToDevice` method.
         * @return list of currently attached devices.
         */
        static std::vector<DeviceInfo> GetConnectedDevices();

        /**
         * Returns the version string of the Record3D library.
         * @return version string (e.g. "1.4.1").
         */
        static std::string GetVersion();

        /**
         * Connect to a selected iDevice via USB. The `onNewFrame()` callback will be called upon receiving
         * a new RGBD frame from the iDevice.
         *
         * The callback function is going to be called on non-main thread, therefore it is not suitable for performing
         * UI-related tasks (such as displaying images via OpenCV).
         *
         * The callback function is intended only for copying the passed RGB and Depth buffers into your application's
         * internal buffers. No computation should be done in it as such behaviour could introduce stream lag when
         * spending too much time in it.
         *
         * You can be connected to only one device in a `Record3DStream` instance. Trying to connect to a different
         * iDevice using this function while being already connected does nothing. You will need to first disconnect
         * from the current device via `Disconnect()`.
         *
         * @param $device an iDevice from the list of devices obtained via `GetConnectedDevices()`.
         * @returns boolean value indicating whether connection has been established.
         */
        bool ConnectToDevice(const DeviceInfo &$device);

        /**
         * Manually terminate the current streaming session.
         */
        void Disconnect();

    private:
        /**
         * Runloop that reads from the connected iDevice and processes received data.
         */
        void StreamProcessingRunloop();

        /**
         * Performs one-shot disconnect cleanup: closes the socket and fires the onStreamStopped callback.
         * Safe to call from multiple threads — only the first call has effect.
         */
        void PerformDisconnectCleanup();

        /**
         * Decompresses incoming compressed depth frame into $destinationBuffer of known size.
         *
         * @param $compressedBuffer the buffer containing LZFSE-compressed depth frame.
         * @param $compressedBufferSize size of the compressed buffer.
         * @param $destinationBuffer buffer into which the decompressed depth frame is going to be written.
         * @returns pointer to the decompressed buffer. In case of decompression failure, `nullptr` is returned.
         */
        uint8_t* DecompressBuffer(const uint8_t* $compressedBuffer, size_t $compressedBufferSize, std::vector<uint8_t> &$destinationBuffer);

        /**
         * Wraps the standard `recv()` function to ensure the *exact* amount of bytes (`$numBytesToRead`) is read into the $outputBuffer.
         *
         * @param $socketHandle socket handle obtained as the return value of `usbmuxd_connect()`.
         * @param $outputBuffer buffer that should be written into.
         * @param $numBytesToRead exact amount of bytes that should be read from $socketHandle and written into $socketHandle.
         * @return the amount of bytes actually read. Usually it is equal to $numBytesToRead, but in case connection is interrupted, the amount will be different.
         */
        uint32_t ReceiveWholeBuffer(int $socketHandle, uint8_t* $outputBuffer, uint32_t $numBytesToRead);

    public:
        // Constants
        static constexpr uint16_t DEVICE_PORT{ 1337 }; /** Port on iDevice that we are listening to for RGBD stream. */

#ifdef PYTHON_BINDINGS_BUILD
        /**
         * NOTE: This is alternative API for Python.
         *
         * This function is called when a new RGBD frame is received. The RGBD data and current intrinsic matrix are intended
         * to be read via the `GetCurrentDepthFrame()`, `GetCurrentRGBFrame()` and `GetCurrentIntrinsicMatrix()` accessors.
         */
        std::function<void()> onNewFrame{};
#else
        /**
         * This function is called when a new RGBD frame is received.
         *
         * The callback function is going to be called on non-main thread, therefore it is not suitable for performing
         * UI-related tasks (such as displaying images via OpenCV).
         *
         * The callback function is intended only to copy the passed RGB and Depth buffers into your application's
         * internal buffers. No computation should be done in it as such behaviour could introduce stream lag when
         * spending too much time in it.
         *
         * @param $rgbFrame RGB buffer.
         * @param $depthFrame Depth buffer.
         * @param $confFrame Confidence buffer.
         * @param $miscData Misc buffer (reserved).
         * @param $rgbWidth width of the RGB frame.
         * @param $rgbHeight height of the RGB frame.
         * @param $depthWidth width of the Depth frame.
         * @param $depthHeight height of the Depth frame.
         * @param $confWidth width of Confidence frame (0 if there is no confidence frame).
         * @param $confHeight height of Confidence frame (0 if there is no confidence frame).
         * @param $deviceType type of the camera used for streaming.
         * @param $K four coefficients of the intrinsic matrix corresponding to the RGB frame.
         * @param $cameraPose camera pose in world space.
         */
        std::function<void(const Record3D::BufferRGB &$rgbFrame,
                           const Record3D::BufferDepth &$depthFrame,
                           const Record3D::BufferConfidence &$confFrame,
                           const Record3D::BufferMisc &$miscData,
                           uint32_t   $rgbWidth,
                           uint32_t   $rgbHeight,
                           uint32_t   $depthWidth,
                           uint32_t   $depthHeight,
                           uint32_t   $confWidth,
                           uint32_t   $confHeight,
                           Record3D::DeviceType $deviceType,
                           Record3D::IntrinsicMatrixCoeffs $K,
                           Record3D::CameraPose $cameraPose)> onNewFrame{};
#endif
        /**
         * This function is called when stream ends (that can happen either due to transfer error or by calling the `Disconnect()` method).
         */
        std::function<void()> onStreamStopped{};

#ifdef PYTHON_BINDINGS_BUILD
        /**
         * NOTE: This is alternative API for Python.
         *
         * @returns the current contents of the Depth buffer which holds the lastly received Depth frame.
         */
        py::array_t<float> GetCurrentDepthFrame()
        {
            std::lock_guard<std::mutex> guard{ frameMutex_ };
            size_t currentFrameWidth = currentFrameDepthWidth_;
            size_t currentFrameHeight = currentFrameDepthHeight_;

            size_t bufferSize  = currentFrameWidth * currentFrameHeight * sizeof(float);
            auto result        = py::array_t<float>({ static_cast<int>(currentFrameHeight), static_cast<int>(currentFrameWidth) });
            auto result_buffer = result.request();
            float *result_ptr  = (float *) result_buffer.ptr;

            std::memcpy(result_ptr, depthImageBuffer_.data(), bufferSize);

            return result;
        }

        /**
         * NOTE: This is alternative API for Python.
         *
         * @returns the current contents of the Confidence buffer which holds the lastly received Depth frame.
         */
        py::array_t<uint8_t> GetCurrentConfidenceFrame()
        {
            std::lock_guard<std::mutex> guard{ frameMutex_ };
            size_t currentFrameWidth = currentFrameConfidenceWidth_;
            size_t currentFrameHeight = currentFrameConfidenceHeight_;

            size_t bufferSize  = currentFrameWidth * currentFrameHeight * sizeof(uint8_t);
            auto result        = py::array_t<uint8_t>({ static_cast<int>(currentFrameHeight), static_cast<int>(currentFrameWidth) });
            auto result_buffer = result.request();
            uint8_t *result_ptr  = (uint8_t *) result_buffer.ptr;

            std::memcpy(result_ptr, confidenceImageBuffer_.data(), bufferSize);

            return result;
        }

        /**
         * NOTE: This is alternative API for Python.
         *
         * @returns the current contents of the Misc buffer (reserved).
         */
        py::array_t<uint8_t> GetCurrentMiscData()
        {
            std::lock_guard<std::mutex> guard{ frameMutex_ };
            size_t bufferSize  = miscBuffer_.size();
            auto result        = py::array_t<uint8_t>( bufferSize );
            auto result_buffer = result.request();
            uint8_t *result_ptr  = (uint8_t *) result_buffer.ptr;

            std::memcpy(result_ptr, miscBuffer_.data(), bufferSize);

            return result;
        }

        /**
         * NOTE: This is alternative API for Python.
         *
         * @returns the current contents of the RGB buffer which holds the lastly received RGB frame.
         */
        py::array_t<uint8_t> GetCurrentRGBFrame()
        {
            std::lock_guard<std::mutex> guard{ frameMutex_ };
            size_t currentFrameWidth = currentFrameRGBWidth_;
            size_t currentFrameHeight = currentFrameRGBHeight_;

            constexpr int numChannels = 3;
            size_t bufferSize  = currentFrameWidth * currentFrameHeight * numChannels * sizeof(uint8_t);
            auto result        = py::array_t<uint8_t>({ static_cast<int>(currentFrameHeight), static_cast<int>(currentFrameWidth), numChannels });
            auto result_buffer = result.request();
            uint8_t *result_ptr  = (uint8_t *) result_buffer.ptr;

            std::memcpy(result_ptr, RGBImageBuffer_.data(), bufferSize);

            return result;
        }

        /**
         * NOTE: This is alternative API for Python.
         *
         * @returns the intrinsic matrix of the lastly received Depth frame.
         */
        IntrinsicMatrixCoeffs GetCurrentIntrinsicMatrix()
        {
            std::lock_guard<std::mutex> guard{ frameMutex_ };
            return rgbIntrinsicMatrixCoeffs_;
        }

        /**
         * NOTE: This is alternative API for Python.
         *
         * @returns the intrinsic matrix of the lastly received Depth frame.
         */
        CameraPose GetCurrentCameraPose()
        {
            std::lock_guard<std::mutex> guard{ frameMutex_ };
            return cameraPose_;
        }

        /**
         * NOTE: This is alternative API for Python.
         *
         * @returns the type of camera (TrueDeph = 0, LiDAR = 1).
         */
        uint32_t GetCurrentDeviceType()
        {
            std::lock_guard<std::mutex> guard{ frameMutex_ };
            return (uint32_t) currentDeviceType_;
        }
#endif
    private:
        size_t currentFrameRGBWidth_{ 0 };
        size_t currentFrameRGBHeight_{ 0 };

        size_t currentFrameDepthWidth_{ 0 };
        size_t currentFrameDepthHeight_{ 0 };

        size_t currentFrameConfidenceWidth_{ 0 };
        size_t currentFrameConfidenceHeight_{ 0 };

        DeviceType currentDeviceType_ {};

        uint8_t* lzfseScratchBuffer_{ nullptr }; /** Preallocated LZFSE scratch buffer. */

        int socketHandle_{ -1 }; /** Socket handle representing connection to iDevice. */
        std::atomic<bool> connectionEstablished_{ false }; /** Flag  */
        std::thread runloopThread_; /** Background thread on which the main runloop is running. */

        std::mutex apiCallsMutex_; /** Mutex used for guarding API calls. */
        mutable std::mutex frameMutex_; /** Mutex used for guarding frame buffer access. */
        std::atomic<bool> disconnectCleanupDone_{ true }; /** Ensures disconnect cleanup runs at most once per connection. */

        std::vector<uint8_t> depthImageBuffer_{}; /** Holds the most recent Depth buffer. */
        std::vector<uint8_t> RGBImageBuffer_{}; /** Holds the most recent RGB buffer. */
        std::vector<uint8_t> confidenceImageBuffer_{}; /** Holds the Confidence map of the most recent Depth buffer. Pixel values: Low: 0, Medium: 1, High: 2 */
        std::vector<uint8_t> miscBuffer_{}; /** Reserved */
        IntrinsicMatrixCoeffs rgbIntrinsicMatrixCoeffs_{}; /** Holds the intrinsic matrix of the most recent RGB frame. */
        CameraPose cameraPose_{}; /** Holds the world position of the most recent camera frame. */
    };
}
#endif //CPP_RECORD3DSTREAM_H
