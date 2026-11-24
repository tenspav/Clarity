import CoreML
import CoreVideo
import IOSurface
import Metal

// A single slot. Keep on the inference owner's executor. Keep the slot alive
// until both prediction and the final GPU consumer have finished.
struct ClaritySlot {
    enum BridgeError: Error { case allocate, layout, wrongOutput, texture }
    let input: CVPixelBuffer
    let output: CVPixelBuffer
    let features: MLDictionaryFeatureProvider
    let options: MLPredictionOptions
    let shape: [NSNumber]
    let imageIO: Bool

    init(width: Int, height: Int, packed: Bool, imageIO: Bool = false) throws {
        guard width > 0, height > 0, !packed || (!imageIO && width % 2 == 0 && height % 2 == 0)
        else { throw BridgeError.layout }
        let w = packed ? width / 2 : width
        let h = packed ? height / 2 : height
        let channels = packed ? 4 : 1
        self.imageIO = imageIO
        shape = [1, channels, h, w].map { NSNumber(value: $0) }
        input = try Self.makeSurface(width: w, height: channels * h)
        output = try Self.makeSurface(width: w, height: channels * h)
        options = MLPredictionOptions()
        if imageIO {
            features = try MLDictionaryFeatureProvider(dictionary: ["luma": MLFeatureValue(pixelBuffer: input)])
            options.outputBackings = ["delta": output]
        } else {
            let tensor = MLMultiArray(pixelBuffer: input, shape: shape)
            let delta = MLMultiArray(pixelBuffer: output, shape: shape)
            guard tensor.shape == shape && delta.shape == shape,
                  tensor.dataType == .float16 && delta.dataType == .float16,
                  let actualIn = tensor.pixelBuffer, let actualOut = delta.pixelBuffer,
                  Self.surfaceID(actualIn) == Self.surfaceID(input),
                  Self.surfaceID(actualOut) == Self.surfaceID(output)
            else { throw BridgeError.layout }
            features = try MLDictionaryFeatureProvider(dictionary: ["luma": MLFeatureValue(multiArray: tensor)])
            options.outputBackings = ["delta": delta]
        }
    }

    // Call only after the input Metal producer completes, on a separate
    // inference executor. Do not wait in the presentation or Motion queue.
    func predict(using model: MLModel) throws -> CVPixelBuffer {
        let result = try model.prediction(from: features, options: options)
        let feature = result.featureValue(for: "delta")
        let actual = imageIO ? feature?.imageBufferValue : feature?.multiArrayValue?.pixelBuffer
        guard let actual, let actualID = Self.surfaceID(actual),
              actualID == Self.surfaceID(output)
        else { throw BridgeError.wrongOutput }
        return actual
    }

    static func metalView(_ pixelBuffer: CVPixelBuffer,
                          cache: CVMetalTextureCache,
                          usage: MTLTextureUsage = .shaderRead) throws -> (CVMetalTexture, MTLTexture) {
        var image: CVMetalTexture?
        let attrs = [kCVMetalTextureUsage as String: NSNumber(value: usage.rawValue)] as CFDictionary
        let rc = CVMetalTextureCacheCreateTextureFromImage(
            nil, cache, pixelBuffer, attrs, .r16Float,
            CVPixelBufferGetWidth(pixelBuffer), CVPixelBufferGetHeight(pixelBuffer), 0, &image)
        guard rc == kCVReturnSuccess, let image, let texture = CVMetalTextureGetTexture(image)
        else { throw BridgeError.texture }
        return (image, texture)
    }

    private static func makeSurface(width: Int, height: Int) throws -> CVPixelBuffer {
        let attrs: [String: Any] = [
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        ]
        var pb: CVPixelBuffer?
        let rc = CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_OneComponent16Half,
                                     attrs as CFDictionary, &pb)
        guard rc == kCVReturnSuccess, let pb, surfaceID(pb) != nil else { throw BridgeError.allocate }
        return pb
    }

    private static func surfaceID(_ pb: CVPixelBuffer) -> IOSurfaceID? {
        guard let surface = CVPixelBufferGetIOSurface(pb)?.takeUnretainedValue() else { return nil }
        return IOSurfaceGetID(surface)
    }
}

// Load a precompiled .mlmodelc. The caller owns actor isolation and the deadline.
func loadClarity(_ url: URL) throws -> MLModel {
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    return try MLModel(contentsOf: url, configuration: config)
}
