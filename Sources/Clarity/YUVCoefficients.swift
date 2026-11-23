import Foundation


/// Coefficients for R'G'B' = (Y - yOffset) * yScale + matrix * ((CbCr - 0.5) * cScale), all in 0...1 units.
public struct YUVCoefficients: Equatable, Sendable {
    public var yOffset: Float, yScale: Float, cScale: Float
    public var rCr: Float, gCb: Float, gCr: Float, bCb: Float
    /// Chroma zero point as a normalized texel value (128/255 for 8-bit, 512<<6 / 65535 for 10-bit in 16).
    public var cOffset: Float = 128.0 / 255

    /// `matrix` = kCVImageBufferYCbCrMatrixKey value ("ITU_R_709_2", "ITU_R_601_4", "ITU_R_2020"); nil -> BT.709.
    /// `tenBit`: x420 / xf20 planes read as r16 / rg16 unorm (10 significant bits in the top of 16).
    public static func make(matrix: String?, fullRange: Bool, tenBit: Bool = false) -> YUVCoefficients {
        let (rCr, gCb, gCr, bCb): (Float, Float, Float, Float)
        switch matrix ?? "" {
        case let m where m.contains("601") || m.contains("SMPTE_240"): (rCr, gCb, gCr, bCb) = (1.402, 0.344136, 0.714136, 1.772)
        case let m where m.contains("2020"): (rCr, gCb, gCr, bCb) = (1.4746, 0.164553, 0.571353, 1.8814)
        default: (rCr, gCb, gCr, bCb) = (1.5748, 0.187324, 0.468124, 1.8556)
        }
        if tenBit {
            let one: Float = 65535, step: Float = 64           // 1 code of 10 bits = 64 in 16
            let cOff = 512 * step / one
            return fullRange
                ? YUVCoefficients(yOffset: 0, yScale: one / (1023 * step), cScale: one / (1023 * step), rCr: rCr, gCb: gCb, gCr: gCr, bCb: bCb, cOffset: cOff)
                : YUVCoefficients(yOffset: 64 * step / one, yScale: one / (876 * step), cScale: one / (896 * step),
                                  rCr: rCr, gCb: gCb, gCr: gCr, bCb: bCb, cOffset: cOff)
        }
        return fullRange
            ? YUVCoefficients(yOffset: 0, yScale: 1, cScale: 1, rCr: rCr, gCb: gCb, gCr: gCr, bCb: bCb)
            : YUVCoefficients(yOffset: 16.0 / 255, yScale: 255.0 / 219, cScale: 255.0 / 224, rCr: rCr, gCb: gCb, gCr: gCr, bCb: bCb)
    }

    /// CPU reference of the Metal kernel (8-bit in, 0-255 out), for tests.
    public func rgb(y: UInt8, cb: UInt8, cr: UInt8) -> (r: Int, g: Int, b: Int) {
        let yy = (Float(y) / 255 - yOffset) * yScale
        let pb = (Float(cb) / 255 - 128.0 / 255) * cScale, pr = (Float(cr) / 255 - 128.0 / 255) * cScale
        func q(_ v: Float) -> Int { Int((min(1, max(0, v)) * 255).rounded()) }
        return (q(yy + rCr * pr), q(yy - gCb * pb - gCr * pr), q(yy + bCb * pb))
    }
}
