#import "SplatProcessing.h"

// MARK: - Color Space Conversion

/// Converts sRGB color to linear color space.
/// This is needed because Metal with sRGB framebuffers expects linear input,
/// while SH evaluation produces sRGB output (matching the reference implementation).
inline half3 sRGBToLinear(half3 srgb) {
    // Using gamma 2.2 approximation (standard for 3DGS training)
    return pow(srgb, half3(2.2h));
}

// MARK: - Front-to-Back Particle Formation

// Value noise adapted from Spark's Splat Reveal Effects (MIT).
// License: THIRD_PARTY_NOTICES.md and SceneAssets/SparkRevealReference-LICENSE.txt.
inline float revealHash(float3 p) {
    p = fract(p * 0.3183099f + 0.1f) * 17.0f;
    return fract(p.x * p.y * p.z);
}

inline float revealNoise(float3 p) {
    float3 cell = floor(p);
    float3 f = fract(p);
    f = f * f * (3.0f - 2.0f * f);
    float x00 = mix(revealHash(cell), revealHash(cell + float3(1, 0, 0)), f.x);
    float x10 = mix(revealHash(cell + float3(0, 1, 0)), revealHash(cell + float3(1, 1, 0)), f.x);
    float x01 = mix(revealHash(cell + float3(0, 0, 1)), revealHash(cell + float3(1, 0, 1)), f.x);
    float x11 = mix(revealHash(cell + float3(0, 1, 1)), revealHash(cell + float3(1, 1, 1)), f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

// Rotate both centres and anisotropic Gaussian covariance for Spark's helix effects.
inline float3x3 revealRotation(float3 axis, float angle) {
    float c = cos(angle), s = sin(angle), t = 1.0f - c;
    float x = axis.x, y = axis.y, z = axis.z;
    return float3x3(float3(t*x*x+c, t*x*y+s*z, t*x*z-s*y),
                    float3(t*x*y-s*z, t*y*y+c, t*y*z+s*x),
                    float3(t*x*z+s*y, t*y*z-s*x, t*z*z+c));
}

// Adapted from Spark's Spread, Magic, Unroll and Twister, with bounded 0...1 timing.
// The captured model-space frame is shared by both eyes and remains fixed during a reversal.
inline void applyReveal(thread Splat &splat, Uniforms uniforms) {
    if (uniforms.revealEnabled == 0 || uniforms.revealProgress >= 1.0f) return;
    if (uniforms.revealProgress <= 0.0f) { splat.color.a = 0.0h; return; }

    float progress = uniforms.revealProgress;
    float3 p = float3(splat.position);
    float3 origin = float3(uniforms.revealOriginX, uniforms.revealOriginY, uniforms.revealOriginZ);
    float3 forward = float3(uniforms.revealDirectionX, uniforms.revealDirectionY, uniforms.revealDirectionZ);
    float3 up = float3(uniforms.revealUpX, uniforms.revealUpY, uniforms.revealUpZ);
    float3 center = origin + forward * (0.5f * (uniforms.revealNearDepth + uniforms.revealFarDepth))
                          + up * (0.5f * (uniforms.revealLow + uniforms.revealHigh));
    float3 q = p - center;
    float height = dot(q, up);
    float3 radial = q - up * height;
    float radius = max(0.001f, uniforms.revealRadius);
    float radialPhase = saturate(length(radial) / radius);
    float heightPhase = saturate((dot(p - origin, up) - uniforms.revealLow) /
                                max(0.001f, uniforms.revealHigh - uniforms.revealLow));
    float grain = revealHash(p * 73.19f + float3(3.1f, 7.2f, 1.7f));
    float3 scatter = float3(grain,
        revealHash(p * 47.31f + float3(1.8f, 8.3f, 2.4f)),
        revealHash(p * 59.73f + float3(6.4f, 2.1f, 9.5f))) * 2.0f - 1.0f;
    float phase = 0;
    float growth = 0;
    float opacity = 0;
    float angle = 0;
    float3 position = p;

    if (uniforms.revealStyle == 2) {
        // Magic: angular seed reveal followed by an expanding noisy radial formation ring.
        float3 right = cross(up, forward);
        if (dot(right, right) < 0.0001f) right = cross(up, abs(up.x) < 0.9f ? float3(1,0,0) : float3(0,0,1));
        right = normalize(right);
        float3 tangent = normalize(cross(right, up));
        float angular = atan2(dot(radial, right), dot(radial, tangent)) / (2.0f * M_PI_F) + 0.5f;
        opacity = smoothstep(angular * 0.20f, angular * 0.20f + 0.03f, progress);
        phase = saturate((progress - 0.18f - radialPhase * 0.58f) / 0.24f);
        growth = smoothstep(0.0f, 1.0f, phase);
        float border = abs((progress - 0.18f) / 0.58f - radialPhase);
        float3 boundedQ = q * min(1.0f, radius / max(length(q), 0.001f));
        position -= boundedQ * (0.20f * exp(-20.0f * border) * (1.0f - growth));
        position += scatter * uniforms.revealParticleTravel * 2.0f * (1.0f - growth);
    } else {
        float arrival;
        if (uniforms.revealStyle == 3) {
            // Unroll: a vertical helix unwinds and expands out to the original scene.
            arrival = heightPhase * 0.70f + grain * 0.03f;
        } else if (uniforms.revealStyle == 4) {
            // Twister: radial weather front, with individual rise timing and a shrinking spin.
            arrival = radialPhase * 0.68f + grain * 0.04f;
        } else {
            // Spread: preserve Spark's two-stage scale growth, progressing near to far.
            float depthPhase = saturate((dot(p - origin, forward) - uniforms.revealNearDepth) /
                                        max(0.001f, uniforms.revealFarDepth - uniforms.revealNearDepth));
            float regional = revealNoise(p * uniforms.revealNoiseFrequency);
            arrival = depthPhase * (1.0f - uniforms.revealFrontWidth - 0.06f)
                    + 0.04f * regional + 0.02f * grain;
        }
        phase = saturate((progress - arrival) / uniforms.revealFrontWidth);
        if (phase <= 0.0f) { splat.color.a = 0.0h; return; }
        if (phase >= 1.0f) return;
        opacity = smoothstep(0.0f, 0.12f, phase);
        if (uniforms.revealStyle == 3) {
            growth = smoothstep(0.25f, 1.0f, phase);
            angle = (heightPhase * 8.0f - 4.0f) * M_PI_F * exp(-3.0f * phase) * (1.0f - phase);
            position = center + revealRotation(up, angle) * q * mix(0.2f, 1.0f, smoothstep(0.0f, 1.0f, phase));
        } else if (uniforms.revealStyle == 4) {
            float s = smoothstep(0.0f, 1.0f, phase);
            float rise = pow(s, 0.5f + 2.0f * grain);
            float h = max(0.25f, uniforms.revealHigh - uniforms.revealLow);
            float y = mix(-h * 0.65f, height, rise);
            float3 r = radial * mix(0.5f, 1.0f, rise);
            angle = progress * 8.0f * (1.0f - s) * 0.2f
                  + (y / h) * 20.0f * (1.0f - s) * exp(-length(r) / radius);
            position = center + revealRotation(up, angle) * r + up * y;
            growth = pow(s, 12.0f);
        } else {
            growth = max(0.2f * smoothstep(0.12f, 0.35f, phase), smoothstep(0.32f, 1.0f, phase));
            float remaining = 1.0f - smoothstep(0.15f, 1.0f, phase);
            position += (0.45f * scatter - 0.55f * forward) * uniforms.revealParticleTravel * remaining * remaining;
        }
    }

    // Preserve original colour; birth opacity is brief, the visible effect is particle motion/shape.
    splat.color.a *= half(opacity);
    float remaining = 1.0f - growth;
    float seedVariance = uniforms.revealSeedRadius * uniforms.revealSeedRadius * remaining * remaining;
    float3 a = float3(splat.covA), b = float3(splat.covB);
    float3x3 covariance = float3x3(a, float3(a.y, b.x, b.y), float3(a.z, b.y, b.z));
    if (uniforms.revealStyle >= 3) {
        float3x3 rotation = revealRotation(up, angle);
        covariance = rotation * covariance * transpose(rotation);
    }
    covariance *= growth * growth;
    covariance[0][0] += seedVariance;
    covariance[1][1] += seedVariance;
    covariance[2][2] += seedVariance;
    splat.covA = packed_half3(covariance[0]);
    splat.covB = packed_half3(covariance[1][1], covariance[2][1], covariance[2][2]);
    splat.position = packed_float3(position);
}

// MARK: - Spherical Harmonics Evaluation

/// Evaluates spherical harmonics to compute view-dependent color.
/// - Parameters:
///   - dir: Normalized view direction vector (from camera to splat)
///   - sh0: DC term (raw SH band 0 coefficients, RGB)
///   - shCoeffs: Pointer to higher-order SH coefficients (can be null for degree 0)
///   - shDegree: The SH degree (0-3)
///   - splatIndex: Index of the splat (for computing offset into shCoeffs)
/// - Returns: Final RGB color after SH evaluation
inline half3 evaluateSH(float3 dir,
                        half3 sh0,
                        device const half* shCoeffs,
                        SHDegree shDegree,
                        uint splatIndex) {
    // Degree 0 (constant/ambient)
    float3 result = SH_C0 * float3(sh0);

    if (shDegree >= SHDegree1 && shCoeffs != nullptr) {
        float x = dir.x, y = dir.y, z = dir.z;

        // Calculate offset: SH1=3, SH2=8, SH3=15 coefficients (each is RGB triplet)
        uint coeffsPerSplat = (shDegree == SHDegree1) ? 3 :
                              (shDegree == SHDegree2) ? 8 : 15;
        device const packed_half3* sh = (device const packed_half3*)(shCoeffs + splatIndex * coeffsPerSplat * 3);

        // Degree 1: 3 basis functions
        result -= SH_C1 * y * float3(sh[0]);
        result += SH_C1 * z * float3(sh[1]);
        result -= SH_C1 * x * float3(sh[2]);

        if (shDegree >= SHDegree2) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            float xxMinusYy = xx - yy;
            float xxPlusYy = xx + yy;

            // Degree 2: 5 basis functions
            result += SH_C2_0 * xy * float3(sh[3]);
            result += SH_C2_1 * yz * float3(sh[4]);
            result += SH_C2_2 * (2.0f * zz - xxPlusYy) * float3(sh[5]);
            result += SH_C2_3 * xz * float3(sh[6]);
            result += SH_C2_4 * xxMinusYy * float3(sh[7]);

            if (shDegree >= SHDegree3) {
                float zz4MinusXxYy = 4.0f * zz - xxPlusYy;

                // Degree 3: 7 basis functions
                result += SH_C3_0 * y * (3.0f * xx - yy) * float3(sh[8]);
                result += SH_C3_1 * xy * z * float3(sh[9]);
                result += SH_C3_2 * y * zz4MinusXxYy * float3(sh[10]);
                result += SH_C3_3 * z * (2.0f * zz - 3.0f * xxPlusYy) * float3(sh[11]);
                result += SH_C3_4 * x * zz4MinusXxYy * float3(sh[12]);
                result += SH_C3_5 * z * xxMinusYy * float3(sh[13]);
                result += SH_C3_6 * x * (xx - 3.0f * yy) * float3(sh[14]);
            }
        }
    }

    // Add 0.5 bias and clamp to non-negative (critical for correct colors)
    return half3(max(result + 0.5f, 0.0f));
}

// MARK: - Covariance Projection

float3 calcCovariance2D(float3 viewPos,
                        packed_half3 cov3Da,
                        packed_half3 cov3Db,
                        float4x4 viewMatrix,
                        float focalX,
                        float focalY,
                        float tanHalfFovX,
                        float tanHalfFovY) {
    float invViewPosZ = 1 / viewPos.z;
    float invViewPosZSquared = invViewPosZ * invViewPosZ;

    float limX = 1.3f * tanHalfFovX;
    float limY = 1.3f * tanHalfFovY;
    viewPos.x = clamp(viewPos.x * invViewPosZ, -limX, limX) * viewPos.z;
    viewPos.y = clamp(viewPos.y * invViewPosZ, -limY, limY) * viewPos.z;

    float3x3 J = float3x3(
        focalX * invViewPosZ, 0, 0,
        0, focalY * invViewPosZ, 0,
        -(focalX * viewPos.x) * invViewPosZSquared, -(focalY * viewPos.y) * invViewPosZSquared, 0
    );
    float3x3 W = float3x3(viewMatrix[0].xyz, viewMatrix[1].xyz, viewMatrix[2].xyz);
    float3x3 T = J * W;
    float3x3 Vrk = float3x3(
        cov3Da.x, cov3Da.y, cov3Da.z,
        cov3Da.y, cov3Db.x, cov3Db.y,
        cov3Da.z, cov3Db.y, cov3Db.z
    );
    float3x3 cov = T * Vrk * transpose(T);

    // Apply low-pass filter: every Gaussian should be at least
    // one pixel wide/high. Discard 3rd row and column.
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    return float3(cov[0][0], cov[0][1], cov[1][1]);
}

// cov2D is a flattened 2d covariance matrix. Given
// covariance = | a b |
//              | c d |
// (where b == c because the Gaussian covariance matrix is symmetric),
// cov2D = ( a, b, d )
void decomposeCovariance(float3 cov2D, thread float2 &v1, thread float2 &v2) {
    float a = cov2D.x;
    float b = cov2D.y;
    float d = cov2D.z;
    float det = a * d - b * b; // matrix is symmetric, so "c" is same as "b"
    float trace = a + d;

    float mean = 0.5 * trace;
    float dist = max(0.1, sqrt(mean * mean - det)); // based on cuda_rasterizer/forward.cu in graphdeco-inria diff-gaussian-rasterization

    // Eigenvalues
    float lambda1 = mean + dist;
    float lambda2 = mean - dist;

    float2 eigenvector1;
    if (b == 0) {
        eigenvector1 = (a > d) ? float2(1, 0) : float2(0, 1);
    } else {
        eigenvector1 = normalize(float2(b, d - lambda2));
    }

    // Gaussian axes are orthogonal
    float2 eigenvector2 = float2(eigenvector1.y, -eigenvector1.x);

    v1 = eigenvector1 * sqrt(lambda1);
    v2 = eigenvector2 * sqrt(lambda2);
}


// Soft edge: image coords (X,Y) in [-1,1]; one smooth falloff per axis, multiplied (corners get naturally softer, not a rounded box).
// edgeMix 0 = default "original hard edge, lightly softened" (band edgeExp ≈ 0.05, opacity only, blends into the room);
// edgeMix 1 = immersion (band edgeWidth ≈ 0.17: bright to the edge, outer ring smoothstep→ease-out into black).
// The border only wobbles inward (low-frequency value noise), so the straight data edge never shows; no position change, no re-sort.
inline float edgeProfile(float a, float edge, float w, float nn) {
    float s = clamp((edge - 0.005f - w * 0.18f * (nn + 1.0f) - a) / w, 0.0f, 1.0f);
    float g = s * s * (3.0f - 2.0f * s);
    return g * (2.0f - g);
}

inline bool edgeScreen(Uniforms uniforms, float3 p, thread float &X, thread float &Y, thread float &pz) {
    pz = uniforms.edgeM[6]*p.x + uniforms.edgeM[7]*p.y + uniforms.edgeM[8]*p.z + uniforms.edgeT[2];
    if (pz < 0.05f) return false;
    float px = uniforms.edgeM[0]*p.x + uniforms.edgeM[1]*p.y + uniforms.edgeM[2]*p.z + uniforms.edgeT[0];
    float py = uniforms.edgeM[3]*p.x + uniforms.edgeM[4]*p.y + uniforms.edgeM[5]*p.z + uniforms.edgeT[1];
    X = uniforms.edgeAX * px / pz + uniforms.edgeBX;
    Y = uniforms.edgeAY * py / pz + uniforms.edgeBY;
    return true;
}

// Multiply the final sRGB colour by k (via the SH0 coefficients): a smooth fade on black.
inline void scaleSRGB(thread Splat &splat, float k) {
    float3 c = SH_C0 * float3(splat.color.rgb) + 0.5f;
    splat.color.rgb = half3((c * k - 0.5f) / SH_C0);
}

inline void applyEdgeBreath(thread Splat &splat, Uniforms uniforms, float3 p) {
    if (uniforms.edgeEnabled == 0) return;
    float X, Y, pz;
    if (!edgeScreen(uniforms, p, X, Y, pz)) return;
    float mixv = clamp(uniforms.edgeMix, 0.0f, 1.0f);
    float w = mix(uniforms.edgeExp, uniforms.edgeWidth, mixv);
    float wy = w * uniforms.edgeAY / max(uniforms.edgeAX, 1e-3f);
    float t = uniforms.edgeTime;
    float breath = 0.5f + 0.5f * sin(t * (2.0f * M_PI_F / 6.0f));
    float edge = 1.0f - uniforms.edgeAmp * breath * mixv;
    float ax = abs(X), ay = abs(Y);
    if (ax < edge - 0.01f - 1.37f * w && ay < edge - 0.01f - 1.37f * wy) return;   // Skip the image centre (including faces)
    float n1 = revealNoise(float3(X * 1.7f, Y * 1.7f, 0.05f * t));
    float n2 = revealNoise(float3(X * 4.3f + 3.7f, Y * 4.3f + 1.3f, 0.09f * t + 5.0f));
    float nn = (0.7f * n1 + 0.3f * n2) * 2.0f - 1.0f;
    float f = edgeProfile(ax, edge, w, nn) * edgeProfile(ay, edge, wy, nn);
    float bx = min(smoothstep(uniforms.edgeBoxX0 - 0.08f, uniforms.edgeBoxX0, X), 1.0f - smoothstep(uniforms.edgeBoxX1, uniforms.edgeBoxX1 + 0.08f, X));
    float by = min(smoothstep(uniforms.edgeBoxY0 - 0.08f, uniforms.edgeBoxY0, Y), 1.0f - smoothstep(uniforms.edgeBoxY1, uniforms.edgeBoxY1 + 0.08f, Y));
    f = max(f, bx * by * mixv);
    if (f >= 0.999f) return;
    if (mixv > 0.001f) scaleSRGB(splat, mix(1.0f, f, mixv));
    splat.color.a *= half(mix(f, sqrt(f), mixv));
}

// Wind: moves centres by a few centimetres (scaled with depth so tree tops visibly sway); no re-sort.
// Weight: higher above the floor moves more; green (leaves/grass) moves more; 0 inside the still box; water only shimmers.
inline void applyWind(thread Splat &splat, Uniforms u, float3 p) {
    if (u.windEnabled == 0 || u.windStrength <= 0.001f || u.edgeEnabled == 0) return;
    float X, Y, pz;
    if (!edgeScreen(u, p, X, Y, pz) || pz < 0.3f) return;
    float h = u.windGround + u.windUp * p.y;
    float3 c = SH_C0 * float3(splat.color.rgb) + 0.5f;
    float leafy = smoothstep(0.0f, 0.07f, c.g - 0.5f * (c.r + c.b));
    float wTree = smoothstep(1.2f, 4.0f, h) * mix(0.35f, 1.0f, leafy);
    float wBush = leafy * smoothstep(0.25f, 1.0f, h) * 0.8f;
    float wGrass = leafy * (1.0f - smoothstep(0.3f, 0.8f, h)) * 0.3f;
    float w = max(max(wTree, wBush), wGrass);
    float sx = min(smoothstep(u.windBoxX0 - 0.06f, u.windBoxX0, X), 1.0f - smoothstep(u.windBoxX1, u.windBoxX1 + 0.06f, X));
    float sy = min(smoothstep(u.windBoxY0 - 0.06f, u.windBoxY0, Y), 1.0f - smoothstep(u.windBoxY1, u.windBoxY1 + 0.06f, Y));
    float sz = 1.0f - smoothstep(u.windBoxZ - 1.0f, u.windBoxZ + 1.0f, pz);
    float still = sx * sy * sz;
    w *= 1.0f - still;
    float t = u.windTime;
    if (w > 0.002f) {
        float A = u.windStrength * min(0.25f, 0.012f + 0.0035f * pz) * w;
        float phase = revealNoise(p * 0.6f) * 6.2831853f;
        float2 dir = float2(u.windDirX, u.windDirZ);
        float gust = 0.55f + 0.45f * revealNoise(float3(t * 0.23f, 3.1f, 7.7f));
        float wave = revealNoise(float3(p.x * 0.12f - t * 0.9f * dir.x, p.z * 0.12f - t * 0.9f * dir.y, 1.3f));
        float sway = 0.5f * sin(t * 1.7f + phase) + 0.25f * sin(t * 2.9f + phase * 1.7f)
                   + 0.5f * (revealNoise(p * 0.9f + float3(0.0f, t * 0.7f, 0.0f)) - 0.5f);
        float flutter = 0.15f * leafy * sin(t * 6.0f + revealHash(p * 37.0f) * 6.2831853f);
        float m = gust * (0.6f * sway + 0.8f * (wave - 0.35f));
        float3 d = A * float3(dir.x * m + flutter, 0.25f * sin(t * 4.1f + phase * 3.0f) * leafy, dir.y * m);
        splat.position = packed_float3(float3(splat.position) + d);
    }
    if (u.windShimmer > 0.0f) {
        float wx = min(smoothstep(u.windWaterX0 - 0.05f, u.windWaterX0, X), 1.0f - smoothstep(u.windWaterX1, u.windWaterX1 + 0.05f, X));
        float wy = min(smoothstep(u.windWaterY0 - 0.03f, u.windWaterY0, Y), 1.0f - smoothstep(u.windWaterY1, u.windWaterY1 + 0.03f, Y));
        float water = wx * wy * (1.0f - leafy) * (1.0f - smoothstep(0.15f, 0.6f, h)) * (1.0f - still);
        if (water > 0.01f) {
            float r = revealNoise(float3(p.x * 2.5f + t * 0.4f, p.z * 4.0f - t * 0.6f, t * 0.3f));
            scaleSRGB(splat, 1.0f + u.windShimmer * u.windStrength * water * (r - 0.5f) * 2.0f);
        }
    }
}

// Reach dissolve: Gaussians near a reaching hand slowly fade and drift like dust, then gather back when the hand withdraws.
// Depends on the hand only (not head distance); changes opacity and a little position; no re-sort.
inline float touchFactor(float3 p, float3 c, float R, float amount, float feather, float t) {
    if (amount <= 0.001f || R <= 0.0f) return 0.0f;
    float3 v = p - c;
    float d = length(v);
    float fw = max(R * feather, 1e-3f);
    if (d > R + 1.5f * fw) return 0.0f;
    float n = revealNoise(p * (2.5f / R) + float3(0.0f, t * 0.15f, 0.0f)) - 0.5f;
    float k = 1.0f - smoothstep(R - fw, R + fw, d + n * fw * 1.6f);
    return amount * k;
}

inline void applyTouchDissolve(thread Splat &splat, Uniforms u, float3 p) {
    if (u.touchAAmount <= 0.001f && u.touchBAmount <= 0.001f) return;
    float t = u.touchTime;
    float3 ca = float3(u.touchAX, u.touchAY, u.touchAZ), cb = float3(u.touchBX, u.touchBY, u.touchBZ);
    float ka = touchFactor(p, ca, u.touchAR, u.touchAAmount, u.touchFeather, t);
    float kb = touchFactor(p, cb, u.touchBR, u.touchBAmount, u.touchFeather, t);
    float k = max(ka, kb);
    if (k <= 0.001f) return;
    float3 c = ka >= kb ? ca : cb;
    float3 out = p - c;
    float len = length(out);
    out = len > 1e-4f ? out / len : float3(0.0f);
    float3 up = float3(u.touchUpX, u.touchUpY, u.touchUpZ);
    float3 jitter = float3(revealHash(p * 41.3f), revealHash(p * 53.7f + 2.1f), revealHash(p * 67.9f + 5.3f)) * 2.0f - 1.0f;
    float3 drift = normalize(out * 0.6f + up * 0.45f + jitter * 0.35f + float3(1e-4f));
    float ease = k * k;
    splat.position = packed_float3(float3(splat.position) + drift * (u.touchDrift * ease * (0.6f + 0.4f * sin(t * 0.7f + revealHash(p * 13.1f) * 6.2831853f))));
    float shrink = 1.0f - 0.45f * ease;
    splat.covA = packed_half3(half3(splat.covA) * half(shrink * shrink));
    splat.covB = packed_half3(half3(splat.covB) * half(shrink * shrink));
    float fade = 1.0f - k;
    splat.color.a *= half(fade * sqrt(fade));
}

FragmentIn splatVertex(Splat splat,
                       Uniforms uniforms,
                       uint relativeVertexIndex,
                       device const half* shCoefficients,
                       SHDegree shDegree,
                       uint splatIndex) {
    FragmentIn out;

    float3 edgeNative = splat.position;
    applyReveal(splat, uniforms);
    applyEdgeBreath(splat, uniforms, edgeNative);
    applyWind(splat, uniforms, edgeNative);
    applyTouchDissolve(splat, uniforms, edgeNative);

    // Fragments below 1/255 alpha cannot change an 8-bit target. Splats that never reach it are
    // culled; the rest get a quad that ends where alpha falls to 1/255 (capped at 3 sigma).
    // Same visible footprint, fewer rasterized fragments. Runs after effects so reveal fades apply.
    float opacity = float(splat.color.a);
    if (!(opacity * 255.0f > 1.0f)) {
        out.position = float4(1, 1, 0, 1);
        return out;
    }
    half quadRadius = half(min(float(kBoundsRadius), sqrt(2.0f * log(opacity * 255.0f))));

    float4 viewPosition4 = uniforms.viewMatrix * float4(splat.position, 1);
    float3 viewPosition3 = viewPosition4.xyz;

    // Early cull: skip splats behind or at the camera plane
    // (In view space, camera looks down -z, so z >= 0 means behind camera)
    if (viewPosition3.z >= 0) {
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    // Reject the same out-of-view centres before expensive SH/covariance work.
    float4 projectedCenter = uniforms.projectionMatrix * viewPosition4;

    float bounds = 1.2 * projectedCenter.w;
    if (projectedCenter.z < 0.0 ||
        projectedCenter.z > projectedCenter.w ||
        projectedCenter.x < -bounds ||
        projectedCenter.x > bounds ||
        projectedCenter.y < -bounds ||
        projectedCenter.y > bounds) {
        out.position = float4(1, 1, 0, 1);
        return out;
    }

    half3 srgbColor;
    if (shDegree == SHDegree0) {
        // Fast path for SH0
        srgbColor = half3(max(SH_C0 * float3(splat.color.rgb) + 0.5f, 0.0f));
    } else {
        float3 worldPosition = float3(splat.position);
        float3 viewDir = normalize(worldPosition - float3(uniforms.cameraPosition));
        srgbColor = evaluateSH(viewDir, splat.color.rgb, shCoefficients, shDegree, splatIndex);
    }

    float3 cov2D = calcCovariance2D(viewPosition3, splat.covA, splat.covB,
                                    uniforms.viewMatrix,
                                    uniforms.focalX, uniforms.focalY,
                                    uniforms.tanHalfFovX, uniforms.tanHalfFovY);

    float2 axis1;
    float2 axis2;
    decomposeCovariance(cov2D, axis1, axis2);

    const half2 relativeCoordinatesArray[] = { { -1, -1 }, { -1, 1 }, { 1, -1 }, { 1, 1 } };
    half2 relativeCoordinates = relativeCoordinatesArray[relativeVertexIndex];
    half2 screenSizeFloat = half2(uniforms.screenSize.x, uniforms.screenSize.y);
    half2 projectedScreenDelta =
        (relativeCoordinates.x * half2(axis1) + relativeCoordinates.y * half2(axis2))
        * 2
        * quadRadius
        / screenSizeFloat;

    out.position = float4(projectedCenter.x + projectedScreenDelta.x * projectedCenter.w,
                          projectedCenter.y + projectedScreenDelta.y * projectedCenter.w,
                          projectedCenter.z,
                          projectedCenter.w);
    out.relativePosition = quadRadius * relativeCoordinates;

    // Convert from sRGB to linear to match Metal expectations for shader color output
    out.color = half4(sRGBToLinear(srgbColor), splat.color.a);
    return out;
}

half splatFragmentAlpha(half2 relativePosition, half splatAlpha) {
    half negativeMagnitudeSquared = -dot(relativePosition, relativePosition);
    return (negativeMagnitudeSquared < -kBoundsRadiusSquared) ? 0 : exp(0.5 * negativeMagnitudeSquared) * splatAlpha;
}
