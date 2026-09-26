// Sparse direct-residual matcher. Four warps share a block; each warp owns
// one event/template proposal. No cross-warp synchronization is required.
// FP32 accumulators are retained: INT16 is an I/O encoding, not score math.

// Fuse normalization, temporal peak detection and bounded packing. Adjacent
// lanes access adjacent contacts; no full time-by-channel score array.
extern "C" __global__ void detect_residual_peaks(
    const float* residual, const float* noise, long long nt, int nc,
    float floor, unsigned long long* count, long long* output,
    unsigned long long capacity)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long size = nt * nc;
    bool keep = false;
    if (i < size) {
        const int c = i % nc;
        const long long t = i / nc;
        const float q = fabsf(residual[i]) / noise[c];
        keep = q > floor;
        if (keep && t > 0) keep = q > fabsf(residual[i-nc]) / noise[c];
        if (keep && t+1 < nt) keep = q >= fabsf(residual[i+nc]) / noise[c];
    }
    const unsigned int bits = __ballot_sync(0xffffffff, keep);
    const int lane = threadIdx.x & 31;
    unsigned long long base = 0;
    if (lane == 0 && bits) base = atomicAdd(count, (unsigned long long)__popc(bits));
    base = __shfl_sync(0xffffffff, base, 0);
    if (keep) {
        const unsigned int before = lane == 0 ? 0 : bits & ((1u << lane)-1u);
        const unsigned long long slot = base + __popc(before);
        if (slot < capacity) output[slot] = i;
    }
}

// Three-tap temporal smoothing and local-peak detection in one pass. A block
// cooperatively stages an 8-sample time tile plus two halo samples per side.
// Warps own one time row, so lanes remain coalesced over neighboring contacts.
extern "C" __global__ void detect_residual_peaks_smooth3_shared(
    const float* residual, const float* noise, long long nt, int nc,
    float floor, unsigned long long* count, long long* output,
    unsigned long long capacity)
{
    const int cx = threadIdx.x;
    const int ty = threadIdx.y;
    const int c = (int)blockIdx.x * 32 + cx;
    const long long base = (long long)blockIdx.y * 8;
    __shared__ float tile[12][32];

    for (int row = ty; row < 12; row += 8) {
        long long t = base + row - 2;
        if (t < 0) t = 0;
        if (t >= nt) t = nt - 1;
        tile[row][cx] = c < nc ? residual[t * (long long)nc + c] : 0.f;
    }
    __syncthreads();

    const long long t = base + ty;
    bool keep = false;
    if (t < nt && c < nc && isfinite(noise[c]) && noise[c] > 0.f) {
        const float center_raw = tile[ty + 2][cx];
        const float center = (t == 0 || t == nt - 1)
            ? center_raw
            : 0.25f * (tile[ty + 1][cx] + 2.f * center_raw + tile[ty + 3][cx]);
        const float q = fabsf(center) / noise[c];
        keep = q > floor;
        if (keep && t > 0) {
            const long long previous_t = t - 1;
            const float previous_raw = tile[ty + 1][cx];
            const float previous = (previous_t == 0 || previous_t == nt - 1)
                ? previous_raw
                : 0.25f * (tile[ty][cx] + 2.f * previous_raw + tile[ty + 2][cx]);
            keep = q > fabsf(previous) / noise[c];
        }
        if (keep && t + 1 < nt) {
            const long long next_t = t + 1;
            const float next_raw = tile[ty + 3][cx];
            const float next = (next_t == 0 || next_t == nt - 1)
                ? next_raw
                : 0.25f * (tile[ty + 2][cx] + 2.f * next_raw + tile[ty + 4][cx]);
            keep = q >= fabsf(next) / noise[c];
        }
    }

    const unsigned int bits = __ballot_sync(0xffffffff, keep);
    const int lane = cx;
    unsigned long long base_out = 0;
    if (lane == 0 && bits)
        base_out = atomicAdd(count, (unsigned long long)__popc(bits));
    base_out = __shfl_sync(0xffffffff, base_out, 0);
    if (keep) {
        const unsigned int before = lane == 0 ? 0 : bits & ((1u << lane) - 1u);
        const unsigned long long slot = base_out + __popc(before);
        if (slot < capacity)
            output[slot] = t * (long long)nc + c;
    }
}

__device__ __forceinline__ float session_warp_sum(float x) {
    for (int d = 16; d; d >>= 1)
        x += __shfl_down_sync(0xffffffff, x, d);
    return x;
}

extern "C" __global__ void score_residual_pairs(
    const float* __restrict__ residual, const int* __restrict__ event_time,
    const float* __restrict__ templates, const int* __restrict__ channels,
    const int* __restrict__ pair_event, const int* __restrict__ pair_unit,
    const float* __restrict__ noise_weight,
    int n_pairs, int n_samples, int n_channels, int width, int half_width,
    const int* __restrict__ shift_radii,
    float score_floor, float amplitude_min, float amplitude_max,
    float* __restrict__ scores, float* __restrict__ amplitudes,
    float* __restrict__ gains, int* __restrict__ shifts)
{
    const int p = blockIdx.x * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (p >= n_pairs) return;
    const int event = pair_event[p], unit = pair_unit[p];
    const int shift_radius = shift_radii[unit];
    const int center = event_time[event];
    const int channel = lane < width ? channels[unit * width + lane] : -1;
    const float* model = templates + unit * 61 * width;
    float best_gain = 0.f, best_score = -1.f, best_amplitude = 0.f;
    int best_shift = 0;
    for (int shift = -shift_radius; shift <= shift_radius; ++shift) {
        const int t0 = center + shift;
        float dot = 0.f, aa = 0.f, bb = 0.f;
        float full_dot = 0.f, full_aa = 0.f;
        if (channel >= 0 && t0 >= 30 && t0 + 30 < n_samples) {
            const float weight = noise_weight[channel];
            #pragma unroll 1
            for (int dt = -30; dt <= 30; ++dt) {
                const float a = model[(30 + dt) * width + lane];
                const float b = residual[(t0 + dt) * n_channels + channel];
                full_dot += a * b * weight;
                full_aa += a * a * weight;
                if (dt >= -half_width && dt <= half_width) {
                    dot += a * b * weight;
                    aa += a * a * weight;
                    bb += b * b * weight;
                }
            }
        }
        dot = session_warp_sum(dot);
        aa = session_warp_sum(aa);
        bb = session_warp_sum(bb);
        full_dot = session_warp_sum(full_dot);
        full_aa = session_warp_sum(full_aa);
        if (lane == 0 && aa > 1e-9f && bb > 1e-9f && full_aa > 1e-9f) {
            const float score = dot * rsqrtf(aa * bb);
            const float amp = full_dot / full_aa;
            const float gain = 2.f * amp * full_dot - amp * amp * full_aa;
            if (score >= score_floor && amp >= amplitude_min && amp <= amplitude_max &&
                (gain > best_gain || (gain == best_gain && abs(shift) < abs(best_shift)))) {
                best_gain = gain; best_score = score;
                best_amplitude = amp; best_shift = shift;
            }
        }
    }
    if (lane == 0) {
        scores[p] = best_score; amplitudes[p] = best_amplitude;
        gains[p] = best_gain; shifts[p] = best_shift;
    }
}

// Proposals selected on the host are disjoint on any common contact within
// one waveform span. Atomic subtraction remains safe if a caller relaxes that
// selection policy in the future.
extern "C" __global__ void subtract_fits(
    float* __restrict__ residual, const float* __restrict__ templates,
    const int* __restrict__ channels, const int* __restrict__ times,
    const int* __restrict__ units, const float* __restrict__ amplitudes,
    int n_fits, int n_samples, int n_channels, int width)
{
    const int fit = blockIdx.x;
    if (fit >= n_fits) return;
    const int center = times[fit], unit = units[fit];
    const float amplitude = amplitudes[fit];
    for (int k = threadIdx.x; k < 61 * width; k += blockDim.x) {
        const int dt = k / width - 30;
        const int slot = k % width;
        const int channel = channels[unit * width + slot];
        const int t = center + dt;
        if (channel >= 0 && t >= 0 && t < n_samples)
            atomicAdd(residual + t * n_channels + channel,
                      -amplitude * templates[unit * 61 * width + k]);
    }
}
