/* tslint:disable */
/* eslint-disable */

/**
 * Acoustic summary, rounded like the Python engine reports it.
 */
export class Features {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    brightness_hz: number;
    distinct_hits: number;
    dominant_pitch_hz: number;
    loudness_db_over_floor: number;
    ring_time_s: number;
}

/**
 * Accepts Float64Array; `features_f32` accepts Float32Array.
 */
export function features(x: Float64Array): Features;

export function features_f32(x: Float32Array): Features;

/**
 * Full chain (synthesize + room_and_mic) returned as Float32Array for `AudioBuffer.copyToChannel`.
 */
export function render_f32(events: Float64Array, freqs: Float64Array, decay: Float64Array, gains: Float64Array, seed: bigint, seconds: number, rustle: boolean, tail_s: number): Float32Array;

/**
 * Default reverb tail time constant (s) to pass as `tail_s`.
 */
export function reverb_tail_s(): number;

/**
 * `tail_s`: reverb tail time constant in seconds (use `reverb_tail_s()` for the default).
 */
export function room_and_mic(x: Float64Array, seed: bigint, rustle: boolean, tail_s: number): Float64Array;

/**
 * Sample rate the engine renders at (Hz).
 */
export function sample_rate(): number;

/**
 * `events` is a flat Float64Array `[t0, J0, tau0, t1, J1, tau1, ...]`.
 */
export function synthesize(events: Float64Array, freqs: Float64Array, decay: Float64Array, gains: Float64Array, seed: bigint, seconds: number): Float64Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_features_free: (a: number, b: number) => void;
    readonly __wbg_get_features_brightness_hz: (a: number) => number;
    readonly __wbg_get_features_distinct_hits: (a: number) => number;
    readonly __wbg_get_features_dominant_pitch_hz: (a: number) => number;
    readonly __wbg_get_features_loudness_db_over_floor: (a: number) => number;
    readonly __wbg_get_features_ring_time_s: (a: number) => number;
    readonly __wbg_set_features_brightness_hz: (a: number, b: number) => void;
    readonly __wbg_set_features_distinct_hits: (a: number, b: number) => void;
    readonly __wbg_set_features_dominant_pitch_hz: (a: number, b: number) => void;
    readonly __wbg_set_features_loudness_db_over_floor: (a: number, b: number) => void;
    readonly __wbg_set_features_ring_time_s: (a: number, b: number) => void;
    readonly features: (a: number, b: number) => [number, number, number];
    readonly features_f32: (a: number, b: number) => [number, number, number];
    readonly render_f32: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: bigint, j: number, k: number, l: number) => [number, number, number, number];
    readonly reverb_tail_s: () => number;
    readonly room_and_mic: (a: number, b: number, c: bigint, d: number, e: number) => [number, number, number, number];
    readonly sample_rate: () => number;
    readonly synthesize: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: bigint, j: number) => [number, number, number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
