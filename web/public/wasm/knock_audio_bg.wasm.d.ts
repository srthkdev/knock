/* tslint:disable */
/* eslint-disable */
export const memory: WebAssembly.Memory;
export const __wbg_features_free: (a: number, b: number) => void;
export const __wbg_get_features_brightness_hz: (a: number) => number;
export const __wbg_get_features_distinct_hits: (a: number) => number;
export const __wbg_get_features_dominant_pitch_hz: (a: number) => number;
export const __wbg_get_features_loudness_db_over_floor: (a: number) => number;
export const __wbg_get_features_ring_time_s: (a: number) => number;
export const __wbg_set_features_brightness_hz: (a: number, b: number) => void;
export const __wbg_set_features_distinct_hits: (a: number, b: number) => void;
export const __wbg_set_features_dominant_pitch_hz: (a: number, b: number) => void;
export const __wbg_set_features_loudness_db_over_floor: (a: number, b: number) => void;
export const __wbg_set_features_ring_time_s: (a: number, b: number) => void;
export const features: (a: number, b: number) => [number, number, number];
export const features_f32: (a: number, b: number) => [number, number, number];
export const render_f32: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: bigint, j: number, k: number, l: number) => [number, number, number, number];
export const reverb_tail_s: () => number;
export const room_and_mic: (a: number, b: number, c: bigint, d: number, e: number) => [number, number, number, number];
export const sample_rate: () => number;
export const synthesize: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: bigint, j: number) => [number, number, number, number];
export const __wbindgen_externrefs: WebAssembly.Table;
export const __wbindgen_malloc: (a: number, b: number) => number;
export const __externref_table_dealloc: (a: number) => void;
export const __wbindgen_free: (a: number, b: number, c: number) => void;
export const __wbindgen_start: () => void;
