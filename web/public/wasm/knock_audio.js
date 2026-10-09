/* @ts-self-types="./knock_audio.d.ts" */

/**
 * Acoustic summary, rounded like the Python engine reports it.
 */
export class Features {
    static __wrap(ptr) {
        const obj = Object.create(Features.prototype);
        obj.__wbg_ptr = ptr;
        FeaturesFinalization.register(obj, obj.__wbg_ptr, obj);
        return obj;
    }
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        FeaturesFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_features_free(ptr, 0);
    }
    /**
     * @returns {number}
     */
    get brightness_hz() {
        const ret = wasm.__wbg_get_features_brightness_hz(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get distinct_hits() {
        const ret = wasm.__wbg_get_features_distinct_hits(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @returns {number}
     */
    get dominant_pitch_hz() {
        const ret = wasm.__wbg_get_features_dominant_pitch_hz(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get loudness_db_over_floor() {
        const ret = wasm.__wbg_get_features_loudness_db_over_floor(this.__wbg_ptr);
        return ret;
    }
    /**
     * @returns {number}
     */
    get ring_time_s() {
        const ret = wasm.__wbg_get_features_ring_time_s(this.__wbg_ptr);
        return ret;
    }
    /**
     * @param {number} arg0
     */
    set brightness_hz(arg0) {
        wasm.__wbg_set_features_brightness_hz(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set distinct_hits(arg0) {
        wasm.__wbg_set_features_distinct_hits(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set dominant_pitch_hz(arg0) {
        wasm.__wbg_set_features_dominant_pitch_hz(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set loudness_db_over_floor(arg0) {
        wasm.__wbg_set_features_loudness_db_over_floor(this.__wbg_ptr, arg0);
    }
    /**
     * @param {number} arg0
     */
    set ring_time_s(arg0) {
        wasm.__wbg_set_features_ring_time_s(this.__wbg_ptr, arg0);
    }
}
if (Symbol.dispose) Features.prototype[Symbol.dispose] = Features.prototype.free;

/**
 * Accepts Float64Array; `features_f32` accepts Float32Array.
 * @param {Float64Array} x
 * @returns {Features}
 */
export function features(x) {
    const ptr0 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.features(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return Features.__wrap(ret[0]);
}

/**
 * @param {Float32Array} x
 * @returns {Features}
 */
export function features_f32(x) {
    const ptr0 = passArrayF32ToWasm0(x, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.features_f32(ptr0, len0);
    if (ret[2]) {
        throw takeFromExternrefTable0(ret[1]);
    }
    return Features.__wrap(ret[0]);
}

/**
 * Full chain (synthesize + room_and_mic) returned as Float32Array for `AudioBuffer.copyToChannel`.
 * @param {Float64Array} events
 * @param {Float64Array} freqs
 * @param {Float64Array} decay
 * @param {Float64Array} gains
 * @param {bigint} seed
 * @param {number} seconds
 * @param {boolean} rustle
 * @param {number} tail_s
 * @returns {Float32Array}
 */
export function render_f32(events, freqs, decay, gains, seed, seconds, rustle, tail_s) {
    const ptr0 = passArrayF64ToWasm0(events, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(freqs, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(decay, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(gains, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.render_f32(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, seed, seconds, rustle, tail_s);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF32FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 4, 4);
    return v5;
}

/**
 * Default reverb tail time constant (s) to pass as `tail_s`.
 * @returns {number}
 */
export function reverb_tail_s() {
    const ret = wasm.reverb_tail_s();
    return ret;
}

/**
 * `tail_s`: reverb tail time constant in seconds (use `reverb_tail_s()` for the default).
 * @param {Float64Array} x
 * @param {bigint} seed
 * @param {boolean} rustle
 * @param {number} tail_s
 * @returns {Float64Array}
 */
export function room_and_mic(x, seed, rustle, tail_s) {
    const ptr0 = passArrayF64ToWasm0(x, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ret = wasm.room_and_mic(ptr0, len0, seed, rustle, tail_s);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v2 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v2;
}

/**
 * Sample rate the engine renders at (Hz).
 * @returns {number}
 */
export function sample_rate() {
    const ret = wasm.sample_rate();
    return ret;
}

/**
 * `events` is a flat Float64Array `[t0, J0, tau0, t1, J1, tau1, ...]`.
 * @param {Float64Array} events
 * @param {Float64Array} freqs
 * @param {Float64Array} decay
 * @param {Float64Array} gains
 * @param {bigint} seed
 * @param {number} seconds
 * @returns {Float64Array}
 */
export function synthesize(events, freqs, decay, gains, seed, seconds) {
    const ptr0 = passArrayF64ToWasm0(events, wasm.__wbindgen_malloc);
    const len0 = WASM_VECTOR_LEN;
    const ptr1 = passArrayF64ToWasm0(freqs, wasm.__wbindgen_malloc);
    const len1 = WASM_VECTOR_LEN;
    const ptr2 = passArrayF64ToWasm0(decay, wasm.__wbindgen_malloc);
    const len2 = WASM_VECTOR_LEN;
    const ptr3 = passArrayF64ToWasm0(gains, wasm.__wbindgen_malloc);
    const len3 = WASM_VECTOR_LEN;
    const ret = wasm.synthesize(ptr0, len0, ptr1, len1, ptr2, len2, ptr3, len3, seed, seconds);
    if (ret[3]) {
        throw takeFromExternrefTable0(ret[2]);
    }
    var v5 = getArrayF64FromWasm0(ret[0], ret[1]).slice();
    wasm.__wbindgen_free(ret[0], ret[1] * 8, 8);
    return v5;
}
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg_Error_30c8987f7c2ed4e2: function(arg0, arg1) {
            const ret = Error(getStringFromWasm0(arg0, arg1));
            return ret;
        },
        __wbg___wbindgen_throw_41e9ee4f547fc59a: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbindgen_init_externref_table: function() {
            const table = wasm.__wbindgen_externrefs;
            const offset = table.grow(4);
            table.set(0, undefined);
            table.set(offset + 0, undefined);
            table.set(offset + 1, null);
            table.set(offset + 2, true);
            table.set(offset + 3, false);
        },
    };
    return {
        __proto__: null,
        "./knock_audio_bg.js": import0,
    };
}

const FeaturesFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_features_free(ptr, 1));

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayF64FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat64ArrayMemory0().subarray(ptr / 8, ptr / 8 + len);
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

let cachedFloat64ArrayMemory0 = null;
function getFloat64ArrayMemory0() {
    if (cachedFloat64ArrayMemory0 === null || cachedFloat64ArrayMemory0.byteLength === 0) {
        cachedFloat64ArrayMemory0 = new Float64Array(wasm.memory.buffer);
    }
    return cachedFloat64ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF64ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 8, 8) >>> 0;
    getFloat64ArrayMemory0().set(arg, ptr / 8);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function takeFromExternrefTable0(idx) {
    const value = wasm.__wbindgen_externrefs.get(idx);
    wasm.__externref_table_dealloc(idx);
    return value;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedFloat32ArrayMemory0 = null;
    cachedFloat64ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    wasm.__wbindgen_start();
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (!module.ok) {
            throw new Error(`failed to fetch Wasm: ${module.status} ${module.statusText} fetching '${module.url}'`);
        }

        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('knock_audio_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
