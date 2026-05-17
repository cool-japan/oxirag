/*
 * napi_stub.c — Weak stub implementations for Node.js N-API functions.
 *
 * These stubs satisfy the linker when building non-Node.js artifacts
 * (test binaries, examples, the oxirag-server binary) with the `nodejs`
 * feature active.  In a real `.node` addon loaded by Node.js, these weak
 * symbols are overridden by the actual napi_* implementations exported by
 * the Node.js process.
 *
 * The stubs all return 0 (napi_ok) and write NULL/0 to output pointers.
 * They are never called at runtime in non-Node.js builds.
 */
#include <stddef.h>
#include <stdint.h>

typedef void* napi_env;
typedef void* napi_value;
typedef void* napi_ref;
typedef void* napi_deferred;
typedef void* napi_threadsafe_function;
typedef void* napi_callback_info;
typedef int   napi_status;
typedef int   napi_valuetype;
typedef void  (*napi_threadsafe_function_call_js)(napi_env, napi_value, void*, void*);
typedef void  (*napi_finalize)(napi_env, void*, void*);

#define NAPI_OK 0

#define STUB0(ret, name) \
    __attribute__((weak)) ret name(void) { return NAPI_OK; }

#define STUB_PTR(name, ...) \
    __attribute__((weak)) napi_status name(__VA_ARGS__) { return NAPI_OK; }

STUB_PTR(napi_call_function, napi_env e, napi_value recv, napi_value fn, size_t argc, const napi_value* argv, napi_value* result)
STUB_PTR(napi_call_threadsafe_function, napi_threadsafe_function fn, void* data, int mode)
STUB_PTR(napi_coerce_to_string, napi_env e, napi_value val, napi_value* result)
STUB_PTR(napi_create_array_with_length, napi_env e, size_t len, napi_value* result)
STUB_PTR(napi_create_double, napi_env e, double val, napi_value* result)
STUB_PTR(napi_create_error, napi_env e, napi_value code, napi_value msg, napi_value* result)
STUB_PTR(napi_create_object, napi_env e, napi_value* result)
STUB_PTR(napi_create_promise, napi_env e, napi_deferred* deferred, napi_value* promise)
STUB_PTR(napi_create_reference, napi_env e, napi_value val, uint32_t cnt, napi_ref* ref)
STUB_PTR(napi_create_string_utf8, napi_env e, const char* str, size_t len, napi_value* result)
STUB_PTR(napi_create_threadsafe_function, napi_env e, napi_value fn, napi_value async_resource, napi_value async_resource_name, size_t max_queue_size, size_t initial_thread_count, void* thread_finalize_data, napi_finalize thread_finalize_cb, void* context, napi_threadsafe_function_call_js call_js_cb, napi_threadsafe_function* result)
STUB_PTR(napi_create_uint32, napi_env e, uint32_t val, napi_value* result)
STUB_PTR(napi_delete_reference, napi_env e, napi_ref ref)
STUB_PTR(napi_get_and_clear_last_exception, napi_env e, napi_value* result)
STUB_PTR(napi_get_array_length, napi_env e, napi_value val, uint32_t* result)
STUB_PTR(napi_get_cb_info, napi_env e, napi_callback_info info, size_t* argc, napi_value* argv, napi_value* thisArg, void** data)
STUB_PTR(napi_get_element, napi_env e, napi_value obj, uint32_t idx, napi_value* result)
STUB_PTR(napi_get_global, napi_env e, napi_value* result)
STUB_PTR(napi_get_named_property, napi_env e, napi_value obj, const char* name, napi_value* result)
STUB_PTR(napi_get_null, napi_env e, napi_value* result)
STUB_PTR(napi_get_reference_value, napi_env e, napi_ref ref, napi_value* result)
STUB_PTR(napi_get_undefined, napi_env e, napi_value* result)
STUB_PTR(napi_get_value_bool, napi_env e, napi_value val, int* result)
STUB_PTR(napi_get_value_double, napi_env e, napi_value val, double* result)
STUB_PTR(napi_get_value_string_utf8, napi_env e, napi_value val, char* buf, size_t bufsize, size_t* result)
STUB_PTR(napi_get_value_uint32, napi_env e, napi_value val, uint32_t* result)
STUB_PTR(napi_is_array, napi_env e, napi_value val, int* result)
STUB_PTR(napi_is_error, napi_env e, napi_value val, int* result)
STUB_PTR(napi_is_exception_pending, napi_env e, int* result)
STUB_PTR(napi_new_instance, napi_env e, napi_value cons, size_t argc, const napi_value* argv, napi_value* result)
STUB_PTR(napi_reference_unref, napi_env e, napi_ref ref, uint32_t* result)
STUB_PTR(napi_reject_deferred, napi_env e, napi_deferred deferred, napi_value rejection)
STUB_PTR(napi_release_threadsafe_function, napi_threadsafe_function fn, int mode)
STUB_PTR(napi_resolve_deferred, napi_env e, napi_deferred deferred, napi_value resolution)
STUB_PTR(napi_set_element, napi_env e, napi_value obj, uint32_t idx, napi_value val)
STUB_PTR(napi_set_named_property, napi_env e, napi_value obj, const char* name, napi_value val)
STUB_PTR(napi_throw, napi_env e, napi_value error)
STUB_PTR(napi_typeof, napi_env e, napi_value val, napi_valuetype* result)
STUB_PTR(napi_unwrap, napi_env e, napi_value val, void** result)
STUB_PTR(napi_wrap, napi_env e, napi_value val, void* native_obj, napi_finalize finalize_cb, void* finalize_hint, napi_ref* result)
