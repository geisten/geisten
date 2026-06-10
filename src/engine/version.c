/*
 * src/engine/version.c — geist_version_string and components.
 *
 * Layer: ENGINE.
 */
#define GEIST_INTERNAL_ENGINE_LAYER

#include <geist.h>

const char *geist_version_string(void) {
    return GEIST_VERSION_STRING;
}

void geist_version_components(int *major, int *minor, int *patch) {
    if (major != nullptr) {
        *major = GEIST_VERSION_MAJOR;
    }
    if (minor != nullptr) {
        *minor = GEIST_VERSION_MINOR;
    }
    if (patch != nullptr) {
        *patch = GEIST_VERSION_PATCH;
    }
}
