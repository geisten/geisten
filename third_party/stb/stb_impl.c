/*
 * third_party/stb/stb_impl.c — single implementation TU for stb headers.
 *
 * Both stb_image.h and stb_image_resize2.h are single-header public-
 * domain libraries; one TU in the project must instantiate them with the
 * IMPLEMENTATION macros. That's this file. All other code includes the
 * headers without the macros to pick up just the declarations.
 *
 * The headers warn under -Wpedantic / -Wshadow / -Wundef on some clang
 * versions; this TU compiles with relaxed flags via a per-file override
 * in mk/common.mk.
 */
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG       1
#define STBI_ONLY_JPEG      1
#define STBI_ONLY_BMP       1
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"
