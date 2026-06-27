/*
 * src/backends/vulkan/backend.c - optional Vulkan backend skeleton.
 *
 * Layer: BACKEND.
 *
 * This backend is optional and dynamically loads the Vulkan loader. The
 * current increment provides lifecycle, device selection, command submission,
 * and buffer upload/download/map policy. Transformer acceleration lands in
 * later TDD steps.
 */
#define GEIST_INTERNAL_BACKEND_LAYER
#define VK_NO_PROTOTYPES

#include <geist.h>
#include <geist_backend.h>

#include <vulkan/vulkan.h>

#include "gguf_quant.h"

#include "shaders/add_f32_spv.h"
#include "shaders/attention_f32_spv.h"
#include "shaders/attention_f32_dyn_spv.h"
#include "shaders/argmax_f32_batch_spv.h"
#include "shaders/argmax_f32_spv.h"
#include "shaders/embed_lookup_scaled_dyn_spv.h"
#include "shaders/embed_lookup_scaled_spv.h"
#include "shaders/ffn_gate_up_gelu_q4k_spv.h"
#include "shaders/ffn_gate_up_gelu_q4k_nt4_spv.h"
#include "shaders/gelu_tanh_f32_spv.h"
#include "shaders/gelu_tanh_mul_f32_spv.h"
#include "shaders/kv_append_f32_dyn_spv.h"
#include "shaders/matvec_f16bf16_spv.h"
#include "shaders/matvec_f32_spv.h"
#include "shaders/matmul_f32_spv.h"
#include "shaders/matvec_q4k_nt4_spv.h"
#include "shaders/matvec_q4k_nt4_pair_spv.h"
#include "shaders/matvec_q4k_nt4_subgroup_spv.h"
#include "shaders/matvec_q4k_nt4_w4a8_spv.h"
#include "shaders/matvec_q4k_spv.h"
#include "shaders/matvec_q4k_n4_spv.h"
#include "shaders/matmul_q4k_spv.h"
#include "shaders/matmul_q4k_m8_spv.h"
#include "shaders/matvec_q6k_spv.h"
#include "shaders/matvec_q6k_n4_spv.h"
#include "shaders/matmul_q6k_spv.h"
#include "shaders/mul_f32_spv.h"
#include "shaders/q4k_quant_x_spv.h"
#include "shaders/rope_f32_dyn_spv.h"
#include "shaders/rope_f32_spv.h"
#include "shaders/rmsnorm_rope_f32_dyn_spv.h"
#include "shaders/rmsnorm_add_f32_spv.h"
#include "shaders/rmsnorm_f32_spv.h"
#include "shaders/scale_f32_spv.h"

#include <dlfcn.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdalign.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct geist_buffer;

enum {
    VULKAN_DESCRIPTOR_POOL_DEFAULT_SETS = 256u,
    VULKAN_DESCRIPTOR_POOL_DEFAULT_STORAGE_DESCRIPTORS = 1024u,
    VULKAN_DESCRIPTOR_POOL_DECODE_CAPTURE_SETS = 4096u,
    VULKAN_DESCRIPTOR_POOL_DECODE_CAPTURE_STORAGE_DESCRIPTORS = 16384u,
    VULKAN_PROFILE_MAX_STATS = 64u,
    VULKAN_CAPTURE_SET_CACHE_MAX = 2048u,
};

struct vulkan_profile_stat {
    const char *label;
    uint64_t count;
    double total_ms;
    double max_ms;
};

struct vulkan_capture_set_cache_entry {
    VkDescriptorSetLayout layout;
    VkDescriptorSet set;
};

struct vulkan_decode_params {
    uint32_t token_id;
    uint32_t q_position;
    uint32_t kv_len;
    uint32_t token_output_offset;
};
static_assert(sizeof(struct vulkan_decode_params) == 16);

struct vulkan_q4k_nt4_cache_entry {
    struct geist_buffer *src;
    size_t src_offset;
    size_t n_in;
    size_t n_out;
    size_t raw_bytes;
    size_t packed_bytes;
    struct geist_buffer *packed;
    struct vulkan_q4k_nt4_cache_entry *next;
};

struct vulkan_state {
    struct geist_backend *backend;
    void *loader_handle;
    const char *loader_name;

    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkPhysicalDeviceMemoryProperties memory_props;
    VkPhysicalDeviceProperties device_props;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    uint32_t queue_timestamp_valid_bits;
    VkCommandPool command_pool;
    VkCommandPool transient_command_pool;
    VkCommandBuffer transient_command_buffer;
    VkCommandPool decode_replay_command_pool;
    VkCommandBuffer decode_replay_command_buffer;
    VkFence decode_replay_fence;
    bool decode_replay_valid;
    bool decode_replay_recording;
    bool decode_replay_profile_recorded;
    bool decode_replay_enabled;
    bool decode_replay_env_checked;
    uint32_t decode_replay_vocab_size;
    uint32_t decode_replay_token_count;
    bool capture_active;
    VkCommandBuffer capture_command_buffer;
    int capture_token_next;
    VkPipelineCache pipeline_cache;
    VkDescriptorPool transient_descriptor_pool;
    uint32_t transient_descriptor_pool_sets;
    uint32_t transient_descriptor_pool_storage_descriptors;
    VkDescriptorPool block_descriptor_pool;
    uint32_t block_descriptor_pool_sets;
    uint32_t block_descriptor_pool_storage_descriptors;
    VkQueryPool profile_query_pool;
    bool profile_env_checked;
    bool profile_enabled;
    bool profile_transient_recorded;
    bool profile_capture_recorded;
    const char *profile_capture_label;
    uint32_t profile_stat_count;
    struct vulkan_profile_stat profile_stats[VULKAN_PROFILE_MAX_STATS];
    bool capture_set_cache_active;
    bool capture_set_cache_enabled;
    bool capture_set_cache_env_checked;
    uint32_t capture_set_cache_cursor;
    uint32_t capture_set_cache_count;
    struct vulkan_capture_set_cache_entry
        capture_set_cache[VULKAN_CAPTURE_SET_CACHE_MAX];
    VkDescriptorSetLayout add_set_layout;
    VkPipelineLayout add_pipeline_layout;
    VkPipeline add_pipeline;
    VkDescriptorSetLayout scale_set_layout;
    VkPipelineLayout scale_pipeline_layout;
    VkPipeline scale_pipeline;
    VkDescriptorSetLayout mul_set_layout;
    VkPipelineLayout mul_pipeline_layout;
    VkPipeline mul_pipeline;
    VkDescriptorSetLayout gelu_tanh_set_layout;
    VkPipelineLayout gelu_tanh_pipeline_layout;
    VkPipeline gelu_tanh_pipeline;
    VkDescriptorSetLayout gelu_tanh_mul_set_layout;
    VkPipelineLayout gelu_tanh_mul_pipeline_layout;
    VkPipeline gelu_tanh_mul_pipeline;
    VkDescriptorSetLayout rmsnorm_set_layout;
    VkPipelineLayout rmsnorm_pipeline_layout;
    VkPipeline rmsnorm_pipeline;
    VkDescriptorSetLayout rmsnorm_add_set_layout;
    VkPipelineLayout rmsnorm_add_pipeline_layout;
    VkPipeline rmsnorm_add_pipeline;
    VkDescriptorSetLayout rmsnorm_rope_dyn_set_layout;
    VkPipelineLayout rmsnorm_rope_dyn_pipeline_layout;
    VkPipeline rmsnorm_rope_dyn_pipeline;
    VkDescriptorSetLayout matvec_set_layout;
    VkPipelineLayout matvec_pipeline_layout;
    VkPipeline matvec_pipeline;
    VkPipelineLayout matmul_f32_pipeline_layout;
    VkPipeline matmul_f32_pipeline;
    VkDescriptorSetLayout matvec_f16bf16_set_layout;
    VkPipelineLayout matvec_f16bf16_pipeline_layout;
    VkPipeline matvec_f16bf16_pipeline;
    VkDescriptorSetLayout matvec_q4k_set_layout;
    VkPipelineLayout matvec_q4k_pipeline_layout;
    VkPipeline matvec_q4k_pipeline;
    VkPipelineLayout matvec_q4k_n4_pipeline_layout;
    VkPipeline matvec_q4k_n4_pipeline;
    VkPipelineLayout matvec_q4k_nt4_pipeline_layout;
    VkPipeline matvec_q4k_nt4_pipeline;
    VkPipelineLayout matvec_q4k_nt4_subgroup_pipeline_layout;
    VkPipeline matvec_q4k_nt4_subgroup_pipeline;
    VkDescriptorSetLayout matvec_q4k_w4a8_set_layout;
    VkPipelineLayout matvec_q4k_w4a8_pipeline_layout;
    VkPipeline matvec_q4k_w4a8_pipeline;
    VkDescriptorSetLayout q4k_quant_x_set_layout;
    VkPipelineLayout q4k_quant_x_pipeline_layout;
    VkPipeline q4k_quant_x_pipeline;
    struct geist_buffer *q4k_w4a8_xq_buffer;
    struct geist_buffer *q4k_w4a8_scale_buffer;
    size_t q4k_w4a8_xq_capacity;
    VkDescriptorSetLayout matvec_q4k_nt4_pair_set_layout;
    VkPipelineLayout matvec_q4k_nt4_pair_pipeline_layout;
    VkPipeline matvec_q4k_nt4_pair_pipeline;
    VkDescriptorSetLayout ffn_gate_up_gelu_q4k_set_layout;
    VkPipelineLayout ffn_gate_up_gelu_q4k_pipeline_layout;
    VkPipeline ffn_gate_up_gelu_q4k_pipeline;
    VkDescriptorSetLayout ffn_gate_up_gelu_q4k_nt4_set_layout;
    VkPipelineLayout ffn_gate_up_gelu_q4k_nt4_pipeline_layout;
    VkPipeline ffn_gate_up_gelu_q4k_nt4_pipeline;
    VkPipelineLayout matmul_q4k_pipeline_layout;
    VkPipeline matmul_q4k_pipeline;
    VkPipelineLayout matmul_q4k_m8_pipeline_layout;
    VkPipeline matmul_q4k_m8_pipeline;
    VkDescriptorSetLayout matvec_q6k_set_layout;
    VkPipelineLayout matvec_q6k_pipeline_layout;
    VkPipeline matvec_q6k_pipeline;
    VkPipelineLayout matvec_q6k_n4_pipeline_layout;
    VkPipeline matvec_q6k_n4_pipeline;
    VkPipelineLayout matmul_q6k_pipeline_layout;
    VkPipeline matmul_q6k_pipeline;
    VkDescriptorSetLayout argmax_set_layout;
    VkPipelineLayout argmax_pipeline_layout;
    VkPipeline argmax_pipeline;
    VkPipelineLayout argmax_batch_pipeline_layout;
    VkPipeline argmax_batch_pipeline;
    VkDescriptorPool argmax_descriptor_pool;
    VkDescriptorSet argmax_descriptor_set;
    struct geist_buffer *argmax_result_buffer;
    size_t argmax_result_capacity;
    bool captured_greedy_token_pending;
    uint32_t captured_greedy_vocab_size;
    uint32_t captured_greedy_token_count;
    VkDescriptorSetLayout embed_lookup_scaled_set_layout;
    VkPipelineLayout embed_lookup_scaled_pipeline_layout;
    VkPipeline embed_lookup_scaled_pipeline;
    VkDescriptorSetLayout embed_lookup_scaled_dyn_set_layout;
    VkPipelineLayout embed_lookup_scaled_dyn_pipeline_layout;
    VkPipeline embed_lookup_scaled_dyn_pipeline;
    struct geist_buffer *decode_params_buffer;
    bool dynamic_decode_params_enabled;
    bool dynamic_decode_params_env_checked;
    enum geist_command_sequence_kind capture_kind;
    VkDescriptorSetLayout rope_set_layout;
    VkPipelineLayout rope_pipeline_layout;
    VkPipeline rope_pipeline;
    VkDescriptorSetLayout rope_dyn_set_layout;
    VkPipelineLayout rope_dyn_pipeline_layout;
    VkPipeline rope_dyn_pipeline;
    VkDescriptorSetLayout attention_set_layout;
    VkPipelineLayout attention_pipeline_layout;
    VkPipeline attention_pipeline;
    VkDescriptorSetLayout attention_dyn_set_layout;
    VkPipelineLayout attention_dyn_pipeline_layout;
    VkPipeline attention_dyn_pipeline;
    VkDescriptorSetLayout kv_append_dyn_set_layout;
    VkPipelineLayout kv_append_dyn_pipeline_layout;
    VkPipeline kv_append_dyn_pipeline;
    struct vulkan_decode_params decode_params;
    struct vulkan_q4k_nt4_cache_entry *q4k_nt4_cache;

    bool has_compute_queue;
    bool has_subgroup_basic;
    bool has_subgroup_arithmetic;
    bool has_shader_integer_dot_product;
    bool has_descriptor_indexing;
    bool has_timeline_semaphore;

    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr vkGetDeviceProcAddr;
    PFN_vkEnumerateInstanceExtensionProperties vkEnumerateInstanceExtensionProperties;
    PFN_vkCreateInstance vkCreateInstance;
    PFN_vkDestroyInstance vkDestroyInstance;
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices;
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkGetPhysicalDeviceQueueFamilyProperties;
    PFN_vkGetPhysicalDeviceMemoryProperties vkGetPhysicalDeviceMemoryProperties;
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties;
    PFN_vkGetPhysicalDeviceProperties2 vkGetPhysicalDeviceProperties2;
    PFN_vkGetPhysicalDeviceFeatures2 vkGetPhysicalDeviceFeatures2;
    PFN_vkEnumerateDeviceExtensionProperties vkEnumerateDeviceExtensionProperties;
    PFN_vkCreateDevice vkCreateDevice;
    PFN_vkDestroyDevice vkDestroyDevice;
    PFN_vkGetDeviceQueue vkGetDeviceQueue;
    PFN_vkCreateBuffer vkCreateBuffer;
    PFN_vkDestroyBuffer vkDestroyBuffer;
    PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements;
    PFN_vkAllocateMemory vkAllocateMemory;
    PFN_vkFreeMemory vkFreeMemory;
    PFN_vkBindBufferMemory vkBindBufferMemory;
    PFN_vkMapMemory vkMapMemory;
    PFN_vkUnmapMemory vkUnmapMemory;
    PFN_vkFlushMappedMemoryRanges vkFlushMappedMemoryRanges;
    PFN_vkInvalidateMappedMemoryRanges vkInvalidateMappedMemoryRanges;
    PFN_vkCreateShaderModule vkCreateShaderModule;
    PFN_vkDestroyShaderModule vkDestroyShaderModule;
    PFN_vkCreatePipelineCache vkCreatePipelineCache;
    PFN_vkDestroyPipelineCache vkDestroyPipelineCache;
    PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout;
    PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout;
    PFN_vkCreatePipelineLayout vkCreatePipelineLayout;
    PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout;
    PFN_vkCreateComputePipelines vkCreateComputePipelines;
    PFN_vkDestroyPipeline vkDestroyPipeline;
    PFN_vkCreateDescriptorPool vkCreateDescriptorPool;
    PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool;
    PFN_vkCreateQueryPool vkCreateQueryPool;
    PFN_vkDestroyQueryPool vkDestroyQueryPool;
    PFN_vkCreateFence vkCreateFence;
    PFN_vkDestroyFence vkDestroyFence;
    PFN_vkResetFences vkResetFences;
    PFN_vkWaitForFences vkWaitForFences;
    PFN_vkResetDescriptorPool vkResetDescriptorPool;
    PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets;
    PFN_vkFreeDescriptorSets vkFreeDescriptorSets;
    PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets;
    PFN_vkCreateCommandPool vkCreateCommandPool;
    PFN_vkDestroyCommandPool vkDestroyCommandPool;
    PFN_vkResetCommandPool vkResetCommandPool;
    PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers;
    PFN_vkFreeCommandBuffers vkFreeCommandBuffers;
    PFN_vkBeginCommandBuffer vkBeginCommandBuffer;
    PFN_vkEndCommandBuffer vkEndCommandBuffer;
    PFN_vkCmdCopyBuffer vkCmdCopyBuffer;
    PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier;
    PFN_vkCmdBindPipeline vkCmdBindPipeline;
    PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets;
    PFN_vkCmdPushConstants vkCmdPushConstants;
    PFN_vkCmdDispatch vkCmdDispatch;
    PFN_vkCmdResetQueryPool vkCmdResetQueryPool;
    PFN_vkCmdWriteTimestamp vkCmdWriteTimestamp;
    PFN_vkGetQueryPoolResults vkGetQueryPoolResults;
    PFN_vkQueueSubmit vkQueueSubmit;
    PFN_vkQueueWaitIdle vkQueueWaitIdle;
};

struct geist_buffer {
    struct vulkan_state *owner;
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize bytes;
    enum geist_buffer_role role;
    unsigned int memory_flags;
    bool host_visible;
    bool host_coherent;
    void *mapped;
};

static void vulkan_buffer_destroy_internal(struct geist_backend *be,
                                           struct geist_buffer *buf);
static void vulkan_destroy_q4k_nt4_cache(struct geist_backend *be);
static void vulkan_decode_replay_invalidate(struct vulkan_state *st);
static void vulkan_remove_q4k_nt4_cache_for_source(
    struct geist_backend *be,
    const struct geist_buffer *src);
[[nodiscard]] static enum geist_status
vulkan_invalidate_if_needed(struct vulkan_state *st,
                            const struct geist_buffer *buf);
[[nodiscard]] static enum geist_status
vulkan_flush_if_needed(struct vulkan_state *st,
                       const struct geist_buffer *buf);
[[nodiscard]] static enum geist_status
vulkan_flush_decode_params(struct geist_backend *be);
static void vulkan_ffn_pre_barrier(struct vulkan_state *st,
                                   VkCommandBuffer cmd);
static void vulkan_ffn_post_barrier(struct vulkan_state *st,
                                    VkCommandBuffer cmd);
static bool vulkan_q6k_n4_enabled(void);

/* Element-level pipeline teardown. Each destroys one handle if live and nulls
 * it; callers guard st/device once up front. These collapse the otherwise
 * identical 6-line "if handle != NULL { destroy; null }" blocks repeated across
 * every vulkan_destroy_*_pipeline function below. */
static inline void vulkan_drop_pipeline(struct vulkan_state *st,
                                        VkPipeline *p) {
    if (*p != VK_NULL_HANDLE) {
        st->vkDestroyPipeline(st->device, *p, nullptr);
        *p = VK_NULL_HANDLE;
    }
}

static inline void vulkan_drop_pipeline_layout(struct vulkan_state *st,
                                               VkPipelineLayout *l) {
    if (*l != VK_NULL_HANDLE) {
        st->vkDestroyPipelineLayout(st->device, *l, nullptr);
        *l = VK_NULL_HANDLE;
    }
}

static inline void vulkan_drop_set_layout(struct vulkan_state *st,
                                          VkDescriptorSetLayout *s) {
    if (*s != VK_NULL_HANDLE) {
        st->vkDestroyDescriptorSetLayout(st->device, *s, nullptr);
        *s = VK_NULL_HANDLE;
    }
}

static void vulkan_destroy_add_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->add_pipeline);
    vulkan_drop_pipeline_layout(st, &st->add_pipeline_layout);
    vulkan_drop_set_layout(st, &st->add_set_layout);
}

static void vulkan_destroy_scale_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->scale_pipeline);
    vulkan_drop_pipeline_layout(st, &st->scale_pipeline_layout);
    vulkan_drop_set_layout(st, &st->scale_set_layout);
}

static void vulkan_destroy_mul_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->mul_pipeline);
    vulkan_drop_pipeline_layout(st, &st->mul_pipeline_layout);
    vulkan_drop_set_layout(st, &st->mul_set_layout);
}

static void vulkan_destroy_gelu_tanh_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->gelu_tanh_pipeline);
    vulkan_drop_pipeline_layout(st, &st->gelu_tanh_pipeline_layout);
    vulkan_drop_set_layout(st, &st->gelu_tanh_set_layout);
}

static void vulkan_destroy_gelu_tanh_mul_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->gelu_tanh_mul_pipeline);
    vulkan_drop_pipeline_layout(st, &st->gelu_tanh_mul_pipeline_layout);
    vulkan_drop_set_layout(st, &st->gelu_tanh_mul_set_layout);
}

static void vulkan_destroy_rmsnorm_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->rmsnorm_pipeline);
    vulkan_drop_pipeline_layout(st, &st->rmsnorm_pipeline_layout);
    vulkan_drop_set_layout(st, &st->rmsnorm_set_layout);
}

static void vulkan_destroy_rmsnorm_add_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->rmsnorm_add_pipeline);
    vulkan_drop_pipeline_layout(st, &st->rmsnorm_add_pipeline_layout);
    vulkan_drop_set_layout(st, &st->rmsnorm_add_set_layout);
}

static void vulkan_destroy_rmsnorm_rope_dyn_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->rmsnorm_rope_dyn_pipeline);
    vulkan_drop_pipeline_layout(st, &st->rmsnorm_rope_dyn_pipeline_layout);
    vulkan_drop_set_layout(st, &st->rmsnorm_rope_dyn_set_layout);
}

static void vulkan_destroy_matvec_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->matvec_pipeline);
    vulkan_drop_pipeline_layout(st, &st->matvec_pipeline_layout);
    vulkan_drop_set_layout(st, &st->matvec_set_layout);
}

static void vulkan_destroy_matmul_f32_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->matmul_f32_pipeline);
    vulkan_drop_pipeline_layout(st, &st->matmul_f32_pipeline_layout);
}

static void vulkan_destroy_matvec_f16bf16_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->matvec_f16bf16_pipeline);
    vulkan_drop_pipeline_layout(st, &st->matvec_f16bf16_pipeline_layout);
    vulkan_drop_set_layout(st, &st->matvec_f16bf16_set_layout);
}

static void vulkan_destroy_q4k_w4a8_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->matvec_q4k_w4a8_pipeline);
    vulkan_drop_pipeline(st, &st->q4k_quant_x_pipeline);
    vulkan_drop_pipeline_layout(st, &st->matvec_q4k_w4a8_pipeline_layout);
    vulkan_drop_pipeline_layout(st, &st->q4k_quant_x_pipeline_layout);
    vulkan_drop_set_layout(st, &st->matvec_q4k_w4a8_set_layout);
    vulkan_drop_set_layout(st, &st->q4k_quant_x_set_layout);
}

static void vulkan_destroy_matvec_q4k_nt4_subgroup_pipeline(
    struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->matvec_q4k_nt4_subgroup_pipeline);
    vulkan_drop_pipeline_layout(st,
                                &st->matvec_q4k_nt4_subgroup_pipeline_layout);
}

static void vulkan_destroy_matvec_q4k_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_destroy_q4k_w4a8_pipeline(st);
    vulkan_destroy_matvec_q4k_nt4_subgroup_pipeline(st);
    vulkan_drop_pipeline(st, &st->matvec_q4k_pipeline);
    vulkan_drop_pipeline(st, &st->matvec_q4k_n4_pipeline);
    vulkan_drop_pipeline(st, &st->matvec_q4k_nt4_pipeline);
    vulkan_drop_pipeline(st, &st->matvec_q4k_nt4_pair_pipeline);
    vulkan_drop_pipeline(st, &st->ffn_gate_up_gelu_q4k_pipeline);
    vulkan_drop_pipeline(st, &st->ffn_gate_up_gelu_q4k_nt4_pipeline);
    vulkan_drop_pipeline_layout(st, &st->matvec_q4k_pipeline_layout);
    vulkan_drop_pipeline_layout(st, &st->matvec_q4k_n4_pipeline_layout);
    vulkan_drop_pipeline_layout(st, &st->matvec_q4k_nt4_pipeline_layout);
    vulkan_drop_pipeline_layout(st, &st->matvec_q4k_nt4_pair_pipeline_layout);
    vulkan_drop_pipeline_layout(st, &st->ffn_gate_up_gelu_q4k_pipeline_layout);
    vulkan_drop_pipeline_layout(st,
                                &st->ffn_gate_up_gelu_q4k_nt4_pipeline_layout);
    vulkan_drop_set_layout(st, &st->matvec_q4k_nt4_pair_set_layout);
    vulkan_drop_set_layout(st, &st->ffn_gate_up_gelu_q4k_set_layout);
    vulkan_drop_set_layout(st, &st->ffn_gate_up_gelu_q4k_nt4_set_layout);
    vulkan_drop_set_layout(st, &st->matvec_q4k_set_layout);
}

static void vulkan_destroy_matmul_q4k_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->matmul_q4k_pipeline);
    vulkan_drop_pipeline_layout(st, &st->matmul_q4k_pipeline_layout);
}

static void vulkan_destroy_matmul_q4k_m8_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->matmul_q4k_m8_pipeline);
    vulkan_drop_pipeline_layout(st, &st->matmul_q4k_m8_pipeline_layout);
}

static void vulkan_destroy_matvec_q6k_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->matvec_q6k_pipeline);
    vulkan_drop_pipeline(st, &st->matvec_q6k_n4_pipeline);
    vulkan_drop_pipeline_layout(st, &st->matvec_q6k_pipeline_layout);
    vulkan_drop_pipeline_layout(st, &st->matvec_q6k_n4_pipeline_layout);
    vulkan_drop_set_layout(st, &st->matvec_q6k_set_layout);
}

static void vulkan_destroy_matmul_q6k_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->matmul_q6k_pipeline);
    vulkan_drop_pipeline_layout(st, &st->matmul_q6k_pipeline_layout);
}

static void vulkan_destroy_argmax_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    if (st->argmax_descriptor_pool != VK_NULL_HANDLE) {
        st->vkDestroyDescriptorPool(st->device, st->argmax_descriptor_pool,
                                    nullptr);
        st->argmax_descriptor_pool = VK_NULL_HANDLE;
        st->argmax_descriptor_set = VK_NULL_HANDLE;
    }
    vulkan_drop_pipeline(st, &st->argmax_pipeline);
    vulkan_drop_pipeline_layout(st, &st->argmax_pipeline_layout);
    vulkan_drop_set_layout(st, &st->argmax_set_layout);
}

static void vulkan_destroy_argmax_batch_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->argmax_batch_pipeline);
    vulkan_drop_pipeline_layout(st, &st->argmax_batch_pipeline_layout);
}

static void vulkan_destroy_embed_lookup_scaled_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->embed_lookup_scaled_pipeline);
    vulkan_drop_pipeline_layout(st, &st->embed_lookup_scaled_pipeline_layout);
    vulkan_drop_set_layout(st, &st->embed_lookup_scaled_set_layout);
}

static void vulkan_destroy_embed_lookup_scaled_dyn_pipeline(
    struct vulkan_state *st) {

    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->embed_lookup_scaled_dyn_pipeline);
    vulkan_drop_pipeline_layout(st,
                                &st->embed_lookup_scaled_dyn_pipeline_layout);
    vulkan_drop_set_layout(st, &st->embed_lookup_scaled_dyn_set_layout);
}

static void vulkan_destroy_rope_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->rope_pipeline);
    vulkan_drop_pipeline_layout(st, &st->rope_pipeline_layout);
    vulkan_drop_set_layout(st, &st->rope_set_layout);
}

static void vulkan_destroy_rope_dyn_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->rope_dyn_pipeline);
    vulkan_drop_pipeline_layout(st, &st->rope_dyn_pipeline_layout);
    vulkan_drop_set_layout(st, &st->rope_dyn_set_layout);
}

static void vulkan_destroy_attention_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->attention_pipeline);
    vulkan_drop_pipeline_layout(st, &st->attention_pipeline_layout);
    vulkan_drop_set_layout(st, &st->attention_set_layout);
}

static void vulkan_destroy_attention_dyn_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->attention_dyn_pipeline);
    vulkan_drop_pipeline_layout(st, &st->attention_dyn_pipeline_layout);
    vulkan_drop_set_layout(st, &st->attention_dyn_set_layout);
}

static void vulkan_destroy_kv_append_dyn_pipeline(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_drop_pipeline(st, &st->kv_append_dyn_pipeline);
    vulkan_drop_pipeline_layout(st, &st->kv_append_dyn_pipeline_layout);
    vulkan_drop_set_layout(st, &st->kv_append_dyn_set_layout);
}

static void vulkan_destroy_block_descriptor_pool(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    vulkan_decode_replay_invalidate(st);
    st->capture_set_cache_active = false;
    st->capture_set_cache_cursor = 0;
    st->capture_set_cache_count = 0;
    if (st->block_descriptor_pool != VK_NULL_HANDLE) {
        st->vkDestroyDescriptorPool(st->device, st->block_descriptor_pool,
                                    nullptr);
        st->block_descriptor_pool = VK_NULL_HANDLE;
    }
    st->block_descriptor_pool_sets = 0;
    st->block_descriptor_pool_storage_descriptors = 0;
}

static void vulkan_destroy_transient_descriptor_pool(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    if (st->transient_descriptor_pool != VK_NULL_HANDLE) {
        st->vkDestroyDescriptorPool(st->device, st->transient_descriptor_pool,
                                    nullptr);
        st->transient_descriptor_pool = VK_NULL_HANDLE;
    }
    st->transient_descriptor_pool_sets = 0;
    st->transient_descriptor_pool_storage_descriptors = 0;
}

static void vulkan_destroy_transient_command_pool(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    if (st->transient_command_pool != VK_NULL_HANDLE) {
        st->vkDestroyCommandPool(st->device, st->transient_command_pool,
                                 nullptr);
        st->transient_command_pool = VK_NULL_HANDLE;
        st->transient_command_buffer = VK_NULL_HANDLE;
    }
}

static void vulkan_destroy_profile_query_pool(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    if (st->profile_query_pool != VK_NULL_HANDLE) {
        st->vkDestroyQueryPool(st->device, st->profile_query_pool, nullptr);
        st->profile_query_pool = VK_NULL_HANDLE;
    }
    st->profile_transient_recorded = false;
    st->profile_capture_recorded = false;
    st->profile_capture_label = nullptr;
}

static void vulkan_destroy_decode_replay_command_pool(struct vulkan_state *st) {
    if (st == nullptr || st->device == VK_NULL_HANDLE) {
        return;
    }
    if (st->decode_replay_command_pool != VK_NULL_HANDLE) {
        st->vkDestroyCommandPool(st->device, st->decode_replay_command_pool,
                                 nullptr);
        st->decode_replay_command_pool = VK_NULL_HANDLE;
        st->decode_replay_command_buffer = VK_NULL_HANDLE;
    }
    if (st->decode_replay_fence != VK_NULL_HANDLE) {
        st->vkDestroyFence(st->device, st->decode_replay_fence, nullptr);
        st->decode_replay_fence = VK_NULL_HANDLE;
    }
    st->decode_replay_valid = false;
    st->decode_replay_recording = false;
    st->decode_replay_profile_recorded = false;
    st->decode_replay_vocab_size = 0;
    st->decode_replay_token_count = 0;
}

static bool vulkan_env_enabled(const char *name) {
    const char *value = getenv(name);
    return value != nullptr && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool vulkan_env_default_enabled(const char *name) {
    const char *value = getenv(name);
    return value == nullptr || value[0] == '\0' || strcmp(value, "0") != 0;
}

/* Strict opt-in: enabled only when the var is exactly "1". */
static bool vulkan_env_is_on(const char *name) {
    const char *env = getenv(name);
    return env != nullptr && env[0] == '1' && env[1] == '\0';
}

/* Default-on toggle: unset/empty -> true, exactly "0" -> false,
 * exactly "1" -> true, anything else -> false. */
static bool vulkan_env_toggle_default_on(const char *name) {
    const char *env = getenv(name);
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    if (env[0] == '0' && env[1] == '\0') {
        return false;
    }
    return env[0] == '1' && env[1] == '\0';
}

static bool vulkan_profile_enabled(struct vulkan_state *st) {
    if (st == nullptr) {
        return false;
    }
    if (!st->profile_env_checked) {
        st->profile_enabled =
            vulkan_env_enabled("GEIST_VULKAN_PROFILE") &&
            st->device_props.limits.timestampPeriod > 0.0f &&
            st->queue_timestamp_valid_bits > 0;
        st->profile_env_checked = true;
    }
    return st->profile_enabled;
}

[[nodiscard]] static enum geist_status
vulkan_ensure_profile_query_pool(struct geist_backend *be) {
    struct vulkan_state *st = be->state;
    if (st->profile_query_pool != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkQueryPoolCreateInfo qpci = {
        .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = 2,
    };
    VkResult rc = st->vkCreateQueryPool(st->device, &qpci, nullptr,
                                        &st->profile_query_pool);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan profile: vkCreateQueryPool failed (%d)", (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vulkan_profile_begin(struct geist_backend *be,
                     VkCommandBuffer cmd,
                     bool *out_recorded) {
    if (out_recorded != nullptr) {
        *out_recorded = false;
    }
    if (be == nullptr || be->state == nullptr || cmd == VK_NULL_HANDLE ||
        out_recorded == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct vulkan_state *st = be->state;
    if (!vulkan_profile_enabled(st)) {
        return GEIST_OK;
    }

    enum geist_status s = vulkan_ensure_profile_query_pool(be);
    if (s != GEIST_OK) {
        return s;
    }
    st->vkCmdResetQueryPool(cmd, st->profile_query_pool, 0, 2);
    st->vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            st->profile_query_pool, 0);
    *out_recorded = true;
    return GEIST_OK;
}

static void vulkan_profile_end(struct vulkan_state *st,
                               VkCommandBuffer cmd,
                               bool recorded) {
    if (st == nullptr || cmd == VK_NULL_HANDLE || !recorded) {
        return;
    }
    st->vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            st->profile_query_pool, 1);
}

static void vulkan_profile_add_sample(struct vulkan_state *st,
                                      const char *label,
                                      double ms) {
    if (st == nullptr || label == nullptr) {
        return;
    }
    for (uint32_t i = 0; i < st->profile_stat_count; i++) {
        struct vulkan_profile_stat *stat = &st->profile_stats[i];
        if (strcmp(stat->label, label) == 0) {
            stat->count++;
            stat->total_ms += ms;
            if (ms > stat->max_ms) {
                stat->max_ms = ms;
            }
            return;
        }
    }
    if (st->profile_stat_count >= VULKAN_PROFILE_MAX_STATS) {
        return;
    }
    struct vulkan_profile_stat *stat =
        &st->profile_stats[st->profile_stat_count++];
    *stat = (struct vulkan_profile_stat){
        .label = label,
        .count = 1,
        .total_ms = ms,
        .max_ms = ms,
    };
}

static void vulkan_profile_report(struct vulkan_state *st,
                                  const char *label,
                                  bool recorded) {
    if (st == nullptr || !recorded ||
        st->profile_query_pool == VK_NULL_HANDLE) {
        return;
    }
    uint64_t timestamps[2] = {0, 0};
    VkResult rc = st->vkGetQueryPoolResults(
        st->device, st->profile_query_pool, 0, 2, sizeof(timestamps),
        timestamps, sizeof(timestamps[0]),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (rc != VK_SUCCESS || label == nullptr) {
        return;
    }
    uint64_t delta = 0;
    if (st->queue_timestamp_valid_bits < 64u) {
        const uint64_t mask = (UINT64_C(1) << st->queue_timestamp_valid_bits) -
                              UINT64_C(1);
        const uint64_t begin = timestamps[0] & mask;
        const uint64_t end = timestamps[1] & mask;
        delta = end >= begin ? end - begin : (mask - begin) + end + 1u;
    } else if (timestamps[1] >= timestamps[0]) {
        delta = timestamps[1] - timestamps[0];
    } else {
        return;
    }
    const double ns =
        (double) delta *
        (double) st->device_props.limits.timestampPeriod;
    vulkan_profile_add_sample(st, label, ns / 1000000.0);
}

static void vulkan_profile_print_summary(struct vulkan_state *st) {
    if (st == nullptr || !st->profile_enabled ||
        st->profile_stat_count == 0) {
        return;
    }
    fprintf(stderr, "vulkan profile summary:\n");
    for (uint32_t i = 0; i < st->profile_stat_count; i++) {
        const struct vulkan_profile_stat *stat = &st->profile_stats[i];
        const double avg_ms = stat->count > 0
                                  ? stat->total_ms / (double) stat->count
                                  : 0.0;
        fprintf(stderr,
                "  %-32s count=%" PRIu64 " total=%.3f ms avg=%.3f ms max=%.3f ms\n",
                stat->label, stat->count, stat->total_ms, avg_ms,
                stat->max_ms);
    }
}

static const char *vulkan_command_sequence_label(
    enum geist_command_sequence_kind kind) {
    switch (kind) {
    case GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP:
        return "sequence_decode_layer_loop";
    case GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP:
        return "sequence_decode_greedy_step";
    case GEIST_COMMAND_SEQUENCE_PREFILL_TEXT:
        return "sequence_prefill_text";
    case GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY:
        return "sequence_verify_greedy";
    default:
        return "sequence";
    }
}

static bool vulkan_capture_set_cache_enabled(struct vulkan_state *st) {
    if (st == nullptr) {
        return false;
    }
    if (!st->capture_set_cache_env_checked) {
        st->capture_set_cache_enabled =
            vulkan_env_enabled("GEIST_VULKAN_CAPTURE_SET_CACHE");
        st->capture_set_cache_env_checked = true;
    }
    return st->capture_set_cache_enabled;
}

static bool vulkan_dynamic_decode_params_enabled(struct vulkan_state *st) {
    if (st == nullptr) {
        return false;
    }
    if (!st->dynamic_decode_params_env_checked) {
        st->dynamic_decode_params_enabled =
            vulkan_env_default_enabled("GEIST_VULKAN_DYNAMIC_DECODE_PARAMS");
        st->dynamic_decode_params_env_checked = true;
    }
    return st->dynamic_decode_params_enabled;
}

static bool vulkan_decode_replay_enabled(struct vulkan_state *st) {
    if (st == nullptr) {
        return false;
    }
    if (!vulkan_dynamic_decode_params_enabled(st)) {
        return false;
    }
    if (!st->decode_replay_env_checked) {
        st->decode_replay_enabled =
            vulkan_env_default_enabled("GEIST_VULKAN_DECODE_REPLAY");
        st->decode_replay_env_checked = true;
    }
    return st->decode_replay_enabled;
}

static void vulkan_decode_replay_invalidate(struct vulkan_state *st) {
    if (st == nullptr) {
        return;
    }
    st->decode_replay_valid = false;
    st->decode_replay_recording = false;
    st->decode_replay_profile_recorded = false;
    st->decode_replay_vocab_size = 0;
    st->decode_replay_token_count = 0;
}

static void vulkan_capture_set_cache_clear(struct vulkan_state *st) {
    if (st == nullptr) {
        return;
    }
    st->capture_set_cache_active = false;
    st->capture_set_cache_cursor = 0;
    st->capture_set_cache_count = 0;
    for (uint32_t i = 0; i < VULKAN_CAPTURE_SET_CACHE_MAX; i++) {
        st->capture_set_cache[i] =
            (struct vulkan_capture_set_cache_entry){0};
    }
}

[[nodiscard]] static enum geist_status
vulkan_alloc_block_descriptor_sets_cached(
    struct geist_backend *be,
    const char *label,
    uint32_t set_count,
    const VkDescriptorSetLayout layouts[static set_count],
    VkDescriptorSet sets[static set_count],
    bool reused[static set_count]) {

    if (be == nullptr || be->state == nullptr || set_count == 0 ||
        layouts == nullptr || sets == nullptr || reused == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct vulkan_state *st = be->state;
    for (uint32_t i = 0; i < set_count; i++) {
        sets[i] = VK_NULL_HANDLE;
        reused[i] = false;
    }

    if (!st->capture_set_cache_active) {
        const VkDescriptorSetAllocateInfo dsai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = st->block_descriptor_pool,
            .descriptorSetCount = set_count,
            .pSetLayouts = layouts,
        };
        VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai, sets);
        if (rc != VK_SUCCESS) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "vulkan %s: vkAllocateDescriptorSets failed (%d)",
                label, (int) rc);
            return GEIST_E_BACKEND;
        }
        return GEIST_OK;
    }

    for (uint32_t i = 0; i < set_count; i++) {
        const uint32_t slot = st->capture_set_cache_cursor++;
        if (slot < st->capture_set_cache_count &&
            st->capture_set_cache[slot].layout == layouts[i] &&
            st->capture_set_cache[slot].set != VK_NULL_HANDLE) {
            sets[i] = st->capture_set_cache[slot].set;
            reused[i] = true;
            continue;
        }

        const VkDescriptorSetAllocateInfo dsai = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = st->block_descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &layouts[i],
        };
        VkDescriptorSet set = VK_NULL_HANDLE;
        VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai, &set);
        if (rc != VK_SUCCESS) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "vulkan %s: cached vkAllocateDescriptorSets failed (%d)",
                label, (int) rc);
            return GEIST_E_BACKEND;
        }
        sets[i] = set;
        if (slot < VULKAN_CAPTURE_SET_CACHE_MAX) {
            if (slot >= st->capture_set_cache_count) {
                st->capture_set_cache_count = slot + 1u;
            }
            st->capture_set_cache[slot] =
                (struct vulkan_capture_set_cache_entry){
                    .layout = layouts[i],
                    .set = set,
                };
        }
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vulkan_ensure_transient_command_buffer(struct geist_backend *be) {
    struct vulkan_state *st = be->state;
    if (st->transient_command_buffer != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    if (st->transient_command_pool == VK_NULL_HANDLE) {
        const VkCommandPoolCreateInfo cpci = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = st->queue_family,
        };
        VkResult rc = st->vkCreateCommandPool(
            st->device, &cpci, nullptr, &st->transient_command_pool);
        if (rc != VK_SUCCESS) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "vulkan: vkCreateCommandPool(transient) failed (%d)",
                (int) rc);
            return GEIST_E_BACKEND;
        }
    }

    const VkCommandBufferAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = st->transient_command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkResult rc = st->vkAllocateCommandBuffers(
        st->device, &ai, &st->transient_command_buffer);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkAllocateCommandBuffers(transient) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vulkan_ensure_decode_replay_command_buffer(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct vulkan_state *st = be->state;
    if (st->decode_replay_command_buffer != VK_NULL_HANDLE &&
        st->decode_replay_fence != VK_NULL_HANDLE) {
        return GEIST_OK;
    }
    if (st->decode_replay_command_pool == VK_NULL_HANDLE) {
        const VkCommandPoolCreateInfo cpci = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = st->queue_family,
        };
        VkResult rc = st->vkCreateCommandPool(
            st->device, &cpci, nullptr, &st->decode_replay_command_pool);
        if (rc != VK_SUCCESS) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "vulkan: vkCreateCommandPool(decode replay) failed (%d)",
                (int) rc);
            return GEIST_E_BACKEND;
        }
    }

    VkResult rc = VK_SUCCESS;
    if (st->decode_replay_command_buffer == VK_NULL_HANDLE) {
        const VkCommandBufferAllocateInfo ai = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = st->decode_replay_command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        rc = st->vkAllocateCommandBuffers(
            st->device, &ai, &st->decode_replay_command_buffer);
        if (rc != VK_SUCCESS) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "vulkan: vkAllocateCommandBuffers(decode replay) failed (%d)",
                (int) rc);
            return GEIST_E_BACKEND;
        }
    }
    if (st->decode_replay_fence == VK_NULL_HANDLE) {
        const VkFenceCreateInfo fci = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        };
        rc = st->vkCreateFence(st->device, &fci, nullptr,
                               &st->decode_replay_fence);
        if (rc != VK_SUCCESS) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "vulkan: vkCreateFence(decode replay) failed (%d)",
                (int) rc);
            return GEIST_E_BACKEND;
        }
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vulkan_begin_transient_command_buffer(struct geist_backend *be,
                                      const char *label,
                                      VkCommandBuffer *out) {
    if (be == nullptr || be->state == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct vulkan_state *st = be->state;
    if (st->capture_active) {
        if (st->capture_command_buffer == VK_NULL_HANDLE) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "vulkan %s: command capture has no command buffer", label);
            return GEIST_E_BACKEND;
        }
        *out = st->capture_command_buffer;
        return GEIST_OK;
    }
    st->captured_greedy_token_pending = false;
    st->captured_greedy_vocab_size = 0;

    enum geist_status s = vulkan_ensure_transient_command_buffer(be);
    if (s != GEIST_OK) {
        return s;
    }

    VkResult rc = st->vkResetCommandPool(st->device,
                                         st->transient_command_pool, 0);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan %s: vkResetCommandPool(transient) failed (%d)",
            label, (int) rc);
        return GEIST_E_BACKEND;
    }
    if (vulkan_profile_enabled(st)) {
        s = vulkan_ensure_profile_query_pool(be);
        if (s != GEIST_OK) {
            return s;
        }
    }

    const VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    rc = st->vkBeginCommandBuffer(st->transient_command_buffer, &bi);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan %s: vkBeginCommandBuffer failed (%d)",
            label, (int) rc);
        return GEIST_E_BACKEND;
    }
    s = vulkan_profile_begin(be, st->transient_command_buffer,
                             &st->profile_transient_recorded);
    if (s != GEIST_OK) {
        return s;
    }

    *out = st->transient_command_buffer;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vulkan_submit_transient_command_buffer(struct geist_backend *be,
                                       VkCommandBuffer cmd,
                                       const char *label) {
    if (be == nullptr || be->state == nullptr || cmd == VK_NULL_HANDLE) {
        return GEIST_E_INVALID_ARG;
    }
    struct vulkan_state *st = be->state;
    if (st->capture_active && cmd == st->capture_command_buffer) {
        return GEIST_OK;
    }

    const bool profiled = st->profile_transient_recorded;
    vulkan_profile_end(st, cmd, profiled);
    VkResult rc = st->vkEndCommandBuffer(cmd);
    if (rc == VK_SUCCESS) {
        const VkSubmitInfo si = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
        };
        rc = st->vkQueueSubmit(st->queue, 1, &si, VK_NULL_HANDLE);
        if (rc == VK_SUCCESS) {
            rc = st->vkQueueWaitIdle(st->queue);
        }
    }
    if (rc != VK_SUCCESS) {
        st->profile_transient_recorded = false;
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan %s: command submission failed (%d)", label, (int) rc);
        return GEIST_E_BACKEND;
    }
    vulkan_profile_report(st, label, profiled);
    st->profile_transient_recorded = false;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vulkan_ensure_transient_descriptor_pool(struct geist_backend *be,
                                        uint32_t min_sets,
                                        uint32_t min_storage_descriptors) {

    struct vulkan_state *st = be->state;
    const uint32_t set_count =
        min_sets > VULKAN_DESCRIPTOR_POOL_DEFAULT_SETS
            ? min_sets
            : VULKAN_DESCRIPTOR_POOL_DEFAULT_SETS;
    const uint32_t descriptor_count =
        min_storage_descriptors >
                VULKAN_DESCRIPTOR_POOL_DEFAULT_STORAGE_DESCRIPTORS
            ? min_storage_descriptors
            : VULKAN_DESCRIPTOR_POOL_DEFAULT_STORAGE_DESCRIPTORS;
    if (st->transient_descriptor_pool != VK_NULL_HANDLE &&
        st->transient_descriptor_pool_sets >= set_count &&
        st->transient_descriptor_pool_storage_descriptors >=
            descriptor_count) {
        return GEIST_OK;
    }
    if (st->capture_active) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: transient descriptor pool too small during capture "
            "(sets=%u/%u descriptors=%u/%u)",
            st->transient_descriptor_pool_sets, set_count,
            st->transient_descriptor_pool_storage_descriptors,
            descriptor_count);
        return GEIST_E_BACKEND;
    }
    vulkan_destroy_transient_descriptor_pool(st);

    const VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = descriptor_count,
    };
    const VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = set_count,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    VkResult rc = st->vkCreateDescriptorPool(st->device, &dpci, nullptr,
                                             &st->transient_descriptor_pool);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorPool(transient) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    st->transient_descriptor_pool_sets = set_count;
    st->transient_descriptor_pool_storage_descriptors = descriptor_count;
    return GEIST_OK;
}

static void vulkan_free_transient_descriptor_sets(
    struct vulkan_state *st,
    uint32_t set_count,
    VkDescriptorSet sets[static set_count]) {

    if (st != nullptr && st->device != VK_NULL_HANDLE &&
        st->transient_descriptor_pool != VK_NULL_HANDLE && set_count > 0) {
        if (st->capture_active) {
            return;
        }
        (void) st->vkFreeDescriptorSets(st->device,
                                        st->transient_descriptor_pool,
                                        set_count, sets);
    }
}

[[nodiscard]] static enum geist_status
vulkan_ensure_block_descriptor_pool(struct geist_backend *be,
                                    uint32_t min_sets,
                                    uint32_t min_storage_descriptors) {

    struct vulkan_state *st = be->state;
    const uint32_t set_count =
        min_sets > VULKAN_DESCRIPTOR_POOL_DEFAULT_SETS
            ? min_sets
            : VULKAN_DESCRIPTOR_POOL_DEFAULT_SETS;
    const uint32_t descriptor_count =
        min_storage_descriptors >
                VULKAN_DESCRIPTOR_POOL_DEFAULT_STORAGE_DESCRIPTORS
            ? min_storage_descriptors
            : VULKAN_DESCRIPTOR_POOL_DEFAULT_STORAGE_DESCRIPTORS;
    if (st->block_descriptor_pool != VK_NULL_HANDLE &&
        st->block_descriptor_pool_sets >= set_count &&
        st->block_descriptor_pool_storage_descriptors >= descriptor_count) {
        return GEIST_OK;
    }
    if (st->capture_active) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: block descriptor pool too small during capture "
            "(sets=%u/%u descriptors=%u/%u)",
            st->block_descriptor_pool_sets, set_count,
            st->block_descriptor_pool_storage_descriptors,
            descriptor_count);
        return GEIST_E_BACKEND;
    }
    vulkan_destroy_block_descriptor_pool(st);

    const VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = descriptor_count,
    };
    const VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = set_count,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    VkResult rc = st->vkCreateDescriptorPool(st->device, &dpci, nullptr,
                                             &st->block_descriptor_pool);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorPool(block fastpath) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    st->block_descriptor_pool_sets = set_count;
    st->block_descriptor_pool_storage_descriptors = descriptor_count;
    return GEIST_OK;
}

static void vulkan_free_block_descriptor_sets(struct vulkan_state *st,
                                              uint32_t set_count,
                                              VkDescriptorSet sets[static set_count]) {
    if (st != nullptr && st->device != VK_NULL_HANDLE &&
        st->block_descriptor_pool != VK_NULL_HANDLE && set_count > 0) {
        if (st->capture_active) {
            return;
        }
        (void) st->vkFreeDescriptorSets(st->device,
                                        st->block_descriptor_pool,
                                        set_count, sets);
    }
}

[[nodiscard]] static enum geist_status
vulkan_reset_descriptor_pool_if_present(struct geist_backend *be,
                                        VkDescriptorPool pool,
                                        const char *label) {
    if (pool == VK_NULL_HANDLE) {
        return GEIST_OK;
    }
    struct vulkan_state *st = be->state;
    if (pool == st->block_descriptor_pool ||
        pool == st->transient_descriptor_pool) {
        vulkan_decode_replay_invalidate(st);
    }
    VkResult rc = st->vkResetDescriptorPool(st->device, pool, 0);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan %s: vkResetDescriptorPool failed (%d)", label, (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_command_sequence_begin(
    struct geist_backend *be,
    enum geist_command_sequence_kind kind,
    int *out_token) {

    if (be == nullptr || be->state == nullptr || out_token == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_token = 0;
    if (kind != GEIST_COMMAND_SEQUENCE_DECODE_LAYER_LOOP &&
        kind != GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP &&
        kind != GEIST_COMMAND_SEQUENCE_PREFILL_TEXT &&
        kind != GEIST_COMMAND_SEQUENCE_VERIFY_GREEDY) {
        return GEIST_E_UNSUPPORTED;
    }
    struct vulkan_state *st = be->state;
    if (st->capture_active) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan command sequence: nested capture");
        return GEIST_E_BACKEND;
    }
    const bool use_capture_set_cache =
        kind == GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP &&
        vulkan_capture_set_cache_enabled(st);
    const bool record_decode_replay =
        kind == GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP &&
        vulkan_decode_replay_enabled(st) &&
        !st->decode_replay_valid;

    enum geist_status s = record_decode_replay
                               ? vulkan_ensure_decode_replay_command_buffer(be)
                               : vulkan_ensure_transient_command_buffer(be);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ensure_transient_descriptor_pool(
        be, VULKAN_DESCRIPTOR_POOL_DECODE_CAPTURE_SETS,
        VULKAN_DESCRIPTOR_POOL_DECODE_CAPTURE_STORAGE_DESCRIPTORS);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ensure_block_descriptor_pool(
        be, VULKAN_DESCRIPTOR_POOL_DECODE_CAPTURE_SETS,
        VULKAN_DESCRIPTOR_POOL_DECODE_CAPTURE_STORAGE_DESCRIPTORS);
    if (s != GEIST_OK) { return s; }
    s = vulkan_reset_descriptor_pool_if_present(
        be, st->transient_descriptor_pool, "command_sequence_begin");
    if (s != GEIST_OK) { return s; }
    if (!use_capture_set_cache || st->capture_set_cache_count == 0) {
        s = vulkan_reset_descriptor_pool_if_present(
            be, st->block_descriptor_pool, "command_sequence_begin");
        if (s != GEIST_OK) { return s; }
        vulkan_capture_set_cache_clear(st);
    }

    VkCommandPool capture_pool = record_decode_replay
                                     ? st->decode_replay_command_pool
                                     : st->transient_command_pool;
    VkCommandBuffer capture_cmd = record_decode_replay
                                      ? st->decode_replay_command_buffer
                                      : st->transient_command_buffer;
    VkResult rc = st->vkResetCommandPool(st->device, capture_pool, 0);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan command_sequence_begin: vkResetCommandPool failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    if (vulkan_profile_enabled(st)) {
        s = vulkan_ensure_profile_query_pool(be);
        if (s != GEIST_OK) {
            return s;
        }
    }

    const VkCommandBufferBeginInfo bi = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = record_decode_replay
                     ? VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT
                     : VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    rc = st->vkBeginCommandBuffer(capture_cmd, &bi);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan command_sequence_begin: vkBeginCommandBuffer failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    s = vulkan_profile_begin(be, capture_cmd, &st->profile_capture_recorded);
    if (s != GEIST_OK) {
        return s;
    }

    if (st->capture_token_next <= 0) {
        st->capture_token_next = 1;
    }
    const int token = st->capture_token_next++;
    st->capture_active = true;
    st->capture_command_buffer = capture_cmd;
    st->capture_kind = kind;
    st->decode_replay_recording = record_decode_replay;
    st->decode_params = (struct vulkan_decode_params){0};
    st->captured_greedy_token_pending = false;
    st->captured_greedy_vocab_size = 0;
    st->captured_greedy_token_count = 0;
    st->profile_capture_label = vulkan_command_sequence_label(kind);
    st->capture_set_cache_active = use_capture_set_cache;
    st->capture_set_cache_cursor = 0;
    *out_token = token;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_command_sequence_end(
    struct geist_backend *be,
    int token,
    bool submit) {

    if (be == nullptr || be->state == nullptr || token <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    struct vulkan_state *st = be->state;
    if (!st->capture_active ||
        st->capture_command_buffer == VK_NULL_HANDLE) {
        return GEIST_E_INVALID_ARG;
    }

    VkCommandBuffer cmd = st->capture_command_buffer;
    const bool replay_recording = st->decode_replay_recording;
    const bool profiled = st->profile_capture_recorded;
    const char *profile_label = st->profile_capture_label;
    st->capture_active = false;
    st->capture_command_buffer = VK_NULL_HANDLE;
    st->capture_kind = (enum geist_command_sequence_kind) 0;
    st->decode_replay_recording = false;
    st->capture_set_cache_active = false;
    st->profile_capture_recorded = false;
    st->profile_capture_label = nullptr;

    VkResult rc = VK_SUCCESS;
    if (submit) {
        vulkan_profile_end(st, cmd, profiled);
        rc = st->vkEndCommandBuffer(cmd);
        if (rc == VK_SUCCESS) {
            const VkSubmitInfo si = {
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1,
                .pCommandBuffers = &cmd,
            };
            rc = st->vkQueueSubmit(st->queue, 1, &si, VK_NULL_HANDLE);
            if (rc == VK_SUCCESS) {
                rc = st->vkQueueWaitIdle(st->queue);
            }
        }
    } else {
        if (replay_recording) {
            vulkan_decode_replay_invalidate(st);
        }
        st->captured_greedy_token_pending = false;
        st->captured_greedy_vocab_size = 0;
        st->captured_greedy_token_count = 0;
        VkCommandPool pool = replay_recording
                                 ? st->decode_replay_command_pool
                                 : st->transient_command_pool;
        rc = st->vkResetCommandPool(st->device, pool, 0);
    }
    if (rc != VK_SUCCESS) {
        if (replay_recording) {
            vulkan_decode_replay_invalidate(st);
        }
        st->captured_greedy_token_pending = false;
        st->captured_greedy_vocab_size = 0;
        st->captured_greedy_token_count = 0;
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan command_sequence_end: command %s failed (%d)",
            submit ? "submission" : "discard", (int) rc);
        return GEIST_E_BACKEND;
    }
    if (submit) {
        if (replay_recording) {
            if (st->captured_greedy_token_pending &&
                st->captured_greedy_vocab_size > 0 &&
                st->captured_greedy_token_count > 0) {
                st->decode_replay_valid = true;
                st->decode_replay_profile_recorded = profiled;
                st->decode_replay_vocab_size = st->captured_greedy_vocab_size;
                st->decode_replay_token_count =
                    st->captured_greedy_token_count;
            } else {
                vulkan_decode_replay_invalidate(st);
            }
        }
        vulkan_profile_report(st, profile_label, profiled);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_command_sequence_read_token(
    struct geist_backend *be,
    geist_token_t *out_token) {

    if (be == nullptr || be->state == nullptr || out_token == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct vulkan_state *st = be->state;
    if (st->capture_active || !st->captured_greedy_token_pending ||
        st->argmax_result_buffer == nullptr) {
        return GEIST_E_INVALID_STATE;
    }

    enum geist_status s =
        vulkan_invalidate_if_needed(st, st->argmax_result_buffer);
    if (s != GEIST_OK) {
        st->captured_greedy_token_pending = false;
        st->captured_greedy_token_count = 0;
        return s;
    }
    uint32_t token_u32 = 0;
    memcpy(&token_u32, st->argmax_result_buffer->mapped, sizeof(token_u32));
    if (token_u32 >= st->captured_greedy_vocab_size ||
        token_u32 > (uint32_t) INT32_MAX) {
        st->captured_greedy_token_pending = false;
        st->captured_greedy_vocab_size = 0;
        st->captured_greedy_token_count = 0;
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan command_sequence_read_token: invalid token id %u",
            token_u32);
        return GEIST_E_BACKEND;
    }
    st->captured_greedy_token_pending = false;
    st->captured_greedy_vocab_size = 0;
    st->captured_greedy_token_count = 0;
    *out_token = (geist_token_t) token_u32;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_command_sequence_read_tokens(
    struct geist_backend *be,
    size_t n,
    geist_token_t out_tokens[static n]) {

    if (be == nullptr || be->state == nullptr ||
        (n > 0 && out_tokens == nullptr)) {
        return GEIST_E_INVALID_ARG;
    }
    if (n == 0 || n > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    struct vulkan_state *st = be->state;
    if (st->capture_active || !st->captured_greedy_token_pending ||
        st->argmax_result_buffer == nullptr ||
        st->captured_greedy_token_count < n ||
        st->argmax_result_capacity < n) {
        return GEIST_E_INVALID_STATE;
    }

    enum geist_status s =
        vulkan_invalidate_if_needed(st, st->argmax_result_buffer);
    if (s != GEIST_OK) {
        st->captured_greedy_token_pending = false;
        st->captured_greedy_token_count = 0;
        return s;
    }

    for (size_t i = 0; i < n; i++) {
        uint32_t token_u32 = 0;
        memcpy(&token_u32,
               (const uint8_t *) st->argmax_result_buffer->mapped +
                   i * sizeof(uint32_t),
               sizeof(token_u32));
        if (token_u32 >= st->captured_greedy_vocab_size ||
            token_u32 > (uint32_t) INT32_MAX) {
            st->captured_greedy_token_pending = false;
            st->captured_greedy_vocab_size = 0;
            st->captured_greedy_token_count = 0;
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "vulkan command_sequence_read_tokens: invalid token id %u",
                token_u32);
            return GEIST_E_BACKEND;
        }
        out_tokens[i] = (geist_token_t) token_u32;
    }

    st->captured_greedy_token_pending = false;
    st->captured_greedy_vocab_size = 0;
    st->captured_greedy_token_count = 0;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vulkan_command_sequence_replay_decode_greedy_step(
    struct geist_backend *be,
    geist_token_t token_id,
    size_t q_position,
    geist_token_t *out_token) {

    if (be == nullptr || be->state == nullptr || out_token == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_token = -1;
    struct vulkan_state *st = be->state;
    if (st->capture_active ||
        !vulkan_decode_replay_enabled(st) ||
        !st->decode_replay_valid ||
        st->decode_replay_command_buffer == VK_NULL_HANDLE ||
        st->argmax_result_buffer == nullptr ||
        st->decode_replay_vocab_size == 0 ||
        st->decode_replay_token_count == 0) {
        return GEIST_E_UNSUPPORTED;
    }
    if (token_id < 0 ||
        (uint64_t) token_id > UINT32_MAX ||
        q_position >= UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }

    st->decode_params.token_id = (uint32_t) token_id;
    st->decode_params.q_position = (uint32_t) q_position;
    st->decode_params.kv_len = (uint32_t) q_position + 1u;
    st->decode_params.token_output_offset = 0;
    enum geist_status s = vulkan_flush_decode_params(be);
    if (s != GEIST_OK) {
        return s;
    }

    const VkSubmitInfo si = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &st->decode_replay_command_buffer,
    };
    if (st->decode_replay_fence == VK_NULL_HANDLE) {
        vulkan_decode_replay_invalidate(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan decode replay: missing replay fence");
        return GEIST_E_BACKEND;
    }
    VkResult rc = st->vkResetFences(st->device, 1, &st->decode_replay_fence);
    if (rc == VK_SUCCESS) {
        rc = st->vkQueueSubmit(st->queue, 1, &si, st->decode_replay_fence);
    }
    if (rc == VK_SUCCESS) {
        rc = st->vkWaitForFences(st->device, 1, &st->decode_replay_fence,
                                 VK_TRUE, UINT64_MAX);
    }
    if (rc != VK_SUCCESS) {
        vulkan_decode_replay_invalidate(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan decode replay: command submission failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    vulkan_profile_report(st, "decode_replay_greedy_step",
                          st->decode_replay_profile_recorded);

    s = vulkan_invalidate_if_needed(st, st->argmax_result_buffer);
    if (s != GEIST_OK) {
        return s;
    }
    uint32_t token_u32 = 0;
    memcpy(&token_u32, st->argmax_result_buffer->mapped, sizeof(token_u32));
    if (token_u32 >= st->decode_replay_vocab_size ||
        token_u32 > (uint32_t) INT32_MAX) {
        vulkan_decode_replay_invalidate(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan decode replay: invalid token id %u", token_u32);
        return GEIST_E_BACKEND;
    }
    *out_token = (geist_token_t) token_u32;
    return GEIST_OK;
}

static const char *const vulkan_loader_candidates[] = {
#if defined(__APPLE__)
    "libvulkan.1.dylib",
    "libvulkan.dylib",
    "libMoltenVK.dylib",
    "/opt/homebrew/lib/libvulkan.1.dylib",
    "/opt/homebrew/lib/libvulkan.dylib",
    "/opt/homebrew/lib/libMoltenVK.dylib",
    "/usr/local/lib/libvulkan.1.dylib",
    "/usr/local/lib/libvulkan.dylib",
    "/usr/local/lib/libMoltenVK.dylib",
#else
    "libvulkan.so.1",
    "libvulkan.so",
#endif
    nullptr,
};

static bool vulkan_load_symbol(struct vulkan_state *st, const char *name,
                               void *out_fn, size_t out_fn_size) {
    void *sym = dlsym(st->loader_handle, name);
    if (sym == nullptr || out_fn_size != sizeof(sym)) {
        return false;
    }
    memcpy(out_fn, &sym, sizeof(sym));
    return true;
}

static bool vulkan_load_device_symbol(struct vulkan_state *st,
                                      const char *name,
                                      void *out_fn,
                                      size_t out_fn_size) {
    if (st == nullptr || st->device == VK_NULL_HANDLE ||
        st->vkGetDeviceProcAddr == nullptr) {
        return false;
    }
    PFN_vkVoidFunction sym = st->vkGetDeviceProcAddr(st->device, name);
    if (sym == nullptr || out_fn_size != sizeof(sym)) {
        return false;
    }
    memcpy(out_fn, &sym, sizeof(sym));
    return true;
}

#define VULKAN_LOAD(st, name) \
    do { \
        if (!vulkan_load_symbol((st), #name, &(st)->name, sizeof((st)->name))) { \
            return false; \
        } \
    } while (0)

#define VULKAN_LOAD_DEVICE(be, st, name) \
    do { \
        if (!vulkan_load_device_symbol((st), #name, &(st)->name, sizeof((st)->name))) { \
            geist_backend_set_error((be), GEIST_E_BACKEND, \
                                    "vulkan: vkGetDeviceProcAddr(%s) failed", \
                                    #name); \
            return GEIST_E_BACKEND; \
        } \
    } while (0)

static bool vulkan_load_core_symbols(struct vulkan_state *st) {
    VULKAN_LOAD(st, vkGetInstanceProcAddr);
    VULKAN_LOAD(st, vkGetDeviceProcAddr);
    VULKAN_LOAD(st, vkEnumerateInstanceExtensionProperties);
    VULKAN_LOAD(st, vkCreateInstance);
    VULKAN_LOAD(st, vkDestroyInstance);
    VULKAN_LOAD(st, vkEnumeratePhysicalDevices);
    VULKAN_LOAD(st, vkGetPhysicalDeviceQueueFamilyProperties);
    VULKAN_LOAD(st, vkGetPhysicalDeviceMemoryProperties);
    VULKAN_LOAD(st, vkGetPhysicalDeviceProperties);
    VULKAN_LOAD(st, vkGetPhysicalDeviceProperties2);
    VULKAN_LOAD(st, vkGetPhysicalDeviceFeatures2);
    VULKAN_LOAD(st, vkEnumerateDeviceExtensionProperties);
    VULKAN_LOAD(st, vkCreateDevice);
    VULKAN_LOAD(st, vkDestroyDevice);
    VULKAN_LOAD(st, vkGetDeviceQueue);
    VULKAN_LOAD(st, vkCreateBuffer);
    VULKAN_LOAD(st, vkDestroyBuffer);
    VULKAN_LOAD(st, vkGetBufferMemoryRequirements);
    VULKAN_LOAD(st, vkAllocateMemory);
    VULKAN_LOAD(st, vkFreeMemory);
    VULKAN_LOAD(st, vkBindBufferMemory);
    VULKAN_LOAD(st, vkMapMemory);
    VULKAN_LOAD(st, vkUnmapMemory);
    VULKAN_LOAD(st, vkFlushMappedMemoryRanges);
    VULKAN_LOAD(st, vkInvalidateMappedMemoryRanges);
    VULKAN_LOAD(st, vkCreateShaderModule);
    VULKAN_LOAD(st, vkDestroyShaderModule);
    VULKAN_LOAD(st, vkCreatePipelineCache);
    VULKAN_LOAD(st, vkDestroyPipelineCache);
    VULKAN_LOAD(st, vkCreateDescriptorSetLayout);
    VULKAN_LOAD(st, vkDestroyDescriptorSetLayout);
    VULKAN_LOAD(st, vkCreatePipelineLayout);
    VULKAN_LOAD(st, vkDestroyPipelineLayout);
    VULKAN_LOAD(st, vkCreateComputePipelines);
    VULKAN_LOAD(st, vkDestroyPipeline);
    VULKAN_LOAD(st, vkCreateDescriptorPool);
    VULKAN_LOAD(st, vkDestroyDescriptorPool);
    VULKAN_LOAD(st, vkCreateQueryPool);
    VULKAN_LOAD(st, vkDestroyQueryPool);
    VULKAN_LOAD(st, vkCreateFence);
    VULKAN_LOAD(st, vkDestroyFence);
    VULKAN_LOAD(st, vkResetFences);
    VULKAN_LOAD(st, vkWaitForFences);
    VULKAN_LOAD(st, vkAllocateDescriptorSets);
    VULKAN_LOAD(st, vkFreeDescriptorSets);
    VULKAN_LOAD(st, vkUpdateDescriptorSets);
    VULKAN_LOAD(st, vkCreateCommandPool);
    VULKAN_LOAD(st, vkDestroyCommandPool);
    VULKAN_LOAD(st, vkAllocateCommandBuffers);
    VULKAN_LOAD(st, vkFreeCommandBuffers);
    VULKAN_LOAD(st, vkBeginCommandBuffer);
    VULKAN_LOAD(st, vkEndCommandBuffer);
    VULKAN_LOAD(st, vkCmdCopyBuffer);
    VULKAN_LOAD(st, vkCmdPipelineBarrier);
    VULKAN_LOAD(st, vkCmdBindPipeline);
    VULKAN_LOAD(st, vkCmdBindDescriptorSets);
    VULKAN_LOAD(st, vkCmdPushConstants);
    VULKAN_LOAD(st, vkCmdDispatch);
    VULKAN_LOAD(st, vkCmdResetQueryPool);
    VULKAN_LOAD(st, vkCmdWriteTimestamp);
    VULKAN_LOAD(st, vkGetQueryPoolResults);
    VULKAN_LOAD(st, vkQueueSubmit);
    VULKAN_LOAD(st, vkQueueWaitIdle);
    return true;
}

[[nodiscard]] static enum geist_status vulkan_load_loader(
    struct geist_backend *be,
    struct vulkan_state *st) {

    for (size_t i = 0; vulkan_loader_candidates[i] != nullptr; i++) {
        const char *candidate = vulkan_loader_candidates[i];
        void *loader = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
        if (loader == nullptr) {
            continue;
        }
        st->loader_handle = loader;
        st->loader_name = candidate;
        if (!vulkan_load_core_symbols(st)) {
            dlclose(loader);
            st->loader_handle = nullptr;
            st->loader_name = nullptr;
            continue;
        }
        return GEIST_OK;
    }

    geist_backend_set_error(be, GEIST_E_BACKEND,
                            "vulkan: Vulkan loader not found; install Vulkan "
                            "Loader or MoltenVK and ensure it is visible via "
                            "the dynamic loader path");
    return GEIST_E_BACKEND;
}

static bool vulkan_instance_extension_available(struct vulkan_state *st,
                                                const char *name) {
    uint32_t count = 0;
    if (st->vkEnumerateInstanceExtensionProperties(
            nullptr, &count, nullptr) != VK_SUCCESS || count == 0) {
        return false;
    }
    VkExtensionProperties props[64];
    if (count > 64) {
        count = 64;
    }
    if (st->vkEnumerateInstanceExtensionProperties(
            nullptr, &count, props) != VK_SUCCESS) {
        return false;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (strcmp(props[i].extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

static bool vulkan_device_extension_available(struct vulkan_state *st,
                                              VkPhysicalDevice physical_device,
                                              const char *name) {
    uint32_t count = 0;
    if (st->vkEnumerateDeviceExtensionProperties(
            physical_device, nullptr, &count, nullptr) != VK_SUCCESS || count == 0) {
        return false;
    }
    VkExtensionProperties props[128];
    if (count > 128) {
        count = 128;
    }
    if (st->vkEnumerateDeviceExtensionProperties(
            physical_device, nullptr, &count, props) != VK_SUCCESS) {
        return false;
    }
    for (uint32_t i = 0; i < count; i++) {
        if (strcmp(props[i].extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] static enum geist_status vulkan_create_instance(
    struct geist_backend *be,
    struct vulkan_state *st) {

    const char *extensions[4];
    uint32_t extension_count = 0;
    VkInstanceCreateFlags flags = 0;
    if (vulkan_instance_extension_available(st, "VK_KHR_portability_enumeration")) {
        extensions[extension_count++] = "VK_KHR_portability_enumeration";
#ifdef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
        flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#else
        flags |= (VkInstanceCreateFlags) 0x00000001u;
#endif
    }

    const VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "geist",
        .applicationVersion = VK_MAKE_VERSION(0, 2, 1),
        .pEngineName = "geist",
        .engineVersion = VK_MAKE_VERSION(0, 2, 1),
        .apiVersion = VK_API_VERSION_1_1,
    };
    const VkInstanceCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .flags = flags,
        .pApplicationInfo = &app,
        .enabledExtensionCount = extension_count,
        .ppEnabledExtensionNames = extensions,
    };
    const VkResult rc = st->vkCreateInstance(&ci, nullptr, &st->instance);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateInstance failed (%d)", (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

static uint64_t vulkan_device_local_bytes(const struct vulkan_state *st) {
    uint64_t total = 0;
    for (uint32_t i = 0; i < st->memory_props.memoryHeapCount; i++) {
        const VkMemoryHeap *heap = &st->memory_props.memoryHeaps[i];
        if ((heap->flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            total += (uint64_t) heap->size;
        }
    }
    return total;
}

static void vulkan_probe_physical_device_features(struct vulkan_state *st) {
    st->vkGetPhysicalDeviceProperties(st->physical_device, &st->device_props);

    VkPhysicalDeviceSubgroupProperties subgroup_props = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 props2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &subgroup_props,
    };
    st->vkGetPhysicalDeviceProperties2(st->physical_device, &props2);
    st->has_subgroup_basic =
        (subgroup_props.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0;
    st->has_subgroup_arithmetic =
        (subgroup_props.supportedOperations &
         VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;

    VkPhysicalDeviceFeatures2 features2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    };
    void **tail = &features2.pNext;

#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES)
    VkPhysicalDeviceShaderIntegerDotProductFeatures dot_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
    };
    *tail = &dot_features;
    tail = &dot_features.pNext;
#endif

#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES)
    VkPhysicalDeviceDescriptorIndexingFeatures descriptor_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
    };
    *tail = &descriptor_features;
    tail = &descriptor_features.pNext;
#endif

#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES)
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
    };
    *tail = &timeline_features;
    tail = &timeline_features.pNext;
#endif
    (void) tail;

    st->vkGetPhysicalDeviceFeatures2(st->physical_device, &features2);

#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES)
    st->has_shader_integer_dot_product =
        dot_features.shaderIntegerDotProduct == VK_TRUE;
#else
    st->has_shader_integer_dot_product = false;
#endif

#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES)
    st->has_descriptor_indexing =
        descriptor_features.runtimeDescriptorArray == VK_TRUE &&
        descriptor_features.descriptorBindingPartiallyBound == VK_TRUE;
#else
    st->has_descriptor_indexing = false;
#endif

#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES)
    st->has_timeline_semaphore =
        timeline_features.timelineSemaphore == VK_TRUE;
#else
    st->has_timeline_semaphore = false;
#endif
}

[[nodiscard]] static enum geist_status vulkan_choose_device(
    struct geist_backend *be,
    struct vulkan_state *st) {

    uint32_t device_count = 0;
    VkResult rc = st->vkEnumeratePhysicalDevices(st->instance, &device_count, nullptr);
    if (rc != VK_SUCCESS || device_count == 0) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: no physical devices found (%d)", (int) rc);
        return GEIST_E_BACKEND;
    }
    VkPhysicalDevice devices[16];
    if (device_count > 16) {
        device_count = 16;
    }
    rc = st->vkEnumeratePhysicalDevices(st->instance, &device_count, devices);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: enumerate physical devices failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    for (uint32_t di = 0; di < device_count; di++) {
        uint32_t queue_count = 0;
        st->vkGetPhysicalDeviceQueueFamilyProperties(devices[di], &queue_count, nullptr);
        if (queue_count == 0) {
            continue;
        }
        VkQueueFamilyProperties queues[32];
        if (queue_count > 32) {
            queue_count = 32;
        }
        st->vkGetPhysicalDeviceQueueFamilyProperties(devices[di], &queue_count, queues);
        for (uint32_t qi = 0; qi < queue_count; qi++) {
            if ((queues[qi].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                st->physical_device = devices[di];
                st->queue_family = qi;
                st->queue_timestamp_valid_bits = queues[qi].timestampValidBits;
                st->vkGetPhysicalDeviceMemoryProperties(
                    st->physical_device, &st->memory_props);
                st->has_compute_queue = true;
                vulkan_probe_physical_device_features(st);
                return GEIST_OK;
            }
        }
    }

    geist_backend_set_error(be, GEIST_E_BACKEND,
                            "vulkan: no compute-capable queue family found");
    return GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status vulkan_create_device(
    struct geist_backend *be,
    struct vulkan_state *st) {

    const float priority = 1.0f;
    const VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = st->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };

    const char *extensions[8];
    uint32_t extension_count = 0;
    if (vulkan_device_extension_available(
            st, st->physical_device, "VK_KHR_portability_subset")) {
        extensions[extension_count++] = "VK_KHR_portability_subset";
    }
    if (vulkan_device_extension_available(
            st, st->physical_device, "VK_KHR_shader_integer_dot_product")) {
        extensions[extension_count++] = "VK_KHR_shader_integer_dot_product";
    }
    if (vulkan_device_extension_available(
            st, st->physical_device, "VK_EXT_descriptor_indexing")) {
        extensions[extension_count++] = "VK_EXT_descriptor_indexing";
    }
    if (vulkan_device_extension_available(
            st, st->physical_device, "VK_KHR_timeline_semaphore")) {
        extensions[extension_count++] = "VK_KHR_timeline_semaphore";
    }

    VkPhysicalDeviceFeatures2 features2 = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
    };
    void **tail = &features2.pNext;

#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES)
    VkPhysicalDeviceShaderIntegerDotProductFeatures dot_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
        .shaderIntegerDotProduct =
            st->has_shader_integer_dot_product ? VK_TRUE : VK_FALSE,
    };
    *tail = &dot_features;
    tail = &dot_features.pNext;
#endif

#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES)
    VkPhysicalDeviceDescriptorIndexingFeatures descriptor_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
        .runtimeDescriptorArray =
            st->has_descriptor_indexing ? VK_TRUE : VK_FALSE,
        .descriptorBindingPartiallyBound =
            st->has_descriptor_indexing ? VK_TRUE : VK_FALSE,
    };
    *tail = &descriptor_features;
    tail = &descriptor_features.pNext;
#endif

#if defined(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES)
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
        .timelineSemaphore =
            st->has_timeline_semaphore ? VK_TRUE : VK_FALSE,
    };
    *tail = &timeline_features;
    tail = &timeline_features.pNext;
#endif
    (void) tail;

    const VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &features2,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
        .enabledExtensionCount = extension_count,
        .ppEnabledExtensionNames = extensions,
    };
    VkResult rc = st->vkCreateDevice(st->physical_device, &dci, nullptr, &st->device);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateDevice failed (%d)", (int) rc);
        return GEIST_E_BACKEND;
    }
    VULKAN_LOAD_DEVICE(be, st, vkResetCommandPool);
    VULKAN_LOAD_DEVICE(be, st, vkResetDescriptorPool);
    st->vkGetDeviceQueue(st->device, st->queue_family, 0, &st->queue);

    const VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = st->queue_family,
    };
    rc = st->vkCreateCommandPool(st->device, &cpci, nullptr, &st->command_pool);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateCommandPool failed (%d)", (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPipelineCacheCreateInfo pcci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
    };
    rc = st->vkCreatePipelineCache(st->device, &pcci, nullptr,
                                   &st->pipeline_cache);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreatePipelineCache failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_add_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->add_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[3] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->add_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateDescriptorSetLayout(add) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 4u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->add_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->add_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_add_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreatePipelineLayout(add) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_add_f32_spv_len,
        .pCode = (const uint32_t *) vulkan_add_f32_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_add_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateShaderModule(add) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->add_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr, &st->add_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_add_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateComputePipelines(add) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_scale_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->scale_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[2] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->scale_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateDescriptorSetLayout(scale) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 4u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->scale_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->scale_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_scale_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreatePipelineLayout(scale) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_scale_f32_spv_len,
        .pCode = (const uint32_t *) vulkan_scale_f32_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_scale_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateShaderModule(scale) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->scale_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr, &st->scale_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_scale_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateComputePipelines(scale) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_mul_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->mul_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[3] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->mul_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateDescriptorSetLayout(mul) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 4u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->mul_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->mul_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_mul_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreatePipelineLayout(mul) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_mul_f32_spv_len,
        .pCode = (const uint32_t *) vulkan_mul_f32_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_mul_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateShaderModule(mul) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->mul_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr, &st->mul_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_mul_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateComputePipelines(mul) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_gelu_tanh_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->gelu_tanh_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[2] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->gelu_tanh_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateDescriptorSetLayout(gelu_tanh) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 4u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->gelu_tanh_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->gelu_tanh_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_gelu_tanh_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreatePipelineLayout(gelu_tanh) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_gelu_tanh_f32_spv_len,
        .pCode = (const uint32_t *) vulkan_gelu_tanh_f32_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_gelu_tanh_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateShaderModule(gelu_tanh) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->gelu_tanh_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr, &st->gelu_tanh_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_gelu_tanh_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateComputePipelines(gelu_tanh) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_rmsnorm_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->rmsnorm_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[3] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->rmsnorm_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateDescriptorSetLayout(rmsnorm) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 6u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->rmsnorm_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->rmsnorm_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_rmsnorm_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreatePipelineLayout(rmsnorm) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_rmsnorm_f32_spv_len,
        .pCode = (const uint32_t *) vulkan_rmsnorm_f32_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_rmsnorm_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateShaderModule(rmsnorm) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->rmsnorm_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr, &st->rmsnorm_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_rmsnorm_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateComputePipelines(rmsnorm) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_rmsnorm_add_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->rmsnorm_add_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[4] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->rmsnorm_add_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorSetLayout(rmsnorm_add) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 7u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->rmsnorm_add_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->rmsnorm_add_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_rmsnorm_add_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(rmsnorm_add) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_rmsnorm_add_f32_spv_len,
        .pCode = (const uint32_t *) vulkan_rmsnorm_add_f32_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_rmsnorm_add_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(rmsnorm_add) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->rmsnorm_add_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr,
                                      &st->rmsnorm_add_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_rmsnorm_add_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(rmsnorm_add) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_gelu_tanh_mul_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->gelu_tanh_mul_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[3] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->gelu_tanh_mul_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorSetLayout(gelu_tanh_mul) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 4u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->gelu_tanh_mul_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->gelu_tanh_mul_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_gelu_tanh_mul_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(gelu_tanh_mul) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_gelu_tanh_mul_f32_spv_len,
        .pCode = (const uint32_t *) vulkan_gelu_tanh_mul_f32_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_gelu_tanh_mul_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(gelu_tanh_mul) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->gelu_tanh_mul_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr,
                                      &st->gelu_tanh_mul_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_gelu_tanh_mul_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(gelu_tanh_mul) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vulkan_ensure_rmsnorm_rope_dyn_pipeline(struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->rmsnorm_rope_dyn_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[5] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 4,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 5,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->rmsnorm_rope_dyn_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorSetLayout(rmsnorm_rope_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 8u * sizeof(uint32_t) + sizeof(float),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->rmsnorm_rope_dyn_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->rmsnorm_rope_dyn_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_rmsnorm_rope_dyn_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(rmsnorm_rope_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_rmsnorm_rope_f32_dyn_spv_len,
        .pCode = (const uint32_t *) vulkan_rmsnorm_rope_f32_dyn_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_rmsnorm_rope_dyn_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(rmsnorm_rope_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->rmsnorm_rope_dyn_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr,
                                      &st->rmsnorm_rope_dyn_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_rmsnorm_rope_dyn_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(rmsnorm_rope_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_matvec_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->matvec_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[3] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->matvec_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateDescriptorSetLayout(matvec) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 5u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->matvec_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->matvec_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreatePipelineLayout(matvec) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_matvec_f32_spv_len,
        .pCode = (const uint32_t *) vulkan_matvec_f32_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateShaderModule(matvec) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->matvec_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr, &st->matvec_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateComputePipelines(matvec) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_matmul_f32_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->matmul_f32_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    enum geist_status s = vulkan_ensure_matvec_pipeline(be);
    if (s != GEIST_OK) { return s; }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 8u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->matvec_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    VkResult rc = st->vkCreatePipelineLayout(
        st->device, &plci, nullptr, &st->matmul_f32_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matmul_f32_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(matmul_f32) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_matmul_f32_spv_len,
        .pCode = (const uint32_t *) vulkan_matmul_f32_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matmul_f32_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(matmul_f32) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->matmul_f32_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr,
                                      &st->matmul_f32_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matmul_f32_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(matmul_f32) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vulkan_ensure_matvec_f16bf16_pipeline(struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->matvec_f16bf16_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[3] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->matvec_f16bf16_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorSetLayout(matvec_f16bf16) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 6u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->matvec_f16bf16_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->matvec_f16bf16_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_f16bf16_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(matvec_f16bf16) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_matvec_f16bf16_spv_len,
        .pCode = (const uint32_t *) vulkan_matvec_f16bf16_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_f16bf16_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(matvec_f16bf16) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->matvec_f16bf16_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr,
                                      &st->matvec_f16bf16_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_f16bf16_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(matvec_f16bf16) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_matvec_q4k_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->matvec_q4k_pipeline != VK_NULL_HANDLE &&
        st->matvec_q4k_n4_pipeline != VK_NULL_HANDLE &&
        st->matvec_q4k_nt4_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[3] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->matvec_q4k_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorSetLayout(matvec_q4k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 6u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->matvec_q4k_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->matvec_q4k_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(matvec_q4k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_matvec_q4k_spv_len,
        .pCode = (const uint32_t *) vulkan_matvec_q4k_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(matvec_q4k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->matvec_q4k_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr,
                                      &st->matvec_q4k_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(matvec_q4k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->matvec_q4k_n4_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(matvec_q4k_n4) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci_n4 = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_matvec_q4k_n4_spv_len,
        .pCode = (const uint32_t *) vulkan_matvec_q4k_n4_spv,
    };
    shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci_n4, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(matvec_q4k_n4) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci_n4 = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->matvec_q4k_n4_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci_n4, nullptr,
                                      &st->matvec_q4k_n4_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(matvec_q4k_n4) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->matvec_q4k_nt4_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(matvec_q4k_nt4) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci_nt4 = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_matvec_q4k_nt4_spv_len,
        .pCode = (const uint32_t *) vulkan_matvec_q4k_nt4_spv,
    };
    shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci_nt4, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(matvec_q4k_nt4) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci_nt4 = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->matvec_q4k_nt4_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci_nt4, nullptr,
                                      &st->matvec_q4k_nt4_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(matvec_q4k_nt4) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vulkan_ensure_matvec_q4k_nt4_subgroup_pipeline(struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->matvec_q4k_nt4_subgroup_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    enum geist_status s = vulkan_ensure_matvec_q4k_pipeline(be);
    if (s != GEIST_OK) { return s; }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 6u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->matvec_q4k_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    VkResult rc = st->vkCreatePipelineLayout(
        st->device, &plci, nullptr,
        &st->matvec_q4k_nt4_subgroup_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_nt4_subgroup_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(matvec_q4k_nt4_subgroup) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_matvec_q4k_nt4_subgroup_spv_len,
        .pCode = (const uint32_t *) vulkan_matvec_q4k_nt4_subgroup_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_nt4_subgroup_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(matvec_q4k_nt4_subgroup) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->matvec_q4k_nt4_subgroup_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(
        st->device, st->pipeline_cache, 1, &cpci, nullptr,
        &st->matvec_q4k_nt4_subgroup_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_nt4_subgroup_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(matvec_q4k_nt4_subgroup) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_q4k_w4a8_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->q4k_quant_x_pipeline != VK_NULL_HANDLE &&
        st->matvec_q4k_w4a8_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding quant_bindings[3] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo qdlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = quant_bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &qdlci, nullptr, &st->q4k_quant_x_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorSetLayout(q4k_quant_x) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange quant_push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 2u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo qplci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->q4k_quant_x_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &quant_push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &qplci, nullptr,
                                    &st->q4k_quant_x_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_q4k_w4a8_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(q4k_quant_x) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo qsmci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_q4k_quant_x_spv_len,
        .pCode = (const uint32_t *) vulkan_q4k_quant_x_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &qsmci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_q4k_w4a8_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(q4k_quant_x) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo qcpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->q4k_quant_x_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &qcpci, nullptr,
                                      &st->q4k_quant_x_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_q4k_w4a8_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(q4k_quant_x) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkDescriptorSetLayoutBinding matvec_bindings[4] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo mdlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4,
        .pBindings = matvec_bindings,
    };
    rc = st->vkCreateDescriptorSetLayout(
        st->device, &mdlci, nullptr, &st->matvec_q4k_w4a8_set_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_q4k_w4a8_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorSetLayout(matvec_q4k_w4a8) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange matvec_push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 5u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo mplci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->matvec_q4k_w4a8_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &matvec_push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &mplci, nullptr,
                                    &st->matvec_q4k_w4a8_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_q4k_w4a8_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(matvec_q4k_w4a8) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo msmci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_matvec_q4k_nt4_w4a8_spv_len,
        .pCode = (const uint32_t *) vulkan_matvec_q4k_nt4_w4a8_spv,
    };
    shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &msmci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_q4k_w4a8_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(matvec_q4k_w4a8) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo mcpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->matvec_q4k_w4a8_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &mcpci, nullptr,
                                      &st->matvec_q4k_w4a8_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_q4k_w4a8_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(matvec_q4k_w4a8) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_matvec_q4k_nt4_pair_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->matvec_q4k_nt4_pair_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[5] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 4,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 5,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr,
        &st->matvec_q4k_nt4_pair_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorSetLayout(matvec_q4k_nt4_pair) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 8u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->matvec_q4k_nt4_pair_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(
        st->device, &plci, nullptr,
        &st->matvec_q4k_nt4_pair_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(matvec_q4k_nt4_pair) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_matvec_q4k_nt4_pair_spv_len,
        .pCode = (const uint32_t *) vulkan_matvec_q4k_nt4_pair_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(matvec_q4k_nt4_pair) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->matvec_q4k_nt4_pair_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr,
                                      &st->matvec_q4k_nt4_pair_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(matvec_q4k_nt4_pair) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vulkan_ensure_ffn_gate_up_gelu_q4k_pipeline(struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->ffn_gate_up_gelu_q4k_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[4] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->ffn_gate_up_gelu_q4k_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorSetLayout(ffn_gate_up_gelu_q4k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 7u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->ffn_gate_up_gelu_q4k_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(
        st->device, &plci, nullptr,
        &st->ffn_gate_up_gelu_q4k_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(ffn_gate_up_gelu_q4k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_ffn_gate_up_gelu_q4k_spv_len,
        .pCode = (const uint32_t *) vulkan_ffn_gate_up_gelu_q4k_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(ffn_gate_up_gelu_q4k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->ffn_gate_up_gelu_q4k_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(
        st->device, st->pipeline_cache, 1, &cpci, nullptr,
        &st->ffn_gate_up_gelu_q4k_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(ffn_gate_up_gelu_q4k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vulkan_ensure_ffn_gate_up_gelu_q4k_nt4_pipeline(struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->ffn_gate_up_gelu_q4k_nt4_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[4] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr,
        &st->ffn_gate_up_gelu_q4k_nt4_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorSetLayout(ffn_gate_up_gelu_q4k_nt4) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 7u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->ffn_gate_up_gelu_q4k_nt4_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(
        st->device, &plci, nullptr,
        &st->ffn_gate_up_gelu_q4k_nt4_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(ffn_gate_up_gelu_q4k_nt4) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_ffn_gate_up_gelu_q4k_nt4_spv_len,
        .pCode = (const uint32_t *) vulkan_ffn_gate_up_gelu_q4k_nt4_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(ffn_gate_up_gelu_q4k_nt4) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->ffn_gate_up_gelu_q4k_nt4_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(
        st->device, st->pipeline_cache, 1, &cpci, nullptr,
        &st->ffn_gate_up_gelu_q4k_nt4_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(ffn_gate_up_gelu_q4k_nt4) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_matvec_q6k_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->matvec_q6k_pipeline != VK_NULL_HANDLE &&
        (!vulkan_q6k_n4_enabled() ||
         st->matvec_q6k_n4_pipeline != VK_NULL_HANDLE)) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[3] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->matvec_q6k_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorSetLayout(matvec_q6k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 6u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->matvec_q6k_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->matvec_q6k_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q6k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(matvec_q6k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_matvec_q6k_spv_len,
        .pCode = (const uint32_t *) vulkan_matvec_q6k_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q6k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(matvec_q6k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->matvec_q6k_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr,
                                      &st->matvec_q6k_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q6k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(matvec_q6k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    if (!vulkan_q6k_n4_enabled()) {
        return GEIST_OK;
    }

    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->matvec_q6k_n4_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q6k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(matvec_q6k_n4) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci_n4 = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_matvec_q6k_n4_spv_len,
        .pCode = (const uint32_t *) vulkan_matvec_q6k_n4_spv,
    };
    shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci_n4, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q6k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(matvec_q6k_n4) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci_n4 = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->matvec_q6k_n4_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci_n4, nullptr,
                                      &st->matvec_q6k_n4_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matvec_q6k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(matvec_q6k_n4) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_matmul_q4k_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->matmul_q4k_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    enum geist_status s = vulkan_ensure_matvec_q4k_pipeline(be);
    if (s != GEIST_OK) { return s; }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 9u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->matvec_q4k_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    VkResult rc = st->vkCreatePipelineLayout(
        st->device, &plci, nullptr, &st->matmul_q4k_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matmul_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(matmul_q4k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_matmul_q4k_spv_len,
        .pCode = (const uint32_t *) vulkan_matmul_q4k_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matmul_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(matmul_q4k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->matmul_q4k_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr,
                                      &st->matmul_q4k_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matmul_q4k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(matmul_q4k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_matmul_q4k_m8_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->matmul_q4k_m8_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    enum geist_status s = vulkan_ensure_matvec_q4k_pipeline(be);
    if (s != GEIST_OK) { return s; }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 9u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->matvec_q4k_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    VkResult rc = st->vkCreatePipelineLayout(
        st->device, &plci, nullptr, &st->matmul_q4k_m8_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matmul_q4k_m8_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(matmul_q4k_m8) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_matmul_q4k_m8_spv_len,
        .pCode = (const uint32_t *) vulkan_matmul_q4k_m8_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matmul_q4k_m8_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(matmul_q4k_m8) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->matmul_q4k_m8_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr,
                                      &st->matmul_q4k_m8_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matmul_q4k_m8_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(matmul_q4k_m8) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_matmul_q6k_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->matmul_q6k_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    enum geist_status s = vulkan_ensure_matvec_q6k_pipeline(be);
    if (s != GEIST_OK) { return s; }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 9u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->matvec_q6k_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    VkResult rc = st->vkCreatePipelineLayout(
        st->device, &plci, nullptr, &st->matmul_q6k_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matmul_q6k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(matmul_q6k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_matmul_q6k_spv_len,
        .pCode = (const uint32_t *) vulkan_matmul_q6k_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matmul_q6k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(matmul_q6k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->matmul_q6k_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr,
                                      &st->matmul_q6k_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_matmul_q6k_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(matmul_q6k) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_argmax_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->argmax_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[2] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->argmax_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateDescriptorSetLayout(argmax) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 3u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->argmax_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->argmax_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_argmax_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreatePipelineLayout(argmax) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_argmax_f32_spv_len,
        .pCode = (const uint32_t *) vulkan_argmax_f32_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_argmax_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateShaderModule(argmax) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->argmax_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr, &st->argmax_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_argmax_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateComputePipelines(argmax) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_argmax_batch_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->argmax_batch_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    enum geist_status s = vulkan_ensure_argmax_pipeline(be);
    if (s != GEIST_OK) { return s; }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 5u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->argmax_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    VkResult rc = st->vkCreatePipelineLayout(
        st->device, &plci, nullptr, &st->argmax_batch_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_argmax_batch_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(argmax_batch) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_argmax_f32_batch_spv_len,
        .pCode = (const uint32_t *) vulkan_argmax_f32_batch_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_argmax_batch_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(argmax_batch) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->argmax_batch_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr,
                                      &st->argmax_batch_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_argmax_batch_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(argmax_batch) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

struct vulkan_embed_lookup_scaled_push {
    uint32_t n_in;
    uint32_t token_id;
    uint32_t dtype;
    uint32_t blocks_per_row;
    uint32_t w_byte_offset;
    uint32_t y_offset;
    float scale;
};

[[nodiscard]] static enum geist_status
vulkan_ensure_embed_lookup_scaled_pipeline(struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->embed_lookup_scaled_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[2] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->embed_lookup_scaled_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorSetLayout(embed_lookup_scaled) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(struct vulkan_embed_lookup_scaled_push),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->embed_lookup_scaled_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(
        st->device, &plci, nullptr,
        &st->embed_lookup_scaled_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_embed_lookup_scaled_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(embed_lookup_scaled) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_embed_lookup_scaled_spv_len,
        .pCode = (const uint32_t *) vulkan_embed_lookup_scaled_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_embed_lookup_scaled_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(embed_lookup_scaled) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->embed_lookup_scaled_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(
        st->device, st->pipeline_cache, 1, &cpci, nullptr,
        &st->embed_lookup_scaled_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_embed_lookup_scaled_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(embed_lookup_scaled) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vulkan_ensure_embed_lookup_scaled_dyn_pipeline(struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->embed_lookup_scaled_dyn_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[3] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr,
        &st->embed_lookup_scaled_dyn_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorSetLayout(embed_lookup_scaled_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(struct vulkan_embed_lookup_scaled_push),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->embed_lookup_scaled_dyn_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(
        st->device, &plci, nullptr,
        &st->embed_lookup_scaled_dyn_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_embed_lookup_scaled_dyn_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(embed_lookup_scaled_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_embed_lookup_scaled_dyn_spv_len,
        .pCode = (const uint32_t *) vulkan_embed_lookup_scaled_dyn_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_embed_lookup_scaled_dyn_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(embed_lookup_scaled_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->embed_lookup_scaled_dyn_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(
        st->device, st->pipeline_cache, 1, &cpci, nullptr,
        &st->embed_lookup_scaled_dyn_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_embed_lookup_scaled_dyn_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(embed_lookup_scaled_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_rope_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->rope_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[3] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->rope_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateDescriptorSetLayout(rope) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 6u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->rope_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->rope_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_rope_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreatePipelineLayout(rope) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_rope_f32_spv_len,
        .pCode = (const uint32_t *) vulkan_rope_f32_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_rope_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateShaderModule(rope) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->rope_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr, &st->rope_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_rope_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateComputePipelines(rope) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_rope_dyn_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->rope_dyn_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[4] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->rope_dyn_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorSetLayout(rope_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 7u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->rope_dyn_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->rope_dyn_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_rope_dyn_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(rope_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_rope_f32_dyn_spv_len,
        .pCode = (const uint32_t *) vulkan_rope_f32_dyn_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_rope_dyn_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(rope_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->rope_dyn_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr,
                                      &st->rope_dyn_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_rope_dyn_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(rope_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_attention_pipeline(
    struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->attention_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[4] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->attention_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateDescriptorSetLayout(attention) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 11u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->attention_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->attention_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_attention_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreatePipelineLayout(attention) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_attention_f32_spv_len,
        .pCode = (const uint32_t *) vulkan_attention_f32_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_attention_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateShaderModule(attention) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->attention_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(st->device, st->pipeline_cache, 1,
                                      &cpci, nullptr, &st->attention_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_attention_pipeline(st);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateComputePipelines(attention) failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vulkan_ensure_attention_dyn_pipeline(struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->attention_dyn_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[5] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 4,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 5,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->attention_dyn_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorSetLayout(attention_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 11u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->attention_dyn_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->attention_dyn_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_attention_dyn_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(attention_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_attention_f32_dyn_spv_len,
        .pCode = (const uint32_t *) vulkan_attention_f32_dyn_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_attention_dyn_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(attention_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->attention_dyn_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(
        st->device, st->pipeline_cache, 1, &cpci, nullptr,
        &st->attention_dyn_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_attention_dyn_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(attention_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status
vulkan_ensure_kv_append_dyn_pipeline(struct geist_backend *be) {

    struct vulkan_state *st = be->state;
    if (st->kv_append_dyn_pipeline != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorSetLayoutBinding bindings[5] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 4,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    const VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 5,
        .pBindings = bindings,
    };
    VkResult rc = st->vkCreateDescriptorSetLayout(
        st->device, &dlci, nullptr, &st->kv_append_dyn_set_layout);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateDescriptorSetLayout(kv_append_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkPushConstantRange push_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = 6u * sizeof(uint32_t),
    };
    const VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &st->kv_append_dyn_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    rc = st->vkCreatePipelineLayout(st->device, &plci, nullptr,
                                    &st->kv_append_dyn_pipeline_layout);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_kv_append_dyn_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreatePipelineLayout(kv_append_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = vulkan_kv_append_f32_dyn_spv_len,
        .pCode = (const uint32_t *) vulkan_kv_append_f32_dyn_spv,
    };
    VkShaderModule shader = VK_NULL_HANDLE;
    rc = st->vkCreateShaderModule(st->device, &smci, nullptr, &shader);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_kv_append_dyn_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateShaderModule(kv_append_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkComputePipelineCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = st->kv_append_dyn_pipeline_layout,
    };
    rc = st->vkCreateComputePipelines(
        st->device, st->pipeline_cache, 1, &cpci, nullptr,
        &st->kv_append_dyn_pipeline);
    st->vkDestroyShaderModule(st->device, shader, nullptr);
    if (rc != VK_SUCCESS) {
        vulkan_destroy_kv_append_dyn_pipeline(st);
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan: vkCreateComputePipelines(kv_append_dyn) failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    return GEIST_OK;
}

static bool vulkan_find_memory_type(const struct vulkan_state *st,
                                    uint32_t type_bits,
                                    VkMemoryPropertyFlags required,
                                    uint32_t *out_type) {
    for (uint32_t i = 0; i < st->memory_props.memoryTypeCount; i++) {
        const uint32_t bit = 1u << i;
        if ((type_bits & bit) == 0) {
            continue;
        }
        const VkMemoryPropertyFlags props =
            st->memory_props.memoryTypes[i].propertyFlags;
        if ((props & required) == required) {
            *out_type = i;
            return true;
        }
    }
    return false;
}

[[nodiscard]] static enum geist_status vulkan_buffer_create_internal(
    struct geist_backend *be,
    size_t bytes,
    enum geist_buffer_role role,
    unsigned int memory_flags,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags required_memory,
    struct geist_buffer **out) {

    if (out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out = nullptr;
    if (bytes == 0) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan: zero-byte buffer requested");
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    struct geist_buffer *buf =
        geist_backend_alloc(be, sizeof(*buf), alignof(struct geist_buffer));
    if (buf == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM,
                                "vulkan: failed to allocate buffer handle");
        return GEIST_E_OOM;
    }
    *buf = (struct geist_buffer){
        .owner = st,
        .bytes = (VkDeviceSize) bytes,
        .role = role,
        .memory_flags = memory_flags,
    };

    const VkBufferCreateInfo bci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = (VkDeviceSize) bytes,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkResult rc = st->vkCreateBuffer(st->device, &bci, nullptr, &buf->buffer);
    if (rc != VK_SUCCESS) {
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkCreateBuffer failed (%d)", (int) rc);
        return GEIST_E_BACKEND;
    }

    VkMemoryRequirements req;
    st->vkGetBufferMemoryRequirements(st->device, buf->buffer, &req);
    uint32_t memory_type = 0;
    if (!vulkan_find_memory_type(st, req.memoryTypeBits, required_memory, &memory_type)) {
        st->vkDestroyBuffer(st->device, buf->buffer, nullptr);
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: no memory type for required flags 0x%x",
                                (unsigned) required_memory);
        return GEIST_E_BACKEND;
    }

    const VkMemoryAllocateInfo mai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = memory_type,
    };
    rc = st->vkAllocateMemory(st->device, &mai, nullptr, &buf->memory);
    if (rc != VK_SUCCESS) {
        st->vkDestroyBuffer(st->device, buf->buffer, nullptr);
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkAllocateMemory failed (%d)", (int) rc);
        return GEIST_E_BACKEND;
    }

    rc = st->vkBindBufferMemory(st->device, buf->buffer, buf->memory, 0);
    if (rc != VK_SUCCESS) {
        st->vkFreeMemory(st->device, buf->memory, nullptr);
        st->vkDestroyBuffer(st->device, buf->buffer, nullptr);
        geist_backend_free(be, buf);
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan: vkBindBufferMemory failed (%d)", (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkMemoryPropertyFlags actual =
        st->memory_props.memoryTypes[memory_type].propertyFlags;
    buf->host_visible = (actual & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    buf->host_coherent = (actual & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    if (buf->host_visible) {
        rc = st->vkMapMemory(st->device, buf->memory, 0, buf->bytes, 0, &buf->mapped);
        if (rc != VK_SUCCESS) {
            st->vkFreeMemory(st->device, buf->memory, nullptr);
            st->vkDestroyBuffer(st->device, buf->buffer, nullptr);
            geist_backend_free(be, buf);
            geist_backend_set_error(be, GEIST_E_BACKEND,
                                    "vulkan: vkMapMemory failed (%d)", (int) rc);
            return GEIST_E_BACKEND;
        }
    }

    *out = buf;
    return GEIST_OK;
}

static void vulkan_buffer_destroy_internal(struct geist_backend *be,
                                           struct geist_buffer *buf) {
    if (buf == nullptr) {
        return;
    }
    vulkan_remove_q4k_nt4_cache_for_source(be, buf);
    struct vulkan_state *st = be->state;
    vulkan_decode_replay_invalidate(st);
    if (buf->mapped != nullptr) {
        st->vkUnmapMemory(st->device, buf->memory);
        buf->mapped = nullptr;
    }
    if (buf->memory != VK_NULL_HANDLE) {
        st->vkFreeMemory(st->device, buf->memory, nullptr);
    }
    if (buf->buffer != VK_NULL_HANDLE) {
        st->vkDestroyBuffer(st->device, buf->buffer, nullptr);
    }
    geist_backend_free(be, buf);
}

static void vulkan_destroy_decode_params_resources(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return;
    }
    struct vulkan_state *st = be->state;
    if (st->decode_params_buffer != nullptr) {
        vulkan_buffer_destroy_internal(be, st->decode_params_buffer);
        st->decode_params_buffer = nullptr;
    }
}

[[nodiscard]] static enum geist_status
vulkan_ensure_decode_params_buffer(struct geist_backend *be) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct vulkan_state *st = be->state;
    if (st->decode_params_buffer != nullptr) {
        return GEIST_OK;
    }
    return vulkan_buffer_create_internal(
        be, sizeof(struct vulkan_decode_params), GEIST_BUFFER_STAGING,
        GEIST_MEMORY_HOST_VISIBLE,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &st->decode_params_buffer);
}

[[nodiscard]] static enum geist_status
vulkan_flush_decode_params(struct geist_backend *be) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct vulkan_state *st = be->state;
    enum geist_status s = vulkan_ensure_decode_params_buffer(be);
    if (s != GEIST_OK) {
        return s;
    }
    if (st->decode_params_buffer == nullptr ||
        st->decode_params_buffer->mapped == nullptr) {
        return GEIST_E_BACKEND;
    }
    memcpy(st->decode_params_buffer->mapped, &st->decode_params,
           sizeof(st->decode_params));
    return vulkan_flush_if_needed(st, st->decode_params_buffer);
}

[[nodiscard]] static enum geist_status
vulkan_write_decode_params_token(struct geist_backend *be,
                                 uint32_t token_id) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct vulkan_state *st = be->state;
    st->decode_params.token_id = token_id;
    return vulkan_flush_decode_params(be);
}

[[nodiscard]] static enum geist_status
vulkan_write_decode_params_attention(struct geist_backend *be,
                                     uint32_t q_position,
                                     uint32_t kv_len) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct vulkan_state *st = be->state;
    st->decode_params.q_position = q_position;
    st->decode_params.kv_len = kv_len;
    return vulkan_flush_decode_params(be);
}

static void vulkan_destroy_argmax_runtime_resources(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return;
    }
    struct vulkan_state *st = be->state;
    if (st->argmax_result_buffer != nullptr) {
        vulkan_buffer_destroy_internal(be, st->argmax_result_buffer);
        st->argmax_result_buffer = nullptr;
        st->argmax_result_capacity = 0;
    }
    if (st->argmax_descriptor_pool != VK_NULL_HANDLE) {
        st->vkDestroyDescriptorPool(st->device, st->argmax_descriptor_pool,
                                    nullptr);
        st->argmax_descriptor_pool = VK_NULL_HANDLE;
        st->argmax_descriptor_set = VK_NULL_HANDLE;
    }
}

[[nodiscard]] static enum geist_status
vulkan_ensure_argmax_resources(struct geist_backend *be, size_t min_tokens) {

    if (be == nullptr || be->state == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (min_tokens == 0 || min_tokens > SIZE_MAX / sizeof(uint32_t)) {
        return GEIST_E_INVALID_ARG;
    }
    struct vulkan_state *st = be->state;

    enum geist_status s = vulkan_ensure_argmax_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    if (st->argmax_result_buffer != nullptr &&
        st->argmax_result_capacity < min_tokens) {
        if (st->capture_active) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "vulkan argmax: result buffer too small during capture");
            return GEIST_E_BACKEND;
        }
        vulkan_destroy_argmax_runtime_resources(be);
    }

    if (st->argmax_result_buffer == nullptr) {
        s = vulkan_buffer_create_internal(
            be, min_tokens * sizeof(uint32_t), GEIST_BUFFER_STAGING,
            GEIST_MEMORY_HOST_VISIBLE,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            &st->argmax_result_buffer);
        if (s != GEIST_OK) {
            return s;
        }
        st->argmax_result_capacity = min_tokens;
    }

    if (st->argmax_descriptor_set != VK_NULL_HANDLE) {
        return GEIST_OK;
    }

    const VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 2,
    };
    const VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    VkResult rc = st->vkCreateDescriptorPool(st->device, &dpci, nullptr,
                                             &st->argmax_descriptor_pool);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan argmax: vkCreateDescriptorPool failed (%d)", (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->argmax_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &st->argmax_set_layout,
    };
    rc = st->vkAllocateDescriptorSets(st->device, &dsai,
                                      &st->argmax_descriptor_set);
    if (rc != VK_SUCCESS) {
        st->vkDestroyDescriptorPool(st->device, st->argmax_descriptor_pool,
                                    nullptr);
        st->argmax_descriptor_pool = VK_NULL_HANDLE;
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan argmax: vkAllocateDescriptorSets failed (%d)", (int) rc);
        return GEIST_E_BACKEND;
    }

    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_copy_buffer(
    struct geist_backend *be,
    struct geist_buffer *src,
    struct geist_buffer *dst,
    VkDeviceSize src_offset,
    VkDeviceSize dst_offset,
    VkDeviceSize bytes) {

    struct vulkan_state *st = be->state;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    enum geist_status s =
        vulkan_begin_transient_command_buffer(be, "buffer_copy", &cmd);
    if (s != GEIST_OK) {
        return s;
    }

    const VkBufferCopy region = {
        .srcOffset = src_offset,
        .dstOffset = dst_offset,
        .size = bytes,
    };
    st->vkCmdCopyBuffer(cmd, src->buffer, dst->buffer, 1, &region);
    return vulkan_submit_transient_command_buffer(be, cmd, "buffer_copy");
}

static VkMemoryPropertyFlags vulkan_memory_flags_for_buffer(
    enum geist_buffer_role role,
    unsigned int memory_flags,
    bool *host_visible) {

    const bool wants_host =
        (memory_flags & (GEIST_MEMORY_HOST | GEIST_MEMORY_HOST_VISIBLE |
                         GEIST_MEMORY_MAPPED)) != 0 ||
        role == GEIST_BUFFER_STAGING || role == GEIST_BUFFER_IO;
    *host_visible = wants_host;
    if (wants_host) {
        return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    }
    return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
}

[[nodiscard]] static enum geist_status vulkan_buffer_create(
    struct geist_backend *be,
    size_t bytes,
    enum geist_buffer_role role,
    unsigned int memory_flags,
    struct geist_buffer **out) {

    bool host_visible = false;
    const VkMemoryPropertyFlags props =
        vulkan_memory_flags_for_buffer(role, memory_flags, &host_visible);
    const VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    return vulkan_buffer_create_internal(be, bytes, role, memory_flags,
                                         usage, props, out);
}

static void vulkan_buffer_destroy(struct geist_backend *be,
                                  struct geist_buffer *buf) {
    vulkan_buffer_destroy_internal(be, buf);
}

static enum geist_status vulkan_flush_if_needed(struct vulkan_state *st,
                                                const struct geist_buffer *buf) {
    if (buf->host_coherent || buf->mapped == nullptr) {
        return GEIST_OK;
    }
    const VkMappedMemoryRange range = {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = buf->memory,
        .offset = 0,
        .size = buf->bytes,
    };
    return st->vkFlushMappedMemoryRanges(st->device, 1, &range) == VK_SUCCESS
               ? GEIST_OK
               : GEIST_E_BACKEND;
}

static enum geist_status vulkan_invalidate_if_needed(
    struct vulkan_state *st,
    const struct geist_buffer *buf) {

    if (buf->host_coherent || buf->mapped == nullptr) {
        return GEIST_OK;
    }
    const VkMappedMemoryRange range = {
        .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
        .memory = buf->memory,
        .offset = 0,
        .size = buf->bytes,
    };
    return st->vkInvalidateMappedMemoryRanges(st->device, 1, &range) == VK_SUCCESS
               ? GEIST_OK
               : GEIST_E_BACKEND;
}

[[nodiscard]] static enum geist_status vulkan_buffer_copy(
    struct geist_buffer *dst,
    size_t dst_offset,
    const struct geist_buffer *src,
    size_t src_offset,
    size_t n_bytes) {

    if (dst == nullptr || src == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (dst->owner == nullptr || dst->owner != src->owner) {
        return GEIST_E_INVALID_ARG;
    }
    if (dst_offset > (size_t) dst->bytes || src_offset > (size_t) src->bytes ||
        n_bytes > (size_t) dst->bytes - dst_offset ||
        n_bytes > (size_t) src->bytes - src_offset) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes == 0) {
        return GEIST_OK;
    }

    struct vulkan_state *st = dst->owner;
    if (src->host_visible) {
        enum geist_status s = vulkan_flush_if_needed(st, src);
        if (s != GEIST_OK) {
            return s;
        }
    }
    enum geist_status s = vulkan_copy_buffer(st->backend,
                                             (struct geist_buffer *) src, dst,
                                             (VkDeviceSize) src_offset,
                                             (VkDeviceSize) dst_offset,
                                             (VkDeviceSize) n_bytes);
    if (s != GEIST_OK) {
        return s;
    }
    if (dst->host_visible) {
        return vulkan_invalidate_if_needed(st, dst);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_buffer_upload(
    struct geist_buffer *buf,
    size_t n_bytes,
    const uint8_t src[static n_bytes]) {

    if (buf == nullptr || src == nullptr || n_bytes > (size_t) buf->bytes) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes == 0) {
        return GEIST_OK;
    }
    struct vulkan_state *st = buf->owner;
    if (buf->host_visible) {
        memcpy(buf->mapped, src, n_bytes);
        return vulkan_flush_if_needed(st, buf);
    }

    struct geist_buffer *staging = nullptr;
    enum geist_status s = vulkan_buffer_create_internal(
        st->backend, n_bytes, GEIST_BUFFER_STAGING, GEIST_MEMORY_HOST_VISIBLE,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &staging);
    if (s != GEIST_OK) {
        return s;
    }
    memcpy(staging->mapped, src, n_bytes);
    s = vulkan_flush_if_needed(st, staging);
    if (s == GEIST_OK) {
        s = vulkan_copy_buffer(st->backend, staging, buf, 0, 0,
                               (VkDeviceSize) n_bytes);
    }
    vulkan_buffer_destroy_internal(st->backend, staging);
    return s;
}

[[nodiscard]] static enum geist_status vulkan_buffer_download(
    size_t n_bytes,
    uint8_t dst[static n_bytes],
    const struct geist_buffer *buf) {

    if (buf == nullptr || dst == nullptr || n_bytes > (size_t) buf->bytes) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes == 0) {
        return GEIST_OK;
    }
    struct vulkan_state *st = buf->owner;
    if (buf->host_visible) {
        enum geist_status s = vulkan_invalidate_if_needed(st, buf);
        if (s != GEIST_OK) {
            return s;
        }
        memcpy(dst, buf->mapped, n_bytes);
        return GEIST_OK;
    }

    struct geist_buffer *staging = nullptr;
    enum geist_status s = vulkan_buffer_create_internal(
        st->backend, n_bytes, GEIST_BUFFER_STAGING, GEIST_MEMORY_HOST_VISIBLE,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &staging);
    if (s != GEIST_OK) {
        return s;
    }
    s = vulkan_copy_buffer(st->backend, (struct geist_buffer *) buf, staging,
                           0, 0, (VkDeviceSize) n_bytes);
    if (s == GEIST_OK) {
        s = vulkan_invalidate_if_needed(st, staging);
    }
    if (s == GEIST_OK) {
        memcpy(dst, staging->mapped, n_bytes);
    }
    vulkan_buffer_destroy_internal(st->backend, staging);
    return s;
}

[[nodiscard]] static enum geist_status vulkan_buffer_download_range(
    const struct geist_buffer *buf,
    size_t offset,
    size_t n_bytes,
    uint8_t dst[static n_bytes]) {

    if (buf == nullptr || dst == nullptr || offset > (size_t) buf->bytes ||
        n_bytes > (size_t) buf->bytes - offset) {
        return GEIST_E_INVALID_ARG;
    }
    if (n_bytes == 0) {
        return GEIST_OK;
    }
    struct vulkan_state *st = buf->owner;
    if (buf->host_visible) {
        enum geist_status s = vulkan_invalidate_if_needed(st, buf);
        if (s != GEIST_OK) {
            return s;
        }
        memcpy(dst, (const uint8_t *) buf->mapped + offset, n_bytes);
        return GEIST_OK;
    }

    struct geist_buffer *staging = nullptr;
    enum geist_status s = vulkan_buffer_create_internal(
        st->backend, n_bytes, GEIST_BUFFER_STAGING, GEIST_MEMORY_HOST_VISIBLE,
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        &staging);
    if (s != GEIST_OK) {
        return s;
    }
    s = vulkan_copy_buffer(st->backend, (struct geist_buffer *) buf, staging,
                           (VkDeviceSize) offset, 0, (VkDeviceSize) n_bytes);
    if (s == GEIST_OK) {
        s = vulkan_invalidate_if_needed(st, staging);
    }
    if (s == GEIST_OK) {
        memcpy(dst, staging->mapped, n_bytes);
    }
    vulkan_buffer_destroy_internal(st->backend, staging);
    return s;
}

static bool vulkan_q4k_nt4_enabled_for(size_t n_out) {
    const char *env = getenv("GEIST_VULKAN_Q4K_NT4");
    if (env == nullptr || env[0] == '\0') {
        (void) n_out;
        return true;
    }
    if (env[0] == '0' && env[1] == '\0') {
        return false;
    }
    if (strcmp(env, "force") == 0) {
        return true;
    }
    (void) n_out;
    return env[0] == '1' && env[1] == '\0';
}

static bool vulkan_q4k_nt4_pair_enabled(void) {
    return vulkan_env_toggle_default_on("GEIST_VULKAN_Q4K_NT4_PAIR");
}

static bool vulkan_q4k_w4a8_enabled(void) {
    return vulkan_env_is_on("GEIST_VULKAN_Q4K_W4A8");
}

static bool vulkan_q4k_nt4_subgroup_enabled_for_state(
    const struct vulkan_state *st) {
    return st != nullptr &&
           st->has_subgroup_arithmetic &&
           vulkan_env_is_on("GEIST_VULKAN_Q4K_NT4_SUBGROUP");
}

[[nodiscard]] static enum geist_status
vulkan_ensure_q4k_nt4_subgroup_if_enabled(struct geist_backend *be) {
    struct vulkan_state *st = be->state;
    if (!vulkan_q4k_nt4_subgroup_enabled_for_state(st)) {
        return GEIST_OK;
    }
    return vulkan_ensure_matvec_q4k_nt4_subgroup_pipeline(be);
}

static bool vulkan_rmsnorm_rope_dyn_enabled(void) {
    return vulkan_env_toggle_default_on("GEIST_VULKAN_RMSNORM_ROPE_DYN");
}

static bool vulkan_q6k_n4_enabled(void) {
    return vulkan_env_toggle_default_on("GEIST_VULKAN_Q6K_N4");
}

static bool vulkan_ffn_gate_up_gelu_q4k_enabled(void) {
    return vulkan_env_is_on("GEIST_VULKAN_FFN_GATE_UP_GELU_Q4K");
}

static bool vulkan_ffn_gate_up_gelu_q4k_nt4_enabled(void) {
    return vulkan_env_is_on("GEIST_VULKAN_FFN_GATE_UP_GELU_Q4K_NT4");
}

static bool vulkan_block_rmsnorm_add_enabled(void) {
    return vulkan_env_toggle_default_on("GEIST_VULKAN_BLOCK_RMSNORM_ADD");
}

static struct geist_buffer *vulkan_q4k_nt4_cache_find(
    const struct vulkan_state *st,
    const struct geist_tensor *w,
    size_t n_in,
    size_t n_out) {

    if (st == nullptr || w == nullptr || w->buffer == nullptr ||
        w->dtype != GEIST_DTYPE_Q4_K ||
        w->layout != GEIST_LAYOUT_BLOCK_QUANTIZED) {
        return nullptr;
    }
    for (struct vulkan_q4k_nt4_cache_entry *entry = st->q4k_nt4_cache;
         entry != nullptr;
         entry = entry->next) {
        if (entry->src == w->buffer && entry->src_offset == w->offset &&
            entry->n_in == n_in && entry->n_out == n_out) {
            return entry->packed;
        }
    }
    return nullptr;
}

static struct geist_buffer *vulkan_q4k_nt4_cache_find_tensor(
    const struct vulkan_state *st,
    const struct geist_tensor *w) {

    if (w == nullptr || w->ndim != 2 ||
        w->shape[0] <= 0 || w->shape[1] <= 0) {
        return nullptr;
    }
    return vulkan_q4k_nt4_cache_find(st, w, (size_t) w->shape[1],
                                     (size_t) w->shape[0]);
}

static void vulkan_destroy_q4k_nt4_cache(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return;
    }
    struct vulkan_state *st = be->state;
    struct vulkan_q4k_nt4_cache_entry *entry = st->q4k_nt4_cache;
    st->q4k_nt4_cache = nullptr;
    while (entry != nullptr) {
        struct vulkan_q4k_nt4_cache_entry *next = entry->next;
        if (entry->packed != nullptr) {
            vulkan_buffer_destroy_internal(be, entry->packed);
        }
        geist_backend_free(be, entry);
        entry = next;
    }
}

static void vulkan_destroy_q4k_w4a8_resources(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return;
    }
    struct vulkan_state *st = be->state;
    if (st->q4k_w4a8_xq_buffer != nullptr) {
        vulkan_buffer_destroy_internal(be, st->q4k_w4a8_xq_buffer);
        st->q4k_w4a8_xq_buffer = nullptr;
    }
    if (st->q4k_w4a8_scale_buffer != nullptr) {
        vulkan_buffer_destroy_internal(be, st->q4k_w4a8_scale_buffer);
        st->q4k_w4a8_scale_buffer = nullptr;
    }
    st->q4k_w4a8_xq_capacity = 0;
}

static void vulkan_remove_q4k_nt4_cache_for_source(
    struct geist_backend *be,
    const struct geist_buffer *src) {

    if (be == nullptr || be->state == nullptr || src == nullptr) {
        return;
    }
    struct vulkan_state *st = be->state;
    struct vulkan_q4k_nt4_cache_entry **link = &st->q4k_nt4_cache;
    while (*link != nullptr) {
        struct vulkan_q4k_nt4_cache_entry *entry = *link;
        if (entry->src != src) {
            link = &entry->next;
            continue;
        }
        *link = entry->next;
        if (entry->packed != nullptr) {
            vulkan_buffer_destroy_internal(be, entry->packed);
        }
        geist_backend_free(be, entry);
    }
}

[[nodiscard]] static enum geist_status vulkan_q4k_nt4_cache_get(
    struct geist_backend *be,
    const struct geist_tensor *w,
    size_t n_in,
    size_t n_out,
    struct geist_buffer **packed_out) {

    if (packed_out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *packed_out = nullptr;
    if (be == nullptr || be->state == nullptr || w == nullptr ||
        w->buffer == nullptr || w->dtype != GEIST_DTYPE_Q4_K ||
        w->layout != GEIST_LAYOUT_BLOCK_QUANTIZED ||
        (n_in % Q4_K_BLOCK_ELEMS) != 0) {
        return GEIST_E_INVALID_ARG;
    }
    struct vulkan_state *st = be->state;
    const size_t blocks_per_row = n_in / Q4_K_BLOCK_ELEMS;
    if (n_out > SIZE_MAX / blocks_per_row ||
        n_out * blocks_per_row > SIZE_MAX / Q4_K_BLOCK_BYTES) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t raw_bytes = n_out * blocks_per_row * Q4_K_BLOCK_BYTES;
    if (w->offset > (size_t) w->buffer->bytes ||
        raw_bytes > (size_t) w->buffer->bytes - w->offset) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t packed_bytes = q4k_predecode_ntile4_size_bytes(n_in, n_out);
    if (packed_bytes == 0) {
        return GEIST_E_INVALID_ARG;
    }

    for (struct vulkan_q4k_nt4_cache_entry *entry = st->q4k_nt4_cache;
         entry != nullptr;
         entry = entry->next) {
        if (entry->src == w->buffer && entry->src_offset == w->offset &&
            entry->n_in == n_in && entry->n_out == n_out) {
            *packed_out = entry->packed;
            return GEIST_OK;
        }
    }

    uint8_t *raw = geist_backend_alloc(be, raw_bytes, alignof(max_align_t));
    uint8_t *packed_host =
        geist_backend_alloc(be, packed_bytes, alignof(max_align_t));
    struct geist_buffer *packed = nullptr;
    struct vulkan_q4k_nt4_cache_entry *entry = nullptr;
    if (raw == nullptr || packed_host == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM,
                                "vulkan q4k nt4 cache: host allocation failed");
        if (raw != nullptr) { geist_backend_free(be, raw); }
        if (packed_host != nullptr) { geist_backend_free(be, packed_host); }
        return GEIST_E_OOM;
    }

    enum geist_status s =
        vulkan_buffer_download_range(w->buffer, w->offset, raw_bytes, raw);
    if (s == GEIST_OK &&
        q4k_predecode_ntile4_pack(raw, n_in, n_out, packed_host) != 0) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan q4k nt4 cache: pack failed");
        s = GEIST_E_BACKEND;
    }
    if (s == GEIST_OK) {
        s = vulkan_buffer_create_internal(
            be, packed_bytes, GEIST_BUFFER_WEIGHT, GEIST_MEMORY_DEVICE,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &packed);
    }
    if (s == GEIST_OK) {
        s = vulkan_buffer_upload(packed, packed_bytes, packed_host);
    }
    if (s == GEIST_OK) {
        entry = geist_backend_alloc(be, sizeof(*entry),
                                    alignof(struct vulkan_q4k_nt4_cache_entry));
        if (entry == nullptr) {
            geist_backend_set_error(be, GEIST_E_OOM,
                                    "vulkan q4k nt4 cache: entry allocation failed");
            s = GEIST_E_OOM;
        }
    }
    if (s != GEIST_OK) {
        if (packed != nullptr) {
            vulkan_buffer_destroy_internal(be, packed);
        }
        geist_backend_free(be, raw);
        geist_backend_free(be, packed_host);
        return s;
    }

    *entry = (struct vulkan_q4k_nt4_cache_entry) {
        .src = w->buffer,
        .src_offset = w->offset,
        .n_in = n_in,
        .n_out = n_out,
        .raw_bytes = raw_bytes,
        .packed_bytes = packed_bytes,
        .packed = packed,
        .next = st->q4k_nt4_cache,
    };
    st->q4k_nt4_cache = entry;
    *packed_out = packed;

    geist_backend_free(be, raw);
    geist_backend_free(be, packed_host);
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ensure_q4k_w4a8_scratch(
    struct geist_backend *be,
    size_t n_in) {

    if (be == nullptr || be->state == nullptr || n_in == 0 ||
        n_in > (SIZE_MAX - 3u)) {
        return GEIST_E_INVALID_ARG;
    }
    struct vulkan_state *st = be->state;
    const size_t xq_bytes = ((n_in + 3u) / 4u) * sizeof(uint32_t);
    if (st->q4k_w4a8_xq_buffer != nullptr &&
        st->q4k_w4a8_scale_buffer != nullptr &&
        st->q4k_w4a8_xq_capacity >= xq_bytes) {
        return GEIST_OK;
    }

    if (st->q4k_w4a8_xq_buffer != nullptr &&
        st->q4k_w4a8_xq_capacity < xq_bytes) {
        vulkan_buffer_destroy_internal(be, st->q4k_w4a8_xq_buffer);
        st->q4k_w4a8_xq_buffer = nullptr;
        st->q4k_w4a8_xq_capacity = 0;
    }
    enum geist_status s = GEIST_OK;
    if (st->q4k_w4a8_xq_buffer == nullptr) {
        s = vulkan_buffer_create_internal(
            be, xq_bytes, GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &st->q4k_w4a8_xq_buffer);
        if (s != GEIST_OK) {
            return s;
        }
        st->q4k_w4a8_xq_capacity = xq_bytes;
    }
    if (st->q4k_w4a8_scale_buffer == nullptr) {
        s = vulkan_buffer_create_internal(
            be, sizeof(float), GEIST_BUFFER_ACTIVATION, GEIST_MEMORY_DEVICE,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            &st->q4k_w4a8_scale_buffer);
        if (s != GEIST_OK) {
            return s;
        }
    }
    return GEIST_OK;
}

static void *vulkan_buffer_map(struct geist_buffer *buf) {
    if (buf == nullptr || !buf->host_visible) {
        return nullptr;
    }
    (void) vulkan_invalidate_if_needed(buf->owner, buf);
    return buf->mapped;
}

static void vulkan_buffer_unmap(struct geist_buffer *buf) {
    if (buf == nullptr || !buf->host_visible) {
        return;
    }
    (void) vulkan_flush_if_needed(buf->owner, buf);
}

static bool vulkan_tensor_f32_dense_count(const struct geist_tensor *t,
                                          size_t *out_n) {
    if (t == nullptr || out_n == nullptr ||
        t->dtype != GEIST_DTYPE_F32 ||
        t->layout != GEIST_LAYOUT_DENSE ||
        t->buffer == nullptr || t->ndim < 1) {
        return false;
    }
    size_t n = 1;
    for (int d = 0; d < t->ndim; d++) {
        if (t->shape[d] <= 0) {
            return false;
        }
        const size_t dim = (size_t) t->shape[d];
        if (n > SIZE_MAX / dim) {
            return false;
        }
        n *= dim;
    }
    *out_n = n;
    return true;
}

static bool vulkan_ranges_overlap(size_t a_offset, size_t a_bytes,
                                  size_t b_offset, size_t b_bytes) {
    if (a_offset < b_offset) {
        return b_offset - a_offset < a_bytes;
    }
    return a_offset - b_offset < b_bytes;
}

[[nodiscard]] static enum geist_status vulkan_embedding_lookup_scaled(
    struct geist_backend *be,
    const struct geist_tensor *embed_table,
    geist_token_t token_id,
    float scale,
    struct geist_tensor *out);

[[nodiscard]] static enum geist_status vulkan_embedding_lookup(
    struct geist_backend *be,
    const struct geist_tensor *embed_table,
    geist_token_t token_id,
    struct geist_tensor *out) {

    if (be == nullptr || be->state == nullptr ||
        embed_table == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (embed_table->dtype != GEIST_DTYPE_F32 ||
        embed_table->layout != GEIST_LAYOUT_DENSE) {
        return vulkan_embedding_lookup_scaled(be, embed_table, token_id,
                                              1.0f, out);
    }

    size_t out_n = 0;
    if (!vulkan_tensor_f32_dense_count(out, &out_n) ||
        embed_table->buffer == nullptr ||
        embed_table->ndim != 2 ||
        (out->ndim != 1 && out->ndim != 2)) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "vulkan embedding_lookup: tensors must be F32 DENSE table [vocab,dim] and F32 DENSE output [dim]");
        return GEIST_E_UNSUPPORTED;
    }

    const int64_t vocab_i64 = embed_table->shape[0];
    const int64_t d_model_i64 = embed_table->shape[1];
    if (vocab_i64 <= 0 || d_model_i64 <= 0 || token_id < 0 ||
        (int64_t) token_id >= vocab_i64) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t d_model = (size_t) d_model_i64;
    if (out_n != d_model ||
        (out->ndim == 2 && (out->shape[0] != 1 ||
                            out->shape[1] != d_model_i64))) {
        return GEIST_E_INVALID_ARG;
    }
    if (d_model > SIZE_MAX / sizeof(float)) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t row_bytes = d_model * sizeof(float);
    if ((size_t) token_id > SIZE_MAX / row_bytes) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *tb = embed_table->buffer;
    struct geist_buffer *ob = out->buffer;
    if (tb->owner == nullptr || tb->owner != ob->owner ||
        tb->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t row_offset = (size_t) token_id * row_bytes;
    if (embed_table->offset > (size_t) tb->bytes ||
        out->offset > (size_t) ob->bytes ||
        row_offset > (size_t) tb->bytes - embed_table->offset ||
        row_bytes > (size_t) tb->bytes - embed_table->offset - row_offset ||
        row_bytes > (size_t) ob->bytes - out->offset) {
        return GEIST_E_INVALID_ARG;
    }
    if (tb == ob && vulkan_ranges_overlap(out->offset, row_bytes,
                                          embed_table->offset + row_offset,
                                          row_bytes)) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan embedding_lookup: output overlaps source row");
        return GEIST_E_INVALID_ARG;
    }

    return vulkan_buffer_copy(ob, out->offset, tb,
                              embed_table->offset + row_offset,
                              row_bytes);
}

[[nodiscard]] static enum geist_status vulkan_embedding_lookup_scaled(
    struct geist_backend *be,
    const struct geist_tensor *embed_table,
    geist_token_t token_id,
    float scale,
    struct geist_tensor *out) {

    if (be == nullptr || be->state == nullptr ||
        embed_table == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t out_n = 0;
    if (!vulkan_tensor_f32_dense_count(out, &out_n) ||
        embed_table->buffer == nullptr ||
        embed_table->ndim != 2 ||
        (out->ndim != 1 && out->ndim != 2)) {
        return GEIST_E_INVALID_ARG;
    }

    const int64_t vocab_i64 = embed_table->shape[0];
    const int64_t d_model_i64 = embed_table->shape[1];
    if (vocab_i64 <= 0 || d_model_i64 <= 0 || token_id < 0 ||
        (int64_t) token_id >= vocab_i64) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t d_model = (size_t) d_model_i64;
    const size_t vocab = (size_t) vocab_i64;
    if (out_n != d_model ||
        d_model > UINT32_MAX || vocab > UINT32_MAX ||
        (size_t) token_id > UINT32_MAX ||
        (out->ndim == 2 && (out->shape[0] != 1 ||
                            out->shape[1] != d_model_i64)) ||
        (out->offset % sizeof(float)) != 0) {
        return GEIST_E_INVALID_ARG;
    }

    size_t row_bytes = 0;
    size_t blocks_per_row = 0;
    if (embed_table->dtype == GEIST_DTYPE_F32 &&
        embed_table->layout == GEIST_LAYOUT_DENSE) {
        if (d_model > SIZE_MAX / sizeof(float)) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = d_model * sizeof(float);
    } else if ((embed_table->dtype == GEIST_DTYPE_F16 ||
                embed_table->dtype == GEIST_DTYPE_BF16) &&
               embed_table->layout == GEIST_LAYOUT_DENSE) {
        if (d_model > SIZE_MAX / sizeof(uint16_t)) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = d_model * sizeof(uint16_t);
    } else if (embed_table->dtype == GEIST_DTYPE_Q4_K &&
               embed_table->layout == GEIST_LAYOUT_BLOCK_QUANTIZED) {
        if ((d_model % 256u) != 0) {
            return GEIST_E_INVALID_ARG;
        }
        blocks_per_row = d_model / 256u;
        if (blocks_per_row > SIZE_MAX / 144u) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = blocks_per_row * 144u;
    } else if (embed_table->dtype == GEIST_DTYPE_Q5_K &&
               embed_table->layout == GEIST_LAYOUT_BLOCK_QUANTIZED) {
        if ((d_model % 256u) != 0) {
            return GEIST_E_INVALID_ARG;
        }
        blocks_per_row = d_model / 256u;
        if (blocks_per_row > SIZE_MAX / 176u) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = blocks_per_row * 176u;
    } else if (embed_table->dtype == GEIST_DTYPE_Q6_K &&
               embed_table->layout == GEIST_LAYOUT_BLOCK_QUANTIZED) {
        if ((d_model % 256u) != 0) {
            return GEIST_E_INVALID_ARG;
        }
        blocks_per_row = d_model / 256u;
        if (blocks_per_row > SIZE_MAX / 210u) {
            return GEIST_E_INVALID_ARG;
        }
        row_bytes = blocks_per_row * 210u;
    } else {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "vulkan embedding_lookup_scaled: unsupported table dtype/layout");
        return GEIST_E_UNSUPPORTED;
    }
    if (row_bytes == 0 ||
        (size_t) token_id > SIZE_MAX / row_bytes ||
        row_bytes > UINT32_MAX ||
        blocks_per_row > UINT32_MAX ||
        embed_table->offset > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *tb = embed_table->buffer;
    struct geist_buffer *ob = out->buffer;
    if (tb->owner == nullptr || tb->owner != ob->owner ||
        tb->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t row_offset = (size_t) token_id * row_bytes;
    const size_t out_bytes = d_model * sizeof(float);
    if (embed_table->offset > (size_t) tb->bytes ||
        out->offset > (size_t) ob->bytes ||
        row_offset > (size_t) tb->bytes - embed_table->offset ||
        row_bytes > (size_t) tb->bytes - embed_table->offset - row_offset ||
        out_bytes > (size_t) ob->bytes - out->offset) {
        return GEIST_E_INVALID_ARG;
    }
    if (tb == ob && vulkan_ranges_overlap(out->offset, out_bytes,
                                          embed_table->offset + row_offset,
                                          row_bytes)) {
        geist_backend_set_error(
            be, GEIST_E_INVALID_ARG,
            "vulkan embedding_lookup_scaled: output overlaps source row");
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    const bool use_dynamic_decode_params =
        st->capture_active &&
        st->capture_kind == GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP &&
        vulkan_dynamic_decode_params_enabled(st);

    enum geist_status s = use_dynamic_decode_params
                              ? vulkan_ensure_embed_lookup_scaled_dyn_pipeline(be)
                              : vulkan_ensure_embed_lookup_scaled_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    if (use_dynamic_decode_params) {
        s = vulkan_write_decode_params_token(be, (uint32_t) token_id);
        if (s != GEIST_OK) {
            return s;
        }
    }
    if (tb->host_visible) {
        s = vulkan_flush_if_needed(st, tb);
        if (s != GEIST_OK) { return s; }
    }

    const uint32_t descriptor_count = use_dynamic_decode_params ? 3u : 2u;
    s = vulkan_ensure_transient_descriptor_pool(be, 1u, descriptor_count);
    if (s != GEIST_OK) {
        return s;
    }

    VkDescriptorSetLayout set_layout = use_dynamic_decode_params
                                           ? st->embed_lookup_scaled_dyn_set_layout
                                           : st->embed_lookup_scaled_set_layout;
    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->transient_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &set_layout,
    };
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai,
                                               &descriptor_set);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan embedding_lookup_scaled: vkAllocateDescriptorSets failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkDescriptorBufferInfo infos[3] = {
        {.buffer = tb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = ob->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {
            .buffer = use_dynamic_decode_params
                          ? st->decode_params_buffer->buffer
                          : VK_NULL_HANDLE,
            .offset = 0,
            .range = sizeof(struct vulkan_decode_params),
        },
    };
    VkWriteDescriptorSet writes[3] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
    };
    st->vkUpdateDescriptorSets(st->device, descriptor_count, writes, 0,
                               nullptr);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(
        be, "embedding_lookup_scaled", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);
        return s;
    }
    rc = VK_SUCCESS;
    if (rc == VK_SUCCESS) {
        const VkMemoryBarrier pre_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                             VK_ACCESS_HOST_WRITE_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT,
        };
        st->vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_HOST_BIT |
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

        const struct vulkan_embed_lookup_scaled_push push = {
            .n_in = (uint32_t) d_model,
            .token_id = (uint32_t) token_id,
            .dtype = (uint32_t) embed_table->dtype,
            .blocks_per_row = (uint32_t) blocks_per_row,
            .w_byte_offset = (uint32_t) embed_table->offset,
            .y_offset = (uint32_t) (out->offset / sizeof(float)),
            .scale = scale,
        };
        VkPipeline pipeline = use_dynamic_decode_params
                                  ? st->embed_lookup_scaled_dyn_pipeline
                                  : st->embed_lookup_scaled_pipeline;
        VkPipelineLayout pipeline_layout =
            use_dynamic_decode_params
                ? st->embed_lookup_scaled_dyn_pipeline_layout
                : st->embed_lookup_scaled_pipeline_layout;
        st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        st->vkCmdBindDescriptorSets(
            cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 1,
            &descriptor_set, 0, nullptr);
        st->vkCmdPushConstants(
            cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
            sizeof(push), &push);
        st->vkCmdDispatch(cmd, (uint32_t) ((d_model + 255u) / 256u), 1, 1);

        const VkMemoryBarrier post_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                             VK_ACCESS_SHADER_READ_BIT |
                             VK_ACCESS_HOST_READ_BIT,
        };
        st->vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 1, &post_barrier, 0, nullptr, 0, nullptr);
    }

    s = vulkan_submit_transient_command_buffer(
        be, cmd, "embedding_lookup_scaled");
    vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);

    if (s != GEIST_OK) {
        return s;
    }
    if (ob->host_visible) {
        return vulkan_invalidate_if_needed(st, ob);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_matvec_f32_dense(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr ||
        x == nullptr || w == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t nx = 0, nw = 0, ny = 0;
    if (!vulkan_tensor_f32_dense_count(x, &nx) ||
        !vulkan_tensor_f32_dense_count(w, &nw) ||
        !vulkan_tensor_f32_dense_count(y, &ny) ||
        x->ndim != 1 || w->ndim != 2 || y->ndim != 1) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "vulkan matvec: tensors must be F32 DENSE [in], [out,in], [out]");
        return GEIST_E_UNSUPPORTED;
    }
    const int64_t n_out_i64 = w->shape[0];
    const int64_t n_in_i64 = w->shape[1];
    if (n_in_i64 <= 0 || n_out_i64 <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t n_in = (size_t) n_in_i64;
    const size_t n_out = (size_t) n_out_i64;
    if (n_in > SIZE_MAX / n_out) {
        return GEIST_E_INVALID_ARG;
    }
    if (nx != n_in || ny != n_out || nw != n_in * n_out) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan matvec: shape mismatch (x=%zu w=%zux%zu y=%zu)",
                                nx, n_out, n_in, ny);
        return GEIST_E_INVALID_ARG;
    }
    if (n_in > UINT32_MAX || n_out > UINT32_MAX ||
        (x->offset % sizeof(float)) != 0 ||
        (w->offset % sizeof(float)) != 0 ||
        (y->offset % sizeof(float)) != 0) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *xb = x->buffer;
    struct geist_buffer *wb = w->buffer;
    struct geist_buffer *yb = y->buffer;
    if (xb->owner == nullptr || xb->owner != wb->owner ||
        xb->owner != yb->owner || xb->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t x_bytes = nx * sizeof(float);
    const size_t w_bytes = nw * sizeof(float);
    const size_t y_bytes = ny * sizeof(float);
    if (x->offset > (size_t) xb->bytes ||
        w->offset > (size_t) wb->bytes ||
        y->offset > (size_t) yb->bytes ||
        x_bytes > (size_t) xb->bytes - x->offset ||
        w_bytes > (size_t) wb->bytes - w->offset ||
        y_bytes > (size_t) yb->bytes - y->offset) {
        return GEIST_E_INVALID_ARG;
    }
    if ((yb == xb && vulkan_ranges_overlap(y->offset, y_bytes, x->offset, x_bytes)) ||
        (yb == wb && vulkan_ranges_overlap(y->offset, y_bytes, w->offset, w_bytes))) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan matvec: y must not overlap inputs");
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    enum geist_status s = vulkan_ensure_matvec_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    if (xb->host_visible) {
        s = vulkan_flush_if_needed(st, xb);
        if (s != GEIST_OK) { return s; }
    }
    if (wb->host_visible && wb != xb) {
        s = vulkan_flush_if_needed(st, wb);
        if (s != GEIST_OK) { return s; }
    }

    s = vulkan_ensure_transient_descriptor_pool(be, 1u, 3u);
    if (s != GEIST_OK) {
        return s;
    }

    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->transient_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &st->matvec_set_layout,
    };
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai,
                                               &descriptor_set);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
            "vulkan matvec: vkAllocateDescriptorSets failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkDescriptorBufferInfo infos[3] = {
        {.buffer = xb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = wb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = yb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[3] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 3, writes, 0, nullptr);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "matvec", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);
        return s;
    }
    rc = VK_SUCCESS;
    if (rc == VK_SUCCESS) {
        const VkMemoryBarrier pre_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                             VK_ACCESS_HOST_WRITE_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT,
        };
        st->vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_HOST_BIT |
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

        const uint32_t push[5] = {
            (uint32_t) n_in,
            (uint32_t) n_out,
            (uint32_t) (x->offset / sizeof(float)),
            (uint32_t) (w->offset / sizeof(float)),
            (uint32_t) (y->offset / sizeof(float)),
        };
        st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              st->matvec_pipeline);
        st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    st->matvec_pipeline_layout, 0, 1,
                                    &descriptor_set, 0, nullptr);
        st->vkCmdPushConstants(cmd, st->matvec_pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(push), push);
        st->vkCmdDispatch(cmd, (uint32_t) n_out, 1, 1);

        const VkMemoryBarrier post_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                             VK_ACCESS_SHADER_READ_BIT |
                             VK_ACCESS_HOST_READ_BIT,
        };
        st->vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 1, &post_barrier, 0, nullptr, 0, nullptr);
    }

    s = vulkan_submit_transient_command_buffer(be, cmd, "matvec");
    vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);

    if (s != GEIST_OK) {
        return s;
    }
    if (yb->host_visible) {
        return vulkan_invalidate_if_needed(st, yb);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_matmul_f32_dense(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr ||
        x == nullptr || w == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t nx = 0, nw = 0, ny = 0;
    if (!vulkan_tensor_f32_dense_count(x, &nx) ||
        !vulkan_tensor_f32_dense_count(w, &nw) ||
        !vulkan_tensor_f32_dense_count(y, &ny) ||
        x->ndim != 2 || w->ndim != 2 || y->ndim != 2) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "vulkan matmul: tensors must be F32 DENSE [rows,in], [out,in], [rows,out]");
        return GEIST_E_UNSUPPORTED;
    }

    const int64_t rows_i64 = x->shape[0];
    const int64_t n_in_i64 = x->shape[1];
    const int64_t n_out_i64 = w->shape[0];
    if (rows_i64 <= 0 || n_in_i64 <= 0 || n_out_i64 <= 0 ||
        w->shape[1] != n_in_i64 ||
        y->shape[0] != rows_i64 ||
        y->shape[1] != n_out_i64) {
        geist_backend_set_error(
            be, GEIST_E_INVALID_ARG,
            "vulkan matmul: shape contract failed "
            "x[%lld,%lld] w[%lld,%lld] y[%lld,%lld]",
            (long long) x->shape[0], (long long) x->shape[1],
            (long long) w->shape[0], (long long) w->shape[1],
            (long long) y->shape[0], (long long) y->shape[1]);
        return GEIST_E_INVALID_ARG;
    }
    const size_t rows = (size_t) rows_i64;
    const size_t n_in = (size_t) n_in_i64;
    const size_t n_out = (size_t) n_out_i64;
    if (rows > UINT32_MAX || n_in > UINT32_MAX || n_out > UINT32_MAX ||
        rows > SIZE_MAX / n_in ||
        rows > SIZE_MAX / n_out ||
        n_out > SIZE_MAX / n_in ||
        (x->offset % sizeof(float)) != 0 ||
        (w->offset % sizeof(float)) != 0 ||
        (y->offset % sizeof(float)) != 0) {
        geist_backend_set_error(
            be, GEIST_E_INVALID_ARG,
            "vulkan matmul: size/offset contract failed rows=%zu n_in=%zu n_out=%zu",
            rows, n_in, n_out);
        return GEIST_E_INVALID_ARG;
    }
    if (nx != rows * n_in || nw != n_out * n_in || ny != rows * n_out) {
        geist_backend_set_error(
            be, GEIST_E_INVALID_ARG,
            "vulkan matmul: shape mismatch (x=%zux%zu w=%zux%zu y=%zux%zu)",
            rows, n_in, n_out, n_in, rows, n_out);
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *xb = x->buffer;
    struct geist_buffer *wb = w->buffer;
    struct geist_buffer *yb = y->buffer;
    if (xb->owner == nullptr || xb->owner != wb->owner ||
        xb->owner != yb->owner || xb->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t x_bytes = nx * sizeof(float);
    const size_t w_bytes = nw * sizeof(float);
    const size_t y_bytes = ny * sizeof(float);
    if (x->offset > (size_t) xb->bytes ||
        w->offset > (size_t) wb->bytes ||
        y->offset > (size_t) yb->bytes ||
        x_bytes > (size_t) xb->bytes - x->offset ||
        w_bytes > (size_t) wb->bytes - w->offset ||
        y_bytes > (size_t) yb->bytes - y->offset) {
        geist_backend_set_error(
            be, GEIST_E_INVALID_ARG,
            "vulkan matmul: range contract failed x_bytes=%zu w_bytes=%zu y_bytes=%zu",
            x_bytes, w_bytes, y_bytes);
        return GEIST_E_INVALID_ARG;
    }
    if ((yb == xb && vulkan_ranges_overlap(y->offset, y_bytes,
                                           x->offset, x_bytes)) ||
        (yb == wb && vulkan_ranges_overlap(y->offset, y_bytes,
                                           w->offset, w_bytes))) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan matmul: y must not overlap inputs");
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    enum geist_status s = vulkan_ensure_matmul_f32_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    if (xb->host_visible) {
        s = vulkan_flush_if_needed(st, xb);
        if (s != GEIST_OK) { return s; }
    }
    if (wb->host_visible && wb != xb) {
        s = vulkan_flush_if_needed(st, wb);
        if (s != GEIST_OK) { return s; }
    }

    s = vulkan_ensure_transient_descriptor_pool(be, 1u, 3u);
    if (s != GEIST_OK) {
        return s;
    }

    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->transient_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &st->matvec_set_layout,
    };
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai,
                                               &descriptor_set);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan matmul: vkAllocateDescriptorSets failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkDescriptorBufferInfo infos[3] = {
        {.buffer = xb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = wb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = yb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[3] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 3, writes, 0, nullptr);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "matmul", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);
        return s;
    }

    const VkMemoryBarrier pre_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                         VK_ACCESS_HOST_WRITE_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

    const uint32_t push[8] = {
        (uint32_t) n_in,
        (uint32_t) n_out,
        (uint32_t) rows,
        (uint32_t) (x->offset / sizeof(float)),
        (uint32_t) (w->offset / sizeof(float)),
        (uint32_t) (y->offset / sizeof(float)),
        (uint32_t) n_in,
        (uint32_t) n_out,
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->matmul_f32_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->matmul_f32_pipeline_layout, 0, 1,
                                &descriptor_set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->matmul_f32_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), push);
    st->vkCmdDispatch(cmd, (uint32_t) n_out, (uint32_t) rows, 1);

    const VkMemoryBarrier post_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                         VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_HOST_READ_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &post_barrier, 0, nullptr, 0, nullptr);

    s = vulkan_submit_transient_command_buffer(be, cmd, "matmul");
    vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);
    if (s != GEIST_OK) {
        return s;
    }
    if (yb->host_visible) {
        return vulkan_invalidate_if_needed(st, yb);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_matvec_q4k_w4a8(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_buffer *wb_shader,
    struct geist_tensor *y,
    size_t n_in,
    size_t n_out,
    size_t blocks_per_row) {

    struct vulkan_state *st = be->state;
    enum geist_status s = vulkan_ensure_q4k_w4a8_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }
    s = vulkan_ensure_q4k_w4a8_scratch(be, n_in);
    if (s != GEIST_OK) {
        return s;
    }
    s = vulkan_ensure_transient_descriptor_pool(be, 2u, 7u);
    if (s != GEIST_OK) {
        return s;
    }

    VkDescriptorSetLayout layouts[2] = {
        st->q4k_quant_x_set_layout,
        st->matvec_q4k_w4a8_set_layout,
    };
    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->transient_descriptor_pool,
        .descriptorSetCount = 2,
        .pSetLayouts = layouts,
    };
    VkDescriptorSet sets[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai, sets);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan matvec_q4k_w4a8: vkAllocateDescriptorSets failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const struct geist_buffer *xb = x->buffer;
    const struct geist_buffer *yb = y->buffer;
    const VkDescriptorBufferInfo infos[7] = {
        {.buffer = xb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = st->q4k_w4a8_xq_buffer->buffer,
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = st->q4k_w4a8_scale_buffer->buffer,
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = st->q4k_w4a8_xq_buffer->buffer,
         .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = wb_shader->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = yb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = st->q4k_w4a8_scale_buffer->buffer,
         .offset = 0,
         .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[7] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sets[0],
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sets[0],
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sets[0],
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sets[1],
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[3],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sets[1],
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[4],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sets[1],
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[5],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = sets[1],
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[6],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 7, writes, 0, nullptr);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "matvec_q4k_w4a8", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_transient_descriptor_sets(st, 2u, sets);
        return s;
    }

    const VkMemoryBarrier pre_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                         VK_ACCESS_HOST_WRITE_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

    const uint32_t quant_push[2] = {
        (uint32_t) n_in,
        (uint32_t) (x->offset / sizeof(float)),
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->q4k_quant_x_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->q4k_quant_x_pipeline_layout, 0, 1,
                                &sets[0], 0, nullptr);
    st->vkCmdPushConstants(cmd, st->q4k_quant_x_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(quant_push), quant_push);
    st->vkCmdDispatch(cmd, 1, 1, 1);

    const VkMemoryBarrier quant_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &quant_barrier, 0, nullptr, 0, nullptr);

    const uint32_t matvec_push[5] = {
        (uint32_t) n_in,
        (uint32_t) n_out,
        (uint32_t) blocks_per_row,
        0u,
        (uint32_t) (y->offset / sizeof(float)),
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->matvec_q4k_w4a8_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->matvec_q4k_w4a8_pipeline_layout, 0, 1,
                                &sets[1], 0, nullptr);
    st->vkCmdPushConstants(cmd, st->matvec_q4k_w4a8_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(matvec_push), matvec_push);
    st->vkCmdDispatch(cmd, (uint32_t) ((n_out + 3u) / 4u), 1, 1);

    const VkMemoryBarrier post_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                         VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_HOST_READ_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &post_barrier, 0, nullptr, 0, nullptr);

    s = vulkan_submit_transient_command_buffer(be, cmd, "matvec_q4k_w4a8");
    vulkan_free_transient_descriptor_sets(st, 2u, sets);
    if (s != GEIST_OK) {
        return s;
    }
    if (yb->host_visible) {
        return vulkan_invalidate_if_needed(st, yb);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_matvec_q4k(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr ||
        x == nullptr || w == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t nx = 0, ny = 0;
    if (!vulkan_tensor_f32_dense_count(x, &nx) ||
        !vulkan_tensor_f32_dense_count(y, &ny) ||
        x->ndim != 1 || y->ndim != 1 ||
        w->buffer == nullptr ||
        w->dtype != GEIST_DTYPE_Q4_K ||
        w->layout != GEIST_LAYOUT_BLOCK_QUANTIZED ||
        w->ndim != 2) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "vulkan matvec_q4k: tensors must be F32 DENSE [in], Q4_K BLOCK_QUANTIZED [out,in], F32 DENSE [out]");
        return GEIST_E_UNSUPPORTED;
    }

    const int64_t n_out_i64 = w->shape[0];
    const int64_t n_in_i64 = w->shape[1];
    if (n_in_i64 <= 0 || n_out_i64 <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t n_in = (size_t) n_in_i64;
    const size_t n_out = (size_t) n_out_i64;
    if (nx != n_in || ny != n_out) {
        geist_backend_set_error(
            be, GEIST_E_INVALID_ARG,
            "vulkan matvec_q4k: shape mismatch (x=%zu w=%zux%zu y=%zu)",
            nx, n_out, n_in, ny);
        return GEIST_E_INVALID_ARG;
    }
    if ((n_in % 256u) != 0 || n_in > UINT32_MAX ||
        n_out > UINT32_MAX || (x->offset % sizeof(float)) != 0 ||
        (y->offset % sizeof(float)) != 0 || w->offset > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t blocks_per_row = n_in / 256u;
    if (blocks_per_row > UINT32_MAX ||
        n_out > SIZE_MAX / blocks_per_row) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t n_blocks = n_out * blocks_per_row;
    if (n_blocks > SIZE_MAX / 144u) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *xb = x->buffer;
    struct geist_buffer *wb = w->buffer;
    struct geist_buffer *yb = y->buffer;
    if (xb->owner == nullptr || xb->owner != wb->owner ||
        xb->owner != yb->owner || xb->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t x_bytes = nx * sizeof(float);
    const size_t w_bytes = n_blocks * 144u;
    const size_t y_bytes = ny * sizeof(float);
    if (x->offset > (size_t) xb->bytes ||
        w->offset > (size_t) wb->bytes ||
        y->offset > (size_t) yb->bytes ||
        x_bytes > (size_t) xb->bytes - x->offset ||
        w_bytes > (size_t) wb->bytes - w->offset ||
        y_bytes > (size_t) yb->bytes - y->offset) {
        return GEIST_E_INVALID_ARG;
    }
    if ((yb == xb && vulkan_ranges_overlap(y->offset, y_bytes,
                                           x->offset, x_bytes)) ||
        (yb == wb && vulkan_ranges_overlap(y->offset, y_bytes,
                                           w->offset, w_bytes))) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan matvec_q4k: y must not overlap inputs");
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    enum geist_status s = vulkan_ensure_matvec_q4k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    if (xb->host_visible) {
        s = vulkan_flush_if_needed(st, xb);
        if (s != GEIST_OK) { return s; }
    }
    if (wb->host_visible && wb != xb) {
        s = vulkan_flush_if_needed(st, wb);
        if (s != GEIST_OK) { return s; }
    }

    struct geist_buffer *wb_shader = wb;
    uint32_t w_byte_offset = (uint32_t) w->offset;
    bool use_nt4 = false;
    if (vulkan_q4k_nt4_enabled_for(n_out)) {
        s = vulkan_q4k_nt4_cache_get(be, w, n_in, n_out, &wb_shader);
        if (s != GEIST_OK) {
            return s;
        }
        w_byte_offset = 0;
        use_nt4 = true;
    }
    if (use_nt4 && vulkan_q4k_w4a8_enabled()) {
        return vulkan_matvec_q4k_w4a8(be, x, wb_shader, y, n_in, n_out,
                                      blocks_per_row);
    }
    const bool use_nt4_subgroup =
        use_nt4 && vulkan_q4k_nt4_subgroup_enabled_for_state(st);
    if (use_nt4_subgroup) {
        s = vulkan_ensure_matvec_q4k_nt4_subgroup_pipeline(be);
        if (s != GEIST_OK) {
            return s;
        }
    }

    s = vulkan_ensure_transient_descriptor_pool(be, 1u, 3u);
    if (s != GEIST_OK) {
        return s;
    }

    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->transient_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &st->matvec_q4k_set_layout,
    };
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai,
                                               &descriptor_set);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan matvec_q4k: vkAllocateDescriptorSets failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkDescriptorBufferInfo infos[3] = {
        {.buffer = xb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = wb_shader->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = yb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[3] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 3, writes, 0, nullptr);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "matvec_q4k", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);
        return s;
    }
    rc = VK_SUCCESS;
    if (rc == VK_SUCCESS) {
        const VkMemoryBarrier pre_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                             VK_ACCESS_HOST_WRITE_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT,
        };
        st->vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_HOST_BIT |
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

        const uint32_t push[6] = {
            (uint32_t) n_in,
            (uint32_t) n_out,
            (uint32_t) blocks_per_row,
            (uint32_t) (x->offset / sizeof(float)),
            w_byte_offset,
            (uint32_t) (y->offset / sizeof(float)),
        };
        if (use_nt4) {
            const VkPipeline pipeline =
                use_nt4_subgroup ? st->matvec_q4k_nt4_subgroup_pipeline
                                 : st->matvec_q4k_nt4_pipeline;
            const VkPipelineLayout layout =
                use_nt4_subgroup ? st->matvec_q4k_nt4_subgroup_pipeline_layout
                                 : st->matvec_q4k_nt4_pipeline_layout;
            st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  pipeline);
            st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        layout, 0, 1, &descriptor_set, 0,
                                        nullptr);
            st->vkCmdPushConstants(cmd, layout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(push), push);
            st->vkCmdDispatch(cmd, (uint32_t) ((n_out + 3u) / 4u), 1, 1);
        } else {
            st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  st->matvec_q4k_pipeline);
            st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        st->matvec_q4k_pipeline_layout, 0, 1,
                                        &descriptor_set, 0, nullptr);
            st->vkCmdPushConstants(cmd, st->matvec_q4k_pipeline_layout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(push), push);
            st->vkCmdDispatch(cmd, (uint32_t) n_out, 1, 1);
        }

        const VkMemoryBarrier post_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                             VK_ACCESS_SHADER_READ_BIT |
                             VK_ACCESS_HOST_READ_BIT,
        };
        st->vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 1, &post_barrier, 0, nullptr, 0, nullptr);
    }

    s = vulkan_submit_transient_command_buffer(be, cmd, "matvec_q4k");
    vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);

    if (s != GEIST_OK) {
        return s;
    }
    if (yb->host_visible) {
        return vulkan_invalidate_if_needed(st, yb);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_prepare_weight_layout(
    struct geist_backend *be,
    const struct geist_tensor *w) {

    if (be == nullptr || be->state == nullptr || w == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (w->dtype != GEIST_DTYPE_Q4_K ||
        w->layout != GEIST_LAYOUT_BLOCK_QUANTIZED ||
        w->ndim != 2 ||
        w->shape[0] <= 0 ||
        w->shape[1] <= 0) {
        return GEIST_OK;
    }
    const size_t n_out = (size_t) w->shape[0];
    const size_t n_in = (size_t) w->shape[1];
    if (!vulkan_q4k_nt4_enabled_for(n_out)) {
        return GEIST_OK;
    }
    struct vulkan_state *st = be->state;
    if (vulkan_q4k_nt4_cache_find(st, w, n_in, n_out) != nullptr) {
        return GEIST_OK;
    }
    struct geist_buffer *packed = nullptr;
    return vulkan_q4k_nt4_cache_get(be, w, n_in, n_out, &packed);
}

[[nodiscard]] static enum geist_status vulkan_matvec_q6k(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr ||
        x == nullptr || w == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t nx = 0, ny = 0;
    if (!vulkan_tensor_f32_dense_count(x, &nx) ||
        !vulkan_tensor_f32_dense_count(y, &ny) ||
        x->ndim != 1 || y->ndim != 1 ||
        w->buffer == nullptr ||
        w->dtype != GEIST_DTYPE_Q6_K ||
        w->layout != GEIST_LAYOUT_BLOCK_QUANTIZED ||
        w->ndim != 2) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "vulkan matvec_q6k: tensors must be F32 DENSE [in], Q6_K BLOCK_QUANTIZED [out,in], F32 DENSE [out]");
        return GEIST_E_UNSUPPORTED;
    }

    const int64_t n_out_i64 = w->shape[0];
    const int64_t n_in_i64 = w->shape[1];
    if (n_in_i64 <= 0 || n_out_i64 <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t n_in = (size_t) n_in_i64;
    const size_t n_out = (size_t) n_out_i64;
    if (nx != n_in || ny != n_out) {
        geist_backend_set_error(
            be, GEIST_E_INVALID_ARG,
            "vulkan matvec_q6k: shape mismatch (x=%zu w=%zux%zu y=%zu)",
            nx, n_out, n_in, ny);
        return GEIST_E_INVALID_ARG;
    }
    if ((n_in % 256u) != 0 || n_in > UINT32_MAX ||
        n_out > UINT32_MAX || (x->offset % sizeof(float)) != 0 ||
        (y->offset % sizeof(float)) != 0 || w->offset > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t blocks_per_row = n_in / 256u;
    if (blocks_per_row > UINT32_MAX ||
        n_out > SIZE_MAX / blocks_per_row) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t n_blocks = n_out * blocks_per_row;
    if (n_blocks > SIZE_MAX / 210u) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *xb = x->buffer;
    struct geist_buffer *wb = w->buffer;
    struct geist_buffer *yb = y->buffer;
    if (xb->owner == nullptr || xb->owner != wb->owner ||
        xb->owner != yb->owner || xb->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t x_bytes = nx * sizeof(float);
    const size_t w_bytes = n_blocks * 210u;
    const size_t y_bytes = ny * sizeof(float);
    if (x->offset > (size_t) xb->bytes ||
        w->offset > (size_t) wb->bytes ||
        y->offset > (size_t) yb->bytes ||
        x_bytes > (size_t) xb->bytes - x->offset ||
        w_bytes > (size_t) wb->bytes - w->offset ||
        y_bytes > (size_t) yb->bytes - y->offset) {
        return GEIST_E_INVALID_ARG;
    }
    if ((yb == xb && vulkan_ranges_overlap(y->offset, y_bytes,
                                           x->offset, x_bytes)) ||
        (yb == wb && vulkan_ranges_overlap(y->offset, y_bytes,
                                           w->offset, w_bytes))) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan matvec_q6k: y must not overlap inputs");
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    enum geist_status s = vulkan_ensure_matvec_q6k_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    if (xb->host_visible) {
        s = vulkan_flush_if_needed(st, xb);
        if (s != GEIST_OK) { return s; }
    }
    if (wb->host_visible && wb != xb) {
        s = vulkan_flush_if_needed(st, wb);
        if (s != GEIST_OK) { return s; }
    }

    s = vulkan_ensure_transient_descriptor_pool(be, 1u, 3u);
    if (s != GEIST_OK) {
        return s;
    }

    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->transient_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &st->matvec_q6k_set_layout,
    };
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai,
                                               &descriptor_set);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan matvec_q6k: vkAllocateDescriptorSets failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkDescriptorBufferInfo infos[3] = {
        {.buffer = xb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = wb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = yb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[3] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 3, writes, 0, nullptr);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "matvec_q6k", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);
        return s;
    }
    rc = VK_SUCCESS;
    if (rc == VK_SUCCESS) {
        const VkMemoryBarrier pre_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                             VK_ACCESS_HOST_WRITE_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT,
        };
        st->vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_HOST_BIT |
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

        const uint32_t push[6] = {
            (uint32_t) n_in,
            (uint32_t) n_out,
            (uint32_t) blocks_per_row,
            (uint32_t) (x->offset / sizeof(float)),
            (uint32_t) w->offset,
            (uint32_t) (y->offset / sizeof(float)),
        };
        const bool use_n4 = vulkan_q6k_n4_enabled();
        st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              use_n4 ? st->matvec_q6k_n4_pipeline :
                                       st->matvec_q6k_pipeline);
        st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    use_n4 ? st->matvec_q6k_n4_pipeline_layout :
                                             st->matvec_q6k_pipeline_layout,
                                    0, 1,
                                    &descriptor_set, 0, nullptr);
        st->vkCmdPushConstants(cmd,
                               use_n4 ? st->matvec_q6k_n4_pipeline_layout :
                                        st->matvec_q6k_pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(push), push);
        st->vkCmdDispatch(cmd,
                          use_n4 ? (uint32_t) ((n_out + 3u) / 4u) :
                                   (uint32_t) n_out,
                          1, 1);

        const VkMemoryBarrier post_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                             VK_ACCESS_SHADER_READ_BIT |
                             VK_ACCESS_HOST_READ_BIT,
        };
        st->vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 1, &post_barrier, 0, nullptr, 0, nullptr);
    }

    s = vulkan_submit_transient_command_buffer(be, cmd, "matvec_q6k");
    vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);

    if (s != GEIST_OK) {
        return s;
    }
    if (yb->host_visible) {
        return vulkan_invalidate_if_needed(st, yb);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_argmax_f32(
    struct geist_backend *be,
    const struct geist_tensor *logits,
    geist_token_t *out_token) {

    if (be == nullptr || be->state == nullptr ||
        logits == nullptr || out_token == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    *out_token = 0;
    struct vulkan_state *st = be->state;
    if (st->capture_active) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "vulkan argmax: captured argmax requires greedy_head token readback");
        return GEIST_E_UNSUPPORTED;
    }

    size_t n = 0;
    if (!vulkan_tensor_f32_dense_count(logits, &n) ||
        !((logits->ndim == 1) ||
          (logits->ndim == 2 && logits->shape[0] == 1))) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "vulkan argmax: logits must be F32 DENSE [n] or [1,n]");
        return GEIST_E_UNSUPPORTED;
    }
    if (n == 0 || n > UINT32_MAX ||
        n > (size_t) INT32_MAX ||
        (logits->offset % sizeof(float)) != 0) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *xb = logits->buffer;
    if (xb->owner == nullptr || xb->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t x_bytes = n * sizeof(float);
    if (logits->offset > (size_t) xb->bytes ||
        x_bytes > (size_t) xb->bytes - logits->offset) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = vulkan_ensure_argmax_resources(be, 1);
    if (s != GEIST_OK) {
        return s;
    }

    struct geist_buffer *out_buf = st->argmax_result_buffer;
    VkDescriptorSet descriptor_set = st->argmax_descriptor_set;
    uint32_t zero = 0;
    memcpy(out_buf->mapped, &zero, sizeof(zero));
    s = vulkan_flush_if_needed(st, out_buf);
    if (s != GEIST_OK) {
        return s;
    }

    if (xb->host_visible) {
        s = vulkan_flush_if_needed(st, xb);
        if (s != GEIST_OK) {
            return s;
        }
    }

    const VkDescriptorBufferInfo infos[2] = {
        {.buffer = xb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = out_buf->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 2, writes, 0, nullptr);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "argmax", &cmd);
    if (s != GEIST_OK) {
        return s;
    }
    VkResult rc = VK_SUCCESS;
    if (rc == VK_SUCCESS) {
        const VkMemoryBarrier pre_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                             VK_ACCESS_HOST_WRITE_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT,
        };
        st->vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_HOST_BIT |
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

        const uint32_t push[3] = {
            (uint32_t) n,
            (uint32_t) (logits->offset / sizeof(float)),
            0u,
        };
        st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              st->argmax_pipeline);
        st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    st->argmax_pipeline_layout, 0, 1,
                                    &descriptor_set, 0, nullptr);
        st->vkCmdPushConstants(cmd, st->argmax_pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(push), push);
        st->vkCmdDispatch(cmd, 1, 1, 1);

        const VkMemoryBarrier post_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT |
                             VK_ACCESS_TRANSFER_READ_BIT |
                             VK_ACCESS_SHADER_READ_BIT,
        };
        st->vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT |
                                     VK_PIPELINE_STAGE_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &post_barrier, 0, nullptr, 0, nullptr);
    }

    s = vulkan_submit_transient_command_buffer(be, cmd, "argmax");
    if (s != GEIST_OK) {
        return s;
    }

    s = vulkan_invalidate_if_needed(st, out_buf);
    if (s == GEIST_OK) {
        uint32_t token_u32 = 0;
        memcpy(&token_u32, out_buf->mapped, sizeof(token_u32));
        if (token_u32 >= n || token_u32 > (uint32_t) INT32_MAX) {
            s = GEIST_E_BACKEND;
            geist_backend_set_error(be, GEIST_E_BACKEND,
                                    "vulkan argmax: invalid token id %u",
                                    token_u32);
        } else {
            *out_token = (geist_token_t) token_u32;
        }
    }
    return s;
}

static void vulkan_cmd_argmax_batch(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_tensor *logits,
    size_t rows,
    size_t n,
    size_t out_offset) {

    const uint32_t push[5] = {
        (uint32_t) n,
        (uint32_t) rows,
        (uint32_t) (logits->offset / sizeof(float)),
        (uint32_t) n,
        (uint32_t) out_offset,
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->argmax_batch_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->argmax_batch_pipeline_layout, 0, 1,
                                &set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->argmax_batch_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), push);
    st->vkCmdDispatch(cmd, (uint32_t) rows, 1, 1);
}

[[nodiscard]] static enum geist_status vulkan_argmax_f32_batch(
    struct geist_backend *be,
    const struct geist_tensor *logits,
    geist_token_t out_tokens[static logits->shape[0]]) {

    if (be == nullptr || be->state == nullptr ||
        logits == nullptr || out_tokens == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    struct vulkan_state *st = be->state;
    if (st->capture_active) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "vulkan argmax_batch: captured standalone argmax is unsupported");
        return GEIST_E_UNSUPPORTED;
    }
    if (logits->dtype != GEIST_DTYPE_F32 ||
        logits->layout != GEIST_LAYOUT_DENSE ||
        logits->ndim != 2 ||
        logits->shape[0] <= 0 ||
        logits->shape[1] <= 0 ||
        logits->shape[0] > UINT32_MAX ||
        logits->shape[1] > UINT32_MAX ||
        logits->shape[1] > INT32_MAX ||
        logits->stride[0] != logits->shape[1] ||
        logits->stride[1] != 1 ||
        (logits->offset % sizeof(float)) != 0) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "vulkan argmax_batch: expected F32 DENSE [rows,n]");
        return GEIST_E_UNSUPPORTED;
    }

    const size_t rows = (size_t) logits->shape[0];
    const size_t n = (size_t) logits->shape[1];
    if (rows > SIZE_MAX / n) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t elems = rows * n;
    const size_t u32_elem_limit = (size_t) UINT32_MAX + 1u;
    const size_t elem_offset = logits->offset / sizeof(float);
    if (elem_offset >= u32_elem_limit ||
        elems > u32_elem_limit - elem_offset) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *xb = logits->buffer;
    if (xb == nullptr || xb->owner == nullptr || xb->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }
    if (logits->offset > (size_t) xb->bytes ||
        elems > ((size_t) xb->bytes - logits->offset) / sizeof(float)) {
        return GEIST_E_INVALID_ARG;
    }

    enum geist_status s = vulkan_ensure_argmax_batch_pipeline(be);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ensure_argmax_resources(be, rows);
    if (s != GEIST_OK) { return s; }

    struct geist_buffer *out_buf = st->argmax_result_buffer;
    VkDescriptorSet descriptor_set = st->argmax_descriptor_set;
    memset(out_buf->mapped, 0, rows * sizeof(uint32_t));
    s = vulkan_flush_if_needed(st, out_buf);
    if (s != GEIST_OK) { return s; }

    if (xb->host_visible) {
        s = vulkan_flush_if_needed(st, xb);
        if (s != GEIST_OK) { return s; }
    }

    const VkDescriptorBufferInfo infos[2] = {
        {.buffer = xb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = out_buf->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 2, writes, 0, nullptr);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "argmax_batch", &cmd);
    if (s != GEIST_OK) { return s; }
    vulkan_ffn_pre_barrier(st, cmd);
    vulkan_cmd_argmax_batch(st, cmd, descriptor_set, logits, rows, n, 0);
    vulkan_ffn_post_barrier(st, cmd);
    s = vulkan_submit_transient_command_buffer(be, cmd, "argmax_batch");
    if (s != GEIST_OK) { return s; }

    s = vulkan_invalidate_if_needed(st, out_buf);
    if (s != GEIST_OK) { return s; }
    for (size_t row = 0; row < rows; row++) {
        uint32_t token_u32 = 0;
        memcpy(&token_u32,
               (const uint8_t *) out_buf->mapped + row * sizeof(uint32_t),
               sizeof(token_u32));
        if (token_u32 >= n || token_u32 > (uint32_t) INT32_MAX) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "vulkan argmax_batch: invalid token id %u", token_u32);
            return GEIST_E_BACKEND;
        }
        out_tokens[row] = (geist_token_t) token_u32;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_rope_apply(
    struct geist_backend *be,
    struct geist_tensor *x,
    const struct geist_tensor *cos,
    const struct geist_tensor *sin) {

    if (be == nullptr || be->state == nullptr ||
        x == nullptr || cos == nullptr || sin == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t nx = 0, nc = 0, ns = 0;
    if (!vulkan_tensor_f32_dense_count(x, &nx) ||
        !vulkan_tensor_f32_dense_count(cos, &nc) ||
        !vulkan_tensor_f32_dense_count(sin, &ns) ||
        x->ndim != 3 || cos->ndim != 2 || sin->ndim != 2) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "vulkan rope: tensors must be F32 DENSE x[seq,heads,dim], cos/sin[seq,dim]");
        return GEIST_E_UNSUPPORTED;
    }

    const int64_t seq_i = x->shape[0];
    const int64_t heads_i = x->shape[1];
    const int64_t head_dim_i = x->shape[2];
    if (seq_i <= 0 || heads_i <= 0 || head_dim_i <= 0 ||
        (head_dim_i % 2) != 0 ||
        cos->shape[0] != seq_i || sin->shape[0] != seq_i ||
        cos->shape[1] != head_dim_i || sin->shape[1] != head_dim_i ||
        nc != ns || nc != (size_t) seq_i * (size_t) head_dim_i) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan rope: shape mismatch");
        return GEIST_E_INVALID_ARG;
    }

    const size_t seq = (size_t) seq_i;
    const size_t heads = (size_t) heads_i;
    const size_t head_dim = (size_t) head_dim_i;
    const size_t half_dim = head_dim / 2u;
    if (seq > UINT32_MAX || heads > UINT32_MAX || head_dim > UINT32_MAX ||
        half_dim == 0 ||
        seq > SIZE_MAX / heads ||
        seq * heads > SIZE_MAX / half_dim) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t total_pairs = seq * heads * half_dim;
    if (total_pairs == 0 || total_pairs > UINT32_MAX ||
        (x->offset % sizeof(float)) != 0 ||
        (cos->offset % sizeof(float)) != 0 ||
        (sin->offset % sizeof(float)) != 0 ||
        x->offset / sizeof(float) > UINT32_MAX ||
        cos->offset / sizeof(float) > UINT32_MAX ||
        sin->offset / sizeof(float) > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *xb = x->buffer;
    struct geist_buffer *cb = cos->buffer;
    struct geist_buffer *sb = sin->buffer;
    if (xb->owner == nullptr || xb->owner != cb->owner ||
        xb->owner != sb->owner || xb->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t x_bytes = nx * sizeof(float);
    const size_t rope_bytes = nc * sizeof(float);
    if (x->offset > (size_t) xb->bytes ||
        cos->offset > (size_t) cb->bytes ||
        sin->offset > (size_t) sb->bytes ||
        x_bytes > (size_t) xb->bytes - x->offset ||
        rope_bytes > (size_t) cb->bytes - cos->offset ||
        rope_bytes > (size_t) sb->bytes - sin->offset) {
        return GEIST_E_INVALID_ARG;
    }
    if ((xb == cb && vulkan_ranges_overlap(x->offset, x_bytes,
                                           cos->offset, rope_bytes)) ||
        (xb == sb && vulkan_ranges_overlap(x->offset, x_bytes,
                                           sin->offset, rope_bytes))) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan rope: x must not overlap cos/sin");
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    enum geist_status s = vulkan_ensure_rope_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    if (xb->host_visible) {
        s = vulkan_flush_if_needed(st, xb);
        if (s != GEIST_OK) { return s; }
    }
    if (cb->host_visible) {
        s = vulkan_flush_if_needed(st, cb);
        if (s != GEIST_OK) { return s; }
    }
    if (sb->host_visible && sb != cb) {
        s = vulkan_flush_if_needed(st, sb);
        if (s != GEIST_OK) { return s; }
    }

    s = vulkan_ensure_transient_descriptor_pool(be, 1u, 3u);
    if (s != GEIST_OK) {
        return s;
    }

    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->transient_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &st->rope_set_layout,
    };
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai,
                                               &descriptor_set);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan rope: vkAllocateDescriptorSets failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkDescriptorBufferInfo infos[3] = {
        {.buffer = xb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = cb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = sb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[3] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 3, writes, 0, nullptr);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "rope", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);
        return s;
    }
    rc = VK_SUCCESS;
    if (rc == VK_SUCCESS) {
        const VkMemoryBarrier pre_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                             VK_ACCESS_HOST_WRITE_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT,
        };
        st->vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_HOST_BIT |
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

        const uint32_t push[6] = {
            (uint32_t) total_pairs,
            (uint32_t) heads,
            (uint32_t) head_dim,
            (uint32_t) (x->offset / sizeof(float)),
            (uint32_t) (cos->offset / sizeof(float)),
            (uint32_t) (sin->offset / sizeof(float)),
        };
        st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              st->rope_pipeline);
        st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    st->rope_pipeline_layout, 0, 1,
                                    &descriptor_set, 0, nullptr);
        st->vkCmdPushConstants(cmd, st->rope_pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(push), push);
        st->vkCmdDispatch(cmd,
                          (uint32_t) ((total_pairs + 255u) / 256u),
                          1, 1);

        const VkMemoryBarrier post_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                             VK_ACCESS_SHADER_READ_BIT |
                             VK_ACCESS_HOST_READ_BIT,
        };
        st->vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 1, &post_barrier, 0, nullptr, 0, nullptr);
    }

    s = vulkan_submit_transient_command_buffer(be, cmd, "rope");
    vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);

    if (s != GEIST_OK) {
        return s;
    }
    if (xb->host_visible) {
        return vulkan_invalidate_if_needed(st, xb);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_attention(
    struct geist_backend *be,
    const struct geist_tensor *q,
    const struct geist_tensor *k,
    const struct geist_tensor *value,
    size_t q_offset,
    size_t sliding_window,
    struct geist_tensor *out) {

    if (be == nullptr || be->state == nullptr ||
        q == nullptr || k == nullptr || value == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t nq = 0, nk = 0, nv = 0, no = 0;
    if (!vulkan_tensor_f32_dense_count(q, &nq) ||
        !vulkan_tensor_f32_dense_count(k, &nk) ||
        !vulkan_tensor_f32_dense_count(value, &nv) ||
        !vulkan_tensor_f32_dense_count(out, &no) ||
        q->ndim != 3 || k->ndim != 3 ||
        value->ndim != 3 || out->ndim != 3) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "vulkan attention: tensors must be F32 DENSE 3D");
        return GEIST_E_UNSUPPORTED;
    }

    const int64_t n_q_i = q->shape[0];
    const int64_t n_q_heads_i = q->shape[1];
    const int64_t head_dim_i = q->shape[2];
    const int64_t n_kv_i = k->shape[0];
    const int64_t n_kv_heads_i = k->shape[1];
    if (n_q_i <= 0 || n_q_heads_i <= 0 || head_dim_i <= 0 ||
        n_kv_i <= 0 || n_kv_heads_i <= 0 ||
        k->shape[2] != head_dim_i ||
        value->shape[0] != n_kv_i ||
        value->shape[1] != n_kv_heads_i ||
        value->shape[2] != head_dim_i ||
        out->shape[0] != n_q_i ||
        out->shape[1] != n_q_heads_i ||
        out->shape[2] != head_dim_i) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan attention: shape mismatch");
        return GEIST_E_INVALID_ARG;
    }

    const size_t n_q = (size_t) n_q_i;
    const size_t n_q_heads = (size_t) n_q_heads_i;
    const size_t head_dim = (size_t) head_dim_i;
    const size_t n_kv = (size_t) n_kv_i;
    const size_t n_kv_heads = (size_t) n_kv_heads_i;
    if (head_dim > 512u ||
        n_q_heads % n_kv_heads != 0 ||
        n_q > UINT32_MAX ||
        n_kv > UINT32_MAX ||
        n_q_heads > UINT32_MAX ||
        n_kv_heads > UINT32_MAX ||
        head_dim > UINT32_MAX ||
        q_offset > UINT32_MAX ||
        sliding_window > UINT32_MAX ||
        q_offset > SIZE_MAX - n_q ||
        q_offset + n_q > UINT32_MAX ||
        (q->offset % sizeof(float)) != 0 ||
        (k->offset % sizeof(float)) != 0 ||
        (value->offset % sizeof(float)) != 0 ||
        (out->offset % sizeof(float)) != 0 ||
        q->offset / sizeof(float) > UINT32_MAX ||
        k->offset / sizeof(float) > UINT32_MAX ||
        value->offset / sizeof(float) > UINT32_MAX ||
        out->offset / sizeof(float) > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    if (nq != n_q * n_q_heads * head_dim ||
        nk != n_kv * n_kv_heads * head_dim ||
        nv != nk ||
        no != nq) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *qb = q->buffer;
    struct geist_buffer *kb = k->buffer;
    struct geist_buffer *vb = value->buffer;
    struct geist_buffer *ob = out->buffer;
    if (qb->owner == nullptr || qb->owner != kb->owner ||
        qb->owner != vb->owner || qb->owner != ob->owner ||
        qb->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t q_bytes = nq * sizeof(float);
    const size_t kv_bytes = nk * sizeof(float);
    const size_t out_bytes = no * sizeof(float);
    if (q->offset > (size_t) qb->bytes ||
        k->offset > (size_t) kb->bytes ||
        value->offset > (size_t) vb->bytes ||
        out->offset > (size_t) ob->bytes ||
        q_bytes > (size_t) qb->bytes - q->offset ||
        kv_bytes > (size_t) kb->bytes - k->offset ||
        kv_bytes > (size_t) vb->bytes - value->offset ||
        out_bytes > (size_t) ob->bytes - out->offset) {
        return GEIST_E_INVALID_ARG;
    }
    if ((ob == qb && vulkan_ranges_overlap(out->offset, out_bytes,
                                           q->offset, q_bytes)) ||
        (ob == kb && vulkan_ranges_overlap(out->offset, out_bytes,
                                           k->offset, kv_bytes)) ||
        (ob == vb && vulkan_ranges_overlap(out->offset, out_bytes,
                                           value->offset, kv_bytes))) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan attention: out must not overlap inputs");
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    enum geist_status s = vulkan_ensure_attention_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    if (qb->host_visible) {
        s = vulkan_flush_if_needed(st, qb);
        if (s != GEIST_OK) { return s; }
    }
    if (kb->host_visible) {
        s = vulkan_flush_if_needed(st, kb);
        if (s != GEIST_OK) { return s; }
    }
    if (vb->host_visible && vb != kb) {
        s = vulkan_flush_if_needed(st, vb);
        if (s != GEIST_OK) { return s; }
    }

    s = vulkan_ensure_transient_descriptor_pool(be, 1u, 4u);
    if (s != GEIST_OK) {
        return s;
    }

    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->transient_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &st->attention_set_layout,
    };
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai,
                                               &descriptor_set);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan attention: vkAllocateDescriptorSets failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkDescriptorBufferInfo infos[4] = {
        {.buffer = qb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = kb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = vb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = ob->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[4] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[3],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 4, writes, 0, nullptr);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "attention", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);
        return s;
    }
    rc = VK_SUCCESS;
    if (rc == VK_SUCCESS) {
        const VkMemoryBarrier pre_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                             VK_ACCESS_HOST_WRITE_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                             VK_ACCESS_SHADER_WRITE_BIT,
        };
        st->vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_HOST_BIT |
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

        const uint32_t push[11] = {
            (uint32_t) n_q,
            (uint32_t) n_kv,
            (uint32_t) n_q_heads,
            (uint32_t) n_kv_heads,
            (uint32_t) head_dim,
            (uint32_t) q_offset,
            (uint32_t) sliding_window,
            (uint32_t) (q->offset / sizeof(float)),
            (uint32_t) (k->offset / sizeof(float)),
            (uint32_t) (value->offset / sizeof(float)),
            (uint32_t) (out->offset / sizeof(float)),
        };
        st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              st->attention_pipeline);
        st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    st->attention_pipeline_layout, 0, 1,
                                    &descriptor_set, 0, nullptr);
        st->vkCmdPushConstants(cmd, st->attention_pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(push), push);
        st->vkCmdDispatch(cmd, (uint32_t) n_q, (uint32_t) n_q_heads, 1);

        const VkMemoryBarrier post_barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                             VK_ACCESS_SHADER_READ_BIT |
                             VK_ACCESS_HOST_READ_BIT,
        };
        st->vkCmdPipelineBarrier(cmd,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                     VK_PIPELINE_STAGE_HOST_BIT,
                                 0, 1, &post_barrier, 0, nullptr, 0, nullptr);
    }

    s = vulkan_submit_transient_command_buffer(be, cmd, "attention");
    vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);

    if (s != GEIST_OK) {
        return s;
    }
    if (ob->host_visible) {
        return vulkan_invalidate_if_needed(st, ob);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_scale_f32(
    struct geist_backend *be,
    const struct geist_tensor *x,
    float scale,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr || x == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t nx = 0, ny = 0;
    if (!vulkan_tensor_f32_dense_count(x, &nx) ||
        !vulkan_tensor_f32_dense_count(y, &ny)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "vulkan scale_f32: tensors must be F32 DENSE");
        return GEIST_E_UNSUPPORTED;
    }
    if (nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan scale_f32: shape mismatch (x=%zu y=%zu)",
                                nx, ny);
        return GEIST_E_INVALID_ARG;
    }
    if (nx == 0) {
        return GEIST_OK;
    }
    if (nx > UINT32_MAX ||
        (x->offset % sizeof(float)) != 0 ||
        (y->offset % sizeof(float)) != 0) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *xb = x->buffer;
    struct geist_buffer *yb = y->buffer;
    if (xb->owner == nullptr || xb->owner != yb->owner ||
        xb->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t n_bytes = nx * sizeof(float);
    if (x->offset > (size_t) xb->bytes ||
        y->offset > (size_t) yb->bytes ||
        n_bytes > (size_t) xb->bytes - x->offset ||
        n_bytes > (size_t) yb->bytes - y->offset) {
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    enum geist_status s = vulkan_ensure_scale_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    if (xb->host_visible) {
        s = vulkan_flush_if_needed(st, xb);
        if (s != GEIST_OK) { return s; }
    }

    s = vulkan_ensure_transient_descriptor_pool(be, 1u, 2u);
    if (s != GEIST_OK) {
        return s;
    }

    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->transient_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &st->scale_set_layout,
    };
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai,
                                               &descriptor_set);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan scale_f32: vkAllocateDescriptorSets failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkDescriptorBufferInfo infos[2] = {
        {.buffer = xb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = yb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 2, writes, 0, nullptr);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "scale_f32", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);
        return s;
    }

    const VkMemoryBarrier pre_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                         VK_ACCESS_HOST_WRITE_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

    const struct {
        uint32_t n;
        uint32_t x_offset;
        uint32_t y_offset;
        float scale;
    } push = {
        .n = (uint32_t) nx,
        .x_offset = (uint32_t) (x->offset / sizeof(float)),
        .y_offset = (uint32_t) (y->offset / sizeof(float)),
        .scale = scale,
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->scale_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->scale_pipeline_layout, 0, 1,
                                &descriptor_set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->scale_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), &push);
    st->vkCmdDispatch(cmd, (uint32_t) ((nx + 255u) / 256u), 1, 1);

    const VkMemoryBarrier post_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                         VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_HOST_READ_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &post_barrier, 0, nullptr, 0, nullptr);

    s = vulkan_submit_transient_command_buffer(be, cmd, "scale_f32");
    vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);

    if (s != GEIST_OK) {
        return s;
    }
    if (yb->host_visible) {
        return vulkan_invalidate_if_needed(st, yb);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_mul(
    struct geist_backend *be,
    const struct geist_tensor *a,
    const struct geist_tensor *b,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr ||
        a == nullptr || b == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t na = 0, nb = 0, ny = 0;
    if (!vulkan_tensor_f32_dense_count(a, &na) ||
        !vulkan_tensor_f32_dense_count(b, &nb) ||
        !vulkan_tensor_f32_dense_count(y, &ny)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "vulkan mul: all tensors must be F32 DENSE");
        return GEIST_E_UNSUPPORTED;
    }
    if (na != nb || na != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan mul: shape mismatch (a=%zu b=%zu y=%zu)",
                                na, nb, ny);
        return GEIST_E_INVALID_ARG;
    }
    if (na == 0) {
        return GEIST_OK;
    }
    if (na > UINT32_MAX ||
        (a->offset % sizeof(float)) != 0 ||
        (b->offset % sizeof(float)) != 0 ||
        (y->offset % sizeof(float)) != 0) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *ab = a->buffer;
    struct geist_buffer *bb = b->buffer;
    struct geist_buffer *yb = y->buffer;
    if (ab->owner == nullptr || ab->owner != bb->owner ||
        ab->owner != yb->owner || ab->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t n_bytes = na * sizeof(float);
    if (a->offset > (size_t) ab->bytes ||
        b->offset > (size_t) bb->bytes ||
        y->offset > (size_t) yb->bytes ||
        n_bytes > (size_t) ab->bytes - a->offset ||
        n_bytes > (size_t) bb->bytes - b->offset ||
        n_bytes > (size_t) yb->bytes - y->offset) {
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    enum geist_status s = vulkan_ensure_mul_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    if (ab->host_visible) {
        s = vulkan_flush_if_needed(st, ab);
        if (s != GEIST_OK) { return s; }
    }
    if (bb->host_visible && bb != ab) {
        s = vulkan_flush_if_needed(st, bb);
        if (s != GEIST_OK) { return s; }
    }

    s = vulkan_ensure_transient_descriptor_pool(be, 1u, 3u);
    if (s != GEIST_OK) {
        return s;
    }

    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->transient_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &st->mul_set_layout,
    };
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai,
                                               &descriptor_set);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan mul: vkAllocateDescriptorSets failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkDescriptorBufferInfo infos[3] = {
        {.buffer = ab->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = bb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = yb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[3] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 3, writes, 0, nullptr);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "mul", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);
        return s;
    }

    const VkMemoryBarrier pre_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                         VK_ACCESS_HOST_WRITE_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

    const uint32_t push[4] = {
        (uint32_t) na,
        (uint32_t) (a->offset / sizeof(float)),
        (uint32_t) (b->offset / sizeof(float)),
        (uint32_t) (y->offset / sizeof(float)),
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->mul_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->mul_pipeline_layout, 0, 1,
                                &descriptor_set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->mul_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), push);
    st->vkCmdDispatch(cmd, (uint32_t) ((na + 255u) / 256u), 1, 1);

    const VkMemoryBarrier post_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                         VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_HOST_READ_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &post_barrier, 0, nullptr, 0, nullptr);

    s = vulkan_submit_transient_command_buffer(be, cmd, "mul");
    vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);

    if (s != GEIST_OK) {
        return s;
    }
    if (yb->host_visible) {
        return vulkan_invalidate_if_needed(st, yb);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_gelu_tanh(
    struct geist_backend *be,
    const struct geist_tensor *x,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr || x == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t nx = 0, ny = 0;
    if (!vulkan_tensor_f32_dense_count(x, &nx) ||
        !vulkan_tensor_f32_dense_count(y, &ny)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "vulkan gelu_tanh: tensors must be F32 DENSE");
        return GEIST_E_UNSUPPORTED;
    }
    if (nx != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan gelu_tanh: shape mismatch (x=%zu y=%zu)",
                                nx, ny);
        return GEIST_E_INVALID_ARG;
    }
    if (nx == 0) {
        return GEIST_OK;
    }
    if (nx > UINT32_MAX ||
        (x->offset % sizeof(float)) != 0 ||
        (y->offset % sizeof(float)) != 0) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *xb = x->buffer;
    struct geist_buffer *yb = y->buffer;
    if (xb->owner == nullptr || xb->owner != yb->owner ||
        xb->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t n_bytes = nx * sizeof(float);
    if (x->offset > (size_t) xb->bytes ||
        y->offset > (size_t) yb->bytes ||
        n_bytes > (size_t) xb->bytes - x->offset ||
        n_bytes > (size_t) yb->bytes - y->offset) {
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    enum geist_status s = vulkan_ensure_gelu_tanh_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    if (xb->host_visible) {
        s = vulkan_flush_if_needed(st, xb);
        if (s != GEIST_OK) { return s; }
    }

    s = vulkan_ensure_transient_descriptor_pool(be, 1u, 2u);
    if (s != GEIST_OK) {
        return s;
    }

    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->transient_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &st->gelu_tanh_set_layout,
    };
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai,
                                               &descriptor_set);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan gelu_tanh: vkAllocateDescriptorSets failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkDescriptorBufferInfo infos[2] = {
        {.buffer = xb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = yb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[2] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 2, writes, 0, nullptr);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "gelu_tanh", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);
        return s;
    }

    const VkMemoryBarrier pre_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                         VK_ACCESS_HOST_WRITE_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

    const uint32_t push[4] = {
        (uint32_t) nx,
        (uint32_t) (x->offset / sizeof(float)),
        (uint32_t) (y->offset / sizeof(float)),
        0u,
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->gelu_tanh_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->gelu_tanh_pipeline_layout, 0, 1,
                                &descriptor_set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->gelu_tanh_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), push);
    st->vkCmdDispatch(cmd, (uint32_t) ((nx + 255u) / 256u), 1, 1);

    const VkMemoryBarrier post_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                         VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_HOST_READ_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &post_barrier, 0, nullptr, 0, nullptr);

    s = vulkan_submit_transient_command_buffer(be, cmd, "gelu_tanh");
    vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);

    if (s != GEIST_OK) {
        return s;
    }
    if (yb->host_visible) {
        return vulkan_invalidate_if_needed(st, yb);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_add(
    struct geist_backend *be,
    const struct geist_tensor *a,
    const struct geist_tensor *b,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr ||
        a == nullptr || b == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t na = 0, nb = 0, ny = 0;
    if (!vulkan_tensor_f32_dense_count(a, &na) ||
        !vulkan_tensor_f32_dense_count(b, &nb) ||
        !vulkan_tensor_f32_dense_count(y, &ny)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "vulkan add: all tensors must be F32 DENSE");
        return GEIST_E_UNSUPPORTED;
    }
    if (na != nb || na != ny) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan add: shape mismatch (a=%zu b=%zu y=%zu)",
                                na, nb, ny);
        return GEIST_E_INVALID_ARG;
    }
    if (na == 0) {
        return GEIST_OK;
    }
    if (na > UINT32_MAX ||
        (a->offset % sizeof(float)) != 0 ||
        (b->offset % sizeof(float)) != 0 ||
        (y->offset % sizeof(float)) != 0) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *ab = a->buffer;
    struct geist_buffer *bb = b->buffer;
    struct geist_buffer *yb = y->buffer;
    if (ab->owner == nullptr || ab->owner != bb->owner ||
        ab->owner != yb->owner || ab->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t n_bytes = na * sizeof(float);
    if (a->offset > (size_t) ab->bytes ||
        b->offset > (size_t) bb->bytes ||
        y->offset > (size_t) yb->bytes ||
        n_bytes > (size_t) ab->bytes - a->offset ||
        n_bytes > (size_t) bb->bytes - b->offset ||
        n_bytes > (size_t) yb->bytes - y->offset) {
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    enum geist_status s = vulkan_ensure_add_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    if (ab->host_visible) {
        s = vulkan_flush_if_needed(st, ab);
        if (s != GEIST_OK) { return s; }
    }
    if (bb->host_visible && bb != ab) {
        s = vulkan_flush_if_needed(st, bb);
        if (s != GEIST_OK) { return s; }
    }

    s = vulkan_ensure_transient_descriptor_pool(be, 1u, 3u);
    if (s != GEIST_OK) {
        return s;
    }

    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->transient_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &st->add_set_layout,
    };
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai,
                                               &descriptor_set);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan add: vkAllocateDescriptorSets failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkDescriptorBufferInfo infos[3] = {
        {.buffer = ab->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = bb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = yb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[3] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 3, writes, 0, nullptr);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "add", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);
        return s;
    }

    const VkMemoryBarrier pre_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                         VK_ACCESS_HOST_WRITE_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

    const uint32_t push[4] = {
        (uint32_t) na,
        (uint32_t) (a->offset / sizeof(float)),
        (uint32_t) (b->offset / sizeof(float)),
        (uint32_t) (y->offset / sizeof(float)),
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->add_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->add_pipeline_layout, 0, 1,
                                &descriptor_set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->add_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), push);
    st->vkCmdDispatch(cmd, (uint32_t) ((na + 255u) / 256u), 1, 1);

    const VkMemoryBarrier post_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                         VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_HOST_READ_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &post_barrier, 0, nullptr, 0, nullptr);

    s = vulkan_submit_transient_command_buffer(be, cmd, "add");
    vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);

    if (s != GEIST_OK) {
        return s;
    }
    if (yb->host_visible) {
        return vulkan_invalidate_if_needed(st, yb);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_rmsnorm(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    float eps,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr ||
        x == nullptr || w == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t nx = 0, nw = 0, ny = 0;
    if (!vulkan_tensor_f32_dense_count(x, &nx) ||
        !vulkan_tensor_f32_dense_count(w, &nw) ||
        !vulkan_tensor_f32_dense_count(y, &ny) ||
        x->ndim < 1 || w->ndim != 1) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "vulkan rmsnorm: tensors must be F32 DENSE");
        return GEIST_E_UNSUPPORTED;
    }
    const int64_t feat_i64 = x->shape[x->ndim - 1];
    if (feat_i64 <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t feat = (size_t) feat_i64;
    if (nw != feat || nx != ny || nx % feat != 0) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan rmsnorm: shape mismatch (x=%zu w=%zu y=%zu feat=%zu)",
                                nx, nw, ny, feat);
        return GEIST_E_INVALID_ARG;
    }
    const size_t rows = nx / feat;
    if (rows == 0) {
        return GEIST_OK;
    }
    if (rows > UINT32_MAX || feat > UINT32_MAX ||
        (x->offset % sizeof(float)) != 0 ||
        (w->offset % sizeof(float)) != 0 ||
        (y->offset % sizeof(float)) != 0) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *xb = x->buffer;
    struct geist_buffer *wb = w->buffer;
    struct geist_buffer *yb = y->buffer;
    if (xb->owner == nullptr || xb->owner != wb->owner ||
        xb->owner != yb->owner || xb->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t x_bytes = nx * sizeof(float);
    const size_t w_bytes = nw * sizeof(float);
    const size_t y_bytes = ny * sizeof(float);
    if (x->offset > (size_t) xb->bytes ||
        w->offset > (size_t) wb->bytes ||
        y->offset > (size_t) yb->bytes ||
        x_bytes > (size_t) xb->bytes - x->offset ||
        w_bytes > (size_t) wb->bytes - w->offset ||
        y_bytes > (size_t) yb->bytes - y->offset) {
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    enum geist_status s = vulkan_ensure_rmsnorm_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    if (xb->host_visible) {
        s = vulkan_flush_if_needed(st, xb);
        if (s != GEIST_OK) { return s; }
    }
    if (wb->host_visible && wb != xb) {
        s = vulkan_flush_if_needed(st, wb);
        if (s != GEIST_OK) { return s; }
    }

    s = vulkan_ensure_transient_descriptor_pool(be, 1u, 3u);
    if (s != GEIST_OK) {
        return s;
    }

    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->transient_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &st->rmsnorm_set_layout,
    };
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai,
                                               &descriptor_set);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan rmsnorm: vkAllocateDescriptorSets failed (%d)",
                                (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkDescriptorBufferInfo infos[3] = {
        {.buffer = xb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = wb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = yb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[3] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 3, writes, 0, nullptr);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "rmsnorm", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);
        return s;
    }

    const VkMemoryBarrier pre_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                         VK_ACCESS_HOST_WRITE_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

    struct rmsnorm_push {
        uint32_t rows;
        uint32_t feat;
        uint32_t x_offset;
        uint32_t w_offset;
        uint32_t y_offset;
        float eps;
    };
    const struct rmsnorm_push push = {
        .rows = (uint32_t) rows,
        .feat = (uint32_t) feat,
        .x_offset = (uint32_t) (x->offset / sizeof(float)),
        .w_offset = (uint32_t) (w->offset / sizeof(float)),
        .y_offset = (uint32_t) (y->offset / sizeof(float)),
        .eps = eps,
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->rmsnorm_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->rmsnorm_pipeline_layout, 0, 1,
                                &descriptor_set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->rmsnorm_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), &push);
    st->vkCmdDispatch(cmd, (uint32_t) rows, 1, 1);

    const VkMemoryBarrier post_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                         VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_HOST_READ_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &post_barrier, 0, nullptr, 0, nullptr);

    s = vulkan_submit_transient_command_buffer(be, cmd, "rmsnorm");
    vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);

    if (s != GEIST_OK) {
        return s;
    }
    if (yb->host_visible) {
        return vulkan_invalidate_if_needed(st, yb);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_gelu_tanh_mul(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *z,
    struct geist_tensor *y) {

    if (be == nullptr || be->state == nullptr ||
        x == nullptr || z == nullptr || y == nullptr) {
        return GEIST_E_INVALID_ARG;
    }

    size_t nx = 0, nz = 0, ny = 0;
    if (!vulkan_tensor_f32_dense_count(x, &nx) ||
        !vulkan_tensor_f32_dense_count(z, &nz) ||
        !vulkan_tensor_f32_dense_count(y, &ny)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "vulkan gelu_tanh_mul: all tensors must be F32 DENSE");
        return GEIST_E_UNSUPPORTED;
    }
    if (nx != nz || nx != ny) {
        geist_backend_set_error(
            be, GEIST_E_INVALID_ARG,
            "vulkan gelu_tanh_mul: shape mismatch (x=%zu z=%zu y=%zu)",
            nx, nz, ny);
        return GEIST_E_INVALID_ARG;
    }
    if (nx == 0) {
        return GEIST_OK;
    }
    if (nx > UINT32_MAX ||
        (x->offset % sizeof(float)) != 0 ||
        (z->offset % sizeof(float)) != 0 ||
        (y->offset % sizeof(float)) != 0) {
        return GEIST_E_INVALID_ARG;
    }

    struct geist_buffer *xb = x->buffer;
    struct geist_buffer *zb = z->buffer;
    struct geist_buffer *yb = y->buffer;
    if (xb->owner == nullptr || xb->owner != zb->owner ||
        xb->owner != yb->owner || xb->owner != be->state) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t n_bytes = nx * sizeof(float);
    if (x->offset > (size_t) xb->bytes ||
        z->offset > (size_t) zb->bytes ||
        y->offset > (size_t) yb->bytes ||
        n_bytes > (size_t) xb->bytes - x->offset ||
        n_bytes > (size_t) zb->bytes - z->offset ||
        n_bytes > (size_t) yb->bytes - y->offset) {
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    enum geist_status s = vulkan_ensure_gelu_tanh_mul_pipeline(be);
    if (s != GEIST_OK) {
        return s;
    }

    if (xb->host_visible) {
        s = vulkan_flush_if_needed(st, xb);
        if (s != GEIST_OK) { return s; }
    }
    if (zb->host_visible && zb != xb) {
        s = vulkan_flush_if_needed(st, zb);
        if (s != GEIST_OK) { return s; }
    }

    s = vulkan_ensure_transient_descriptor_pool(be, 1u, 3u);
    if (s != GEIST_OK) {
        return s;
    }

    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->transient_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &st->gelu_tanh_mul_set_layout,
    };
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai,
                                               &descriptor_set);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan gelu_tanh_mul: vkAllocateDescriptorSets failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    const VkDescriptorBufferInfo infos[3] = {
        {.buffer = xb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = zb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = yb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[3] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 3, writes, 0, nullptr);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "gelu_tanh_mul", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);
        return s;
    }

    const VkMemoryBarrier pre_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                         VK_ACCESS_HOST_WRITE_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &pre_barrier, 0, nullptr, 0, nullptr);

    const uint32_t push[4] = {
        (uint32_t) nx,
        (uint32_t) (x->offset / sizeof(float)),
        (uint32_t) (z->offset / sizeof(float)),
        (uint32_t) (y->offset / sizeof(float)),
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->gelu_tanh_mul_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->gelu_tanh_mul_pipeline_layout, 0, 1,
                                &descriptor_set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->gelu_tanh_mul_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), push);
    st->vkCmdDispatch(cmd, (uint32_t) ((nx + 255u) / 256u), 1, 1);

    const VkMemoryBarrier post_barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                         VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_HOST_READ_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &post_barrier, 0, nullptr, 0, nullptr);

    s = vulkan_submit_transient_command_buffer(be, cmd, "gelu_tanh_mul");
    vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);

    if (s != GEIST_OK) {
        return s;
    }
    if (yb->host_visible) {
        return vulkan_invalidate_if_needed(st, yb);
    }
    return GEIST_OK;
}

static bool vulkan_ffn_f32_1d_exact(const struct geist_tensor *t, size_t n) {
    return t != nullptr &&
           t->buffer != nullptr &&
           t->dtype == GEIST_DTYPE_F32 &&
           t->layout == GEIST_LAYOUT_DENSE &&
           t->ndim == 1 &&
           t->shape[0] == (int64_t) n &&
           t->stride[0] == 1;
}

static bool vulkan_ffn_f32_2d_exact(const struct geist_tensor *t,
                                    size_t rows,
                                    size_t cols) {
    return t != nullptr &&
           t->buffer != nullptr &&
           t->dtype == GEIST_DTYPE_F32 &&
           t->layout == GEIST_LAYOUT_DENSE &&
           t->ndim == 2 &&
           t->shape[0] == (int64_t) rows &&
           t->shape[1] == (int64_t) cols &&
           t->stride[0] == (int64_t) cols &&
           t->stride[1] == 1;
}

static bool vulkan_rope_f32_2d_compatible(const struct geist_tensor *t,
                                          size_t rows,
                                          size_t cols,
                                          size_t q_position,
                                          bool *uses_positioned_table) {
    if (uses_positioned_table != nullptr) {
        *uses_positioned_table = false;
    }
    if (t == nullptr ||
        t->buffer == nullptr ||
        t->dtype != GEIST_DTYPE_F32 ||
        t->layout != GEIST_LAYOUT_DENSE ||
        t->ndim != 2 ||
        t->shape[0] <= 0 ||
        t->shape[1] != (int64_t) cols ||
        t->stride[0] != (int64_t) cols ||
        t->stride[1] != 1 ||
        q_position > SIZE_MAX - rows) {
        return false;
    }
    const size_t table_rows = (size_t) t->shape[0];
    if (table_rows == rows) {
        return true;
    }
    if (table_rows >= q_position + rows) {
        if (uses_positioned_table != nullptr) {
            *uses_positioned_table = true;
        }
        return true;
    }
    return false;
}

static bool vulkan_ffn_weight_2d_shape(const struct geist_tensor *t,
                                       size_t rows,
                                       size_t cols) {
    return t != nullptr &&
           t->buffer != nullptr &&
           t->ndim == 2 &&
           t->shape[0] == (int64_t) rows &&
           t->shape[1] == (int64_t) cols;
}

static bool vulkan_ffn_weight_supported(const struct geist_tensor *w) {
    if (w == nullptr) {
        return false;
    }
    if (w->dtype == GEIST_DTYPE_F32 && w->layout == GEIST_LAYOUT_DENSE) {
        return true;
    }
    if ((w->dtype == GEIST_DTYPE_F16 || w->dtype == GEIST_DTYPE_BF16) &&
        w->layout == GEIST_LAYOUT_DENSE) {
        return true;
    }
    return w->layout == GEIST_LAYOUT_BLOCK_QUANTIZED &&
           (w->dtype == GEIST_DTYPE_Q4_K || w->dtype == GEIST_DTYPE_Q6_K);
}

static bool vulkan_ffn_f32_3d_exact(const struct geist_tensor *t,
                                    size_t d0,
                                    size_t d1,
                                    size_t d2) {
    return t != nullptr &&
           t->buffer != nullptr &&
           t->dtype == GEIST_DTYPE_F32 &&
           t->layout == GEIST_LAYOUT_DENSE &&
           t->ndim == 3 &&
           t->shape[0] == (int64_t) d0 &&
           t->shape[1] == (int64_t) d1 &&
           t->shape[2] == (int64_t) d2 &&
           t->stride[0] == (int64_t) (d1 * d2) &&
           t->stride[1] == (int64_t) d2 &&
           t->stride[2] == 1;
}

static struct geist_tensor vulkan_ffn_1d_view(const struct geist_tensor *t,
                                              size_t n) {
    struct geist_tensor out = *t;
    out.ndim = 1;
    for (size_t i = 0; i < 8; i++) {
        out.shape[i] = 0;
        out.stride[i] = 0;
    }
    out.shape[0] = (int64_t) n;
    out.stride[0] = 1;
    return out;
}

static void vulkan_ffn_shader_barrier(struct vulkan_state *st,
                                      VkCommandBuffer cmd) {
    const VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
}

static void vulkan_ffn_pre_barrier(struct vulkan_state *st,
                                   VkCommandBuffer cmd) {
    const VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
                         VK_ACCESS_HOST_WRITE_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_SHADER_WRITE_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
}

static void vulkan_ffn_post_barrier(struct vulkan_state *st,
                                    VkCommandBuffer cmd) {
    const VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT |
                         VK_ACCESS_SHADER_READ_BIT |
                         VK_ACCESS_HOST_READ_BIT,
    };
    st->vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT |
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_HOST_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
}

static void vulkan_ffn_write_set(struct vulkan_state *st,
                                 VkDescriptorSet set,
                                 const struct geist_tensor *a,
                                 const struct geist_tensor *b,
                                 const struct geist_tensor *c) {
    const struct geist_buffer *ab = a->buffer;
    const struct geist_buffer *bb = b->buffer;
    const struct geist_buffer *packed_b =
        vulkan_q4k_nt4_cache_find_tensor(st, b);
    if (packed_b != nullptr) {
        bb = packed_b;
    }
    const struct geist_buffer *cb = c->buffer;
    const VkDescriptorBufferInfo infos[3] = {
        {.buffer = ab->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = bb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = cb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[3] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 3, writes, 0, nullptr);
}

static void vulkan_ffn_write_q4k_nt4_pair_set(
    struct vulkan_state *st,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *w0,
    const struct geist_tensor *w1,
    const struct geist_tensor *y0,
    const struct geist_tensor *y1) {

    const struct geist_buffer *w0_packed =
        vulkan_q4k_nt4_cache_find_tensor(st, w0);
    const struct geist_buffer *w1_packed =
        vulkan_q4k_nt4_cache_find_tensor(st, w1);
    const VkDescriptorBufferInfo infos[5] = {
        {.buffer = x->buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = w0_packed->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = w1_packed->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = y0->buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = y1->buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[5] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[3],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[4],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 5, writes, 0, nullptr);
}

static void vulkan_rope_dyn_write_set(
    struct vulkan_state *st,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *cos,
    const struct geist_tensor *sin) {

    const VkDescriptorBufferInfo infos[4] = {
        {.buffer = x->buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = cos->buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = sin->buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {
            .buffer = st->decode_params_buffer->buffer,
            .offset = 0,
            .range = sizeof(struct vulkan_decode_params),
        },
    };
    const VkWriteDescriptorSet writes[4] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[3],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 4, writes, 0, nullptr);
}

static void vulkan_rmsnorm_rope_dyn_write_set(
    struct vulkan_state *st,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    const struct geist_tensor *cos,
    const struct geist_tensor *sin) {

    const VkDescriptorBufferInfo infos[5] = {
        {.buffer = x->buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = w->buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = cos->buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = sin->buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {
            .buffer = st->decode_params_buffer->buffer,
            .offset = 0,
            .range = sizeof(struct vulkan_decode_params),
        },
    };
    const VkWriteDescriptorSet writes[5] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[3],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[4],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 5, writes, 0, nullptr);
}

static void vulkan_ffn_write_gate_up_gelu_q4k_nt4_set(
    struct vulkan_state *st,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *gate_w,
    const struct geist_tensor *up_w,
    const struct geist_tensor *y) {

    const struct geist_buffer *gate_packed =
        vulkan_q4k_nt4_cache_find_tensor(st, gate_w);
    const struct geist_buffer *up_packed =
        vulkan_q4k_nt4_cache_find_tensor(st, up_w);
    const VkDescriptorBufferInfo infos[4] = {
        {.buffer = x->buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = gate_packed->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = up_packed->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = y->buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[4] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[3],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 4, writes, 0, nullptr);
}

static void vulkan_attn_write_set4(struct vulkan_state *st,
                                   VkDescriptorSet set,
                                   const struct geist_tensor *a,
                                   const struct geist_tensor *b,
                                   const struct geist_tensor *c,
                                   const struct geist_tensor *d) {
    const struct geist_buffer *ab = a->buffer;
    const struct geist_buffer *bb = b->buffer;
    const struct geist_buffer *cb = c->buffer;
    const struct geist_buffer *db = d->buffer;
    const VkDescriptorBufferInfo infos[4] = {
        {.buffer = ab->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = bb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = cb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = db->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[4] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[3],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 4, writes, 0, nullptr);
}

static void vulkan_attn_write_set5_params(struct vulkan_state *st,
                                          VkDescriptorSet set,
                                          const struct geist_tensor *a,
                                          const struct geist_tensor *b,
                                          const struct geist_tensor *c,
                                          const struct geist_tensor *d) {
    const struct geist_buffer *ab = a->buffer;
    const struct geist_buffer *bb = b->buffer;
    const struct geist_buffer *cb = c->buffer;
    const struct geist_buffer *db = d->buffer;
    const VkDescriptorBufferInfo infos[5] = {
        {.buffer = ab->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = bb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = cb->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = db->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {
            .buffer = st->decode_params_buffer->buffer,
            .offset = 0,
            .range = sizeof(struct vulkan_decode_params),
        },
    };
    const VkWriteDescriptorSet writes[5] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[3],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[4],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 5, writes, 0, nullptr);
}

static void vulkan_kv_append_dyn_write_set(
    struct vulkan_state *st,
    VkDescriptorSet set,
    const struct geist_tensor *k_scratch,
    const struct geist_tensor *v_scratch,
    const struct geist_tensor *k_cache,
    const struct geist_tensor *v_cache) {

    const VkDescriptorBufferInfo infos[5] = {
        {.buffer = k_scratch->buffer->buffer, .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = v_scratch->buffer->buffer, .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = k_cache->buffer->buffer, .offset = 0,
         .range = VK_WHOLE_SIZE},
        {.buffer = v_cache->buffer->buffer, .offset = 0,
         .range = VK_WHOLE_SIZE},
        {
            .buffer = st->decode_params_buffer->buffer,
            .offset = 0,
            .range = sizeof(struct vulkan_decode_params),
        },
    };
    const VkWriteDescriptorSet writes[5] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[3],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 4,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[4],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 5, writes, 0, nullptr);
}

static bool vulkan_ffn_range_ok(const struct geist_tensor *t,
                                size_t bytes) {
    if (t == nullptr || t->buffer == nullptr) {
        return false;
    }
    const struct geist_buffer *b = t->buffer;
    return t->offset <= (size_t) b->bytes &&
           bytes <= (size_t) b->bytes - t->offset;
}

static bool vulkan_ffn_owner_ok(const struct geist_backend *be,
                                const struct geist_tensor *t) {
    return t != nullptr &&
           t->buffer != nullptr &&
           t->buffer->owner != nullptr &&
           t->buffer->owner == be->state;
}

[[nodiscard]] static enum geist_status vulkan_ffn_flush_read(
    struct vulkan_state *st,
    const struct geist_tensor *t) {

    if (t != nullptr && t->buffer != nullptr && t->buffer->host_visible) {
        return vulkan_flush_if_needed(st, t->buffer);
    }
    return GEIST_OK;
}

struct vulkan_ffn_matvec_dims {
    size_t n_in;
    size_t n_out;
    size_t blocks_per_row;
    size_t weight_bytes;
};

[[nodiscard]] static enum geist_status vulkan_ffn_matvec_dims(
    struct geist_backend *be,
    const struct geist_tensor *w,
    size_t expected_out,
    size_t expected_in,
    struct vulkan_ffn_matvec_dims *out) {

    if (w == nullptr || out == nullptr ||
        w->shape[0] != (int64_t) expected_out ||
        w->shape[1] != (int64_t) expected_in ||
        expected_in == 0 || expected_out == 0 ||
        expected_in > UINT32_MAX ||
        expected_out > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }

    if (w->dtype == GEIST_DTYPE_F32 && w->layout == GEIST_LAYOUT_DENSE) {
        if (expected_out > SIZE_MAX / expected_in ||
            w->offset % sizeof(float) != 0) {
            return GEIST_E_INVALID_ARG;
        }
        *out = (struct vulkan_ffn_matvec_dims){
            .n_in = expected_in,
            .n_out = expected_out,
            .blocks_per_row = 0,
            .weight_bytes = expected_out * expected_in * sizeof(float),
        };
        return GEIST_OK;
    }

    if ((w->dtype == GEIST_DTYPE_F16 || w->dtype == GEIST_DTYPE_BF16) &&
        w->layout == GEIST_LAYOUT_DENSE) {
        if (expected_out > SIZE_MAX / expected_in ||
            w->offset % sizeof(uint32_t) != 0 ||
            (expected_in % 2u) != 0) {
            return GEIST_E_INVALID_ARG;
        }
        *out = (struct vulkan_ffn_matvec_dims){
            .n_in = expected_in,
            .n_out = expected_out,
            .blocks_per_row = 0,
            .weight_bytes = expected_out * expected_in * sizeof(uint16_t),
        };
        return GEIST_OK;
    }

    if (w->layout != GEIST_LAYOUT_BLOCK_QUANTIZED ||
        (w->dtype != GEIST_DTYPE_Q4_K && w->dtype != GEIST_DTYPE_Q6_K)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "vulkan ffn_geglu_block: unsupported weight dtype/layout");
        return GEIST_E_UNSUPPORTED;
    }
    if ((expected_in % 256u) != 0 || w->offset > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t blocks_per_row = expected_in / 256u;
    const size_t block_bytes = (w->dtype == GEIST_DTYPE_Q4_K) ? 144u : 210u;
    if (blocks_per_row > UINT32_MAX ||
        expected_out > SIZE_MAX / blocks_per_row) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t n_blocks = expected_out * blocks_per_row;
    if (n_blocks > SIZE_MAX / block_bytes) {
        return GEIST_E_INVALID_ARG;
    }
    *out = (struct vulkan_ffn_matvec_dims){
        .n_in = expected_in,
        .n_out = expected_out,
        .blocks_per_row = blocks_per_row,
        .weight_bytes = n_blocks * block_bytes,
    };
    return GEIST_OK;
}

static VkDescriptorSetLayout vulkan_ffn_matvec_set_layout(
    const struct vulkan_state *st,
    const struct geist_tensor *w) {

    if (w->dtype == GEIST_DTYPE_Q4_K) {
        return st->matvec_q4k_set_layout;
    }
    if (w->dtype == GEIST_DTYPE_Q6_K) {
        return st->matvec_q6k_set_layout;
    }
    if (w->dtype == GEIST_DTYPE_F16 || w->dtype == GEIST_DTYPE_BF16) {
        return st->matvec_f16bf16_set_layout;
    }
    return st->matvec_set_layout;
}

static void vulkan_ffn_cmd_matmul_f32(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    const struct geist_tensor *y,
    size_t rows,
    size_t n_in,
    size_t n_out) {

    const uint32_t push[8] = {
        (uint32_t) n_in,
        (uint32_t) n_out,
        (uint32_t) rows,
        (uint32_t) (x->offset / sizeof(float)),
        (uint32_t) (w->offset / sizeof(float)),
        (uint32_t) (y->offset / sizeof(float)),
        (uint32_t) n_in,
        (uint32_t) n_out,
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->matmul_f32_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->matmul_f32_pipeline_layout, 0, 1,
                                &set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->matmul_f32_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), push);
    st->vkCmdDispatch(cmd, (uint32_t) n_out, (uint32_t) rows, 1);
}

static void vulkan_ffn_cmd_matmul_q4k(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    const struct geist_tensor *y,
    size_t rows,
    const struct vulkan_ffn_matvec_dims *dims) {

    const uint32_t push[9] = {
        (uint32_t) dims->n_in,
        (uint32_t) dims->n_out,
        (uint32_t) dims->blocks_per_row,
        (uint32_t) rows,
        (uint32_t) (x->offset / sizeof(float)),
        (uint32_t) w->offset,
        (uint32_t) (y->offset / sizeof(float)),
        (uint32_t) dims->n_in,
        (uint32_t) dims->n_out,
    };
    if (rows > 1u) {
        st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              st->matmul_q4k_m8_pipeline);
        st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    st->matmul_q4k_m8_pipeline_layout, 0, 1,
                                    &set, 0, nullptr);
        st->vkCmdPushConstants(cmd, st->matmul_q4k_m8_pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(push), push);
        st->vkCmdDispatch(cmd, (uint32_t) dims->n_out,
                          (uint32_t) ((rows + 7u) / 8u), 1);
        return;
    }

    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->matmul_q4k_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->matmul_q4k_pipeline_layout, 0, 1,
                                &set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->matmul_q4k_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), push);
    st->vkCmdDispatch(cmd, (uint32_t) dims->n_out, 1, 1);
}

static void vulkan_ffn_cmd_matmul_q6k(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    const struct geist_tensor *y,
    size_t rows,
    const struct vulkan_ffn_matvec_dims *dims) {

    const uint32_t push[9] = {
        (uint32_t) dims->n_in,
        (uint32_t) dims->n_out,
        (uint32_t) dims->blocks_per_row,
        (uint32_t) rows,
        (uint32_t) (x->offset / sizeof(float)),
        (uint32_t) w->offset,
        (uint32_t) (y->offset / sizeof(float)),
        (uint32_t) dims->n_in,
        (uint32_t) dims->n_out,
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->matmul_q6k_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->matmul_q6k_pipeline_layout, 0, 1,
                                &set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->matmul_q6k_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), push);
    st->vkCmdDispatch(cmd, (uint32_t) dims->n_out, (uint32_t) rows, 1);
}

[[nodiscard]] static enum geist_status vulkan_matmul_qk(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    struct geist_tensor *y,
    bool q6) {

    if (be == nullptr || be->state == nullptr ||
        x == nullptr || w == nullptr || y == nullptr ||
        x->ndim != 2 || w->ndim != 2 || y->ndim != 2 ||
        x->shape[0] <= 0 || x->shape[1] <= 0 ||
        w->shape[0] <= 0 || w->shape[1] != x->shape[1] ||
        y->shape[0] != x->shape[0] || y->shape[1] != w->shape[0]) {
        geist_backend_set_error(
            be, GEIST_E_INVALID_ARG,
            "vulkan matmul_qk: shape contract failed "
            "x[%lld,%lld] w[%lld,%lld] y[%lld,%lld]",
            x != nullptr ? (long long) x->shape[0] : -1ll,
            x != nullptr ? (long long) x->shape[1] : -1ll,
            w != nullptr ? (long long) w->shape[0] : -1ll,
            w != nullptr ? (long long) w->shape[1] : -1ll,
            y != nullptr ? (long long) y->shape[0] : -1ll,
            y != nullptr ? (long long) y->shape[1] : -1ll);
        return GEIST_E_INVALID_ARG;
    }
    if (w->dtype != (q6 ? GEIST_DTYPE_Q6_K : GEIST_DTYPE_Q4_K) ||
        w->layout != GEIST_LAYOUT_BLOCK_QUANTIZED) {
        return GEIST_E_UNSUPPORTED;
    }

    const size_t rows = (size_t) x->shape[0];
    const size_t n_in = (size_t) x->shape[1];
    const size_t n_out = (size_t) w->shape[0];
    if (!vulkan_ffn_f32_2d_exact(x, rows, n_in) ||
        !vulkan_ffn_f32_2d_exact(y, rows, n_out) ||
        !vulkan_ffn_weight_2d_shape(w, n_out, n_in) ||
        rows > UINT32_MAX) {
        geist_backend_set_error(
            be, GEIST_E_INVALID_ARG,
            "vulkan matmul_qk: tensor layout contract failed "
            "x_stride[%lld,%lld] y_stride[%lld,%lld] rows=%zu",
            (long long) x->stride[0], (long long) x->stride[1],
            (long long) y->stride[0], (long long) y->stride[1], rows);
        return GEIST_E_INVALID_ARG;
    }

    if (!q6 && rows == 1u && vulkan_q4k_nt4_enabled_for(n_out)) {
        struct geist_tensor x1 = *x;
        struct geist_tensor y1 = *y;
        x1.ndim = 1;
        x1.shape[0] = (int64_t) n_in;
        x1.shape[1] = 0;
        x1.stride[0] = 1;
        x1.stride[1] = 0;
        y1.ndim = 1;
        y1.shape[0] = (int64_t) n_out;
        y1.shape[1] = 0;
        y1.stride[0] = 1;
        y1.stride[1] = 0;
        return vulkan_matvec_q4k(be, &x1, w, &y1);
    }
    if (q6 && rows == 1u) {
        struct geist_tensor x1 = *x;
        struct geist_tensor y1 = *y;
        x1.ndim = 1;
        x1.shape[0] = (int64_t) n_in;
        x1.shape[1] = 0;
        x1.stride[0] = 1;
        x1.stride[1] = 0;
        y1.ndim = 1;
        y1.shape[0] = (int64_t) n_out;
        y1.shape[1] = 0;
        y1.stride[0] = 1;
        y1.stride[1] = 0;
        return vulkan_matvec_q6k(be, &x1, w, &y1);
    }

    struct vulkan_ffn_matvec_dims dims;
    enum geist_status s =
        vulkan_ffn_matvec_dims(be, w, n_out, n_in, &dims);
    if (s != GEIST_OK) {
        return s;
    }
    const size_t x_bytes = rows * n_in * sizeof(float);
    const size_t y_bytes = rows * n_out * sizeof(float);
    if (!vulkan_ffn_owner_ok(be, x) ||
        !vulkan_ffn_owner_ok(be, w) ||
        !vulkan_ffn_owner_ok(be, y) ||
        !vulkan_ffn_range_ok(x, x_bytes) ||
        !vulkan_ffn_range_ok(w, dims.weight_bytes) ||
        !vulkan_ffn_range_ok(y, y_bytes)) {
        geist_backend_set_error(
            be, GEIST_E_INVALID_ARG,
            "vulkan matmul_qk: owner/range contract failed "
            "x_bytes=%zu w_bytes=%zu y_bytes=%zu",
            x_bytes, dims.weight_bytes, y_bytes);
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    s = q6 ? vulkan_ensure_matmul_q6k_pipeline(be)
           : (rows > 1u ? vulkan_ensure_matmul_q4k_m8_pipeline(be)
                         : vulkan_ensure_matmul_q4k_pipeline(be));
    if (s != GEIST_OK) {
        return s;
    }
    s = vulkan_ffn_flush_read(st, x);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ffn_flush_read(st, w);
    if (s != GEIST_OK) { return s; }

    s = vulkan_ensure_transient_descriptor_pool(be, 1u, 3u);
    if (s != GEIST_OK) {
        return s;
    }

    const VkDescriptorSetLayout set_layout =
        q6 ? st->matvec_q6k_set_layout : st->matvec_q4k_set_layout;
    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->transient_descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &set_layout,
    };
    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai,
                                               &descriptor_set);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan matmul_qk: vkAllocateDescriptorSets failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }
    vulkan_ffn_write_set(st, descriptor_set, x, w, y);

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "matmul_qk", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);
        return s;
    }
    vulkan_ffn_pre_barrier(st, cmd);
    if (q6) {
        vulkan_ffn_cmd_matmul_q6k(st, cmd, descriptor_set, x, w, y,
                                  rows, &dims);
    } else {
        vulkan_ffn_cmd_matmul_q4k(st, cmd, descriptor_set, x, w, y,
                                  rows, &dims);
    }
    vulkan_ffn_post_barrier(st, cmd);

    s = vulkan_submit_transient_command_buffer(be, cmd, "matmul_qk");
    vulkan_free_transient_descriptor_sets(st, 1u, &descriptor_set);
    if (s != GEIST_OK) {
        return s;
    }
    if (y->buffer->host_visible) {
        return vulkan_invalidate_if_needed(st, y->buffer);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_matmul_q4k(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    struct geist_tensor *y) {

    return vulkan_matmul_qk(be, x, w, y, false);
}

[[nodiscard]] static enum geist_status vulkan_matmul_q6k(
    struct geist_backend *be,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    struct geist_tensor *y) {

    return vulkan_matmul_qk(be, x, w, y, true);
}

static void vulkan_ffn_cmd_matvec(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    const struct geist_tensor *y,
    const struct vulkan_ffn_matvec_dims *dims) {

    if (w->dtype == GEIST_DTYPE_Q4_K) {
        const bool use_nt4 =
            vulkan_q4k_nt4_cache_find(st, w, dims->n_in,
                                      dims->n_out) != nullptr;
        const uint32_t push[6] = {
            (uint32_t) dims->n_in,
            (uint32_t) dims->n_out,
            (uint32_t) dims->blocks_per_row,
            (uint32_t) (x->offset / sizeof(float)),
            use_nt4 ? 0u : (uint32_t) w->offset,
            (uint32_t) (y->offset / sizeof(float)),
        };
        if (use_nt4) {
            const bool use_subgroup =
                vulkan_q4k_nt4_subgroup_enabled_for_state(st) &&
                st->matvec_q4k_nt4_subgroup_pipeline != VK_NULL_HANDLE;
            const VkPipeline pipeline =
                use_subgroup ? st->matvec_q4k_nt4_subgroup_pipeline
                             : st->matvec_q4k_nt4_pipeline;
            const VkPipelineLayout layout =
                use_subgroup ? st->matvec_q4k_nt4_subgroup_pipeline_layout
                             : st->matvec_q4k_nt4_pipeline_layout;
            st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  pipeline);
            st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        layout, 0, 1, &set, 0, nullptr);
            st->vkCmdPushConstants(cmd, layout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(push), push);
        } else {
            st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  st->matvec_q4k_n4_pipeline);
            st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        st->matvec_q4k_n4_pipeline_layout, 0,
                                        1, &set, 0, nullptr);
            st->vkCmdPushConstants(cmd, st->matvec_q4k_n4_pipeline_layout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(push), push);
        }
        st->vkCmdDispatch(cmd, (uint32_t) ((dims->n_out + 3u) / 4u), 1, 1);
        return;
    }
    if (w->dtype == GEIST_DTYPE_Q6_K) {
        const uint32_t push[6] = {
            (uint32_t) dims->n_in,
            (uint32_t) dims->n_out,
            (uint32_t) dims->blocks_per_row,
            (uint32_t) (x->offset / sizeof(float)),
            (uint32_t) w->offset,
            (uint32_t) (y->offset / sizeof(float)),
        };
        const bool use_n4 = vulkan_q6k_n4_enabled();
        st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              use_n4 ? st->matvec_q6k_n4_pipeline :
                                       st->matvec_q6k_pipeline);
        st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    use_n4 ? st->matvec_q6k_n4_pipeline_layout :
                                             st->matvec_q6k_pipeline_layout,
                                    0, 1,
                                    &set, 0, nullptr);
        st->vkCmdPushConstants(cmd,
                               use_n4 ? st->matvec_q6k_n4_pipeline_layout :
                                        st->matvec_q6k_pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(push), push);
        st->vkCmdDispatch(cmd,
                          use_n4 ? (uint32_t) ((dims->n_out + 3u) / 4u) :
                                   (uint32_t) dims->n_out,
                          1, 1);
        return;
    }
    if (w->dtype == GEIST_DTYPE_F16 || w->dtype == GEIST_DTYPE_BF16) {
        const uint32_t push[6] = {
            (uint32_t) dims->n_in,
            (uint32_t) dims->n_out,
            (uint32_t) w->dtype,
            (uint32_t) (x->offset / sizeof(float)),
            (uint32_t) (w->offset / sizeof(uint16_t)),
            (uint32_t) (y->offset / sizeof(float)),
        };
        st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              st->matvec_f16bf16_pipeline);
        st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    st->matvec_f16bf16_pipeline_layout, 0, 1,
                                    &set, 0, nullptr);
        st->vkCmdPushConstants(cmd, st->matvec_f16bf16_pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(push), push);
        st->vkCmdDispatch(cmd, (uint32_t) dims->n_out, 1, 1);
        return;
    }

    const uint32_t push[5] = {
        (uint32_t) dims->n_in,
        (uint32_t) dims->n_out,
        (uint32_t) (x->offset / sizeof(float)),
        (uint32_t) (w->offset / sizeof(float)),
        (uint32_t) (y->offset / sizeof(float)),
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->matvec_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->matvec_pipeline_layout, 0, 1,
                                &set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->matvec_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), push);
    st->vkCmdDispatch(cmd, (uint32_t) dims->n_out, 1, 1);
}

static void vulkan_ffn_cmd_matvec_q4k_nt4_pair(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *w0,
    const struct geist_tensor *w1,
    const struct geist_tensor *y0,
    const struct geist_tensor *y1,
    const struct vulkan_ffn_matvec_dims *dims) {

    (void) w0;
    (void) w1;
    const uint32_t push[8] = {
        (uint32_t) dims->n_in,
        (uint32_t) dims->n_out,
        (uint32_t) dims->blocks_per_row,
        (uint32_t) (x->offset / sizeof(float)),
        0u,
        0u,
        (uint32_t) (y0->offset / sizeof(float)),
        (uint32_t) (y1->offset / sizeof(float)),
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->matvec_q4k_nt4_pair_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->matvec_q4k_nt4_pair_pipeline_layout, 0,
                                1, &set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->matvec_q4k_nt4_pair_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                           push);
    st->vkCmdDispatch(cmd, (uint32_t) ((dims->n_out + 3u) / 4u), 1, 1);
}

static void vulkan_ffn_cmd_gate_up_gelu_q4k(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *gate_w,
    const struct geist_tensor *up_w,
    const struct geist_tensor *y,
    const struct vulkan_ffn_matvec_dims *dims) {

    const uint32_t push[7] = {
        (uint32_t) dims->n_in,
        (uint32_t) dims->n_out,
        (uint32_t) dims->blocks_per_row,
        (uint32_t) (x->offset / sizeof(float)),
        (uint32_t) gate_w->offset,
        (uint32_t) up_w->offset,
        (uint32_t) (y->offset / sizeof(float)),
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->ffn_gate_up_gelu_q4k_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->ffn_gate_up_gelu_q4k_pipeline_layout, 0,
                                1, &set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->ffn_gate_up_gelu_q4k_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                           push);
    st->vkCmdDispatch(cmd, (uint32_t) ((dims->n_out + 3u) / 4u), 1, 1);
}

static void vulkan_ffn_cmd_gate_up_gelu_q4k_nt4(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *gate_w,
    const struct geist_tensor *up_w,
    const struct geist_tensor *y,
    const struct vulkan_ffn_matvec_dims *dims) {

    (void) gate_w;
    (void) up_w;
    const uint32_t push[7] = {
        (uint32_t) dims->n_in,
        (uint32_t) dims->n_out,
        (uint32_t) dims->blocks_per_row,
        (uint32_t) (x->offset / sizeof(float)),
        0u,
        0u,
        (uint32_t) (y->offset / sizeof(float)),
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->ffn_gate_up_gelu_q4k_nt4_pipeline);
    st->vkCmdBindDescriptorSets(
        cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        st->ffn_gate_up_gelu_q4k_nt4_pipeline_layout, 0, 1, &set, 0,
        nullptr);
    st->vkCmdPushConstants(
        cmd, st->ffn_gate_up_gelu_q4k_nt4_pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
    st->vkCmdDispatch(cmd, (uint32_t) ((dims->n_out + 3u) / 4u), 1, 1);
}

static void vulkan_ffn_cmd_matmul_or_matvec(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    const struct geist_tensor *y,
    size_t rows,
    const struct vulkan_ffn_matvec_dims *dims) {

    if (rows <= 1u) {
        vulkan_ffn_cmd_matvec(st, cmd, set, x, w, y, dims);
        return;
    }
    if (w->dtype == GEIST_DTYPE_F32) {
        vulkan_ffn_cmd_matmul_f32(st, cmd, set, x, w, y, rows,
                                  dims->n_in, dims->n_out);
        return;
    }
    if (w->dtype == GEIST_DTYPE_Q4_K) {
        vulkan_ffn_cmd_matmul_q4k(st, cmd, set, x, w, y, rows, dims);
        return;
    }
    if (w->dtype == GEIST_DTYPE_Q6_K) {
        vulkan_ffn_cmd_matmul_q6k(st, cmd, set, x, w, y, rows, dims);
        return;
    }
}

static void vulkan_attn_cmd_rmsnorm(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    const struct geist_tensor *y,
    size_t rows,
    size_t feat,
    float eps) {

    struct rmsnorm_push {
        uint32_t rows;
        uint32_t feat;
        uint32_t x_offset;
        uint32_t w_offset;
        uint32_t y_offset;
        float eps;
    };
    const struct rmsnorm_push push = {
        .rows = (uint32_t) rows,
        .feat = (uint32_t) feat,
        .x_offset = (uint32_t) (x->offset / sizeof(float)),
        .w_offset = (uint32_t) (w->offset / sizeof(float)),
        .y_offset = (uint32_t) (y->offset / sizeof(float)),
        .eps = eps,
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->rmsnorm_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->rmsnorm_pipeline_layout, 0, 1,
                                &set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->rmsnorm_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), &push);
    st->vkCmdDispatch(cmd, (uint32_t) rows, 1, 1);
}

static void vulkan_write_rmsnorm_add_set(
    struct vulkan_state *st,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    const struct geist_tensor *residual,
    const struct geist_tensor *y) {

    const VkDescriptorBufferInfo infos[4] = {
        {.buffer = x->buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = w->buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = residual->buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
        {.buffer = y->buffer->buffer, .offset = 0, .range = VK_WHOLE_SIZE},
    };
    const VkWriteDescriptorSet writes[4] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[2],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &infos[3],
        },
    };
    st->vkUpdateDescriptorSets(st->device, 4, writes, 0, nullptr);
}

static void vulkan_cmd_rmsnorm_add(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    const struct geist_tensor *residual,
    const struct geist_tensor *y,
    size_t rows,
    size_t feat,
    float eps) {

    struct rmsnorm_add_push {
        uint32_t rows;
        uint32_t feat;
        uint32_t x_offset;
        uint32_t w_offset;
        uint32_t residual_offset;
        uint32_t y_offset;
        float eps;
    };
    const struct rmsnorm_add_push push = {
        .rows = (uint32_t) rows,
        .feat = (uint32_t) feat,
        .x_offset = (uint32_t) (x->offset / sizeof(float)),
        .w_offset = (uint32_t) (w->offset / sizeof(float)),
        .residual_offset = (uint32_t) (residual->offset / sizeof(float)),
        .y_offset = (uint32_t) (y->offset / sizeof(float)),
        .eps = eps,
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->rmsnorm_add_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->rmsnorm_add_pipeline_layout, 0, 1,
                                &set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->rmsnorm_add_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), &push);
    st->vkCmdDispatch(cmd, (uint32_t) rows, 1, 1);
}

[[nodiscard]] static enum geist_status vulkan_ple_block(
    struct geist_backend *be,
    const struct geist_backend_ple_block *block) {

    if (be == nullptr || be->state == nullptr || block == nullptr ||
        block->struct_size < sizeof(*block) ||
        block->seq == 0 ||
        block->d_model == 0 ||
        block->hidden_per_layer == 0) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t rows = block->seq;
    const size_t d = block->d_model;
    const size_t hpl = block->hidden_per_layer;
    if (rows > UINT32_MAX || d > UINT32_MAX || hpl > UINT32_MAX ||
        rows > SIZE_MAX / d ||
        rows > SIZE_MAX / hpl) {
        return GEIST_E_INVALID_ARG;
    }
    if (!vulkan_ffn_f32_2d_exact(block->hidden, rows, d) ||
        !vulkan_ffn_f32_2d_exact(block->per_layer_input, rows, hpl) ||
        !vulkan_ffn_weight_2d_shape(block->per_layer_gate_weight, hpl, d) ||
        !vulkan_ffn_weight_2d_shape(block->per_layer_proj_weight, d, hpl) ||
        !vulkan_ffn_f32_1d_exact(block->post_per_layer_norm_weight, d) ||
        !vulkan_ffn_f32_2d_exact(block->gate_scratch, rows, hpl) ||
        !vulkan_ffn_f32_2d_exact(block->proj_scratch, rows, d) ||
        !vulkan_ffn_f32_2d_exact(block->out, rows, d)) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan ple_block: tensor shape mismatch");
        return GEIST_E_INVALID_ARG;
    }
    if (!vulkan_ffn_weight_supported(block->per_layer_gate_weight) ||
        !vulkan_ffn_weight_supported(block->per_layer_proj_weight)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "vulkan ple_block: unsupported weight dtype/layout");
        return GEIST_E_UNSUPPORTED;
    }

    struct vulkan_ffn_matvec_dims gate_dims;
    struct vulkan_ffn_matvec_dims proj_dims;
    enum geist_status s =
        vulkan_ffn_matvec_dims(be, block->per_layer_gate_weight, hpl, d,
                               &gate_dims);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ffn_matvec_dims(be, block->per_layer_proj_weight, d, hpl,
                               &proj_dims);
    if (s != GEIST_OK) { return s; }

    const size_t hidden_elems = rows * d;
    const size_t gate_elems = rows * hpl;
    const size_t hidden_bytes = hidden_elems * sizeof(float);
    const size_t gate_bytes = gate_elems * sizeof(float);
    if (!vulkan_ffn_owner_ok(be, block->hidden) ||
        !vulkan_ffn_owner_ok(be, block->per_layer_input) ||
        !vulkan_ffn_owner_ok(be, block->per_layer_gate_weight) ||
        !vulkan_ffn_owner_ok(be, block->per_layer_proj_weight) ||
        !vulkan_ffn_owner_ok(be, block->post_per_layer_norm_weight) ||
        !vulkan_ffn_owner_ok(be, block->gate_scratch) ||
        !vulkan_ffn_owner_ok(be, block->proj_scratch) ||
        !vulkan_ffn_owner_ok(be, block->out) ||
        !vulkan_ffn_range_ok(block->hidden, hidden_bytes) ||
        !vulkan_ffn_range_ok(block->per_layer_input, gate_bytes) ||
        !vulkan_ffn_range_ok(block->per_layer_gate_weight,
                             gate_dims.weight_bytes) ||
        !vulkan_ffn_range_ok(block->per_layer_proj_weight,
                             proj_dims.weight_bytes) ||
        !vulkan_ffn_range_ok(block->post_per_layer_norm_weight,
                             d * sizeof(float)) ||
        !vulkan_ffn_range_ok(block->gate_scratch, gate_bytes) ||
        !vulkan_ffn_range_ok(block->proj_scratch, hidden_bytes) ||
        !vulkan_ffn_range_ok(block->out, hidden_bytes) ||
        (block->hidden->offset % sizeof(float)) != 0 ||
        (block->per_layer_input->offset % sizeof(float)) != 0 ||
        (block->post_per_layer_norm_weight->offset % sizeof(float)) != 0 ||
        (block->gate_scratch->offset % sizeof(float)) != 0 ||
        (block->proj_scratch->offset % sizeof(float)) != 0 ||
        (block->out->offset % sizeof(float)) != 0) {
        return GEIST_E_INVALID_ARG;
    }

    if (block->per_layer_gate_weight->dtype == GEIST_DTYPE_Q4_K ||
        block->per_layer_proj_weight->dtype == GEIST_DTYPE_Q4_K) {
        s = rows > 1u ? vulkan_ensure_matmul_q4k_m8_pipeline(be)
                      : vulkan_ensure_matvec_q4k_pipeline(be);
        if (s != GEIST_OK) { return s; }
        if (rows == 1u) {
            s = vulkan_ensure_q4k_nt4_subgroup_if_enabled(be);
            if (s != GEIST_OK) { return s; }
        }
    }
    if (block->per_layer_gate_weight->dtype == GEIST_DTYPE_Q6_K ||
        block->per_layer_proj_weight->dtype == GEIST_DTYPE_Q6_K) {
        s = rows > 1u ? vulkan_ensure_matmul_q6k_pipeline(be)
                      : vulkan_ensure_matvec_q6k_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }
    if (block->per_layer_gate_weight->dtype == GEIST_DTYPE_F32 ||
        block->per_layer_proj_weight->dtype == GEIST_DTYPE_F32) {
        s = rows > 1u ? vulkan_ensure_matmul_f32_pipeline(be)
                      : vulkan_ensure_matvec_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }
    s = vulkan_ensure_gelu_tanh_mul_pipeline(be);
    if (s != GEIST_OK) { return s; }
    const bool use_rmsnorm_add =
        vulkan_env_is_on("GEIST_VULKAN_PLE_RMSNORM_ADD");
    if (use_rmsnorm_add) {
        s = vulkan_ensure_rmsnorm_add_pipeline(be);
        if (s != GEIST_OK) { return s; }
    } else {
        s = vulkan_ensure_rmsnorm_pipeline(be);
        if (s != GEIST_OK) { return s; }
        s = vulkan_ensure_add_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }

    struct vulkan_state *st = be->state;
    s = vulkan_ffn_flush_read(st, block->hidden);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ffn_flush_read(st, block->per_layer_input);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ffn_flush_read(st, block->per_layer_gate_weight);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ffn_flush_read(st, block->per_layer_proj_weight);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ffn_flush_read(st, block->post_per_layer_norm_weight);
    if (s != GEIST_OK) { return s; }

    const uint32_t set_count = use_rmsnorm_add ? 4u : 5u;
    const uint32_t descriptor_count = use_rmsnorm_add ? 13u : 15u;
    s = vulkan_ensure_block_descriptor_pool(be, set_count, descriptor_count);
    if (s != GEIST_OK) { return s; }
    VkDescriptorSetLayout layouts[5] = {
        vulkan_ffn_matvec_set_layout(st, block->per_layer_gate_weight),
        st->gelu_tanh_mul_set_layout,
        vulkan_ffn_matvec_set_layout(st, block->per_layer_proj_weight),
        use_rmsnorm_add ? st->rmsnorm_add_set_layout : st->rmsnorm_set_layout,
        st->add_set_layout,
    };
    VkDescriptorSet sets[5] = {VK_NULL_HANDLE};
    bool sets_reused[5] = {false};
    s = vulkan_alloc_block_descriptor_sets_cached(
        be, "ple_block", set_count, layouts, sets, sets_reused);
    if (s != GEIST_OK) { return s; }

    if (!sets_reused[0]) {
        vulkan_ffn_write_set(st, sets[0], block->hidden,
                             block->per_layer_gate_weight,
                             block->gate_scratch);
    }
    if (!sets_reused[1]) {
        vulkan_ffn_write_set(st, sets[1], block->gate_scratch,
                             block->per_layer_input, block->gate_scratch);
    }
    if (!sets_reused[2]) {
        vulkan_ffn_write_set(st, sets[2], block->gate_scratch,
                             block->per_layer_proj_weight,
                             block->proj_scratch);
    }
    if (use_rmsnorm_add) {
        if (!sets_reused[3]) {
            vulkan_write_rmsnorm_add_set(st, sets[3], block->proj_scratch,
                                         block->post_per_layer_norm_weight,
                                         block->hidden, block->out);
        }
    } else {
        if (!sets_reused[3]) {
            vulkan_ffn_write_set(st, sets[3], block->proj_scratch,
                                 block->post_per_layer_norm_weight,
                                 block->proj_scratch);
        }
        if (!sets_reused[4]) {
            vulkan_ffn_write_set(st, sets[4], block->hidden,
                                 block->proj_scratch, block->out);
        }
    }

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "ple_block", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_block_descriptor_sets(st, set_count, sets);
        return s;
    }

    vulkan_ffn_pre_barrier(st, cmd);
    vulkan_ffn_cmd_matmul_or_matvec(st, cmd, sets[0], block->hidden,
                                    block->per_layer_gate_weight,
                                    block->gate_scratch, rows, &gate_dims);
    vulkan_ffn_shader_barrier(st, cmd);

    const uint32_t gelu_push[4] = {
        (uint32_t) gate_elems,
        (uint32_t) (block->gate_scratch->offset / sizeof(float)),
        (uint32_t) (block->per_layer_input->offset / sizeof(float)),
        (uint32_t) (block->gate_scratch->offset / sizeof(float)),
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->gelu_tanh_mul_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->gelu_tanh_mul_pipeline_layout, 0, 1,
                                &sets[1], 0, nullptr);
    st->vkCmdPushConstants(cmd, st->gelu_tanh_mul_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(gelu_push), gelu_push);
    st->vkCmdDispatch(cmd, (uint32_t) ((gate_elems + 255u) / 256u), 1, 1);
    vulkan_ffn_shader_barrier(st, cmd);

    vulkan_ffn_cmd_matmul_or_matvec(st, cmd, sets[2], block->gate_scratch,
                                    block->per_layer_proj_weight,
                                    block->proj_scratch, rows, &proj_dims);
    vulkan_ffn_shader_barrier(st, cmd);

    if (use_rmsnorm_add) {
        vulkan_cmd_rmsnorm_add(st, cmd, sets[3], block->proj_scratch,
                               block->post_per_layer_norm_weight,
                               block->hidden, block->out, rows, d,
                               block->eps);
    } else {
        vulkan_attn_cmd_rmsnorm(st, cmd, sets[3], block->proj_scratch,
                                block->post_per_layer_norm_weight,
                                block->proj_scratch, rows, d, block->eps);
        vulkan_ffn_shader_barrier(st, cmd);
        const uint32_t add_push[4] = {
            (uint32_t) hidden_elems,
            (uint32_t) (block->hidden->offset / sizeof(float)),
            (uint32_t) (block->proj_scratch->offset / sizeof(float)),
            (uint32_t) (block->out->offset / sizeof(float)),
        };
        st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              st->add_pipeline);
        st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    st->add_pipeline_layout, 0, 1, &sets[4],
                                    0, nullptr);
        st->vkCmdPushConstants(cmd, st->add_pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(add_push), add_push);
        st->vkCmdDispatch(cmd, (uint32_t) ((hidden_elems + 255u) / 256u), 1,
                          1);
    }
    vulkan_ffn_post_barrier(st, cmd);

    s = vulkan_submit_transient_command_buffer(be, cmd, "ple_block");
    vulkan_free_block_descriptor_sets(st, set_count, sets);
    if (s != GEIST_OK) {
        return s;
    }
    if (block->out->buffer->host_visible) {
        return vulkan_invalidate_if_needed(st, block->out->buffer);
    }
    return GEIST_OK;
}

static void vulkan_attn_cmd_rope(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *cos,
    const struct geist_tensor *sin,
    size_t seq,
    size_t heads,
    size_t head_dim,
    size_t rope_row_offset,
    bool use_dynamic_decode_params,
    bool use_decode_position) {

    const size_t total_pairs = seq * heads * (head_dim / 2u);
    const uint32_t push[7] = {
        (uint32_t) total_pairs,
        (uint32_t) heads,
        (uint32_t) head_dim,
        (uint32_t) (x->offset / sizeof(float)),
        (uint32_t) (cos->offset / sizeof(float) +
                    rope_row_offset * head_dim),
        (uint32_t) (sin->offset / sizeof(float) +
                    rope_row_offset * head_dim),
        use_decode_position ? 1u : 0u,
    };
    VkPipeline pipeline = use_dynamic_decode_params
                              ? st->rope_dyn_pipeline
                              : st->rope_pipeline;
    VkPipelineLayout pipeline_layout =
        use_dynamic_decode_params
            ? st->rope_dyn_pipeline_layout
            : st->rope_pipeline_layout;
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout, 0, 1, &set, 0, nullptr);
    st->vkCmdPushConstants(cmd, pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           (use_dynamic_decode_params ? 7u : 6u) *
                               sizeof(uint32_t),
                           push);
    st->vkCmdDispatch(cmd, (uint32_t) ((total_pairs + 255u) / 256u),
                      1, 1);
}

static void vulkan_attn_cmd_rmsnorm_rope_dyn(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_tensor *x,
    const struct geist_tensor *w,
    const struct geist_tensor *cos,
    const struct geist_tensor *sin,
    size_t seq,
    size_t heads,
    size_t head_dim,
    bool use_decode_position,
    float eps) {

    struct rmsnorm_rope_push {
        uint32_t rows;
        uint32_t heads;
        uint32_t head_dim;
        uint32_t x_offset;
        uint32_t w_offset;
        uint32_t cos_offset;
        uint32_t sin_offset;
        uint32_t use_decode_position;
        float eps;
    };
    const struct rmsnorm_rope_push push = {
        .rows = (uint32_t) seq,
        .heads = (uint32_t) heads,
        .head_dim = (uint32_t) head_dim,
        .x_offset = (uint32_t) (x->offset / sizeof(float)),
        .w_offset = (uint32_t) (w->offset / sizeof(float)),
        .cos_offset = (uint32_t) (cos->offset / sizeof(float)),
        .sin_offset = (uint32_t) (sin->offset / sizeof(float)),
        .use_decode_position = use_decode_position ? 1u : 0u,
        .eps = eps,
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->rmsnorm_rope_dyn_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->rmsnorm_rope_dyn_pipeline_layout, 0, 1,
                                &set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->rmsnorm_rope_dyn_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                           &push);
    st->vkCmdDispatch(cmd, (uint32_t) (seq * heads), 1, 1);
}

static void vulkan_attn_cmd_attention(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_backend_attention_block *block,
    bool use_dynamic_decode_params) {

    const uint32_t push[11] = {
        (uint32_t) block->residual->shape[0],
        (uint32_t) block->kv_len,
        (uint32_t) block->q_heads,
        (uint32_t) block->kv_heads,
        (uint32_t) block->head_dim,
        (uint32_t) block->q_position,
        (uint32_t) block->sliding_window,
        (uint32_t) (block->q_scratch->offset / sizeof(float)),
        (uint32_t) (block->k_cache->offset / sizeof(float)),
        (uint32_t) (block->v_cache->offset / sizeof(float)),
        (uint32_t) (block->attn_scratch->offset / sizeof(float)),
    };
    VkPipeline pipeline = use_dynamic_decode_params
                              ? st->attention_dyn_pipeline
                              : st->attention_pipeline;
    VkPipelineLayout pipeline_layout =
        use_dynamic_decode_params
            ? st->attention_dyn_pipeline_layout
            : st->attention_pipeline_layout;
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout, 0, 1, &set, 0, nullptr);
    st->vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), push);
    st->vkCmdDispatch(cmd, (uint32_t) block->residual->shape[0],
                      (uint32_t) block->q_heads, 1);
}

static void vulkan_attn_query_cmd_attention(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_backend_attention_query_block *block,
    bool use_dynamic_decode_params) {

    const uint32_t push[11] = {
        (uint32_t) block->residual->shape[0],
        (uint32_t) block->kv_len,
        (uint32_t) block->q_heads,
        (uint32_t) block->kv_heads,
        (uint32_t) block->head_dim,
        (uint32_t) block->q_position,
        (uint32_t) block->sliding_window,
        (uint32_t) (block->q_scratch->offset / sizeof(float)),
        (uint32_t) (block->k_cache->offset / sizeof(float)),
        (uint32_t) (block->v_cache->offset / sizeof(float)),
        (uint32_t) (block->attn_scratch->offset / sizeof(float)),
    };
    VkPipeline pipeline = use_dynamic_decode_params
                              ? st->attention_dyn_pipeline
                              : st->attention_pipeline;
    VkPipelineLayout pipeline_layout =
        use_dynamic_decode_params
            ? st->attention_dyn_pipeline_layout
            : st->attention_pipeline_layout;
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout, 0, 1, &set, 0, nullptr);
    st->vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), push);
    st->vkCmdDispatch(cmd, (uint32_t) block->residual->shape[0],
                      (uint32_t) block->q_heads, 1);
}

static void vulkan_attn_cmd_kv_append_dyn(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_backend_attention_block *block,
    size_t rows,
    size_t kv_out) {

    const uint32_t push[6] = {
        (uint32_t) (rows * kv_out),
        (uint32_t) kv_out,
        (uint32_t) (block->k_scratch->offset / sizeof(float)),
        (uint32_t) (block->v_scratch->offset / sizeof(float)),
        (uint32_t) (block->k_cache->offset / sizeof(float)),
        (uint32_t) (block->v_cache->offset / sizeof(float)),
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->kv_append_dyn_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->kv_append_dyn_pipeline_layout, 0, 1,
                                &set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->kv_append_dyn_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), push);
    st->vkCmdDispatch(cmd, (uint32_t) (((rows * kv_out) + 255u) / 256u),
                      1, 1);
}

static void vulkan_attn_cmd_add(
    struct vulkan_state *st,
    VkCommandBuffer cmd,
    VkDescriptorSet set,
    const struct geist_tensor *a,
    const struct geist_tensor *b,
    const struct geist_tensor *y,
    size_t n) {

    const uint32_t push[4] = {
        (uint32_t) n,
        (uint32_t) (a->offset / sizeof(float)),
        (uint32_t) (b->offset / sizeof(float)),
        (uint32_t) (y->offset / sizeof(float)),
    };
    st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                          st->add_pipeline);
    st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                st->add_pipeline_layout, 0, 1,
                                &set, 0, nullptr);
    st->vkCmdPushConstants(cmd, st->add_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(push), push);
    st->vkCmdDispatch(cmd, (uint32_t) ((n + 255u) / 256u), 1, 1);
}

[[nodiscard]] static enum geist_status vulkan_greedy_head(
    struct geist_backend *be,
    const struct geist_backend_greedy_head *head,
    geist_token_t *out_token) {

    if (be == nullptr || be->state == nullptr || head == nullptr ||
        out_token == nullptr ||
        head->struct_size < sizeof(*head) ||
        head->hidden == nullptr || head->norm_weight == nullptr ||
        head->lm_head_weight == nullptr || head->normed_scratch == nullptr ||
        head->logits == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (head->d_model == 0 || head->vocab_size == 0 ||
        head->d_model > UINT32_MAX || head->vocab_size > UINT32_MAX ||
        head->vocab_size > (size_t) INT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }

    size_t hidden_n = 0;
    size_t norm_n = 0;
    size_t normed_n = 0;
    size_t logits_n = 0;
    if (!vulkan_tensor_f32_dense_count(head->hidden, &hidden_n) ||
        !vulkan_tensor_f32_dense_count(head->norm_weight, &norm_n) ||
        !vulkan_tensor_f32_dense_count(head->normed_scratch, &normed_n) ||
        !vulkan_tensor_f32_dense_count(head->logits, &logits_n) ||
        hidden_n != head->d_model ||
        norm_n != head->d_model ||
        normed_n != head->d_model ||
        logits_n != head->vocab_size ||
        head->hidden->ndim != 1 ||
        head->norm_weight->ndim != 1 ||
        head->normed_scratch->ndim != 1 ||
        !((head->logits->ndim == 1) ||
          (head->logits->ndim == 2 && head->logits->shape[0] == 1))) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "vulkan greedy_head: expected F32 DENSE hidden/norm/logits");
        return GEIST_E_UNSUPPORTED;
    }

    struct vulkan_ffn_matvec_dims matvec_dims;
    enum geist_status s = vulkan_ffn_matvec_dims(
        be, head->lm_head_weight, head->vocab_size, head->d_model,
        &matvec_dims);
    if (s != GEIST_OK) {
        return s;
    }

    const struct geist_tensor *offset_tensors[] = {
        head->hidden, head->norm_weight, head->normed_scratch, head->logits,
    };
    for (size_t i = 0; i < sizeof(offset_tensors) / sizeof(offset_tensors[0]);
         i++) {
        if ((offset_tensors[i]->offset % sizeof(float)) != 0 ||
            offset_tensors[i]->offset / sizeof(float) > UINT32_MAX) {
            return GEIST_E_INVALID_ARG;
        }
    }

    const struct geist_tensor *owner_tensors[] = {
        head->hidden, head->norm_weight, head->lm_head_weight,
        head->normed_scratch, head->logits,
    };
    for (size_t i = 0; i < sizeof(owner_tensors) / sizeof(owner_tensors[0]);
         i++) {
        if (!vulkan_ffn_owner_ok(be, owner_tensors[i])) {
            return GEIST_E_INVALID_ARG;
        }
    }

    const size_t d_bytes = head->d_model * sizeof(float);
    const size_t logits_bytes = head->vocab_size * sizeof(float);
    if (!vulkan_ffn_range_ok(head->hidden, d_bytes) ||
        !vulkan_ffn_range_ok(head->norm_weight, d_bytes) ||
        !vulkan_ffn_range_ok(head->lm_head_weight, matvec_dims.weight_bytes) ||
        !vulkan_ffn_range_ok(head->normed_scratch, d_bytes) ||
        !vulkan_ffn_range_ok(head->logits, logits_bytes)) {
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    s = vulkan_ensure_rmsnorm_pipeline(be);
    if (s != GEIST_OK) { return s; }
    if (head->lm_head_weight->dtype == GEIST_DTYPE_F32) {
        s = vulkan_ensure_matvec_pipeline(be);
    } else if (head->lm_head_weight->dtype == GEIST_DTYPE_F16 ||
               head->lm_head_weight->dtype == GEIST_DTYPE_BF16) {
        s = vulkan_ensure_matvec_f16bf16_pipeline(be);
    } else if (head->lm_head_weight->dtype == GEIST_DTYPE_Q4_K) {
        s = vulkan_ensure_matvec_q4k_pipeline(be);
        if (s == GEIST_OK) {
            s = vulkan_ensure_q4k_nt4_subgroup_if_enabled(be);
        }
    } else if (head->lm_head_weight->dtype == GEIST_DTYPE_Q6_K) {
        s = vulkan_ensure_matvec_q6k_pipeline(be);
    } else {
        s = GEIST_E_UNSUPPORTED;
    }
    if (s != GEIST_OK) { return s; }
    if (head->token_output_offset >= UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }
    s = vulkan_ensure_argmax_resources(be, head->token_output_offset + 1);
    if (s != GEIST_OK) { return s; }

    const struct geist_tensor *read_tensors[] = {
        head->hidden, head->norm_weight, head->lm_head_weight,
    };
    for (size_t i = 0; i < sizeof(read_tensors) / sizeof(read_tensors[0]);
         i++) {
        s = vulkan_ffn_flush_read(st, read_tensors[i]);
        if (s != GEIST_OK) { return s; }
    }

    struct geist_buffer *argmax_out = st->argmax_result_buffer;
    uint32_t zero = 0;
    memcpy((uint8_t *) argmax_out->mapped +
               head->token_output_offset * sizeof(uint32_t),
           &zero, sizeof(zero));
    s = vulkan_flush_if_needed(st, argmax_out);
    if (s != GEIST_OK) { return s; }

    constexpr uint32_t set_count = 3u;
    s = vulkan_ensure_block_descriptor_pool(be, set_count, 8u);
    if (s != GEIST_OK) { return s; }

    const VkDescriptorSetLayout layouts[3] = {
        st->rmsnorm_set_layout,
        vulkan_ffn_matvec_set_layout(st, head->lm_head_weight),
        st->argmax_set_layout,
    };
    VkDescriptorSet sets[3] = {VK_NULL_HANDLE};
    bool sets_reused[3] = {false};
    s = vulkan_alloc_block_descriptor_sets_cached(
        be, "greedy_head", set_count, layouts, sets, sets_reused);
    if (s != GEIST_OK) { return s; }

    if (!sets_reused[0]) {
        vulkan_ffn_write_set(st, sets[0], head->hidden, head->norm_weight,
                             head->normed_scratch);
    }
    if (!sets_reused[1]) {
        vulkan_ffn_write_set(st, sets[1], head->normed_scratch,
                             head->lm_head_weight, head->logits);
    }
    if (!sets_reused[2]) {
        const VkDescriptorBufferInfo infos[2] = {
            {.buffer = head->logits->buffer->buffer,
             .offset = 0,
             .range = VK_WHOLE_SIZE},
            {.buffer = argmax_out->buffer,
             .offset = 0,
             .range = VK_WHOLE_SIZE},
        };
        const VkWriteDescriptorSet writes[2] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = sets[2],
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &infos[0],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = sets[2],
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &infos[1],
            },
        };
        st->vkUpdateDescriptorSets(st->device, 2, writes, 0, nullptr);
    }

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "greedy_head", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_block_descriptor_sets(st, set_count, sets);
        return s;
    }

    vulkan_ffn_pre_barrier(st, cmd);
    vulkan_attn_cmd_rmsnorm(st, cmd, sets[0], head->hidden,
                            head->norm_weight, head->normed_scratch,
                            1u, head->d_model, head->eps);
    vulkan_ffn_shader_barrier(st, cmd);
    vulkan_ffn_cmd_matvec(st, cmd, sets[1], head->normed_scratch,
                          head->lm_head_weight, head->logits,
                          &matvec_dims);
    vulkan_ffn_shader_barrier(st, cmd);
    {
        const uint32_t push[3] = {
            (uint32_t) head->vocab_size,
            (uint32_t) (head->logits->offset / sizeof(float)),
            (uint32_t) head->token_output_offset,
        };
        st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              st->argmax_pipeline);
        st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    st->argmax_pipeline_layout, 0, 1,
                                    &sets[2], 0, nullptr);
        st->vkCmdPushConstants(cmd, st->argmax_pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(push), push);
        st->vkCmdDispatch(cmd, 1, 1, 1);
    }
    vulkan_ffn_post_barrier(st, cmd);

    s = vulkan_submit_transient_command_buffer(be, cmd, "greedy_head");
    vulkan_free_block_descriptor_sets(st, set_count, sets);
    if (s != GEIST_OK) {
        return s;
    }
    if (st->capture_active) {
        st->captured_greedy_token_pending = true;
        st->captured_greedy_vocab_size = (uint32_t) head->vocab_size;
        if (head->token_output_offset + 1 >
            st->captured_greedy_token_count) {
            st->captured_greedy_token_count =
                (uint32_t) (head->token_output_offset + 1);
        }
        *out_token = -1;
        return GEIST_OK;
    }

    s = vulkan_invalidate_if_needed(st, argmax_out);
    if (s != GEIST_OK) {
        return s;
    }
    uint32_t token_u32 = 0;
    memcpy(&token_u32,
           (const uint8_t *) argmax_out->mapped +
               head->token_output_offset * sizeof(uint32_t),
           sizeof(token_u32));
    if (token_u32 >= head->vocab_size ||
        token_u32 > (uint32_t) INT32_MAX) {
        geist_backend_set_error(be, GEIST_E_BACKEND,
                                "vulkan greedy_head: invalid token id %u",
                                token_u32);
        return GEIST_E_BACKEND;
    }
    *out_token = (geist_token_t) token_u32;
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_greedy_head_batch(
    struct geist_backend *be,
    const struct geist_backend_greedy_head_batch *head,
    geist_token_t out_tokens[static head->row_count]) {

    if (be == nullptr || be->state == nullptr || head == nullptr ||
        out_tokens == nullptr ||
        head->struct_size < sizeof(*head) ||
        head->hidden == nullptr || head->norm_weight == nullptr ||
        head->lm_head_weight == nullptr || head->normed_scratch == nullptr ||
        head->logits == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (head->d_model == 0 || head->vocab_size == 0 ||
        head->row_count == 0 ||
        head->d_model > UINT32_MAX || head->vocab_size > UINT32_MAX ||
        head->row_count > UINT32_MAX ||
        head->vocab_size > (size_t) INT32_MAX ||
        head->token_output_offset > UINT32_MAX ||
        head->row_count >
            (size_t) UINT32_MAX - head->token_output_offset) {
        return GEIST_E_INVALID_ARG;
    }

    size_t hidden_n = 0;
    size_t norm_n = 0;
    size_t normed_n = 0;
    size_t logits_n = 0;
    if (head->row_count > SIZE_MAX / head->d_model ||
        head->row_count > SIZE_MAX / head->vocab_size) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t hidden_elems = head->row_count * head->d_model;
    const size_t logits_elems = head->row_count * head->vocab_size;
    if (!vulkan_tensor_f32_dense_count(head->hidden, &hidden_n) ||
        !vulkan_tensor_f32_dense_count(head->norm_weight, &norm_n) ||
        !vulkan_tensor_f32_dense_count(head->normed_scratch, &normed_n) ||
        !vulkan_tensor_f32_dense_count(head->logits, &logits_n) ||
        hidden_n != hidden_elems ||
        norm_n != head->d_model ||
        normed_n != hidden_elems ||
        logits_n != logits_elems ||
        head->hidden->ndim != 2 ||
        head->hidden->shape[0] != (int64_t) head->row_count ||
        head->hidden->shape[1] != (int64_t) head->d_model ||
        head->norm_weight->ndim != 1 ||
        head->normed_scratch->ndim != 2 ||
        head->normed_scratch->shape[0] != (int64_t) head->row_count ||
        head->normed_scratch->shape[1] != (int64_t) head->d_model ||
        head->logits->ndim != 2 ||
        head->logits->shape[0] != (int64_t) head->row_count ||
        head->logits->shape[1] != (int64_t) head->vocab_size) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "vulkan greedy_head_batch: expected F32 DENSE batch tensors");
        return GEIST_E_UNSUPPORTED;
    }

    struct vulkan_ffn_matvec_dims matvec_dims;
    enum geist_status s = vulkan_ffn_matvec_dims(
        be, head->lm_head_weight, head->vocab_size, head->d_model,
        &matvec_dims);
    if (s != GEIST_OK) {
        return s;
    }

    const struct geist_tensor *f32_tensors[] = {
        head->hidden, head->norm_weight, head->normed_scratch, head->logits,
    };
    const size_t f32_counts[] = {
        hidden_elems, head->d_model, hidden_elems, logits_elems,
    };
    const size_t u32_elem_limit = (size_t) UINT32_MAX + 1u;
    for (size_t i = 0; i < sizeof(f32_tensors) / sizeof(f32_tensors[0]);
         i++) {
        if ((f32_tensors[i]->offset % sizeof(float)) != 0) {
            return GEIST_E_INVALID_ARG;
        }
        const size_t elem_offset = f32_tensors[i]->offset / sizeof(float);
        if (elem_offset >= u32_elem_limit ||
            f32_counts[i] > u32_elem_limit - elem_offset) {
            return GEIST_E_INVALID_ARG;
        }
    }
    if (head->lm_head_weight->dtype == GEIST_DTYPE_F32) {
        if (head->lm_head_weight->offset % sizeof(float) != 0) {
            return GEIST_E_INVALID_ARG;
        }
        const size_t weight_offset =
            head->lm_head_weight->offset / sizeof(float);
        const size_t weight_elems = head->vocab_size * head->d_model;
        if (weight_offset >= u32_elem_limit ||
            weight_elems > u32_elem_limit - weight_offset) {
            return GEIST_E_INVALID_ARG;
        }
    }

    const struct geist_tensor *owner_tensors[] = {
        head->hidden, head->norm_weight, head->lm_head_weight,
        head->normed_scratch, head->logits,
    };
    for (size_t i = 0; i < sizeof(owner_tensors) / sizeof(owner_tensors[0]);
         i++) {
        if (!vulkan_ffn_owner_ok(be, owner_tensors[i])) {
            return GEIST_E_INVALID_ARG;
        }
    }

    const size_t hidden_bytes = hidden_elems * sizeof(float);
    const size_t norm_bytes = head->d_model * sizeof(float);
    const size_t logits_bytes = logits_elems * sizeof(float);
    if (!vulkan_ffn_range_ok(head->hidden, hidden_bytes) ||
        !vulkan_ffn_range_ok(head->norm_weight, norm_bytes) ||
        !vulkan_ffn_range_ok(head->lm_head_weight, matvec_dims.weight_bytes) ||
        !vulkan_ffn_range_ok(head->normed_scratch, hidden_bytes) ||
        !vulkan_ffn_range_ok(head->logits, logits_bytes)) {
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    s = vulkan_ensure_rmsnorm_pipeline(be);
    if (s != GEIST_OK) { return s; }
    if (head->lm_head_weight->dtype == GEIST_DTYPE_F32) {
        s = vulkan_ensure_matmul_f32_pipeline(be);
    } else if (head->lm_head_weight->dtype == GEIST_DTYPE_F16 ||
               head->lm_head_weight->dtype == GEIST_DTYPE_BF16) {
        s = vulkan_ensure_matvec_f16bf16_pipeline(be);
    } else if (head->lm_head_weight->dtype == GEIST_DTYPE_Q4_K) {
        s = head->row_count > 1u ? vulkan_ensure_matmul_q4k_m8_pipeline(be)
                                 : vulkan_ensure_matmul_q4k_pipeline(be);
    } else if (head->lm_head_weight->dtype == GEIST_DTYPE_Q6_K) {
        s = vulkan_ensure_matmul_q6k_pipeline(be);
    } else {
        s = GEIST_E_UNSUPPORTED;
    }
    if (s != GEIST_OK) { return s; }
    s = vulkan_ensure_argmax_batch_pipeline(be);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ensure_argmax_resources(
        be, head->token_output_offset + head->row_count);
    if (s != GEIST_OK) { return s; }

    const struct geist_tensor *read_tensors[] = {
        head->hidden, head->norm_weight, head->lm_head_weight,
    };
    for (size_t i = 0; i < sizeof(read_tensors) / sizeof(read_tensors[0]);
         i++) {
        s = vulkan_ffn_flush_read(st, read_tensors[i]);
        if (s != GEIST_OK) { return s; }
    }

    struct geist_buffer *argmax_out = st->argmax_result_buffer;
    memset((uint8_t *) argmax_out->mapped +
               head->token_output_offset * sizeof(uint32_t),
           0, head->row_count * sizeof(uint32_t));
    s = vulkan_flush_if_needed(st, argmax_out);
    if (s != GEIST_OK) { return s; }

    constexpr uint32_t set_count = 3u;
    s = vulkan_ensure_block_descriptor_pool(be, set_count, 8u);
    if (s != GEIST_OK) { return s; }

    const VkDescriptorSetLayout layouts[3] = {
        st->rmsnorm_set_layout,
        vulkan_ffn_matvec_set_layout(st, head->lm_head_weight),
        st->argmax_set_layout,
    };
    VkDescriptorSet sets[3] = {VK_NULL_HANDLE};
    const VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = st->block_descriptor_pool,
        .descriptorSetCount = set_count,
        .pSetLayouts = layouts,
    };
    VkResult rc = st->vkAllocateDescriptorSets(st->device, &dsai, sets);
    if (rc != VK_SUCCESS) {
        geist_backend_set_error(
            be, GEIST_E_BACKEND,
            "vulkan greedy_head_batch: vkAllocateDescriptorSets failed (%d)",
            (int) rc);
        return GEIST_E_BACKEND;
    }

    vulkan_ffn_write_set(st, sets[0], head->hidden, head->norm_weight,
                         head->normed_scratch);
    vulkan_ffn_write_set(st, sets[1], head->normed_scratch,
                         head->lm_head_weight, head->logits);
    {
        const VkDescriptorBufferInfo infos[2] = {
            {.buffer = head->logits->buffer->buffer,
             .offset = 0,
             .range = VK_WHOLE_SIZE},
            {.buffer = argmax_out->buffer,
             .offset = 0,
             .range = VK_WHOLE_SIZE},
        };
        const VkWriteDescriptorSet writes[2] = {
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = sets[2],
                .dstBinding = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &infos[0],
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = sets[2],
                .dstBinding = 1,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &infos[1],
            },
        };
        st->vkUpdateDescriptorSets(st->device, 2, writes, 0, nullptr);
    }

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "greedy_head_batch", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_block_descriptor_sets(st, set_count, sets);
        return s;
    }

    vulkan_ffn_pre_barrier(st, cmd);
    vulkan_attn_cmd_rmsnorm(st, cmd, sets[0], head->hidden,
                            head->norm_weight, head->normed_scratch,
                            head->row_count, head->d_model, head->eps);
    vulkan_ffn_shader_barrier(st, cmd);
    if (head->lm_head_weight->dtype == GEIST_DTYPE_F32) {
        vulkan_ffn_cmd_matmul_f32(st, cmd, sets[1],
                                  head->normed_scratch,
                                  head->lm_head_weight,
                                  head->logits,
                                  head->row_count,
                                  head->d_model,
                                  head->vocab_size);
        vulkan_ffn_shader_barrier(st, cmd);
    } else if (head->lm_head_weight->dtype == GEIST_DTYPE_Q6_K) {
        vulkan_ffn_cmd_matmul_q6k(st, cmd, sets[1],
                                  head->normed_scratch,
                                  head->lm_head_weight,
                                  head->logits,
                                  head->row_count,
                                  &matvec_dims);
        vulkan_ffn_shader_barrier(st, cmd);
    } else if (head->lm_head_weight->dtype == GEIST_DTYPE_Q4_K) {
        vulkan_ffn_cmd_matmul_q4k(st, cmd, sets[1],
                                  head->normed_scratch,
                                  head->lm_head_weight,
                                  head->logits,
                                  head->row_count,
                                  &matvec_dims);
        vulkan_ffn_shader_barrier(st, cmd);
    }
    if (head->lm_head_weight->dtype != GEIST_DTYPE_F32 &&
        head->lm_head_weight->dtype != GEIST_DTYPE_Q4_K &&
        head->lm_head_weight->dtype != GEIST_DTYPE_Q6_K) {
        for (size_t row = 0; row < head->row_count; row++) {
            struct geist_tensor logits_row = *head->logits;
            logits_row.ndim = 1;
            logits_row.shape[0] = (int64_t) head->vocab_size;
            logits_row.shape[1] = 0;
            logits_row.stride[0] = 1;
            logits_row.stride[1] = 0;
            logits_row.offset += row * head->vocab_size * sizeof(float);

            struct geist_tensor normed_row = *head->normed_scratch;
            normed_row.ndim = 1;
            normed_row.shape[0] = (int64_t) head->d_model;
            normed_row.shape[1] = 0;
            normed_row.stride[0] = 1;
            normed_row.stride[1] = 0;
            normed_row.offset += row * head->d_model * sizeof(float);

            vulkan_ffn_cmd_matvec(st, cmd, sets[1], &normed_row,
                                  head->lm_head_weight, &logits_row,
                                  &matvec_dims);
            vulkan_ffn_shader_barrier(st, cmd);
        }
    }
    vulkan_cmd_argmax_batch(st, cmd, sets[2], head->logits,
                            head->row_count, head->vocab_size,
                            head->token_output_offset);
    vulkan_ffn_post_barrier(st, cmd);

    s = vulkan_submit_transient_command_buffer(be, cmd, "greedy_head_batch");
    vulkan_free_block_descriptor_sets(st, set_count, sets);
    if (s != GEIST_OK) {
        return s;
    }
    if (st->capture_active) {
        st->captured_greedy_token_pending = true;
        st->captured_greedy_vocab_size = (uint32_t) head->vocab_size;
        st->captured_greedy_token_count =
            (uint32_t) (head->token_output_offset + head->row_count);
        for (size_t row = 0; row < head->row_count; row++) {
            out_tokens[row] = -1;
        }
        return GEIST_OK;
    }

    s = vulkan_invalidate_if_needed(st, argmax_out);
    if (s != GEIST_OK) {
        return s;
    }
    for (size_t row = 0; row < head->row_count; row++) {
        uint32_t token_u32 = 0;
        memcpy(&token_u32,
               (const uint8_t *) argmax_out->mapped +
                   (head->token_output_offset + row) * sizeof(uint32_t),
               sizeof(token_u32));
        if (token_u32 >= head->vocab_size ||
            token_u32 > (uint32_t) INT32_MAX) {
            geist_backend_set_error(
                be, GEIST_E_BACKEND,
                "vulkan greedy_head_batch: invalid token id %u",
                token_u32);
            return GEIST_E_BACKEND;
        }
        out_tokens[row] = (geist_token_t) token_u32;
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_attention_block_recorded(
    struct geist_backend *be,
    const struct geist_backend_attention_block *block) {

    struct vulkan_state *st = be->state;
    const bool use_post_rmsnorm_add =
        block->post_attn_norm_weight != nullptr &&
        vulkan_block_rmsnorm_add_enabled();
    const bool use_dynamic_decode_params =
        st->capture_active &&
        st->capture_kind == GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP &&
        vulkan_dynamic_decode_params_enabled(st);
    enum geist_status s = vulkan_ensure_rmsnorm_pipeline(be);
    if (s != GEIST_OK) { return s; }
    s = use_dynamic_decode_params
            ? vulkan_ensure_rope_dyn_pipeline(be)
            : vulkan_ensure_rope_pipeline(be);
    if (s != GEIST_OK) { return s; }
    s = use_dynamic_decode_params
            ? vulkan_ensure_attention_dyn_pipeline(be)
            : vulkan_ensure_attention_pipeline(be);
    if (s != GEIST_OK) { return s; }
    if (use_dynamic_decode_params) {
        s = vulkan_ensure_kv_append_dyn_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }
    if (use_post_rmsnorm_add) {
        s = vulkan_ensure_rmsnorm_add_pipeline(be);
        if (s != GEIST_OK) { return s; }
    } else {
        s = vulkan_ensure_add_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }

    const struct geist_tensor *weights[4] = {
        block->q_proj_weight,
        block->k_proj_weight,
        block->v_proj_weight,
        block->o_proj_weight,
    };
    bool need_f32 = false;
    bool need_f16bf16 = false;
    bool need_q4 = false;
    bool need_q6 = false;
    for (size_t i = 0; i < 4; i++) {
        if (weights[i]->dtype == GEIST_DTYPE_F32) { need_f32 = true; }
        if (weights[i]->dtype == GEIST_DTYPE_F16 ||
            weights[i]->dtype == GEIST_DTYPE_BF16) {
            need_f16bf16 = true;
        }
        if (weights[i]->dtype == GEIST_DTYPE_Q4_K) { need_q4 = true; }
        if (weights[i]->dtype == GEIST_DTYPE_Q6_K) { need_q6 = true; }
    }
    if (need_f32) {
        s = vulkan_ensure_matvec_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }
    if (need_f16bf16) {
        s = vulkan_ensure_matvec_f16bf16_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }
    if (need_q4) {
        s = vulkan_ensure_matvec_q4k_pipeline(be);
        if (s != GEIST_OK) { return s; }
        s = vulkan_ensure_q4k_nt4_subgroup_if_enabled(be);
        if (s != GEIST_OK) { return s; }
    }
    if (need_q6) {
        s = vulkan_ensure_matvec_q6k_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }

    const size_t rows = (size_t) block->residual->shape[0];
    if (rows > 1u) {
        if (need_f32) {
            s = vulkan_ensure_matmul_f32_pipeline(be);
            if (s != GEIST_OK) { return s; }
        }
        if (need_q4) {
            s = vulkan_ensure_matmul_q4k_m8_pipeline(be);
            if (s != GEIST_OK) { return s; }
        }
        if (need_q6) {
            s = vulkan_ensure_matmul_q6k_pipeline(be);
            if (s != GEIST_OK) { return s; }
        }
    }
    const size_t d = block->d_model;
    if (block->q_heads > SIZE_MAX / block->head_dim ||
        block->kv_heads > SIZE_MAX / block->head_dim) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t q_out = block->q_heads * block->head_dim;
    const size_t kv_out = block->kv_heads * block->head_dim;
    struct geist_tensor normed_1d =
        vulkan_ffn_1d_view(block->normed_scratch, d);
    struct geist_tensor q_1d =
        vulkan_ffn_1d_view(block->q_scratch, q_out);
    struct geist_tensor k_1d =
        vulkan_ffn_1d_view(block->k_scratch, kv_out);
    struct geist_tensor v_1d =
        vulkan_ffn_1d_view(block->v_scratch, kv_out);
    struct geist_tensor attn_1d =
        vulkan_ffn_1d_view(block->attn_scratch, q_out);
    struct geist_tensor o_1d =
        vulkan_ffn_1d_view(block->o_scratch, d);

    struct vulkan_ffn_matvec_dims q_dims;
    struct vulkan_ffn_matvec_dims k_dims;
    struct vulkan_ffn_matvec_dims v_dims;
    struct vulkan_ffn_matvec_dims o_dims;
    s = vulkan_ffn_matvec_dims(be, block->q_proj_weight, q_out, d, &q_dims);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ffn_matvec_dims(be, block->k_proj_weight, kv_out, d, &k_dims);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ffn_matvec_dims(be, block->v_proj_weight, kv_out, d, &v_dims);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ffn_matvec_dims(be, block->o_proj_weight, d, q_out, &o_dims);
    if (s != GEIST_OK) { return s; }

    const bool use_kv_pair =
        vulkan_q4k_nt4_pair_enabled() &&
        rows == 1u &&
        block->k_proj_weight->dtype == GEIST_DTYPE_Q4_K &&
        block->v_proj_weight->dtype == GEIST_DTYPE_Q4_K &&
        k_dims.n_in == v_dims.n_in &&
        k_dims.n_out == v_dims.n_out &&
        k_dims.blocks_per_row == v_dims.blocks_per_row &&
        vulkan_q4k_nt4_cache_find(st, block->k_proj_weight, k_dims.n_in,
                                  k_dims.n_out) != nullptr &&
        vulkan_q4k_nt4_cache_find(st, block->v_proj_weight, v_dims.n_in,
                                  v_dims.n_out) != nullptr;
    if (use_kv_pair) {
        s = vulkan_ensure_matvec_q4k_nt4_pair_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }

    if (rows == 0 ||
        rows > UINT32_MAX ||
        d > UINT32_MAX || q_out > UINT32_MAX || kv_out > UINT32_MAX ||
        block->q_heads > UINT32_MAX || block->kv_heads > UINT32_MAX ||
        block->head_dim > UINT32_MAX || block->kv_len > UINT32_MAX ||
        block->q_position > UINT32_MAX ||
        block->sliding_window > UINT32_MAX ||
        block->q_position > SIZE_MAX - rows ||
        block->q_position + rows > block->kv_len ||
        (block->head_dim % 2u) != 0 ||
        block->head_dim == 0 || block->head_dim > 512u ||
        block->q_heads == 0 || block->kv_heads == 0 ||
        block->q_heads % block->kv_heads != 0 ||
        block->q_position >= block->kv_len) {
        return GEIST_E_INVALID_ARG;
    }

    const struct geist_tensor *residual_add =
        block->post_attn_norm_weight != nullptr
            ? block->post_attn_scratch
            : block->o_scratch;
    const struct geist_tensor *offset_tensors[] = {
        block->residual, block->attn_norm_weight, block->q_norm_weight,
        block->k_norm_weight, block->v_norm_weight, block->cos, block->sin,
        block->k_cache, block->v_cache, block->normed_scratch,
        block->q_scratch, block->k_scratch, block->v_scratch,
        block->attn_scratch, block->o_scratch, residual_add, block->out,
        block->post_attn_norm_weight, block->post_attn_scratch,
    };
    for (size_t i = 0; i < sizeof(offset_tensors) / sizeof(offset_tensors[0]);
         i++) {
        if (offset_tensors[i] != nullptr &&
            (offset_tensors[i]->offset % sizeof(float)) != 0) {
            return GEIST_E_INVALID_ARG;
        }
    }

    const struct geist_tensor *owner_tensors[] = {
        block->residual, block->attn_norm_weight, block->q_proj_weight,
        block->k_proj_weight, block->v_proj_weight, block->q_norm_weight,
        block->k_norm_weight, block->v_norm_weight, block->cos, block->sin,
        block->k_cache, block->v_cache, block->o_proj_weight,
        block->normed_scratch, block->q_scratch, block->k_scratch,
        block->v_scratch, block->attn_scratch, block->o_scratch,
        residual_add, block->out, block->post_attn_norm_weight,
        block->post_attn_scratch,
    };
    for (size_t i = 0; i < sizeof(owner_tensors) / sizeof(owner_tensors[0]);
         i++) {
        if (owner_tensors[i] != nullptr &&
            !vulkan_ffn_owner_ok(be, owner_tensors[i])) {
            return GEIST_E_INVALID_ARG;
        }
    }

    if (block->kv_len > SIZE_MAX / kv_out ||
        block->kv_len * kv_out > SIZE_MAX / sizeof(float)) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t k_cache_bytes = block->kv_len * kv_out * sizeof(float);
    const size_t qkv_bytes = rows * q_out * sizeof(float);
    const size_t kv_bytes = rows * kv_out * sizeof(float);
    const size_t d_bytes = rows * d * sizeof(float);
    const size_t norm_bytes = d * sizeof(float);
    const size_t head_norm_bytes = block->head_dim * sizeof(float);
    bool cos_uses_positioned_table = false;
    bool sin_uses_positioned_table = false;
    if (!vulkan_rope_f32_2d_compatible(block->cos, rows, block->head_dim,
                                       block->q_position,
                                       &cos_uses_positioned_table) ||
        !vulkan_rope_f32_2d_compatible(block->sin, rows, block->head_dim,
                                       block->q_position,
                                       &sin_uses_positioned_table) ||
        cos_uses_positioned_table != sin_uses_positioned_table) {
        return GEIST_E_INVALID_ARG;
    }
    const bool rope_uses_positioned_table = cos_uses_positioned_table;
    const size_t rope_bytes =
        (size_t) block->cos->shape[0] * block->head_dim * sizeof(float);
    const size_t sin_rope_bytes =
        (size_t) block->sin->shape[0] * block->head_dim * sizeof(float);
    if (!vulkan_ffn_range_ok(block->residual, d_bytes) ||
        !vulkan_ffn_range_ok(block->attn_norm_weight, norm_bytes) ||
        !vulkan_ffn_range_ok(block->q_proj_weight, q_dims.weight_bytes) ||
        !vulkan_ffn_range_ok(block->k_proj_weight, k_dims.weight_bytes) ||
        !vulkan_ffn_range_ok(block->v_proj_weight, v_dims.weight_bytes) ||
        !vulkan_ffn_range_ok(block->q_norm_weight,
                             head_norm_bytes) ||
        !vulkan_ffn_range_ok(block->k_norm_weight,
                             head_norm_bytes) ||
        !vulkan_ffn_range_ok(block->v_norm_weight,
                             head_norm_bytes) ||
        !vulkan_ffn_range_ok(block->cos, rope_bytes) ||
        !vulkan_ffn_range_ok(block->sin, sin_rope_bytes) ||
        !vulkan_ffn_range_ok(block->k_cache, k_cache_bytes) ||
        !vulkan_ffn_range_ok(block->v_cache, k_cache_bytes) ||
        !vulkan_ffn_range_ok(block->o_proj_weight, o_dims.weight_bytes) ||
        !vulkan_ffn_range_ok(block->normed_scratch, d_bytes) ||
        !vulkan_ffn_range_ok(block->q_scratch, qkv_bytes) ||
        !vulkan_ffn_range_ok(block->k_scratch, kv_bytes) ||
        !vulkan_ffn_range_ok(block->v_scratch, kv_bytes) ||
        !vulkan_ffn_range_ok(block->attn_scratch, qkv_bytes) ||
        !vulkan_ffn_range_ok(block->o_scratch, d_bytes) ||
        !vulkan_ffn_range_ok(residual_add, d_bytes) ||
        !vulkan_ffn_range_ok(block->out, d_bytes) ||
        (block->post_attn_norm_weight != nullptr &&
         (!vulkan_ffn_range_ok(block->post_attn_norm_weight, norm_bytes) ||
          !vulkan_ffn_range_ok(block->post_attn_scratch, d_bytes)))) {
        return GEIST_E_INVALID_ARG;
    }

    const struct geist_tensor *read_tensors[] = {
        block->residual, block->attn_norm_weight, block->q_proj_weight,
        block->k_proj_weight, block->v_proj_weight, block->q_norm_weight,
        block->k_norm_weight, block->v_norm_weight, block->cos, block->sin,
        block->k_cache, block->v_cache, block->o_proj_weight,
        block->post_attn_norm_weight,
    };
    for (size_t i = 0; i < sizeof(read_tensors) / sizeof(read_tensors[0]);
         i++) {
        s = vulkan_ffn_flush_read(st, read_tensors[i]);
        if (s != GEIST_OK) { return s; }
    }
    if (use_dynamic_decode_params) {
        s = vulkan_write_decode_params_attention(
            be, (uint32_t) block->q_position, (uint32_t) block->kv_len);
        if (s != GEIST_OK) { return s; }
    }

    const uint32_t base_set_count =
        block->post_attn_norm_weight == nullptr || use_post_rmsnorm_add
            ? 12u
            : 13u;
    const uint32_t kv_pair_set_index = base_set_count;
    const uint32_t kv_append_set_index =
        base_set_count + (use_kv_pair ? 1u : 0u);
    const uint32_t set_count = base_set_count +
        (use_kv_pair ? 1u : 0u) +
        (use_dynamic_decode_params ? 1u : 0u);
    s = vulkan_ensure_block_descriptor_pool(
        be, set_count,
        set_count * 4u + (use_kv_pair ? 1u : 0u) +
            (use_dynamic_decode_params ? 2u : 0u));
    if (s != GEIST_OK) { return s; }

    VkDescriptorSetLayout layouts[15] = {
        st->rmsnorm_set_layout,
        vulkan_ffn_matvec_set_layout(st, block->q_proj_weight),
        vulkan_ffn_matvec_set_layout(st, block->k_proj_weight),
        vulkan_ffn_matvec_set_layout(st, block->v_proj_weight),
        st->rmsnorm_set_layout,
        st->rmsnorm_set_layout,
        st->rmsnorm_set_layout,
        use_dynamic_decode_params ? st->rope_dyn_set_layout
                                  : st->rope_set_layout,
        use_dynamic_decode_params ? st->rope_dyn_set_layout
                                  : st->rope_set_layout,
        use_dynamic_decode_params ? st->attention_dyn_set_layout
                                  : st->attention_set_layout,
        vulkan_ffn_matvec_set_layout(st, block->o_proj_weight),
        block->post_attn_norm_weight != nullptr
            ? (use_post_rmsnorm_add ? st->rmsnorm_add_set_layout
                                    : st->rmsnorm_set_layout)
            : st->add_set_layout,
        st->add_set_layout,
        VK_NULL_HANDLE,
        VK_NULL_HANDLE,
    };
    if (use_kv_pair) {
        layouts[kv_pair_set_index] = st->matvec_q4k_nt4_pair_set_layout;
    }
    if (use_dynamic_decode_params) {
        layouts[kv_append_set_index] = st->kv_append_dyn_set_layout;
    }
    VkDescriptorSet sets[15] = {VK_NULL_HANDLE};
    bool sets_reused[15] = {false};
    s = vulkan_alloc_block_descriptor_sets_cached(
        be, "attention_block", set_count, layouts, sets, sets_reused);
    if (s != GEIST_OK) { return s; }

    if (!sets_reused[0]) {
        vulkan_ffn_write_set(st, sets[0], block->residual,
                             block->attn_norm_weight, block->normed_scratch);
    }
    if (!sets_reused[1]) {
        vulkan_ffn_write_set(st, sets[1], &normed_1d,
                             block->q_proj_weight, &q_1d);
    }
    if (!sets_reused[2]) {
        vulkan_ffn_write_set(st, sets[2], &normed_1d,
                             block->k_proj_weight, &k_1d);
    }
    if (!sets_reused[3]) {
        vulkan_ffn_write_set(st, sets[3], &normed_1d,
                             block->v_proj_weight, &v_1d);
    }
    if (use_kv_pair && !sets_reused[kv_pair_set_index]) {
        vulkan_ffn_write_q4k_nt4_pair_set(
            st, sets[kv_pair_set_index], &normed_1d,
            block->k_proj_weight, block->v_proj_weight, &k_1d, &v_1d);
    }
    if (!sets_reused[4]) {
        vulkan_ffn_write_set(st, sets[4], block->q_scratch,
                             block->q_norm_weight, block->q_scratch);
    }
    if (!sets_reused[5]) {
        vulkan_ffn_write_set(st, sets[5], block->k_scratch,
                             block->k_norm_weight, block->k_scratch);
    }
    if (!sets_reused[6]) {
        vulkan_ffn_write_set(st, sets[6], block->v_scratch,
                             block->v_norm_weight, block->v_scratch);
    }
    if (!sets_reused[7]) {
        if (use_dynamic_decode_params) {
            vulkan_rope_dyn_write_set(st, sets[7], block->q_scratch,
                                      block->cos, block->sin);
        } else {
            vulkan_ffn_write_set(st, sets[7], block->q_scratch,
                                 block->cos, block->sin);
        }
    }
    if (!sets_reused[8]) {
        if (use_dynamic_decode_params) {
            vulkan_rope_dyn_write_set(st, sets[8], block->k_scratch,
                                      block->cos, block->sin);
        } else {
            vulkan_ffn_write_set(st, sets[8], block->k_scratch,
                                 block->cos, block->sin);
        }
    }
    if (!sets_reused[9]) {
        if (use_dynamic_decode_params) {
            vulkan_attn_write_set5_params(st, sets[9], block->q_scratch,
                                          block->k_cache, block->v_cache,
                                          block->attn_scratch);
        } else {
            vulkan_attn_write_set4(st, sets[9], block->q_scratch,
                                   block->k_cache, block->v_cache,
                                   block->attn_scratch);
        }
    }
    if (!sets_reused[10]) {
        vulkan_ffn_write_set(st, sets[10], &attn_1d,
                             block->o_proj_weight, &o_1d);
    }
    if (use_dynamic_decode_params && !sets_reused[kv_append_set_index]) {
        vulkan_kv_append_dyn_write_set(st, sets[kv_append_set_index],
                                       block->k_scratch, block->v_scratch,
                                       block->k_cache, block->v_cache);
    }
    if (block->post_attn_norm_weight != nullptr) {
        if (use_post_rmsnorm_add) {
            if (!sets_reused[11]) {
                vulkan_write_rmsnorm_add_set(st, sets[11], block->o_scratch,
                                             block->post_attn_norm_weight,
                                             block->residual, block->out);
            }
        } else {
            if (!sets_reused[11]) {
                vulkan_ffn_write_set(st, sets[11], block->o_scratch,
                                     block->post_attn_norm_weight,
                                     block->post_attn_scratch);
            }
            if (!sets_reused[12]) {
                vulkan_ffn_write_set(st, sets[12], block->residual,
                                     block->post_attn_scratch, block->out);
            }
        }
    } else {
        if (!sets_reused[11]) {
            vulkan_ffn_write_set(st, sets[11], block->residual,
                                 block->o_scratch, block->out);
        }
    }

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "attention_block", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_block_descriptor_sets(st, set_count, sets);
        return s;
    }
    vulkan_ffn_pre_barrier(st, cmd);

        vulkan_attn_cmd_rmsnorm(st, cmd, sets[0], block->residual,
                                block->attn_norm_weight,
                                block->normed_scratch, rows, d, block->eps);
        vulkan_ffn_shader_barrier(st, cmd);

        vulkan_ffn_cmd_matmul_or_matvec(st, cmd, sets[1], &normed_1d,
                                        block->q_proj_weight, &q_1d,
                                        rows, &q_dims);
        vulkan_ffn_shader_barrier(st, cmd);
        if (use_kv_pair) {
            vulkan_ffn_cmd_matvec_q4k_nt4_pair(
                st, cmd, sets[kv_pair_set_index], &normed_1d,
                block->k_proj_weight, block->v_proj_weight, &k_1d, &v_1d,
                &k_dims);
        } else {
            vulkan_ffn_cmd_matmul_or_matvec(st, cmd, sets[2], &normed_1d,
                                            block->k_proj_weight, &k_1d,
                                            rows, &k_dims);
            vulkan_ffn_shader_barrier(st, cmd);
            vulkan_ffn_cmd_matmul_or_matvec(st, cmd, sets[3], &normed_1d,
                                            block->v_proj_weight, &v_1d,
                                            rows, &v_dims);
        }
        vulkan_ffn_shader_barrier(st, cmd);

        vulkan_attn_cmd_rmsnorm(st, cmd, sets[4], block->q_scratch,
                                block->q_norm_weight, block->q_scratch,
                                rows * block->q_heads, block->head_dim,
                                block->eps);
        vulkan_ffn_shader_barrier(st, cmd);
        vulkan_attn_cmd_rmsnorm(st, cmd, sets[5], block->k_scratch,
                                block->k_norm_weight, block->k_scratch,
                                rows * block->kv_heads, block->head_dim,
                                block->eps);
        vulkan_ffn_shader_barrier(st, cmd);
        vulkan_attn_cmd_rmsnorm(st, cmd, sets[6], block->v_scratch,
                                block->v_norm_weight, block->v_scratch,
                                rows * block->kv_heads, block->head_dim,
                                block->eps);
        vulkan_ffn_shader_barrier(st, cmd);

        vulkan_attn_cmd_rope(st, cmd, sets[7], block->q_scratch,
                             block->cos, block->sin, rows, block->q_heads,
                             block->head_dim,
                             (!use_dynamic_decode_params &&
                              rope_uses_positioned_table)
                                 ? block->q_position
                                 : 0u,
                             use_dynamic_decode_params,
                             rope_uses_positioned_table);
        vulkan_ffn_shader_barrier(st, cmd);
        vulkan_attn_cmd_rope(st, cmd, sets[8], block->k_scratch,
                             block->cos, block->sin, rows, block->kv_heads,
                             block->head_dim,
                             (!use_dynamic_decode_params &&
                              rope_uses_positioned_table)
                                 ? block->q_position
                                 : 0u,
                             use_dynamic_decode_params,
                             rope_uses_positioned_table);
        vulkan_ffn_shader_barrier(st, cmd);

        if (use_dynamic_decode_params) {
            vulkan_attn_cmd_kv_append_dyn(st, cmd,
                                          sets[kv_append_set_index],
                                          block, rows, kv_out);
            vulkan_ffn_shader_barrier(st, cmd);
        } else {
            const VkBufferCopy k_copy = {
                .srcOffset = block->k_scratch->offset,
                .dstOffset = block->k_cache->offset +
                             block->q_position * kv_out * sizeof(float),
                .size = kv_bytes,
            };
            const VkBufferCopy v_copy = {
                .srcOffset = block->v_scratch->offset,
                .dstOffset = block->v_cache->offset +
                             block->q_position * kv_out * sizeof(float),
                .size = kv_bytes,
            };
            st->vkCmdCopyBuffer(cmd, block->k_scratch->buffer->buffer,
                                block->k_cache->buffer->buffer, 1, &k_copy);
            st->vkCmdCopyBuffer(cmd, block->v_scratch->buffer->buffer,
                                block->v_cache->buffer->buffer, 1, &v_copy);
            const VkMemoryBarrier copy_barrier = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            };
            st->vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     0, 1, &copy_barrier, 0, nullptr,
                                     0, nullptr);
        }

        vulkan_attn_cmd_attention(st, cmd, sets[9], block,
                                  use_dynamic_decode_params);
        vulkan_ffn_shader_barrier(st, cmd);

        vulkan_ffn_cmd_matmul_or_matvec(st, cmd, sets[10], &attn_1d,
                                        block->o_proj_weight, &o_1d,
                                        rows, &o_dims);
        vulkan_ffn_shader_barrier(st, cmd);

        uint32_t add_set_index = 11u;
        if (block->post_attn_norm_weight != nullptr) {
            if (use_post_rmsnorm_add) {
                vulkan_cmd_rmsnorm_add(st, cmd, sets[11], block->o_scratch,
                                       block->post_attn_norm_weight,
                                       block->residual, block->out, rows, d,
                                       block->eps);
                add_set_index = UINT32_MAX;
            } else {
                vulkan_attn_cmd_rmsnorm(st, cmd, sets[11], block->o_scratch,
                                        block->post_attn_norm_weight,
                                        block->post_attn_scratch,
                                        rows, d, block->eps);
                vulkan_ffn_shader_barrier(st, cmd);
                add_set_index = 12u;
            }
        }
        if (add_set_index != UINT32_MAX) {
            vulkan_attn_cmd_add(st, cmd, sets[add_set_index],
                                block->residual, residual_add, block->out,
                                rows * d);
        }

    vulkan_ffn_post_barrier(st, cmd);

    s = vulkan_submit_transient_command_buffer(be, cmd, "attention_block");
    vulkan_free_block_descriptor_sets(st, set_count, sets);
    if (s != GEIST_OK) {
        return s;
    }
    if (block->out->buffer->host_visible) {
        return vulkan_invalidate_if_needed(st, block->out->buffer);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_attention_block(
    struct geist_backend *be,
    const struct geist_backend_attention_block *block) {

    if (be == nullptr || be->state == nullptr || block == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (block->struct_size != sizeof(*block)) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t d = block->d_model;
    if (d == 0 || block->q_heads == 0 || block->kv_heads == 0 ||
        block->head_dim == 0 ||
        block->q_heads > SIZE_MAX / block->head_dim ||
        block->kv_heads > SIZE_MAX / block->head_dim ||
        block->q_position >= block->kv_len) {
        return GEIST_E_INVALID_ARG;
    }
    if (block->residual == nullptr || block->residual->ndim != 2 ||
        block->residual->shape[0] <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t rows = (size_t) block->residual->shape[0];
    if (block->q_position > SIZE_MAX - rows ||
        block->q_position + rows > block->kv_len) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t q_out = block->q_heads * block->head_dim;
    const size_t kv_out = block->kv_heads * block->head_dim;
    bool cos_uses_positioned_table = false;
    bool sin_uses_positioned_table = false;
    if (!vulkan_ffn_f32_2d_exact(block->residual, rows, d) ||
        !vulkan_ffn_f32_1d_exact(block->attn_norm_weight, d) ||
        !vulkan_ffn_weight_2d_shape(block->q_proj_weight, q_out, d) ||
        !vulkan_ffn_weight_2d_shape(block->k_proj_weight, kv_out, d) ||
        !vulkan_ffn_weight_2d_shape(block->v_proj_weight, kv_out, d) ||
        !vulkan_ffn_weight_2d_shape(block->o_proj_weight, d, q_out) ||
        !vulkan_ffn_weight_supported(block->q_proj_weight) ||
        !vulkan_ffn_weight_supported(block->k_proj_weight) ||
        !vulkan_ffn_weight_supported(block->v_proj_weight) ||
        !vulkan_ffn_weight_supported(block->o_proj_weight) ||
        !vulkan_ffn_f32_1d_exact(block->q_norm_weight, block->head_dim) ||
        !vulkan_ffn_f32_1d_exact(block->k_norm_weight, block->head_dim) ||
        !vulkan_ffn_f32_1d_exact(block->v_norm_weight, block->head_dim) ||
        !vulkan_rope_f32_2d_compatible(block->cos, rows, block->head_dim,
                                       block->q_position,
                                       &cos_uses_positioned_table) ||
        !vulkan_rope_f32_2d_compatible(block->sin, rows, block->head_dim,
                                       block->q_position,
                                       &sin_uses_positioned_table) ||
        cos_uses_positioned_table != sin_uses_positioned_table ||
        !vulkan_ffn_f32_3d_exact(block->k_cache, block->kv_len,
                                 block->kv_heads, block->head_dim) ||
        !vulkan_ffn_f32_3d_exact(block->v_cache, block->kv_len,
                                 block->kv_heads, block->head_dim) ||
        !vulkan_ffn_f32_2d_exact(block->normed_scratch, rows, d) ||
        !vulkan_ffn_f32_2d_exact(block->q_scratch, rows, q_out) ||
        !vulkan_ffn_f32_2d_exact(block->k_scratch, rows, kv_out) ||
        !vulkan_ffn_f32_2d_exact(block->v_scratch, rows, kv_out) ||
        !vulkan_ffn_f32_2d_exact(block->attn_scratch, rows, q_out) ||
        !vulkan_ffn_f32_2d_exact(block->o_scratch, rows, d) ||
        !vulkan_ffn_f32_2d_exact(block->out, rows, d)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "vulkan attention_block: unsupported tensor shape/layout");
        return GEIST_E_UNSUPPORTED;
    }
    if (block->post_attn_norm_weight != nullptr &&
        (!vulkan_ffn_f32_1d_exact(block->post_attn_norm_weight, d) ||
         !vulkan_ffn_f32_2d_exact(block->post_attn_scratch, rows, d))) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "vulkan attention_block: unsupported post norm tensors");
        return GEIST_E_UNSUPPORTED;
    }
    if (block->post_attn_norm_weight == nullptr &&
        block->post_attn_scratch != nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    return vulkan_attention_block_recorded(be, block);
}

[[nodiscard]] static enum geist_status vulkan_attention_query_block_recorded(
    struct geist_backend *be,
    const struct geist_backend_attention_query_block *block) {

    struct vulkan_state *st = be->state;
    const bool use_post_rmsnorm_add =
        block->post_attn_norm_weight != nullptr &&
        vulkan_block_rmsnorm_add_enabled();
    const bool use_dynamic_decode_params =
        st->capture_active &&
        st->capture_kind == GEIST_COMMAND_SEQUENCE_DECODE_GREEDY_STEP &&
        vulkan_dynamic_decode_params_enabled(st);
    const bool use_q_norm_rope =
        use_dynamic_decode_params && vulkan_rmsnorm_rope_dyn_enabled();
    enum geist_status s = vulkan_ensure_rmsnorm_pipeline(be);
    if (s != GEIST_OK) { return s; }
    s = use_dynamic_decode_params && !use_q_norm_rope
            ? vulkan_ensure_rope_dyn_pipeline(be)
            : vulkan_ensure_rope_pipeline(be);
    if (s != GEIST_OK) { return s; }
    if (use_q_norm_rope) {
        s = vulkan_ensure_rmsnorm_rope_dyn_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }
    s = use_dynamic_decode_params
            ? vulkan_ensure_attention_dyn_pipeline(be)
            : vulkan_ensure_attention_pipeline(be);
    if (s != GEIST_OK) { return s; }
    if (use_post_rmsnorm_add) {
        s = vulkan_ensure_rmsnorm_add_pipeline(be);
        if (s != GEIST_OK) { return s; }
    } else {
        s = vulkan_ensure_add_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }

    if (block->q_proj_weight->dtype == GEIST_DTYPE_F32 ||
        block->o_proj_weight->dtype == GEIST_DTYPE_F32) {
        s = vulkan_ensure_matvec_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }
    if (block->q_proj_weight->dtype == GEIST_DTYPE_F16 ||
        block->q_proj_weight->dtype == GEIST_DTYPE_BF16 ||
        block->o_proj_weight->dtype == GEIST_DTYPE_F16 ||
        block->o_proj_weight->dtype == GEIST_DTYPE_BF16) {
        s = vulkan_ensure_matvec_f16bf16_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }
    if (block->q_proj_weight->dtype == GEIST_DTYPE_Q4_K ||
        block->o_proj_weight->dtype == GEIST_DTYPE_Q4_K) {
        s = vulkan_ensure_matvec_q4k_pipeline(be);
        if (s != GEIST_OK) { return s; }
        s = vulkan_ensure_q4k_nt4_subgroup_if_enabled(be);
        if (s != GEIST_OK) { return s; }
    }
    if (block->q_proj_weight->dtype == GEIST_DTYPE_Q6_K ||
        block->o_proj_weight->dtype == GEIST_DTYPE_Q6_K) {
        s = vulkan_ensure_matvec_q6k_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }

    const size_t rows = (size_t) block->residual->shape[0];
    if (rows > 1u) {
        if (block->q_proj_weight->dtype == GEIST_DTYPE_F32 ||
            block->o_proj_weight->dtype == GEIST_DTYPE_F32) {
            s = vulkan_ensure_matmul_f32_pipeline(be);
            if (s != GEIST_OK) { return s; }
        }
        if (block->q_proj_weight->dtype == GEIST_DTYPE_Q4_K ||
            block->o_proj_weight->dtype == GEIST_DTYPE_Q4_K) {
            s = vulkan_ensure_matmul_q4k_m8_pipeline(be);
            if (s != GEIST_OK) { return s; }
        }
        if (block->q_proj_weight->dtype == GEIST_DTYPE_Q6_K ||
            block->o_proj_weight->dtype == GEIST_DTYPE_Q6_K) {
            s = vulkan_ensure_matmul_q6k_pipeline(be);
            if (s != GEIST_OK) { return s; }
        }
    }
    const size_t d = block->d_model;
    if (block->q_heads > SIZE_MAX / block->head_dim ||
        block->kv_heads > SIZE_MAX / block->head_dim) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t q_out = block->q_heads * block->head_dim;
    const size_t kv_out = block->kv_heads * block->head_dim;
    struct geist_tensor normed_1d =
        vulkan_ffn_1d_view(block->normed_scratch, d);
    struct geist_tensor q_1d =
        vulkan_ffn_1d_view(block->q_scratch, q_out);
    struct geist_tensor attn_1d =
        vulkan_ffn_1d_view(block->attn_scratch, q_out);
    struct geist_tensor o_1d =
        vulkan_ffn_1d_view(block->o_scratch, d);

    struct vulkan_ffn_matvec_dims q_dims;
    struct vulkan_ffn_matvec_dims o_dims;
    s = vulkan_ffn_matvec_dims(be, block->q_proj_weight, q_out, d, &q_dims);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ffn_matvec_dims(be, block->o_proj_weight, d, q_out, &o_dims);
    if (s != GEIST_OK) { return s; }

    if (rows == 0 ||
        rows > UINT32_MAX ||
        d > UINT32_MAX || q_out > UINT32_MAX || kv_out > UINT32_MAX ||
        block->q_heads > UINT32_MAX || block->kv_heads > UINT32_MAX ||
        block->head_dim > UINT32_MAX || block->kv_len > UINT32_MAX ||
        block->q_position > UINT32_MAX ||
        block->sliding_window > UINT32_MAX ||
        block->q_position > SIZE_MAX - rows ||
        block->q_position + rows > block->kv_len ||
        (block->head_dim % 2u) != 0 ||
        block->head_dim == 0 || block->head_dim > 512u ||
        block->q_heads == 0 || block->kv_heads == 0 ||
        block->q_heads % block->kv_heads != 0 ||
        block->q_position >= block->kv_len) {
        return GEIST_E_INVALID_ARG;
    }

    const struct geist_tensor *residual_add =
        block->post_attn_norm_weight != nullptr
            ? block->post_attn_scratch
            : block->o_scratch;
    const struct geist_tensor *offset_tensors[] = {
        block->residual, block->attn_norm_weight, block->q_norm_weight,
        block->cos, block->sin, block->k_cache, block->v_cache,
        block->normed_scratch, block->q_scratch, block->attn_scratch,
        block->o_scratch, residual_add, block->out,
        block->post_attn_norm_weight, block->post_attn_scratch,
    };
    for (size_t i = 0; i < sizeof(offset_tensors) / sizeof(offset_tensors[0]);
         i++) {
        if (offset_tensors[i] != nullptr &&
            (offset_tensors[i]->offset % sizeof(float)) != 0) {
            return GEIST_E_INVALID_ARG;
        }
    }

    const struct geist_tensor *owner_tensors[] = {
        block->residual, block->attn_norm_weight, block->q_proj_weight,
        block->q_norm_weight, block->cos, block->sin, block->k_cache,
        block->v_cache, block->o_proj_weight, block->normed_scratch,
        block->q_scratch, block->attn_scratch, block->o_scratch,
        residual_add, block->out, block->post_attn_norm_weight,
        block->post_attn_scratch,
    };
    for (size_t i = 0; i < sizeof(owner_tensors) / sizeof(owner_tensors[0]);
         i++) {
        if (owner_tensors[i] != nullptr &&
            !vulkan_ffn_owner_ok(be, owner_tensors[i])) {
            return GEIST_E_INVALID_ARG;
        }
    }

    if (block->kv_len > SIZE_MAX / kv_out ||
        block->kv_len * kv_out > SIZE_MAX / sizeof(float)) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t k_cache_bytes = block->kv_len * kv_out * sizeof(float);
    const size_t q_bytes = rows * q_out * sizeof(float);
    const size_t d_bytes = rows * d * sizeof(float);
    const size_t norm_bytes = d * sizeof(float);
    const size_t head_norm_bytes = block->head_dim * sizeof(float);
    bool cos_uses_positioned_table = false;
    bool sin_uses_positioned_table = false;
    if (!vulkan_rope_f32_2d_compatible(block->cos, rows, block->head_dim,
                                       block->q_position,
                                       &cos_uses_positioned_table) ||
        !vulkan_rope_f32_2d_compatible(block->sin, rows, block->head_dim,
                                       block->q_position,
                                       &sin_uses_positioned_table) ||
        cos_uses_positioned_table != sin_uses_positioned_table) {
        return GEIST_E_INVALID_ARG;
    }
    const bool rope_uses_positioned_table = cos_uses_positioned_table;
    const size_t rope_bytes =
        (size_t) block->cos->shape[0] * block->head_dim * sizeof(float);
    const size_t sin_rope_bytes =
        (size_t) block->sin->shape[0] * block->head_dim * sizeof(float);
    if (!vulkan_ffn_range_ok(block->residual, d_bytes) ||
        !vulkan_ffn_range_ok(block->attn_norm_weight, norm_bytes) ||
        !vulkan_ffn_range_ok(block->q_proj_weight, q_dims.weight_bytes) ||
        !vulkan_ffn_range_ok(block->q_norm_weight,
                             head_norm_bytes) ||
        !vulkan_ffn_range_ok(block->cos, rope_bytes) ||
        !vulkan_ffn_range_ok(block->sin, sin_rope_bytes) ||
        !vulkan_ffn_range_ok(block->k_cache, k_cache_bytes) ||
        !vulkan_ffn_range_ok(block->v_cache, k_cache_bytes) ||
        !vulkan_ffn_range_ok(block->o_proj_weight, o_dims.weight_bytes) ||
        !vulkan_ffn_range_ok(block->normed_scratch, d_bytes) ||
        !vulkan_ffn_range_ok(block->q_scratch, q_bytes) ||
        !vulkan_ffn_range_ok(block->attn_scratch, q_bytes) ||
        !vulkan_ffn_range_ok(block->o_scratch, d_bytes) ||
        !vulkan_ffn_range_ok(residual_add, d_bytes) ||
        !vulkan_ffn_range_ok(block->out, d_bytes) ||
        (block->post_attn_norm_weight != nullptr &&
         (!vulkan_ffn_range_ok(block->post_attn_norm_weight, norm_bytes) ||
          !vulkan_ffn_range_ok(block->post_attn_scratch, d_bytes)))) {
        return GEIST_E_INVALID_ARG;
    }

    const struct geist_tensor *read_tensors[] = {
        block->residual, block->attn_norm_weight, block->q_proj_weight,
        block->q_norm_weight, block->cos, block->sin, block->k_cache,
        block->v_cache, block->o_proj_weight, block->post_attn_norm_weight,
    };
    for (size_t i = 0; i < sizeof(read_tensors) / sizeof(read_tensors[0]);
         i++) {
        s = vulkan_ffn_flush_read(st, read_tensors[i]);
        if (s != GEIST_OK) { return s; }
    }
    if (use_dynamic_decode_params) {
        s = vulkan_write_decode_params_attention(
            be, (uint32_t) block->q_position, (uint32_t) block->kv_len);
        if (s != GEIST_OK) { return s; }
    }

    const uint32_t set_count =
        block->post_attn_norm_weight == nullptr || use_post_rmsnorm_add
            ? 7u
            : 8u;
    s = vulkan_ensure_block_descriptor_pool(
        be, set_count,
        set_count * 4u + (use_dynamic_decode_params ? 1u : 0u));
    if (s != GEIST_OK) { return s; }

    const VkDescriptorSetLayout layouts[8] = {
        st->rmsnorm_set_layout,
        vulkan_ffn_matvec_set_layout(st, block->q_proj_weight),
        use_q_norm_rope ? st->rmsnorm_rope_dyn_set_layout
                        : st->rmsnorm_set_layout,
        use_dynamic_decode_params ? st->rope_dyn_set_layout
                                  : st->rope_set_layout,
        use_dynamic_decode_params ? st->attention_dyn_set_layout
                                  : st->attention_set_layout,
        vulkan_ffn_matvec_set_layout(st, block->o_proj_weight),
        block->post_attn_norm_weight != nullptr
            ? (use_post_rmsnorm_add ? st->rmsnorm_add_set_layout
                                    : st->rmsnorm_set_layout)
            : st->add_set_layout,
        st->add_set_layout,
    };
    VkDescriptorSet sets[8] = {VK_NULL_HANDLE};
    bool sets_reused[8] = {false};
    s = vulkan_alloc_block_descriptor_sets_cached(
        be, "attention_query_block", set_count, layouts, sets, sets_reused);
    if (s != GEIST_OK) { return s; }

    if (!sets_reused[0]) {
        vulkan_ffn_write_set(st, sets[0], block->residual,
                             block->attn_norm_weight, block->normed_scratch);
    }
    if (!sets_reused[1]) {
        vulkan_ffn_write_set(st, sets[1], &normed_1d,
                             block->q_proj_weight, &q_1d);
    }
    if (!sets_reused[2]) {
        if (use_q_norm_rope) {
            vulkan_rmsnorm_rope_dyn_write_set(st, sets[2],
                                              block->q_scratch,
                                              block->q_norm_weight,
                                              block->cos, block->sin);
        } else {
            vulkan_ffn_write_set(st, sets[2], block->q_scratch,
                                 block->q_norm_weight, block->q_scratch);
        }
    }
    if (!use_q_norm_rope && !sets_reused[3]) {
        if (use_dynamic_decode_params) {
            vulkan_rope_dyn_write_set(st, sets[3], block->q_scratch,
                                      block->cos, block->sin);
        } else {
            vulkan_ffn_write_set(st, sets[3], block->q_scratch,
                                 block->cos, block->sin);
        }
    }
    if (!sets_reused[4]) {
        if (use_dynamic_decode_params) {
            vulkan_attn_write_set5_params(st, sets[4], block->q_scratch,
                                          block->k_cache, block->v_cache,
                                          block->attn_scratch);
        } else {
            vulkan_attn_write_set4(st, sets[4], block->q_scratch,
                                   block->k_cache, block->v_cache,
                                   block->attn_scratch);
        }
    }
    if (!sets_reused[5]) {
        vulkan_ffn_write_set(st, sets[5], &attn_1d,
                             block->o_proj_weight, &o_1d);
    }
    if (block->post_attn_norm_weight != nullptr) {
        if (use_post_rmsnorm_add) {
            if (!sets_reused[6]) {
                vulkan_write_rmsnorm_add_set(st, sets[6], block->o_scratch,
                                             block->post_attn_norm_weight,
                                             block->residual, block->out);
            }
        } else {
            if (!sets_reused[6]) {
                vulkan_ffn_write_set(st, sets[6], block->o_scratch,
                                     block->post_attn_norm_weight,
                                     block->post_attn_scratch);
            }
            if (!sets_reused[7]) {
                vulkan_ffn_write_set(st, sets[7], block->residual,
                                     block->post_attn_scratch, block->out);
            }
        }
    } else {
        if (!sets_reused[6]) {
            vulkan_ffn_write_set(st, sets[6], block->residual,
                                 block->o_scratch, block->out);
        }
    }

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "attention_query_block", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_block_descriptor_sets(st, set_count, sets);
        return s;
    }

    vulkan_ffn_pre_barrier(st, cmd);
    vulkan_attn_cmd_rmsnorm(st, cmd, sets[0], block->residual,
                            block->attn_norm_weight, block->normed_scratch,
                            rows, d, block->eps);
    vulkan_ffn_shader_barrier(st, cmd);

    vulkan_ffn_cmd_matmul_or_matvec(st, cmd, sets[1], &normed_1d,
                                    block->q_proj_weight, &q_1d,
                                    rows, &q_dims);
    vulkan_ffn_shader_barrier(st, cmd);

    if (use_q_norm_rope) {
        vulkan_attn_cmd_rmsnorm_rope_dyn(
            st, cmd, sets[2], block->q_scratch, block->q_norm_weight,
            block->cos, block->sin, rows, block->q_heads, block->head_dim,
            rope_uses_positioned_table, block->eps);
    } else {
        vulkan_attn_cmd_rmsnorm(st, cmd, sets[2], block->q_scratch,
                                block->q_norm_weight, block->q_scratch,
                                rows * block->q_heads, block->head_dim,
                                block->eps);
        vulkan_ffn_shader_barrier(st, cmd);

        vulkan_attn_cmd_rope(st, cmd, sets[3], block->q_scratch,
                             block->cos, block->sin, rows, block->q_heads,
                             block->head_dim,
                             (!use_dynamic_decode_params &&
                              rope_uses_positioned_table)
                                 ? block->q_position
                                 : 0u,
                             use_dynamic_decode_params,
                             rope_uses_positioned_table);
    }
    vulkan_ffn_shader_barrier(st, cmd);

    vulkan_attn_query_cmd_attention(st, cmd, sets[4], block,
                                    use_dynamic_decode_params);
    vulkan_ffn_shader_barrier(st, cmd);

    vulkan_ffn_cmd_matmul_or_matvec(st, cmd, sets[5], &attn_1d,
                                    block->o_proj_weight, &o_1d,
                                    rows, &o_dims);
    vulkan_ffn_shader_barrier(st, cmd);

    uint32_t add_set_index = 6u;
    if (block->post_attn_norm_weight != nullptr) {
        if (use_post_rmsnorm_add) {
            vulkan_cmd_rmsnorm_add(st, cmd, sets[6], block->o_scratch,
                                   block->post_attn_norm_weight,
                                   block->residual, block->out, rows, d,
                                   block->eps);
            add_set_index = UINT32_MAX;
        } else {
            vulkan_attn_cmd_rmsnorm(st, cmd, sets[6], block->o_scratch,
                                    block->post_attn_norm_weight,
                                    block->post_attn_scratch, rows, d,
                                    block->eps);
            vulkan_ffn_shader_barrier(st, cmd);
            add_set_index = 7u;
        }
    }
    if (add_set_index != UINT32_MAX) {
        vulkan_attn_cmd_add(st, cmd, sets[add_set_index], block->residual,
                            residual_add, block->out, rows * d);
    }

    vulkan_ffn_post_barrier(st, cmd);

    s = vulkan_submit_transient_command_buffer(be, cmd,
                                               "attention_query_block");
    vulkan_free_block_descriptor_sets(st, set_count, sets);
    if (s != GEIST_OK) {
        return s;
    }
    if (block->out->buffer->host_visible) {
        return vulkan_invalidate_if_needed(st, block->out->buffer);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_attention_query_block(
    struct geist_backend *be,
    const struct geist_backend_attention_query_block *block) {

    if (be == nullptr || be->state == nullptr || block == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (block->struct_size != sizeof(*block)) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t d = block->d_model;
    if (d == 0 || block->q_heads == 0 || block->kv_heads == 0 ||
        block->head_dim == 0 ||
        block->q_heads > SIZE_MAX / block->head_dim ||
        block->kv_heads > SIZE_MAX / block->head_dim ||
        block->q_position >= block->kv_len) {
        return GEIST_E_INVALID_ARG;
    }
    if (block->residual == nullptr || block->residual->ndim != 2 ||
        block->residual->shape[0] <= 0) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t rows = (size_t) block->residual->shape[0];
    if (block->q_position > SIZE_MAX - rows ||
        block->q_position + rows > block->kv_len) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t q_out = block->q_heads * block->head_dim;
    bool cos_uses_positioned_table = false;
    bool sin_uses_positioned_table = false;
    if (!vulkan_ffn_f32_2d_exact(block->residual, rows, d) ||
        !vulkan_ffn_f32_1d_exact(block->attn_norm_weight, d) ||
        !vulkan_ffn_weight_2d_shape(block->q_proj_weight, q_out, d) ||
        !vulkan_ffn_weight_2d_shape(block->o_proj_weight, d, q_out) ||
        !vulkan_ffn_weight_supported(block->q_proj_weight) ||
        !vulkan_ffn_weight_supported(block->o_proj_weight) ||
        !vulkan_ffn_f32_1d_exact(block->q_norm_weight, block->head_dim) ||
        !vulkan_rope_f32_2d_compatible(block->cos, rows, block->head_dim,
                                       block->q_position,
                                       &cos_uses_positioned_table) ||
        !vulkan_rope_f32_2d_compatible(block->sin, rows, block->head_dim,
                                       block->q_position,
                                       &sin_uses_positioned_table) ||
        cos_uses_positioned_table != sin_uses_positioned_table ||
        !vulkan_ffn_f32_3d_exact(block->k_cache, block->kv_len,
                                 block->kv_heads, block->head_dim) ||
        !vulkan_ffn_f32_3d_exact(block->v_cache, block->kv_len,
                                 block->kv_heads, block->head_dim) ||
        !vulkan_ffn_f32_2d_exact(block->normed_scratch, rows, d) ||
        !vulkan_ffn_f32_2d_exact(block->q_scratch, rows, q_out) ||
        !vulkan_ffn_f32_2d_exact(block->attn_scratch, rows, q_out) ||
        !vulkan_ffn_f32_2d_exact(block->o_scratch, rows, d) ||
        !vulkan_ffn_f32_2d_exact(block->out, rows, d)) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "vulkan attention_query_block: unsupported tensor shape/layout");
        return GEIST_E_UNSUPPORTED;
    }
    if (block->post_attn_norm_weight != nullptr &&
        (!vulkan_ffn_f32_1d_exact(block->post_attn_norm_weight, d) ||
         !vulkan_ffn_f32_2d_exact(block->post_attn_scratch, rows, d))) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "vulkan attention_query_block: unsupported post norm tensors");
        return GEIST_E_UNSUPPORTED;
    }
    if (block->post_attn_norm_weight == nullptr &&
        block->post_attn_scratch != nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    return vulkan_attention_query_block_recorded(be, block);
}

[[nodiscard]] static enum geist_status vulkan_ffn_geglu_block_recorded(
    struct geist_backend *be,
    const struct geist_backend_ffn_geglu_block *block) {

    struct vulkan_state *st = be->state;
    const bool use_post_rmsnorm_add =
        block->post_ffw_norm_weight != nullptr &&
        vulkan_block_rmsnorm_add_enabled();
    enum geist_status s = vulkan_ensure_rmsnorm_pipeline(be);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ensure_gelu_tanh_mul_pipeline(be);
    if (s != GEIST_OK) { return s; }
    if (use_post_rmsnorm_add) {
        s = vulkan_ensure_rmsnorm_add_pipeline(be);
        if (s != GEIST_OK) { return s; }
    } else {
        s = vulkan_ensure_add_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }
    const bool batched = block->seq > 1u;
    if (batched &&
        (block->gate_weight->dtype == GEIST_DTYPE_F16 ||
         block->gate_weight->dtype == GEIST_DTYPE_BF16 ||
         block->up_weight->dtype == GEIST_DTYPE_F16 ||
         block->up_weight->dtype == GEIST_DTYPE_BF16 ||
         block->down_weight->dtype == GEIST_DTYPE_F16 ||
         block->down_weight->dtype == GEIST_DTYPE_BF16)) {
        geist_backend_set_error(
            be, GEIST_E_UNSUPPORTED,
            "vulkan ffn_geglu_block: batched F16/BF16 weights unsupported");
        return GEIST_E_UNSUPPORTED;
    }
    if (block->gate_weight->dtype == GEIST_DTYPE_Q4_K ||
        block->up_weight->dtype == GEIST_DTYPE_Q4_K ||
        block->down_weight->dtype == GEIST_DTYPE_Q4_K) {
        s = batched ? vulkan_ensure_matmul_q4k_m8_pipeline(be)
                    : vulkan_ensure_matvec_q4k_pipeline(be);
        if (s != GEIST_OK) { return s; }
        if (!batched) {
            s = vulkan_ensure_q4k_nt4_subgroup_if_enabled(be);
            if (s != GEIST_OK) { return s; }
        }
    }
    if (block->gate_weight->dtype == GEIST_DTYPE_Q6_K ||
        block->up_weight->dtype == GEIST_DTYPE_Q6_K ||
        block->down_weight->dtype == GEIST_DTYPE_Q6_K) {
        s = batched ? vulkan_ensure_matmul_q6k_pipeline(be)
                    : vulkan_ensure_matvec_q6k_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }
    if (block->gate_weight->dtype == GEIST_DTYPE_F32 ||
        block->up_weight->dtype == GEIST_DTYPE_F32 ||
        block->down_weight->dtype == GEIST_DTYPE_F32) {
        s = batched ? vulkan_ensure_matmul_f32_pipeline(be)
                    : vulkan_ensure_matvec_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }
    if (block->gate_weight->dtype == GEIST_DTYPE_F16 ||
        block->gate_weight->dtype == GEIST_DTYPE_BF16 ||
        block->up_weight->dtype == GEIST_DTYPE_F16 ||
        block->up_weight->dtype == GEIST_DTYPE_BF16 ||
        block->down_weight->dtype == GEIST_DTYPE_F16 ||
        block->down_weight->dtype == GEIST_DTYPE_BF16) {
        s = vulkan_ensure_matvec_f16bf16_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }

    struct geist_tensor pre_ff_1d =
        vulkan_ffn_1d_view(block->pre_ff_scratch, block->d_model);
    struct geist_tensor gate_1d =
        vulkan_ffn_1d_view(block->gate_scratch, block->inter);
    struct geist_tensor up_1d =
        vulkan_ffn_1d_view(block->up_scratch, block->inter);
    struct geist_tensor ffn_out_1d =
        vulkan_ffn_1d_view(block->ffn_out_scratch, block->d_model);

    struct vulkan_ffn_matvec_dims gate_dims;
    struct vulkan_ffn_matvec_dims up_dims;
    struct vulkan_ffn_matvec_dims down_dims;
    s = vulkan_ffn_matvec_dims(be, block->gate_weight, block->inter,
                               block->d_model, &gate_dims);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ffn_matvec_dims(be, block->up_weight, block->inter,
                               block->d_model, &up_dims);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ffn_matvec_dims(be, block->down_weight, block->d_model,
                               block->inter, &down_dims);
    if (s != GEIST_OK) { return s; }

    const bool use_gate_up_gelu_q4k =
        vulkan_ffn_gate_up_gelu_q4k_enabled() &&
        !batched &&
        block->gate_weight->dtype == GEIST_DTYPE_Q4_K &&
        block->up_weight->dtype == GEIST_DTYPE_Q4_K &&
        gate_dims.n_in == up_dims.n_in &&
        gate_dims.n_out == up_dims.n_out &&
        gate_dims.blocks_per_row == up_dims.blocks_per_row;
    if (use_gate_up_gelu_q4k) {
        s = vulkan_ensure_ffn_gate_up_gelu_q4k_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }
    const bool use_gate_up_pair =
        !use_gate_up_gelu_q4k &&
        vulkan_q4k_nt4_pair_enabled() &&
        !batched &&
        block->gate_weight->dtype == GEIST_DTYPE_Q4_K &&
        block->up_weight->dtype == GEIST_DTYPE_Q4_K &&
        gate_dims.n_in == up_dims.n_in &&
        gate_dims.n_out == up_dims.n_out &&
        gate_dims.blocks_per_row == up_dims.blocks_per_row &&
        vulkan_q4k_nt4_cache_find(st, block->gate_weight, gate_dims.n_in,
                                  gate_dims.n_out) != nullptr &&
        vulkan_q4k_nt4_cache_find(st, block->up_weight, up_dims.n_in,
                                  up_dims.n_out) != nullptr;
    if (use_gate_up_pair) {
        s = vulkan_ensure_matvec_q4k_nt4_pair_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }
    const bool use_gate_up_gelu =
        !use_gate_up_gelu_q4k &&
        vulkan_ffn_gate_up_gelu_q4k_nt4_enabled() &&
        !use_gate_up_pair &&
        !batched &&
        block->gate_weight->dtype == GEIST_DTYPE_Q4_K &&
        block->up_weight->dtype == GEIST_DTYPE_Q4_K &&
        gate_dims.n_in == up_dims.n_in &&
        gate_dims.n_out == up_dims.n_out &&
        gate_dims.blocks_per_row == up_dims.blocks_per_row &&
        vulkan_q4k_nt4_cache_find(st, block->gate_weight, gate_dims.n_in,
                                  gate_dims.n_out) != nullptr &&
        vulkan_q4k_nt4_cache_find(st, block->up_weight, up_dims.n_in,
                                  up_dims.n_out) != nullptr;
    if (use_gate_up_gelu) {
        s = vulkan_ensure_ffn_gate_up_gelu_q4k_nt4_pipeline(be);
        if (s != GEIST_OK) { return s; }
    }

    const struct geist_tensor *residual_add =
        block->post_ffw_norm_weight != nullptr
            ? block->post_ff_scratch
            : block->ffn_out_scratch;
    const size_t rows = block->seq;
    const size_t d = block->d_model;
    const size_t inter = block->inter;
    if (rows > UINT32_MAX || d > UINT32_MAX || inter > UINT32_MAX ||
        rows > SIZE_MAX / d ||
        rows > SIZE_MAX / inter ||
        block->residual->offset % sizeof(float) != 0 ||
        block->ffn_norm_weight->offset % sizeof(float) != 0 ||
        block->pre_ff_scratch->offset % sizeof(float) != 0 ||
        block->gate_scratch->offset % sizeof(float) != 0 ||
        block->up_scratch->offset % sizeof(float) != 0 ||
        block->ffn_out_scratch->offset % sizeof(float) != 0 ||
        residual_add->offset % sizeof(float) != 0 ||
        block->out->offset % sizeof(float) != 0 ||
        (block->post_ffw_norm_weight != nullptr &&
         (block->post_ffw_norm_weight->offset % sizeof(float) != 0 ||
          block->post_ff_scratch->offset % sizeof(float) != 0))) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t act_d = rows * d;
    const size_t act_inter = rows * inter;
    if (act_d > UINT32_MAX || act_inter > UINT32_MAX) {
        return GEIST_E_INVALID_ARG;
    }

    if (!vulkan_ffn_owner_ok(be, block->residual) ||
        !vulkan_ffn_owner_ok(be, block->ffn_norm_weight) ||
        !vulkan_ffn_owner_ok(be, block->gate_weight) ||
        !vulkan_ffn_owner_ok(be, block->up_weight) ||
        !vulkan_ffn_owner_ok(be, block->down_weight) ||
        !vulkan_ffn_owner_ok(be, block->pre_ff_scratch) ||
        !vulkan_ffn_owner_ok(be, block->gate_scratch) ||
        !vulkan_ffn_owner_ok(be, block->up_scratch) ||
        !vulkan_ffn_owner_ok(be, block->ffn_out_scratch) ||
        !vulkan_ffn_owner_ok(be, residual_add) ||
        !vulkan_ffn_owner_ok(be, block->out) ||
        (block->post_ffw_norm_weight != nullptr &&
         (!vulkan_ffn_owner_ok(be, block->post_ffw_norm_weight) ||
          !vulkan_ffn_owner_ok(be, block->post_ff_scratch)))) {
        return GEIST_E_INVALID_ARG;
    }

    if (!vulkan_ffn_range_ok(block->residual, act_d * sizeof(float)) ||
        !vulkan_ffn_range_ok(block->ffn_norm_weight, d * sizeof(float)) ||
        !vulkan_ffn_range_ok(block->gate_weight, gate_dims.weight_bytes) ||
        !vulkan_ffn_range_ok(block->up_weight, up_dims.weight_bytes) ||
        !vulkan_ffn_range_ok(block->down_weight, down_dims.weight_bytes) ||
        !vulkan_ffn_range_ok(block->pre_ff_scratch, act_d * sizeof(float)) ||
        !vulkan_ffn_range_ok(block->gate_scratch, act_inter * sizeof(float)) ||
        !vulkan_ffn_range_ok(block->up_scratch, act_inter * sizeof(float)) ||
        !vulkan_ffn_range_ok(block->ffn_out_scratch, act_d * sizeof(float)) ||
        !vulkan_ffn_range_ok(residual_add, act_d * sizeof(float)) ||
        !vulkan_ffn_range_ok(block->out, act_d * sizeof(float)) ||
        (block->post_ffw_norm_weight != nullptr &&
         (!vulkan_ffn_range_ok(block->post_ffw_norm_weight,
                               d * sizeof(float)) ||
          !vulkan_ffn_range_ok(block->post_ff_scratch,
                               act_d * sizeof(float))))) {
        return GEIST_E_INVALID_ARG;
    }

    s = vulkan_ffn_flush_read(st, block->residual);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ffn_flush_read(st, block->ffn_norm_weight);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ffn_flush_read(st, block->gate_weight);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ffn_flush_read(st, block->up_weight);
    if (s != GEIST_OK) { return s; }
    s = vulkan_ffn_flush_read(st, block->down_weight);
    if (s != GEIST_OK) { return s; }
    if (block->post_ffw_norm_weight != nullptr) {
        s = vulkan_ffn_flush_read(st, block->post_ffw_norm_weight);
        if (s != GEIST_OK) { return s; }
    }

    const uint32_t base_set_count =
        block->post_ffw_norm_weight != nullptr && !use_post_rmsnorm_add
            ? 7u
            : 6u;
    const uint32_t pair_set_index = base_set_count;
    const uint32_t fused_set_index =
        base_set_count + (use_gate_up_pair ? 1u : 0u);
    const bool use_any_gate_up_gelu =
        use_gate_up_gelu_q4k || use_gate_up_gelu;
    const uint32_t set_count = base_set_count +
        (use_gate_up_pair ? 1u : 0u) +
        (use_any_gate_up_gelu ? 1u : 0u);
    const uint32_t descriptor_count = base_set_count * 4u +
        (use_gate_up_pair ? 5u : 0u) +
        (use_any_gate_up_gelu ? 4u : 0u);
    s = vulkan_ensure_block_descriptor_pool(be, set_count, descriptor_count);
    if (s != GEIST_OK) { return s; }

    VkDescriptorSetLayout layouts[9] = {
        st->rmsnorm_set_layout,
        vulkan_ffn_matvec_set_layout(st, block->gate_weight),
        vulkan_ffn_matvec_set_layout(st, block->up_weight),
        st->gelu_tanh_mul_set_layout,
        vulkan_ffn_matvec_set_layout(st, block->down_weight),
        block->post_ffw_norm_weight != nullptr
            ? (use_post_rmsnorm_add ? st->rmsnorm_add_set_layout
                                    : st->rmsnorm_set_layout)
            : st->add_set_layout,
        st->add_set_layout,
    };
    if (use_gate_up_pair) {
        layouts[pair_set_index] = st->matvec_q4k_nt4_pair_set_layout;
    }
    if (use_gate_up_gelu_q4k) {
        layouts[fused_set_index] = st->ffn_gate_up_gelu_q4k_set_layout;
    } else if (use_gate_up_gelu) {
        layouts[fused_set_index] = st->ffn_gate_up_gelu_q4k_nt4_set_layout;
    }
    VkDescriptorSet sets[9] = {VK_NULL_HANDLE};
    bool sets_reused[9] = {false};
    s = vulkan_alloc_block_descriptor_sets_cached(
        be, "ffn_geglu_block", set_count, layouts, sets, sets_reused);
    if (s != GEIST_OK) { return s; }

    if (!sets_reused[0]) {
        vulkan_ffn_write_set(st, sets[0], block->residual,
                             block->ffn_norm_weight, block->pre_ff_scratch);
    }
    if (use_any_gate_up_gelu) {
        if (!sets_reused[fused_set_index]) {
            if (use_gate_up_gelu_q4k) {
                vulkan_attn_write_set4(st, sets[fused_set_index],
                                       &pre_ff_1d, block->gate_weight,
                                       block->up_weight, &gate_1d);
            } else {
                vulkan_ffn_write_gate_up_gelu_q4k_nt4_set(
                    st, sets[fused_set_index], &pre_ff_1d,
                    block->gate_weight, block->up_weight, &gate_1d);
            }
        }
    } else {
        if (!sets_reused[1]) {
            vulkan_ffn_write_set(st, sets[1], &pre_ff_1d,
                                 block->gate_weight, &gate_1d);
        }
        if (!sets_reused[2]) {
            vulkan_ffn_write_set(st, sets[2], &pre_ff_1d,
                                 block->up_weight, &up_1d);
        }
        if (use_gate_up_pair) {
            if (!sets_reused[pair_set_index]) {
                vulkan_ffn_write_q4k_nt4_pair_set(
                    st, sets[pair_set_index], &pre_ff_1d,
                    block->gate_weight, block->up_weight, &gate_1d, &up_1d);
            }
        }
        if (!sets_reused[3]) {
            vulkan_ffn_write_set(st, sets[3], block->gate_scratch,
                                 block->up_scratch, block->gate_scratch);
        }
    }
    if (!sets_reused[4]) {
        vulkan_ffn_write_set(st, sets[4], &gate_1d,
                             block->down_weight, &ffn_out_1d);
    }
    if (block->post_ffw_norm_weight != nullptr) {
        if (use_post_rmsnorm_add) {
            if (!sets_reused[5]) {
                vulkan_write_rmsnorm_add_set(st, sets[5],
                                             block->ffn_out_scratch,
                                             block->post_ffw_norm_weight,
                                             block->residual, block->out);
            }
        } else {
            if (!sets_reused[5]) {
                vulkan_ffn_write_set(st, sets[5], block->ffn_out_scratch,
                                     block->post_ffw_norm_weight,
                                     block->post_ff_scratch);
            }
            if (!sets_reused[6]) {
                vulkan_ffn_write_set(st, sets[6], block->residual,
                                     block->post_ff_scratch, block->out);
            }
        }
    } else {
        if (!sets_reused[5]) {
            vulkan_ffn_write_set(st, sets[5], block->residual,
                                 block->ffn_out_scratch, block->out);
        }
    }

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    s = vulkan_begin_transient_command_buffer(be, "ffn_geglu_block", &cmd);
    if (s != GEIST_OK) {
        vulkan_free_block_descriptor_sets(st, set_count, sets);
        return s;
    }
    vulkan_ffn_pre_barrier(st, cmd);

        struct rmsnorm_push {
            uint32_t rows;
            uint32_t feat;
            uint32_t x_offset;
            uint32_t w_offset;
            uint32_t y_offset;
            float eps;
        };
        const struct rmsnorm_push pre_push = {
            .rows = (uint32_t) rows,
            .feat = (uint32_t) d,
            .x_offset = (uint32_t) (block->residual->offset / sizeof(float)),
            .w_offset =
                (uint32_t) (block->ffn_norm_weight->offset / sizeof(float)),
            .y_offset =
                (uint32_t) (block->pre_ff_scratch->offset / sizeof(float)),
            .eps = block->eps,
        };
        st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              st->rmsnorm_pipeline);
        st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    st->rmsnorm_pipeline_layout, 0, 1,
                                    &sets[0], 0, nullptr);
        st->vkCmdPushConstants(cmd, st->rmsnorm_pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(pre_push), &pre_push);
        st->vkCmdDispatch(cmd, (uint32_t) rows, 1, 1);
        vulkan_ffn_shader_barrier(st, cmd);

        if (use_any_gate_up_gelu) {
            if (use_gate_up_gelu_q4k) {
                vulkan_ffn_cmd_gate_up_gelu_q4k(
                    st, cmd, sets[fused_set_index], &pre_ff_1d,
                    block->gate_weight, block->up_weight, &gate_1d,
                    &gate_dims);
            } else {
                vulkan_ffn_cmd_gate_up_gelu_q4k_nt4(
                    st, cmd, sets[fused_set_index], &pre_ff_1d,
                    block->gate_weight, block->up_weight, &gate_1d,
                    &gate_dims);
            }
            vulkan_ffn_shader_barrier(st, cmd);
        } else {
            if (use_gate_up_pair) {
                vulkan_ffn_cmd_matvec_q4k_nt4_pair(
                    st, cmd, sets[pair_set_index], &pre_ff_1d,
                    block->gate_weight, block->up_weight, &gate_1d, &up_1d,
                    &gate_dims);
            } else {
                vulkan_ffn_cmd_matmul_or_matvec(st, cmd, sets[1], &pre_ff_1d,
                                                block->gate_weight, &gate_1d,
                                                rows, &gate_dims);
                vulkan_ffn_cmd_matmul_or_matvec(st, cmd, sets[2], &pre_ff_1d,
                                                block->up_weight, &up_1d,
                                                rows, &up_dims);
            }
            vulkan_ffn_shader_barrier(st, cmd);

            const uint32_t gelu_push[4] = {
                (uint32_t) act_inter,
                (uint32_t) (block->gate_scratch->offset / sizeof(float)),
                (uint32_t) (block->up_scratch->offset / sizeof(float)),
                (uint32_t) (block->gate_scratch->offset / sizeof(float)),
            };
            st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  st->gelu_tanh_mul_pipeline);
            st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        st->gelu_tanh_mul_pipeline_layout, 0,
                                        1, &sets[3], 0, nullptr);
            st->vkCmdPushConstants(cmd, st->gelu_tanh_mul_pipeline_layout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(gelu_push), gelu_push);
            st->vkCmdDispatch(cmd, (uint32_t) ((act_inter + 255u) / 256u), 1,
                              1);
            vulkan_ffn_shader_barrier(st, cmd);
        }

        vulkan_ffn_cmd_matmul_or_matvec(st, cmd, sets[4], &gate_1d,
                                        block->down_weight, &ffn_out_1d,
                                        rows, &down_dims);
        vulkan_ffn_shader_barrier(st, cmd);

        uint32_t add_set_index = 5u;
        if (block->post_ffw_norm_weight != nullptr) {
            if (use_post_rmsnorm_add) {
                vulkan_cmd_rmsnorm_add(st, cmd, sets[5],
                                       block->ffn_out_scratch,
                                       block->post_ffw_norm_weight,
                                       block->residual, block->out, rows, d,
                                       block->eps);
                add_set_index = UINT32_MAX;
            } else {
                const struct rmsnorm_push post_push = {
                    .rows = (uint32_t) rows,
                    .feat = (uint32_t) d,
                    .x_offset =
                        (uint32_t) (block->ffn_out_scratch->offset /
                                    sizeof(float)),
                    .w_offset =
                        (uint32_t) (block->post_ffw_norm_weight->offset /
                                    sizeof(float)),
                    .y_offset =
                        (uint32_t) (block->post_ff_scratch->offset /
                                    sizeof(float)),
                    .eps = block->eps,
                };
                st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                      st->rmsnorm_pipeline);
                st->vkCmdBindDescriptorSets(
                    cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                    st->rmsnorm_pipeline_layout, 0, 1, &sets[5], 0, nullptr);
                st->vkCmdPushConstants(cmd, st->rmsnorm_pipeline_layout,
                                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                       sizeof(post_push), &post_push);
                st->vkCmdDispatch(cmd, (uint32_t) rows, 1, 1);
                vulkan_ffn_shader_barrier(st, cmd);
                add_set_index = 6u;
            }
        }

        if (add_set_index != UINT32_MAX) {
            const uint32_t add_push[4] = {
                (uint32_t) act_d,
                (uint32_t) (block->residual->offset / sizeof(float)),
                (uint32_t) (residual_add->offset / sizeof(float)),
                (uint32_t) (block->out->offset / sizeof(float)),
            };
            st->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                  st->add_pipeline);
            st->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                        st->add_pipeline_layout, 0, 1,
                                        &sets[add_set_index], 0, nullptr);
            st->vkCmdPushConstants(cmd, st->add_pipeline_layout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(add_push), add_push);
            st->vkCmdDispatch(cmd, (uint32_t) ((act_d + 255u) / 256u), 1, 1);
        }

    vulkan_ffn_post_barrier(st, cmd);

    s = vulkan_submit_transient_command_buffer(be, cmd, "ffn_geglu_block");
    vulkan_free_block_descriptor_sets(st, set_count, sets);
    if (s != GEIST_OK) {
        return s;
    }
    if (block->out->buffer->host_visible) {
        return vulkan_invalidate_if_needed(st, block->out->buffer);
    }
    return GEIST_OK;
}

[[nodiscard]] static enum geist_status vulkan_ffn_geglu_block(
    struct geist_backend *be,
    const struct geist_backend_ffn_geglu_block *block) {

    if (be == nullptr || be->state == nullptr || block == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    if (block->struct_size < sizeof(*block) ||
        block->seq == 0 ||
        block->d_model == 0 ||
        block->inter == 0) {
        return GEIST_E_INVALID_ARG;
    }

    const size_t rows = block->seq;
    const size_t d = block->d_model;
    const size_t inter = block->inter;
    const bool has_post_norm = block->post_ffw_norm_weight != nullptr;
    if (!vulkan_ffn_f32_2d_exact(block->residual, rows, d) ||
        !vulkan_ffn_f32_1d_exact(block->ffn_norm_weight, d) ||
        !vulkan_ffn_weight_2d_shape(block->gate_weight, inter, d) ||
        !vulkan_ffn_weight_2d_shape(block->up_weight, inter, d) ||
        !vulkan_ffn_weight_2d_shape(block->down_weight, d, inter) ||
        !vulkan_ffn_f32_2d_exact(block->pre_ff_scratch, rows, d) ||
        !vulkan_ffn_f32_2d_exact(block->gate_scratch, rows, inter) ||
        !vulkan_ffn_f32_2d_exact(block->up_scratch, rows, inter) ||
        !vulkan_ffn_f32_2d_exact(block->ffn_out_scratch, rows, d) ||
        !vulkan_ffn_f32_2d_exact(block->out, rows, d) ||
        (has_post_norm &&
         (!vulkan_ffn_f32_1d_exact(block->post_ffw_norm_weight, d) ||
          !vulkan_ffn_f32_2d_exact(block->post_ff_scratch, rows, d)))) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan ffn_geglu_block: tensor shape mismatch");
        return GEIST_E_INVALID_ARG;
    }
    if (!vulkan_ffn_weight_supported(block->gate_weight) ||
        !vulkan_ffn_weight_supported(block->up_weight) ||
        !vulkan_ffn_weight_supported(block->down_weight)) {
        geist_backend_set_error(be, GEIST_E_UNSUPPORTED,
                                "vulkan ffn_geglu_block: unsupported weight dtype/layout");
        return GEIST_E_UNSUPPORTED;
    }

    return vulkan_ffn_geglu_block_recorded(be, block);
}

[[nodiscard]] static enum geist_status vulkan_create(
    struct geist_backend *be,
    const struct geist_backend_opts *opts) {

    (void) opts;
    struct vulkan_state *st =
        geist_backend_alloc(be, sizeof(*st), alignof(struct vulkan_state));
    if (st == nullptr) {
        geist_backend_set_error(be, GEIST_E_OOM,
                                "vulkan: failed to allocate %zu-byte state",
                                sizeof(*st));
        return GEIST_E_OOM;
    }
    *st = (struct vulkan_state){0};
    st->backend = be;

    enum geist_status s = vulkan_load_loader(be, st);
    if (s != GEIST_OK) {
        geist_backend_free(be, st);
        return s;
    }

    s = vulkan_create_instance(be, st);
    if (s == GEIST_OK) {
        s = vulkan_choose_device(be, st);
    }
    if (s == GEIST_OK) {
        s = vulkan_create_device(be, st);
    }
    if (s != GEIST_OK) {
        vulkan_destroy_add_pipeline(st);
        vulkan_destroy_scale_pipeline(st);
        vulkan_destroy_mul_pipeline(st);
        vulkan_destroy_gelu_tanh_pipeline(st);
        vulkan_destroy_gelu_tanh_mul_pipeline(st);
        vulkan_destroy_rmsnorm_pipeline(st);
        vulkan_destroy_rmsnorm_add_pipeline(st);
        vulkan_destroy_rmsnorm_rope_dyn_pipeline(st);
        vulkan_destroy_matmul_f32_pipeline(st);
        vulkan_destroy_matvec_pipeline(st);
        vulkan_destroy_matvec_f16bf16_pipeline(st);
        vulkan_destroy_matmul_q4k_pipeline(st);
        vulkan_destroy_matmul_q4k_m8_pipeline(st);
        vulkan_destroy_matvec_q4k_pipeline(st);
        vulkan_destroy_matmul_q6k_pipeline(st);
        vulkan_destroy_matvec_q6k_pipeline(st);
        vulkan_destroy_argmax_batch_pipeline(st);
        vulkan_destroy_argmax_pipeline(st);
        vulkan_destroy_embed_lookup_scaled_pipeline(st);
        vulkan_destroy_embed_lookup_scaled_dyn_pipeline(st);
        vulkan_destroy_rope_pipeline(st);
        vulkan_destroy_rope_dyn_pipeline(st);
        vulkan_destroy_attention_pipeline(st);
        vulkan_destroy_attention_dyn_pipeline(st);
        vulkan_destroy_decode_replay_command_pool(st);
        vulkan_destroy_q4k_w4a8_resources(be);
        vulkan_destroy_decode_params_resources(be);
        vulkan_destroy_profile_query_pool(st);
        vulkan_destroy_transient_command_pool(st);
        vulkan_destroy_transient_descriptor_pool(st);
        vulkan_destroy_block_descriptor_pool(st);
        if (st->pipeline_cache != VK_NULL_HANDLE) {
            st->vkDestroyPipelineCache(st->device, st->pipeline_cache, nullptr);
        }
        if (st->command_pool != VK_NULL_HANDLE) {
            st->vkDestroyCommandPool(st->device, st->command_pool, nullptr);
        }
        if (st->device != VK_NULL_HANDLE) {
            st->vkDestroyDevice(st->device, nullptr);
        }
        if (st->instance != VK_NULL_HANDLE) {
            st->vkDestroyInstance(st->instance, nullptr);
        }
        if (st->loader_handle != nullptr) {
            dlclose(st->loader_handle);
        }
        geist_backend_free(be, st);
        return s;
    }

    be->state = st;
    return GEIST_OK;
}

static void vulkan_destroy(struct geist_backend *be) {
    if (be == nullptr || be->state == nullptr) {
        return;
    }
    struct vulkan_state *st = be->state;
    if (st->device != VK_NULL_HANDLE) {
        st->vkQueueWaitIdle(st->queue);
    }
    vulkan_profile_print_summary(st);
    vulkan_destroy_q4k_nt4_cache(be);
    vulkan_destroy_q4k_w4a8_resources(be);
    vulkan_destroy_decode_replay_command_pool(st);
    vulkan_destroy_argmax_runtime_resources(be);
    vulkan_destroy_decode_params_resources(be);
    vulkan_destroy_add_pipeline(st);
    vulkan_destroy_scale_pipeline(st);
    vulkan_destroy_mul_pipeline(st);
    vulkan_destroy_gelu_tanh_pipeline(st);
    vulkan_destroy_gelu_tanh_mul_pipeline(st);
    vulkan_destroy_rmsnorm_pipeline(st);
    vulkan_destroy_rmsnorm_add_pipeline(st);
    vulkan_destroy_rmsnorm_rope_dyn_pipeline(st);
    vulkan_destroy_matmul_f32_pipeline(st);
    vulkan_destroy_matvec_pipeline(st);
    vulkan_destroy_matvec_f16bf16_pipeline(st);
    vulkan_destroy_matmul_q4k_pipeline(st);
    vulkan_destroy_matmul_q4k_m8_pipeline(st);
    vulkan_destroy_matvec_q4k_pipeline(st);
    vulkan_destroy_matmul_q6k_pipeline(st);
    vulkan_destroy_matvec_q6k_pipeline(st);
    vulkan_destroy_argmax_batch_pipeline(st);
    vulkan_destroy_argmax_pipeline(st);
    vulkan_destroy_embed_lookup_scaled_pipeline(st);
    vulkan_destroy_embed_lookup_scaled_dyn_pipeline(st);
    vulkan_destroy_rope_pipeline(st);
    vulkan_destroy_rope_dyn_pipeline(st);
    vulkan_destroy_attention_pipeline(st);
    vulkan_destroy_attention_dyn_pipeline(st);
    vulkan_destroy_kv_append_dyn_pipeline(st);
    vulkan_destroy_profile_query_pool(st);
    vulkan_destroy_transient_command_pool(st);
    vulkan_destroy_transient_descriptor_pool(st);
    vulkan_destroy_block_descriptor_pool(st);
    if (st->pipeline_cache != VK_NULL_HANDLE) {
        st->vkDestroyPipelineCache(st->device, st->pipeline_cache, nullptr);
        st->pipeline_cache = VK_NULL_HANDLE;
    }
    if (st->command_pool != VK_NULL_HANDLE) {
        st->vkDestroyCommandPool(st->device, st->command_pool, nullptr);
        st->command_pool = VK_NULL_HANDLE;
    }
    if (st->device != VK_NULL_HANDLE) {
        st->vkDestroyDevice(st->device, nullptr);
        st->device = VK_NULL_HANDLE;
    }
    if (st->instance != VK_NULL_HANDLE) {
        st->vkDestroyInstance(st->instance, nullptr);
        st->instance = VK_NULL_HANDLE;
    }
    if (st->loader_handle != nullptr) {
        dlclose(st->loader_handle);
        st->loader_handle = nullptr;
    }
    geist_backend_free(be, be->state);
    be->state = nullptr;
}

static enum geist_support vulkan_supports_op(
    struct geist_backend *be,
    const struct geist_op_support_query *query) {

    (void) be;
    if (query == nullptr) {
        return GEIST_SUPPORT_NONE;
    }
    if (query->op == GEIST_OP_LINEAR &&
        query->input_count >= 2 &&
        query->output_count >= 1 &&
        query->inputs[0].dtype == GEIST_DTYPE_F32 &&
        query->inputs[0].layout == GEIST_LAYOUT_DENSE &&
        query->outputs[0].dtype == GEIST_DTYPE_F32 &&
        query->outputs[0].layout == GEIST_LAYOUT_DENSE) {
        if (query->inputs[1].dtype == GEIST_DTYPE_F32 &&
            query->inputs[1].layout == GEIST_LAYOUT_DENSE) {
            return GEIST_SUPPORT_NATIVE;
        }
        if (query->inputs[1].dtype == GEIST_DTYPE_Q4_K &&
            query->inputs[1].layout == GEIST_LAYOUT_BLOCK_QUANTIZED) {
            return GEIST_SUPPORT_NATIVE;
        }
        if (query->inputs[1].dtype == GEIST_DTYPE_Q6_K &&
            query->inputs[1].layout == GEIST_LAYOUT_BLOCK_QUANTIZED) {
            return GEIST_SUPPORT_NATIVE;
        }
    }
    if (query->op == GEIST_OP_RESIDUAL_ADD &&
        query->input_count >= 2 &&
        query->output_count >= 1 &&
        query->inputs[0].dtype == GEIST_DTYPE_F32 &&
        query->inputs[0].layout == GEIST_LAYOUT_DENSE &&
        query->inputs[1].dtype == GEIST_DTYPE_F32 &&
        query->inputs[1].layout == GEIST_LAYOUT_DENSE &&
        query->outputs[0].dtype == GEIST_DTYPE_F32 &&
        query->outputs[0].layout == GEIST_LAYOUT_DENSE) {
        return GEIST_SUPPORT_NATIVE;
    }
    if (query->op == GEIST_OP_RMSNORM &&
        query->input_count >= 2 &&
        query->output_count >= 1 &&
        query->inputs[0].dtype == GEIST_DTYPE_F32 &&
        query->inputs[0].layout == GEIST_LAYOUT_DENSE &&
        query->inputs[1].dtype == GEIST_DTYPE_F32 &&
        query->inputs[1].layout == GEIST_LAYOUT_DENSE &&
        query->outputs[0].dtype == GEIST_DTYPE_F32 &&
        query->outputs[0].layout == GEIST_LAYOUT_DENSE) {
        return GEIST_SUPPORT_NATIVE;
    }
    if (query->op == GEIST_OP_EMBEDDING_LOOKUP &&
        query->input_count >= 1 &&
        query->output_count >= 1 &&
        query->outputs[0].dtype == GEIST_DTYPE_F32 &&
        query->outputs[0].layout == GEIST_LAYOUT_DENSE) {
        if (query->inputs[0].dtype == GEIST_DTYPE_F32 &&
            query->inputs[0].layout == GEIST_LAYOUT_DENSE) {
            return GEIST_SUPPORT_NATIVE;
        }
        if (query->inputs[0].dtype == GEIST_DTYPE_Q4_K &&
            query->inputs[0].layout == GEIST_LAYOUT_BLOCK_QUANTIZED) {
            return GEIST_SUPPORT_NATIVE;
        }
        if (query->inputs[0].dtype == GEIST_DTYPE_Q6_K &&
            query->inputs[0].layout == GEIST_LAYOUT_BLOCK_QUANTIZED) {
            return GEIST_SUPPORT_NATIVE;
        }
    }
    if (query->op == GEIST_OP_ROPE &&
        query->input_count >= 1 &&
        query->output_count >= 1 &&
        query->inputs[0].dtype == GEIST_DTYPE_F32 &&
        query->inputs[0].layout == GEIST_LAYOUT_DENSE &&
        query->outputs[0].dtype == GEIST_DTYPE_F32 &&
        query->outputs[0].layout == GEIST_LAYOUT_DENSE) {
        return GEIST_SUPPORT_NATIVE;
    }
    if (query->op == GEIST_OP_ATTENTION &&
        query->input_count >= 3 &&
        query->output_count >= 1 &&
        query->inputs[0].dtype == GEIST_DTYPE_F32 &&
        query->inputs[0].layout == GEIST_LAYOUT_DENSE &&
        query->inputs[1].dtype == GEIST_DTYPE_F32 &&
        query->inputs[1].layout == GEIST_LAYOUT_DENSE &&
        query->inputs[2].dtype == GEIST_DTYPE_F32 &&
        query->inputs[2].layout == GEIST_LAYOUT_DENSE &&
        query->outputs[0].dtype == GEIST_DTYPE_F32 &&
        query->outputs[0].layout == GEIST_LAYOUT_DENSE) {
        return GEIST_SUPPORT_NATIVE;
    }
    return GEIST_SUPPORT_NONE;
}

[[nodiscard]] static enum geist_status vulkan_query_accel_caps(
    struct geist_backend *be,
    struct geist_backend_accel_caps *out) {

    if (be == nullptr || be->state == nullptr || out == nullptr) {
        return GEIST_E_INVALID_ARG;
    }
    const size_t caller_size = out->struct_size;
    if (caller_size < sizeof(*out)) {
        geist_backend_set_error(be, GEIST_E_INVALID_ARG,
                                "vulkan: accel caps struct too small (%zu < %zu)",
                                caller_size, sizeof(*out));
        return GEIST_E_INVALID_ARG;
    }

    struct vulkan_state *st = be->state;
    *out = (struct geist_backend_accel_caps){
        .struct_size = sizeof(*out),
        .device_resident_buffers = true,
        .compute_queue = st->has_compute_queue,
        .pipeline_cache = true,
        .subgroup_basic = st->has_subgroup_basic,
        .shader_integer_dot_product = st->has_shader_integer_dot_product,
        .descriptor_indexing = st->has_descriptor_indexing,
        .timeline_semaphore = st->has_timeline_semaphore,
        .device_local_bytes = vulkan_device_local_bytes(st),
    };
    snprintf(out->device_name, sizeof(out->device_name), "%s",
             st->device_props.deviceName);
    return GEIST_OK;
}

static const struct geist_backend_vtbl vulkan_vtbl = {
    .create = vulkan_create,
    .destroy = vulkan_destroy,
    .supports_op = vulkan_supports_op,
    .query_accel_caps = vulkan_query_accel_caps,
    .buffer_create = vulkan_buffer_create,
    .buffer_destroy = vulkan_buffer_destroy,
    .buffer_create_aliased = nullptr,
    .buffer_upload = vulkan_buffer_upload,
    .buffer_download = vulkan_buffer_download,
    .buffer_copy = vulkan_buffer_copy,
    .buffer_map = vulkan_buffer_map,
    .buffer_unmap = vulkan_buffer_unmap,
    .prepare_weight_layout = vulkan_prepare_weight_layout,
    .matvec_f32_dense = vulkan_matvec_f32_dense,
    .matmul_f32_dense = vulkan_matmul_f32_dense,
    .matvec_q4k = vulkan_matvec_q4k,
    .matmul_q4k = vulkan_matmul_q4k,
    .matvec_q6k = vulkan_matvec_q6k,
    .matmul_q6k = vulkan_matmul_q6k,
    .argmax_f32 = vulkan_argmax_f32,
    .argmax_f32_batch = vulkan_argmax_f32_batch,
    .rope_apply = vulkan_rope_apply,
    .embedding_lookup = vulkan_embedding_lookup,
    .embedding_lookup_scaled = vulkan_embedding_lookup_scaled,
    .attention = vulkan_attention,
    .add = vulkan_add,
    .mul = vulkan_mul,
    .scale_f32 = vulkan_scale_f32,
    .rmsnorm = vulkan_rmsnorm,
    .gelu_tanh = vulkan_gelu_tanh,
    .gelu_tanh_mul = vulkan_gelu_tanh_mul,
    .ffn_geglu_block = vulkan_ffn_geglu_block,
    .attention_block = vulkan_attention_block,
    .attention_query_block = vulkan_attention_query_block,
    .ple_block = vulkan_ple_block,
    .greedy_head = vulkan_greedy_head,
    .greedy_head_batch = vulkan_greedy_head_batch,
    .command_sequence_begin = vulkan_command_sequence_begin,
    .command_sequence_end = vulkan_command_sequence_end,
    .command_sequence_read_token = vulkan_command_sequence_read_token,
    .command_sequence_read_tokens = vulkan_command_sequence_read_tokens,
    .command_sequence_replay_decode_greedy_step =
        vulkan_command_sequence_replay_decode_greedy_step,
};

const struct geist_backend_descriptor geist_backend_vulkan = {
    .name = "vulkan",
    .vtbl = &vulkan_vtbl,
    .caps = nullptr,
    .n_caps = 0,
};
