#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct alignas(16) KernelParams {
  uint32_t a;
  uint32_t b;
  uint32_t slot;
  uint32_t tag;
};

struct Options {
  std::vector<size_t> command_counts{242, 640};
  size_t repetitions = 1000;
  size_t warmup_repetitions = 5;
  std::string json_out;
  bool acknowledge_gpu_use = false;
};

struct Timings {
  double host_encode_us = 0.0;
  double wall_us = 0.0;
  double gpu_us = 0.0;
  uint64_t command_buffers = 0;
};

struct ReadMismatch {
  size_t slot = 0;
  uint32_t expected = 0;
  uint32_t actual = 0;
};

struct Validation {
  uint32_t expected_state = 0;
  uint32_t actual_state = 0;
  uint32_t expected_counter = 0;
  uint32_t actual_counter = 0;
  size_t read_mismatches = 0;
  std::vector<ReadMismatch> mismatch_examples;

  bool exact() const {
    return expected_state == actual_state &&
           expected_counter == actual_counter && read_mismatches == 0;
  }
};

struct CaseResult {
  size_t command_count = 0;
  size_t read_slots = 0;
  size_t repetitions = 0;
  size_t warmup_repetitions = 0;
  double icb_preencode_us = 0.0;
  size_t icb_size_bytes = 0;
  Timings direct;
  Timings icb;
  Validation direct_validation;
  Validation icb_validation;
  uint64_t icb_encoded_commands = 0;
  uint64_t icb_execute_calls = 0;

  bool passed() const {
    return direct_validation.exact() && icb_validation.exact() &&
           icb_size_bytes > 0 && icb_encoded_commands == command_count &&
           icb_execute_calls == repetitions;
  }
};

struct Pipelines {
  id<MTLComputePipelineState> write;
  id<MTLComputePipelineState> transform;
  id<MTLComputePipelineState> read;
};

struct Buffers {
  id<MTLBuffer> state;
  id<MTLBuffer> counter;
  id<MTLBuffer> params;
  id<MTLBuffer> reads;
};

constexpr uint32_t kInitialState = 0x12345678u;
constexpr uint32_t kReadSentinel = 0xDEADBEEFu;

double micros(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}

std::string jsonEscape(const std::string &value) {
  std::ostringstream out;
  for (unsigned char c : value) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (c < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<unsigned>(c) << std::dec;
        } else {
          out << c;
        }
    }
  }
  return out.str();
}

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

size_t parsePositive(const std::string &text, const char *name) {
  size_t consumed = 0;
  unsigned long long value = 0;
  try {
    value = std::stoull(text, &consumed);
  } catch (...) {
    fail(std::string("invalid ") + name + ": " + text);
  }
  if (consumed != text.size() || value == 0) {
    fail(std::string("invalid ") + name + ": " + text);
  }
  return static_cast<size_t>(value);
}

std::vector<size_t> parseCommandCounts(const std::string &text) {
  std::vector<size_t> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    values.push_back(parsePositive(item, "command count"));
  }
  if (values.empty()) {
    fail("--commands requires at least one count");
  }
  return values;
}

void printUsage(const char *program) {
  std::cout
      << "Usage: " << program << " --acknowledge-gpu-use [options]\n"
      << "\n"
      << "Options:\n"
      << "  --commands 242,640       Exact ICB command counts (default: 242,640)\n"
      << "  --repetitions N          Timed replay count (default: 1000)\n"
      << "  --warmup-repetitions N   Untimed replay count (default: 5)\n"
      << "  --json-out PATH          Also write the JSON result to PATH\n"
      << "  --acknowledge-gpu-use    Required fail-closed GPU acknowledgement\n"
      << "  --help                   Show this text without creating a Metal device\n";
}

Options parseOptions(int argc, const char *argv[]) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto requireValue = [&](const char *name) -> std::string {
      if (++i >= argc) fail(std::string(name) + " requires a value");
      return argv[i];
    };
    if (arg == "--commands") {
      options.command_counts = parseCommandCounts(requireValue("--commands"));
    } else if (arg == "--repetitions") {
      options.repetitions = parsePositive(requireValue("--repetitions"), "repetitions");
    } else if (arg == "--warmup-repetitions") {
      options.warmup_repetitions = parsePositive(
          requireValue("--warmup-repetitions"), "warmup repetitions");
    } else if (arg == "--json-out") {
      options.json_out = requireValue("--json-out");
    } else if (arg == "--acknowledge-gpu-use") {
      options.acknowledge_gpu_use = true;
    } else if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    } else {
      fail("unknown argument: " + arg);
    }
  }
  return options;
}

std::vector<KernelParams> makeParams(size_t command_count, size_t *read_slots) {
  std::vector<KernelParams> params;
  params.reserve(command_count);
  uint32_t slot = 0;
  for (size_t i = 0; i < command_count; ++i) {
    const uint32_t index = static_cast<uint32_t>(i);
    KernelParams value{};
    switch (i % 3) {
      case 0:
        value.a = 0x9E3779B9u ^ (index * 0x45D9F3Bu);
        break;
      case 1:
        value.a = (0x0019660Du + index * 2u) | 1u;
        value.b = 0x3C6EF35Fu ^ (index * 0x27D4EB2Du);
        break;
      default:
        value.slot = slot++;
        value.tag = 0xA5A5A5A5u ^ (index * 0x9E3779B9u);
        break;
    }
    params.push_back(value);
  }
  *read_slots = slot;
  return params;
}

Validation expectedFor(const std::vector<KernelParams> &params,
                       size_t read_slots, size_t repetitions,
                       std::vector<uint32_t> *expected_reads) {
  uint32_t state = kInitialState;
  expected_reads->assign(std::max<size_t>(1, read_slots), kReadSentinel);
  for (size_t repetition = 0; repetition < repetitions; ++repetition) {
    for (size_t i = 0; i < params.size(); ++i) {
      const auto &p = params[i];
      switch (i % 3) {
        case 0: state += p.a; break;
        case 1: state = state * p.a + p.b; break;
        default: (*expected_reads)[p.slot] = state ^ p.tag; break;
      }
    }
  }
  const uint64_t expected_counter = params.size() * repetitions;
  if (expected_counter > UINT32_MAX) fail("mechanism counter would overflow uint32");
  Validation result;
  result.expected_state = state;
  result.expected_counter = static_cast<uint32_t>(expected_counter);
  return result;
}

void resetBuffers(const Buffers &buffers, size_t read_slots) {
  *static_cast<uint32_t *>(buffers.state.contents) = kInitialState;
  *static_cast<uint32_t *>(buffers.counter.contents) = 0;
  auto *reads = static_cast<uint32_t *>(buffers.reads.contents);
  std::fill(reads, reads + std::max<size_t>(1, read_slots), kReadSentinel);
}

Validation validateBuffers(const Buffers &buffers,
                           const std::vector<KernelParams> &params,
                           size_t read_slots, size_t repetitions) {
  std::vector<uint32_t> expected_reads;
  Validation result = expectedFor(params, read_slots, repetitions, &expected_reads);
  result.actual_state = *static_cast<uint32_t *>(buffers.state.contents);
  result.actual_counter = *static_cast<uint32_t *>(buffers.counter.contents);
  const auto *actual_reads = static_cast<const uint32_t *>(buffers.reads.contents);
  for (size_t i = 0; i < expected_reads.size(); ++i) {
    if (actual_reads[i] != expected_reads[i]) {
      ++result.read_mismatches;
      if (result.mismatch_examples.size() < 8) {
        result.mismatch_examples.push_back({i, expected_reads[i], actual_reads[i]});
      }
    }
  }
  return result;
}

id<MTLComputePipelineState> makePipeline(id<MTLDevice> device,
                                         id<MTLLibrary> library,
                                         NSString *function_name) {
  id<MTLFunction> function = [library newFunctionWithName:function_name];
  if (!function) fail("Metal function missing: " + std::string(function_name.UTF8String));
  MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
  descriptor.label = [@"ICB P0 " stringByAppendingString:function_name];
  descriptor.computeFunction = function;
  descriptor.supportIndirectCommandBuffers = YES;
  NSError *error = nil;
  id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithDescriptor:descriptor
                                             options:MTLPipelineOptionNone
                                          reflection:nil
                                               error:&error];
  if (!pipeline) {
    fail("pipeline creation failed for " + std::string(function_name.UTF8String) +
         ": " + std::string(error.localizedDescription.UTF8String));
  }
  if (!pipeline.supportIndirectCommandBuffers) {
    fail("pipeline did not preserve supportIndirectCommandBuffers for " +
         std::string(function_name.UTF8String));
  }
  return pipeline;
}

Pipelines makePipelines(id<MTLDevice> device) {
  static NSString *source = @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Params {
  uint a;
  uint b;
  uint slot;
  uint tag;
};

kernel void write_stage(device uint *state [[buffer(0)]],
                        device atomic_uint *counter [[buffer(1)]],
                        constant Params &p [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {
  if (tid == 0) {
    state[0] += p.a;
    atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
  }
}

kernel void transform_stage(device uint *state [[buffer(0)]],
                            device atomic_uint *counter [[buffer(1)]],
                            constant Params &p [[buffer(2)]],
                            uint tid [[thread_position_in_grid]]) {
  if (tid == 0) {
    state[0] = state[0] * p.a + p.b;
    atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
  }
}

kernel void read_stage(device uint *state [[buffer(0)]],
                       device atomic_uint *counter [[buffer(1)]],
                       constant Params &p [[buffer(2)]],
                       device uint *reads [[buffer(3)]],
                       uint tid [[thread_position_in_grid]]) {
  if (tid == 0) {
    reads[p.slot] = state[0] ^ p.tag;
    atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
  }
}
)METAL";
  MTLCompileOptions *options = [MTLCompileOptions new];
  options.fastMathEnabled = NO;
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
  if (!library) {
    fail("Metal source compilation failed: " +
         std::string(error.localizedDescription.UTF8String));
  }
  return {makePipeline(device, library, @"write_stage"),
          makePipeline(device, library, @"transform_stage"),
          makePipeline(device, library, @"read_stage")};
}

Buffers makeBuffers(id<MTLDevice> device, const std::vector<KernelParams> &params,
                    size_t read_slots) {
  const MTLResourceOptions options = MTLResourceStorageModeShared;
  Buffers buffers;
  buffers.state = [device newBufferWithLength:sizeof(uint32_t) options:options];
  buffers.counter = [device newBufferWithLength:sizeof(uint32_t) options:options];
  buffers.params = [device newBufferWithLength:params.size() * sizeof(KernelParams)
                                       options:options];
  buffers.reads = [device newBufferWithLength:std::max<size_t>(1, read_slots) * sizeof(uint32_t)
                                      options:options];
  if (!buffers.state || !buffers.counter || !buffers.params || !buffers.reads) {
    fail("failed to allocate shared Metal buffers");
  }
  std::memcpy(buffers.params.contents, params.data(), params.size() * sizeof(KernelParams));
  resetBuffers(buffers, read_slots);
  return buffers;
}

id<MTLComputePipelineState> pipelineFor(const Pipelines &pipelines, size_t index) {
  switch (index % 3) {
    case 0: return pipelines.write;
    case 1: return pipelines.transform;
    default: return pipelines.read;
  }
}

void bindDirect(id<MTLComputeCommandEncoder> encoder, const Buffers &buffers,
                size_t index) {
  [encoder setBuffer:buffers.state offset:0 atIndex:0];
  [encoder setBuffer:buffers.counter offset:0 atIndex:1];
  [encoder setBuffer:buffers.params offset:index * sizeof(KernelParams) atIndex:2];
  if (index % 3 == 2) [encoder setBuffer:buffers.reads offset:0 atIndex:3];
}

void bindIndirect(id<MTLIndirectComputeCommand> command, const Buffers &buffers,
                  size_t index) {
  [command setKernelBuffer:buffers.state offset:0 atIndex:0];
  [command setKernelBuffer:buffers.counter offset:0 atIndex:1];
  [command setKernelBuffer:buffers.params offset:index * sizeof(KernelParams) atIndex:2];
  if (index % 3 == 2) [command setKernelBuffer:buffers.reads offset:0 atIndex:3];
}

void checkCommandBuffer(id<MTLCommandBuffer> command_buffer, const char *lane) {
  if (command_buffer.status == MTLCommandBufferStatusError) {
    std::string detail = command_buffer.error
        ? command_buffer.error.localizedDescription.UTF8String : "unknown error";
    fail(std::string(lane) + " command buffer failed: " + detail);
  }
}

Timings runDirect(id<MTLCommandQueue> queue, const Pipelines &pipelines,
                  const Buffers &buffers, size_t command_count,
                  size_t repetitions) {
  Timings result;
  const auto wall_start = Clock::now();
  for (size_t repetition = 0; repetition < repetitions; ++repetition) {
    const auto encode_start = Clock::now();
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    if (!command_buffer || !encoder) fail("failed to create direct command encoder");
    encoder.label = @"ICB P0 direct baseline";
    for (size_t i = 0; i < command_count; ++i) {
      [encoder setComputePipelineState:pipelineFor(pipelines, i)];
      bindDirect(encoder, buffers, i);
      [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      if (i + 1 < command_count) {
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }
    }
    [encoder endEncoding];
    result.host_encode_us += micros(encode_start, Clock::now());
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    checkCommandBuffer(command_buffer, "direct");
    if (command_buffer.GPUEndTime >= command_buffer.GPUStartTime) {
      result.gpu_us += (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1.0e6;
    }
    ++result.command_buffers;
  }
  result.wall_us = micros(wall_start, Clock::now());
  return result;
}

id<MTLIndirectCommandBuffer> makeICB(id<MTLDevice> device,
                                     const Pipelines &pipelines,
                                     const Buffers &buffers,
                                     size_t command_count,
                                     double *preencode_us) {
  MTLIndirectCommandBufferDescriptor *descriptor =
      [MTLIndirectCommandBufferDescriptor new];
  descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatchThreads;
  descriptor.inheritPipelineState = NO;
  descriptor.inheritBuffers = NO;
  descriptor.maxKernelBufferBindCount = 4;
  id<MTLIndirectCommandBuffer> icb =
      [device newIndirectCommandBufferWithDescriptor:descriptor
                                      maxCommandCount:command_count
                                                options:MTLResourceStorageModeShared];
  if (!icb) fail("device rejected a compute indirect command buffer");
  const auto start = Clock::now();
  for (size_t i = 0; i < command_count; ++i) {
    id<MTLIndirectComputeCommand> command = [icb indirectComputeCommandAtIndex:i];
    if (!command) fail("indirectComputeCommandAtIndex returned nil");
    // A command's barrier waits for commands before it and must be set first.
    if (i > 0) [command setBarrier];
    [command setComputePipelineState:pipelineFor(pipelines, i)];
    bindIndirect(command, buffers, i);
    [command concurrentDispatchThreads:MTLSizeMake(1, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  }
  *preencode_us = micros(start, Clock::now());
  if (icb.size == 0) fail("indirect command buffer reports zero size after encoding");
  return icb;
}

Timings runICB(id<MTLCommandQueue> queue, id<MTLIndirectCommandBuffer> icb,
               const Buffers &buffers, size_t command_count,
               size_t repetitions) {
  Timings result;
  const auto wall_start = Clock::now();
  for (size_t repetition = 0; repetition < repetitions; ++repetition) {
    const auto encode_start = Clock::now();
    id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command_buffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    if (!command_buffer || !encoder) fail("failed to create ICB replay encoder");
    encoder.label = @"ICB P0 reusable replay";
    [encoder useResource:buffers.state usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [encoder useResource:buffers.counter usage:MTLResourceUsageRead | MTLResourceUsageWrite];
    [encoder useResource:buffers.params usage:MTLResourceUsageRead];
    [encoder useResource:buffers.reads usage:MTLResourceUsageWrite];
    [encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0, command_count)];
    [encoder endEncoding];
    result.host_encode_us += micros(encode_start, Clock::now());
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    checkCommandBuffer(command_buffer, "ICB");
    if (command_buffer.GPUEndTime >= command_buffer.GPUStartTime) {
      result.gpu_us += (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1.0e6;
    }
    ++result.command_buffers;
  }
  result.wall_us = micros(wall_start, Clock::now());
  return result;
}

CaseResult runCase(id<MTLDevice> device, id<MTLCommandQueue> queue,
                   const Pipelines &pipelines, size_t command_count,
                   size_t repetitions, size_t warmup_repetitions) {
  CaseResult result;
  result.command_count = command_count;
  result.repetitions = repetitions;
  result.warmup_repetitions = warmup_repetitions;
  auto params = makeParams(command_count, &result.read_slots);

  Buffers direct_buffers = makeBuffers(device, params, result.read_slots);
  runDirect(queue, pipelines, direct_buffers, command_count, warmup_repetitions);
  resetBuffers(direct_buffers, result.read_slots);
  result.direct = runDirect(queue, pipelines, direct_buffers, command_count, repetitions);
  result.direct_validation = validateBuffers(
      direct_buffers, params, result.read_slots, repetitions);

  Buffers icb_buffers = makeBuffers(device, params, result.read_slots);
  id<MTLIndirectCommandBuffer> icb = makeICB(
      device, pipelines, icb_buffers, command_count, &result.icb_preencode_us);
  runICB(queue, icb, icb_buffers, command_count, warmup_repetitions);
  resetBuffers(icb_buffers, result.read_slots);
  result.icb = runICB(queue, icb, icb_buffers, command_count, repetitions);
  result.icb_validation = validateBuffers(
      icb_buffers, params, result.read_slots, repetitions);
  result.icb_size_bytes = icb.size;
  result.icb_encoded_commands = command_count;
  result.icb_execute_calls = repetitions;

  return result;
}

void emitValidation(std::ostream &out, const Validation &validation) {
  out << "{\"exact\":" << (validation.exact() ? "true" : "false")
      << ",\"expected_state\":" << validation.expected_state
      << ",\"actual_state\":" << validation.actual_state
      << ",\"expected_gpu_commands\":" << validation.expected_counter
      << ",\"observed_gpu_commands\":" << validation.actual_counter
      << ",\"read_mismatches\":" << validation.read_mismatches
      << ",\"mismatch_examples\":[";
  for (size_t i = 0; i < validation.mismatch_examples.size(); ++i) {
    const auto &mismatch = validation.mismatch_examples[i];
    out << "{\"slot\":" << mismatch.slot
        << ",\"expected\":" << mismatch.expected
        << ",\"actual\":" << mismatch.actual << "}"
        << (i + 1 == validation.mismatch_examples.size() ? "" : ",");
  }
  out << "]}";
}

void emitFailureDiagnostics(const char *lane, size_t command_count,
                            const Validation &validation) {
  if (validation.exact()) return;
  std::cerr << "VALIDATION_FAIL lane=" << lane
            << " commands=" << command_count
            << " expected_state=" << validation.expected_state
            << " actual_state=" << validation.actual_state
            << " expected_gpu_commands=" << validation.expected_counter
            << " observed_gpu_commands=" << validation.actual_counter
            << " read_mismatches=" << validation.read_mismatches << "\n";
  for (const auto &mismatch : validation.mismatch_examples) {
    std::cerr << "READ_MISMATCH lane=" << lane
              << " commands=" << command_count
              << " slot=" << mismatch.slot
              << " expected=" << mismatch.expected
              << " actual=" << mismatch.actual << "\n";
  }
}

void emitTimings(std::ostream &out, const Timings &timings,
                 size_t command_count, size_t repetitions) {
  const double commands = static_cast<double>(command_count) * repetitions;
  out << "{\"host_encode_total_us\":" << timings.host_encode_us
      << ",\"host_encode_us_per_replay\":" << timings.host_encode_us / repetitions
      << ",\"host_encode_us_per_command\":" << timings.host_encode_us / commands
      << ",\"wall_total_us\":" << timings.wall_us
      << ",\"effective_wall_us_per_command\":" << timings.wall_us / commands
      << ",\"gpu_total_us\":" << timings.gpu_us
      << ",\"gpu_us_per_command\":" << timings.gpu_us / commands
      << ",\"command_buffers\":" << timings.command_buffers << "}";
}

std::string makeJSON(id<MTLDevice> device, const Options &options,
                     const std::vector<CaseResult> &results) {
  const bool passed = std::all_of(results.begin(), results.end(),
                                  [](const CaseResult &result) {
                                    return result.passed();
                                  });
  std::ostringstream out;
  out << std::setprecision(10);
  out << "{\n  \"schema\":\"mlx-uag.native-compute-icb-p0.v1\",\n"
      << "  \"device\":\"" << jsonEscape(device.name.UTF8String) << "\",\n"
      << "  \"repetitions\":" << options.repetitions << ",\n"
      << "  \"warmup_repetitions\":" << options.warmup_repetitions << ",\n"
      << "  \"passed\":" << (passed ? "true" : "false") << ",\n"
      << "  \"mechanism_proof\":{"
      << "\"pipeline_states_support_icb\":3,"
      << "\"descriptor_command_type\":\"concurrent_dispatch_threads\","
      << "\"fail_closed_on_zero_icb_size\":true,"
      << "\"fail_closed_on_counter_or_dependency_mismatch\":true},\n"
      << "  \"cases\":[\n";
  for (size_t i = 0; i < results.size(); ++i) {
    const auto &result = results[i];
    const double direct_host = result.direct.host_encode_us;
    const double icb_host = result.icb.host_encode_us;
    const double direct_wall = result.direct.wall_us;
    const double icb_wall = result.icb.wall_us;
    out << "    {\"command_count\":" << result.command_count
        << ",\"read_slots\":" << result.read_slots
        << ",\"icb_preencode_us\":" << result.icb_preencode_us
        << ",\"icb_size_bytes\":" << result.icb_size_bytes
        << ",\"icb_encoded_commands\":" << result.icb_encoded_commands
        << ",\"icb_execute_calls\":" << result.icb_execute_calls
        << ",\"passed\":" << (result.passed() ? "true" : "false")
        << ",\"direct_timed_barriers\":"
        << (result.command_count - 1) * result.repetitions
        << ",\"icb_preencoded_barriers\":" << result.command_count - 1
        << ",\"icb_timed_parent_barriers\":0"
        << ",\"icb_execution_ranges_per_replay\":1"
        << ",\"direct\":";
    emitTimings(out, result.direct, result.command_count, result.repetitions);
    out << ",\"icb\":";
    emitTimings(out, result.icb, result.command_count, result.repetitions);
    out << ",\"host_encode_speedup\":" << direct_host / icb_host
        << ",\"effective_wall_speedup\":" << direct_wall / icb_wall
        << ",\"direct_validation\":";
    emitValidation(out, result.direct_validation);
    out << ",\"icb_validation\":";
    emitValidation(out, result.icb_validation);
    out << "}" << (i + 1 == results.size() ? "\n" : ",\n");
  }
  out << "  ],\n  \"status\":\"" << (passed ? "PASS" : "FAIL") << "\"\n}\n";
  return out.str();
}

int run(const Options &options) {
  if (!options.acknowledge_gpu_use) {
    std::cerr << "Refusing GPU work: pass --acknowledge-gpu-use through the guarded run script.\n";
    return 2;
  }
  if (@available(macOS 11.0, *)) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) fail("MTLCreateSystemDefaultDevice returned nil");
    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) fail("failed to create Metal command queue");
    Pipelines pipelines = makePipelines(device);
    std::vector<CaseResult> results;
    for (size_t command_count : options.command_counts) {
      results.push_back(runCase(device, queue, pipelines, command_count,
                                options.repetitions, options.warmup_repetitions));
    }
    bool passed = true;
    for (const auto &result : results) {
      emitFailureDiagnostics("direct", result.command_count,
                             result.direct_validation);
      emitFailureDiagnostics("icb", result.command_count,
                             result.icb_validation);
      passed = passed && result.passed();
    }
    std::string json = makeJSON(device, options, results);
    std::cout << json;
    if (!options.json_out.empty()) {
      std::ofstream output(options.json_out);
      if (!output) fail("could not open --json-out path: " + options.json_out);
      output << json;
      if (!output) fail("could not write --json-out path: " + options.json_out);
    }
    return passed ? 0 : 1;
  }
  fail("compute ICB requires macOS 11 or newer");
}

}  // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    try {
      Options options = parseOptions(argc, argv);
      return run(options);
    } catch (const std::exception &error) {
      std::cerr << "FAIL: " << error.what() << "\n";
      return 1;
    }
  }
}
