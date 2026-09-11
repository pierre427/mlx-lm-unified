#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
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

constexpr size_t kCommands = 6;
constexpr size_t kTableStride = 12;
constexpr size_t kScheduleStride = 8;
constexpr size_t kScratchFloats = 112698;
constexpr NSUInteger kThreadgroups = 40;
constexpr NSUInteger kThreads = 512;
constexpr NSUInteger kQMVThreadgroupBytes = 2560 * sizeof(float);
constexpr NSUInteger kNormThreadgroupBytes = (16 + 4) * sizeof(float);
constexpr double kOracleRTol = 1.0e-4;
constexpr double kOracleATol = 1.0e-5;
constexpr const char *kManifestSchema =
    "mlx-uag.qwen4-phase-family-icb-p1.v1";
constexpr const char *kReceiptSchema =
    "mlx-uag.qwen4-phase-family-icb-p1-receipt.v1";

struct PhaseParams {
  uint32_t op;
  uint32_t entry;
  uint32_t src;
  uint32_t dst;
  uint32_t arg0;
  uint32_t arg1;
  uint32_t arg2;
  uint32_t barrier;
  uint32_t command;
};

static_assert(sizeof(PhaseParams) == 9 * sizeof(uint32_t));

constexpr std::array<std::array<uint32_t, kScheduleStride>, kCommands>
    kExpectedSchedule{{
        {{4, 0, 0, 20480, 10240, 2560, 0, 2}},
        {{1, 1, 20480, 30720, 4, 3, 0, 2}},
        {{1, 2, 30720, 31040, 4, 2, 0, 2}},
        {{5, UINT32_MAX, 31040, 41280, 4, 2560, 0, 2}},
        {{1, 3, 20480, 43840, 1, 4, 2, 1}},
        {{1, 4, 41280, 46404, 2, 0, 0, 0}},
    }};

struct OutputRange {
  const char *name;
  uint32_t offset;
  uint32_t floats;
};

constexpr std::array<OutputRange, kCommands> kExpectedOutputs{{
    {"NORMED", 20480, 10240},
    {"HC_LR", 30720, 320},
    {"HC_W", 31040, 10240},
    {"MIXED", 41280, 2560},
    {"INJECT", 43840, 4},
    {"GDN_QKV", 46404, 10240},
}};

struct Options {
  std::string artifact;
  std::string manifest_sha256;
  std::string receipt_out;
  std::string direct_scratch_out;
  std::string icb_scratch_out;
  size_t repetitions = 100;
  size_t warmups = 5;
  bool acknowledge_gpu_use = false;
  bool validate_only = false;
};

struct Timings {
  double host_encode_us = 0.0;
  double wall_us = 0.0;
  double gpu_us = 0.0;
  size_t submissions = 0;
};

struct Pipelines {
  id<MTLComputePipelineState> norm;
  id<MTLComputePipelineState> qmv;
  id<MTLComputePipelineState> mix;
};

struct LaneBuffers {
  id<MTLBuffer> scratch;
  id<MTLBuffer> receipts;
};

struct Inputs {
  NSDictionary *manifest;
  std::string manifest_digest;
  NSData *weight;
  NSData *table;
  NSData *schedule;
  NSData *scratch_input;
  NSData *scratch_expected;
  NSString *source;
  std::array<PhaseParams, kCommands> params;
};

struct AccuracyCheck {
  NSDictionary *json;
  bool passed;
  std::string failure;
};

[[noreturn]] void fail(const std::string &message) {
  throw std::runtime_error(message);
}

double micros(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}

NSString *ns(const std::string &value) {
  return [NSString stringWithUTF8String:value.c_str()];
}

std::string text(NSString *value) {
  return value ? std::string(value.UTF8String) : std::string();
}

NSDictionary *dictionary(id value, const char *where) {
  if (![value isKindOfClass:[NSDictionary class]]) {
    fail(std::string(where) + " must be a JSON object");
  }
  return (NSDictionary *)value;
}

NSArray *array(id value, const char *where) {
  if (![value isKindOfClass:[NSArray class]]) {
    fail(std::string(where) + " must be a JSON array");
  }
  return (NSArray *)value;
}

NSString *stringValue(id value, const char *where) {
  if (![value isKindOfClass:[NSString class]]) {
    fail(std::string(where) + " must be a string");
  }
  return (NSString *)value;
}

uint64_t integerValue(id value, const char *where) {
  if (![value isKindOfClass:[NSNumber class]]) {
    fail(std::string(where) + " must be an integer");
  }
  return [(NSNumber *)value unsignedLongLongValue];
}

double numberValue(id value, const char *where) {
  if (![value isKindOfClass:[NSNumber class]])
    fail(std::string(where) + " must be numeric");
  return [(NSNumber *)value doubleValue];
}

NSData *readData(NSString *path) {
  NSData *data = [NSData dataWithContentsOfFile:path];
  if (!data) fail("cannot read " + text(path));
  return data;
}

std::string sha256(NSData *data) {
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(data.bytes, static_cast<CC_LONG>(data.length), digest);
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (unsigned char value : digest) out << std::setw(2) << unsigned(value);
  return out.str();
}

void writeData(NSData *data, const std::string &path) {
  if (![data writeToFile:ns(path) options:NSDataWritingAtomic error:nil]) {
    fail("cannot write " + path);
  }
}

void writeJSON(NSDictionary *value, const std::string &path) {
  NSError *error = nil;
  NSData *data = [NSJSONSerialization dataWithJSONObject:value
                                                  options:NSJSONWritingPrettyPrinted
                                                    error:&error];
  if (!data) fail("cannot encode receipt JSON: " + text(error.localizedDescription));
  writeData(data, path);
}

size_t parsePositive(const std::string &value, const char *name) {
  size_t used = 0;
  unsigned long long parsed = 0;
  try {
    parsed = std::stoull(value, &used);
  } catch (...) {
    fail(std::string("invalid ") + name + ": " + value);
  }
  if (!parsed || used != value.size()) fail(std::string("invalid ") + name);
  return static_cast<size_t>(parsed);
}

void usage(const char *program) {
  std::cout
      << "Usage: " << program << " --acknowledge-gpu-use --artifact DIR\n"
      << "  --manifest-sha256 HEX      Required explicit manifest binding\n"
      << "  --receipt-out PATH         Native receipt JSON\n"
      << "  --direct-scratch-out PATH  Direct scratch image\n"
      << "  --icb-scratch-out PATH     ICB scratch image\n"
      << "  --repetitions N            Timed replays (default: 100)\n"
      << "  --warmups N                Untimed replays (default: 5)\n"
      << "  --validate-only            CPU-only artifact/ABI preflight\n";
}

Options parseOptions(int argc, const char *argv[]) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    auto next = [&](const char *name) {
      if (++i >= argc) fail(std::string(name) + " requires a value");
      return std::string(argv[i]);
    };
    if (arg == "--artifact") options.artifact = next("--artifact");
    else if (arg == "--manifest-sha256")
      options.manifest_sha256 = next("--manifest-sha256");
    else if (arg == "--receipt-out") options.receipt_out = next("--receipt-out");
    else if (arg == "--direct-scratch-out")
      options.direct_scratch_out = next("--direct-scratch-out");
    else if (arg == "--icb-scratch-out")
      options.icb_scratch_out = next("--icb-scratch-out");
    else if (arg == "--repetitions")
      options.repetitions = parsePositive(next("--repetitions"), "repetitions");
    else if (arg == "--warmups")
      options.warmups = parsePositive(next("--warmups"), "warmups");
    else if (arg == "--acknowledge-gpu-use") options.acknowledge_gpu_use = true;
    else if (arg == "--validate-only") options.validate_only = true;
    else if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else fail("unknown option: " + arg);
  }
  if (options.artifact.empty() || options.manifest_sha256.empty())
    fail("artifact and manifest digest are required");
  if (options.validate_only) return options;
  if (!options.acknowledge_gpu_use) fail("--acknowledge-gpu-use is required");
  if (options.receipt_out.empty() || options.direct_scratch_out.empty() ||
      options.icb_scratch_out.empty())
    fail("receipt and both scratch outputs are required");
  return options;
}

NSString *artifactPath(const Options &options, NSString *name) {
  return [ns(options.artifact) stringByAppendingPathComponent:name];
}

NSData *validatedArtifact(const Options &options, NSDictionary *manifest,
                          NSString *key) {
  NSDictionary *artifacts = dictionary(manifest[@"artifacts"], "artifacts");
  NSDictionary *metadata = dictionary(artifacts[key], text(key).c_str());
  NSString *file = stringValue(metadata[@"file"], "artifact file");
  NSData *data = readData(artifactPath(options, file));
  if (data.length != integerValue(metadata[@"bytes"], "artifact bytes"))
    fail("artifact byte count differs: " + text(file));
  if (sha256(data) != text(stringValue(metadata[@"sha256"], "artifact sha256")))
    fail("artifact digest differs: " + text(file));
  return data;
}

void validateManifestContract(NSDictionary *manifest, NSData *weight,
                              NSData *schedule, NSData *table,
                              NSData *scratch) {
  if (text(stringValue(manifest[@"schema"], "manifest schema")) != kManifestSchema)
    fail("manifest schema mismatch");
  NSDictionary *pack = dictionary(manifest[@"pack_abi"], "pack_abi");
  if (integerValue(pack[@"groups"], "pack groups") != 1 ||
      integerValue(pack[@"table_stride_words"], "table stride") != kTableStride ||
      text(stringValue(pack[@"sb_layout"], "scale/bias layout")) != "interleaved")
    fail("pack ABI is not one-group/table-12/interleaved");
  if (table.length != 5 * kTableStride * sizeof(uint32_t))
    fail("table must contain exactly five entries");
  const uint32_t *T = static_cast<const uint32_t *>(table.bytes);
  const std::array<std::array<uint32_t, 4>, 5> specs{{
      {{1, 10240, 0, 0}},
      {{320, 10240, 4, 64}},
      {{10240, 320, 4, 64}},
      {{4, 10240, 4, 64}},
      {{10240, 2560, 4, 64}},
  }};
  const uint64_t weightWords = weight.length / sizeof(uint32_t);
  if (weight.length % sizeof(uint32_t))
    fail("weight group is not a whole number of uint32 words");
  for (size_t i = 0; i < specs.size(); ++i) {
    const uint32_t *row = T + i * kTableStride;
    const uint32_t expectedKind = i == 0 ? 1u : 0u;
    if (row[0] != 0 || row[1] != expectedKind || row[2] != specs[i][0] ||
        row[3] != specs[i][1] || row[4] != 0 || row[5] != specs[i][2] ||
        row[6] != specs[i][3])
      fail("table geometry differs at entry " + std::to_string(i));
    if (uint64_t(row[7]) + row[9] > weightWords ||
        uint64_t(row[8]) + row[10] > weightWords)
      fail("table range exceeds weight group at entry " + std::to_string(i));
  }
  NSDictionary *sched = dictionary(manifest[@"schedule_abi"], "schedule_abi");
  if (integerValue(sched[@"commands"], "schedule commands") != kCommands ||
      integerValue(sched[@"step_stride_words"], "schedule stride") !=
          kScheduleStride ||
      integerValue(sched[@"icb_barriers"], "ICB barriers") != kCommands - 1)
    fail("schedule ABI counts differ");
  if (schedule.length != kCommands * kScheduleStride * sizeof(uint32_t))
    fail("schedule binary has wrong byte count");
  const uint32_t *words = static_cast<const uint32_t *>(schedule.bytes);
  for (size_t command = 0; command < kCommands; ++command)
    for (size_t word = 0; word < kScheduleStride; ++word)
      if (words[command * kScheduleStride + word] !=
          kExpectedSchedule[command][word])
        fail("schedule binary differs at command " + std::to_string(command) +
             " word " + std::to_string(word));
  NSDictionary *contract = dictionary(manifest[@"scratch_contract"],
                                      "scratch_contract");
  if (integerValue(contract[@"full_scratch_floats"], "scratch floats") !=
      kScratchFloats || scratch.length != kScratchFloats * sizeof(float))
    fail("scratch geometry differs");
  NSArray *inputs = array(contract[@"inputs"], "scratch inputs");
  if (inputs.count != 1) fail("scratch contract must have one input");
  NSDictionary *input = dictionary(inputs[0], "scratch input");
  if (text(stringValue(input[@"name"], "input name")) != "RESID_A" ||
      integerValue(input[@"offset"], "input offset") != 0 ||
      integerValue(input[@"floats"], "input floats") != 10240)
    fail("scratch input contract differs");
  NSArray *outputs = array(contract[@"outputs"], "scratch outputs");
  if (outputs.count != kExpectedOutputs.size())
    fail("scratch contract must expose exactly six outputs");
  for (size_t i = 0; i < kExpectedOutputs.size(); ++i) {
    NSDictionary *output = dictionary(outputs[i], "scratch output");
    const auto &expected = kExpectedOutputs[i];
    if (text(stringValue(output[@"name"], "output name")) != expected.name ||
        integerValue(output[@"offset"], "output offset") != expected.offset ||
        integerValue(output[@"floats"], "output floats") != expected.floats)
      fail("scratch output contract differs at index " + std::to_string(i));
  }
  NSDictionary *comparison = dictionary(contract[@"comparison"],
                                        "scratch comparison");
  if (text(stringValue(comparison[@"direct_vs_icb"],
                       "direct/ICB comparison")) != "bitwise_uint32" ||
      text(stringValue(comparison[@"oracle"], "oracle comparison")) !=
          "abs(candidate-oracle) <= atol + rtol*abs(oracle)" ||
      numberValue(comparison[@"oracle_rtol"], "oracle rtol") != kOracleRTol ||
      numberValue(comparison[@"oracle_atol"], "oracle atol") != kOracleATol)
    fail("scratch accuracy policy differs");
  NSDictionary *geometry = dictionary(manifest[@"native_geometry"],
                                      "native_geometry");
  if (integerValue(geometry[@"threadgroups"], "threadgroups") != kThreadgroups ||
      integerValue(geometry[@"threads_per_threadgroup"], "threads") != kThreads ||
      integerValue(geometry[@"qmv_threadgroup_bytes"], "qmv tg bytes") !=
          kQMVThreadgroupBytes ||
      integerValue(geometry[@"norm_threadgroup_bytes"], "norm tg bytes") !=
          kNormThreadgroupBytes)
    fail("native geometry differs");
}

Inputs loadInputs(const Options &options) {
  NSString *manifestPath = artifactPath(options, @"manifest.json");
  NSData *manifestData = readData(manifestPath);
  const std::string digest = sha256(manifestData);
  if (digest != options.manifest_sha256)
    fail("--manifest-sha256 does not match manifest.json");
  NSString *stamp = [[NSString alloc]
      initWithData:readData(artifactPath(options, @"MANIFEST.sha256"))
          encoding:NSUTF8StringEncoding];
  NSArray<NSString *> *parts = [stamp componentsSeparatedByCharactersInSet:
      [NSCharacterSet whitespaceAndNewlineCharacterSet]];
  NSString *stamped = nil;
  for (NSString *part in parts) if (part.length) { stamped = part; break; }
  if (!stamped || text(stamped) != digest) fail("MANIFEST.sha256 mismatch");
  NSError *error = nil;
  NSDictionary *manifest = dictionary(
      [NSJSONSerialization JSONObjectWithData:manifestData options:0 error:&error],
      "manifest");
  Inputs result;
  result.manifest = manifest;
  result.manifest_digest = digest;
  result.weight = validatedArtifact(options, manifest, @"weight_group_0");
  result.table = validatedArtifact(options, manifest, @"table");
  result.schedule = validatedArtifact(options, manifest, @"schedule");
  result.scratch_input = validatedArtifact(options, manifest, @"scratch_input");
  result.scratch_expected = validatedArtifact(
      options, manifest, @"scratch_expected");
  NSData *sourceData = validatedArtifact(options, manifest, @"phase_source");
  result.source = [[NSString alloc] initWithData:sourceData
                                        encoding:NSUTF8StringEncoding];
  if (!result.source) fail("phase source is not UTF-8");
  validateManifestContract(manifest, result.weight, result.schedule,
                           result.table, result.scratch_input);
  if (result.scratch_expected.length != result.scratch_input.length)
    fail("expected scratch byte count differs from input scratch");
  const uint32_t *scheduleWords =
      static_cast<const uint32_t *>(result.schedule.bytes);
  for (size_t i = 0; i < kCommands; ++i) {
    std::memcpy(&result.params[i], scheduleWords + i * kScheduleStride,
                kScheduleStride * sizeof(uint32_t));
    result.params[i].command = static_cast<uint32_t>(i);
  }
  return result;
}

Pipelines makePipelines(id<MTLDevice> device, NSString *source) {
  MTLCompileOptions *options = [MTLCompileOptions new];
  options.languageVersion = MTLLanguageVersion3_2;
  options.fastMathEnabled = NO;
  NSError *error = nil;
  id<MTLLibrary> library = [device newLibraryWithSource:source
                                                options:options
                                                  error:&error];
  if (!library) fail("Metal source compilation failed: " +
                     text(error.localizedDescription));
  auto pipeline = [&](NSString *name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) fail("missing Metal function " + text(name));
    MTLComputePipelineDescriptor *descriptor = [MTLComputePipelineDescriptor new];
    descriptor.label = [@"Qwen4 phase-family P1 " stringByAppendingString:name];
    descriptor.computeFunction = function;
    descriptor.supportIndirectCommandBuffers = YES;
    NSError *pipelineError = nil;
    id<MTLComputePipelineState> state =
        [device newComputePipelineStateWithDescriptor:descriptor
                                               options:MTLPipelineOptionNone
                                            reflection:nil
                                                 error:&pipelineError];
    if (!state) fail("pipeline creation failed for " + text(name) + ": " +
                     text(pipelineError.localizedDescription));
    if (!state.supportIndirectCommandBuffers)
      fail("pipeline did not retain ICB support: " + text(name));
    if (state.maxTotalThreadsPerThreadgroup < kThreads)
      fail("pipeline does not admit 512 threads: " + text(name));
    return state;
  };
  return {pipeline(@"p1_group_rmsnorm"), pipeline(@"p1_qmv"),
          pipeline(@"p1_hc_mix")};
}

id<MTLComputePipelineState> pipelineFor(const Pipelines &pipelines, size_t i) {
  if (i == 0) return pipelines.norm;
  if (i == 3) return pipelines.mix;
  return pipelines.qmv;
}

NSUInteger threadgroupBytes(size_t i) {
  if (i == 0) return kNormThreadgroupBytes;
  if (i == 3) return 0;
  return kQMVThreadgroupBytes;
}

id<MTLBuffer> bufferWithData(id<MTLDevice> device, NSData *data) {
  id<MTLBuffer> buffer = [device newBufferWithBytes:data.bytes
                                             length:data.length
                                            options:MTLResourceStorageModeShared];
  if (!buffer) fail("Metal buffer allocation failed");
  return buffer;
}

LaneBuffers makeLane(id<MTLDevice> device, NSData *scratch) {
  LaneBuffers lane;
  lane.scratch = bufferWithData(device, scratch);
  lane.receipts = [device newBufferWithLength:kCommands * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];
  if (!lane.receipts) fail("receipt buffer allocation failed");
  std::memset(lane.receipts.contents, 0, lane.receipts.length);
  return lane;
}

void resetLane(const LaneBuffers &lane, NSData *scratch) {
  std::memcpy(lane.scratch.contents, scratch.bytes, scratch.length);
  std::memset(lane.receipts.contents, 0, lane.receipts.length);
}

void bindDirect(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> weight,
                id<MTLBuffer> table, const LaneBuffers &lane,
                id<MTLBuffer> params, size_t command) {
  [encoder setBuffer:weight offset:0 atIndex:0];
  [encoder setBuffer:table offset:0 atIndex:1];
  [encoder setBuffer:lane.scratch offset:0 atIndex:2];
  [encoder setBuffer:params offset:command * sizeof(PhaseParams) atIndex:3];
  [encoder setBuffer:lane.receipts offset:0 atIndex:4];
  [encoder setThreadgroupMemoryLength:threadgroupBytes(command) atIndex:0];
}

void bindIndirect(id<MTLIndirectComputeCommand> command, id<MTLBuffer> weight,
                  id<MTLBuffer> table, const LaneBuffers &lane,
                  id<MTLBuffer> params, size_t index) {
  [command setKernelBuffer:weight offset:0 atIndex:0];
  [command setKernelBuffer:table offset:0 atIndex:1];
  [command setKernelBuffer:lane.scratch offset:0 atIndex:2];
  [command setKernelBuffer:params offset:index * sizeof(PhaseParams) atIndex:3];
  [command setKernelBuffer:lane.receipts offset:0 atIndex:4];
  [command setThreadgroupMemoryLength:threadgroupBytes(index) atIndex:0];
}

void checkCommandBuffer(id<MTLCommandBuffer> command, const char *lane) {
  if (command.status == MTLCommandBufferStatusError)
    fail(std::string(lane) + " command buffer failed: " +
         text(command.error.localizedDescription));
}

Timings submitDirect(id<MTLCommandQueue> queue, const Pipelines &pipelines,
                     id<MTLBuffer> weight, id<MTLBuffer> table,
                     const LaneBuffers &lane, id<MTLBuffer> params) {
  Timings result;
  const auto wallStart = Clock::now();
  const auto encodeStart = Clock::now();
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder =
      [command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  if (!command || !encoder) fail("cannot create direct command encoder");
  for (size_t i = 0; i < kCommands; ++i) {
    [encoder setComputePipelineState:pipelineFor(pipelines, i)];
    bindDirect(encoder, weight, table, lane, params, i);
    [encoder dispatchThreadgroups:MTLSizeMake(kThreadgroups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
    if (i + 1 < kCommands) [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
  }
  [encoder endEncoding];
  result.host_encode_us = micros(encodeStart, Clock::now());
  [command commit];
  [command waitUntilCompleted];
  checkCommandBuffer(command, "direct");
  result.wall_us = micros(wallStart, Clock::now());
  result.gpu_us = (command.GPUEndTime - command.GPUStartTime) * 1.0e6;
  result.submissions = 1;
  return result;
}

id<MTLIndirectCommandBuffer> makeICB(
    id<MTLDevice> device, const Pipelines &pipelines, id<MTLBuffer> weight,
    id<MTLBuffer> table, const LaneBuffers &lane, id<MTLBuffer> params,
    double *preencode_us) {
  MTLIndirectCommandBufferDescriptor *descriptor =
      [MTLIndirectCommandBufferDescriptor new];
  descriptor.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
  descriptor.inheritPipelineState = NO;
  descriptor.inheritBuffers = NO;
  descriptor.maxKernelBufferBindCount = 5;
  descriptor.maxKernelThreadgroupMemoryBindCount = 1;
  id<MTLIndirectCommandBuffer> icb =
      [device newIndirectCommandBufferWithDescriptor:descriptor
                                      maxCommandCount:kCommands
                                                options:MTLResourceStorageModeShared];
  if (!icb) fail("device rejected the compute indirect command buffer");
  const auto start = Clock::now();
  for (size_t i = 0; i < kCommands; ++i) {
    id<MTLIndirectComputeCommand> command = [icb indirectComputeCommandAtIndex:i];
    if (!command) fail("indirectComputeCommandAtIndex returned nil");
    if (i > 0) [command setBarrier];
    [command setComputePipelineState:pipelineFor(pipelines, i)];
    bindIndirect(command, weight, table, lane, params, i);
    [command concurrentDispatchThreadgroups:MTLSizeMake(kThreadgroups, 1, 1)
                         threadsPerThreadgroup:MTLSizeMake(kThreads, 1, 1)];
  }
  *preencode_us = micros(start, Clock::now());
  if (!icb.size) fail("encoded ICB reports zero size");
  return icb;
}

Timings submitICB(id<MTLCommandQueue> queue, id<MTLIndirectCommandBuffer> icb,
                  id<MTLBuffer> weight, id<MTLBuffer> table,
                  const LaneBuffers &lane, id<MTLBuffer> params) {
  Timings result;
  const auto wallStart = Clock::now();
  const auto encodeStart = Clock::now();
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLComputeCommandEncoder> encoder =
      [command computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
  if (!command || !encoder) fail("cannot create ICB replay encoder");
  [encoder useResource:weight usage:MTLResourceUsageRead];
  [encoder useResource:table usage:MTLResourceUsageRead];
  [encoder useResource:lane.scratch usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  [encoder useResource:params usage:MTLResourceUsageRead];
  [encoder useResource:lane.receipts usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  [encoder executeCommandsInBuffer:icb withRange:NSMakeRange(0, kCommands)];
  [encoder endEncoding];
  result.host_encode_us = micros(encodeStart, Clock::now());
  [command commit];
  [command waitUntilCompleted];
  checkCommandBuffer(command, "ICB");
  result.wall_us = micros(wallStart, Clock::now());
  result.gpu_us = (command.GPUEndTime - command.GPUStartTime) * 1.0e6;
  result.submissions = 1;
  return result;
}

void add(Timings *total, const Timings &value) {
  total->host_encode_us += value.host_encode_us;
  total->wall_us += value.wall_us;
  total->gpu_us += value.gpu_us;
  total->submissions += value.submissions;
}

NSArray *receiptCounts(const LaneBuffers &lane) {
  const uint32_t *values = static_cast<const uint32_t *>(lane.receipts.contents);
  NSMutableArray *result = [NSMutableArray arrayWithCapacity:kCommands];
  for (size_t i = 0; i < kCommands; ++i) [result addObject:@(values[i])];
  return result;
}

uint32_t receiptTotal(const LaneBuffers &lane) {
  const uint32_t *values = static_cast<const uint32_t *>(lane.receipts.contents);
  uint32_t total = 0;
  for (size_t i = 0; i < kCommands; ++i) {
    if (values[i] != 1) fail("device receipt differs at command " + std::to_string(i));
    total += values[i];
  }
  return total;
}

std::string firstOffsets(const std::vector<uint32_t> &offsets) {
  std::ostringstream out;
  for (size_t i = 0; i < offsets.size(); ++i) {
    if (i) out << ",";
    out << offsets[i];
  }
  return out.str();
}

NSArray *offsetsJSON(const std::vector<uint32_t> &offsets) {
  NSMutableArray *result = [NSMutableArray arrayWithCapacity:offsets.size()];
  for (uint32_t value : offsets) [result addObject:@(value)];
  return result;
}

AccuracyCheck validateOutputs(NSData *expectedData, const LaneBuffers &direct,
                              const LaneBuffers &icb) {
  const uint32_t *expected =
      static_cast<const uint32_t *>(expectedData.bytes);
  const float *expectedFloats = static_cast<const float *>(expectedData.bytes);
  const uint32_t *directWords =
      static_cast<const uint32_t *>(direct.scratch.contents);
  const float *directFloats = static_cast<const float *>(direct.scratch.contents);
  const uint32_t *icbWords =
      static_cast<const uint32_t *>(icb.scratch.contents);
  const float *icbFloats = static_cast<const float *>(icb.scratch.contents);
  std::ostringstream failures;
  size_t failedRanges = 0;
  NSMutableArray *outputs = [NSMutableArray arrayWithCapacity:kCommands];
  for (const auto &range : kExpectedOutputs) {
    size_t outsideDirect = 0, outsideICB = 0, directICB = 0;
    double maxDirect = 0.0, maxICB = 0.0;
    double totalDirect = 0.0, totalICB = 0.0;
    std::vector<uint32_t> firstOutsideDirect;
    std::vector<uint32_t> firstOutsideICB;
    std::vector<uint32_t> firstDirectICB;
    for (uint32_t i = 0; i < range.floats; ++i) {
      const size_t at = range.offset + i;
      const double reference = expectedFloats[at];
      const double directValue = directFloats[at];
      const double icbValue = icbFloats[at];
      const double directError = std::abs(directValue - reference);
      const double icbError = std::abs(icbValue - reference);
      const double allowed = kOracleATol + kOracleRTol * std::abs(reference);
      const bool directAccepted = expected[at] == directWords[at] ||
          (std::isfinite(reference) && std::isfinite(directValue) &&
           directError <= allowed);
      const bool icbAccepted = expected[at] == icbWords[at] ||
          (std::isfinite(reference) && std::isfinite(icbValue) &&
           icbError <= allowed);
      if (std::isfinite(directError)) {
        maxDirect = std::max(maxDirect, directError);
        totalDirect += directError;
      }
      if (std::isfinite(icbError)) {
        maxICB = std::max(maxICB, icbError);
        totalICB += icbError;
      }
      if (!directAccepted) {
        ++outsideDirect;
        if (firstOutsideDirect.size() < 8) firstOutsideDirect.push_back(i);
      }
      if (!icbAccepted) {
        ++outsideICB;
        if (firstOutsideICB.size() < 8) firstOutsideICB.push_back(i);
      }
      if (directWords[at] != icbWords[at]) {
        ++directICB;
        if (firstDirectICB.size() < 8) firstDirectICB.push_back(i);
      }
    }
    const bool passed = !outsideDirect && !outsideICB && !directICB;
    [outputs addObject:@{
      @"name": ns(range.name),
      @"floats": @(range.floats),
      @"oracle_vs_direct": @{
        @"max_abs": @(maxDirect),
        @"mean_abs": @(totalDirect / range.floats),
        @"outside_tolerance": @(outsideDirect),
        @"first_relative_word_offsets": offsetsJSON(firstOutsideDirect),
      },
      @"oracle_vs_icb": @{
        @"max_abs": @(maxICB),
        @"mean_abs": @(totalICB / range.floats),
        @"outside_tolerance": @(outsideICB),
        @"first_relative_word_offsets": offsetsJSON(firstOutsideICB),
      },
      @"direct_vs_icb": @{
        @"mismatch_words": @(directICB),
        @"first_relative_word_offsets": offsetsJSON(firstDirectICB),
      },
      @"passed": @(passed),
    }];
    if (!passed) {
      ++failedRanges;
      failures << " " << range.name
               << " oracle/direct_outside=" << outsideDirect << "["
               << firstOffsets(firstOutsideDirect) << "]"
               << " oracle/icb_outside=" << outsideICB << "["
               << firstOffsets(firstOutsideICB) << "]"
               << " direct/icb=" << directICB << "["
               << firstOffsets(firstDirectICB) << "]";
    }
  }
  AccuracyCheck result;
  result.passed = failedRanges == 0;
  result.failure = failedRanges
      ? "output validation failed across " + std::to_string(failedRanges) +
            " ranges:" + failures.str()
      : "";
  result.json = @{
    @"direct_icb_comparison": @"bitwise_uint32",
    @"oracle_comparison": @"abs(candidate-oracle) <= atol + rtol*abs(oracle)",
    @"oracle_rtol": @(kOracleRTol),
    @"oracle_atol": @(kOracleATol),
    @"outputs": outputs,
    @"passed": @(result.passed),
  };
  return result;
}

NSDictionary *timingJSON(const Timings &value, size_t repetitions) {
  const double commands = static_cast<double>(repetitions * kCommands);
  return @{
    @"repetitions": @(repetitions),
    @"submissions": @(value.submissions),
    @"host_encode_us_total": @(value.host_encode_us),
    @"host_encode_us_per_replay": @(value.host_encode_us / repetitions),
    @"host_encode_us_per_command": @(value.host_encode_us / commands),
    @"wall_us_total": @(value.wall_us),
    @"wall_us_per_replay": @(value.wall_us / repetitions),
    @"gpu_us_total": @(value.gpu_us),
    @"gpu_us_per_replay": @(value.gpu_us / repetitions),
    @"gpu_us_per_command": @(value.gpu_us / commands),
  };
}

int run(const Options &options) {
  Inputs inputs = loadInputs(options);
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) fail("no Metal device");
  if (device.maxThreadgroupMemoryLength < kQMVThreadgroupBytes)
    fail("device threadgroup memory is below 10 KiB");
  id<MTLCommandQueue> queue = [device newCommandQueue];
  if (!queue) fail("cannot create Metal command queue");
  Pipelines pipelines = makePipelines(device, inputs.source);
  id<MTLBuffer> weight = bufferWithData(device, inputs.weight);
  id<MTLBuffer> table = bufferWithData(device, inputs.table);
  NSData *paramsData = [NSData dataWithBytes:inputs.params.data()
                                      length:sizeof(inputs.params)];
  id<MTLBuffer> params = bufferWithData(device, paramsData);

  LaneBuffers direct = makeLane(device, inputs.scratch_input);
  LaneBuffers icbLane = makeLane(device, inputs.scratch_input);
  double preencodeUs = 0.0;
  id<MTLIndirectCommandBuffer> icb = makeICB(
      device, pipelines, weight, table, icbLane, params, &preencodeUs);

  resetLane(direct, inputs.scratch_input);
  submitDirect(queue, pipelines, weight, table, direct, params);
  NSArray *directCounts = receiptCounts(direct);
  const uint32_t directTotal = receiptTotal(direct);
  writeData([NSData dataWithBytes:direct.scratch.contents
                           length:direct.scratch.length],
            options.direct_scratch_out);

  resetLane(icbLane, inputs.scratch_input);
  submitICB(queue, icb, weight, table, icbLane, params);
  NSArray *icbCounts = receiptCounts(icbLane);
  const uint32_t icbTotal = receiptTotal(icbLane);
  writeData([NSData dataWithBytes:icbLane.scratch.contents
                           length:icbLane.scratch.length],
            options.icb_scratch_out);
  AccuracyCheck accuracy = validateOutputs(
      inputs.scratch_expected, direct, icbLane);
  NSDictionary *mechanism = @{
    @"direct_commands": @(kCommands),
    @"direct_barriers": @(kCommands - 1),
    @"direct_submissions": @1,
    @"direct_device_receipts": @(directTotal),
    @"direct_receipt_slots": directCounts,
    @"icb_encoded_commands": @(kCommands),
    @"icb_barriers": @(kCommands - 1),
    @"icb_execute_calls": @1,
    @"icb_submissions": @1,
    @"icb_device_receipts": @(icbTotal),
    @"icb_receipt_slots": icbCounts,
    @"icb_size_bytes": @(icb.size),
  };
  if (!accuracy.passed) {
    NSDictionary *failureReceipt = @{
      @"schema": ns(kReceiptSchema),
      @"passed": @NO,
      @"manifest_sha256": ns(inputs.manifest_digest),
      @"implementation": inputs.manifest[@"implementation"],
      @"device": device.name,
      @"mechanism": mechanism,
      @"accuracy": accuracy.json,
      @"failures": @[ns(accuracy.failure)],
    };
    writeJSON(failureReceipt, options.receipt_out);
    std::cerr << "FAIL: " << accuracy.failure << "\n";
    return 1;
  }

  for (size_t i = 0; i < options.warmups; ++i) {
    resetLane(direct, inputs.scratch_input);
    submitDirect(queue, pipelines, weight, table, direct, params);
    resetLane(icbLane, inputs.scratch_input);
    submitICB(queue, icb, weight, table, icbLane, params);
  }

  Timings directTiming, icbTiming;
  for (size_t i = 0; i < options.repetitions; ++i) {
    resetLane(direct, inputs.scratch_input);
    add(&directTiming, submitDirect(queue, pipelines, weight, table, direct, params));
    resetLane(icbLane, inputs.scratch_input);
    add(&icbTiming, submitICB(queue, icb, weight, table, icbLane, params));
  }

  NSDictionary *timing = @{
    @"icb_preencode_us": @(preencodeUs),
    @"direct": timingJSON(directTiming, options.repetitions),
    @"icb": timingJSON(icbTiming, options.repetitions),
  };
  NSDictionary *receipt = @{
    @"schema": ns(kReceiptSchema),
    @"passed": @YES,
    @"manifest_sha256": ns(inputs.manifest_digest),
    @"implementation": inputs.manifest[@"implementation"],
    @"device": device.name,
    @"mechanism": mechanism,
    @"accuracy": accuracy.json,
    @"timing": timing,
  };
  writeJSON(receipt, options.receipt_out);
  NSData *stdoutData = [NSJSONSerialization dataWithJSONObject:receipt
                                                       options:NSJSONWritingPrettyPrinted
                                                         error:nil];
  std::cout << std::string(static_cast<const char *>(stdoutData.bytes),
                           stdoutData.length)
            << "\n";
  return 0;
}

int validateOnly(const Options &options) {
  Inputs inputs = loadInputs(options);
  NSDictionary *result = @{
    @"schema": @"mlx-uag.qwen4-phase-family-icb-p1-native-preflight.v1",
    @"passed": @YES,
    @"gpu_touched": @NO,
    @"manifest_sha256": ns(inputs.manifest_digest),
    @"commands": @(kCommands),
    @"outputs": @(kExpectedOutputs.size()),
  };
  NSData *data = [NSJSONSerialization dataWithJSONObject:result
                                                  options:NSJSONWritingPrettyPrinted
                                                    error:nil];
  std::cout << std::string(static_cast<const char *>(data.bytes), data.length)
            << "\n";
  return 0;
}

}  // namespace

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    Options options;
    try {
      options = parseOptions(argc, argv);
      return options.validate_only ? validateOnly(options) : run(options);
    } catch (const std::exception &error) {
      std::cerr << "FAIL: " << error.what() << "\n";
      if (!options.receipt_out.empty()) {
        NSDictionary *failure = @{
          @"schema": ns(kReceiptSchema),
          @"passed": @NO,
          @"failures": @[ns(error.what())],
        };
        try {
          writeJSON(failure, options.receipt_out);
        } catch (...) {
        }
      }
      return 1;
    }
  }
}
