#!/usr/bin/env python3
"""NPU性能分析脚本 — 使用torch_npu.profiler分析算子性能。

用法:
    python profile.py --op_name <算子名> [--verify_dir <目录>] 
                     [--profile_dir <分析结果目录>] [--duration <时长>]

指标:
    - 详细的性能分析数据（通过torch_npu.profiler收集）
    - 各种算子的执行时间统计
"""

import argparse
import os
import sys
import json
import time
import statistics
import torch
import torch_npu  # noqa: F401


def profile_implementations(op_name, verify_dir, profile_dir, duration=10):
    """使用torch_npu.profiler分析框架实现和生成实现的性能"""
    sys.path.insert(0, verify_dir)
    
    # 加载模块
    torch_module = __import__(f"{op_name}_torch")
    FrameworkModel = torch_module.Model
    get_inputs = torch_module.get_inputs
    get_init_inputs = torch_module.get_init_inputs
    
    impl_module = __import__(f"{op_name}_triton_ascend_impl")
    ModelNew = impl_module.ModelNew
    
    device = torch.device("npu")
    
    # 初始化模型
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    init_params = get_init_inputs()
    framework_model = FrameworkModel(*init_params).to(device)
    impl_model = ModelNew(*init_params).to(device)
    
    # 准备输入数据
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    inputs_impl = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]
    
    torch.manual_seed(0)
    torch.npu.manual_seed(0)
    inputs_framework = [x.to(device) if isinstance(x, torch.Tensor) else x for x in get_inputs()]
    
    def measure_latency(model, inputs, n):
        """测量 n 次运行的延迟（毫秒）"""
        latencies = []
        for _ in range(n):
            torch.npu.synchronize()
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(*inputs)
            torch.npu.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # 转换为毫秒
        return latencies
    
    # 执行性能分析
    print(f"开始性能分析 (持续时间: {duration} 秒)...")
    
    # 创建分析结果目录
    os.makedirs(profile_dir, exist_ok=True)
    
    # 生成实现的性能分析
    print("分析生成实现...")
    impl_profile_path = os.path.join(profile_dir, f"{op_name}_impl_profile")
    
    # 使用torch_npu.profiler进行分析
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.NPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=duration, repeat=1)
    ) as prof:
        for i in range(10):  # 运行10次以收集足够的数据
            torch.npu.synchronize()
            with torch.no_grad():
                _ = impl_model(*inputs_impl)
            torch.npu.synchronize()
            prof.step()
    
    # 保存生成实现的分析结果
    impl_prof_file = impl_profile_path + ".json"
    prof.export_chrome_trace(impl_prof_file)
    print(f"生成实现分析结果已保存到: {impl_prof_file}")
    
    # 框架实现的性能分析
    print("分析框架实现...")
    framework_profile_path = os.path.join(profile_dir, f"{op_name}_framework_profile")
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.NPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=duration, repeat=1)
    ) as prof:
        for i in range(10):  # 运行10次以收集足够的数据
            torch.npu.synchronize()
            with torch.no_grad():
                _ = framework_model(*inputs_framework)
            torch.npu.synchronize()
            prof.step()
    
    # 保存框架实现的分析结果
    framework_prof_file = framework_profile_path + ".json"
    prof.export_chrome_trace(framework_prof_file)
    print(f"框架实现分析结果已保存到: {framework_prof_file}")
    
    # 基本性能测试
    print("执行基本性能测试...")
    
    # 重置内存统计
    torch.npu.reset_peak_memory_stats()
    
    # Warmup
    print("执行 warmup...")
    _ = measure_latency(impl_model, inputs_impl, 5)
    torch.npu.synchronize()
    
    # 正式测试：生成实现
    print("测试生成实现...")
    impl_latencies = measure_latency(impl_model, inputs_impl, 50)
    impl_peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)  # MB
    
    # 重置内存统计
    torch.npu.reset_peak_memory_stats()
    
    # Warmup 框架实现
    _ = measure_latency(framework_model, inputs_framework, 5)
    torch.npu.synchronize()
    
    # 正式测试：框架实现
    print("测试框架实现...")
    framework_latencies = measure_latency(framework_model, inputs_framework, 50)
    framework_peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)  # MB
    
    # 计算统计指标
    def calc_stats(latencies):
        sorted_lat = sorted(latencies)
        n = len(sorted_lat)
        return {
            "avg": statistics.mean(latencies),
            "p50": sorted_lat[n // 2] if n % 2 == 1 else (sorted_lat[n // 2 - 1] + sorted_lat[n // 2]) / 2,
            "p99": sorted_lat[int(n * 0.99)] if n > 1 else sorted_lat[0]
        }
    
    impl_stats = calc_stats(impl_latencies)
    framework_stats = calc_stats(framework_latencies)
    
    # 计算加速比
    speedup = framework_stats["avg"] / impl_stats["avg"] if impl_stats["avg"] > 0 else 0
    
    result = {
        "op_name": op_name,
        "profile_dir": profile_dir,
        "duration": duration,
        "framework": {
            "avg_latency_ms": round(framework_stats["avg"], 4),
            "p50_latency_ms": round(framework_stats["p50"], 4),
            "p99_latency_ms": round(framework_stats["p99"], 4),
            "peak_memory_mb": round(framework_peak_memory, 2)
        },
        "implementation": {
            "avg_latency_ms": round(impl_stats["avg"], 4),
            "p50_latency_ms": round(impl_stats["p50"], 4),
            "p99_latency_ms": round(impl_stats["p99"], 4),
            "peak_memory_mb": round(impl_peak_memory, 2)
        },
        "speedup_vs_torch": round(speedup, 2)
    }
    
    return result


def main():
    parser = argparse.ArgumentParser(description="NPU性能分析脚本")
    parser.add_argument("--op_name", required=True, help="算子名称")
    parser.add_argument("--verify_dir", default=".", help="验证目录路径（默认当前目录）")
    parser.add_argument("--profile_dir", default="./profiles", help="分析结果保存目录（默认 ./profiles）")
    parser.add_argument("--duration", type=int, default=10, help="分析持续时间（秒，默认 10）")
    args = parser.parse_args()
    
    verify_dir = os.path.abspath(args.verify_dir)
    if not os.path.isdir(verify_dir):
        print(f"错误: 验证目录不存在: {verify_dir}", file=sys.stderr)
        sys.exit(1)
    
    try:
        result = profile_implementations(
            args.op_name, 
            verify_dir, 
            args.profile_dir,
            args.duration
        )
        
        # 打印结果
        print("\n性能分析结果:")
        print(f"  框架实现 - 平均延迟: {result['framework']['avg_latency_ms']:.4f} ms")
        print(f"  生成实现 - 平均延迟: {result['implementation']['avg_latency_ms']:.4f} ms")
        print(f"  加速比: {result['speedup_vs_torch']:.2f}x")
        print(f"  生成实现 - 峰值内存: {result['implementation']['peak_memory_mb']:.2f} MB")
        
        # 保存结果到JSON文件
        result_file = os.path.join(args.profile_dir, f"{args.op_name}_profile_result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n详细结果已保存到: {result_file}")
        
        sys.exit(0)
        
    except Exception as e:
        print(f"性能分析失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()