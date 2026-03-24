# NPU Profiler

这是一个使用torch_npu.profiler来分析算子性能的技能。

## 功能

- 使用torch_npu.profiler对算子进行详细的性能分析
- 收集CPU和NPU的性能数据
- 生成Chrome Trace格式的分析报告
- 提供基本的性能指标（延迟、内存使用等）

## 使用方法

```bash
python skills/npu-profiler/scripts/profile.py --op_name <算子名> [--verify_dir <目录>] [--profile_dir <分析结果目录>] [--duration <时长>]
```

### 参数说明

- `--op_name`: 算子名称（必需）
- `--verify_dir`: 验证目录路径（默认当前目录）
- `--profile_dir`: 分析结果保存目录（默认 ./profiles）
- `--duration`: 分析持续时间（秒，默认 10）

## 输出

- Chrome Trace格式的性能分析文件（.json）
- JSON格式的性能统计结果
- 控制台输出的性能指标

## 依赖

- torch_npu
- torch