"""
Performance monitoring and diagnostics for CRAG system.

This module provides timing decorators and diagnostic tools to identify
performance bottlenecks in the CRAG pipeline.
"""
from __future__ import annotations

import time
from functools import wraps
from typing import Callable, Dict, List, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.
    
    Usage:
        @timing_decorator
        def my_function():
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"⏱️  {func.__name__} 耗时: {elapsed:.2f}秒")
        return result
    return wrapper


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
    
    def record(self, operation: str, duration: float):
        """Record a performance metric."""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all recorded operations."""
        stats = {}
        for operation, durations in self.metrics.items():
            if durations:
                stats[operation] = {
                    "count": len(durations),
                    "total": sum(durations),
                    "avg": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations),
                }
        return stats
    
    def print_summary(self):
        """Print performance summary."""
        print("\n" + "="*60)
        print("📊 CRAG 性能统计")
        print("="*60)
        
        stats = self.get_stats()
        for operation, stat in stats.items():
            print(f"\n{operation}:")
            print(f"  调用次数: {stat['count']}")
            print(f"  总耗时: {stat['total']:.2f}秒")
            print(f"  平均耗时: {stat['avg']:.3f}秒")
            print(f"  最快: {stat['min']:.3f}秒")
            print(f"  最慢: {stat['max']:.3f}秒")
        
        print("="*60 + "\n")


def check_gpu_status() -> Dict[str, any]:
    """
    Check GPU availability and status.
    
    Returns:
        Dictionary with GPU information.
    """
    info = {
        "cuda_available": False,
        "device_name": None,
        "gpu_memory_total": None,
        "gpu_memory_allocated": None,
    }
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        info["cuda_available"] = True
        info["device_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info["gpu_memory_allocated"] = torch.cuda.memory_allocated(0) / 1024**3
    
    return info


def print_gpu_status():
    """Print GPU status information."""
    status = check_gpu_status()
    
    print("\n" + "="*60)
    print("🖥️  GPU 状态检查")
    print("="*60)
    
    if status["cuda_available"]:
        print(f"✅ GPU 可用: {status['device_name']}")
        print(f"   总内存: {status['gpu_memory_total']:.2f} GB")
        print(f"   已使用: {status['gpu_memory_allocated']:.2f} GB")
    else:
        print("❌ GPU 不可用，将使用 CPU（性能较慢）")
    
    print("="*60 + "\n")


def diagnostic_test(
    evaluator,
    num_documents: int = 12,
    batch_size: int = 4,
) -> Dict[str, float]:
    """
    Run diagnostic test to identify performance issues.
    
    Args:
        evaluator: SemanticRetrievalEvaluator instance.
        num_documents: Number of test documents.
        batch_size: Batch size for evaluation.
        
    Returns:
        Dictionary with diagnostic results.
    """
    print("\n" + "="*60)
    print("🔍 CRAG 性能诊断测试")
    print("="*60)
    
    # Check GPU
    print_gpu_status()
    
    # Test data
    test_query = "什么是机器学习"
    test_documents = [
        "机器学习是人工智能的一个分支，主要研究计算机如何模拟人类学习行为。",
        "深度学习是机器学习的一个子领域，使用神经网络进行特征学习。",
        "自然语言处理是人工智能的另一个重要分支。",
        "计算机视觉主要研究如何让计算机理解和解释视觉信息。",
    ] * (num_documents // 4 + 1)
    test_documents = test_documents[:num_documents]
    
    print(f"\n测试配置:")
    print(f"  查询: {test_query}")
    print(f"  文档数: {len(test_documents)}")
    print(f"  批处理大小: {batch_size}")
    
    # Test batch evaluation
    print(f"\n开始批量评估...")
    start_time = time.time()
    
    scores = evaluator.evaluate_batch(test_query, test_documents)
    
    eval_time = time.time() - start_time
    
    # Results
    results = {
        "num_documents": len(test_documents),
        "eval_time": eval_time,
        "avg_time_per_doc": eval_time / len(test_documents),
        "throughput": len(test_documents) / eval_time,
    }
    
    print(f"\n📊 诊断结果:")
    print(f"  评估 {len(test_documents)} 个文档耗时: {eval_time:.2f}秒")
    print(f"  平均每个文档: {results['avg_time_per_doc']:.3f}秒")
    print(f"  吞吐量: {results['throughput']:.2f} 文档/秒")
    
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  GPU内存使用: {gpu_mem:.2f} GB")
    
    # Performance assessment
    print(f"\n📈 性能评估:")
    if results['avg_time_per_doc'] < 0.1:
        print("  ✅ 优秀: 平均 < 0.1秒/文档")
    elif results['avg_time_per_doc'] < 0.5:
        print("  ✅ 良好: 平均 < 0.5秒/文档")
    elif results['avg_time_per_doc'] < 1.0:
        print("  ⚠️  一般: 平均 < 1.0秒/文档")
    else:
        print("  🔴 较慢: 平均 > 1.0秒/文档")
        print("  建议: 检查GPU使用、批处理大小、模型加载")
    
    print("="*60 + "\n")
    
    return results

