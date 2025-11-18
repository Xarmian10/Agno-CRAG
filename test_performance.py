"""
Performance test script for CRAG system.

Run this to diagnose performance issues:
    python test_performance.py
"""
from __future__ import annotations

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from crag_core import SemanticRetrievalEvaluator
    from performance_monitor import diagnostic_test, print_gpu_status
    EVALUATOR_AVAILABLE = True
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保已安装所有依赖: pip install torch transformers sentencepiece")
    EVALUATOR_AVAILABLE = False


def main():
    """Run performance diagnostic test."""
    print("="*60)
    print("CRAG 性能诊断工具")
    print("="*60)
    
    if not EVALUATOR_AVAILABLE:
        return
    
    # Check GPU status
    print_gpu_status()
    
    # Initialize evaluator
    try:
        model_path = os.getenv("T5_EVALUATOR_PATH", "finetuned_t5_evaluator")
        batch_size = int(os.getenv("T5_BATCH_SIZE", "4"))
        
        print(f"\n初始化语义评估器...")
        print(f"  模型路径: {model_path}")
        print(f"  批处理大小: {batch_size}")
        
        evaluator = SemanticRetrievalEvaluator(
            model_path=model_path,
            batch_size=batch_size,
        )
        
        # Run diagnostic test
        results = diagnostic_test(
            evaluator=evaluator,
            num_documents=12,
            batch_size=batch_size,
        )
        
        # Print performance stats
        stats = evaluator.get_performance_stats()
        print("\n📊 评估器统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Recommendations
        print("\n💡 优化建议:")
        if results['avg_time_per_doc'] > 1.0:
            print("  1. 检查GPU是否正确使用 (nvidia-smi)")
            print("  2. 增加批处理大小 (设置 T5_BATCH_SIZE=8 或更高)")
            print("  3. 确保模型在GPU上 (检查设备输出)")
            print("  4. 考虑使用更小的模型或量化")
        elif results['avg_time_per_doc'] > 0.5:
            print("  1. 可以尝试增加批处理大小以提高吞吐量")
            print("  2. 检查是否有其他进程占用GPU")
        else:
            print("  ✅ 性能良好！")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

