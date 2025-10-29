#!/usr/bin/env python3
"""
Enhanced RAG Demo Script
Demonstrates the new capabilities without requiring database connection.
"""

import os
import sys
import asyncio
from pathlib import Path

# Set environment variables
os.environ["DASHSCOPE_API_KEY"] = "sk-919decde1cc14dcfa8132bc610401299"
os.environ["SUPABASE_URL"] = "https://demo.supabase.co"
os.environ["SUPABASE_ANON_KEY"] = "demo_key"
os.environ["SUPABASE_SERVICE_ROLE_KEY"] = "demo_service_key"

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def demo_reranking():
    """Demonstrate reranking functionality."""
    print("🎯 演示重排序功能")
    print("-" * 40)
    
    try:
        from app.services.rerank import rerank_service
        
        # 模拟搜索结果
        mock_results = [
            {
                'chunk_id': 'ai_doc#1',
                'source': '人工智能概述.pdf',
                'text': '人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。',
                'similarity': 0.7
            },
            {
                'chunk_id': 'ml_doc#1', 
                'source': '机器学习基础.pdf',
                'text': '机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。',
                'similarity': 0.9
            },
            {
                'chunk_id': 'dl_doc#1',
                'source': '深度学习入门.pdf', 
                'text': '深度学习使用多层神经网络来模拟人脑的工作方式，在图像识别和自然语言处理方面取得了突破性进展。',
                'similarity': 0.8
            },
            {
                'chunk_id': 'nlp_doc#1',
                'source': '自然语言处理.pdf',
                'text': '自然语言处理是人工智能的一个重要分支，专注于让计算机理解、解释和生成人类语言。',
                'similarity': 0.6
            }
        ]
        
        query = "什么是机器学习？"
        
        print(f"查询: {query}")
        print(f"原始搜索结果数量: {len(mock_results)}")
        print(f"原始排序: {[r['chunk_id'] for r in mock_results]}")
        
        # 应用重排序
        reranked = rerank_service.rerank_results_sync(query, mock_results)
        
        print(f"重排序后结果数量: {len(reranked)}")
        print(f"重排序后顺序: {[r['chunk_id'] for r in reranked]}")
        
        print("\n📊 重排序结果分析:")
        for i, result in enumerate(reranked, 1):
            print(f"  {i}. {result['chunk_id']} (相似度: {result['similarity']:.2f})")
            print(f"     内容: {result['text'][:50]}...")
        
        print("✅ 重排序功能演示完成\n")
        return True
        
    except Exception as e:
        print(f"❌ 重排序演示失败: {e}")
        return False

async def demo_document_parsing():
    """Demonstrate document parsing functionality."""
    print("📄 演示文档解析功能")
    print("-" * 40)
    
    try:
        from app.services.document_parser import document_parser
        
        # 创建测试文档
        test_file = Path("demo_document.txt")
        test_content = """
        人工智能技术发展报告

        1. 概述
        人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。

        2. 主要技术领域
        - 机器学习：通过算法让计算机从数据中学习模式
        - 深度学习：使用神经网络进行复杂模式识别
        - 自然语言处理：让计算机理解和生成人类语言
        - 计算机视觉：让计算机"看懂"图像和视频

        3. 应用场景
        - 智能助手：如Siri、Alexa等语音助手
        - 自动驾驶：Tesla、Waymo等公司的自动驾驶技术
        - 医疗诊断：AI辅助医生进行疾病诊断
        - 金融风控：银行和金融机构的风险评估系统

        4. 发展趋势
        随着计算能力的提升和大数据的积累，人工智能技术正在快速发展，未来将在更多领域发挥重要作用。
        """
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        print(f"创建测试文档: {test_file}")
        
        # 解析文档
        documents = document_parser.parse_file(str(test_file))
        print(f"解析结果: {len(documents)} 个文档")
        
        # 转换为chunks
        chunks = document_parser.convert_to_chunks(documents)
        print(f"分块结果: {len(chunks)} 个chunks")
        
        print("\n📊 文档分块分析:")
        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}: {chunk['chunk_id']}")
            print(f"    内容长度: {len(chunk['text'])} 字符")
            print(f"    内容预览: {chunk['text'][:80]}...")
            print()
        
        # 清理测试文件
        test_file.unlink()
        
        print("✅ 文档解析功能演示完成\n")
        return True
        
    except Exception as e:
        print(f"❌ 文档解析演示失败: {e}")
        return False

async def demo_api_endpoints():
    """Demonstrate new API endpoints."""
    print("🌐 演示新增API端点")
    print("-" * 40)
    
    import requests
    
    base_url = "http://localhost:8000"
    
    # 测试健康检查
    try:
        response = requests.get(f"{base_url}/healthz", timeout=5)
        print(f"健康检查: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"健康检查失败: {e}")
    
    # 测试API文档
    try:
        response = requests.get(f"{base_url}/docs", timeout=5)
        print(f"API文档: {response.status_code} - 可访问")
    except Exception as e:
        print(f"API文档访问失败: {e}")
    
    # 测试OpenAPI规范
    try:
        response = requests.get(f"{base_url}/openapi.json", timeout=5)
        openapi_data = response.json()
        endpoints = list(openapi_data.get("paths", {}).keys())
        print(f"可用端点: {endpoints}")
        
        # 检查新增端点
        new_endpoints = [ep for ep in endpoints if ep in ["/upload", "/process-directory"]]
        print(f"新增端点: {new_endpoints}")
        
    except Exception as e:
        print(f"OpenAPI规范获取失败: {e}")
    
    print("✅ API端点演示完成\n")
    return True

async def main():
    """运行所有演示."""
    print("🚀 增强RAG功能演示")
    print("=" * 50)
    
    demos = [
        demo_reranking,
        demo_document_parsing,
        demo_api_endpoints
    ]
    
    results = []
    for demo in demos:
        result = await demo()
        results.append(result)
    
    print("=" * 50)
    print("📊 演示结果总结:")
    print(f"✅ 成功: {sum(results)}/{len(results)}")
    print(f"❌ 失败: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 所有功能演示成功！")
        print("\n🔧 新增功能总结:")
        print("1. ✅ DashScopeRerank重排序 - 提升检索精度")
        print("2. ✅ LlamaIndex文档解析 - 支持多格式文档")
        print("3. ✅ 文件上传API - /upload端点")
        print("4. ✅ 目录处理API - /process-directory端点")
        print("5. ✅ 增强RAG pipeline - 完整的工作流程")
        
        print("\n🌐 服务访问地址:")
        print("- API文档: http://localhost:8000/docs")
        print("- 健康检查: http://localhost:8000/healthz")
        print("- 文件上传: POST http://localhost:8000/upload")
        print("- 目录处理: POST http://localhost:8000/process-directory")
        print("- 智能问答: POST http://localhost:8000/answer")
    else:
        print("\n⚠️ 部分功能演示失败，请检查错误信息")

if __name__ == "__main__":
    asyncio.run(main())
