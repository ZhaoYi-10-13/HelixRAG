#!/usr/bin/env python3
"""
数据库连接测试脚本
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_database_connection():
    """测试数据库连接"""
    print("🔍 测试数据库连接...")
    
    try:
        from app.core.database import db
        import asyncio
        
        async def test_connection():
            await db.connect()
            print("✅ 数据库连接成功！")
            
            # 测试基本查询
            try:
                stats = await db.supabase.rpc('get_chunk_stats').execute()
                print(f"📊 数据库状态: {stats.data}")
                return True
            except Exception as e:
                print(f"⚠️ 数据库查询测试失败: {e}")
                return False
        
        result = asyncio.run(test_connection())
        return result
        
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}")
        print("\n🔧 请检查以下配置:")
        print("1. SUPABASE_URL 是否正确")
        print("2. SUPABASE_ANON_KEY 是否正确") 
        print("3. SUPABASE_SERVICE_ROLE_KEY 是否正确")
        print("4. 是否在 Supabase 中执行了 sql/init_supabase.sql")
        return False

if __name__ == "__main__":
    print("🚀 数据库连接测试")
    print("=" * 40)
    
    # 检查环境变量
    required_vars = ['SUPABASE_URL', 'SUPABASE_ANON_KEY', 'SUPABASE_SERVICE_ROLE_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var) or os.getenv(var).startswith('your-')]
    
    if missing_vars:
        print(f"❌ 缺少环境变量: {missing_vars}")
        print("\n请更新 .env 文件中的 Supabase 配置")
        sys.exit(1)
    
    success = test_database_connection()
    
    if success:
        print("\n🎉 数据库连接测试通过！")
        print("现在可以正常使用 RAG 功能了")
    else:
        print("\n⚠️ 数据库连接测试失败")
        print("请按照上述步骤配置 Supabase")
