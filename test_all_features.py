#!/usr/bin/env python3
"""
Comprehensive test script for OpenScholar features:
- Score-based filtering with different threshold types
- Google feedback retrieval 
- OpenScholar reranker integration
- Self-reflective generation pipeline
"""

import subprocess
import json
import os
import time
from datetime import datetime

def run_test(name, command, description):
    """Run a test scenario and capture results"""
    print(f"\n{'='*80}")
    print(f"🧪 TEST: {name}")
    print(f"📝 Description: {description}")
    print(f"🚀 Command: {command}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)
        end_time = time.time()
        
        print(f"✅ Test completed in {end_time - start_time:.1f} seconds")
        print(f"📤 Exit code: {result.returncode}")
        
        if result.stdout:
            print("📝 Output:")
            print(result.stdout[-1000:])  # Last 1000 chars to avoid spam
            
        if result.stderr:
            print("⚠️  Errors:")
            print(result.stderr[-1000:])  # Last 1000 chars
            
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def main():
    """Run comprehensive test suite"""
    
    print("🔬 OpenScholar Comprehensive Feature Test Suite")
    print(f"🎯 Model: OpenAI GPT-4o-mini | Feedback: Google Search Only")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test input file
    input_file = "comprehensive_test_input.jsonl"
    
    # Test scenarios
    tests = [
        {
            "name": "1. Basic Score Filtering (Average Threshold)",
            "command": f"""python run.py \\
                --input_file {input_file} \\
                --model_name "gpt-4o-mini" \\
                --api "openai" \\
                --api_key_fp $OPENAI_API_KEY \\
                --output_file test_score_avg.json \\
                --use_contexts \\
                --ranking_ce \\
                --reranker OpenScholar/OpenScholar_Reranker \\
                --use_score_threshold \\
                --score_threshold_type average \\
                --zero_shot""",
            "description": "Tests score-based filtering with average threshold using OpenScholar reranker and GPT-4o-mini"
        },
        
        {
            "name": "2. Aggressive Score Filtering (75th Percentile)",
            "command": f"""python run.py \\
                --input_file {input_file} \\
                --model_name "gpt-4o-mini" \\
                --api "openai" \\
                --api_key_fp $OPENAI_API_KEY \\
                --output_file test_score_p75.json \\
                --use_contexts \\
                --ranking_ce \\
                --reranker OpenScholar/OpenScholar_Reranker \\
                --use_score_threshold \\
                --score_threshold_type percentile_75 \\
                --zero_shot""",
            "description": "Tests aggressive score filtering keeping only top 25% of documents with GPT-4o-mini"
        },
        
        {
            "name": "3. Google Feedback Retrieval",
            "command": f"""python run.py \\
                --input_file {input_file} \\
                --model_name "gpt-4o-mini" \\
                --api "openai" \\
                --api_key_fp $OPENAI_API_KEY \\
                --output_file test_google_feedback.json \\
                --use_contexts \\
                --feedback \\
                --use_google_feedback \\
                --ranking_ce \\
                --reranker OpenScholar/OpenScholar_Reranker \\
                --zero_shot""",
            "description": "Tests Google feedback retrieval with GPT-4o-mini"
        },
        
        {
            "name": "4. Combined Features - Score Filtering + Google Feedback",
            "command": f"""python run.py \\
                --input_file {input_file} \\
                --model_name "gpt-4o-mini" \\
                --api "openai" \\
                --api_key_fp $OPENAI_API_KEY \\
                --output_file test_combined.json \\
                --use_contexts \\
                --feedback \\
                --use_google_feedback \\
                --ranking_ce \\
                --reranker OpenScholar/OpenScholar_Reranker \\
                --use_score_threshold \\
                --score_threshold_type average \\
                --zero_shot""",
            "description": "Tests combined score filtering + Google feedback retrieval with GPT-4o-mini"
        },
        
        {
            "name": "5. Full Pipeline with Google Feedback",
            "command": f"""python run.py \\
                --input_file {input_file} \\
                --model_name "gpt-4o-mini" \\
                --api "openai" \\
                --api_key_fp $OPENAI_API_KEY \\
                --output_file test_full_pipeline.json \\
                --use_contexts \\
                --feedback \\
                --use_google_feedback \\
                --ranking_ce \\
                --reranker OpenScholar/OpenScholar_Reranker \\
                --use_score_threshold \\
                --score_threshold_type percentile_75 \\
                --feedback_threshold_type percentile_90 \\
                --posthoc_at \\
                --use_abstract \\
                --norm_cite \\
                --zero_shot""",
            "description": "Tests complete pipeline: score filtering + Google feedback + post-hoc attribution + abstracts with GPT-4o-mini"
        },
        
        {
            "name": "6. Threshold Comparison Test (Median)",
            "command": f"""python run.py \\
                --input_file {input_file} \\
                --model_name "gpt-4o-mini" \\
                --api "openai" \\
                --api_key_fp $OPENAI_API_KEY \\
                --output_file test_median_threshold.json \\
                --use_contexts \\
                --ranking_ce \\
                --reranker OpenScholar/OpenScholar_Reranker \\
                --use_score_threshold \\
                --score_threshold_type median \\
                --zero_shot""",
            "description": "Tests median threshold for comparison with average threshold results using GPT-4o-mini"
        },
        
        {
            "name": "7. Conservative Score Filtering (25th Percentile)",
            "command": f"""python run.py \\
                --input_file {input_file} \\
                --model_name "gpt-4o-mini" \\
                --api "openai" \\
                --api_key_fp $OPENAI_API_KEY \\
                --output_file test_conservative.json \\
                --use_contexts \\
                --ranking_ce \\
                --reranker OpenScholar/OpenScholar_Reranker \\
                --use_score_threshold \\
                --score_threshold_type percentile_25 \\
                --zero_shot""",
            "description": "Tests conservative score filtering keeping top 75% of documents with GPT-4o-mini"
        }
    ]
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  WARNING: OPENAI_API_KEY environment variable not set!")
        print("   Please set it with: export OPENAI_API_KEY='your-api-key-here'")
        print("   Tests may fail without proper API key configuration.\n")
    
    # Run all tests
    results = []
    for test in tests:
        success = run_test(test["name"], test["command"], test["description"])
        results.append({
            "name": test["name"],
            "success": success,
            "description": test["description"]
        })
        
        # Small delay between tests
        time.sleep(2)
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 TEST SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    
    for result in results:
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{status} {result['name']}")
    
    print(f"\n🎯 Overall: {successful}/{total} tests passed ({successful/total*100:.1f}%)")
    
    # List output files created
    print(f"\n📁 Output files created:")
    output_files = [
        "test_score_avg.json",
        "test_score_p75.json", 
        "test_google_feedback.json",
        "test_combined.json",
        "test_full_pipeline.json",
        "test_median_threshold.json",
        "test_conservative.json"
    ]
    
    for file in output_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"  📄 {file} ({size:,} bytes)")
    
    print(f"\n🏁 Test suite completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Configuration: GPT-4o-mini + Google Feedback + OpenScholar Reranker")
    print(f"🔑 API: OpenAI (requires OPENAI_API_KEY environment variable)")

if __name__ == "__main__":
    main() 