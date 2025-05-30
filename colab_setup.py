#!/usr/bin/env python3
"""
OpenScholar v2.0.0 - Google Colab Setup Script
==============================================

This script automates the setup and testing of OpenScholar v2.0.0 
in Google Colab environment.

Usage:
    python colab_setup.py

Features:
- Automatic environment setup
- API key configuration
- Test data creation
- Multi-source demo execution
- Results analysis
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path

class ColabSetup:
    def __init__(self):
        self.repo_url = "https://github.com/ryanchen0327/OpenScholarForSciFy.git"
        self.repo_dir = "OpenScholarForSciFy"
        
    def print_header(self, title):
        """Print a formatted header"""
        print(f"\n{'='*60}")
        print(f"🎓 {title}")
        print(f"{'='*60}")
    
    def run_command(self, command, description=""):
        """Run a shell command and handle errors"""
        if description:
            print(f"▶️ {description}")
        
        try:
            result = subprocess.run(command, shell=True, check=True, 
                                  capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Error: {e}")
            if e.stderr:
                print(f"Details: {e.stderr}")
            return False
    
    def setup_environment(self):
        """Setup the Colab environment"""
        self.print_header("Environment Setup")
        
        # Check if we're in Colab
        try:
            import google.colab
            print("✅ Running in Google Colab")
        except ImportError:
            print("⚠️ Not running in Google Colab, but continuing...")
        
        # Update system
        print("📦 Updating system packages...")
        self.run_command("apt-get update -qq", "Updating package lists")
        
        # Clone repository
        if os.path.exists(self.repo_dir):
            print(f"📁 Repository {self.repo_dir} already exists, updating...")
            os.chdir(self.repo_dir)
            self.run_command("git pull", "Updating repository")
        else:
            print(f"📥 Cloning repository from {self.repo_url}")
            self.run_command(f"git clone {self.repo_url}", "Cloning repository")
            os.chdir(self.repo_dir)
        
        # Verify location
        current_dir = os.getcwd()
        print(f"📍 Current directory: {current_dir}")
        
        return True
    
    def install_dependencies(self):
        """Install Python dependencies"""
        self.print_header("Installing Dependencies")
        
        # Install requirements
        if os.path.exists("requirements.txt"):
            print("📋 Installing from requirements.txt...")
            self.run_command("pip install -r requirements.txt", "Installing requirements")
        
        # Install additional Colab-specific packages
        additional_packages = [
            "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
            "transformers>=4.20.0",
            "accelerate",
            "sentence-transformers", 
            "FlagEmbedding"
        ]
        
        for package in additional_packages:
            print(f"📦 Installing {package.split()[0]}...")
            self.run_command(f"pip install {package}")
        
        # Install spacy model
        print("🔤 Downloading spacy model...")
        self.run_command("python -m spacy download en_core_web_sm", "Installing spacy model")
        
        return True
    
    def configure_apis(self):
        """Configure API keys interactively"""
        self.print_header("API Configuration")
        
        print("🔑 API Key Setup (All Optional)")
        print("Press Enter to skip any API you don't have\n")
        
        # Semantic Scholar
        s2_key = input("Semantic Scholar API Key (or Enter for dummy): ").strip()
        os.environ['S2_API_KEY'] = s2_key if s2_key else 'dummy_key'
        
        # Google Custom Search
        google_key = input("Google Custom Search API Key (optional): ").strip()
        if google_key:
            os.environ['GOOGLE_API_KEY'] = google_key
            google_cx = input("Google Custom Search Engine ID: ").strip()
            os.environ['GOOGLE_CX'] = google_cx
        
        # You.com
        you_key = input("You.com API Key (optional): ").strip()
        if you_key:
            os.environ['YOUR_API_KEY'] = you_key
        
        # Summary
        print("\n✅ API Configuration Summary:")
        print(f"  Semantic Scholar: {'✅' if os.environ.get('S2_API_KEY') else '❌'}")
        print(f"  Google Search: {'✅' if os.environ.get('GOOGLE_API_KEY') else '❌ (optional)'}")
        print(f"  You.com Search: {'✅' if os.environ.get('YOUR_API_KEY') else '❌ (optional)'}")
        
        return True
    
    def create_test_data(self):
        """Create test input files"""
        self.print_header("Creating Test Data")
        
        test_questions = [
            {
                "question": "What are the latest developments in transformer models for natural language processing?",
                "id": "q1"
            },
            {
                "question": "How do large language models handle reasoning tasks?", 
                "id": "q2"
            },
            {
                "question": "What are the environmental impacts of training large neural networks?",
                "id": "q3"
            }
        ]
        
        # Save test file
        test_file = "colab_test_input.jsonl"
        with open(test_file, 'w') as f:
            for q in test_questions:
                f.write(json.dumps(q) + '\n')
        
        print(f"✅ Created test file: {test_file}")
        print("\n📋 Test Questions:")
        for i, q in enumerate(test_questions, 1):
            print(f"  {i}. {q['question']}")
        
        return test_file
    
    def run_basic_demo(self, input_file):
        """Run basic OpenScholar demo"""
        self.print_header("Basic OpenScholar Demo")
        
        output_file = "colab_basic_output.json"
        
        cmd = [
            "python run.py",
            f"--input_file {input_file}",
            "--model_name gpt2",
            "--use_contexts",
            "--use_score_threshold",
            "--score_threshold_type average", 
            f"--output_file {output_file}",
            "--zero_shot"
        ]
        
        command = " ".join(cmd)
        print(f"🚀 Running: {command}")
        
        if self.run_command(command, "Basic demo execution"):
            print(f"✅ Basic demo completed! Output: {output_file}")
            return output_file
        else:
            print("❌ Basic demo failed")
            return None
    
    def run_multisource_demo(self, input_file):
        """Run multi-source feedback demo"""
        self.print_header("Multi-Source Feedback Demo")
        
        output_file = "colab_multisource_output.json"
        
        cmd = [
            "python run.py",
            f"--input_file {input_file}",
            "--model_name gpt2",
            "--use_contexts",
            "--feedback",
            "--ss_retriever",
            "--use_score_threshold",
            "--score_threshold_type average",
            f"--output_file {output_file}",
            "--zero_shot"
        ]
        
        # Add optional APIs if available
        if os.environ.get('GOOGLE_API_KEY'):
            cmd.append("--use_google_feedback")
        if os.environ.get('YOUR_API_KEY'):
            cmd.append("--use_youcom_feedback")
        
        command = " ".join(cmd)
        print(f"🚀 Running: {command}")
        
        if self.run_command(command, "Multi-source demo execution"):
            print(f"✅ Multi-source demo completed! Output: {output_file}")
            return output_file
        else:
            print("❌ Multi-source demo failed")
            return None
    
    def analyze_results(self, output_files):
        """Analyze and display results"""
        self.print_header("Results Analysis")
        
        for output_file in output_files:
            if not output_file or not os.path.exists(output_file):
                continue
                
            print(f"\n📄 Analysis of {output_file}:")
            print("-" * 40)
            
            try:
                with open(output_file, 'r') as f:
                    results = json.load(f)
                
                for i, result in enumerate(results, 1):
                    question = result.get('question', 'N/A')
                    answer = result.get('output', 'N/A')
                    contexts = result.get('ctxs', [])
                    
                    print(f"\n🔸 Question {i}: {question[:80]}...")
                    print(f"📝 Answer: {answer[:150]}...")
                    print(f"📚 Retrieved documents: {len(contexts)}")
                    
                    if contexts:
                        # Show source diversity
                        sources = set(ctx.get('type', 'unknown') for ctx in contexts)
                        print(f"🔍 Sources: {', '.join(sources)}")
                    
                    print()
                    
            except Exception as e:
                print(f"❌ Error analyzing {output_file}: {e}")
    
    def run_interactive_test(self):
        """Run interactive question testing"""
        self.print_header("Interactive Testing")
        
        print("🎯 Enter your own research question:")
        custom_question = input("Your question: ").strip()
        
        if not custom_question:
            print("❌ No question provided, skipping interactive test")
            return
        
        # Create custom input file
        custom_input = {"question": custom_question, "id": "custom"}
        custom_file = "custom_input.jsonl"
        
        with open(custom_file, 'w') as f:
            json.dump(custom_input, f)
        
        # Run with best available configuration
        output_file = "custom_output.json"
        
        cmd = [
            "python run.py",
            f"--input_file {custom_file}",
            "--model_name gpt2",
            "--use_contexts",
            "--feedback",
            "--ss_retriever",
            "--use_score_threshold",
            "--score_threshold_type percentile_75",
            f"--output_file {output_file}",
            "--zero_shot"
        ]
        
        if os.environ.get('GOOGLE_API_KEY'):
            cmd.append("--use_google_feedback")
        if os.environ.get('YOUR_API_KEY'):
            cmd.append("--use_youcom_feedback")
        
        command = " ".join(cmd)
        print(f"🚀 Running custom question: {command}")
        
        if self.run_command(command, "Custom question processing"):
            self.analyze_results([output_file])
        else:
            print("❌ Custom question processing failed")
    
    def show_next_steps(self):
        """Show next steps and resources"""
        self.print_header("Next Steps & Resources")
        
        print("🎉 OpenScholar v2.0.0 Setup Complete!")
        print("\n🚀 What you can do next:")
        print("  1. 🔗 Visit: https://github.com/ryanchen0327/OpenScholarForSciFy")
        print("  2. 📖 Read the documentation files in this directory")
        print("  3. 🔑 Get API keys for full multi-source capabilities")
        print("  4. 🎯 Try more complex research questions")
        print("  5. ⭐ Star the repository if you find it useful!")
        
        print("\n📚 Documentation available:")
        docs = [
            "README.md - Main documentation",
            "MULTI_SOURCE_FEEDBACK_README.md - Multi-source setup guide", 
            "SCORE_FILTERING_README.md - Filtering documentation",
            "CHANGELOG.md - What's new in v2.0.0",
            "LICENSE_COMPLIANCE.md - Apache 2.0 compliance"
        ]
        
        for doc in docs:
            doc_file = doc.split(' - ')[0]
            if os.path.exists(doc_file):
                print(f"  ✅ {doc}")
            else:
                print(f"  📄 {doc}")
        
        print("\n📧 Questions or Issues?")
        print("   Create an issue at: https://github.com/ryanchen0327/OpenScholarForSciFy/issues")
        
        print("\n📄 License: Apache 2.0 - Free for commercial and research use")
        print("✨ Happy researching with OpenScholar v2.0.0!")
    
    def run_full_setup(self):
        """Run the complete setup process"""
        print("🎓 OpenScholar v2.0.0 - Google Colab Setup")
        print("==========================================")
        
        # Setup steps
        steps = [
            ("Environment Setup", self.setup_environment),
            ("Install Dependencies", self.install_dependencies),
            ("Configure APIs", self.configure_apis),
        ]
        
        for step_name, step_func in steps:
            try:
                if not step_func():
                    print(f"❌ Failed at step: {step_name}")
                    return False
            except Exception as e:
                print(f"❌ Error in {step_name}: {e}")
                return False
        
        # Create test data and run demos
        try:
            test_file = self.create_test_data()
            
            # Run demos
            output_files = []
            
            print("\n🎯 Choose demos to run:")
            print("1. Basic demo (fastest)")
            print("2. Multi-source demo (recommended)")
            print("3. Both demos")
            print("4. Interactive test only")
            
            choice = input("Enter choice (1-4, or Enter for option 2): ").strip()
            if not choice:
                choice = "2"
            
            if choice in ["1", "3"]:
                basic_output = self.run_basic_demo(test_file)
                if basic_output:
                    output_files.append(basic_output)
            
            if choice in ["2", "3"]:
                multi_output = self.run_multisource_demo(test_file)
                if multi_output:
                    output_files.append(multi_output)
            
            # Analyze results
            if output_files:
                self.analyze_results(output_files)
            
            # Interactive test
            if choice == "4" or input("\n🎮 Run interactive test? (y/N): ").lower() == 'y':
                self.run_interactive_test()
            
            # Show next steps
            self.show_next_steps()
            
            return True
            
        except Exception as e:
            print(f"❌ Error in demo execution: {e}")
            return False

def main():
    """Main function to run the setup"""
    setup = ColabSetup()
    
    try:
        success = setup.run_full_setup()
        if success:
            print("\n🎉 Setup completed successfully!")
        else:
            print("\n❌ Setup encountered some issues")
            
    except KeyboardInterrupt:
        print("\n⚠️ Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 