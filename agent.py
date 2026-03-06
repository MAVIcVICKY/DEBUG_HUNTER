"""
🐛 Agentic Bug Hunter - Main Pipeline Orchestrator
=====================================================
Orchestrates the complete multi-agent pipeline:

  Dataset Loader → Code Parser → Bug Detector → MCP Retriever
      → Explanation Generator → Validator → CSV Output

Usage:
    # Set your Gemini API key
    set GEMINI_API_KEY=your_key_here

    # Make sure MCP server is running first:
    cd server && python mcp_server.py

    # Then run the agent pipeline:
    python agent.py

    # Or analyze a single sample:
    python agent.py --sample 16
"""

import asyncio
import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.dataset_loader import DatasetLoaderAgent
from agents.code_parser import CodeParserAgent
from agents.bug_detector import BugDetectionAgent
from agents.mcp_retriever import MCPRetrievalAgent
from agents.explanation_agent import ExplanationAgent
from agents.validator import ValidationAgent
from agents.csv_output import CSVOutputAgent


class AgenticBugHunter:
    """Main orchestrator for the multi-agent bug hunting pipeline."""

    def __init__(self, gemini_api_key: str, mcp_server_url: str = "http://localhost:8003"):
        print("\n" + "=" * 60)
        print("  🐛 AGENTIC BUG HUNTER - Initializing Pipeline")
        print("=" * 60)

        # Initialize all agents
        self.dataset_loader = DatasetLoaderAgent()
        self.code_parser = CodeParserAgent()
        self.bug_detector = BugDetectionAgent(gemini_api_key)
        self.mcp_retriever = MCPRetrievalAgent(mcp_server_url)
        self.explanation_agent = ExplanationAgent(gemini_api_key)
        self.validator = ValidationAgent()
        self.csv_output = CSVOutputAgent()

        print("\n✅ All 7 agents initialized:")
        print("  1. DatasetLoaderAgent")
        print("  2. CodeParserAgent")
        print("  3. BugDetectionAgent (Gemini LLM)")
        print("  4. MCPRetrievalAgent (MCP Server)")
        print("  5. ExplanationAgent (Gemini LLM)")
        print("  6. ValidationAgent")
        print("  7. CSVOutputAgent")
        print()

    async def run_pipeline(self, sample_id: int = None) -> str:
        """
        Run the complete multi-agent pipeline.
        
        Args:
            sample_id: Optional specific sample ID to analyze.
                       If None, processes all samples.
                       
        Returns:
            Path to the generated output CSV.
        """
        start_time = time.time()

        # ─── Step 1: Load Dataset ───
        print("\n" + "─" * 50)
        print("📦 STEP 1: Loading Dataset")
        print("─" * 50)
        
        if sample_id is not None:
            sample = self.dataset_loader.load_single_sample(sample_id)
            if sample is None:
                print(f"❌ Sample ID {sample_id} not found!")
                return None
            samples = [sample]
        else:
            samples = self.dataset_loader.load_all_samples()

        # Process each sample through the pipeline
        processed_results = []
        
        for i, sample in enumerate(samples):
            print(f"\n{'🔄' * 20}")
            print(f"  Processing Sample {i+1}/{len(samples)} (ID: {sample['id']})")
            print(f"{'🔄' * 20}")

            try:
                # ─── Step 2: Parse Code ───
                print(f"\n  📝 STEP 2: Parsing Code...")
                sample = self.code_parser.parse(sample)

                # ─── Step 3: Detect Bugs ───
                print(f"\n  🔍 STEP 3: Detecting Bugs (LLM)...")
                sample = self.bug_detector.detect(sample)

                # ─── Step 4: Retrieve Documentation ───
                print(f"\n  📚 STEP 4: Retrieving MCP Documentation...")
                sample = await self.mcp_retriever.retrieve(sample)

                # ─── Step 5: Generate Explanations ───
                print(f"\n  💬 STEP 5: Generating Explanations (LLM)...")
                sample = self.explanation_agent.explain(sample)

                # ─── Step 6: Validate Output ───
                print(f"\n  ✔️  STEP 6: Validating Results...")
                sample = self.validator.validate(sample)

                processed_results.append(sample)

                # Print intermediate result
                self._print_sample_result(sample)

            except Exception as e:
                print(f"\n  ❌ ERROR processing sample {sample['id']}: {e}")
                import traceback
                traceback.print_exc()
                # Add partial result
                sample["explanations"] = {
                    "explanations": [{
                        "bug_line": 0, 
                        "explanation": f"Processing error: {str(e)}"
                    }]
                }
                processed_results.append(sample)

            # Small delay to avoid rate limiting
            if i < len(samples) - 1:
                await asyncio.sleep(1)

        # ─── Step 7: Generate CSV Output ───
        print(f"\n{'─' * 50}")
        print("📊 STEP 7: Generating CSV Output")
        print("─" * 50)
        
        output_path = self.csv_output.generate(processed_results)

        # Final summary
        elapsed = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"  🎉 PIPELINE COMPLETE")
        print(f"  Samples processed: {len(processed_results)}")
        print(f"  Time elapsed: {elapsed:.1f}s")
        print(f"  Output: {output_path}")
        print(f"{'=' * 60}\n")

        return output_path

    def _print_sample_result(self, sample: dict):
        """Print a formatted result for a single sample."""
        print(f"\n  {'─' * 46}")
        print(f"  📋 Result for Sample {sample['id']}:")
        
        explanations = sample.get("explanations", {}).get("explanations", [])
        for exp in explanations:
            line = exp.get("bug_line", "?")
            explanation = exp.get("explanation", "N/A")
            if len(explanation) > 80:
                explanation = explanation[:77] + "..."
            print(f"    Line {line}: {explanation}")
        
        # Compare with ground truth
        gt_explanation = sample.get("explanation", "")
        if gt_explanation:
            print(f"  📌 Ground Truth: {gt_explanation[:80]}...")
        
        validation = sample.get("validation", {})
        if validation:
            status = "✅" if validation.get("all_valid") else "⚠️"
            print(f"  {status} Validation: {validation.get('valid_bugs', 0)}/{validation.get('total_bugs', 0)} valid")


def main():
    parser = argparse.ArgumentParser(
        description="🐛 Agentic Bug Hunter - Multi-Agent Bug Detection System"
    )
    parser.add_argument(
        "--sample", "-s", type=int, default=None,
        help="Analyze a specific sample ID (default: all samples)"
    )
    parser.add_argument(
        "--mcp-url", type=str, default="http://localhost:8003",
        help="MCP Server URL (default: http://localhost:8003)"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    args = parser.parse_args()

    # Get API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("❌ ERROR: Gemini API key required!")
        print("   Set it via: set GEMINI_API_KEY=your_key")
        print("   Or pass: python agent.py --api-key your_key")
        sys.exit(1)

    # Create and run pipeline
    hunter = AgenticBugHunter(
        gemini_api_key=api_key,
        mcp_server_url=args.mcp_url
    )

    output_path = asyncio.run(hunter.run_pipeline(sample_id=args.sample))
    
    if output_path:
        print(f"✅ Results saved to: {output_path}")
    else:
        print("❌ Pipeline failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
