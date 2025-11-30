"""
Run All Tests and Generate Report

This master script runs all 4 tests sequentially and generates comprehensive report:
1. RoWeeder Baseline on WeedMap
2. RoWeeder Baseline on Agriculture-Vision
3. Phase 1 Finetuned on Agriculture-Vision
4. Phase 1 Finetuned on WeedMap
5. Generate comprehensive report

Usage:
    python run_all_tests.py --weedmap_path /path/to/weedmap

Optional: Run specific tests only
    python run_all_tests.py --weedmap_path /path/to/weedmap --tests 1,2,3
"""
import subprocess
import sys
from pathlib import Path
import argparse
import time
import sys
from pathlib import Path


def run_test(script_name, args_dict, test_number, test_name):
    """Run a single test script"""
    
    print("\n" + "="*80)
    print(f"TEST {test_number}: {test_name}")
    print("="*80)
    
    # Build command
    cmd = [sys.executable, script_name]
    for key, value in args_dict.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"Command: {' '.join(cmd)}\n")
    
    # Run
    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"\n✅ Test {test_number} completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ Test {test_number} failed after {elapsed/60:.1f} minutes")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Test {test_number} interrupted by user")
        raise


def main():
    parser = argparse.ArgumentParser(description='Run all tests and generate report')
    parser.add_argument('--weedmap_path', type=str, required=True,
                       help='Path to WeedMap dataset')
    parser.add_argument('--agriculture_path', type=str,
                       default='dataset/finetuning_dataset/agriculture_vision_2019',
                       help='Path to Agriculture-Vision dataset')
    parser.add_argument('--checkpoint', type=str,
                       default='outputs/agriculture_vision_v2/best_model.pth',
                       help='Path to Phase 1 checkpoint')
    parser.add_argument('--output_dir', type=str,
                       default='outputs/test_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--tests', type=str, default='1,2,3,4',
                       help='Which tests to run (comma-separated, e.g., "1,2,3,4")')
    parser.add_argument('--skip_report', action='store_true',
                       help='Skip final report generation')
    
    args = parser.parse_args()
    
    # Parse which tests to run
    tests_to_run = [int(x) for x in args.tests.split(',')]
    
    print("="*80)
    print("COMPREHENSIVE TESTING CAMPAIGN")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  WeedMap: {args.weedmap_path}")
    print(f"  Agriculture-Vision: {args.agriculture_path}")
    print(f"  Phase 1 Checkpoint: {args.checkpoint}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"  Device: {args.device}")
    print(f"  Tests to run: {tests_to_run}")
    print()
    
    # Verify paths
    weedmap_path = Path(args.weedmap_path)
    agriculture_path = Path(args.agriculture_path)
    checkpoint_path = Path(args.checkpoint)
    
    if not weedmap_path.exists():
        print(f"❌ WeedMap path not found: {weedmap_path}")
        print("   Please download WeedMap dataset first!")
        return
    
    if not agriculture_path.exists():
        print(f"❌ Agriculture-Vision path not found: {agriculture_path}")
        return
    
    if 3 in tests_to_run or 4 in tests_to_run:
        if not checkpoint_path.exists():
            print(f"⚠️  Phase 1 checkpoint not found: {checkpoint_path}")
            print("   Tests 3 and 4 will be skipped!")
            tests_to_run = [t for t in tests_to_run if t in [1, 2]]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run tests
    results = {}
    total_start = time.time()
    
    # Test 1: RoWeeder on WeedMap
    if 1 in tests_to_run:
        success = run_test(
            'test_baseline_weedmap.py',
            {
                'weedmap_path': args.weedmap_path,
                'output_dir': args.output_dir,
                'device': args.device
            },
            1,
            "RoWeeder Baseline on WeedMap (verify ~75% F1)"
        )
        results['test1'] = success
    
    # Test 2: RoWeeder on Agriculture-Vision
    if 2 in tests_to_run:
        success = run_test(
            'test_baseline_av.py',
            {
                'agriculture_path': args.agriculture_path,
                'output_dir': args.output_dir,
                'device': args.device
            },
            2,
            "RoWeeder Baseline on Agriculture-Vision (show domain gap <40%)"
        )
        results['test2'] = success
    
    # Test 3: Phase 1 on Agriculture-Vision
    if 3 in tests_to_run:
        success = run_test(
            'test_finetuned_av.py',
            {
                'checkpoint': args.checkpoint,
                'agriculture_path': args.agriculture_path,
                'output_dir': args.output_dir,
                'device': args.device
            },
            3,
            "Phase 1 Finetuned on Agriculture-Vision (confirm 67% F1)"
        )
        results['test3'] = success
    
    # Test 4: Phase 1 on WeedMap
    if 4 in tests_to_run:
        success = run_test(
            'test_finetuned_weedmap.py',
            {
                'checkpoint': args.checkpoint,
                'weedmap_path': args.weedmap_path,
                'output_dir': args.output_dir,
                'device': args.device
            },
            4,
            "Phase 1 Finetuned on WeedMap (show domain specificity)"
        )
        results['test4'] = success
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "="*80)
    print("TESTING CAMPAIGN SUMMARY")
    print("="*80)
    
    completed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\nTests Completed: {completed}/{total}")
    print(f"Total Time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.2f} hours)")
    print()
    
    for test_num, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  Test {test_num[-1]}: {status}")
    
    # Generate report
    if not args.skip_report:
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        try:
            subprocess.run([
                sys.executable,
                'generate_report.py',
                '--results_dir', args.output_dir,
                '--output_dir', args.output_dir
            ], check=True)
            
            print("\n✅ Report generated successfully!")
            print(f"📄 View report: {output_dir}/COMPREHENSIVE_TESTING_REPORT.txt")
            print(f"📊 View plot: {output_dir}/model_comparison_plot.png")
            
        except subprocess.CalledProcessError:
            print("\n❌ Report generation failed")
    
    print("\n" + "="*80)
    print("ALL DONE!")
    print("="*80)
    
    # Exit code
    if all(results.values()):
        print("\n🎉 All tests passed! Your testing campaign is complete!")
        sys.exit(0)
    else:
        print("\n⚠️  Some tests failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(130)