#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_setup.py - Verify installation and run quick test
=======================================================
Version: 1.0.0
Date: Dec 7, 2025

快速验证脚本，检查所有依赖是否正确安装。

Usage:
------
python test_setup.py
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def check_dependencies():
    """Check all required dependencies."""
    print("=" * 60)
    print("LGO vs InterpretML (EBM) - Dependency Check")
    print("=" * 60)
    
    all_ok = True
    
    # NumPy
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError:
        print("✗ NumPy: NOT INSTALLED")
        all_ok = False
    
    # Pandas
    try:
        import pandas as pd
        print(f"✓ Pandas: {pd.__version__}")
    except ImportError:
        print("✗ Pandas: NOT INSTALLED")
        all_ok = False
    
    # Scikit-learn
    try:
        import sklearn
        print(f"✓ Scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("✗ Scikit-learn: NOT INSTALLED")
        all_ok = False
    
    # SciPy
    try:
        import scipy
        print(f"✓ SciPy: {scipy.__version__}")
    except ImportError:
        print("✗ SciPy: NOT INSTALLED")
        all_ok = False
    
    # Matplotlib
    try:
        import matplotlib
        print(f"✓ Matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("✗ Matplotlib: NOT INSTALLED")
        all_ok = False
    
    # DEAP
    try:
        import deap
        print(f"✓ DEAP: {deap.__version__}")
    except ImportError:
        print("✗ DEAP: NOT INSTALLED (required for LGO)")
        all_ok = False
    
    # InterpretML
    try:
        import interpret
        print(f"✓ InterpretML: {interpret.__version__}")
    except ImportError:
        print("✗ InterpretML: NOT INSTALLED (required for EBM)")
        print("  → Install: pip install interpret")
        all_ok = False
    
    print("-" * 60)
    
    # Check local modules
    try:
        from LGO_v2_2 import run_lgo_sr_v3
        print("✓ LGO_v2_2: OK")
    except ImportError as e:
        print(f"✗ LGO_v2_2: {e}")
        all_ok = False
    
    try:
        from InterpretML_v1 import get_ebm, INTERPRET_AVAILABLE
        if INTERPRET_AVAILABLE:
            print("✓ InterpretML_v1: OK")
        else:
            print("✗ InterpretML_v1: InterpretML package not available")
            all_ok = False
    except ImportError as e:
        print(f"✗ InterpretML_v1: {e}")
        all_ok = False
    
    try:
        from run_lgo_interpret_comparison import LGOvsEBMExperiment
        print("✓ run_lgo_interpret_comparison: OK")
    except ImportError as e:
        print(f"✗ run_lgo_interpret_comparison: {e}")
        all_ok = False
    
    print("=" * 60)
    
    if all_ok:
        print("All dependencies OK! ✓")
        return True
    else:
        print("Some dependencies missing. Please install them.")
        return False


def run_quick_test():
    """Run a quick test with synthetic data."""
    print("\n" + "=" * 60)
    print("Quick Test with Synthetic Data")
    print("=" * 60)
    
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    
    X = pd.DataFrame({
        'age': np.random.normal(65, 10, n_samples),
        'heart_rate': np.random.normal(90, 15, n_samples),
        'sbp': np.random.normal(110, 20, n_samples),
        'creatinine': np.random.lognormal(-0.2, 0.5, n_samples),
    })
    
    logits = 0.3 * (X['age'] - 65) / 10 + 0.2 * (X['heart_rate'] - 90) / 15
    prob = 1 / (1 + np.exp(-logits))
    y = (np.random.random(n_samples) < prob).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data: {len(X_train)} train, {len(X_test)} test")
    print(f"Positive rate: {y_train.mean():.2%}")
    
    # Test EBM
    print("\n[Testing EBM]")
    try:
        from InterpretML_v1 import get_ebm
        ebm = get_ebm(verbose=False, random_state=42, interactions=0)
        ebm.fit(X_train, y_train, task='classification')
        y_prob = ebm.predict_proba(X_test)
        
        from sklearn.metrics import roc_auc_score
        auroc = roc_auc_score(y_test, y_prob)
        print(f"  EBM AUROC: {auroc:.4f} ✓")
    except Exception as e:
        print(f"  EBM Error: {e}")
    
    # Test LGO (quick run)
    print("\n[Testing LGO]")
    try:
        from LGO_v2_2 import run_lgo_sr_v3
        
        # Very small budget for quick test
        result_df = run_lgo_sr_v3(
            X=X_train.values,
            y=y_train.values,
            feature_names=list(X_train.columns),
            experiment='lgo_hard',
            pop_size=50,
            ngen=10,
            random_state=42,
        )
        
        print(f"  LGO completed, found {len(result_df)} expressions ✓")
        if len(result_df) > 0:
            print(f"  Best expression: {str(result_df.iloc[0]['expr'])[:80]}...")
    except Exception as e:
        print(f"  LGO Error: {e}")
    
    print("\n" + "=" * 60)
    print("Quick test completed!")
    print("=" * 60)


if __name__ == "__main__":
    if check_dependencies():
        try:
            run_quick_test()
        except Exception as e:
            print(f"\nQuick test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nPlease install missing dependencies first:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
