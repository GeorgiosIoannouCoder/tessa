import importlib.util, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
MOD_PATH = ROOT / "code" / "models" / "svm_senti.py"

spec = importlib.util.spec_from_file_location("svm_senti", MOD_PATH)
svm_senti = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(svm_senti)

def test_svm_small_sample_runs():
    out = svm_senti.train_eval(limit=1200, min_df=1, ngram_max=1,
                               save_dir="models_performance/svm_test", seed=0)
    assert out["macro_f1"] >= 0.42  # non-trivial but robust floor
