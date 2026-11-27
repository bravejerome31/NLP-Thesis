import os
import re
import gc
import json
import torch
import numpy as np
import scipy.special as sp
import xgboost as xgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Prevent PyTorch/XGBoost thread exhaustion & C-level OpenMP crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(2)

class TriagePipeline:
    def __init__(self):
        self.models_loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
        
        # 1. Require every artifact used by the trained stacked model.
        self.model_paths = {
            'tokenizer':   os.path.join(base_dir, 'xlmr-tokenizer'),
            'sentiment':   os.path.join(base_dir, 'xlmr-sentiment'),
            'emotion':     os.path.join(base_dir, 'xlmr-emotion'),
            'baseline':    os.path.join(base_dir, 'xlmr-baseline-priority'),
            'unfine_tuned_baseline': os.path.join(base_dir, 'xlmr-unfine-tuned-priority-control'),
            'xgboost':     os.path.join(base_dir, 'xgboost_triage.json'),
            'feature_cfg': os.path.join(base_dir, 'feature_config.json'),
            'label_maps':  os.path.join(base_dir, 'label_maps.json'),
        }

        try:
            for name, path in self.model_paths.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing artifact: {path}")

            # 2. Load the training contracts used by the saved models.
            with open(self.model_paths['feature_cfg'], encoding='utf-8') as fh:
                feature_cfg = json.load(fh)
            with open(self.model_paths['label_maps'], encoding='utf-8') as fh:
                label_maps = json.load(fh)

            self.FEATURE_NAMES = feature_cfg['feature_names']
            self.OFFICES = feature_cfg['offices']
            self.MAX_LEN = feature_cfg['max_length']
            self.sent_map = {int(v): k for k, v in label_maps['sentiment'].items()}
            self.emot_map = {int(v): k for k, v in label_maps['emotion'].items()}
            self.priority_map = {int(v): k for k, v in label_maps['priority'].items()}

            # 3. Initialize the trained stacked enhancement artifacts.
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_paths['tokenizer'])
            self.xgb_booster = xgb.Booster({"nthread": 2})
            self.xgb_booster.load_model(self.model_paths['xgboost'])
            if self.xgb_booster.num_features() != len(self.FEATURE_NAMES):
                raise ValueError(
                    f"XGBoost expects {self.xgb_booster.num_features()} features, "
                    f"but feature_config.json defines {len(self.FEATURE_NAMES)}."
                )

            self.models_loaded = True
            print("⚡ [Triage Engine] Streamlined Pipeline initialized successfully.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ [Triage Engine] Initialization failed: {e}")
            self.models_loaded = False

    @staticmethod
    def clean_feedback(text):
        text = str(text)
        if text.strip().lower() in ['nan', 'none', '']:
            return ''
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def normalize_office(val):
        val_clean = re.sub(r'\s+', ' ', str(val)).strip()
        val_clean = val_clean.replace("Registrar / Admin", "Admin").replace("Registrar/Admin", "Admin")
        
        low = val_clean.lower()
        if 'admin' in low or 'registrar' in low: return 'Admin'
        if 'osas' in low: return 'OSAS'
        if 'librar' in low: return 'Library'
        if 'infirm' in low or 'clinic' in low: return 'Infirmary'
        return 'Unassigned'

    def _run_transformer_inference(self, model_path, texts, batch_size=16):
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        model.eval()
        all_logits = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=self.MAX_LEN, return_tensors="pt"
                ).to(self.device)
                logits = model(**inputs).logits.float().cpu().numpy()
                all_logits.append(logits)
                del inputs

        del model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return np.vstack(all_logits)

    @staticmethod
    def compute_semantic_weights(texts):
        feats = []
        for t in texts:
            s = str(t)
            w_cnt = max(len(s.split()), 1)
            c_len = max(len(s), 1)
            feats.append([
                float(c_len),
                float(w_cnt),
                s.count("!") / w_cnt,
                s.count("?") / w_cnt,
                sum(1 for c in s if c.isupper()) / c_len,
            ])
        return np.array(feats, dtype=np.float32)

    def process_dataframe(self, df):
        if not self.models_loaded:
            raise RuntimeError("ML Models are not fully mounted.")

        df['Cleaned_Text'] = df['Feedback'].apply(self.clean_feedback)
        df['Norm_Office'] = df['Office'].apply(self.normalize_office)

        df_valid = df[df['Cleaned_Text'].str.len() > 2].reset_index(drop=True)
        if len(df_valid) == 0:
            return df_valid, [], [], [], [], [], []

        cleaned_batch = df_valid['Cleaned_Text'].tolist()
        num_records = len(cleaned_batch)

        # 1. Run all transformer heads used by the trained stacked model.
        s_log = self._run_transformer_inference(self.model_paths['sentiment'], cleaned_batch)
        e_log = self._run_transformer_inference(self.model_paths['emotion'], cleaned_batch)
        p_log = self._run_transformer_inference(self.model_paths['baseline'], cleaned_batch)

        s_pr = sp.softmax(s_log, axis=1)
        e_pr = sp.softmax(e_log, axis=1)
        p_pr = sp.softmax(p_log, axis=1)

        # 2. Extract confidence and decision-margin features.
        p_sorted = np.sort(p_pr, axis=1)
        conf = np.stack([
            s_pr.max(axis=1),
            e_pr.max(axis=1),
            p_pr.max(axis=1),
            p_sorted[:, -1] - p_sorted[:, -2],
        ], axis=1)

        # 3. Office OHE
        office = np.zeros((num_records, len(self.OFFICES)), dtype=np.float32)
        off_lower = df_valid['Norm_Office'].astype(str).str.lower().values
        for i, o in enumerate(self.OFFICES):
            office[:, i] = (off_lower == o.lower()).astype(np.float32)

        # 4. Semantic Weights
        weights = self.compute_semantic_weights(cleaned_batch)

        # 5. Feature fusion matching the saved 39-feature XGBoost contract.
        X_features = np.hstack([
            s_log, s_pr,
            e_log, e_pr,
            p_log, p_pr,
            conf, office, weights,
        ]).astype(np.float32)

        # 6. Pass the exact feature names expected by the trained booster.
        dmat = xgb.DMatrix(X_features, feature_names=self.FEATURE_NAMES)
        xgb_probabilities = self.xgb_booster.predict(dmat, validate_features=False)

        prop_classes = np.argmax(xgb_probabilities, axis=1).tolist()
        prop_probs = np.max(xgb_probabilities, axis=1).tolist()
        sent_classes = np.argmax(s_log, axis=1).tolist()
        emot_classes = np.argmax(e_log, axis=1).tolist()
        sent_confs = np.max(s_pr, axis=1).tolist()
        emot_confs = np.max(e_pr, axis=1).tolist()

        return (
            df_valid, prop_classes, prop_probs,
            sent_classes, emot_classes,
            sent_confs, emot_confs
        )
    
    # -------------------------------------------------------------
    # Helper methods for Developer Panel Support (routes_2.py API)
    # -------------------------------------------------------------
    
    def harmonize_data(self, df):
        df['Cleaned_Text'] = df['Feedback'].apply(self.clean_feedback)
        return df['Feedback'].tolist(), df['Cleaned_Text'].tolist()
        
    def extract_features(self, cleaned_batch, aligned_offices):
        s_log = self._run_transformer_inference(self.model_paths['sentiment'], cleaned_batch)
        e_log = self._run_transformer_inference(self.model_paths['emotion'], cleaned_batch)
        p_log = self._run_transformer_inference(self.model_paths['baseline'], cleaned_batch)

        s_pr = sp.softmax(s_log, axis=1)
        e_pr = sp.softmax(e_log, axis=1)
        p_pr = sp.softmax(p_log, axis=1)

        p_sorted = np.sort(p_pr, axis=1)
        conf = np.stack([
            s_pr.max(axis=1),
            e_pr.max(axis=1),
            p_pr.max(axis=1),
            p_sorted[:, -1] - p_sorted[:, -2],
        ], axis=1)

        office = np.zeros((len(cleaned_batch), len(self.OFFICES)), dtype=np.float32)
        for i, o in enumerate(self.OFFICES):
            office[:, i] = (np.array(aligned_offices) == o).astype(np.float32)

        weights = self.compute_semantic_weights(cleaned_batch)
        X_features = np.hstack([
            s_log, s_pr,
            e_log, e_pr,
            p_log, p_pr,
            conf, office, weights,
        ]).astype(np.float32)

        sent_classes = np.argmax(s_log, axis=1).tolist()
        emot_classes = np.argmax(e_log, axis=1).tolist()
        sent_probs = np.max(s_pr, axis=1).tolist()
        emot_probs = np.max(e_pr, axis=1).tolist()

        return X_features, sent_classes, emot_classes, sent_probs, emot_probs

    def classify_triage(self, X_features):
        dmat = xgb.DMatrix(X_features, feature_names=self.FEATURE_NAMES)
        xgb_probabilities = self.xgb_booster.predict(dmat, validate_features=False)
        prop_classes = np.argmax(xgb_probabilities, axis=1).tolist()
        prop_probs = np.max(xgb_probabilities, axis=1).tolist()
        return prop_classes, prop_probs

    def classify_unfine_tuned_priority(self, texts):
        logits = self._run_transformer_inference(
            self.model_paths['unfine_tuned_baseline'], texts
        )
        return np.argmax(sp.softmax(logits, axis=1), axis=1).tolist()