"""
Evaluation of External Rock Subgenres on Phase 8 Clean Models.
Tests Classic Rock, Hard Rock, Alternative Grunge, and Modern Indie Rock.
"""
import json
from pathlib import Path
from ml.inference.predictor import GenrePredictor

def evaluate_subgenres():
    predictor = GenrePredictor()
    
    subgenres = [
        {
            "name": "Classic Rock (70s Style Overdrive)",
            "path": "dataset/external_test/rock_subgenres/classic_rock.wav",
            "expected": "rock",
        },
        {
            "name": "Hard Rock (High-Gain Power Chords)",
            "path": "dataset/external_test/rock_subgenres/hard_rock.wav",
            "expected": "rock / metal",
        },
        {
            "name": "Alternative / Grunge Rock (Fuzz Bass / Modulated)",
            "path": "dataset/external_test/rock_subgenres/alternative_grunge.wav",
            "expected": "rock",
        },
        {
            "name": "Modern Indie Rock (rock.mp3 / High Sheen Master)",
            "path": "dataset/external_test/rock_modern.mp3",
            "expected": "rock",
        },
    ]

    results = []
    print("\n=======================================================")
    print("EXTERNAL ROCK SUBGENRE EVALUATION ON ACTIVE PHASE 8 MODELS")
    print("=======================================================")

    for item in subgenres:
        pred = predictor.predict_audio(item["path"])
        ann = pred["ann_prediction"]
        cnn = pred["cnn_prediction"]

        res = {
            "subgenre": item["name"],
            "path": item["path"],
            "expected": item["expected"],
            "ann": {
                "pred": ann["predicted_genre"],
                "conf": round(ann["confidence"], 4),
                "entropy": ann["uncertainty"]["entropy"],
                "top3": ann["top3"],
            },
            "cnn": {
                "pred": cnn["predicted_genre"],
                "conf": round(cnn["confidence"], 4),
                "entropy": cnn["uncertainty"]["entropy"],
                "top3": cnn["top3"],
            },
            "models_agree": pred["models_agree"],
            "uncertainty_status": pred["uncertainty_status"],
        }
        results.append(res)

        ann_top3_str = ", ".join([f"{x['genre']}: {x['probability']*100:.1f}%" for x in ann["top3"]])
        cnn_top3_str = ", ".join([f"{x['genre']}: {x['probability']*100:.1f}%" for x in cnn["top3"]])
        print(f"\n--- {item['name']} ---")
        print(f"  Expected: {item['expected']}")
        print(f"  ANN: {ann['predicted_genre']} ({ann['confidence']*100:.1f}%) | Entropy: {ann['uncertainty']['entropy']:.2f}")
        print(f"       Top 3: {ann_top3_str}")
        print(f"  CNN: {cnn['predicted_genre']} ({cnn['confidence']*100:.1f}%) | Entropy: {cnn['uncertainty']['entropy']:.2f}")
        print(f"       Top 3: {cnn_top3_str}")
        print(f"  Agreement: {pred['models_agree']} | Uncertainty: {pred['uncertainty_status']}")

    with open("results/rock_subgenres_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved subgenre evaluation to results/rock_subgenres_evaluation.json")

if __name__ == "__main__":
    evaluate_subgenres()
