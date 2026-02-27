import json
import re
import math

class QualityScorer:
    def __init__(self, weights=None):
        self.weights = weights or {
            "completeness": 0.30,
            "format_compliance": 0.20,
            "coverage": 0.25,
            "clarity": 0.15,
            "validity": 0.10
        }
        self.pass_threshold = 0.70

    def score_submission(self, content, submission_type="json"):
        scores = {
            "completeness": 0.0,
            "format_compliance": 0.0,
            "coverage": 0.0,
            "clarity": 0.0,
            "validity": 0.0
        }
        feedback = []

        # 1. Format Compliance
        if submission_type == "json":
            try:
                data = json.loads(content)
                scores["format_compliance"] = 1.0
                scores["validity"] = 1.0
            except:
                scores["format_compliance"] = 0.2
                feedback.append("Invalid JSON format detected.")
        elif submission_type == "markdown":
            if content.startswith("#"): scores["format_compliance"] += 0.5
            if "##" in content: scores["format_compliance"] += 0.5
            if scores["format_compliance"] < 1.0: feedback.append("Weak Markdown structure.")
        
        # 2. Completeness (Length and structure density)
        words = content.split()
        if len(words) > 200: scores["completeness"] = 1.0
        elif len(words) > 50: scores["completeness"] = 0.6
        else: feedback.append("Content is too brief.")

        # 3. Coverage (Diversity of keys/headers)
        if submission_type == "json":
            keys = re.findall(r'"([^"]+)"\s*:', content)
            if len(set(keys)) > 8: scores["coverage"] = 1.0
            else: scores["coverage"] = len(set(keys)) / 8.0
        
        # 4. Clarity (Readability heuristic)
        avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
        if 4 < avg_word_len < 10: scores["clarity"] = 1.0
        else: scores["clarity"] = 0.5

        # 5. Validity (Placeholder for logic check)
        if scores["validity"] == 0: scores["validity"] = 0.8 # Assume valid if readable

        # Calculate weighted score
        weighted_score = sum(scores[dim] * self.weights[dim] for dim in scores)
        
        quality_rating = "Excellent" if weighted_score > 0.9 else "Good" if weighted_score > 0.7 else "Fair" if weighted_score > 0.5 else "Poor"
        
        return {
            "weighted_score": round(weighted_score, 4),
            "quality_rating": quality_rating,
            "scores": {k: round(v, 2) for k, v in scores.items()},
            "feedback": feedback,
            "pass_threshold": weighted_score >= self.pass_threshold
        }

if __name__ == "__main__":
    scorer = QualityScorer()
    sample = '{"title": "Research Report", "content": "This is a detailed analysis of AI agent revenue streams...", "metadata": {"author": "Pioneer", "date": "2026-02-28"}, "tags": ["finance", "ai"]}'
    print(json.dumps(scorer.score_submission(sample, "json"), indent=2))
