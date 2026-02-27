import json
from scorer import QualityScorer

samples = [
    {"type": "json", "content": '{"name": "test"}'},
    {"type": "json", "content": '{"title": "Full Analysis", "body": "Long body text...", "meta": {"a": 1, "b": 2, "c": 3}, "extra": [1,2,3,4,5,6,7,8,9,0]}'},
    {"type": "markdown", "content": "# Header\nContent"},
    {"type": "markdown", "content": "# Big Report\n## Section 1\nDetailed content about agents and revenue models for 2026."}
] * 5 # Duplicate to get 20

scorer = QualityScorer()
results = []
for i, s in enumerate(samples):
    results.append({
        "submission_id": i + 1,
        "result": scorer.score_submission(s["content"], s["type"])
    })

with open("scorecards.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Generated {len(results)} scorecards.")
