# ContentSplit — AI Content Repurposer API

Turn one piece of content into many. Blog post → tweets, LinkedIn, NOSTR, email, video script.

## Features
- **Blog → Social**: Extract key points, generate platform-specific posts
- **Long → Short**: Summarize articles into bite-sized content
- **Tone Matching**: Adjust formality for each platform
- **Thread Generator**: Turn articles into X/Twitter threads
- **Hashtag Suggestions**: Platform-appropriate tags
- **Batch Processing**: Repurpose multiple articles at once

## API

```
POST /api/repurpose
{
  "content": "Your blog post text...",
  "source_type": "blog",
  "targets": ["twitter_thread", "linkedin", "nostr", "email_newsletter"],
  "tone": "professional",
  "max_tweets": 10
}
```

Response:
```json
{
  "twitter_thread": ["Tweet 1...", "Tweet 2..."],
  "linkedin": "LinkedIn post...",
  "nostr": "NOSTR note...",
  "email_newsletter": "Email content...",
  "hashtags": {"twitter": [...], "linkedin": [...]}
}
```

## Pricing
- Free: 5 repurposes/month
- Starter ($19/mo): 100 repurposes/month
- Pro ($49/mo): Unlimited + custom tones + API access

## Tech Stack
- FastAPI backend
- OpenAI/Claude for generation
- Stripe billing
- Deployed on Railway/Fly.io
