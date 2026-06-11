"""
AI Movie Review Generator — template engine (zero API cost).
Reserved `llm_generate()` placeholder for future API integration.
"""

import random
import re
from typing import Optional


# ── style phrase banks ──────────────────────────────────────────────

STYLE_BANKS = {
    "hype": {
        "name": "Hype Mode",
        "label": "Hype Mode",
        "openings": [
            'If you haven\'t seen {movie} yet, stop whatever you\'re doing and go watch it. This is the movie you\'ll be telling everyone about for the rest of the year.',
            'Alright, let me tell you why {movie} is hands-down the best film I\'ve seen this year. No caveats, no "it\'s good but"—just pure excellence.',
            'A friend asked me if {movie} was worth watching. I just sent them a photo of my ticket stub. You\'ll understand once you see it.',
            'Some movies are disposable entertainment you forget by the time you reach the parking lot. {movie} is the kind that lingers in your head for weeks.',
        ],
        "positives": {
            "default": [
                'The film is remarkably well-crafted from start to finish. The opening sequence alone grabs you by the collar and doesn\'t let go.',
                'What sets this movie apart is that it genuinely respects its audience. Every scene serves a purpose. Nothing is filler.',
                'The pacing is masterful—taut and gripping when it needs to be, then giving you just enough room to breathe before the next wave hits.',
                'The direction shows a level of confidence you rarely see. Every creative choice feels intentional, from the color palette to the blocking of each scene.',
            ],
            "acting": [
                'The performances here are nothing short of extraordinary. The lead doesn\'t just play the character—they become the character, disappearing so completely into the role that you forget you\'re watching an actor at all.',
                'The lead delivers what might be a career-best performance. The emotional range on display—from quiet vulnerability to explosive intensity—is genuinely remarkable.',
                'The supporting cast is equally strong. Every minor character gets a moment to shine, which is rare in films of this scale.',
                'The chemistry between the leads is electric. You believe every glance, every unspoken word between them.',
            ],
            "plot": [
                'The screenplay is incredibly tight. Every setup pays off, every subplot earns its place, and the third act brings everything together in a way that feels both surprising and inevitable.',
                'The story is deceptively simple on the surface, but the layers of meaning underneath reward repeat viewings. This is a film that grows richer each time you watch it.',
                'The narrative structure is clever without being gimmicky. The non-linear elements serve the story rather than showing off, which is a harder trick to pull off than it looks.',
                'The dialogue is sharp and natural. Characters talk like actual human beings, not like screenwriters using them as mouthpieces.',
            ],
            "visuals": [
                'Visually, this film is a feast. Every frame could be a poster. The cinematography doesn\'t just capture the action—it elevates it, using light, shadow, and composition to tell the story in ways words never could.',
                'The visual effects are top-tier, but more importantly, they serve the story rather than overshadowing it. The CGI blends seamlessly with practical effects.',
                'The production design is immersive to an almost ridiculous degree. The world-building is so detailed that you feel like you could step through the screen and walk around in it.',
            ],
            "music": [
                'The score is doing some seriously heavy lifting here, and it does it without ever feeling intrusive. It knows exactly when to swell and when to step back and let silence do the work.',
                'The sound design deserves special mention. Every creak, every whisper, every footstep is placed with surgical precision, creating an atmosphere that pulls you in completely.',
                'I\'ve been listening to the soundtrack on repeat for three days. The composer hasn\'t just written music for the film—they\'ve written music that stands on its own as an artistic statement.',
            ],
        },
        "negatives": {
            "default": [
                'Is it perfect? No. There are a couple of moments where the pacing stumbles slightly, but these are minor quibbles in what is otherwise a stellar experience.',
                'If I had to nitpick, some of the second-act transitions could be smoother. But honestly, that\'s like complaining about the font on a winning lottery ticket.',
            ],
        },
        "closings": [
            'Bottom line: Go see {movie}. Preferably on the biggest screen you can find. It deserves your full attention and it will reward it generously.',
            'Here\'s the thing: {movie} isn\'t perfect, but it reminded me why I love movies in the first place. That feeling is worth more than a flawless runtime.',
            'Alright, hype duty done. If you see it and hate it—fair enough, we just have different taste. But if you see it and love it—come find me. We have a lot to talk about.',
        ],
    },

    "roast": {
        "name": "Roast Mode",
        "label": "Roast Mode",
        "openings": [
            'I walked out of {movie} feeling a very specific kind of disappointment. Not the anger you feel at a genuinely terrible film—but the hollow emptiness of two hours you will never get back.',
            'Let me start with a compliment: {movie} is ambitious. Now let me spend the rest of this review explaining why ambition without execution is just expensive noise.',
            'If someone recommended {movie} to you, I would gently suggest you reevaluate that friendship. Not because the movie is bad—but because a true friend would have warned you.',
        ],
        "positives": {
            "default": [
                'To be fair, the trailer was excellent. Whoever cut that together deserves a raise. The actual film, unfortunately, is a different story.',
                'The first twenty minutes are genuinely intriguing. And then... well. Let\'s just say the film peaks early and spends the rest of its runtime rolling downhill.',
                'The poster design team earned their paycheck. As for everyone else involved in this production—let\'s just say I have questions about the budget allocation.',
            ],
            "acting": [
                'The actors are clearly trying. You can see it on their faces—the desperate hope that somehow, through sheer effort, they can elevate this material. They cannot. But the effort is almost touching.',
                'The lead delivers their lines with the resigned energy of someone who read the script, realized what they\'d signed up for, and decided to collect the paycheck with as little eye contact as possible.',
            ],
            "plot": [
                'The plot unfolds like a drunk person trying to walk in a straight line. You see the intention. You appreciate the effort. But the result is fundamentally, irreparably crooked.',
                'The screenwriter appears to have confused "paying homage" with "doing the exact same thing but worse." It\'s less an homage and more a poorly traced copy.',
            ],
            "visuals": [
                'The CGI isn\'t terrible, but when you consider the budget—this is the visual equivalent of paying Michelin-star prices for instant noodles.',
            ],
        },
        "negatives": {
            "default": [
                'The fundamental problem with {movie} is that it has no idea what it wants to be. It cribs from every genre playbook without ever developing a personality of its own.',
                'The plot holes are so numerous you could use them as a sieve. Characters make decisions that defy not just logic, but basic self-preservation instincts.',
                'The story advances entirely because characters make inexplicably stupid choices. If anyone in this film acted like a real human being, the credits would roll at the thirty-minute mark.',
            ],
            "plot": [
                'The script\'s problem isn\'t that it\'s formulaic—formulas work when executed well. The problem is that it\'s formulaic AND messy, like three writers wrote separate drafts and someone just shuffled the pages together.',
                'The ending twist isn\'t shocking in the way the filmmakers intended. It\'s shocking because of how little setup there was. A good twist makes you rethink everything. This one makes you recheck the runtime.',
            ],
        },
        "closings": [
            'Verdict: If you have two hours to spare, go for a walk. Call your mom. Learn a new hobby. Almost anything would be a better use of your time than {movie}.',
            '{movie} reminded me of that Oscar Wilde line—"Everything is about sex, except sex. Sex is about power." This movie is about nothing, except looking like it\'s about something.',
            'If you absolutely must watch it, bring a friend who enjoys sarcasm. The shared eye-rolling will at least provide some entertainment value.',
        ],
    },

    "analysis": {
        "name": "Deep Analysis",
        "label": "Deep Analysis",
        "openings": [
            '{movie} can\'t be reduced to "good" or "bad." It\'s a work that demands serious engagement—a film that takes real creative risks, even if not all of them pay off.',
            '{movie} is one of those rare films that has been simultaneously overrated and underrated by different camps. Here\'s my attempt to cut through the noise and engage with what\'s actually on screen.',
            'Let me state my position upfront: {movie} is a flawed masterpiece. This review isn\'t about delivering a verdict—it\'s about understanding what the film is trying to do, how it goes about doing it, and where it succeeds and fails along the way.',
        ],
        "positives": {
            "default": [
                'From a structural perspective, the director has chosen an unconventional narrative approach. This isn\'t showing off—it\'s a formal choice that mirrors the film\'s thematic concerns about fragmentation and memory.',
                'On a craft level, the film demonstrates exceptional technical control. The long takes, the sound-image relationships, the rhythmic editing—all of it rewards close analysis.',
            ],
            "acting": [
                'The performances deserve extended discussion. The lead abandons showy theatrics in favor of something rarer: a quiet, lived-in naturalism. In a market that rewards "loud" acting, this is a risky and admirable choice.',
                'The ensemble construction is also noteworthy. Every supporting character feels like a complete person with a life outside the frame, rather than a narrative convenience for the protagonist.',
            ],
            "plot": [
                'Analyzing the screenplay structurally reveals a fascinating formal experiment. The film adopts a classic three-act framework but introduces a deliberate deformation in the middle of Act Two—extending the protagonist\'s period of maximum crisis beyond conventional tolerance.',
            ],
        },
        "negatives": {
            "default": [
                'However, the third act reveals significant structural problems. The careful groundwork laid in the first two-thirds is abandoned in favor of a rushed, almost panicked narrative pace—as if the director suddenly learned the budget had been cut.',
                'The genre hybridization, while ambitious, is not uniformly successful. Certain references lack the necessary transformation to feel organic rather than derivative.',
            ],
        },
        "closings": [
            'In the end, {movie} is not a perfect film. But it is an honest one, and an ambitious one—and in an increasingly risk-averse industry, that alone merits serious consideration.',
            'The conversation around {movie} is only beginning. I look forward to more perspectives, more readings, and more arguments. A film that provokes this much discussion is, almost by definition, doing something right.',
        ],
    },

    "literary": {
        "name": "Literary Style",
        "label": "Literary Style",
        "openings": [
            'Some films are meant to be watched. Others are meant to be felt. {movie} belongs to the latter category. It doesn\'t demand your analysis—it asks for your presence, and in return, it gives you something quiet and lasting.',
            'Watching {movie}, I kept thinking about something Roger Ebert once said: "Movies are like a machine that generates empathy." This film is that machine, operating at full capacity.',
            'The theater was quiet when the credits rolled—not the silence of indifference, but the silence of people who had just been moved and didn\'t quite know what to do with the feeling yet.',
        ],
        "positives": {
            "default": [
                'The most beautiful moments in {movie} are the ones most commercial films would have cut—an unnecessary glance, a wordless walk, a tree swaying in the wind. These "redundant" moments are precisely what elevate it beyond its genre.',
                'Great cinema tells its story through images, not dialogue. Some of {movie}\'s most powerful passages work with the sound off—not because the score isn\'t good (it is), but because the visual storytelling is that complete.',
            ],
        },
        "negatives": {
            "default": [
                'This is not an easy film to enter. Its rhythms are slow, its emotional register is subdued, and it requires you to meet it on its own terms. Not everyone will have the patience—and that\'s okay.',
            ],
        },
        "closings": [
            'So I\'m not sure whether to recommend {movie}. If you\'re exhausted and just want something light to unwind with—this isn\'t it. But if the world has been feeling too loud lately, and you need something (someone) to sit quietly with you for two hours—go.',
            'I wrote most of this review in my head during the walk home from the theater. The streetlights were on. The shadows were long. And I realized {movie} had already become part of how I see things.',
        ],
    },
}


# ── movie knowledge base ─────────────────────────────────────────────

MOVIE_KB = {
    "inception": {"title_en": "Inception", "director": "Christopher Nolan", "year": "2010", "genre": "Sci-Fi / Action"},
    "interstellar": {"title_en": "Interstellar", "director": "Christopher Nolan", "year": "2014", "genre": "Sci-Fi / Drama"},
    "parasite": {"title_en": "Parasite", "director": "Bong Joon-ho", "year": "2019", "genre": "Thriller / Drama"},
    "dune": {"title_en": "Dune", "director": "Denis Villeneuve", "year": "2021", "genre": "Sci-Fi / Epic"},
    "oppenheimer": {"title_en": "Oppenheimer", "director": "Christopher Nolan", "year": "2023", "genre": "Biography / Drama"},
    "barbie": {"title_en": "Barbie", "director": "Greta Gerwig", "year": "2023", "genre": "Comedy / Fantasy"},
    "everything_everywhere": {"title_en": "Everything Everywhere All at Once", "director": "Daniels", "year": "2022", "genre": "Sci-Fi / Comedy-Drama"},
    "the_dark_knight": {"title_en": "The Dark Knight", "director": "Christopher Nolan", "year": "2008", "genre": "Action / Crime"},
    "spirited_away": {"title_en": "Spirited Away", "director": "Hayao Miyazaki", "year": "2001", "genre": "Animation / Fantasy"},
    "the_godfather": {"title_en": "The Godfather", "director": "Francis Ford Coppola", "year": "1972", "genre": "Crime / Drama"},
    "pulp_fiction": {"title_en": "Pulp Fiction", "director": "Quentin Tarantino", "year": "1994", "genre": "Crime / Drama"},
    "blade_runner_2049": {"title_en": "Blade Runner 2049", "director": "Denis Villeneuve", "year": "2017", "genre": "Sci-Fi / Noir"},
}


# ── utility ───────────────────────────────────────────────────────────

def _pick(aspect_key, bank):
    positives = bank["positives"]
    negatives = bank["negatives"]
    pool = positives.get(aspect_key, positives.get("default", [""]))
    pool_neg = negatives.get(aspect_key, negatives.get("default", [""]))
    return random.choice(pool), random.choice(pool_neg)


def _extract_aspects(feeling):
    aspect_map = {
        "effects": "visuals", "visuals": "visuals", "CGI": "visuals", "visual": "visuals",
        "cinematography": "visuals", "photography": "visuals", "design": "visuals",
        "plot": "plot", "story": "plot", "script": "plot", "writing": "plot",
        "screenplay": "plot", "narrative": "plot", "pacing": "plot", "ending": "plot",
        "twist": "plot", "logic": "plot", "structure": "plot", "dialogue": "plot",
        "acting": "acting", "performance": "acting", "actor": "acting",
        "cast": "acting", "lead": "acting", "character": "acting",
        "score": "music", "soundtrack": "music", "music": "music", "sound": "music",
        "audio": "music", "BGM": "music",
    }
    found = set()
    lower = feeling.lower()
    for word, aspect in aspect_map.items():
        if word.lower() in lower:
            found.add(aspect)
    if not found:
        found.add("default")
    return list(found)


def _estimate_sentiment_from_text(feeling):
    positive_words = [
        "amazing", "brilliant", "excellent", "great", "good", "fantastic", "love",
        "beautiful", "stunning", "masterpiece", "incredible", "wonderful", "best",
        "impressive", "outstanding", "recommend", "must-watch", "must-see",
        "phenomenal", "perfect", "superb", "breathtaking", "awesome", "enjoyed",
    ]
    negative_words = [
        "bad", "terrible", "boring", "disappointing", "awful", "worst", "weak",
        "poor", "waste", "overrated", "mess", "lazy", "dull", "forgettable",
        "cringe", "embarrassing", "mediocre", "fails", "unwatchable", "sucks",
    ]
    lower = feeling.lower()
    pos_count = sum(1 for w in positive_words if w in lower)
    neg_count = sum(1 for w in negative_words if w in lower)
    if pos_count > neg_count:
        return 75
    elif neg_count > pos_count:
        return 30
    return 55


# ── template engine ───────────────────────────────────────────────────

def template_generate(movie_name, short_feeling, style="hype", sentiment_ratio=None):
    bank = STYLE_BANKS.get(style, STYLE_BANKS["hype"])
    aspects = _extract_aspects(short_feeling)
    if sentiment_ratio is None:
        sentiment_ratio = _estimate_sentiment_from_text(short_feeling)
    selected_aspects = random.sample(aspects, min(3, len(aspects)))
    parts = []

    opening = random.choice(bank["openings"]).format(movie=movie_name)
    parts.append(opening)

    for aspect in selected_aspects:
        pos, neg = _pick(aspect, bank)
        roll = random.randint(0, 100)
        if roll <= sentiment_ratio:
            parts.append(pos)
            if roll > 80 and neg:
                parts.append(neg)
        else:
            parts.append(neg)
            if roll > 30 and pos:
                parts.append(pos)

    if sentiment_ratio >= 70:
        parts.append(random.choice(bank["positives"]["default"]))
    elif sentiment_ratio <= 35:
        parts.append(random.choice(bank["negatives"]["default"]))
    else:
        pos, neg = _pick("default", bank)
        parts.append(pos)
        parts.append(neg)

    closing = random.choice(bank["closings"]).format(movie=movie_name)
    parts.append(closing)

    review = "\n\n".join(parts)
    review = re.sub(r"([.!\?;])([A-Za-z])", r"\1 \2", review)
    return {
        "review": review,
        "meta": {
            "style": bank["label"],
            "sentiment_ratio": sentiment_ratio,
            "aspects_covered": selected_aspects,
            "engine": "template",
        },
    }


# ── LLM API ──────────────────────────────────────────────────────────

def llm_generate(movie_name, short_feeling, style="hype", sentiment_ratio=70,
                 api_key=None, model="deepseek-chat", max_words=200):
    """Generate a review via DeepSeek API. Falls back to template if no key or on error."""
    if not api_key:
        result = template_generate(movie_name, short_feeling, style, sentiment_ratio)
        result["meta"]["engine"] = "template (no API key)"
        return result

    from openai import OpenAI

    bank = STYLE_BANKS.get(style, STYLE_BANKS["hype"])
    aspects = _extract_aspects(short_feeling)
    if sentiment_ratio is None:
        sentiment_ratio = _estimate_sentiment_from_text(short_feeling)

    tone_guide = {
        "hype": "Enthusiastic, energetic, and overwhelmingly positive. Like a friend who just saw the best movie of their life.",
        "roast": "Sarcastic, witty, and mercilessly critical. Like a comedian tearing apart a bad movie.",
        "analysis": "Thoughtful, balanced, and intellectually rigorous. Like a film scholar writing for a serious publication.",
        "literary": "Poetic, introspective, and emotionally resonant. Like a novelist reflecting on how a film moved them.",
    }

    system_prompt = (
        f"You are a professional film critic writing in '{bank['label']}' style. "
        f"{tone_guide.get(style, tone_guide['hype'])}\n\n"
        f"Rules:\n"
        f"- Write a complete movie review of 2-3 short paragraphs.\n"
        f"- The review MUST be no more than {max_words} words total. Be concise.\n"
        f"- The overall sentiment should be roughly {sentiment_ratio}% positive.\n"
        f"- Cover these aspects where relevant: {', '.join(aspects[:4])}.\n"
        f"- Do NOT use placeholder text or brackets.\n"
        f"- End with a clear verdict or recommendation."
    )

    user_prompt = (
        f'Movie: "{movie_name}"\n\n'
        f"My quick take: {short_feeling}\n\n"
        f"Write a {bank['label']} review."
    )

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
        resp = client.chat.completions.create(
            model=model,
            max_tokens=400,
            temperature=0.85,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        review_text = resp.choices[0].message.content
        return {
            "review": review_text,
            "meta": {
                "style": bank["label"],
                "sentiment_ratio": sentiment_ratio,
                "aspects_covered": aspects,
                "engine": f"deepseek ({model})",
            },
        }
    except Exception as e:
        # Fallback to template on any API error
        result = template_generate(movie_name, short_feeling, style, sentiment_ratio)
        result["meta"]["engine"] = f"template (API fallback: {e})"
        return result
