"""Conservative local intake. Recognition rules do not establish answerability."""

import re

from .explanation import scope_boundary


def question_intake(question):
    question = question.strip()
    if not 3 <= len(question) <= 2000:
        raise ValueError("Vul een vraag van drie tot tweeduizend tekens in.")
    boundary = scope_boundary(question)
    base = dict(
        question=question,
        request=None,
        sources=[],
        paid_calls=0,
        answer_generated=False,
        estimated_max_cost_usd=0,
    )
    if boundary:
        return dict(**base, status="needs_additional_sources", reason=boundary)
    topics = r"toelat|aanmeld|weiger|afwij|inschrij|verwijder|schors|\bopp\b|ontwikkelingsperspectief|\btlv\b|toelaatbaar|ondersteun|begeleiding|handelingsdeel|schoolrapport|havo|mavo|multidisciplin|zorgleerling|thuiszit"
    school_support_context = re.search(r"\bschool\b", question.lower()) and re.search(
        r"samenwerkingsverband|behandelplek|onderwijs.{0,20}zorg|zorg.{0,20}onderwijs",
        question.lower(),
    )
    if not re.search(topics, question.lower()) and not school_support_context:
        return dict(
            **base,
            status="needs_clarification",
            clarification_question="Gaat je vraag over aanmelden bij een school, stoppen op de huidige school, of het plan voor extra ondersteuning?",
        )
    return None
