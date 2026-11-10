import pytest

from gpo_local.question_grounding import no_parent_source_requests


def test_description_of_missing_evidence_is_allowed():
    no_parent_source_requests(
        "In de aangeleverde bronnen staat echter geen expliciete passage over groepsgrootte. "
        "Wel bevatten de bronnen een beoordeling van de toelatingscriteria."
    )


@pytest.mark.parametrize(
    "request_text",
    [
        "Lever de passage uit het advies aan.",
        "Welke bronpassage bedoel je?",
        "Kunt u het zaaknummer aanleveren?",
        "Wil je de bron-ID opsturen?",
    ],
)
def test_requests_for_parent_supplied_evidence_are_still_blocked(request_text):
    with pytest.raises(ValueError):
        no_parent_source_requests(request_text)
