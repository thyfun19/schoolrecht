import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from conftest import BASE, ROOT
from gpo_app import pinecone_search
from gpo_app.api import create_app
from gpo_app.chunking import LocalTokenCounter, prepare_chunks
from gpo_app.ingestion import import_prepared
from gpo_app.card_search import excerpt, rank_cards, search_cards
from gpo_app.citation_search import CitationIndex


def test_parent_question_finds_responsibility_when_lessons_move():
    index = CitationIndex(
        [
            (
                "general",
                "De school geeft uitleg over de activiteiten van de andere voorziening.",
            ),
            (
                "responsibility",
                "De school van herkomst blijft verantwoordelijk voor het onderwijsaanbod.",
            ),
        ]
    )
    ranked = index.ranking(
        "Wie had de taak toen mijn kind elders les kreeg en ingeschreven bleef?",
        {"general", "responsibility"},
        2,
    )
    assert ranked[0][0] == "responsibility"
    assert not index.ranking(
        "Wat staat er over brandveiligheid?", {"general", "responsibility"}, 2
    )


def test_plan_duty_requires_nearby_condition_in_same_section():
    question = (
        "Wie moest het plan aanpassen toen ze elders les kreeg maar ingeschreven bleef?"
    )
    cases = [dict(processing_id=c) for c in ("split", "together")]
    duty = "Het bevoegd gezag was verantwoordelijk voor het bijstellen van het ontwikkelingsperspectief."
    condition = "Bij een proefplaatsing bleef de school waar zij was ingeschreven verantwoordelijk."
    rows = [
        dict(
            processing_id=c["processing_id"],
            source_id=c["processing_id"] + str(i),
            section_key=i if c["processing_id"] == "split" else 0,
            page_number=3,
            quote=q,
        )
        for c in cases
        for i, q in enumerate([duty, condition])
    ]
    first, cite = search_cards(question, cases, rows, None, 1)[0]
    assert first["processing_id"] == "together"
    assert cite["source_id"] == "together0" and cite["quote"] == duty


def test_card_keeps_condition_at_end_of_short_paragraph():
    quote = (
        "Bij de bespreking van het plan met de ouders " * 6
        + "gold de volgende voorwaarde."
    )
    assert excerpt("Wat werd besproken?", quote) == quote


def test_passage_refinement_keeps_case_order_and_source_identity():
    tokenizer = SimpleNamespace(
        no_padding=lambda: None, encode=lambda *args: SimpleNamespace(ids=[1])
    )
    scores = {"alpha": 3, "beta": 4, "gamma": 2, "delta": 5, "last": 1}
    model = SimpleNamespace(
        tokenizer=tokenizer, score=lambda pairs: [scores[pairs[0]["passage"]]]
    )

    def citation(key):
        return {"source_id": key, "quote": key}

    candidates = [
        ({"processing_id": str(i)}, [citation(key) for key in keys])
        for i, keys in enumerate([["gamma", "delta"], ["alpha", "beta"], ["last"]])
    ]
    refined = rank_cards("vraag", candidates, model, 2)
    assert [(c["processing_id"], p["source_id"]) for c, p in refined] == [
        ("1", "beta"),
        ("0", "delta"),
    ]
    assert rank_cards("vraag", candidates, None, 1) == [
        (candidates[0][0], candidates[0][1][0])
    ]


def test_deadline_explained_before_any_search(monkeypatch):
    def unexpected_search():
        raise AssertionError("Scope check must precede cloud and database access.")

    monkeypatch.setattr(pinecone_search, "pinecone_search", unexpected_search)
    with TestClient(create_app(object())) as client:
        response = client.post(
            "/api/v1/search/pinecone",
            json={
                "question": "Tot wanneer kan ik bezwaar maken en waar moet ik het indienen?"
            },
        )
    assert response.status_code == 422
    assert "Actuele procedurebronnen" in response.json()["detail"]


def test_cloud_fallback_ranks_real_questions_and_preserves_citations(
    engine, prepared, monkeypatch, tmp_path
):
    monkeypatch.delenv("GPO_SEARCH_MODE", raising=False)
    import_prepared(engine, prepared)
    prepare_chunks(engine, LocalTokenCounter(BASE / "data/e5/model/tokenizer.json"))
    monkeypatch.setattr(pinecone_search, "BASE", tmp_path)
    monkeypatch.setattr(pinecone_search, "pinecone_configured", lambda: False)
    search = pinecone_search.PineconeSearch()
    monkeypatch.setattr(pinecone_search, "pinecone_search", lambda: search)
    questions = json.loads(
        (ROOT / "onderzoek_gpo/besliscontrole/referentie.json").read_text("utf-8")
    )["questions"]
    rankings = {}
    with TestClient(create_app(engine)) as client:
        for question in questions[:4]:
            response = client.post(
                "/api/v1/search/pinecone",
                json={"question": question["question"], "limit": 9},
            )
            assert response.status_code == 200
            result = response.json()
            assert result["method"] == "postgres_text"
            rankings[question["id"]] = result["cases"][:3]
            case = result["results"][0]
            preview = case["preview"]
            source = client.get(
                "/api/v1/sources/" + preview["source_id"],
                params={"processing_id": case["processing_id"]},
            ).json()
            assert source["quote"] == preview["quote"]
            assert source["page_number"] == preview["page_number"]
            assert preview["excerpt"].strip("…") in source["quote"]
            chunk = case["matched_chunks"][0]
            citation = client.get(
                "/api/v1/sources/" + chunk["source_ids"][0],
                params={"processing_id": case["processing_id"]},
            ).json()
            assert (citation["case_number"], citation["page_number"]) == (
                case["case_number"],
                chunk["page"],
            )
    assert rankings["V3"] != rankings["V4"]


def test_fusion_counts_each_case_once_per_method_and_deduplicates_chunks():
    def hit(case, chunk):
        return dict(id=chunk, metadata=dict(case=case, processing_id=case))

    text = [hit("a", "1"), hit("a", "2"), hit("b", "3")]
    vectors = [hit("b", "3"), hit("c", "4")]
    ranked = pinecone_search.fuse_case_matches(text, vectors)
    assert [m["id"] for m in ranked] == ["3", "1", "2", "4"]
    assert ranked[1]["case_score"] == ranked[2]["case_score"] == 1 / 61


def test_citation_discovery_finds_case_outside_vector_pool(
    engine, prepared, monkeypatch, tmp_path
):
    monkeypatch.delenv("GPO_SEARCH_MODE", raising=False)
    monkeypatch.delenv("GPO_HYBRID_SEARCH", raising=False)
    import_prepared(engine, prepared)
    monkeypatch.setattr(pinecone_search, "BASE", tmp_path)
    monkeypatch.setattr(pinecone_search, "pinecone_configured", lambda: True)
    search = pinecone_search.PineconeSearch()
    monkeypatch.setattr(search, "query_vectors", lambda *args, **kwargs: [])
    monkeypatch.setattr(pinecone_search, "pinecone_search", lambda: search)
    with TestClient(create_app(engine)) as client:
        body = {
            "question": "Welke instemming is nodig bij het handelingsdeel van het OPP?",
            "limit": 3,
        }
        results = client.post("/api/v1/search/pinecone", json=body).json()["results"]
        assert results and len({r["case_number"] for r in results}) == len(results)
        for entry in results:
            assert entry["retrieval_source"] == "registered_citation"
            preview = entry["preview"]
            canonical = client.get(
                "/api/v1/sources/" + preview["source_id"],
                params={"processing_id": entry["processing_id"]},
            ).json()
            assert (canonical["quote"], canonical["case_number"]) == (
                preview["quote"],
                entry["case_number"],
            )
        assert (
            client.post(
                "/api/v1/search/pinecone", json={**body, "sector": "onbekend"}
            ).json()["results"]
            == []
        )


def test_parent_intents_prefer_commission_findings_over_party_contentions_and_false_cognates():
    question = "De directeur zegt dat de klas al vol zit. Hoe woog de commissie de beschikbare plaatsen bij aanmelding?"
    cases = [dict(processing_id="115205")]
    contention = "Verweerder heeft aangevoerd dat de reden voor de weigering om [leerling] tot de school toe te laten, is gelegen in het feit dat de school ten tijde van de aanmelding van [leerling] vol was."
    finding = "Het moet voor ouders duidelijk zijn hoeveel plaatsruimte er is en hoeveel aanmeldingen er zijn."
    rows = [
        dict(processing_id="115205", source_id="s1", page_number=4, quote=contention),
        dict(processing_id="115205", source_id="s2", page_number=4, quote=finding),
    ]
    _, picked = search_cards(question, cases, rows, None, 1)[0]
    assert picked["quote"] == finding

    verzuim_q = "Welke hulp zette school volgens de commissie in toen het door verzuim niet lukte om weer lessen te volgen?"
    verzuimd_row = dict(
        processing_id="a",
        source_id="a1",
        page_number=5,
        quote="Daarnaast heeft de school verzuimd voor [leerling] onderzoek in te stellen naar zijn ondersteuningsbehoeften.",
    )
    verzuim_row = dict(
        processing_id="b",
        source_id="b1",
        page_number=5,
        quote="Nadat bleek dat de ingezette begeleiding niet leidde tot verbetering en sprake was van veelvuldig verzuim, heeft verweerder zowel de jeugdarts als de leerplichtambtenaar betrokken.",
    )
    tokenizer = SimpleNamespace(
        no_padding=lambda: None, encode=lambda *args: SimpleNamespace(ids=[1])
    )
    model = SimpleNamespace(
        tokenizer=tokenizer,
        score=lambda pairs: [10.0 if "verzuimd" in pairs[0]["passage"] else 5.0],
    )
    cards = search_cards(
        verzuim_q,
        [dict(processing_id="a"), dict(processing_id="b")],
        [verzuimd_row, verzuim_row],
        model,
        1,
    )
    assert cards[0][0]["processing_id"] == "b"
    assert cards[0][1]["quote"] == verzuim_row["quote"]
