"""Hindi facts are never merged into a fact about someone else (#339).

Seen with qwen3:8b and nomic-embed-text: "प्रिया पुणे में रहती है" was filed as an update to "Riya lives in Pune", and a
Hindi Karan fact was skipped as a duplicate of a Kabir fact. The embedding model scores such texts close together and
the small model then picks UPDATE or SKIP. Different subjects are now caught before the model decides."""

from sentient.memory.facts import FactMemory, covered_by, same_subject, same_thing, subject_aliases
from sentient.memory.scripts import devanagari_subjects, name_key, names_match, script_words
from sentient.memory.vectors import VecTable
from sentient.store.db import Store
from tests.conftest import FakeProvider

ANALYSIS = {"topics": ["Relationships & Social Life"], "memory_type": "long-term", "duration": None}


class SameVectorProvider(FakeProvider):
    """An embedding model that can't tell texts apart: every fact scores 1.0 against every other."""

    async def embed(self, texts, *, model=None):
        return [[1.0] + [0.0] * (self.dim - 1) for _ in texts]


def test_names_match_across_scripts():
    assert names_match("माया", "maya") and names_match("प्रिया", "priya") and names_match("कबीर", "kabir")
    assert names_match("करण", "karan") and names_match("श्रेया", "shreya") and names_match("पूजा", "pooja")
    assert not names_match("प्रिया", "riya")
    assert not names_match("करण", "kabir")
    assert not names_match("करण", "kiran")
    assert not names_match("riya", "riyaz")
    assert name_key("Maya") == name_key("माया")


def test_devanagari_subjects():
    assert devanagari_subjects("प्रिया पुणे में रहती है।") == {"प्रिया"}
    assert devanagari_subjects("पिछले साल कबीर दिल्ली चला गया।") == {"कबीर"}  # past "last year"
    assert devanagari_subjects("मैं पुणे में रहती हूँ", "Maya") == {"maya"}
    assert devanagari_subjects("मेरी बहन पुणे में रहती है", "Maya") == {"maya's sister"}
    assert devanagari_subjects("माया की बहन रिया पुणे में रहती है") == {"माया's sister"}
    assert devanagari_subjects("वह पुणे में रहती है") == set()  # "she": unclear
    # a fact that starts in Latin script keeps the English reading
    assert subject_aliases("Priya पुणे में रहती है", "Maya") == {"priya"}


def test_same_subject_across_scripts():
    assert not same_subject("प्रिया पुणे में रहती है।", "Riya lives in Pune.", "Maya")
    assert not same_subject("करण गोवा गया।", "Kabir moved to Delhi.", "Maya")
    assert not same_subject("करण गोवा गया।", "पिछले साल कबीर दिल्ली चला गया।", "Maya")
    assert same_subject("माया अब दिल्ली में रहती है।", "Maya lives in Pune.", "Maya")
    assert same_subject("कबीर दिल्ली चला गया।", "Kabir moved to Delhi.", "Maya")
    assert same_subject("मैं शाकाहारी हूँ", "Maya is vegetarian.", "Maya")  # "I" is the user
    assert not same_subject("माया की बहन पुणे में रहती है", "Maya lives in Pune.", "Maya")
    assert not same_thing("प्रिया अब दिल्ली में रहती है।", "Riya lives in Pune.", "Maya")


def test_hindi_words_are_not_covered_by_an_english_fact():
    assert script_words("करण गोवा गया।") == {"करण", "गोवा", "गया"}
    # nothing in the English word checks reads Hindi, so this used to count as "already said"
    assert not covered_by("करण गोवा गया।", "Kabir went to Goa.", "Maya")
    assert not covered_by("प्रिया पुणे में रहती है।", "Riya lives in Pune.", "Maya")
    assert covered_by("प्रिया पुणे में रहती है", "प्रिया पुणे में रहती है।", "Maya")


async def _memory(config, isolated_home, name, llm=None):
    store = await Store(isolated_home / name).open()
    llm = llm or SameVectorProvider()
    return store, llm, FactMemory(store, llm, config)


async def test_hindi_fact_about_someone_else_is_added_not_skipped(config, isolated_home):
    # every fact embeds to the same vector, so on main the near-duplicate shortcut skipped the Hindi fact
    store, _llm, mem = await _memory(config, isolated_home, "dup.db")
    try:
        riya = await mem.remember("Riya lives in Pune.", use_llm=False)
        out = await mem.remember("प्रिया पुणे में रहती है।", use_llm=False)
        assert out["action"] == "ADD" and out["id"] != riya["id"]
        # the same person again is still a duplicate
        again = await mem.remember("Riya lives in Pune now.", use_llm=False)
        assert again["action"] == "SKIP"
    finally:
        await store.close()


async def test_model_update_or_skip_across_subjects_is_kept_as_new(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "cud.db")
    config.memory.duplicate_similarity = 0.999
    try:
        riya = await mem.remember("Riya lives in Pune.", use_llm=False)
        kabir = await mem.remember("Kabir moved to Delhi last year.", use_llm=False)
        llm.json_replies.extend([
            {"action": "UPDATE", "fact_id": riya["id"], "content": "Riya lives in Pune.", "analysis": None},
            ANALYSIS,
            {"action": "SKIP", "fact_id": kabir["id"], "content": "", "analysis": None},
            ANALYSIS,
            {"action": "DELETE", "fact_id": riya["id"], "content": "", "analysis": None},
            ANALYSIS,
        ])
        priya = await mem.remember("प्रिया अब दिल्ली में रहती है।")
        karan = await mem.remember("करण गोवा गया।")
        third = await mem.remember("पिछले साल प्रिया मुंबई गई।")
        assert priya["action"] == karan["action"] == third["action"] == "ADD"
        stored = {f["content"] for f in await mem.list_facts()}
        assert {"Riya lives in Pune.", "Kabir moved to Delhi last year.", "प्रिया अब दिल्ली में रहती है।",
                "करण गोवा गया।", "पिछले साल प्रिया मुंबई गई।"} <= stored
        # the facts about other people were never offered to the model
        prompt = llm.calls[0]["messages"][-1]["content"]
        assert "Riya lives in Pune" not in prompt and "Kabir" not in prompt
    finally:
        await store.close()


async def test_same_person_across_scripts_can_still_be_updated(config, isolated_home):
    store, llm, mem = await _memory(config, isolated_home, "same.db")
    config.memory.duplicate_similarity = 0.999
    try:
        maya = await mem.remember("Maya lives in Pune.", use_llm=False)
        llm.json_replies.extend([
            {"action": "UPDATE", "fact_id": maya["id"], "content": "Maya lives in Delhi.", "analysis": None},
            ANALYSIS,
        ])
        out = await mem.remember("माया अब दिल्ली में रहती है।")
        assert out["action"] == "UPDATE" and out["id"] == maya["id"]
        assert "Maya lives in Pune" in llm.calls[0]["messages"][-1]["content"]
    finally:
        await store.close()


class SwitchableProvider(FakeProvider):
    """Two embedding models with the same dimension that place texts differently."""

    def __init__(self):
        super().__init__()
        self.embedding = "ollama/nomic-embed-text"

    def model_for(self, role):
        return self.embedding if role == "embedding" else super().model_for(role)

    async def embed(self, texts, *, model=None):
        vecs = await super().embed(texts)
        return vecs if self.embedding.endswith("nomic-embed-text") else [list(reversed(v)) for v in vecs]


async def test_switching_embedding_model_of_the_same_size_re_embeds_memory(config, isolated_home):
    llm = SwitchableProvider()
    store, llm, mem = await _memory(config, isolated_home, "switch.db", llm)
    try:
        first = await mem.remember("Maya likes long walks by the sea.", use_llm=False)
        assert await store.get_meta("facts_vec_model") == "ollama/nomic-embed-text"
        llm.embedding = "ollama/paraphrase-multilingual"  # also 768 dimensions in real life
        mem2 = FactMemory(store, llm, config)  # a restart with the new setting
        found = await mem2.recall("Maya likes long walks by the sea.", min_similarity=0.0, track=False)
        assert found and found[0]["id"] == first["id"] and found[0]["similarity"] > 0.99
        assert await store.get_meta("facts_vec_model") == "ollama/paraphrase-multilingual"
        stored = (await mem2.vec.all_vectors())[first["id"]]
        [expected] = await llm.embed(["Maya likes long walks by the sea."])
        assert max(abs(a - b) for a, b in zip(stored, expected, strict=True)) < 1e-6
    finally:
        await store.close()


async def test_vec_table_keeps_an_index_when_the_model_was_never_recorded(isolated_home):
    store = await Store(isolated_home / "old.db").open()
    try:
        rebuilt: list[bool] = []

        async def rebuild():
            rebuilt.append(True)

        old = VecTable(store, "t_vec")
        await old.ensure(4)  # an older build: no model recorded
        await old.upsert(1, [1.0, 0.0, 0.0, 0.0])
        new = VecTable(store, "t_vec")
        assert await new.ensure(4, rebuild, "ollama/bge-m3") is False and not rebuilt
        assert await store.get_meta("t_vec_model") == "ollama/bge-m3"
        other = VecTable(store, "t_vec")
        assert await other.ensure(4, rebuild, "ollama/nomic-embed-text") is True and rebuilt
    finally:
        await store.close()


async def test_a_failed_re_embed_is_tried_again_after_a_restart(isolated_home):
    store = await Store(isolated_home / "retry.db").open()
    try:
        first = VecTable(store, "r_vec")
        await first.ensure(4, None, "ollama/nomic-embed-text")

        async def broken():
            raise RuntimeError("Ollama went away")

        switched = VecTable(store, "r_vec")
        assert await switched.ensure(4, broken, "ollama/bge-m3") is True
        # the old model stays recorded, so the index isn't trusted on the next start
        assert await store.get_meta("r_vec_model") == "ollama/nomic-embed-text"
        rebuilt: list[bool] = []

        async def works():
            rebuilt.append(True)

        again = VecTable(store, "r_vec")
        assert await again.ensure(4, works, "ollama/bge-m3") is True and rebuilt
        assert await store.get_meta("r_vec_model") == "ollama/bge-m3"
    finally:
        await store.close()


async def test_user_model_ignores_insight_vectors_from_another_embedding_model(app, monkeypatch):
    um = app.user_model
    await um.add_insight("Maya is vegetarian", "preferences")
    assert await um._insight_vectors()  # read once and cached for this model
    monkeypatch.setattr(app.fake, "model_for", lambda role: "ollama/bge-m3" if role == "embedding" else "fake/x")
    assert await um._insight_vectors() == {}  # never compared with a query from the new model
    for task in list(um._background):
        await task
    assert await app.store.get_meta("user_insights_vec_model") == "ollama/bge-m3"
    assert await um._insight_vectors()  # re-embedded with the new model
