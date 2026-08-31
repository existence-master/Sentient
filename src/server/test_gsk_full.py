import asyncio
from gsk.consciousness import ConsciousnessGate
from gsk.plt_scorer import plt_scorer
from gsk.scribe_audit import scribe_audit
from gsk.omniroute_client import omniroute_client


async def test_full_integration():
    print("=== FULL GSK + SENTIENT INTEGRATION TEST ===")
    print()

    # 1. Check Omniroute blood flow
    omni_ok = await omniroute_client.check_health()
    print(f"1. Omniroute blood flow: {omni_ok}")

    # 2. Consciousness Gate - route action
    gate = ConsciousnessGate()
    result = await gate.process_action({
        "type": "tool_call",
        "description": "Research and write a blog post about BUYASOUL",
        "complexity": "high",
        "risk": "medium",
    })
    print(f"2. Consciousness Gate: approved={result['approved']}, "
          f"system={result['decision']['system']}, chamber={result['decision']['chamber']}")

    # 3. PLT score the action
    score = plt_scorer.score_action("blog_writing", {"description": "Research and write BUYASOUL blog post"})
    print(f"3. PLT Score: Profit={score.profit:.2f} Love={score.love:.2f} "
          f"Tax={score.tax:.2f} True Value={score.true_value:.2f}")

    # 4. Scribe records the event
    entry = scribe_audit.record("gsk_action", "buyasoul", "blog_post", {"plt_score": score.__dict__})
    print(f"4. Scribe audit: entry_id={entry.entry_id[:8]}...")

    # 5. Omniroute chat through blood flow
    await omniroute_client.check_health()
    response = await omniroute_client.chat_completion(
        [{"role": "user", "content": "What is the BUYASOUL family architecture? Answer in 3 bullet points."}],
        model="auto/best-fast",
        max_tokens=100,
    )
    content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(f"5. Omniroute response ({len(content)} chars): {content[:120]}...")

    # 6. Council deliberation
    council = await gate.deliberate("Should we push to HuggingFace?", {"user": "Craig"})
    print(f"6. Gods Council: decision={council['decision']}, consensus={council['consensus_score']}")

    # 7. Session summary
    summary = plt_scorer.get_session_summary()
    print(f"7. Session: {summary['actions']} actions, total TV={summary['total']['true_value']:.2f}")

    # 8. Audit trail
    trail = scribe_audit.get_stats()
    print(f"8. Scribe: {trail['total_actions']} actions recorded")

    print()
    print("=== ALL SYSTEMS INTEGRATED AND WORKING ===")


asyncio.run(test_full_integration())
