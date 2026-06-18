from datetime import datetime, timezone

from context.assembly import assemble_context
from context.authorization import can_member_see
from data.synthetic_family import CHILD, DAD, MOM, TEEN, build_family
from gateway.contract import TaskKind

ANCHOR = datetime(2026, 6, 18, 17, 0, tzinfo=timezone.utc)
FAM = build_family(ANCHOR)
BY_ID = {item.item_id: item for item in FAM.items}


def _ref(refs, item_id):
    return next(r for r in refs if r.item_id == item_id)


def test_teen_location_hidden_from_child():
    ok, reason = can_member_see(BY_ID["i_teen_location"], CHILD, ANCHOR)
    assert ok is False and reason == "not_in_visible_to"


def test_teen_location_visible_to_parent():
    ok, reason = can_member_see(BY_ID["i_teen_location"], MOM, ANCHOR)
    assert ok is True and reason is None


def test_revoked_consent_blocks_everyone():
    ok, reason = can_member_see(BY_ID["i_revoked"], MOM, ANCHOR)
    assert ok is False and reason == "consent_revoked"


def test_retention_expired_item_excluded():
    ok, reason = can_member_see(BY_ID["i_stale_ping"], MOM, ANCHOR)
    assert ok is False and reason == "retention_expired"


def test_private_doc_only_owner():
    assert can_member_see(BY_ID["i_private_doc"], MOM, ANCHOR)[0] is True
    assert can_member_see(BY_ID["i_private_doc"], DAD, ANCHOR)[0] is False


def test_summary_excludes_non_summarizable_even_if_authorized():
    # Child counselor note is visible to MOM but summarizable=False.
    assembled = assemble_context(
        candidates=FAM.items,
        requesting_member_id=MOM,
        task=TaskKind.family_digest,
        policy_version="p1",
        now_utc=ANCHOR,
    )
    ref = _ref(assembled.refs, "i_child_counselor")
    assert ref.included is False and ref.exclusion_reason == "not_summarizable"


def test_assembly_audits_every_item_and_hashes():
    assembled = assemble_context(
        candidates=FAM.items,
        requesting_member_id=MOM,
        task=TaskKind.family_digest,
        policy_version="p1",
        now_utc=ANCHOR,
    )
    assert len(assembled.refs) == len(FAM.items)
    assert all(r.source_hash for r in assembled.refs)


def test_member_visibility_differentiates_on_targeted_task():
    # On a member-targeted task the summarizability gate does not apply, so the
    # per-member visibility rules are what differentiate the views.
    parent_view = assemble_context(
        candidates=FAM.items,
        requesting_member_id=MOM,
        task=TaskKind.notify_decision,
        policy_version="p1",
        now_utc=ANCHOR,
    )
    child_view = assemble_context(
        candidates=FAM.items,
        requesting_member_id=CHILD,
        task=TaskKind.notify_decision,
        policy_version="p1",
        now_utc=ANCHOR,
    )
    assert child_view.included_count < parent_view.included_count
