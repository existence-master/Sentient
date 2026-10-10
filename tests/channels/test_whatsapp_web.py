"""The neonize bridge turns WhatsApp's protobuf messages into ``WAMessage`` (offline: no connection is made).

Skipped unless the optional ``whatsapp`` extra (neonize) is installed.
"""

from __future__ import annotations

import pytest

from sentient.channels.whatsapp_web import _ensure_magic, _jid, available, to_message

if not available():
    pytest.skip("the optional 'whatsapp' extra (neonize) is not installed", allow_module_level=True)
_ensure_magic()

from neonize.proto.Neonize_pb2 import JID, MessageInfo, MessageSource  # noqa: E402
from neonize.proto.Neonize_pb2 import Message as Event  # noqa: E402
from neonize.proto.waE2E.WAWebProtobufsE2E_pb2 import (  # noqa: E402
    AudioMessage,
    ContextInfo,
    DocumentMessage,
    ExtendedTextMessage,
    FutureProofMessage,
    ImageMessage,
    Message,
    ProtocolMessage,
    ReactionMessage,
)

ME = JID(User="15550001111", Server="s.whatsapp.net")


def event(msg: Message, *, chat: JID = ME, from_me: bool = True) -> Event:
    source = MessageSource(Chat=chat, Sender=JID(User="15550001111", Device=3, Server="s.whatsapp.net"), IsFromMe=from_me)
    return Event(Info=MessageInfo(ID="ABC", Pushname="Maya", MessageSource=source), Message=msg)


def test_text_and_reply():
    m = to_message(event(Message(conversation="hi")))
    assert (m.id, m.chat, m.from_me, m.text, m.reply_to) == ("ABC", "15550001111@s.whatsapp.net", True, "hi", None)
    m = to_message(event(Message(extendedTextMessage=ExtendedTextMessage(text="1", contextInfo=ContextInfo(stanzaID="S9")))))
    assert (m.text, m.reply_to) == ("1", "S9")


def test_media_inside_wrappers():
    voice = Message(audioMessage=AudioMessage(PTT=True, mimetype="audio/ogg; codecs=opus", fileLength=1234))
    m = to_message(event(Message(ephemeralMessage=FutureProofMessage(message=voice))))
    assert [(x.kind, x.name, x.size) for x in m.media] == [("voice", "whatsapp-voice-ABC.ogg", 1234)]
    m = to_message(event(Message(imageMessage=ImageMessage(caption="what is this", mimetype="image/jpeg"))))
    assert m.text == "what is this" and m.media[0].kind == "image"
    doc = Message(documentMessage=DocumentMessage(fileName="r.pdf", caption="see", mimetype="application/pdf"))
    m = to_message(event(Message(documentWithCaptionMessage=FutureProofMessage(message=doc))))
    assert (m.text, m.media[0].kind, m.media[0].name) == ("see", "document", "r.pdf")


def test_protocol_messages_and_reactions_are_skipped():
    assert to_message(event(Message(protocolMessage=ProtocolMessage()))) is None
    assert to_message(event(Message(reactionMessage=ReactionMessage(text="+")))) is None
    group = to_message(event(Message(conversation="g"), chat=JID(User="1203630", Server="g.us"), from_me=False))
    assert group.chat == "1203630@g.us"


def test_jid_from_text():
    jid = _jid("15550001111@s.whatsapp.net")
    assert (jid.User, jid.Server) == ("15550001111", "s.whatsapp.net")
