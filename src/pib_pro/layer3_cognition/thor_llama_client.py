"""Local Thor ``llama-server`` OpenAI-compatible client with offline fallback."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from pib_pro.layer3_cognition.omni_dialogue import (
    ProsodyFeatures,
    WorldStateContextToken,
    build_world_state_context_token,
)
from pib_pro.world_state_bus.ring_buffer import WorldStateRingBuffer


@dataclass(frozen=True, slots=True)
class ThorLlamaClientConfig:
    base_url: str = "http://127.0.0.1:8080"
    model: str = "gemma-4-26B-A4B-it"
    timeout_s: float = 1.5


@dataclass(frozen=True, slots=True)
class ThorDialogueReply:
    utterance: str
    stressed_word: str
    target_object_id: int | None
    used_remote_model: bool


def _format_system_prompt(
    token: WorldStateContextToken,
    prosody: ProsodyFeatures,
) -> str:
    objects = ", ".join(str(oid) for oid in token.active_object_ids) or "none"
    return (
        "You are pib.Pro embodied dialogue on Thor. Reply with JSON only: "
        '{"utterance": str, "stressed_word": str, "target_object_id": int|null}. '
        f"Objects={objects}; joint_attention={token.joint_attention_object_id}; "
        f"surprise={token.surprise_scalar:.2f}; pointing={token.human_pointing_active}; "
        f"prosody_question={prosody.is_question}; hesitation={prosody.hesitation_ratio:.2f}."
    )


def _format_user_prompt(utterance: str) -> str:
    return f"User said: {utterance.strip()}"


def _deterministic_reply(
    token: WorldStateContextToken,
    prosody: ProsodyFeatures,
    utterance: str,
) -> ThorDialogueReply:
    selected = token.joint_attention_object_id if token.joint_attention_object_id >= 0 else None
    if selected is None and token.active_object_ids:
        selected = token.active_object_ids[0]
    if prosody.is_question and len(token.active_object_ids) > 1:
        return ThorDialogueReply(
            utterance="Do you mean THIS one?",
            stressed_word="THIS",
            target_object_id=selected,
            used_remote_model=False,
        )
    color = "red" if token.human_pointing_active else "that"
    text = utterance.strip() or f"Sure, I will take the {color} one."
    if "red" in text.lower():
        color = "red"
    return ThorDialogueReply(
        utterance=text if text else f"Sure, I will take the {color} one.",
        stressed_word=color,
        target_object_id=selected,
        used_remote_model=False,
    )


def _parse_model_content(content: str) -> ThorDialogueReply | None:
    text = content.strip()
    if not text:
        return None
    try:
        if text.startswith("{"):
            data = json.loads(text)
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                return None
            data = json.loads(text[start : end + 1])
        utterance = str(data.get("utterance", ""))
        stressed = str(data.get("stressed_word", "that"))
        target_raw = data.get("target_object_id")
        target = int(target_raw) if target_raw is not None else None
        if not utterance:
            return None
        return ThorDialogueReply(
            utterance=utterance,
            stressed_word=stressed,
            target_object_id=target,
            used_remote_model=True,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


@dataclass(slots=True)
class ThorLlamaServerClient:
    """HTTP client for Thor ``llama-server`` ``/v1/chat/completions`` with safe fallback."""

    config: ThorLlamaClientConfig

    def complete(
        self,
        token: WorldStateContextToken,
        prosody: ProsodyFeatures,
        user_utterance: str,
    ) -> ThorDialogueReply:
        """Call local llama-server; never raises on network failure."""
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": _format_system_prompt(token, prosody)},
                {"role": "user", "content": _format_user_prompt(user_utterance)},
            ],
            "temperature": 0.2,
            "max_tokens": 128,
        }
        url = f"{self.config.base_url.rstrip('/')}/v1/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            choices = payload.get("choices") or []
            if not choices:
                return _deterministic_reply(token, prosody, user_utterance)
            message = choices[0].get("message") or {}
            content = str(message.get("content", ""))
            parsed = _parse_model_content(content)
            if parsed is None:
                return _deterministic_reply(token, prosody, user_utterance)
            return parsed
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            json.JSONDecodeError,
            OSError,
            ValueError,
        ):
            return _deterministic_reply(token, prosody, user_utterance)

    def complete_from_bus(
        self,
        bus: WorldStateRingBuffer,
        prosody: ProsodyFeatures,
        user_utterance: str,
        *,
        stamp_monotonic_s: float,
    ) -> ThorDialogueReply:
        token = build_world_state_context_token(
            bus,
            stamp_monotonic_s=stamp_monotonic_s,
        )
        if token is None:
            empty = WorldStateContextToken(
                active_object_ids=(),
                joint_attention_object_id=-1,
                object_positions_xyz=(),
                surprise_scalar=0.0,
                human_pointing_active=False,
                human_head_shake_detected=False,
                stamp_monotonic_s=stamp_monotonic_s,
            )
            return _deterministic_reply(empty, prosody, user_utterance)
        return self.complete(token, prosody, user_utterance)
