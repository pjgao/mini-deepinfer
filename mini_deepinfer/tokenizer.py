class ByteTokenizer:
    """Zero-download UTF-8 tokenizer, NOT the official DeepSeek tokenizer."""

    bos_token_id = 256
    eos_token_id = 257
    pad_token_id = 258

    def encode(self, text: str) -> list[int]:
        return [self.bos_token_id, *text.encode("utf-8")]

    def decode(self, tokens: list[int]) -> str:
        return bytes(t for t in tokens if 0 <= t < 256).decode("utf-8", errors="replace")
