"""Answer generation semantics without loading torch or model weights."""
import contextlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from tests.helpers import ROOT
sys.path.insert(0, str(ROOT / 'dashboard'))
import live_model as LM


class Batch(dict):
    def to(self, device):
        return self


class TestLiveAnswer(unittest.TestCase):
    def setup_model(self, new_tokens):
        tokenizer = MagicMock()
        tokenizer.return_value = {'input_ids': [1, 2]}
        tokenizer.apply_chat_template.return_value = Batch(input_ids=np.array([[101, 1, 2, 102]]))
        tokenizer.decode.side_effect = lambda ids, **kw: 'input question' if list(ids) == [1, 2] else 'model answer'
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = 99
        model = SimpleNamespace(generate=MagicMock(return_value=np.array([[101, 1, 2, 102] + new_tokens])),
                                generation_config=SimpleNamespace(eos_token_id=99))
        return LM.Loaded(model, tokenizer, 'cpu', 'bfloat16', 0.0)

    def test_uses_instruction_template_and_decodes_only_new_tokens(self):
        m = self.setup_model([7, 8, 99])
        fake_torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        with patch.dict(sys.modules, {'torch': fake_torch}):
            out = LM.generate(m, 'input question', 64)
        self.assertEqual(out.text, 'model answer')
        self.assertEqual(out.output_tokens, 3)
        self.assertFalse(out.hit_limit)
        self.assertEqual(m.tokenizer.apply_chat_template.call_args.args[0], [
            {'role': 'system', 'content': LM.ANSWER_CONTEXT},
            {'role': 'user', 'content': 'input question'},
        ])
        self.assertEqual(m.tokenizer.decode.call_args.args[0], [7, 8, 99])
        self.assertTrue(m.model.generate.call_args.kwargs['use_cache'])
        self.assertFalse(m.model.generate.call_args.kwargs['output_router_logits'])

    def test_length_limit_is_reported(self):
        m = self.setup_model([7, 8])
        with patch.dict(sys.modules, {'torch': SimpleNamespace(inference_mode=contextlib.nullcontext)}):
            out = LM.generate(m, 'question', 2)
        self.assertTrue(out.hit_limit)

    def test_invalid_inputs_fail_before_importing_torch(self):
        for prompt, count in [('', 64), ('question', 0), ('question', 257)]:
            with self.subTest(prompt=prompt, count=count), self.assertRaises(ValueError):
                LM.generate(None, prompt, count)


if __name__ == '__main__':
    unittest.main()
