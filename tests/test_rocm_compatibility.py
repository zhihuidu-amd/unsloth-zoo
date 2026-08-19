"""
AMD ROCm aiter attention tests for unsloth-zoo.

Covers the three symbols added by PR #920 (now merged):
  get_amd_attention_implementation, get_amd_flash_attn_func,
  replace_sdpa_with_amd_aiter

FlashInfer env clearing, sm_cap guard and _default_target_gb are already
tested in tests/test_vllm_flashinfer_hip.py,
tests/test_vllm_utils_xpu_sm_cap.py and tests/test_tiled_mlp_target_gb.py.
"""

import sys
from unittest import mock
import pytest
import torch
import unsloth_zoo.device_type as dt


def _make_hip_torch():
    v = mock.MagicMock(); v.hip = "7.0.0"; v.cuda = None; return v

def _make_cuda_torch():
    v = mock.MagicMock(); v.hip = None; v.cuda = "12.1"; return v


@pytest.fixture(autouse=True)
def _clear_caches():
    yield
    dt.is_hip.cache_clear()
    if hasattr(dt, "get_amd_attention_implementation"):
        dt.get_amd_attention_implementation.cache_clear()
    if hasattr(dt, "get_amd_flash_attn_func"):
        dt.get_amd_flash_attn_func.cache_clear()
    if hasattr(dt, "_detect_gfx_arch"):
        dt._detect_gfx_arch.cache_clear()


class TestAmdAiterDetection:
    def test_returns_sdpa_on_nvidia(self):
        with mock.patch.object(torch, "version", _make_cuda_torch()):
            assert dt.get_amd_attention_implementation() == "sdpa"

    def test_returns_sdpa_without_aiter(self):
        with mock.patch.object(torch, "version", _make_hip_torch()):
            with mock.patch.object(dt, "_detect_rocm_major_minor", return_value="7.0"):
                with mock.patch.object(dt, "_detect_gfx_arch", return_value="gfx942"):
                    with mock.patch("importlib.util.find_spec", return_value=None):
                        assert dt.get_amd_attention_implementation() == "sdpa"

    def test_returns_sdpa_on_rocm_lt7_even_with_aiter_present(self):
        """ROCm version gate fires before the aiter attribute check."""
        mock_aiter = mock.MagicMock(); mock_aiter.flash_attn_func = mock.MagicMock()
        with mock.patch.object(torch, "version", _make_hip_torch()):
            with mock.patch.object(dt, "_detect_rocm_major_minor", return_value="6.2"):
                with mock.patch("importlib.util.find_spec", return_value=True):
                    with mock.patch.dict(sys.modules, {"aiter": mock_aiter}):
                        assert dt.get_amd_attention_implementation() == "sdpa"

    def test_returns_sdpa_on_unsupported_arch(self):
        """Consumer RDNA (gfx1200) is not in the CDNA allowlist."""
        mock_aiter = mock.MagicMock(); mock_aiter.flash_attn_func = mock.MagicMock()
        with mock.patch.object(torch, "version", _make_hip_torch()):
            with mock.patch.object(dt, "_detect_rocm_major_minor", return_value="7.0"):
                with mock.patch.object(dt, "_detect_gfx_arch", return_value="gfx1200"):
                    with mock.patch("importlib.util.find_spec", return_value=True):
                        with mock.patch.dict(sys.modules, {"aiter": mock_aiter}):
                            assert dt.get_amd_attention_implementation() == "sdpa"

    def test_returns_amd_aiter_on_gfx942_with_aiter(self):
        mock_aiter = mock.MagicMock(); mock_aiter.flash_attn_func = mock.MagicMock()
        with mock.patch.object(torch, "version", _make_hip_torch()):
            with mock.patch.object(dt, "_detect_rocm_major_minor", return_value="7.0"):
                with mock.patch.object(dt, "_detect_gfx_arch", return_value="gfx942"):
                    with mock.patch("importlib.util.find_spec", return_value=True):
                        with mock.patch.dict(sys.modules, {"aiter": mock_aiter}):
                            assert dt.get_amd_attention_implementation() == "amd_aiter"

    def test_returns_amd_aiter_on_gfx950_with_aiter(self):
        mock_aiter = mock.MagicMock(); mock_aiter.flash_attn_func = mock.MagicMock()
        with mock.patch.object(torch, "version", _make_hip_torch()):
            with mock.patch.object(dt, "_detect_rocm_major_minor", return_value="7.0"):
                with mock.patch.object(dt, "_detect_gfx_arch", return_value="gfx950"):
                    with mock.patch("importlib.util.find_spec", return_value=True):
                        with mock.patch.dict(sys.modules, {"aiter": mock_aiter}):
                            assert dt.get_amd_attention_implementation() == "amd_aiter"


class TestAmdFlashAttnFunc:
    def test_returns_none_on_nvidia(self):
        with mock.patch.object(torch, "version", _make_cuda_torch()):
            assert dt.get_amd_flash_attn_func() is None

    def test_returns_flash_attn_func_callable(self):
        mock_fn = mock.MagicMock()
        mock_aiter = mock.MagicMock(spec=["flash_attn_func"])
        mock_aiter.flash_attn_func = mock_fn
        with mock.patch.object(dt, "get_amd_attention_implementation",
                               return_value="amd_aiter"):
            with mock.patch.dict(sys.modules, {"aiter": mock_aiter}):
                assert dt.get_amd_flash_attn_func() is mock_fn

    def test_returns_none_when_only_flashattnfunc_class_available(self):
        """FlashAttnFunc.apply() requires 13+ positional args — not safely wrappable."""
        mock_aiter = mock.MagicMock(spec=["FlashAttnFunc"])
        mock_aiter.FlashAttnFunc = mock.MagicMock()
        del mock_aiter.flash_attn_func
        with mock.patch.object(dt, "get_amd_attention_implementation",
                               return_value="amd_aiter"):
            with mock.patch.dict(sys.modules, {"aiter": mock_aiter}):
                assert dt.get_amd_flash_attn_func() is None


class TestReplaceSDPAWithAmdAiter:
    def setup_method(self):
        from unsloth_zoo.compiler import replace_sdpa_with_amd_aiter
        self.rewrite = replace_sdpa_with_amd_aiter

    def _patch_aiter(self):
        return mock.patch("unsloth_zoo.device_type.get_amd_attention_implementation",
                          return_value="amd_aiter")

    def test_noop_on_nvidia(self):
        src = "out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)"
        with mock.patch("unsloth_zoo.device_type.get_amd_attention_implementation",
                        return_value="sdpa"):
            assert self.rewrite(src) == src

    def test_is_idempotent(self):
        src = "    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)"
        with self._patch_aiter():
            once = self.rewrite(src)
            twice = self.rewrite(once)
        assert once == twice

    def test_noop_when_not_literal_true(self):
        src = "    out = scaled_dot_product_attention(q, k, v, is_causal=self.causal)"
        with self._patch_aiter():
            assert self.rewrite(src) == src

    def test_noop_when_attn_mask_present(self):
        src = "    out = scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=True)"
        with self._patch_aiter():
            assert self.rewrite(src) == src

    def test_noop_for_disable_compile_shim(self):
        src = "    out = disable_compile_scaled_dot_product_attention(q, k, v, is_causal=True)"
        with self._patch_aiter():
            assert self.rewrite(src) == src

    def test_noop_for_attribute_assignment(self):
        src = "    self.out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)"
        with self._patch_aiter():
            assert self.rewrite(src) == src

    def test_noop_for_chained_call(self):
        src = "    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2)"
        with self._patch_aiter():
            assert self.rewrite(src) == src

    def test_generated_code_parses(self):
        import ast, textwrap
        src = textwrap.dedent("""\
            import torch
            def forward(q, k, v):
                out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                return out
        """)
        with self._patch_aiter():
            result = self.rewrite(src)
        ast.parse(result)
