"""Unit tests for STEP tokenizer."""

import pytest
from pathlib import Path

from cadling.backend.step.tokenizer import STEPTokenizer


class TestSTEPTokenizerInit:
    """Test STEPTokenizer initialization."""

    def test_init_default(self):
        """Test default initialization."""
        tokenizer = STEPTokenizer()

        assert tokenizer.vocab_size == 50000
        assert tokenizer.PAD_ID == 0
        assert tokenizer.UNK_ID == 1
        assert tokenizer.SEP_ID == 2
        assert tokenizer.CLS_ID == 3

    def test_init_custom_vocab_size(self):
        """Test initialization with custom vocab size."""
        tokenizer = STEPTokenizer(vocab_size=10000)

        assert tokenizer.vocab_size == 10000

    def test_special_tokens_set(self):
        """Test special tokens are set correctly."""
        tokenizer = STEPTokenizer()

        assert '<PAD>' in tokenizer.special_tokens
        assert '<UNK>' in tokenizer.special_tokens
        assert '<SEP>' in tokenizer.special_tokens
        assert '<CLS>' in tokenizer.special_tokens

        assert tokenizer.special_tokens['<PAD>'] == 0
        assert tokenizer.special_tokens['<UNK>'] == 1
        assert tokenizer.special_tokens['<SEP>'] == 2
        assert tokenizer.special_tokens['<CLS>'] == 3

    def test_vocab_built(self):
        """Test vocab is built during initialization."""
        tokenizer = STEPTokenizer()

        assert tokenizer.vocab is not None
        assert len(tokenizer.vocab) > 0
        assert tokenizer.id_to_token is not None

    def test_vocab_contains_entity_types(self):
        """Test vocab contains common entity types."""
        tokenizer = STEPTokenizer()

        # Check for common entity types
        assert 'CARTESIAN_POINT' in tokenizer.vocab
        assert 'DIRECTION' in tokenizer.vocab
        assert 'CIRCLE' in tokenizer.vocab
        assert 'PLANE' in tokenizer.vocab
        assert 'ADVANCED_FACE' in tokenizer.vocab


class TestBuildVocab:
    """Test _build_vocab method."""

    def test_vocab_includes_special_tokens(self):
        """Test vocab includes special tokens."""
        tokenizer = STEPTokenizer()
        vocab = tokenizer.vocab

        assert '<PAD>' in vocab
        assert '<UNK>' in vocab
        assert '<SEP>' in vocab
        assert '<CLS>' in vocab

    def test_vocab_includes_entity_types(self):
        """Test vocab includes entity types."""
        tokenizer = STEPTokenizer()
        vocab = tokenizer.vocab

        # Points and directions
        assert 'CARTESIAN_POINT' in vocab
        assert 'DIRECTION' in vocab
        assert 'VECTOR' in vocab

        # Curves
        assert 'LINE' in vocab
        assert 'CIRCLE' in vocab
        assert 'B_SPLINE_CURVE' in vocab

        # Surfaces
        assert 'PLANE' in vocab
        assert 'CYLINDRICAL_SURFACE' in vocab
        assert 'SPHERICAL_SURFACE' in vocab

        # Topology
        assert 'VERTEX_POINT' in vocab
        assert 'EDGE_CURVE' in vocab
        assert 'ADVANCED_FACE' in vocab
        assert 'CLOSED_SHELL' in vocab

    def test_vocab_includes_keywords(self):
        """Test vocab includes STEP keywords."""
        tokenizer = STEPTokenizer()
        vocab = tokenizer.vocab

        assert '.T.' in vocab
        assert '.F.' in vocab
        assert '.UNSPECIFIED.' in vocab
        assert '$' in vocab
        assert '*' in vocab

    def test_vocab_includes_operators(self):
        """Test vocab includes operators."""
        tokenizer = STEPTokenizer()
        vocab = tokenizer.vocab

        assert '=' in vocab
        assert '(' in vocab
        assert ')' in vocab
        assert ',' in vocab
        assert ';' in vocab
        assert '#' in vocab

    def test_vocab_size_reasonable(self):
        """Test vocab size is reasonable."""
        tokenizer = STEPTokenizer()

        # Should have special tokens + entity types + keywords + operators
        # At least 50 tokens
        assert tokenizer.get_vocab_size() > 50


class TestTokenize:
    """Test tokenize method."""

    def test_tokenize_entity_reference(self):
        """Test tokenizing entity references."""
        tokenizer = STEPTokenizer()

        tokens = tokenizer.tokenize("#123")
        assert tokens == ["#123"]

        tokens = tokenizer.tokenize("#1 #456 #789")
        assert tokens == ["#1", "#456", "#789"]

    def test_tokenize_entity_type(self):
        """Test tokenizing entity type names."""
        tokenizer = STEPTokenizer()

        tokens = tokenizer.tokenize("CARTESIAN_POINT")
        assert tokens == ["CARTESIAN_POINT"]

        tokens = tokenizer.tokenize("CARTESIAN_POINT DIRECTION")
        assert tokens == ["CARTESIAN_POINT", "DIRECTION"]

    def test_tokenize_numbers(self):
        """Test tokenizing numbers."""
        tokenizer = STEPTokenizer()

        # Integer
        tokens = tokenizer.tokenize("123")
        assert tokens == ["123"]

        # Float
        tokens = tokenizer.tokenize("123.456")
        assert tokens == ["123.456"]

        # Negative
        tokens = tokenizer.tokenize("-123.456")
        assert tokens == ["-123.456"]

        # Scientific notation
        tokens = tokenizer.tokenize("1.23E-10")
        assert tokens == ["1.23E-10"]

    def test_tokenize_keywords(self):
        """Test tokenizing STEP keywords."""
        tokenizer = STEPTokenizer()

        tokens = tokenizer.tokenize(".T.")
        assert tokens == [".T."]

        tokens = tokenizer.tokenize(".F.")
        assert tokens == [".F."]

        tokens = tokenizer.tokenize(".UNSPECIFIED.")
        assert tokens == [".UNSPECIFIED."]

    def test_tokenize_operators(self):
        """Test tokenizing operators."""
        tokenizer = STEPTokenizer()

        tokens = tokenizer.tokenize("=")
        assert tokens == ["="]

        tokens = tokenizer.tokenize("()")
        assert tokens == ["(", ")"]

        tokens = tokenizer.tokenize(",;")
        assert tokens == [",", ";"]

    def test_tokenize_string_literals(self):
        """Test tokenizing string literals."""
        tokenizer = STEPTokenizer()

        tokens = tokenizer.tokenize("'hello world'")
        assert tokens == ["'hello world'"]

    def test_tokenize_complex_line(self):
        """Test tokenizing complex STEP line."""
        tokenizer = STEPTokenizer()

        text = "#123=CARTESIAN_POINT('origin',(0.0,0.0,0.0));"
        tokens = tokenizer.tokenize(text)

        expected = ["#123", "=", "CARTESIAN_POINT", "(", "'origin'", ",",
                   "(", "0.0", ",", "0.0", ",", "0.0", ")", ")", ";"]
        assert tokens == expected

    def test_tokenize_with_null_and_derived(self):
        """Test tokenizing $ and * symbols."""
        tokenizer = STEPTokenizer()

        tokens = tokenizer.tokenize("$")
        assert "$" in tokens

        tokens = tokenizer.tokenize("*")
        assert "*" in tokens


class TestEncode:
    """Test encode method."""

    def test_encode_known_tokens(self):
        """Test encoding known tokens."""
        tokenizer = STEPTokenizer()

        # Entity type tokens (special tokens like <PAD> are vocab only, not STEP text)
        ids = tokenizer.encode("CARTESIAN_POINT")
        assert len(ids) == 1
        assert ids[0] == tokenizer.vocab["CARTESIAN_POINT"]

        ids = tokenizer.encode("DIRECTION")
        assert len(ids) == 1
        assert ids[0] == tokenizer.vocab["DIRECTION"]

    def test_encode_entity_types(self):
        """Test encoding entity types."""
        tokenizer = STEPTokenizer()

        ids = tokenizer.encode("CARTESIAN_POINT")
        assert len(ids) == 1
        assert ids[0] == tokenizer.vocab["CARTESIAN_POINT"]

    def test_encode_unknown_tokens_hashed(self):
        """Test unknown tokens are hashed."""
        tokenizer = STEPTokenizer()

        # Unknown token should be hashed
        ids = tokenizer.encode("UNKNOWN_ENTITY_TYPE_12345")
        assert len(ids) == 1
        assert 0 <= ids[0] < tokenizer.vocab_size

    def test_encode_complex_text(self):
        """Test encoding complex STEP text."""
        tokenizer = STEPTokenizer()

        text = "CARTESIAN_POINT ( 0.0 , 1.0 , 2.0 )"
        ids = tokenizer.encode(text)

        # Should have multiple IDs
        assert len(ids) > 0
        assert all(isinstance(id, int) for id in ids)

    def test_encode_consistency(self):
        """Test encoding same text twice gives same result."""
        tokenizer = STEPTokenizer()

        text = "CARTESIAN_POINT"
        ids1 = tokenizer.encode(text)
        ids2 = tokenizer.encode(text)

        assert ids1 == ids2


class TestDecode:
    """Test decode method."""

    def test_decode_known_ids(self):
        """Test decoding known token IDs."""
        tokenizer = STEPTokenizer()

        # Special tokens
        text = tokenizer.decode([tokenizer.PAD_ID])
        assert "<PAD>" in text

        text = tokenizer.decode([tokenizer.UNK_ID])
        assert "<UNK>" in text

    def test_decode_entity_types(self):
        """Test decoding entity type IDs."""
        tokenizer = STEPTokenizer()

        cartesian_id = tokenizer.vocab["CARTESIAN_POINT"]
        text = tokenizer.decode([cartesian_id])
        assert "CARTESIAN_POINT" in text

    def test_decode_unknown_ids(self):
        """Test decoding unknown IDs returns <UNK>."""
        tokenizer = STEPTokenizer()

        # Use an ID not in id_to_token
        unknown_id = 99999
        text = tokenizer.decode([unknown_id])
        assert "<UNK>" in text

    def test_decode_multiple_ids(self):
        """Test decoding multiple IDs."""
        tokenizer = STEPTokenizer()

        ids = [
            tokenizer.vocab["CARTESIAN_POINT"],
            tokenizer.vocab["("],
            tokenizer.vocab[")"]
        ]
        text = tokenizer.decode(ids)

        assert "CARTESIAN_POINT" in text
        assert "(" in text
        assert ")" in text

    def test_encode_decode_roundtrip(self):
        """Test encode-decode roundtrip for known tokens."""
        tokenizer = STEPTokenizer()

        original = "CARTESIAN_POINT"
        ids = tokenizer.encode(original)
        decoded = tokenizer.decode(ids)

        # Decoded text should contain original (may have spaces)
        assert "CARTESIAN_POINT" in decoded


class TestBatchEncode:
    """Test batch_encode method."""

    def test_batch_encode_basic(self):
        """Test basic batch encoding."""
        tokenizer = STEPTokenizer()

        texts = ["CARTESIAN_POINT", "DIRECTION"]
        result = tokenizer.batch_encode(texts, add_special_tokens=False)

        assert 'token_ids' in result
        assert len(result['token_ids']) == 2
        assert all(isinstance(ids, list) for ids in result['token_ids'])

    def test_batch_encode_with_special_tokens(self):
        """Test batch encoding with special tokens."""
        tokenizer = STEPTokenizer()

        texts = ["CARTESIAN_POINT"]
        result = tokenizer.batch_encode(texts, add_special_tokens=True)

        ids = result['token_ids'][0]
        # Should have CLS at start and SEP at end
        assert ids[0] == tokenizer.CLS_ID
        assert ids[-1] == tokenizer.SEP_ID

    def test_batch_encode_with_max_length_truncation(self):
        """Test batch encoding with truncation."""
        tokenizer = STEPTokenizer()

        texts = ["CARTESIAN_POINT DIRECTION VECTOR PLANE"]
        max_length = 3
        result = tokenizer.batch_encode(
            texts,
            add_special_tokens=False,
            max_length=max_length
        )

        ids = result['token_ids'][0]
        assert len(ids) <= max_length

    def test_batch_encode_with_padding(self):
        """Test batch encoding with padding."""
        tokenizer = STEPTokenizer()

        texts = ["CARTESIAN_POINT", "DIRECTION VECTOR"]
        max_length = 10
        result = tokenizer.batch_encode(
            texts,
            add_special_tokens=False,
            max_length=max_length,
            padding=True
        )

        # All sequences should have same length
        ids1 = result['token_ids'][0]
        ids2 = result['token_ids'][1]
        assert len(ids1) == max_length
        assert len(ids2) == max_length

    def test_batch_encode_attention_mask(self):
        """Test batch encoding generates attention mask."""
        tokenizer = STEPTokenizer()

        texts = ["CARTESIAN_POINT", "DIRECTION"]
        max_length = 10
        result = tokenizer.batch_encode(
            texts,
            add_special_tokens=False,
            max_length=max_length,
            padding=True
        )

        assert 'attention_mask' in result
        assert len(result['attention_mask']) == 2

        # Attention mask should have 1s for tokens, 0s for padding
        for mask in result['attention_mask']:
            assert len(mask) == max_length
            assert all(m in [0, 1] for m in mask)

    def test_batch_encode_attention_mask_correctness(self):
        """Test attention mask marks padding correctly."""
        tokenizer = STEPTokenizer()

        texts = ["CARTESIAN_POINT"]
        max_length = 10
        result = tokenizer.batch_encode(
            texts,
            add_special_tokens=False,
            max_length=max_length,
            padding=True
        )

        ids = result['token_ids'][0]
        mask = result['attention_mask'][0]

        # Check mask corresponds to ids
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            if token_id == tokenizer.PAD_ID:
                assert mask_val == 0, f"Padding at index {i} should have mask 0"
            else:
                assert mask_val == 1, f"Token at index {i} should have mask 1"


class TestVocabMethods:
    """Test vocabulary getter methods."""

    def test_get_vocab_size(self):
        """Test get_vocab_size returns correct size."""
        tokenizer = STEPTokenizer()

        size = tokenizer.get_vocab_size()
        assert size == len(tokenizer.vocab)
        assert size > 0

    def test_get_special_tokens(self):
        """Test get_special_tokens returns correct mapping."""
        tokenizer = STEPTokenizer()

        special = tokenizer.get_special_tokens()

        assert '<PAD>' in special
        assert '<UNK>' in special
        assert '<SEP>' in special
        assert '<CLS>' in special

        assert special['<PAD>'] == 0
        assert special['<UNK>'] == 1
        assert special['<SEP>'] == 2
        assert special['<CLS>'] == 3


class TestParseStepFile:
    """Test parse_step_file method."""

    @pytest.fixture
    def sample_step_content(self):
        """Sample STEP file content."""
        return """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('CAD model'),'2;1');
FILE_NAME('test.stp','2024-01-01',('Author'),('Organization'),'Preprocessor','Originating System','Authorization');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1=CARTESIAN_POINT('',(0.0,0.0,0.0));
#2=DIRECTION('',(0.0,0.0,1.0));
#3=VECTOR('',#2,1.0);
ENDSEC;
END-ISO-10303-21;
"""

    def test_parse_step_file_structure(self, sample_step_content):
        """Test parsing returns correct structure."""
        tokenizer = STEPTokenizer()
        result = tokenizer.parse_step_file(sample_step_content)

        assert 'header' in result
        assert 'entities' in result
        assert isinstance(result['header'], dict)
        assert isinstance(result['entities'], dict)

    def test_parse_step_file_header(self, sample_step_content):
        """Test header is parsed."""
        tokenizer = STEPTokenizer()
        result = tokenizer.parse_step_file(sample_step_content)

        header = result['header']
        assert 'file_description' in header
        assert 'file_name' in header
        assert 'file_schema' in header

    def test_parse_step_file_entities(self, sample_step_content):
        """Test entities are parsed."""
        tokenizer = STEPTokenizer()
        result = tokenizer.parse_step_file(sample_step_content)

        entities = result['entities']
        assert len(entities) > 0
        assert 1 in entities
        assert 2 in entities
        assert 3 in entities

    def test_parse_step_file_entity_structure(self, sample_step_content):
        """Test entity structure is correct."""
        tokenizer = STEPTokenizer()
        result = tokenizer.parse_step_file(sample_step_content)

        entity = result['entities'][1]
        assert 'type' in entity
        assert 'params' in entity
        assert 'raw' in entity
        assert entity['type'] == 'CARTESIAN_POINT'


class TestParseHeader:
    """Test _parse_header method."""

    def test_parse_header_file_description(self):
        """Test parsing FILE_DESCRIPTION."""
        tokenizer = STEPTokenizer()

        lines = ["FILE_DESCRIPTION(('test'),'2;1');"]
        header = tokenizer._parse_header(lines)

        assert header['file_description'] is not None
        assert 'FILE_DESCRIPTION' in header['file_description']

    def test_parse_header_file_name(self):
        """Test parsing FILE_NAME."""
        tokenizer = STEPTokenizer()

        lines = ["FILE_NAME('test.stp','2024-01-01',('Author'),('Org'),'Pre','Sys','Auth');"]
        header = tokenizer._parse_header(lines)

        assert header['file_name'] is not None
        assert 'FILE_NAME' in header['file_name']

    def test_parse_header_file_schema(self):
        """Test parsing FILE_SCHEMA."""
        tokenizer = STEPTokenizer()

        lines = ["FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));"]
        header = tokenizer._parse_header(lines)

        assert header['file_schema'] is not None
        assert 'FILE_SCHEMA' in header['file_schema']

    def test_parse_header_multiple_lines(self):
        """Test parsing multiple header lines."""
        tokenizer = STEPTokenizer()

        lines = [
            "FILE_DESCRIPTION(('test'),'2;1');",
            "FILE_NAME('test.stp','2024-01-01',('Author'),('Org'),'Pre','Sys','Auth');",
            "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));"
        ]
        header = tokenizer._parse_header(lines)

        assert header['file_description'] is not None
        assert header['file_name'] is not None
        assert header['file_schema'] is not None


class TestParseEntities:
    """Test _parse_entities method."""

    def test_parse_entities_basic(self):
        """Test parsing basic entity."""
        tokenizer = STEPTokenizer()

        lines = ["#1=CARTESIAN_POINT('',(0.0,0.0,0.0));"]
        entities = tokenizer._parse_entities(lines)

        assert 1 in entities
        assert entities[1]['type'] == 'CARTESIAN_POINT'

    def test_parse_entities_multiple(self):
        """Test parsing multiple entities."""
        tokenizer = STEPTokenizer()

        lines = [
            "#1=CARTESIAN_POINT('',(0.0,0.0,0.0));",
            "#2=DIRECTION('',(0.0,0.0,1.0));",
            "#3=VECTOR('',#2,1.0);"
        ]
        entities = tokenizer._parse_entities(lines)

        assert len(entities) == 3
        assert 1 in entities
        assert 2 in entities
        assert 3 in entities

    def test_parse_entities_extracts_id(self):
        """Test entity ID is extracted correctly."""
        tokenizer = STEPTokenizer()

        lines = ["#123=CARTESIAN_POINT('',(0.0,0.0,0.0));"]
        entities = tokenizer._parse_entities(lines)

        assert 123 in entities

    def test_parse_entities_extracts_type(self):
        """Test entity type is extracted correctly."""
        tokenizer = STEPTokenizer()

        lines = ["#1=DIRECTION('',(0.0,0.0,1.0));"]
        entities = tokenizer._parse_entities(lines)

        assert entities[1]['type'] == 'DIRECTION'

    def test_parse_entities_skips_empty_lines(self):
        """Test empty lines are skipped."""
        tokenizer = STEPTokenizer()

        lines = ["", "#1=CARTESIAN_POINT('',(0.0,0.0,0.0));", ""]
        entities = tokenizer._parse_entities(lines)

        assert len(entities) == 1

    def test_parse_entities_skips_non_entity_lines(self):
        """Test non-entity lines are skipped."""
        tokenizer = STEPTokenizer()

        lines = [
            "ENDSEC;",
            "#1=CARTESIAN_POINT('',(0.0,0.0,0.0));",
            "DATA;"
        ]
        entities = tokenizer._parse_entities(lines)

        assert len(entities) == 1


class TestParseParams:
    """Test _parse_params method."""

    def test_parse_params_empty(self):
        """Test parsing empty parameters."""
        tokenizer = STEPTokenizer()

        params = tokenizer._parse_params("")
        assert params == []

    def test_parse_params_single(self):
        """Test parsing single parameter."""
        tokenizer = STEPTokenizer()

        params = tokenizer._parse_params("'test'")
        assert len(params) == 1

    def test_parse_params_multiple(self):
        """Test parsing multiple parameters."""
        tokenizer = STEPTokenizer()

        params = tokenizer._parse_params("0.0,1.0,2.0")
        assert len(params) == 3

    def test_parse_params_nested_parentheses(self):
        """Test parsing nested parentheses."""
        tokenizer = STEPTokenizer()

        params = tokenizer._parse_params("'',(0.0,1.0,2.0)")
        assert len(params) == 2
        # Second param should be a list
        assert isinstance(params[1], list)

    def test_parse_params_with_strings(self):
        """Test parsing parameters with strings."""
        tokenizer = STEPTokenizer()

        params = tokenizer._parse_params("'name',123")
        assert len(params) == 2
        assert params[0] == 'name'

    def test_parse_params_complex(self):
        """Test parsing complex parameter string."""
        tokenizer = STEPTokenizer()

        params = tokenizer._parse_params("'',#2,1.0")
        assert len(params) == 3


class TestParseSingleParam:
    """Test _parse_single_param method."""

    def test_parse_single_param_null(self):
        """Test parsing null parameter."""
        tokenizer = STEPTokenizer()

        result = tokenizer._parse_single_param("$")
        assert result is None

    def test_parse_single_param_empty(self):
        """Test parsing empty parameter."""
        tokenizer = STEPTokenizer()

        result = tokenizer._parse_single_param("")
        assert result is None

    def test_parse_single_param_string(self):
        """Test parsing string literal."""
        tokenizer = STEPTokenizer()

        result = tokenizer._parse_single_param("'hello'")
        assert result == 'hello'

    def test_parse_single_param_entity_reference(self):
        """Test parsing entity reference."""
        tokenizer = STEPTokenizer()

        result = tokenizer._parse_single_param("#123")
        assert result == '#123'

    def test_parse_single_param_boolean(self):
        """Test parsing boolean/enumeration."""
        tokenizer = STEPTokenizer()

        result = tokenizer._parse_single_param(".T.")
        assert result == '.T.'

        result = tokenizer._parse_single_param(".F.")
        assert result == '.F.'

    def test_parse_single_param_list(self):
        """Test parsing list parameter."""
        tokenizer = STEPTokenizer()

        result = tokenizer._parse_single_param("(1,2,3)")
        assert isinstance(result, list)
        assert len(result) == 3

    def test_parse_single_param_number_string(self):
        """Test parsing numbers (kept as strings)."""
        tokenizer = STEPTokenizer()

        # Integer
        result = tokenizer._parse_single_param("123")
        assert result == "123"

        # Float
        result = tokenizer._parse_single_param("123.456")
        assert result == "123.456"

        # Scientific notation
        result = tokenizer._parse_single_param("1.23E-10")
        assert result == "1.23E-10"


class TestTokenizerIntegration:
    """Integration tests using real STEP data."""

    @pytest.fixture
    def test_data_path(self):
        """Get test data directory."""
        # From cadling/tests/unit/backend/step, go up to cadling/, then to data/test_data
        return Path(__file__).parent.parent.parent.parent.parent / "data" / "test_data"

    @pytest.fixture
    def small_step_file(self, test_data_path):
        """Get a small STEP file for testing."""
        step_path = test_data_path / "step"
        if not step_path.exists():
            pytest.skip("STEP test data not found")

        # Find a small STEP file
        step_files = sorted(list(step_path.glob("*.stp")) + list(step_path.glob("*.step")))
        if not step_files:
            pytest.skip("No STEP files found")

        # Use the smallest file
        smallest_file = min(step_files, key=lambda f: f.stat().st_size)
        return smallest_file

    def test_tokenize_real_step_file(self, small_step_file):
        """Test tokenizing real STEP file content."""
        tokenizer = STEPTokenizer()

        content = small_step_file.read_text(encoding='utf-8', errors='ignore')
        tokens = tokenizer.tokenize(content)

        # Should produce tokens
        assert len(tokens) > 0

        # Should contain typical STEP tokens
        assert any('#' in token for token in tokens)  # Entity references

    def test_encode_real_step_file(self, small_step_file):
        """Test encoding real STEP file content."""
        tokenizer = STEPTokenizer()

        content = small_step_file.read_text(encoding='utf-8', errors='ignore')
        # Take first 1000 characters to avoid too long sequences
        content = content[:1000]

        token_ids = tokenizer.encode(content)

        # Should produce token IDs
        assert len(token_ids) > 0
        assert all(isinstance(tid, int) for tid in token_ids)
        assert all(0 <= tid < tokenizer.vocab_size for tid in token_ids)

    def test_parse_real_step_file(self, small_step_file):
        """Test parsing real STEP file."""
        tokenizer = STEPTokenizer()

        content = small_step_file.read_text(encoding='utf-8', errors='ignore')
        result = tokenizer.parse_step_file(content)

        # Should have header and entities
        assert 'header' in result
        assert 'entities' in result

        # Header should have fields
        assert 'file_description' in result['header']
        assert 'file_name' in result['header']
        assert 'file_schema' in result['header']

        # Should have parsed some entities
        assert len(result['entities']) > 0

        # Entities should have correct structure
        for entity_id, entity_data in result['entities'].items():
            assert isinstance(entity_id, int)
            assert 'type' in entity_data
            assert 'params' in entity_data
            assert 'raw' in entity_data

    def test_batch_encode_multiple_files(self, test_data_path):
        """Test batch encoding multiple STEP file excerpts."""
        tokenizer = STEPTokenizer()

        step_path = test_data_path / "step"
        if not step_path.exists():
            pytest.skip("STEP test data not found")

        step_files = list(step_path.glob("*.stp"))[:3]  # First 3 files
        if len(step_files) < 2:
            pytest.skip("Not enough STEP files for batch test")

        # Extract short excerpts from each file
        texts = []
        for f in step_files:
            content = f.read_text(encoding='utf-8', errors='ignore')
            # Extract first entity line
            for line in content.split('\n'):
                if line.strip().startswith('#'):
                    texts.append(line.strip())
                    break

        if len(texts) < 2:
            pytest.skip("Could not extract entity lines")

        # Batch encode
        result = tokenizer.batch_encode(
            texts,
            add_special_tokens=True,
            max_length=50,
            padding=True
        )

        # Should have token_ids for each text
        assert len(result['token_ids']) == len(texts)

        # All sequences should have same length due to padding
        lengths = [len(ids) for ids in result['token_ids']]
        assert len(set(lengths)) == 1
        assert lengths[0] == 50

        # Should have attention mask
        assert 'attention_mask' in result
        assert len(result['attention_mask']) == len(texts)


class TestMultilineEntityParsing:
    """Test multiline entity parsing."""

    def test_parse_multiline_entity_basic(self):
        """Test parsing entity split across two lines."""
        tokenizer = STEPTokenizer()

        content = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Test'),'2;1');
FILE_NAME('test.stp','2024-01-01',('Author'),('Org'),'Pre','Sys','Auth');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1 = CARTESIAN_POINT('origin',
  (0.0, 0.0, 0.0));
ENDSEC;
END-ISO-10303-21;
"""

        result = tokenizer.parse_step_file(content)

        assert 1 in result['entities']
        assert result['entities'][1]['type'] == 'CARTESIAN_POINT'
        assert len(result['entities'][1]['params']) == 2
        assert result['entities'][1]['params'][0] == 'origin'

    def test_parse_multiline_entity_complex(self):
        """Test parsing complex multiline entity (5+ lines)."""
        tokenizer = STEPTokenizer()

        content = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Test'),'2;1');
FILE_NAME('test.stp','2024-01-01',('Author'),('Org'),'Pre','Sys','Auth');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#11 = ( GEOMETRIC_REPRESENTATION_CONTEXT(3)
GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#15))
GLOBAL_UNIT_ASSIGNED_CONTEXT
((#16,#17,#18))
REPRESENTATION_CONTEXT('Context #1',
  '3D Context with UNIT and UNCERTAINTY') );
ENDSEC;
END-ISO-10303-21;
"""

        result = tokenizer.parse_step_file(content)

        assert 11 in result['entities']
        # Should parse the complex nested structure

    def test_parse_entity_with_comments(self):
        """Test parsing entity with comments."""
        tokenizer = STEPTokenizer()

        content = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Test'),'2;1');
FILE_NAME('test.stp','2024-01-01',('Author'),('Org'),'Pre','Sys','Auth');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1 = /* This is the origin point */ CARTESIAN_POINT('origin', (0.0, 0.0, 0.0));
ENDSEC;
END-ISO-10303-21;
"""

        result = tokenizer.parse_step_file(content)

        assert 1 in result['entities']
        assert result['entities'][1]['type'] == 'CARTESIAN_POINT'
        # Comment should be removed
        assert '/*' not in result['entities'][1]['raw']

    def test_parse_entity_with_multiline_string(self):
        """Test parsing entity with string parameters across lines."""
        tokenizer = STEPTokenizer()

        content = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Test'),'2;1');
FILE_NAME('test.stp','2024-01-01',('Author'),('Org'),'Pre','Sys','Auth');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1 = PRODUCT('Part Name',
  'Part-001',
  'Description here',
  (#2));
ENDSEC;
END-ISO-10303-21;
"""

        result = tokenizer.parse_step_file(content)

        assert 1 in result['entities']
        params = result['entities'][1]['params']
        assert params[0] == 'Part Name'
        assert params[1] == 'Part-001'

    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        tokenizer = STEPTokenizer()

        # Test with newlines and tabs
        text = "ENTITY_TYPE(   \n\t   param1   ,   \n   param2   )"
        normalized = tokenizer._normalize_whitespace(text)

        assert '\n' not in normalized
        assert '\t' not in normalized
        # Should have single spaces
        assert '  ' not in normalized

    def test_comment_removal(self):
        """Test comment removal."""
        tokenizer = STEPTokenizer()

        text = "DATA; /* comment */ #1 = ENTITY()"
        cleaned = tokenizer._remove_comments(text)

        assert '/*' not in cleaned
        assert 'comment' not in cleaned
        assert 'DATA;' in cleaned
        assert '#1 = ENTITY()' in cleaned

    def test_nested_parentheses_across_lines(self):
        """Test parsing nested parentheses across multiple lines."""
        tokenizer = STEPTokenizer()

        content = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('Test'),'2;1');
FILE_NAME('test.stp','2024-01-01',('Author'),('Org'),'Pre','Sys','Auth');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));
ENDSEC;
DATA;
#1 = ADVANCED_FACE('',(
  #2,
  #3
),#4,.T.);
ENDSEC;
END-ISO-10303-21;
"""

        result = tokenizer.parse_step_file(content)

        assert 1 in result['entities']
        assert result['entities'][1]['type'] == 'ADVANCED_FACE'
        # Should have parsed nested parameters
        params = result['entities'][1]['params']
        assert len(params) > 0
