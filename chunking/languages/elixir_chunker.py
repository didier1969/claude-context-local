"""Elixir-specific tree-sitter based chunker."""

from typing import Any, Dict, Set

from chunking.base_chunker import LanguageChunker

# Elixir definition keywords that produce `call` AST nodes
_ELIXIR_DEF_KEYWORDS = {
    'defmodule', 'def', 'defp', 'defmacro', 'defmacrop',
    'defimpl', 'defprotocol', 'defstruct', 'defguard',
    'defguardp', 'defdelegate', 'defoverridable',
}


class ElixirChunker(LanguageChunker):
    """Elixir-specific chunker using tree-sitter."""

    def __init__(self):
        super().__init__('elixir')

    def _get_splittable_node_types(self) -> Set[str]:
        """In Elixir all top-level definitions are `call` nodes."""
        return {'call'}

    def should_chunk_node(self, node: Any) -> bool:
        """Only chunk `call` nodes whose identifier is an Elixir def keyword."""
        if node.type != 'call':
            return False
        for child in node.children:
            if child.type == 'identifier':
                return child.text.decode('utf-8') in _ELIXIR_DEF_KEYWORDS
        return False

    def extract_metadata(self, node: Any, source: bytes) -> Dict[str, Any]:
        """Extract Elixir-specific metadata (keyword, name, docs)."""
        metadata: Dict[str, Any] = {'node_type': node.type}

        # First identifier child is the keyword (def, defmodule, ...)
        keyword = None
        for child in node.children:
            if child.type == 'identifier':
                keyword = self.get_node_text(child, source)
                metadata['keyword'] = keyword
                break

        # Extract name from arguments
        for child in node.children:
            if child.type == 'arguments':
                for arg in child.children:
                    if arg.type == 'alias':
                        # defmodule MyApp.Foo → alias node
                        metadata['name'] = self.get_node_text(arg, source)
                        break
                    elif arg.type == 'identifier':
                        # def my_func → identifier node
                        metadata['name'] = self.get_node_text(arg, source)
                        break
                    elif arg.type == 'call':
                        # def my_func(arg1, arg2) → nested call
                        for sub in arg.children:
                            if sub.type == 'identifier':
                                metadata['name'] = self.get_node_text(sub, source)
                                break
                        break
                    elif arg.type == 'dot':
                        # defimpl MyProtocol, for: MyStruct
                        metadata['name'] = self.get_node_text(arg, source)
                        break
                break

        return metadata
