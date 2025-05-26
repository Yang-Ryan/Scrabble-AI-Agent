class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True

    def is_word(self, word: str) -> bool:
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def rack_words(self, rack: list, max_words=500) -> list:
        """Return all words in dictionary that can be formed with rack"""
        results = set()
        self._search(self.root, [], rack, results, max_words)
        return list(results)

    def _search(self, node, path, rack, results, max_words):
        if len(results) >= max_words:
            return
        if node.is_end:
            results.add(''.join(path))

        used = set()
        for i, ch in enumerate(rack):
            if ch in node.children and ch not in used:
                used.add(ch)
                next_rack = rack[:i] + rack[i+1:]
                path.append(ch)
                self._search(node.children[ch], path, next_rack, results, max_words)
                path.pop()
